# Hybrid Conv1D + LSTM model for CMAPSS FD001 RUL prediction

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from step1_ml_pipeline import load_cmapss_data, data_processing

# Configuration
DATANAME = "FD001"
VAL_FRAC = 0.15
MAX_RUL = 350.0
WINDOW_SIZE = 30
BATCH_SIZE = 256
EPOCHS = 70
LEARNING_RATE = 1e-3
USE_SAMPLE_WEIGHTS = False
TOP_K_CORR_FEATURES = 11

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data Loading & Standardization
train_df_raw, test_df_raw = load_cmapss_data(DATANAME)
train_std, test_std, scaler = data_processing(train_df_raw, test_df_raw)

# Engine-Level Train/Val Split
rng = np.random.default_rng(seed=17)
eng_ids = train_std["engine_id"].unique()
rng.shuffle(eng_ids)

cut = int((1.0 - VAL_FRAC) * len(eng_ids))
val_ids = set(eng_ids[cut:])
mask_val = train_std["engine_id"].isin(val_ids)

train_df = train_std[~mask_val].copy()
val_df = train_std[mask_val].copy()
test_df = test_std.copy()

print(f"Train engines: {train_df['engine_id'].nunique()}, Val engines: {val_df['engine_id'].nunique()}, Test engines: {test_df['engine_id'].nunique()}")

# RUL Capping at 95th percentile
rul_train = train_df["RUL"].values
OPTIMAL_MAX_RUL = np.percentile(rul_train, 95)
print(f"Capping RUL at 95th percentile: {OPTIMAL_MAX_RUL:.1f}")

train_df["RUL"] = np.minimum(train_df["RUL"].values, OPTIMAL_MAX_RUL)
val_df["RUL"] = np.minimum(val_df["RUL"].values, OPTIMAL_MAX_RUL)
test_df["RUL"] = np.minimum(test_df["RUL"].values, OPTIMAL_MAX_RUL)

# Feature Selection via Pearson Correlation
exclude = ["engine_id", "cycle", "RUL"]
original_features = [c for c in train_df.columns if c not in exclude]

corr_series = train_df[original_features + ["RUL"]].corr(method="pearson")["RUL"].drop("RUL")
feature_cols = corr_series.abs().sort_values(ascending=False).head(TOP_K_CORR_FEATURES).index.tolist()
print(f"Selected {len(feature_cols)} features by correlation with RUL")

# Window Builder
def make_windows(df, feature_cols, window_size=30, strategy="all", bins=None, max_per_bin=8, stride=1, cap_rul=None, return_meta=False):
    X_list, y_list, eng_list, cyc_list = [], [], [], []

    if strategy == "uniform_bins" and bins is None:
        bins = np.array([0, 10, 25, 50, 80, 110, 130])

    for eng_id, eng_df in df.groupby("engine_id"):
        eng_df = eng_df.sort_values("cycle").reset_index(drop=True)
        feats = eng_df[feature_cols].to_numpy(dtype=float)
        rul = eng_df["RUL"].to_numpy(dtype=float)
        cycles = eng_df["cycle"].to_numpy(dtype=int)

        if len(eng_df) < window_size:
            continue

        if strategy == "all":
            starts = list(range(0, len(eng_df) - window_size + 1, stride))
        elif strategy == "uniform_bins":
            candidate_starts = list(range(0, len(eng_df) - window_size + 1))
            candidate_ruls = [rul[s + window_size - 1] for s in candidate_starts]
            bin_indices = np.digitize(candidate_ruls, bins) - 1
            
            starts = []
            rng_local = np.random.default_rng(seed=17)
            for b in np.unique(bin_indices):
                b_indices = [candidate_starts[i] for i in range(len(candidate_starts)) if bin_indices[i] == b]
                if len(b_indices) > max_per_bin:
                    b_indices = rng_local.choice(b_indices, size=max_per_bin, replace=False).tolist()
                starts.extend(b_indices)
            starts.sort()

        for s in starts:
            e = s + window_size
            X_seq = feats[s:e]
            y_val = rul[e - 1]
            if cap_rul is not None:
                y_val = min(y_val, cap_rul)

            X_list.append(X_seq)
            y_list.append(y_val)
            if return_meta:
                eng_list.append(eng_id)
                cyc_list.append(cycles[e - 1])

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, window_size, len(feature_cols)))
    y = np.array(y_list, dtype=float)

    if return_meta:
        return X, y, np.array(eng_list), np.array(cyc_list)
    return X, y

# Generate Windows
X_tr, y_tr = make_windows(train_df, feature_cols, window_size=WINDOW_SIZE, strategy="uniform_bins", max_per_bin=8, cap_rul=OPTIMAL_MAX_RUL)
X_val, y_val = make_windows(val_df, feature_cols, window_size=WINDOW_SIZE, strategy="uniform_bins", max_per_bin=8, cap_rul=OPTIMAL_MAX_RUL)
X_te, y_te, eng_te, cyc_te = make_windows(test_df, feature_cols, window_size=WINDOW_SIZE, strategy="all", stride=1, cap_rul=OPTIMAL_MAX_RUL, return_meta=True)

# Model Architecture
class TemporalAttention(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1, use_bias=False)

    def call(self, inputs, mask=None):
        u = self.W(inputs)
        scores = self.v(u)
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

n_features = len(feature_cols)
inp = keras.Input(shape=(WINDOW_SIZE, n_features), batch_size=BATCH_SIZE)

x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.15))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.10))(x)

context = TemporalAttention(units=64)(x)

h = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(context)
h = layers.Dropout(0.2)(h)
out = layers.Dense(1, activation="linear")(h)

model = keras.Model(inputs=inp, outputs=out)
model.summary()

# Compilation & Training
def nasa_metric(y_true, y_pred):
    d = y_pred - y_true
    s = tf.where(d >= 0, tf.exp(d / 13.0) - 1.0, tf.exp(-d / 10.0) - 1.0)
    return tf.reduce_mean(s)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=keras.losses.Huber(delta=10.0),
    metrics=[keras.metrics.MeanAbsoluteError(name="MAE"), nasa_metric]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    keras.callbacks.ModelCheckpoint(str(OUTPUT_DIR / "best_cnn_lstm_rul.keras"), monitor="val_loss", save_best_only=True)
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

# Evaluation
y_pred_te = model.predict(X_te).ravel()

def nasa_score_numpy(y_true, y_pred):
    d = y_pred - y_true
    s = np.where(d >= 0, np.exp(d / 13.0) - 1.0, np.exp(-d / 10.0) - 1.0)
    return float(np.sum(s))

mae = mean_absolute_error(y_te, y_pred_te)
rmse = mean_squared_error(y_te, y_pred_te, squared=False)
r2 = r2_score(y_te, y_pred_te)
nasa = nasa_score_numpy(y_te, y_pred_te)

print(f"\nPer-window Test Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}, NASA: {nasa:.2f}")

df_win = pd.DataFrame({"engine_id": eng_te, "cycle": cyc_te, "true_RUL": y_te, "pred": y_pred_te})

# First window per engine
df_first = df_win.sort_values(["engine_id", "cycle"]).groupby("engine_id").head(1)
y_true_first, y_pred_first = df_first["true_RUL"].to_numpy(), df_first["pred"].to_numpy()
print(f"First-window - MAE: {mean_absolute_error(y_true_first, y_pred_first):.2f}, RMSE: {mean_squared_error(y_true_first, y_pred_first, squared=False):.2f}, R2: {r2_score(y_true_first, y_pred_first):.3f}, NASA: {nasa_score_numpy(y_true_first, y_pred_first):.2f}")

# Last window per engine
df_last = df_win.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
y_true_last, y_pred_last = df_last["true_RUL"].to_numpy(), df_last["pred"].to_numpy()
print(f"Last-window  - MAE: {mean_absolute_error(y_true_last, y_pred_last):.2f}, RMSE: {mean_squared_error(y_true_last, y_pred_last, squared=False):.2f}, R2: {r2_score(y_true_last, y_pred_last):.3f}, NASA: {nasa_score_numpy(y_true_last, y_pred_last):.2f}")

