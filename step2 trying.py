# LSTM + Conv1D model for CMAPSS FD001
# Assumes: train_df, val_df, test_df (pandas), feature_cols (list), and your standardization already applied.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Replace the top of your script with the block below (loads data, creates train/val/test)
import numpy as np
import pandas as pd

from step1_ml_pipeline import load_cmapss_data, data_processing

# Load, preprocess and prepare feature list (paste this before make_windows)
import numpy as np
import pandas as pd

from step1_ml_pipeline import load_cmapss_data, data_processing

# CONFIG
DATANAME = "FD001"
VAL_FRAC = 0.15
MAX_RUL = 125
WINDOW_SIZE = 30  # ensure WINDOW_SIZE is defined before make_windows if used later

# 1) Load and standardize
train_df_raw, test_df_raw = load_cmapss_data(DATANAME)
train_std, test_std, scaler = data_processing(train_df_raw, test_df_raw)

print("Columns from standardized train:", list(train_std.columns))

# 2) Split engines for validation (engine-level split)
rng = np.random.default_rng(seed=42)
eng_ids = train_std["engine_id"].unique()
rng.shuffle(eng_ids)
cut = int((1.0 - VAL_FRAC) * len(eng_ids))
val_ids = set(eng_ids[cut:])
mask_val = train_std["engine_id"].isin(val_ids)

train_df = train_std[~mask_val].copy()
val_df   = train_std[mask_val].copy()
test_df  = test_std.copy()

print(f"# train engines: {train_df['engine_id'].nunique()}")
print(f"#  val  engines: {val_df['engine_id'].nunique()}")
print(f"# test engines:  {test_df['engine_id'].nunique()}")

# 3) Cap RUL (if you want)
train_df["RUL"] = np.minimum(train_df["RUL"].values, MAX_RUL)
val_df["RUL"]   = np.minimum(val_df["RUL"].values, MAX_RUL)
test_df["RUL"]  = np.minimum(test_df["RUL"].values, MAX_RUL)

# 4) Define feature columns (exclude metadata/target)
exclude = ["engine_id", "cycle", "RUL"]
feature_cols = [c for c in train_df.columns if c not in exclude]
print("Num sensor/features:", len(feature_cols))
print("Feature columns:", feature_cols)
# Now continue with feature_cols, windows, model building, etc.
# ---------- hyperparams ----------
WINDOW_SIZE = 30
MAX_RUL = 125.0
BATCH_SIZE = 256
EPOCHS = 70
LEARNING_RATE = 1e-3
CAP_RUL = True           # set True to cap RUL at MAX_RUL
USE_SAMPLE_WEIGHTS = True
LAST_K_WINDOWS = 3       # per-engine windows used for training (1 = last only)
# ---------------------------------

# Optional: cap RUL in dataframes
if CAP_RUL:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["RUL"] = np.minimum(train_df["RUL"].values, MAX_RUL)
    val_df["RUL"]   = np.minimum(val_df["RUL"].values, MAX_RUL)
    test_df["RUL"]  = np.minimum(test_df["RUL"].values, MAX_RUL)


import numpy as np
import pandas as pd

def make_windows(
    df,
    feature_cols,
    window_size=30,
    strategy="all",          # "all", "last_k", "uniform_bins"
    last_k=3,
    bins=None,               # list of bin edges for uniform_bins
    max_per_bin=8,           # max windows per bin per engine
    stride=1,                # stride for "all"
    cap_rul=None,
    return_meta=False,
):
    """
    Build supervised windows with different sampling strategies.
    
    strategy:
      - "all": standard sliding windows over entire engine with given stride.
      - "last_k": last_k windows ending at final cycles (what you had).
      - "uniform_bins": sample up to max_per_bin windows per RUL bin per engine.
      
    If cap_rul is not None, target is clipped to that value.
    """
    X_list, y_list = [], []          # <-- FIXED
    eng_list, cyc_list = [], []
    
    engine_groups = df.groupby("engine_id")
    
    # Prepare bins
    if strategy == "uniform_bins":
        if bins is None:
            # Default edges covering FD001 range
            bins = [0, 10, 25, 50, 80, 110, 130]  # 130 acts as upper bound
        bins = np.array(bins)
    
    for eng_id, eng_df in engine_groups:
        eng_df = eng_df.sort_values("cycle").reset_index(drop=True)
        feats = eng_df[feature_cols].to_numpy(dtype=float)
        rul   = eng_df["RUL"].to_numpy(dtype=float)
        cycles = eng_df["cycle"].to_numpy(dtype=int)
        L = len(eng_df)
        if L < window_size:
            continue
        
        if strategy == "last_k":
            last_start = L - window_size
            starts = list(range(max(0, last_start - (last_k - 1)), last_start + 1))
        elif strategy == "all":
            starts = list(range(0, L - window_size + 1, stride))
        elif strategy == "uniform_bins":
            # Collect all possible windows first
            candidate_starts = list(range(0, L - window_size + 1))
            # Compute end RUL values for these candidates
            candidate_ruls = [rul[s + window_size - 1] for s in candidate_starts]
            # Digitize into bins
            bin_indices = np.digitize(candidate_ruls, bins) - 1  # bin numbers
            # For each bin pick up to max_per_bin starts (random subset if needed)
            starts = []
            rng = np.random.default_rng(seed=42)
            for b in np.unique(bin_indices):
                b_indices = [candidate_starts[i] for i in range(len(candidate_starts)) if bin_indices[i] == b]
                if len(b_indices) > max_per_bin:
                    b_indices = rng.choice(b_indices, size=max_per_bin, replace=False).tolist()
                starts.extend(b_indices)
            # Optional: sort starts to maintain temporal order
            starts.sort()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
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
    else:
        return X, y

# Example usage (replace your old window calls):
CAP_RUL = 125
WINDOW_SIZE = 30

# Balanced windows
X_tr, y_tr = make_windows (train_df, feature_cols, window_size=WINDOW_SIZE,
                           strategy="uniform_bins", cap_rul=CAP_RUL,
                           max_per_bin=8)

X_val, y_val = make_windows(val_df, feature_cols, window_size=WINDOW_SIZE,
                             strategy="uniform_bins", cap_rul=CAP_RUL,
                             max_per_bin=8)

X_te, y_te, eng_te, cyc_te = make_windows(test_df, feature_cols, window_size=WINDOW_SIZE,
                                           strategy="all", cap_rul=CAP_RUL,
                                           return_meta=True, stride=1)

print("Ranges after resampling:")
print("Train y:", y_tr.min(), y_tr.mean(), y_tr.max())
print("Val   y:", y_val.min(), y_val.mean(), y_val.max())
print("Test  y:", y_te.min(), y_te.mean(), y_te.max())

# Build datasets
X_tr, y_tr = make_windows(train_df, feature_cols, WINDOW_SIZE, return_meta=False, last_k=LAST_K_WINDOWS)
X_val, y_val = make_windows(val_df, feature_cols, WINDOW_SIZE, return_meta=False, last_k=LAST_K_WINDOWS)
X_te, y_te, eng_te, cyc_te = make_windows(test_df, feature_cols, WINDOW_SIZE, return_meta=True, last_k=1)

print("Shapes: X_tr, X_val, X_te:", X_tr.shape, X_val.shape, X_te.shape)

# 1) Quick numeric checks (you already printed some; run again for more detail)
import numpy as np
print("y_tr: n, min, mean, max:", len(y_tr), np.min(y_tr), np.mean(y_tr), np.max(y_tr))
print("y_val:", len(y_val), np.min(y_val), np.mean(y_val), np.max(y_val))
print("y_te: n, min, mean, max:", len(y_te), np.min(y_te), np.mean(y_te), np.max(y_te))


# 2) Inspect the RUL columns in your source dataframes
print("train_df RUL stats:", train_df["RUL"].describe())
print("val_df RUL stats:  ", val_df["RUL"].describe())
print("test_df RUL stats: ", test_df["RUL"].describe())

# 3) Inspect engineered features RUL value used in training (if using build_dataset_last_windows)
# If you built df_train_feats earlier:
if 'df_train_feats' in globals():
    print("df_train_feats['RUL'] stats:", df_train_feats["RUL"].describe())
if 'df_test_feats' in globals():
    print("df_test_feats['RUL'] stats:", df_test_feats["RUL"].describe())

# 4) Show a few rows to verify exact values
print("train_df sample rows:\n", train_df.head()[["engine_id","cycle","RUL"]])
print("train_df (tail):\n", train_df.groupby("engine_id").tail(3).head(6))
print("test_df sample rows:\n", test_df.head()[["engine_id","cycle","RUL"]])

# sample weights (emphasize low RUL)
def compute_sample_weights(y, max_rul=MAX_RUL):
    w = 1.0 + (max_rul - np.clip(y, 0, max_rul)) / float(max_rul)
    # map to a reasonable range (optional)
    return w

w_tr = compute_sample_weights(y_tr) if USE_SAMPLE_WEIGHTS else None
w_val = compute_sample_weights(y_val) if USE_SAMPLE_WEIGHTS else None

# ---------- model: Conv1D front-end -> BiLSTM -> Attention -> Dense ----------
class TemporalAttention(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1, use_bias=False)
    def call(self, inputs, mask=None):
        # inputs: (batch, time, dim)
        u = self.W(inputs)                  # (batch, time, units)
        scores = self.v(u)                  # (batch, time, 1)
        weights = tf.nn.softmax(scores, axis=1)  # across time
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

n_features = len(feature_cols)
inp = keras.Input(shape=(WINDOW_SIZE, n_features))

# Conv front-end: two conv blocks
x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

# BiLSTM (return sequences for attention)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.15))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.10))(x)

# Attention pooling
context = TemporalAttention(units=64)(x)

# Dense head
h = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(context)
h = layers.Dropout(0.2)(h)
out = layers.Dense(1, activation="linear")(h)

model = keras.Model(inputs=inp, outputs=out)
model.summary()

# compile
loss_fn = keras.losses.Huber(delta=10.0)
opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
def nasa_metric(y_true, y_pred):
    d = y_pred - y_true
    s = tf.where(
        d >= 0,
        tf.exp(d / 13.0) - 1.0,
        tf.exp(-d / 10.0) - 1.0
    )
    return tf.reduce_mean(s)

model.compile(optimizer=opt, loss=loss_fn, metrics=[keras.metrics.MeanAbsoluteError(name="MAE"), nasa_metric])

# callbacks
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
reduce_lr  = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)
ckpt = keras.callbacks.ModelCheckpoint("best_cnn_lstm_rul.h5", monitor="val_loss", save_best_only=True)

# fit
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val, w_val) if USE_SAMPLE_WEIGHTS else (X_val, y_val),
    sample_weight=w_tr if USE_SAMPLE_WEIGHTS else None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr, ckpt],
    verbose=2
)

# ---------- Evaluation ----------
# per-window test
y_pred_te = model.predict(X_te).ravel()
mae = mean_absolute_error(y_te, y_pred_te)
rmse = mean_squared_error(y_te, y_pred_te, squared=False)
r2 = r2_score(y_te, y_pred_te)
def nasa_score_numpy(y_true, y_pred):
    d = y_pred - y_true
    s = np.where(d >= 0, np.exp(d / 13.0) - 1.0, np.exp(-d / 10.0) - 1.0)
    return float(np.sum(s))
nasa = nasa_score_numpy(y_te, y_pred_te)
print("Per-window Test MAE:", mae, "RMSE:", rmse, "R2:", r2, "NASA:", nasa)

# per-engine last-window evaluation
import pandas as pd
df_win = pd.DataFrame({"engine_id": eng_te, "cycle": cyc_te, "true_RUL": y_te, "pred": y_pred_te})
df_last = df_win.sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1).reset_index(drop=True)
y_true_last = df_last["true_RUL"].to_numpy()
y_pred_last = df_last["pred"].to_numpy()
print("Per-engine last-window MAE:", mean_absolute_error(y_true_last, y_pred_last))
print("Per-engine last-window RMSE:", mean_squared_error(y_true_last, y_pred_last, squared=False))
print("Per-engine last-window R2:", r2_score(y_true_last, y_pred_last))
print("Per-engine last-window NASA:", nasa_score_numpy(y_true_last, y_pred_last))

# Quick diagnostics to check for scaling / exploding outputs
print("y_tr range:", np.min(y_tr), np.mean(y_tr), np.max(y_tr))
print("y_te range:", np.min(y_te), np.mean(y_te), np.max(y_te))
print("y_pred_te range:", np.min(y_pred_te), np.mean(y_pred_te), np.max(y_pred_te))