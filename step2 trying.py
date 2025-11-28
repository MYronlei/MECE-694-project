# =============================================================================
# SECTION: Imports & Global Settings
# Purpose: gather dependencies and configure dataset/training constants.
# =============================================================================

# Hybrid Conv1D + LSTM model for CMAPSS FD001 using helpers from step1_ml_pipeline.

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
VERBOSE_DIAGNOSTICS = False  # set True to print extra range stats at end

USE_SAMPLE_WEIGHTS = True
LAST_K_WINDOWS = 3

CYCLE_EXPORT_TARGETS = [60, 90]  # per-engine snapshots near these cycles

CAP_RUL = MAX_RUL  # numeric cap reused to keep window builder logic intact
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_K_CORR_FEATURES = 11 # number of features to keep by |Pearson| corr with RUL

# =============================================================================
# SECTION: Data Loading & Standardization
# Purpose: ingest CMAPSS FD001 and normalize features for modeling.
# =============================================================================

train_df_raw, test_df_raw = load_cmapss_data(DATANAME)
train_std, test_std, scaler = data_processing(train_df_raw, test_df_raw)

print("Columns from standardized train:", list(train_std.columns))

# =============================================================================
# SECTION: Engine-Level Train/Val Split
# Purpose: split engines into train/validation partitions reproducibly.
# =============================================================================

rng = np.random.default_rng(seed=17)
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

# =============================================================================
# SECTION: Data-Driven RUL Capping
# Purpose: analyze RUL distribution and enforce an empirical ceiling.
# =============================================================================

print("\n=== Analyzing RUL distribution to find optimal cap ===")
rul_train = train_df["RUL"].values
rul_stats = {
    "min": rul_train.min(),
    "mean": rul_train.mean(),
    "median": np.median(rul_train),
    "90th": np.percentile(rul_train, 90),
    "95th": np.percentile(rul_train, 95),
    "99th": np.percentile(rul_train, 99),
    "max": rul_train.max()
}
print(f"RUL stats: min={rul_stats['min']:.1f}, mean={rul_stats['mean']:.1f}, "
      f"median={rul_stats['median']:.1f}, 90%={rul_stats['90th']:.1f}, "
      f"95%={rul_stats['95th']:.1f}, 99%={rul_stats['99th']:.1f}, max={rul_stats['max']:.1f}")

# Test candidate caps: no cap, 95th percentile, 125 (common in literature), current MAX_RUL
candidate_caps = [
    None,  # No cap
    rul_stats['95th'],
    125,  # Common threshold in CMAPSS literature
    MAX_RUL
]

# Compute percentage of data affected by each cap
print("\nCandidate MAX_RUL values and impact:")
for cap in candidate_caps:
    if cap is None:
        pct_affected = 0
        print(f"  No cap: 0.0% of data affected")
    else:
        pct_affected = 100 * np.mean(rul_train > cap)
        print(f"  {cap:.1f}: {pct_affected:.1f}% of data will be capped")

# Use 95th percentile as optimal (balances outlier reduction with data retention)
OPTIMAL_MAX_RUL = rul_stats['95th']
print(f"\n✓ Selected optimal MAX_RUL = {OPTIMAL_MAX_RUL:.1f} (95th percentile)")
print(f"  This will cap {100 * np.mean(rul_train > OPTIMAL_MAX_RUL):.1f}% of high-RUL values")

# Override MAX_RUL and CAP_RUL with optimal value
MAX_RUL = OPTIMAL_MAX_RUL
CAP_RUL = MAX_RUL

# Cap targets to optimal MAX_RUL
if CAP_RUL:
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df["RUL"] = np.minimum(train_df["RUL"].values, MAX_RUL)
    val_df["RUL"]   = np.minimum(val_df["RUL"].values, MAX_RUL)
    test_df["RUL"]  = np.minimum(test_df["RUL"].values, MAX_RUL)

# =============================================================================
# SECTION: Feature Selection via Pearson Correlation
# Purpose: keep top raw features most correlated with RUL (no KPCA).
# =============================================================================

exclude = ["engine_id", "cycle", "RUL"]
original_features = [c for c in train_df.columns if c not in exclude]

corr_series = train_df[original_features + ["RUL"]].corr(method="pearson")["RUL"].drop("RUL")
feature_cols = corr_series.abs().sort_values(ascending=False).head(TOP_K_CORR_FEATURES).index.tolist()
print(f"Selected {len(feature_cols)} features by |Pearson| correlation with RUL:")
print(feature_cols)

# =============================================================================
# SECTION: Window Builder Utility
# Purpose: create sliding windows with multiple sampling strategies.
# =============================================================================

def make_windows(
    df,
    feature_cols,
    window_size=30,
    strategy="all",
    last_k=3,
    bins=None,
    max_per_bin=8,
    stride=1,
    cap_rul=None,
    return_meta=False,
):
    """Create sliding windows using 'all', 'last_k', or 'uniform_bins' sampling."""
    X_list, y_list = [], []
    eng_list, cyc_list = [], []

    engine_groups = df.groupby("engine_id")

    # Prepare bins for uniform sampling
    if strategy == "uniform_bins":
        if bins is None:
            bins = [0, 10, 25, 50, 80, 110, 130]
        bins = np.array(bins)

    for eng_id, eng_df in engine_groups:
        eng_df = eng_df.sort_values("cycle").reset_index(drop=True)

        feats  = eng_df[feature_cols].to_numpy(dtype=float)
        rul    = eng_df["RUL"].to_numpy(dtype=float)
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
            rng_local = np.random.default_rng(seed=17)
            for b in np.unique(bin_indices):
                b_indices = [
                    candidate_starts[i]
                    for i in range(len(candidate_starts))
                    if bin_indices[i] == b
                ]
                if len(b_indices) > max_per_bin:
                    b_indices = rng_local.choice(b_indices, size=max_per_bin, replace=False).tolist()
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

# =============================================================================
# SECTION: Windowed Dataset Construction
# Purpose: generate balanced train/val windows and dense test windows.
# =============================================================================

WINDOW_STRIDE = 1
UNIFORM_MAX_PER_BIN = 8

X_tr, y_tr = make_windows(
    train_df,
    feature_cols,
    window_size=WINDOW_SIZE,
    strategy="uniform_bins",
    max_per_bin=UNIFORM_MAX_PER_BIN,
    cap_rul=CAP_RUL,
)

X_val, y_val = make_windows(
    val_df,
    feature_cols,
    window_size=WINDOW_SIZE,
    strategy="uniform_bins",
    max_per_bin=UNIFORM_MAX_PER_BIN,
    cap_rul=CAP_RUL,
)

X_te, y_te, eng_te, cyc_te = make_windows(
    test_df,
    feature_cols,
    window_size=WINDOW_SIZE,
    strategy="all",
    stride=WINDOW_STRIDE,
    cap_rul=CAP_RUL,
    return_meta=True,
)



# =============================================================================
# SECTION: Sample Weighting
# Purpose: emphasize low-RUL regimes during training via simple heuristics.
# =============================================================================

def compute_sample_weights(y, max_rul=MAX_RUL):
    w = 1.0 + (max_rul - np.clip(y, 0, max_rul)) / float(max_rul)
    return w

w_tr = compute_sample_weights(y_tr) if USE_SAMPLE_WEIGHTS else None

# =============================================================================
# SECTION: Model Definition
# Purpose: assemble Conv-BiLSTM-Attention architecture for RUL regression.
# =============================================================================

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
inp = keras.Input(shape=(WINDOW_SIZE, n_features))

# Conv front-end
x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

# BiLSTM stack
x = layers.Bidirectional(
    layers.LSTM(128, return_sequences=True, dropout=0.15)
)(x)
x = layers.Bidirectional(
    layers.LSTM(64, return_sequences=True, dropout=0.10)
)(x)

# Attention pooling
context = TemporalAttention(units=64)(x)

# Dense regression head
h = layers.Dense(
    64,
    activation="relu",
    kernel_regularizer=keras.regularizers.l2(1e-4),
)(context)
h = layers.Dropout(0.2)(h)
out = layers.Dense(1, activation="linear")(h)

model = keras.Model(inputs=inp, outputs=out)
model.summary()

# =============================================================================
# SECTION: Compilation & Training Setup
# Purpose: configure loss, metrics, and callbacks before fitting.
# =============================================================================

loss_fn = keras.losses.Huber(delta=10.0)
opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

def nasa_metric(y_true, y_pred):
    d = y_pred - y_true
    s = tf.where(
        d >= 0,
        tf.exp(d / 13.0) - 1.0,
        tf.exp(-d / 10.0) - 1.0,
    )
    return tf.reduce_mean(s)

model.compile(optimizer=opt, loss=loss_fn, metrics=[keras.metrics.MeanAbsoluteError(name="MAE"), nasa_metric])

# Training callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1,
)
ckpt = keras.callbacks.ModelCheckpoint(
    str(OUTPUT_DIR / "best_cnn_lstm_rul.keras"),
    monitor="val_loss",
    save_best_only=True,
)

# =============================================================================
# SECTION: Supervised Training
# Purpose: fit the model on windowed data with callbacks and weights.
# =============================================================================

history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    sample_weight=w_tr if USE_SAMPLE_WEIGHTS else None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr, ckpt],
    verbose=2,
)

# =============================================================================
# SECTION: Evaluation & Diagnostics
# Purpose: score the model on dense test windows and engine aggregates.
# =============================================================================

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

df_win = pd.DataFrame(
    {
        "engine_id": eng_te,
        "cycle": cyc_te,
        "true_RUL": y_te,
        "pred": y_pred_te,
    }
)

# Get first window per engine (early prediction)
df_first = (
    df_win.sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .head(1)
    .reset_index(drop=True)
)

y_true_first = df_first["true_RUL"].to_numpy()
y_pred_first = df_first["pred"].to_numpy()

print("Per-engine first-window MAE:", mean_absolute_error(y_true_first, y_pred_first))
print("Per-engine first-window RMSE:", mean_squared_error(y_true_first, y_pred_first, squared=False))
print("Per-engine first-window R2:", r2_score(y_true_first, y_pred_first))
print("Per-engine first-window NASA:", nasa_score_numpy(y_true_first, y_pred_first))

# Also compute last window for comparison
df_last = (
    df_win.sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
    .reset_index(drop=True)
)

y_true_last = df_last["true_RUL"].to_numpy()
y_pred_last = df_last["pred"].to_numpy()

print("Per-engine last-window MAE:", mean_absolute_error(y_true_last, y_pred_last))
print("Per-engine last-window RMSE:", mean_squared_error(y_true_last, y_pred_last, squared=False))
print("Per-engine last-window R2:", r2_score(y_true_last, y_pred_last))
print("Per-engine last-window NASA:", nasa_score_numpy(y_true_last, y_pred_last))

# =============================================================================
# SECTION: Prediction Persistence
# Purpose: export per-window and per-engine CSVs for optimization.
# =============================================================================

per_window_path = OUTPUT_DIR / "rul_predictions_all_windows.csv"
per_engine_first_path = OUTPUT_DIR / "rul_predictions_per_engine_first.csv"
per_engine_last_path = OUTPUT_DIR / "rul_predictions_per_engine_last.csv"

df_win.to_csv(per_window_path, index=False)

# Save first window predictions
df_first_rounded = df_first.copy()
df_first_rounded["pred_RUL"] = np.floor(df_first_rounded["pred"]).astype(int)
df_first_rounded = df_first_rounded.drop(columns=["pred"]).rename(columns={"true_RUL": "true_RUL", "pred_RUL": "pred_RUL"})
df_first_rounded.to_csv(per_engine_first_path, index=False)

# Save last window predictions
df_last_rounded = df_last.copy()
df_last_rounded["pred_RUL"] = np.floor(df_last_rounded["pred"]).astype(int)
df_last_rounded = df_last_rounded.drop(columns=["pred"]).rename(columns={"true_RUL": "true_RUL", "pred_RUL": "pred_RUL"})
df_last_rounded.to_csv(per_engine_last_path, index=False)

# Save per-engine snapshots near target cycles
def save_cycle_snapshot(df_source, cycle_target):
    rows = []
    for eng_id, eng_df in df_source.groupby("engine_id"):
        idx = (eng_df["cycle"] - cycle_target).abs().values.argmin()
        rows.append(eng_df.iloc[idx])
    df_cycle = pd.DataFrame(rows).reset_index(drop=True)
    df_cycle["pred_RUL"] = np.floor(df_cycle["pred"]).astype(int)
    df_cycle = df_cycle.drop(columns=["pred"]).rename(columns={"true_RUL": "true_RUL", "pred_RUL": "pred_RUL"})
    out_path = OUTPUT_DIR / f"rul_predictions_per_engine_cycle{cycle_target}.csv"
    df_cycle.to_csv(out_path, index=False)
    print(f"Saved per-engine cycle~{cycle_target} predictions to {out_path}")

for target_cycle in CYCLE_EXPORT_TARGETS:
    save_cycle_snapshot(df_win, target_cycle)

print(f"Saved per-window predictions to {per_window_path}")
print(f"Saved per-engine FIRST-window predictions to {per_engine_first_path}")
print(f"Saved per-engine LAST-window predictions to {per_engine_last_path}")

# =============================================================================
# SECTION: Final Range Diagnostics
# Purpose: ensure predicted/test RUL ranges remain sane.
# =============================================================================

if VERBOSE_DIAGNOSTICS:
    print("y_tr range:", np.min(y_tr), np.mean(y_tr), np.max(y_tr))
    print("y_te range:", np.min(y_te), np.mean(y_te), np.max(y_te))
    print("y_pred_te range:", np.min(y_pred_te), np.mean(y_pred_te), np.max(y_pred_te))
