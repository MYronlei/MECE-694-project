#!/usr/bin/env python3
"""
Lightweight EDA for CMAPSS FD001 using existing load/data_processing helpers.
Generates a few summary plots into output/eda/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from step1_ml_pipeline import load_cmapss_data, data_processing

DATANAME = "FD001"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "eda"


def plot_rul_hist(train_df):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train_df["RUL"], bins=60, color="#3b82f6", edgecolor="white", alpha=0.9)
    ax.set_title("RUL distribution (train)")
    ax.set_xlabel("RUL")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    return fig


def plot_sensor_spread(train_df, sensor_cols, top_k=15):
    stats = train_df[sensor_cols].agg(["mean", "std"]).T
    top = stats.sort_values("std", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top["std"], color="#10b981", alpha=0.8)
    ax.set_title(f"Top {top_k} sensors by std (scaled)")
    ax.set_xlabel("Std Dev")
    ax.invert_yaxis()
    return fig


def plot_rul_trajectories(train_df, n_engines=6):
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_ids = train_df["engine_id"].unique()[:n_engines]
    for eid in sample_ids:
        eng = train_df[train_df.engine_id == eid].sort_values("cycle")
        ax.plot(eng["cycle"], eng["RUL"], label=f"Engine {eid}")
    ax.set_title("RUL trajectories for sample engines")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    return fig


def plot_corr_heatmap(train_df, sensor_cols, max_cols=12, method="pearson"):
    subset = sensor_cols[:max_cols]
    corr = train_df[subset + ["RUL"]].corr(method=method)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(f"Correlation heatmap (subset, {method})")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="corr")
    return fig


def plot_complete_corr_heatmap(train_df, sensor_cols, method="pearson"):
    """Generate complete correlation heatmap for all features including RUL."""
    # Calculate correlation matrix for all sensors plus RUL
    corr = train_df[sensor_cols + ["RUL"]].corr(method=method)
    
    # Determine figure size based on number of features
    n_features = len(corr.columns)
    fig_size = max(12, n_features * 0.4)  # Scale figure size with number of features
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    
    # Adjust font size based on number of features
    fontsize = max(6, min(10, 120 / n_features))
    ax.set_xticklabels(corr.columns, rotation=90, ha="right", fontsize=fontsize)
    ax.set_yticklabels(corr.index, fontsize=fontsize)
    
    ax.set_title(f"Complete Correlation Heatmap - All Features ({method.capitalize()})", 
                 fontsize=14, fontweight='bold', pad=20)
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    cbar.ax.tick_params(labelsize=10)
    
    # Add gridlines for better readability
    ax.set_xticks([x - 0.5 for x in range(1, len(corr.columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(corr.index))], minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)
    
    return fig


def plot_sensor_time(train_df, test_df, sensor_cols, top_k=3):
    """Plot top-variance sensors vs cycle for one train and one test engine."""
    stats = train_df[sensor_cols].agg(["std"]).T.sort_values("std", ascending=False)
    top_sensors = stats.head(top_k).index.tolist()

    # pick one representative engine from train and test
    train_eid = int(train_df["engine_id"].iloc[0])
    test_eid = int(test_df["engine_id"].iloc[0])

    def _plot_one(df, eid, title):
        fig, ax = plt.subplots(figsize=(8, 5))
        eng = df[df.engine_id == eid].sort_values("cycle")
        for s in top_sensors:
            ax.plot(eng["cycle"], eng[s], label=s)
        ax.set_title(title)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Scaled sensor value")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        return fig

    fig_train = _plot_one(train_df, train_eid, f"Top sensors vs cycle (train engine {train_eid})")
    fig_test = _plot_one(test_df, test_eid, f"Top sensors vs cycle (test engine {test_eid})")
    return fig_train, fig_test


def plot_initial_rul(train_df, test_df):
    """Plot initial RUL per engine for train and test sets."""
    def _initial(df):
        return (
            df.sort_values(["engine_id", "cycle"])
            .groupby("engine_id")
            .head(1)[["engine_id", "RUL"]]
        )

    init_train = _initial(train_df)
    init_test = _initial(test_df)

    def _plot(data, title):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(data["engine_id"], data["RUL"], color="#6366f1", alpha=0.85, width=0.8)
        ax.set_title(title)
        ax.set_xlabel("Engine ID")
        ax.set_ylabel("Initial RUL")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlim(0.5, data["engine_id"].max() + 0.5)
        return fig

    fig_train = _plot(init_train, "Initial RUL per engine (train)")
    fig_test = _plot(init_test, "Initial RUL per engine (test)")
    return fig_train, fig_test


def plot_training_history(history_path):
    """Plot training and validation loss convergence over epochs."""
    if not history_path.exists():
        print(f"Training history file not found: {history_path}")
        return None
    
    df_history = pd.read_csv(history_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    if 'loss' in df_history.columns and 'val_loss' in df_history.columns:
        epochs = range(1, len(df_history) + 1)
        ax1.plot(epochs, df_history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, df_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Model Loss Convergence', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
    
    # MAE plot (if available)
    mae_cols = [c for c in df_history.columns if 'mae' in c.lower()]
    if len(mae_cols) >= 2:
        train_mae = [c for c in mae_cols if 'val' not in c][0]
        val_mae = [c for c in mae_cols if 'val' in c][0]
        epochs = range(1, len(df_history) + 1)
        ax2.plot(epochs, df_history[train_mae], 'b-', label='Training MAE', linewidth=2)
        ax2.plot(epochs, df_history[val_mae], 'r-', label='Validation MAE', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('MAE', fontsize=11)
        ax2.set_title('Mean Absolute Error Convergence', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    else:
        # If no MAE, plot learning rate or other metric
        lr_col = [c for c in df_history.columns if 'lr' in c.lower()]
        if lr_col:
            epochs = range(1, len(df_history) + 1)
            ax2.plot(epochs, df_history[lr_col[0]], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Learning Rate', fontsize=11)
            ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(alpha=0.3)
    
    return fig


def plot_pred_vs_true(pred_path, title, per_engine_last=False):
    """Scatter plot of predicted vs true RUL from model outputs."""
    if not pred_path.exists():
        print(f"Prediction file not found: {pred_path}")
        return None
    df = pd.read_csv(pred_path)
    required = {"true_RUL", "pred"}
    if per_engine_last:
        required |= {"engine_id", "cycle"}
    if not required.issubset(df.columns):
        print(f"Prediction file missing required columns: {pred_path}")
        return None

    if per_engine_last:
        df = (
            df.sort_values(["engine_id", "cycle"])
            .groupby("engine_id")
            .tail(1)
            .reset_index(drop=True)
        )

    err = (df["pred"] - df["true_RUL"]).abs()
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=err.min(), vmax=err.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        df["true_RUL"],
        df["pred"],
        c=err,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        s=14,
        edgecolor="none",
    )
    lims = [
        min(df["true_RUL"].min(), df["pred"].min()),
        max(df["true_RUL"].max(), df["pred"].max()),
    ]
    ax.plot(lims, lims, "k--", lw=1, label="Ideal")
    ax.plot(lims, [l + 20 for l in lims], "k:", lw=1, label="+20")
    ax.plot(lims, [l - 20 for l in lims], "k-.", lw=1, label="-20")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(title)
    ax.legend()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Pred - True|")
    ax.grid(alpha=0.2)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_cmapss_data(DATANAME)
    train_std, test_std, _ = data_processing(train_raw, test_raw)

    exclude = ["engine_id", "cycle", "RUL"]
    sensor_cols = [c for c in train_std.columns if c not in exclude]
    corr_series = train_std[sensor_cols + ["RUL"]].corr(method="pearson")["RUL"].drop("RUL")
    top4_corr_sensors = corr_series.abs().sort_values(ascending=False).head(4).index.tolist()

    print(f"Train shape: {train_std.shape}, engines: {train_std.engine_id.nunique()}, cycles: {train_std.cycle.min()}-{train_std.cycle.max()}")
    print(f"Test shape:  {test_std.shape},  engines: {test_std.engine_id.nunique()}, cycles: {test_std.cycle.min()}-{test_std.cycle.max()}")

    rul = train_std["RUL"]
    print("RUL stats (train): min={:.1f}, mean={:.1f}, median={:.1f}, 90th={:.1f}, max={:.1f}".format(
        rul.min(), rul.mean(), np.median(rul), np.percentile(rul, 90), rul.max()
    ))

    plots = {
        "corr_complete_pearson.png": plot_complete_corr_heatmap(train_std, sensor_cols, method="pearson"),
        "corr_subset_pearson.png": plot_corr_heatmap(train_std, sensor_cols, method="pearson"),
        "corr_top4_pearson.png": plot_corr_heatmap(train_std, top4_corr_sensors, max_cols=4, method="pearson"),
    }


    fig_init_train, fig_init_test = plot_initial_rul(train_std, test_std)
    plots.update({
        "initial_rul_train.png": fig_init_train,
        "initial_rul_test.png": fig_init_test,
    })

    pred_path = Path(__file__).resolve().parent / "output" / "rul_predictions_all_windows.csv"
    scatter_fig = plot_pred_vs_true(
        pred_path,
        "Predicted vs True RUL (test engines: last window)",
        per_engine_last=True,
    )
    if scatter_fig is not None:
        plots["pred_vs_true_last_scatter.png"] = scatter_fig

    # Add training history plot
    history_path = Path(__file__).resolve().parent / "output" / "training_history.csv"
    history_fig = plot_training_history(history_path)
    if history_fig is not None:
        plots["training_convergence.png"] = history_fig

    for name, fig in plots.items():
        out_path = OUTPUT_DIR / name
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")

    print("EDA complete. Inspect PNGs in output/eda/.")


if __name__ == "__main__":
    main()
