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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_cmapss_data(DATANAME)
    train_std, test_std, _ = data_processing(train_raw, test_raw)

    exclude = ["engine_id", "cycle", "RUL"]
    sensor_cols = [c for c in train_std.columns if c not in exclude]

    print(f"Train shape: {train_std.shape}, engines: {train_std.engine_id.nunique()}, cycles: {train_std.cycle.min()}-{train_std.cycle.max()}")
    print(f"Test shape:  {test_std.shape},  engines: {test_std.engine_id.nunique()}, cycles: {test_std.cycle.min()}-{test_std.cycle.max()}")

    rul = train_std["RUL"]
    print("RUL stats (train): min={:.1f}, mean={:.1f}, median={:.1f}, 90th={:.1f}, max={:.1f}".format(
        rul.min(), rul.mean(), np.median(rul), np.percentile(rul, 90), rul.max()
    ))

    plots = {
        "corr_subset_pearson.png": plot_corr_heatmap(train_std, sensor_cols, method="pearson"),
        "corr_subset_spearman.png": plot_corr_heatmap(train_std, sensor_cols, method="spearman"),
    }


    fig_init_train, fig_init_test = plot_initial_rul(train_std, test_std)
    plots.update({
        "initial_rul_train.png": fig_init_train,
        "initial_rul_test.png": fig_init_test,
    })

    for name, fig in plots.items():
        out_path = OUTPUT_DIR / name
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")

    print("EDA complete. Inspect PNGs in output/eda/.")


if __name__ == "__main__":
    main()
