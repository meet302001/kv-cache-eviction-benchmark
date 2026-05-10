"""
Benchmark analysis and visualization.

Reads the benchmark summary CSV and per-run CSVs from loadtest/results/
and generates comparison charts.

Usage:
    python analyze.py                                    # auto-find latest summary
    python analyze.py --summary results/benchmark_summary_20260510.csv
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

RESULTS_DIR = Path(__file__).parent / "results"

POLICY_ORDER = ["fifo", "lru", "sjf", "cost_aware", "random"]
POLICY_COLORS = {
    "fifo": "#4C72B0",
    "lru": "#55A868",
    "sjf": "#C44E52",
    "cost_aware": "#8172B2",
    "random": "#CCB974",
}
WORKLOAD_ORDER = ["chatbot", "batch", "mixed", "burst"]


def load_summary(path: str | None = None) -> pd.DataFrame:
    """Load the benchmark summary CSV (auto-finds latest if not specified)."""
    if path:
        return pd.read_csv(path)

    pattern = str(RESULTS_DIR / "benchmark_summary_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No summary files found matching {pattern}")
        sys.exit(1)

    latest = files[-1]
    print(f"Loading: {latest}")
    return pd.read_csv(latest)


def plot_completion_rate(df: pd.DataFrame, out_dir: Path):
    """Bar chart: completion rate by policy, grouped by workload."""
    fig, ax = plt.subplots(figsize=(12, 6))

    workloads = [w for w in WORKLOAD_ORDER if w in df["workload"].values]
    policies = [p for p in POLICY_ORDER if p in df["policy"].values]
    x = np.arange(len(workloads))
    width = 0.15

    for i, policy in enumerate(policies):
        vals = []
        for wl in workloads:
            row = df[(df["policy"] == policy) & (df["workload"] == wl)]
            vals.append(row["completion_rate"].values[0] * 100 if len(row) else 0)
        ax.bar(x + i * width, vals, width, label=policy, color=POLICY_COLORS.get(policy))

    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Request Completion Rate by Policy and Workload")
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(workloads)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "completion_rate.png", dpi=150)
    plt.close(fig)
    print("  Saved: completion_rate.png")


def plot_ttft(df: pd.DataFrame, out_dir: Path):
    """Grouped bar chart: TTFT p50/p95 by policy, per workload."""
    workloads = [w for w in WORKLOAD_ORDER if w in df["workload"].values]
    policies = [p for p in POLICY_ORDER if p in df["policy"].values]

    fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 6), sharey=True)
    if len(workloads) == 1:
        axes = [axes]

    for ax, wl in zip(axes, workloads):
        subset = df[df["workload"] == wl]
        x = np.arange(len(policies))

        p50 = [subset[subset["policy"] == p]["ttft_p50"].values[0] if len(subset[subset["policy"] == p]) else 0 for p in policies]
        p95 = [subset[subset["policy"] == p]["ttft_p95"].values[0] if len(subset[subset["policy"] == p]) else 0 for p in policies]

        ax.bar(x - 0.15, p50, 0.3, label="p50", color="#4C72B0", alpha=0.8)
        ax.bar(x + 0.15, p95, 0.3, label="p95", color="#C44E52", alpha=0.8)

        ax.set_title(f"{wl}")
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("TTFT (seconds)")
    axes[0].legend()
    fig.suptitle("Time to First Token by Policy and Workload", fontsize=14)

    plt.tight_layout()
    fig.savefig(out_dir / "ttft_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: ttft_comparison.png")


def plot_wasted_compute(df: pd.DataFrame, out_dir: Path):
    """Bar chart: total tokens wasted by policy, grouped by workload."""
    fig, ax = plt.subplots(figsize=(12, 6))

    workloads = [w for w in WORKLOAD_ORDER if w in df["workload"].values]
    policies = [p for p in POLICY_ORDER if p in df["policy"].values]
    x = np.arange(len(workloads))
    width = 0.15

    for i, policy in enumerate(policies):
        vals = []
        for wl in workloads:
            row = df[(df["policy"] == policy) & (df["workload"] == wl)]
            vals.append(row["tokens_wasted"].values[0] if len(row) else 0)
        ax.bar(x + i * width, vals, width, label=policy, color=POLICY_COLORS.get(policy))

    ax.set_ylabel("Tokens Wasted")
    ax.set_title("Wasted Compute (Evicted Tokens) by Policy and Workload")
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(workloads)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "wasted_compute.png", dpi=150)
    plt.close(fig)
    print("  Saved: wasted_compute.png")


def plot_throughput(df: pd.DataFrame, out_dir: Path):
    """Bar chart: throughput (req/s) by policy, grouped by workload."""
    fig, ax = plt.subplots(figsize=(12, 6))

    workloads = [w for w in WORKLOAD_ORDER if w in df["workload"].values]
    policies = [p for p in POLICY_ORDER if p in df["policy"].values]
    x = np.arange(len(workloads))
    width = 0.15

    for i, policy in enumerate(policies):
        vals = []
        for wl in workloads:
            row = df[(df["policy"] == policy) & (df["workload"] == wl)]
            vals.append(row["throughput_rps"].values[0] if len(row) else 0)
        ax.bar(x + i * width, vals, width, label=policy, color=POLICY_COLORS.get(policy))

    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Request Throughput by Policy and Workload")
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(workloads)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "throughput.png", dpi=150)
    plt.close(fig)
    print("  Saved: throughput.png")


def plot_e2e_latency(df: pd.DataFrame, out_dir: Path):
    """Grouped bar chart: E2E latency p50/p95 by policy, per workload."""
    workloads = [w for w in WORKLOAD_ORDER if w in df["workload"].values]
    policies = [p for p in POLICY_ORDER if p in df["policy"].values]

    fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 6), sharey=True)
    if len(workloads) == 1:
        axes = [axes]

    for ax, wl in zip(axes, workloads):
        subset = df[df["workload"] == wl]
        x = np.arange(len(policies))

        p50 = [subset[subset["policy"] == p]["e2e_p50"].values[0] if len(subset[subset["policy"] == p]) else 0 for p in policies]
        p95 = [subset[subset["policy"] == p]["e2e_p95"].values[0] if len(subset[subset["policy"] == p]) else 0 for p in policies]

        ax.bar(x - 0.15, p50, 0.3, label="p50", color="#4C72B0", alpha=0.8)
        ax.bar(x + 0.15, p95, 0.3, label="p95", color="#C44E52", alpha=0.8)

        ax.set_title(f"{wl}")
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("E2E Latency (seconds)")
    axes[0].legend()
    fig.suptitle("End-to-End Latency by Policy and Workload", fontsize=14)

    plt.tight_layout()
    fig.savefig(out_dir / "e2e_latency.png", dpi=150)
    plt.close(fig)
    print("  Saved: e2e_latency.png")


def plot_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap: policy × workload → throughput."""
    workloads = [w for w in WORKLOAD_ORDER if w in df["workload"].values]
    policies = [p for p in POLICY_ORDER if p in df["policy"].values]

    matrix = np.zeros((len(policies), len(workloads)))
    for i, policy in enumerate(policies):
        for j, wl in enumerate(workloads):
            row = df[(df["policy"] == policy) & (df["workload"] == wl)]
            matrix[i, j] = row["throughput_rps"].values[0] if len(row) else 0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(workloads)))
    ax.set_yticks(np.arange(len(policies)))
    ax.set_xticklabels(workloads)
    ax.set_yticklabels(policies)

    for i in range(len(policies)):
        for j in range(len(workloads)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")

    ax.set_title("Throughput Heatmap (req/s): Policy × Workload")
    fig.colorbar(im, label="req/s")

    plt.tight_layout()
    fig.savefig(out_dir / "throughput_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved: throughput_heatmap.png")


def plot_eviction_scatter(df: pd.DataFrame, out_dir: Path):
    """Scatter: tokens wasted vs completion rate, per policy (all workloads)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    policies = [p for p in POLICY_ORDER if p in df["policy"].values]

    for policy in policies:
        subset = df[df["policy"] == policy]
        ax.scatter(
            subset["tokens_wasted"],
            subset["completion_rate"] * 100,
            label=policy,
            color=POLICY_COLORS.get(policy),
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["workload"],
                (row["tokens_wasted"], row["completion_rate"] * 100),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_xlabel("Tokens Wasted")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Wasted Compute vs Completion Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "waste_vs_completion.png", dpi=150)
    plt.close(fig)
    print("  Saved: waste_vs_completion.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--summary", type=str, default=None, help="Path to summary CSV")
    args = parser.parse_args()

    df = load_summary(args.summary)

    out_dir = RESULTS_DIR / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating charts to {out_dir}/\n")

    plot_completion_rate(df, out_dir)
    plot_ttft(df, out_dir)
    plot_wasted_compute(df, out_dir)
    plot_throughput(df, out_dir)
    plot_e2e_latency(df, out_dir)
    plot_heatmap(df, out_dir)
    plot_eviction_scatter(df, out_dir)

    print(f"\nDone! {7} charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
