#!/usr/bin/env python3
"""
Generate throughput plots from bench_plot benchmark results.

Usage:
    cargo bench --no-default-features --bench bench_plot
    python3 benches/plot_benchmarks.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV = "target/bench_plots/throughput.csv"
OUT = "target/bench_plots"

COLORS = {
    "serial": "#2563eb",     # blue
    "parallel": "#ea580c",   # orange
}
MARKERS = {
    "serial": "o",
    "parallel": "s",
}

def fmt_rate(x, _):
    """Format large numbers: 1000 -> 1K, 1000000 -> 1M."""
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def plot_scaling(df, group, title, xlabel, ylabel, filename):
    """Plot serial vs parallel throughput scaling."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for variant in ["serial", "parallel"]:
        sub = df[(df.group == group) & (df.variant == variant)].sort_values("param")
        if sub.empty:
            continue
        ax.plot(
            sub.param, sub["mean"],
            marker=MARKERS[variant], color=COLORS[variant],
            linewidth=2, markersize=7, label=variant.capitalize(),
        )
        ax.fill_between(
            sub.param, sub.ci_low, sub.ci_high,
            alpha=0.15, color=COLORS[variant],
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_rate))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename + ".png"), dpi=150)
    fig.savefig(os.path.join(OUT, filename + ".svg"))
    plt.close()
    print(f"  -> {filename}.png / .svg")


def plot_batch(df):
    """Plot DTL batch-size scaling with 4 variants."""
    sub = df[df.group == "dtl_batch"].copy()
    if sub.empty:
        return

    style = {
        "pergene_serial":      {"color": "#2563eb", "ls": "-",  "marker": "o", "label": "Per-gene serial"},
        "pergene_parallel":    {"color": "#2563eb", "ls": "--", "marker": "s", "label": "Per-gene parallel"},
        "perspecies_serial":   {"color": "#16a34a", "ls": "-",  "marker": "^", "label": "Per-species serial"},
        "perspecies_parallel": {"color": "#16a34a", "ls": "--", "marker": "D", "label": "Per-species parallel"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, st in style.items():
        v = sub[sub.variant == variant].sort_values("param")
        if v.empty:
            continue
        ax.plot(
            v.param, v["mean"],
            linestyle=st["ls"], marker=st["marker"], color=st["color"],
            linewidth=2, markersize=7, label=st["label"],
        )
        ax.fill_between(v.param, v.ci_low, v.ci_high, alpha=0.12, color=st["color"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_rate))
    ax.set_xlabel("Batch Size (number of gene trees)", fontsize=12)
    ax.set_ylabel("Gene Trees / Second", fontsize=12)
    ax.set_title("DTL Batch Throughput (50 species)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dtl_batch_throughput.png"), dpi=150)
    fig.savefig(os.path.join(OUT, "dtl_batch_throughput.svg"))
    plt.close()
    print("  -> dtl_batch_throughput.png / .svg")


def plot_thread_scaling(df):
    """Plot speedup vs thread count."""
    sub = df[df.group == "thread_scaling"].copy()
    if sub.empty:
        return

    style = {
        "bd":             {"color": "#2563eb", "marker": "o", "label": "BD (1000 species)"},
        "dtl_pergene":    {"color": "#ea580c", "marker": "s", "label": "DTL per-gene (100 sp)"},
        "dtl_perspecies":  {"color": "#16a34a", "marker": "^", "label": "DTL per-species (100 sp)"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    max_threads = 1
    for variant, st in style.items():
        v = sub[sub.variant == variant].sort_values("param")
        if v.empty:
            continue
        baseline = v["mean"].iloc[0]
        speedup = v["mean"] / baseline
        speedup_lo = v.ci_low / baseline
        speedup_hi = v.ci_high / baseline
        ax.plot(
            v.param, speedup,
            marker=st["marker"], color=st["color"],
            linewidth=2, markersize=7, label=st["label"],
        )
        ax.fill_between(v.param, speedup_lo, speedup_hi, alpha=0.12, color=st["color"])
        max_threads = max(max_threads, int(v.param.max()))

    # Ideal scaling line
    ideal_x = np.arange(1, max_threads + 1)
    ax.plot(ideal_x, ideal_x, "k--", alpha=0.3, linewidth=1, label="Ideal")

    ax.set_xlabel("Thread Count", fontsize=12)
    ax.set_ylabel("Speedup (vs 1 thread)", fontsize=12)
    ax.set_title("Parallel Scaling Efficiency", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.5, max_threads + 0.5)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "thread_scaling.png"), dpi=150)
    fig.savefig(os.path.join(OUT, "thread_scaling.svg"))
    plt.close()
    print("  -> thread_scaling.png / .svg")


def plot_combined_overview(df):
    """Single figure with all scaling plots in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) BD scaling
    ax = axes[0, 0]
    for variant in ["serial", "parallel"]:
        s = df[(df.group == "bd") & (df.variant == variant)].sort_values("param")
        if s.empty:
            continue
        ax.plot(s.param, s["mean"], marker=MARKERS[variant], color=COLORS[variant],
                linewidth=2, markersize=6, label=variant.capitalize())
        ax.fill_between(s.param, s.ci_low, s.ci_high, alpha=0.15, color=COLORS[variant])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_rate))
    ax.set_xlabel("Number of Species"); ax.set_ylabel("Species Trees / s")
    ax.set_title("Birth-Death Throughput", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2)

    # (0,1) DTL per-gene scaling
    ax = axes[0, 1]
    for variant in ["serial", "parallel"]:
        s = df[(df.group == "dtl_pergene") & (df.variant == variant)].sort_values("param")
        if s.empty:
            continue
        ax.plot(s.param, s["mean"], marker=MARKERS[variant], color=COLORS[variant],
                linewidth=2, markersize=6, label=variant.capitalize())
        ax.fill_between(s.param, s.ci_low, s.ci_high, alpha=0.15, color=COLORS[variant])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_rate))
    ax.set_xlabel("Species Tree Size"); ax.set_ylabel("Gene Trees / s")
    ax.set_title("DTL Per-Gene Throughput", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2)

    # (1,0) DTL batch throughput
    ax = axes[1, 0]
    batch_style = {
        "pergene_serial":      {"color": "#2563eb", "ls": "-",  "marker": "o", "label": "Per-gene serial"},
        "pergene_parallel":    {"color": "#2563eb", "ls": "--", "marker": "s", "label": "Per-gene parallel"},
        "perspecies_serial":   {"color": "#16a34a", "ls": "-",  "marker": "^", "label": "Per-species serial"},
        "perspecies_parallel": {"color": "#16a34a", "ls": "--", "marker": "D", "label": "Per-species parallel"},
    }
    for variant, st in batch_style.items():
        v = df[(df.group == "dtl_batch") & (df.variant == variant)].sort_values("param")
        if v.empty:
            continue
        ax.plot(v.param, v["mean"], linestyle=st["ls"], marker=st["marker"], color=st["color"],
                linewidth=2, markersize=6, label=st["label"])
        ax.fill_between(v.param, v.ci_low, v.ci_high, alpha=0.12, color=st["color"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_rate))
    ax.set_xlabel("Batch Size"); ax.set_ylabel("Gene Trees / s")
    ax.set_title("DTL Batch Throughput (50 species)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.2)

    # (1,1) Thread scaling
    ax = axes[1, 1]
    ts_style = {
        "bd":             {"color": "#2563eb", "marker": "o", "label": "BD (1000 sp)"},
        "dtl_pergene":    {"color": "#ea580c", "marker": "s", "label": "DTL per-gene (100 sp)"},
        "dtl_perspecies":  {"color": "#16a34a", "marker": "^", "label": "DTL per-sp (100 sp)"},
    }
    max_threads = 1
    for variant, st in ts_style.items():
        v = df[(df.group == "thread_scaling") & (df.variant == variant)].sort_values("param")
        if v.empty:
            continue
        baseline = v["mean"].iloc[0]
        ax.plot(v.param, v["mean"] / baseline, marker=st["marker"], color=st["color"],
                linewidth=2, markersize=6, label=st["label"])
        ax.fill_between(v.param, v.ci_low / baseline, v.ci_high / baseline, alpha=0.12, color=st["color"])
        max_threads = max(max_threads, int(v.param.max()))
    ideal_x = np.arange(1, max_threads + 1)
    ax.plot(ideal_x, ideal_x, "k--", alpha=0.3, linewidth=1, label="Ideal")
    ax.set_xlabel("Threads"); ax.set_ylabel("Speedup")
    ax.set_title("Parallel Scaling Efficiency", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    ax.set_xlim(0.5, max_threads + 0.5); ax.set_ylim(bottom=0)

    fig.suptitle("Rustree Benchmark Suite", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overview.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "overview.svg"), bbox_inches="tight")
    plt.close()
    print("  -> overview.png / .svg")


def main():
    if not os.path.exists(CSV):
        print(f"Error: {CSV} not found.")
        print("Run benchmarks first: cargo bench --no-default-features --bench bench_plot")
        return

    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} data points from {CSV}\n")
    print("Generating plots...")

    # Individual plots
    plot_scaling(df, "bd", "Birth-Death Simulation Throughput",
                 "Number of Species", "Species Trees / Second", "bd_throughput")
    plot_scaling(df, "dtl_pergene", "DTL Per-Gene Simulation Throughput",
                 "Species Tree Size", "Gene Trees / Second", "dtl_pergene_throughput")
    plot_scaling(df, "dtl_perspecies", "DTL Per-Species Simulation Throughput",
                 "Species Tree Size", "Gene Trees / Second", "dtl_perspecies_throughput")
    plot_batch(df)
    plot_thread_scaling(df)
    plot_combined_overview(df)

    print(f"\nAll plots saved to {OUT}/")


if __name__ == "__main__":
    main()
