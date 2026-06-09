#!/usr/bin/env python3
"""Plot centered fp32/fp64 gradient-error distributions on the tail subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_TSV = Path(
    "benchmarks/large_dataset_capacity/output/"
    "hogenom_solver_accuracy_distribution_20260608/"
    "distribution_centered_fp32_vs_fp64_tail_p99_clade60k.tsv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_TSV.parent / "plots_centered_fp32_vs_fp64_tail"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def case_name(row: pd.Series) -> str:
    dtype = "fp64" if str(row["dtype"]) == "float64" else "fp32"
    solver = str(row["solver"])
    iterations = int(row["iterations"])
    if solver == "gmres":
        return f"{dtype} centered GMRES{iterations}"
    if solver == "neumann":
        return f"{dtype} centered N{iterations}"
    return f"{dtype} centered {solver}{iterations}"


def case_order_key(case: str) -> tuple[int, int, int]:
    dtype_order = 0 if case.startswith("fp32") else 1
    if " N" in case:
        solver_order = 0
    elif "GMRES" in case:
        solver_order = 1
    else:
        solver_order = 9
    iteration = int("".join(ch for ch in case.split()[-1] if ch.isdigit()))
    return dtype_order, solver_order, iteration


def solver_color(case: str):
    if "GMRES" in case:
        return "#2ca02c"
    if " N" in case:
        return "#1f77b4"
    return "#7f7f7f"


def line_style(case: str) -> tuple[str, float, float]:
    is_fp64 = case.startswith("fp64")
    if "GMRES" in case:
        style = "--"
    else:
        style = "-"
    return style, (3.2 if is_fp64 else 1.8), (0.95 if is_fp64 else 0.55)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["rel_l2"] = pd.to_numeric(df["rel_l2"], errors="coerce")
    df["abs_l2"] = pd.to_numeric(df["abs_l2"], errors="coerce")
    df = df[np.isfinite(df["rel_l2"]) & np.isfinite(df["abs_l2"])].copy()
    df = df[(df["rel_l2"] > 0) & (df["abs_l2"] > 0)].copy()
    df["case"] = df.apply(case_name, axis=1)
    return df


def quantile_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for case, group in df.groupby("case"):
        values = group[metric].to_numpy(dtype=np.float64)
        rows.append(
            {
                "case": case,
                "count": int(values.size),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "p999": float(np.quantile(values, 0.999)),
                "max": float(np.max(values)),
            }
        )
    out = pd.DataFrame(rows)
    out["_order"] = out["case"].map(case_order_key)
    out = out.sort_values("_order").drop(columns=["_order"])
    return out


def plot_survival(df: pd.DataFrame, metric: str, output: Path, *, title: str, xlabel: str) -> None:
    cases = sorted(df["case"].unique(), key=case_order_key)
    fig, ax = plt.subplots(figsize=(12, 7))
    for case in cases:
        values = np.sort(df.loc[df["case"] == case, metric].to_numpy(dtype=np.float64))
        n = values.size
        survival = (n - np.arange(n, dtype=np.float64)) / n
        style, linewidth, alpha = line_style(case)
        ax.step(
            values,
            survival,
            where="post",
            label=case,
            color=solver_color(case),
            linestyle=style,
            linewidth=linewidth,
            alpha=alpha,
        )
    for y, label in [(1e-2, "p99"), (1e-3, "p99.9")]:
        ax.axhline(y, color="0.25", linewidth=1.0, linestyle="-.")
        ax.text(ax.get_xlim()[0], y * 1.05, label, fontsize=10, color="0.2")
    finite = df[metric].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(
        10 ** np.floor(np.log10(max(float(np.min(finite)) * 0.8, 1e-16))),
        10 ** np.ceil(np.log10(float(np.max(finite)) * 1.2)),
    )
    ax.set_ylim(2e-3, 1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of tail families with error >= x")
    ax.grid(True, which="both", linewidth=0.45, alpha=0.35)
    ax.legend(ncol=2, fontsize=8, frameon=True, loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.input_tsv)
    relative_plot = args.output_dir / "centered_fp32_vs_fp64_tail_relative.png"
    absolute_plot = args.output_dir / "centered_fp32_vs_fp64_tail_absolute.png"
    relative_table = args.output_dir / "centered_fp32_vs_fp64_tail_relative_quantiles.tsv"
    absolute_table = args.output_dir / "centered_fp32_vs_fp64_tail_absolute_quantiles.tsv"
    plot_survival(
        df,
        "rel_l2",
        relative_plot,
        title="Centered fp32 vs fp64 relative gradient error on p99 tail families",
        xlabel="relative L2 gradient error vs fp64 Neumann512",
    )
    plot_survival(
        df,
        "abs_l2",
        absolute_plot,
        title="Centered fp32 vs fp64 absolute gradient error on p99 tail families",
        xlabel="absolute L2 gradient error vs fp64 Neumann512",
    )
    quantile_table(df, "rel_l2").to_csv(relative_table, sep="\t", index=False)
    quantile_table(df, "abs_l2").to_csv(absolute_table, sep="\t", index=False)
    print(relative_plot)
    print(absolute_plot)
    print(relative_table)
    print(absolute_table)


if __name__ == "__main__":
    main()
