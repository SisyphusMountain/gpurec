#!/usr/bin/env python3
"""Plot HOGENOM all-family gradient relative-error distributions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_TSV = Path(
    "benchmarks/large_dataset_capacity/output/"
    "hogenom_solver_accuracy_distribution_20260608/distribution_clade60k.tsv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_TSV.parent / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def case_name(row: pd.Series) -> str:
    label = str(row["label"])
    solver = str(row["solver"])
    iterations = int(row["iterations"])
    if label.startswith("centered"):
        if solver == "gmres":
            return f"centered GMRES{iterations}"
        return f"centered N{iterations}"
    if solver == "neumann":
        return f"fp32 N{iterations}"
    if solver == "gmres":
        return f"fp32 GMRES{iterations}"
    return f"{label} {solver}{iterations}"


def case_order_key(name: str) -> tuple[int, int]:
    if name.startswith("fp32 N"):
        return 0, int(name.removeprefix("fp32 N"))
    if name.startswith("fp32 GMRES"):
        return 1, int(name.removeprefix("fp32 GMRES"))
    if name.startswith("centered N"):
        return 2, int(name.removeprefix("centered N"))
    if name.startswith("centered GMRES"):
        return 3, int(name.removeprefix("centered GMRES"))
    return 9, 0


def case_iteration(name: str) -> int:
    digits = []
    for char in reversed(name):
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(reversed(digits))) if digits else 0


def case_line_style(name: str) -> tuple[str, float, float]:
    centered = name.startswith("centered")
    if "GMRES" in name:
        linestyle = "--"
    else:
        linestyle = "-"
    linewidth = 3.0 if centered else 1.6
    alpha = 0.95 if centered else 0.70
    return linestyle, linewidth, alpha


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["rel_l2"] = pd.to_numeric(df["rel_l2"], errors="coerce")
    df["case"] = df.apply(case_name, axis=1)
    df = df[np.isfinite(df["rel_l2"]) & (df["rel_l2"] > 0)].copy()
    return df


def quantile_rows(df: pd.DataFrame, *, metric: str) -> list[dict[str, float | str | int]]:
    probs = {
        "p50": 0.50,
        "p95": 0.95,
        "p99": 0.99,
        "p999": 0.999,
        "p9999": 0.9999,
        "max": 1.0,
    }
    rows: list[dict[str, float | str | int]] = []
    for case, group in df.groupby("case"):
        values = group[metric].to_numpy(dtype=np.float64)
        row: dict[str, float | str | int] = {
            "case": case,
            "count": int(values.size),
        }
        for name, prob in probs.items():
            row[name] = float(np.quantile(values, prob))
        row["frac_le_1e-4"] = float(np.mean(values <= 1e-4))
        row["frac_le_1e-5"] = float(np.mean(values <= 1e-5))
        rows.append(row)
    rows.sort(key=lambda item: case_order_key(str(item["case"])))
    return rows


def plot_survival(df: pd.DataFrame, output: Path, *, metric: str, x_label: str, title: str) -> None:
    cases = sorted(df["case"].unique(), key=case_order_key)
    iterations = sorted({case_iteration(case) for case in cases})
    cmap = plt.get_cmap("tab10")
    color_by_iteration = {
        iteration: cmap(idx % cmap.N)
        for idx, iteration in enumerate(iterations)
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    for case in cases:
        values = np.sort(df.loc[df["case"] == case, metric].to_numpy(dtype=np.float64))
        n = values.size
        survival = (n - np.arange(n, dtype=np.float64)) / n
        linestyle, linewidth, alpha = case_line_style(case)
        ax.step(
            values,
            survival,
            where="post",
            label=case,
            color=color_by_iteration[case_iteration(case)],
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )

    tail_lines = [
        (1e-2, "p99"),
        (1e-3, "p99.9"),
        (1e-4, "p99.99"),
    ]
    for y, label in tail_lines:
        ax.axhline(y, color="0.25", linewidth=1.0, linestyle="-.")
        ax.text(
            1.05e-6,
            y * 1.04,
            label,
            ha="left",
            va="bottom",
            fontsize=10,
            color="0.20",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    finite = df[metric].to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    xmin = 10 ** np.floor(np.log10(max(float(np.min(finite)) * 0.8, 1e-12)))
    xmax = 10 ** np.ceil(np.log10(float(np.max(finite)) * 1.2))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(5e-5, 1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("fraction of families with error >= x")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.45, alpha=0.35)
    ax.legend(ncol=2, fontsize=8, frameon=True, loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_tail_quantiles(rows: list[dict[str, float | str | int]], output: Path) -> None:
    cases = [str(row["case"]) for row in rows]
    x = np.arange(len(cases))
    quantiles = [
        ("p95", "p95"),
        ("p99", "p99"),
        ("p999", "p99.9"),
        ("p9999", "p99.99"),
        ("max", "max"),
    ]
    fig, ax = plt.subplots(figsize=(13, 7))
    for name, label in quantiles:
        y = np.array([float(row[name]) for row in rows], dtype=np.float64)
        linewidth = 2.8 if name in {"p999", "p9999", "max"} else 1.8
        marker = "o" if name != "max" else "X"
        ax.plot(x, y, marker=marker, linewidth=linewidth, markersize=5, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=35, ha="right")
    ax.set_ylabel("relative L2 gradient error")
    ax.set_title("Tail quantiles by solver variant")
    ax.grid(True, which="both", axis="y", linewidth=0.45, alpha=0.35)
    ax.legend(ncol=5, fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_quantiles(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    fieldnames = [
        "case",
        "count",
        "p50",
        "p95",
        "p99",
        "p999",
        "p9999",
        "max",
        "frac_le_1e-4",
        "frac_le_1e-5",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.input_tsv)
    df["abs_l2"] = pd.to_numeric(df["abs_l2"], errors="coerce")
    df = df[np.isfinite(df["abs_l2"]) & (df["abs_l2"] > 0)].copy()
    relative_rows = quantile_rows(df, metric="rel_l2")
    absolute_rows = quantile_rows(df, metric="abs_l2")
    relative_survival_path = args.output_dir / "relative_error_survival_all_cases.png"
    absolute_survival_path = args.output_dir / "absolute_error_survival_all_cases.png"
    quantile_path = args.output_dir / "relative_error_tail_quantiles.png"
    relative_table_path = args.output_dir / "relative_error_tail_quantiles.tsv"
    absolute_table_path = args.output_dir / "absolute_error_tail_quantiles.tsv"
    plot_survival(
        df,
        relative_survival_path,
        metric="rel_l2",
        x_label="per-family relative L2 gradient error vs fp64 Neumann512",
        title="HOGENOM relative gradient error tail distribution",
    )
    plot_survival(
        df,
        absolute_survival_path,
        metric="abs_l2",
        x_label="per-family absolute L2 gradient error vs fp64 Neumann512",
        title="HOGENOM absolute gradient error tail distribution",
    )
    plot_tail_quantiles(relative_rows, quantile_path)
    write_quantiles(relative_table_path, relative_rows)
    write_quantiles(absolute_table_path, absolute_rows)
    print(relative_survival_path)
    print(absolute_survival_path)
    print(quantile_path)
    print(relative_table_path)
    print(absolute_table_path)


if __name__ == "__main__":
    main()
