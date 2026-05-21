#!/usr/bin/env python3
"""Checkout-local branchscale penalty report builder.

This is a one-off report renderer for the original HOGENOM branchscale penalty
sweep, not a general report API.  It discovers only immediate `penalty_*`
children below `--sweep-dir`.  Each complete child is expected to contain
`history.jsonl`, `branchscaled_node_rates_final.tsv`, and a
`tree_plots/rates_final.png` image; `run_config.json` is read when present.

Current HOGENOM launchers can write timestamped child directories, which this
historical script does not discover unless those runs are copied or symlinked
under `penalty_*` names.  The rendered report also preserves original-report
prose, including the May 18, 2026 report date and the "1325 branch multipliers"
caption.  Reuse for another dataset should first migrate those assumptions into
data-driven fields, or archive/delete this script once branchscaled reporting is
owned by the supported CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunSummary:
    label: str
    penalty: float
    run_dir: Path
    history: list[dict[str, Any]]
    config: dict[str, Any]
    top_multipliers: list[dict[str, Any]]

    @property
    def final(self) -> dict[str, Any]:
        return self.history[-1]

    @property
    def final_step(self) -> int:
        return int(self.final["step"])

    @property
    def best_nll(self) -> float:
        best_from_history = self.final.get("best/likelihood_data_nll_bits")
        if best_from_history is not None:
            return float(best_from_history)
        return min(float(row["likelihood/data_nll_bits"]) for row in self.history)

    @property
    def best_step(self) -> int:
        best_from_history = self.final.get("best/likelihood_step")
        if best_from_history is not None:
            return int(best_from_history)
        return int(
            min(
                self.history,
                key=lambda row: float(row["likelihood/data_nll_bits"]),
            )["step"]
        )

    @property
    def best_wait(self) -> int:
        value = self.final.get("best/likelihood_no_improvement_steps")
        if value is None:
            return self.final_step - self.best_step
        return int(value)

    @property
    def stop_reason(self) -> str:
        patience = int(self.config.get("best_likelihood_patience") or 0)
        grad_tol = float(self.config.get("grad_inf_tol") or 0.0)
        step_cap = int(self.config.get("steps") or 0)
        if patience > 0 and self.best_wait >= patience:
            return "best-likelihood patience"
        if grad_tol > 0.0 and float(self.final.get("grad/inf", math.inf)) <= grad_tol:
            return "gradient tolerance"
        if step_cap > 0 and self.final_step + 1 >= step_cap:
            return "step cap"
        return "not stopped"


def load_history(path: Path) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_top_multipliers(path: Path, count: int = 10) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows.sort(key=lambda row: float(row["l_e"]), reverse=True)
    return rows[:count]


def penalty_label(run_dir: Path) -> str:
    return run_dir.name.removeprefix("penalty_")


def load_runs(sweep_dir: Path) -> list[RunSummary]:
    runs: list[RunSummary] = []
    for run_dir in sorted(sweep_dir.glob("penalty_*")):
        history_path = run_dir / "history.jsonl"
        rates_path = run_dir / "branchscaled_node_rates_final.tsv"
        if not history_path.exists() or not rates_path.exists():
            continue
        config = load_json(run_dir / "run_config.json")
        penalty = float(config.get("branchscale_prior_weight", "nan"))
        runs.append(
            RunSummary(
                label=penalty_label(run_dir),
                penalty=penalty,
                run_dir=run_dir,
                history=load_history(history_path),
                config=config,
                top_multipliers=load_top_multipliers(rates_path),
            )
        )
    runs.sort(key=lambda run: run.penalty)
    return runs


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_kbits(bits: float, digits: int = 3) -> str:
    return fmt(bits / 1000.0, digits)


def plot_loss(runs: list[RunSummary], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    for run in runs:
        steps = [int(row["step"]) for row in run.history]
        nll = [float(row["likelihood/data_nll_bits"]) / 1000.0 for row in run.history]
        obj = [float(row["objective/bits"]) / 1000.0 for row in run.history]
        label = f"{run.penalty:g}"
        axes[0].plot(steps, nll, linewidth=1.5, label=label)
        axes[1].plot(steps, obj, linewidth=1.5, label=label)
        axes[0].axvline(run.best_step, color=axes[0].lines[-1].get_color(), alpha=0.18, linewidth=0.8)
    axes[0].set_ylabel("data NLL (kbits)")
    axes[1].set_ylabel("objective (kbits)")
    axes[1].set_xlabel("Adam step")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(title="penalty", fontsize=8, title_fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gradients(runs: list[RunSummary], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    for run in runs:
        steps = [int(row["step"]) for row in run.history]
        grad_norm = [float(row["grad/norm"]) for row in run.history]
        grad_inf = [float(row["grad/inf"]) for row in run.history]
        label = f"{run.penalty:g}"
        axes[0].plot(steps, grad_norm, linewidth=1.5, label=label)
        axes[1].plot(steps, grad_inf, linewidth=1.5, label=label)
    axes[0].set_ylabel("gradient norm")
    axes[1].set_ylabel("gradient infinity norm")
    axes[1].set_xlabel("Adam step")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
    axes[0].legend(title="penalty", fontsize=8, title_fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_patience(runs: list[RunSummary], out_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 4.2))
    for run in runs:
        steps = [int(row["step"]) for row in run.history]
        waits = [int(row.get("best/likelihood_no_improvement_steps") or 0) for row in run.history]
        axis.plot(steps, waits, linewidth=1.5, label=f"{run.penalty:g}")
    patience_values = sorted(
        {
            int(run.config.get("best_likelihood_patience") or 0)
            for run in runs
            if int(run.config.get("best_likelihood_patience") or 0) > 0
        }
    )
    for patience in patience_values:
        axis.axhline(patience, linestyle="--", color="black", linewidth=0.9, alpha=0.5)
    axis.set_xlabel("Adam step")
    axis.set_ylabel("steps since best NLL")
    axis.grid(True, alpha=0.25)
    axis.legend(title="penalty", fontsize=8, title_fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_plots(runs: list[RunSummary], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(runs, plot_dir / "loss_evolution.png")
    plot_gradients(runs, plot_dir / "gradient_evolution.png")
    plot_patience(runs, plot_dir / "best_likelihood_patience.png")


def render_report(sweep_dir: Path, runs: list[RunSummary]) -> str:
    penalties = ", ".join(fmt(run.penalty, 4).rstrip("0").rstrip(".") for run in runs)
    first_config = runs[0].config
    patience = int(first_config.get("best_likelihood_patience") or 0)
    min_delta = float(first_config.get("best_likelihood_min_delta") or 0.0)
    lr = float(first_config.get("lr") or 0.0)
    lr_decay_every = int(first_config.get("lr_decay_every") or 0)
    lr_decay_factor = float(first_config.get("lr_decay_factor") or 0.0)
    steps = int(first_config.get("steps") or 0)
    clade_budget = int(first_config.get("clade_budget") or 0)
    max_wave_size = int(first_config.get("max_wave_size") or 0)

    lines: list[str] = [
        r"\documentclass[11pt]{article}",
        "",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{hyperref}",
        "",
        r"\title{HOGENOM CCP Branchwise Multiplier Penalty Sweep}",
        r"\author{gpurec optimization run}",
        r"\date{May 18, 2026}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section{Setup}",
        "",
        "This report summarizes a local sweep using",
        r"\texttt{scripts/optimize\_hogenom\_ccp\_wandb.py}.",
        "Each run used the branchscaled parameterization, where shared",
        r"D/T/L rates are multiplied by one branch-specific multiplier \(l_e\).",
        "",
        r"\begin{itemize}",
        rf"  \item Penalty weights: {penalties}.",
        rf"  \item Optimizer: Adam, learning rate {lr:g}, decay every {lr_decay_every} steps by {lr_decay_factor:g}, maximum {steps} steps.",
        rf"  \item Regularization: beta prior disabled; branchscale prior weight swept over the penalties above.",
        rf"  \item Stop rule: save \texttt{{checkpoints/best.pt}} whenever data NLL improves by at least {min_delta:g} bits, then stop after {patience} later steps without such an improvement.",
        rf"  \item Solver memory settings: clade budget {clade_budget}, maximum wave size {max_wave_size}.",
        r"\end{itemize}",
        "",
        r"\section{Convergence}",
        "",
        "All completed runs below stopped by the best-likelihood patience rule,",
        "not by the fixed step cap.  The gradient norms remain large because the",
        "criterion used here is improvement in the data likelihood, not first-order",
        "stationarity of the penalized objective.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{r r r r r l}",
        r"\toprule",
        r"Penalty & Stop step & Best step & Best NLL & Final obj. & Stop reason \\",
        r" & & & (kbits) & (kbits) & \\",
        r"\midrule",
    ]
    for run in runs:
        lines.append(
            rf"{fmt(run.penalty, 4).rstrip('0').rstrip('.')} & "
            rf"{run.final_step} & {run.best_step} & "
            rf"{fmt_kbits(run.best_nll)} & "
            rf"{fmt_kbits(float(run.final['objective/bits']))} & "
            rf"{latex_escape(run.stop_reason)} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Convergence summary.  Values are rounded and reported in thousands of bits where applicable.}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{r r r r r r}",
            r"\toprule",
            r"Penalty & Grad norm & Grad inf. & Branch pen. & Median \(l_e\) & P95 \(l_e\) \\",
            r" & \((10^3)\) & \((10^3)\) & (kbits) & & \\",
            r"\midrule",
        ]
    )
    for run in runs:
        final = run.final
        lines.append(
            rf"{fmt(run.penalty, 4).rstrip('0').rstrip('.')} & "
            rf"{fmt(float(final['grad/norm']) / 1000.0, 2)} & "
            rf"{fmt(float(final['grad/inf']) / 1000.0, 2)} & "
            rf"{fmt_kbits(float(final['regularization/branchscale_bits']))} & "
            rf"{fmt(float(final['branchscale/l_median']))} & "
            rf"{fmt(float(final['branchscale/l_p95']))} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Final gradient and multiplier diagnostics.}",
            r"\end{table}",
            "",
            r"\begin{figure}[H]",
            r"  \centering",
            r"  \includegraphics[width=\textwidth]{diagnostic_plots/loss_evolution.png}",
            r"  \caption{Data NLL and penalized objective over Adam steps.  Faint vertical markers in the top panel show the best data-likelihood step for each run.}",
            r"\end{figure}",
            "",
            r"\begin{figure}[H]",
            r"  \centering",
            r"  \includegraphics[width=\textwidth]{diagnostic_plots/gradient_evolution.png}",
            r"  \caption{Gradient norm and gradient infinity norm over Adam steps.}",
            r"\end{figure}",
            "",
            r"\begin{figure}[H]",
            r"  \centering",
            r"  \includegraphics[width=\textwidth]{diagnostic_plots/best_likelihood_patience.png}",
            r"  \caption{Number of steps since the best data NLL.  The dashed horizontal line is the patience threshold.}",
            r"\end{figure}",
            "",
            r"\section{Multiplier Distribution}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{r r r r r r r}",
            r"\toprule",
            r"Penalty & Mean & Median & P95 & Max & \(>2\times\) & \(>4\times\) \\",
            r"\midrule",
        ]
    )
    for run in runs:
        values = [float(row["l_e"]) for row in run.top_multipliers]
        rates_path = run.run_dir / "branchscaled_node_rates_final.tsv"
        with rates_path.open("r", newline="", encoding="utf-8") as handle:
            all_values = [float(row["l_e"]) for row in csv.DictReader(handle, delimiter="\t")]
        count_2x = sum(value > 2.0 for value in all_values)
        count_4x = sum(value > 4.0 for value in all_values)
        final = run.final
        lines.append(
            rf"{fmt(run.penalty, 4).rstrip('0').rstrip('.')} & "
            rf"{fmt(float(final['branchscale/l_mean']))} & "
            rf"{fmt(float(final['branchscale/l_median']))} & "
            rf"{fmt(float(final['branchscale/l_p95']))} & "
            rf"{fmt(max(values))} & {count_2x} & {count_4x} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Final branch multiplier distribution.  Counts are out of 1325 branch multipliers.}",
            r"\end{table}",
            "",
            r"\begin{longtable}{r r p{0.52\textwidth} r}",
            r"\caption{Top ten branchwise rate multipliers for every penalty.}\\",
            r"\toprule",
            r"Penalty & Rank & Branch label & \(l_e\) \\",
            r"\midrule",
            r"\endfirsthead",
            r"\caption[]{Top ten branchwise rate multipliers for every penalty (continued).}\\",
            r"\toprule",
            r"Penalty & Rank & Branch label & \(l_e\) \\",
            r"\midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endfoot",
        ]
    )
    for run in runs:
        for rank, row in enumerate(run.top_multipliers, start=1):
            lines.append(
                rf"{fmt(run.penalty, 4).rstrip('0').rstrip('.')} & "
                rf"{rank} & {latex_escape(row['label'])} & {fmt(float(row['l_e']))} \\"
            )
        lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\end{longtable}",
            "",
            r"\section{Interpretation}",
            "",
            "The penalty sweep shows the expected tradeoff.  Weaker penalties improve",
            "the data likelihood but allow a broad branch multiplier distribution.",
            "Stronger penalties keep most multipliers close to one and pay for that",
            "with a worse data likelihood.  The intermediate penalties are the useful",
            "region for selecting how much branchwise adaptation to allow.",
            "",
            r"\clearpage",
            r"\section{Final Rate Maps}",
            "",
        ]
    )
    for run in runs:
        rel_plot = run.run_dir.relative_to(sweep_dir) / "tree_plots" / "rates_final.png"
        lines.extend(
            [
                r"\begin{figure}[p]",
                r"  \centering",
                rf"  \includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{{{rel_plot.as_posix()}}}",
                rf"  \caption{{Final rate map for branchscale prior weight {fmt(run.penalty, 4).rstrip('0').rstrip('.')}.}}",
                r"\end{figure}",
                "",
            ]
        )
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the HOGENOM branchscale penalty sweep LaTeX report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Expected legacy layout: --sweep-dir contains penalty_* child "
            "directories, each with history.jsonl, "
            "branchscaled_node_rates_final.tsv, and tree_plots/rates_final.png. "
            "Timestamped HOGENOM launcher outputs must be copied or symlinked "
            "under penalty_* names before this historical report builder will "
            "discover them."
        ),
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("tests/data/HOGENOM/hogenom/output_gpurec_wandb_adam_branchscale_penalty_sweep"),
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    runs = load_runs(sweep_dir)
    if not runs:
        raise SystemExit(f"no complete penalty runs found under {sweep_dir}")

    make_plots(runs, sweep_dir / "diagnostic_plots")
    report = render_report(sweep_dir, runs)
    report_path = sweep_dir / "branchscale_penalty_sweep_report.tex"
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
