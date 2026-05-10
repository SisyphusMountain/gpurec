"""Gradient-based HOGENOM per-family DTL-rate optimization.

This is the scalable alternative to coordinate probing in
``optimize_hogenom_fast.py``.  Each iteration computes one batched gradient,
turns it into an Adam-style descent direction for every family, and performs a
small per-family line search along that full vector direction.  The number of
loss probes per iteration is independent of the parameter dimension.

The important practical detail is ``neumann_terms``: low values give noticeably
worse directions on HOGENOM high-rate families.  The default here uses 32 terms.

Example
-------
python profiling/optimize_hogenom_gradient.py \
  --dataset tests/data/hogenom_bench \
  --init-csv tests/data/hogenom_bench/gpurec/model_rates.csv \
  --fixed-pi 20 \
  --reference-pi 160
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch

from gpurec import GeneReconModel


OPTIMIZED_ENV = {
    "GPUREC_FORWARD_LEAF_INDEX": "1",
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS": "64",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
    "GPUREC_FORWARD_DTS_OVERLAP_MODE": "off",
    "GPUREC_KERNELIZED_ACTIVE_MASK": "1",
}


def _set_optimized_env() -> None:
    for key, value in OPTIMIZED_ENV.items():
        os.environ.setdefault(key, value)


def _family_id(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _gene_paths(dataset: Path) -> list[Path]:
    paths = sorted(dataset.glob("g_*.nwk"), key=_family_id)
    if not paths:
        raise FileNotFoundError(f"no g_*.nwk files in {dataset}")
    return paths


def _load_rates(
    init_csv: Path | None,
    gene_paths: list[Path],
    *,
    min_rate: float,
    max_rate: float,
) -> pd.DataFrame:
    if init_csv is None:
        rows = []
        for path in gene_paths:
            family_id = _family_id(path)
            rows.append(
                {
                    "family_id": family_id,
                    "family": f"family_{family_id:04d}",
                    "gene_file": path.name,
                    "D": 0.05,
                    "L": 0.05,
                    "T": 0.05,
                }
            )
        return pd.DataFrame(rows)

    df = pd.read_csv(init_csv).copy()
    if "family_id" not in df.columns:
        df.insert(0, "family_id", range(len(df)))
    df["family_id"] = df["family_id"].astype(int)
    if "family" not in df.columns:
        df.insert(1, "family", [f"family_{i:04d}" for i in df["family_id"]])
    if "gene_file" not in df.columns:
        df.insert(2, "gene_file", [f"g_{i}.nwk" for i in df["family_id"]])
    missing = {"D", "L", "T"} - set(df.columns)
    if missing:
        raise ValueError(f"{init_csv} is missing rate columns: {sorted(missing)}")
    for col in ("D", "L", "T"):
        df[col] = df[col].clip(lower=min_rate, upper=max_rate)
    return df.sort_values("family_id").reset_index(drop=True)


def _build_model(
    dataset: Path,
    gene_paths: list[Path],
    df: pd.DataFrame,
    *,
    fixed_pi: int,
    neumann_terms: int,
    device: str,
    dtype: torch.dtype,
    preprocess_cache_dir: str,
    max_wave_size: int,
) -> GeneReconModel:
    selected = [gene_paths[int(fid)] for fid in df["family_id"]]
    return GeneReconModel.from_trees(
        species_tree=str(dataset / "sp.nwk"),
        gene_trees=[str(path) for path in selected],
        mode="genewise",
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        preprocess_cache_dir=preprocess_cache_dir,
        fixed_iters_Pi=fixed_pi,
        fixed_iters_E=None,
        max_iters_E=2000,
        tol_E=1e-10,
        neumann_terms=neumann_terms,
        use_pruning=False,
        max_wave_size=max_wave_size,
    )


def _copy_rates_to_model(
    model: GeneReconModel,
    df: pd.DataFrame,
    *,
    min_rate: float,
    max_rate: float,
) -> None:
    rates = torch.tensor(
        df[["D", "L", "T"]].to_numpy(),
        device=model.theta.device,
        dtype=model.theta.dtype,
    ).clamp(min_rate, max_rate)
    with torch.no_grad():
        model.theta.copy_(torch.log2(rates))


def _sync(model: GeneReconModel) -> None:
    if model.theta.device.type == "cuda":
        torch.cuda.synchronize(model.theta.device)


@torch.no_grad()
def _eval_nll(model: GeneReconModel) -> torch.Tensor:
    out = model.nll_per_family().detach()
    _sync(model)
    return out


def _adam_direction_line_search(
    model: GeneReconModel,
    current_loss: torch.Tensor,
    *,
    iterations: int,
    max_step_log2: float,
    alpha_grid: list[float],
    min_rate: float,
    max_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Run Adam directions plus per-family line search.

    The line search probes one full-vector descent direction per family.  It is
    therefore dimension-scalable: increasing parameters per family does not
    increase the number of objective probes.
    """
    lo = math.log2(min_rate)
    hi = math.log2(max_rate)
    m = torch.zeros_like(model.theta)
    v = torch.zeros_like(model.theta)
    history: list[dict[str, float | int]] = []
    probes = 0
    t0 = time.perf_counter()

    for iteration in range(1, iterations + 1):
        model.theta.grad = None
        loss_vec = model.nll_per_family()
        loss_vec.sum().backward()
        _sync(model)
        grad = model.theta.grad.detach()

        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        direction = -m_hat / (v_hat.sqrt() + eps)

        row_max = direction.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        direction = direction / row_max * max_step_log2

        start_theta = model.theta.detach().clone()
        best_theta = start_theta.clone()
        best_loss = current_loss.clone()
        accepted = torch.zeros_like(current_loss, dtype=torch.bool)

        for alpha in alpha_grid:
            candidate = (start_theta + float(alpha) * direction).clamp(lo, hi)
            with torch.no_grad():
                model.theta.copy_(candidate)
            candidate_loss = _eval_nll(model)
            probes += 1
            better = candidate_loss < best_loss
            if bool(better.any()):
                best_loss = torch.where(better, candidate_loss, best_loss)
                best_theta[better] = candidate[better]
                accepted |= better

        with torch.no_grad():
            model.theta.copy_(best_theta)
        current_loss = best_loss.detach()
        record = {
            "iteration": iteration,
            "max_step_log2": float(max_step_log2),
            "nll_sum": float(current_loss.sum().cpu()),
            "accepted": int(accepted.sum().item()),
            "probes": probes,
            "elapsed_s": time.perf_counter() - t0,
        }
        history.append(record)
        print(
            f"[adam-ls] iter={iteration} max_step={max_step_log2:g} "
            f"sum={record['nll_sum']:.6f} accepted={record['accepted']} "
            f"probes={probes} elapsed={record['elapsed_s']:.2f}s",
            flush=True,
        )

    return current_loss, history


def _parse_stages(text: str) -> list[tuple[int, float]]:
    stages: list[tuple[int, float]] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        iterations_s, step_s = item.split(":", 1)
        iterations = int(iterations_s)
        step = float(step_s)
        if iterations < 1:
            raise ValueError("stage iterations must be positive")
        if step <= 0:
            raise ValueError("stage step must be positive")
        stages.append((iterations, step))
    if not stages:
        raise ValueError("at least one stage is required")
    return stages


def _write_rates(path: Path, df: pd.DataFrame, model: GeneReconModel, nll: torch.Tensor) -> None:
    out = df[["family_id", "family", "gene_file"]].copy()
    out[["D", "L", "T"]] = model.rates.detach().cpu().numpy()
    out["nll_fixed"] = nll.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.10g")


def _validate(
    dataset: Path,
    gene_paths: list[Path],
    rates: pd.DataFrame,
    *,
    fixed_pi: int,
    reference_pi: int,
    neumann_terms: int,
    device: str,
    dtype: torch.dtype,
    preprocess_cache_dir: str,
    max_wave_size: int,
    min_rate: float,
    max_rate: float,
) -> pd.DataFrame:
    out = rates[["family_id", "family", "gene_file", "D", "L", "T"]].copy()
    values: dict[int, torch.Tensor] = {}
    for passes in (fixed_pi, reference_pi):
        model = _build_model(
            dataset,
            gene_paths,
            out,
            fixed_pi=passes,
            neumann_terms=neumann_terms,
            device=device,
            dtype=dtype,
            preprocess_cache_dir=preprocess_cache_dir,
            max_wave_size=max_wave_size,
        )
        _copy_rates_to_model(model, out, min_rate=min_rate, max_rate=max_rate)
        t0 = time.perf_counter()
        values[passes] = _eval_nll(model).cpu()
        print(
            f"[validate] fixed_pi={passes} sum={float(values[passes].sum()):.6f} "
            f"elapsed={time.perf_counter() - t0:.2f}s",
            flush=True,
        )
    out[f"nll_fixed{fixed_pi}"] = values[fixed_pi].numpy()
    out[f"nll_fixed{reference_pi}"] = values[reference_pi].numpy()
    out["abs_diff"] = (values[fixed_pi] - values[reference_pi]).abs().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("tests/data/hogenom_bench"))
    parser.add_argument("--init-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--preprocess-cache-dir", default="/tmp/gpurec_hogenom_fast_cache")
    parser.add_argument("--fixed-pi", type=int, default=20)
    parser.add_argument("--reference-pi", type=int, default=160)
    parser.add_argument("--neumann-terms", type=int, default=32)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=2.0)
    parser.add_argument("--start-family", type=int, default=0)
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument(
        "--stages",
        default="8:0.125,8:0.0625",
        help="comma-separated iteration:max_log2_step stages",
    )
    parser.add_argument(
        "--alphas",
        default="2,1,0.5,0.25,0.125,0.0625",
        help="comma-separated line-search multipliers",
    )
    args = parser.parse_args()

    _set_optimized_env()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir or (dataset / "gpurec" / "fast_opt")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    gene_paths = _gene_paths(dataset)
    init_csv = args.init_csv
    if init_csv is None:
        default_init = dataset / "gpurec" / "model_rates.csv"
        init_csv = default_init if default_init.exists() else None
    df = _load_rates(init_csv, gene_paths, min_rate=args.min_rate, max_rate=args.max_rate)
    if args.start_family < 0:
        raise ValueError("--start-family must be non-negative")
    df = df[df["family_id"] >= args.start_family]
    if args.max_families is not None:
        if args.max_families < 1:
            raise ValueError("--max-families must be positive when provided")
        df = df.head(args.max_families)
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("empty family selection")

    model = _build_model(
        dataset,
        gene_paths,
        df,
        fixed_pi=args.fixed_pi,
        neumann_terms=args.neumann_terms,
        device=args.device,
        dtype=dtype,
        preprocess_cache_dir=args.preprocess_cache_dir,
        max_wave_size=args.max_wave_size,
    )
    _copy_rates_to_model(model, df, min_rate=args.min_rate, max_rate=args.max_rate)
    current = _eval_nll(model)
    initial_sum = float(current.sum().cpu())
    print(f"[start] fixed_pi={args.fixed_pi} sum={initial_sum:.6f}", flush=True)

    alpha_grid = [float(part) for part in args.alphas.split(",") if part.strip()]
    stages = _parse_stages(args.stages)
    all_history: list[dict[str, float | int]] = []
    total_start = time.perf_counter()
    for stage_index, (iterations, max_step) in enumerate(stages, start=1):
        print(
            f"[stage {stage_index}] iterations={iterations} max_step_log2={max_step:g}",
            flush=True,
        )
        current, history = _adam_direction_line_search(
            model,
            current,
            iterations=iterations,
            max_step_log2=max_step,
            alpha_grid=alpha_grid,
            min_rate=args.min_rate,
            max_rate=args.max_rate,
        )
        for row in history:
            row["stage"] = stage_index
        all_history.extend(history)

    rates_path = output_dir / "model_rates_gradient_adam_line_search.csv"
    _write_rates(rates_path, df, model, current)
    rates_df = pd.read_csv(rates_path)
    validation = _validate(
        dataset,
        gene_paths,
        rates_df,
        fixed_pi=args.fixed_pi,
        reference_pi=args.reference_pi,
        neumann_terms=args.neumann_terms,
        device=args.device,
        dtype=dtype,
        preprocess_cache_dir=args.preprocess_cache_dir,
        max_wave_size=args.max_wave_size,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
    )
    validation_path = output_dir / "fixed_pass_validation_gradient_adam_line_search.csv"
    validation.to_csv(validation_path, index=False, float_format="%.10g")
    abs_diff = validation["abs_diff"]
    summary = {
        "dataset": str(dataset),
        "init_csv": None if init_csv is None else str(init_csv),
        "families": int(len(df)),
        "fixed_pi": args.fixed_pi,
        "reference_pi": args.reference_pi,
        "neumann_terms": args.neumann_terms,
        "stages": [{"iterations": n, "max_step_log2": s} for n, s in stages],
        "alpha_grid": alpha_grid,
        "initial_nll_sum": initial_sum,
        "nll_sum_fixed": float(validation[f"nll_fixed{args.fixed_pi}"].sum()),
        "nll_sum_reference": float(validation[f"nll_fixed{args.reference_pi}"].sum()),
        "max_abs_fixed_vs_reference": float(abs_diff.max()),
        "p99_abs_fixed_vs_reference": float(abs_diff.quantile(0.99)),
        "families_over_0_1": int((abs_diff > 0.1).sum()),
        "rates_path": str(rates_path),
        "validation_path": str(validation_path),
        "elapsed_s": time.perf_counter() - total_start,
        "history": all_history,
    }
    summary_path = output_dir / "summary_gradient_adam_line_search.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
