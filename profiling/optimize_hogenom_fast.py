"""Fast per-family HOGENOM DTL-rate optimization.

This script is intentionally benchmark-oriented rather than production API.
It optimizes independent genewise rates on ``tests/data/hogenom_bench`` using
cheap no-grad coordinate search over log2 rates, then optionally runs a short
batched L-BFGS polish.  The final validation evaluates the same rates with a
larger fixed Pi-pass count and reports the per-family NLL discrepancy.

Example
-------
python profiling/optimize_hogenom_fast.py \
  --dataset tests/data/hogenom_bench \
  --init-csv tests/data/hogenom_bench/gpurec/model_rates.csv \
  --fixed-pi 20 \
  --reference-pi 160 \
  --polish-armijo-iters 2
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
from gpurec.optimization import BatchedLBFGS


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


def _load_init_rates(
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

    df = pd.read_csv(init_csv)
    df = df.copy()
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
        neumann_terms=4,
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


@torch.no_grad()
def _eval_nll(model: GeneReconModel) -> torch.Tensor:
    out = model.nll_per_family().detach()
    if model.theta.device.type == "cuda":
        torch.cuda.synchronize(model.theta.device)
    return out


def _coordinate_search(
    model: GeneReconModel,
    *,
    steps: list[float],
    min_rate: float,
    max_rate: float,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    lo = math.log2(min_rate)
    hi = math.log2(max_rate)
    current = _eval_nll(model)
    history: list[dict[str, float | int]] = []
    evals = 1
    t0 = time.perf_counter()

    for sweep, step in enumerate(steps, start=1):
        accepted_total = 0
        improvement_total = 0.0
        for coord in range(3):
            for sign in (1.0, -1.0):
                old_theta = model.theta.detach().clone()
                candidate = old_theta.clone()
                candidate[:, coord].add_(sign * step).clamp_(lo, hi)
                with torch.no_grad():
                    model.theta.copy_(candidate)
                loss = _eval_nll(model)
                evals += 1

                better = loss < current
                if bool(better.any()):
                    improvement_total += float((current[better] - loss[better]).sum().cpu())
                    accepted_total += int(better.sum().item())
                    old_theta[better] = candidate[better]
                    with torch.no_grad():
                        model.theta.copy_(old_theta)
                    current = torch.where(better, loss, current)
                else:
                    with torch.no_grad():
                        model.theta.copy_(old_theta)

        record = {
            "sweep": sweep,
            "step_log2": float(step),
            "nll_sum": float(current.sum().cpu()),
            "accepted": accepted_total,
            "improvement": improvement_total,
            "evaluations": evals,
            "elapsed_s": time.perf_counter() - t0,
        }
        history.append(record)
        print(
            f"[coord] sweep={sweep} step={step:g} "
            f"sum={record['nll_sum']:.6f} accepted={accepted_total} "
            f"improvement={improvement_total:.6f} evals={evals} "
            f"elapsed={record['elapsed_s']:.2f}s",
            flush=True,
        )
    return current, history


def _armijo_polish(
    model: GeneReconModel,
    *,
    iters: int,
    min_rate: float,
    max_rate: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if iters <= 0:
        return _eval_nll(model), {
            "enabled": 0,
            "elapsed_s": 0.0,
            "gradient_calls": 0,
            "loss_calls": 0,
        }

    opt = BatchedLBFGS(
        [model.theta],
        lr=1.0,
        max_iter=iters,
        max_eval=max(1, iters) * 20,
        history_size=6,
        line_search_fn="armijo",
        max_ls=12,
        tolerance_grad=1e-4,
        tolerance_change=1e-7,
        lower_bound=math.log2(min_rate),
        upper_bound=math.log2(max_rate),
    )
    gradient_calls = 0
    loss_calls = 0

    def loss_closure() -> torch.Tensor:
        nonlocal loss_calls
        loss_calls += 1
        with torch.no_grad():
            return _eval_nll(model)

    def closure() -> torch.Tensor:
        nonlocal gradient_calls
        gradient_calls += 1
        opt.zero_grad(set_to_none=True)
        loss = model.nll_per_family()
        loss.sum().backward()
        if model.theta.device.type == "cuda":
            torch.cuda.synchronize(model.theta.device)
        print(
            f"[polish] grad={gradient_calls} sum={float(loss.detach().sum().cpu()):.6f} "
            f"loss_probes={loss_calls}",
            flush=True,
        )
        return loss

    start = _eval_nll(model)
    t0 = time.perf_counter()
    opt.step(closure, loss_closure=loss_closure)
    elapsed = time.perf_counter() - t0
    final = _eval_nll(model)
    stats = {
        "enabled": 1,
        "elapsed_s": elapsed,
        "gradient_calls": gradient_calls,
        "loss_calls": loss_calls,
        "start_nll_sum": float(start.sum().cpu()),
        "final_nll_sum": float(final.sum().cpu()),
        "improvement": float((start.sum() - final.sum()).cpu()),
    }
    return final, stats


def _write_rates(
    output: Path,
    df: pd.DataFrame,
    model: GeneReconModel,
    nll: torch.Tensor,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    out = df[["family_id", "family", "gene_file"]].copy()
    out[["D", "L", "T"]] = model.rates.detach().cpu().numpy()
    out["nll_fixed"] = nll.detach().cpu().numpy()
    out.to_csv(output, index=False, float_format="%.10g")


def _validate_reference(
    dataset: Path,
    gene_paths: list[Path],
    rates_df: pd.DataFrame,
    *,
    fixed_pi: int,
    reference_pi: int,
    device: str,
    dtype: torch.dtype,
    preprocess_cache_dir: str,
    max_wave_size: int,
) -> pd.DataFrame:
    rows = rates_df[["family_id", "family", "gene_file", "D", "L", "T"]].copy()
    values: dict[int, torch.Tensor] = {}
    for passes in (fixed_pi, reference_pi):
        model = _build_model(
            dataset,
            gene_paths,
            rows,
            fixed_pi=passes,
            device=device,
            dtype=dtype,
            preprocess_cache_dir=preprocess_cache_dir,
            max_wave_size=max_wave_size,
        )
        _copy_rates_to_model(model, rows, min_rate=1e-10, max_rate=2.0)
        t0 = time.perf_counter()
        values[passes] = _eval_nll(model).cpu()
        print(
            f"[validate] fixed_pi={passes} sum={float(values[passes].sum()):.6f} "
            f"elapsed={time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    rows[f"nll_fixed{fixed_pi}"] = values[fixed_pi].numpy()
    rows[f"nll_fixed{reference_pi}"] = values[reference_pi].numpy()
    rows["abs_diff"] = (values[fixed_pi] - values[reference_pi]).abs().numpy()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("tests/data/hogenom_bench"))
    parser.add_argument("--init-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--preprocess-cache-dir", default="/tmp/gpurec_hogenom_fast_cache")
    parser.add_argument("--start-family", type=int, default=0)
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--fixed-pi", type=int, default=20)
    parser.add_argument("--reference-pi", type=int, default=160)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=2.0)
    parser.add_argument(
        "--steps",
        default="0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,"
        "0.00390625,0.001953125,0.0009765625,0.00048828125",
        help="comma-separated log2-rate coordinate steps",
    )
    parser.add_argument("--polish-armijo-iters", type=int, default=0)
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
    df = _load_init_rates(
        init_csv,
        gene_paths,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
    )
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
        device=args.device,
        dtype=dtype,
        preprocess_cache_dir=args.preprocess_cache_dir,
        max_wave_size=args.max_wave_size,
    )
    _copy_rates_to_model(model, df, min_rate=args.min_rate, max_rate=args.max_rate)
    steps = [float(part) for part in args.steps.split(",") if part.strip()]

    t0 = time.perf_counter()
    nll, coord_history = _coordinate_search(
        model,
        steps=steps,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
    )
    nll, polish_stats = _armijo_polish(
        model,
        iters=args.polish_armijo_iters,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
    )
    total_elapsed = time.perf_counter() - t0

    rates_path = output_dir / "model_rates_fast.csv"
    _write_rates(rates_path, df, model, nll)
    rates_df = pd.read_csv(rates_path)
    validation = _validate_reference(
        dataset,
        gene_paths,
        rates_df,
        fixed_pi=args.fixed_pi,
        reference_pi=args.reference_pi,
        device=args.device,
        dtype=dtype,
        preprocess_cache_dir=args.preprocess_cache_dir,
        max_wave_size=args.max_wave_size,
    )
    validation_path = output_dir / "fixed_pass_validation_fast.csv"
    validation.to_csv(validation_path, index=False, float_format="%.10g")

    abs_diff = validation["abs_diff"]
    summary = {
        "dataset": str(dataset),
        "init_csv": None if init_csv is None else str(init_csv),
        "families": int(len(df)),
        "fixed_pi": args.fixed_pi,
        "reference_pi": args.reference_pi,
        "rates_path": str(rates_path),
        "validation_path": str(validation_path),
        "nll_sum_fixed": float(validation[f"nll_fixed{args.fixed_pi}"].sum()),
        "nll_sum_reference": float(validation[f"nll_fixed{args.reference_pi}"].sum()),
        "max_abs_fixed_vs_reference": float(abs_diff.max()),
        "p99_abs_fixed_vs_reference": float(abs_diff.quantile(0.99)),
        "families_over_0_1": int((abs_diff > 0.1).sum()),
        "total_elapsed_s": total_elapsed,
        "coord_history": coord_history,
        "polish_stats": polish_stats,
    }
    summary_path = output_dir / "summary_fast.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
