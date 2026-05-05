#!/usr/bin/env python3
"""Benchmark global/uniform DTL-rate optimization strategies.

The script targets the 3-parameter global mode and uses the public
``GeneReconModel`` autograd bridge. ``recommended-fp32`` uses the production
``optimize_global_rates_lbfgs`` helper when available, and can fall back to the
direct experimental ``GeneReconModel`` LBFGS path with ``--helper-mode direct``.
``bf16-start-fp32-polish`` uses the helper in two phases when bf16 is runnable:
a short bf16 start followed by an fp32 polish seeded from the bf16 rates.
``bf16-resident-threshold-fp32-polish`` drives a pure resident-bf16 initial
phase from this harness and switches to fp32 when configured thresholds fire.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpurec import GeneReconModel


PRODUCTION_HELPER_CANDIDATES = (
    ("gpurec.optimization", "optimize_global_rates_lbfgs"),
    ("gpurec.optimization.global_parameter_optimizer", "optimize_global_rates_lbfgs"),
    ("gpurec.api", "optimize_global_rates_lbfgs"),
    ("gpurec", "optimize_global_rates_lbfgs"),
)

DEFAULT_FLAGS = {
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
    "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
    "GPUREC_FUSED_DTS_BACKWARD_ACCUM": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL": "tree",
    "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
    "GPUREC_BACKWARD_LEAF_INDEX": "1",
    "GPUREC_FUSED_WAVE_PARAM_ACCUM": "1",
    "GPUREC_DTS_PIBAR_UD_FUSION": "1",
    "GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES": "1",
    "GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS": "1",
    "GPUREC_DTS_GRAD_MT_TWO_STAGE": "1",
    "GPUREC_BACKWARD_PARENT_REDUCED_DTS": "tiled",
    "GPUREC_BACKWARD_PARENT_REDUCED_DTS_TILE_SPLITS": "64",
}


@dataclass
class EvalRecord:
    strategy: str
    eval_idx: int
    elapsed_s: float
    nll: float
    grad_inf: float
    d_rate: float
    l_rate: float
    t_rate: float
    max_rate_rel_err: float
    target_nll_gap: float
    phase: str = "unknown"
    dtype: str = "unknown"
    phase_eval_idx: int | None = None
    eval_time_s: float | None = None
    forward_time_s: float | None = None
    backward_time_s: float | None = None
    relative_rate_step: float | None = None
    nll_change: float | None = None


@dataclass(frozen=True)
class ProductionHelper:
    source: str
    func: Callable[..., Any]


def _parse_dtype(text: str) -> torch.dtype:
    text = text.lower().strip()
    if text in ("fp32", "float32"):
        return torch.float32
    if text in ("fp64", "float64"):
        return torch.float64
    if text in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise argparse.ArgumentTypeError("dtype must be fp32, fp64, or bf16")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/data/test_trees_100")
    parser.add_argument("--cache-dir", default="/tmp/gpurec_paramopt_cache")
    parser.add_argument(
        "--strategies",
        default="eval-target,recommended-fp32,bf16-start-fp32-polish,scipy-lbfgsb-fp32,scipy-lbfgsb-fp64-polish,bad-floor-init-guard",
    )
    parser.add_argument("--init-rate", type=float, default=0.05)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--maxfun", type=int, default=60)
    parser.add_argument("--torch-lbfgs-max-iter", type=int, default=12)
    parser.add_argument("--bf16-start-steps", type=int, default=4)
    parser.add_argument("--bf16-start-lr", type=float, default=0.05)
    parser.add_argument("--fp32-polish-steps", type=int, default=8)
    parser.add_argument("--bf16-threshold-min-steps", type=int, default=2)
    parser.add_argument("--bf16-threshold-max-steps", type=int, default=12)
    parser.add_argument("--bf16-switch-rate-rtol", type=float, default=5e-3)
    parser.add_argument("--bf16-switch-nll-abs-tol", type=float, default=1e-2)
    parser.add_argument(
        "--bf16-switch-criteria",
        choices=("any", "all"),
        default="any",
        help=(
            "Threshold handoff rule for resident bf16 phase. 'any' switches "
            "when any enabled threshold fires; 'all' waits for all enabled "
            "thresholds. Set a threshold <= 0 to disable it."
        ),
    )
    parser.add_argument("--adam-steps", type=int, default=3)
    parser.add_argument("--adam-lr", type=float, default=0.35)
    parser.add_argument("--fixed-iters-Pi", type=int, default=6)
    parser.add_argument("--max-wave-size", type=int, default=32768)
    parser.add_argument(
        "--max-families",
        type=int,
        default=None,
        help="Use only the first N sorted g_*.nwk files without modifying the dataset.",
    )
    parser.add_argument(
        "--allow-missing-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow benchmark runs when output_global reference files are absent.",
    )
    parser.add_argument("--nll-tol", type=float, default=5e-2)
    parser.add_argument("--rate-rtol", type=float, default=1e-2)
    parser.add_argument("--gtol", type=float, default=1e-3)
    parser.add_argument("--ftol", type=float, default=1e-7)
    parser.add_argument(
        "--helper-mode",
        choices=("auto", "direct", "require"),
        default="auto",
        help=(
            "For recommended-fp32, use optimize_global_rates_lbfgs when "
            "available (auto), always use the direct GeneReconModel LBFGS path "
            "(direct), or fail unless the helper is importable (require)."
        ),
    )
    parser.add_argument(
        "--allow-floor-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow optimizer initialization at or below --min-rate.",
    )
    parser.add_argument("--print-evals", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _resolve_production_helper() -> ProductionHelper | None:
    for module_name, attr_name in PRODUCTION_HELPER_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        func = getattr(module, attr_name, None)
        if callable(func):
            return ProductionHelper(f"{module_name}.{attr_name}", func)
    return None


def _gene_paths(root: Path, max_families: int | None = None) -> list[str]:
    paths = [str(p) for p in sorted(root.glob("g_*.nwk"))]
    if max_families is None:
        return paths
    if max_families < 1:
        raise ValueError("--max-families must be >= 1")
    return paths[:max_families]


def _reference_rates(root: Path, *, allow_missing: bool = False) -> torch.Tensor:
    path = root / "output_global" / "model_parameters" / "model_parameters.txt"
    if allow_missing and not path.exists():
        return torch.full((3,), float("nan"), dtype=torch.float64)
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4:
            return torch.tensor([float(parts[1]), float(parts[2]), float(parts[3])])
    raise RuntimeError(f"could not parse rates from {path}")


def _reference_nll(root: Path, *, allow_missing: bool = False) -> float:
    path = root / "output_global" / "per_fam_likelihoods.txt"
    if allow_missing and not path.exists():
        return float("nan")
    total_ll = 0.0
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            total_ll += float(parts[1])
    # AleRax writes natural-log likelihoods.  gpurec's internal dynamic
    # program and optimizer report log2 NLL, so convert nats to bits here.
    return -total_ll / math.log(2.0)


def _validate_interior_init_rates(
    init_rates: tuple[float, float, float],
    *,
    min_rate: float,
    strategy: str,
    allow_floor_init: bool,
) -> None:
    if not math.isfinite(min_rate) or min_rate <= 0.0:
        raise ValueError(f"{strategy}: min_rate must be finite and positive, got {min_rate!r}")
    bad_rates = [rate for rate in init_rates if not math.isfinite(rate) or rate <= 0.0]
    if bad_rates:
        raise ValueError(f"{strategy}: init rates must be finite and positive, got {init_rates!r}")
    if allow_floor_init:
        return
    floorish = [rate for rate in init_rates if rate <= min_rate * (1.0 + 1e-12)]
    if floorish:
        raise ValueError(
            f"{strategy}: refusing init_rates={init_rates!r} at/below min_rate={min_rate:.3e}; "
            "use an interior initialization such as 0.02 or 0.05, and keep min_rate as a constraint"
        )


def _make_model(
    root: Path,
    *,
    dtype: torch.dtype,
    init_rates: tuple[float, float, float],
    args: argparse.Namespace,
) -> GeneReconModel:
    build_start = time.perf_counter()
    model = GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=_gene_paths(root, args.max_families),
        mode="global",
        pibar_mode="uniform",
        device="cuda",
        dtype=dtype,
        theta_init_rates=init_rates,
        preprocess_cache_dir=args.cache_dir,
        fixed_iters_Pi=args.fixed_iters_Pi,
        max_wave_size=args.max_wave_size,
        neumann_terms=3,
        use_pruning=True,
        pruning_threshold=1e-6,
    )
    if model.theta.device.type == "cuda":
        torch.cuda.synchronize(model.theta.device)
    model._bench_build_time_s = time.perf_counter() - build_start
    return model


def _rate_metrics(model: GeneReconModel, target_rates: torch.Tensor, target_nll: float) -> tuple[float, float, float, float, float]:
    rates = model.rates.detach().to(dtype=torch.float64, device="cpu")
    target = target_rates.to(dtype=torch.float64)
    if torch.isfinite(target).all():
        rel = torch.max(torch.abs(rates - target) / torch.clamp(target.abs(), min=1e-30)).item()
    else:
        rel = float("nan")
    return float(rates[0]), float(rates[1]), float(rates[2]), float(rel), target_nll


def _evaluate_with_grad(
    model: GeneReconModel,
    *,
    target_rates: torch.Tensor,
    target_nll: float,
    min_rate: float,
) -> tuple[float, torch.Tensor, float, tuple[float, float, float, float, float]]:
    model.clamp_theta_(min_rate=min_rate)
    model.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    torch.cuda.synchronize()
    grad = model.theta.grad.detach().clone()
    grad_inf = float(torch.max(torch.abs(grad)).detach().cpu())
    metrics = _rate_metrics(model, target_rates, target_nll)
    return float(loss.detach().cpu()), grad, grad_inf, metrics


def _record(
    records: list[EvalRecord],
    strategy: str,
    eval_idx: int,
    start_time: float,
    nll: float,
    grad_inf: float,
    metrics: tuple[float, float, float, float, float],
    target_nll: float,
    phase: str = "unknown",
    dtype: torch.dtype | str | None = None,
    phase_eval_idx: int | None = None,
    eval_time_s: float | None = None,
    forward_time_s: float | None = None,
    backward_time_s: float | None = None,
    relative_rate_step: float | None = None,
    nll_change: float | None = None,
) -> None:
    d_rate, l_rate, t_rate, max_rate_rel_err, _ = metrics
    dtype_name = _dtype_name(dtype) if isinstance(dtype, torch.dtype) else (dtype or "unknown")
    records.append(
        EvalRecord(
            strategy=strategy,
            eval_idx=eval_idx,
            elapsed_s=time.perf_counter() - start_time,
            nll=nll,
            grad_inf=grad_inf,
            d_rate=d_rate,
            l_rate=l_rate,
            t_rate=t_rate,
            max_rate_rel_err=max_rate_rel_err,
            target_nll_gap=nll - target_nll,
            phase=phase,
            dtype=str(dtype_name),
            phase_eval_idx=phase_eval_idx,
            eval_time_s=eval_time_s,
            forward_time_s=forward_time_s,
            backward_time_s=backward_time_s,
            relative_rate_step=relative_rate_step,
            nll_change=nll_change,
        )
    )


def _print_record(r: EvalRecord) -> None:
    print(
        "eval",
        "strategy", r.strategy,
        "phase", r.phase,
        "dtype", r.dtype,
        "idx", r.eval_idx,
        "phase_idx", r.phase_eval_idx if r.phase_eval_idx is not None else "n/a",
        "elapsed_s", f"{r.elapsed_s:.6f}",
        "eval_time_s", f"{r.eval_time_s:.6f}" if r.eval_time_s is not None else "n/a",
        "nll", f"{r.nll:.8f}",
        "target_gap", f"{r.target_nll_gap:.8e}",
        "grad_inf", f"{r.grad_inf:.8e}",
        "D", f"{r.d_rate:.10e}",
        "L", f"{r.l_rate:.10e}",
        "T", f"{r.t_rate:.10e}",
        "rate_rel", f"{r.max_rate_rel_err:.8e}",
        "relative_rate_step", f"{r.relative_rate_step:.8e}" if r.relative_rate_step is not None else "n/a",
        "nll_change", f"{r.nll_change:.8e}" if r.nll_change is not None else "n/a",
        flush=True,
    )


def _first_hit(records: list[EvalRecord], *, nll_tol: float, rate_rtol: float) -> EvalRecord | None:
    for r in records:
        if r.target_nll_gap <= nll_tol and r.max_rate_rel_err <= rate_rtol:
            return r
    return None


def _run_scipy_lbfgsb(
    *,
    name: str,
    root: Path,
    dtype: torch.dtype,
    init_rates: tuple[float, float, float],
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> tuple[list[EvalRecord], object]:
    _validate_interior_init_rates(
        init_rates,
        min_rate=args.min_rate,
        strategy=name,
        allow_floor_init=args.allow_floor_init,
    )
    min_theta = math.log2(args.min_rate)
    model = _make_model(root, dtype=dtype, init_rates=init_rates, args=args)
    records: list[EvalRecord] = []
    start = time.perf_counter()
    eval_count = 0

    def fun_and_grad(theta_np: np.ndarray):
        nonlocal eval_count
        eval_start = time.perf_counter()
        theta_np = np.maximum(theta_np, min_theta)
        with torch.no_grad():
            model.theta.copy_(torch.as_tensor(theta_np, dtype=dtype, device=model.theta.device))
        nll, grad, grad_inf, metrics = _evaluate_with_grad(
            model,
            target_rates=target_rates,
            target_nll=target_nll,
            min_rate=args.min_rate,
        )
        eval_count += 1
        _record(
            records,
            name,
            eval_count,
            start,
            nll,
            grad_inf,
            metrics,
            target_nll,
            phase=f"{_dtype_name(dtype)}_scipy_lbfgsb",
            dtype=dtype,
            phase_eval_idx=eval_count,
            eval_time_s=time.perf_counter() - eval_start,
        )
        if args.print_evals:
            _print_record(records[-1])
        grad_np = grad.detach().reshape(-1).to(dtype=torch.float64, device="cpu").numpy()
        np.nan_to_num(grad_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return float(nll), grad_np

    x0 = torch.log2(torch.tensor(init_rates, dtype=torch.float64)).numpy()
    result = minimize(
        fun_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(min_theta, None)] * 3,
        options={
            "maxiter": args.maxiter,
            "maxfun": args.maxfun,
            "gtol": args.gtol,
            "ftol": args.ftol,
            "maxls": 20,
        },
    )
    result.bench_build_time_s = float(getattr(model, "_bench_build_time_s", float("nan")))
    result.bench_optimizer_time_s = records[-1].elapsed_s if records else 0.0
    result.bench_phase_summaries = _phase_summaries_from_records(records, name)
    return records, result


def _run_torch_lbfgs(
    *,
    name: str | None = None,
    root: Path,
    dtype: torch.dtype,
    init_rates: tuple[float, float, float],
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
    print_evals: bool | None = None,
    phase: str | None = None,
) -> tuple[list[EvalRecord], object]:
    name = name or f"torch-lbfgs-{_dtype_name(dtype)}"
    phase_name = phase or ("fp32_lbfgs" if dtype == torch.float32 else "lbfgs")
    _validate_interior_init_rates(
        init_rates,
        min_rate=args.min_rate,
        strategy=name,
        allow_floor_init=args.allow_floor_init,
    )
    should_print_evals = args.print_evals if print_evals is None else print_evals
    model = _make_model(root, dtype=dtype, init_rates=init_rates, args=args)
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=args.torch_lbfgs_max_iter,
        max_eval=args.maxfun,
        history_size=10,
        tolerance_grad=args.gtol,
        tolerance_change=args.ftol,
        line_search_fn="strong_wolfe",
    )
    records: list[EvalRecord] = []
    start = time.perf_counter()
    eval_count = 0

    def closure():
        nonlocal eval_count
        eval_start = time.perf_counter()
        nll, _grad, grad_inf, metrics = _evaluate_with_grad(
            model,
            target_rates=target_rates,
            target_nll=target_nll,
            min_rate=args.min_rate,
        )
        eval_count += 1
        _record(
            records,
            name,
            eval_count,
            start,
            nll,
            grad_inf,
            metrics,
            target_nll,
            phase=phase_name,
            dtype=dtype,
            phase_eval_idx=eval_count,
            eval_time_s=time.perf_counter() - eval_start,
        )
        if should_print_evals:
            _print_record(records[-1])
        return model.theta.grad.new_tensor(nll)

    result = opt.step(closure)
    model.clamp_theta_(min_rate=args.min_rate)
    result_value = _as_float(result)
    return records, SimpleNamespace(
        success=True,
        message="ok" if result_value is None else f"optimizer_return={result_value:.8f}",
        build_time_s=float(getattr(model, "_bench_build_time_s", float("nan"))),
        optimizer_time_s=records[-1].elapsed_s if records else 0.0,
        phase_summaries=_phase_summaries_from_records(records, name),
    )


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().to(dtype=torch.float64, device="cpu"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    out = _as_float(value)
    return out if out is not None and math.isfinite(out) else out


def _dtype_from_phase(phase: str) -> str:
    phase = phase.lower()
    if "bf16" in phase or "bfloat16" in phase:
        return "bf16"
    if "fp64" in phase or "float64" in phase:
        return "fp64"
    if "fp32" in phase or "float32" in phase:
        return "fp32"
    return "unknown"


def _result_success(result: object) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success", True))
    return bool(getattr(result, "success", True))


def _result_message(result: object) -> str:
    if isinstance(result, dict):
        return str(result.get("message", "ok"))
    return str(getattr(result, "message", "ok"))


def _result_metric(result: object, name: str) -> float | None:
    candidates = (name, f"bench_{name}")
    for key in candidates:
        value = None
        if isinstance(result, dict):
            value = result.get(key)
            timing = result.get("timing")
            if value is None and isinstance(timing, dict):
                value = timing.get(key)
        else:
            value = getattr(result, key, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _result_phase_summaries(result: object) -> list[dict[str, object]]:
    for key in ("phase_summaries", "bench_phase_summaries"):
        if isinstance(result, dict):
            value = result.get(key)
        else:
            value = getattr(result, key, None)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if isinstance(result, dict):
        timing = result.get("timing")
        if isinstance(timing, dict):
            phases = timing.get("phases")
            if isinstance(phases, list):
                return [item for item in phases if isinstance(item, dict)]
    return []


def _phase_summaries_from_records(records: list[EvalRecord], strategy: str) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    seen: list[tuple[str, str]] = []
    for record in records:
        key = (record.phase, record.dtype)
        if key not in seen:
            seen.append(key)
    for phase, dtype in seen:
        group = [record for record in records if record.phase == phase and record.dtype == dtype]
        if not group:
            continue
        eval_times = [record.eval_time_s for record in group if record.eval_time_s is not None]
        forward_times = [record.forward_time_s for record in group if record.forward_time_s is not None]
        backward_times = [record.backward_time_s for record in group if record.backward_time_s is not None]
        summaries.append(
            {
                "strategy": strategy,
                "phase": phase,
                "dtype": dtype,
                "evaluations": len(group),
                "time_s": sum(eval_times) if eval_times else max(record.elapsed_s for record in group),
                "avg_eval_time_s": sum(eval_times) / len(eval_times) if eval_times else None,
                "forward_time_s": sum(forward_times) if forward_times else None,
                "backward_time_s": sum(backward_times) if backward_times else None,
            }
        )
    return summaries


def _phase_summaries_from_helper(
    *,
    raw_result: object,
    records: list[EvalRecord],
    strategy: str,
) -> list[dict[str, object]]:
    timing = raw_result.get("timing", {}) if isinstance(raw_result, dict) else {}
    raw_phases = timing.get("phases", []) if isinstance(timing, dict) else []
    summaries: list[dict[str, object]] = []
    for phase in raw_phases:
        if not isinstance(phase, dict):
            continue
        phase_name = str(phase.get("phase", "unknown"))
        group = [record for record in records if record.phase == phase_name]
        dtype = group[0].dtype if group else _dtype_from_phase(phase_name)
        forward_times = [record.forward_time_s for record in group if record.forward_time_s is not None]
        backward_times = [record.backward_time_s for record in group if record.backward_time_s is not None]
        eval_times = [record.eval_time_s for record in group if record.eval_time_s is not None]
        summaries.append(
            {
                "strategy": strategy,
                "phase": phase_name,
                "dtype": dtype,
                "evaluations": int(phase.get("evaluations", len(group))),
                "time_s": _optional_float(phase.get("time_s")),
                "avg_eval_time_s": sum(eval_times) / len(eval_times) if eval_times else None,
                "forward_time_s": sum(forward_times) if forward_times else None,
                "backward_time_s": sum(backward_times) if backward_times else None,
                "optimizer": phase.get("optimizer"),
                "switch_reason": phase.get("switch_reason"),
            }
        )
    if summaries:
        return summaries
    return _phase_summaries_from_records(records, strategy)


def _print_phase_summaries(strategy: str, result: object) -> None:
    for phase in _result_phase_summaries(result):
        phase_name = str(phase.get("phase", "unknown"))
        dtype = str(phase.get("dtype", _dtype_from_phase(phase_name)))
        time_s = _optional_float(phase.get("time_s"))
        avg_eval_time_s = _optional_float(phase.get("avg_eval_time_s"))
        forward_time_s = _optional_float(phase.get("forward_time_s"))
        backward_time_s = _optional_float(phase.get("backward_time_s"))
        switch_reason = phase.get("switch_reason")
        optimizer = phase.get("optimizer")
        print(
            "phase_summary",
            "strategy", str(phase.get("strategy", strategy)),
            "phase", phase_name,
            "dtype", dtype,
            "evals", phase.get("evaluations", "n/a"),
            "time_s", f"{time_s:.6f}" if time_s is not None else "n/a",
            "avg_eval_time_s", f"{avg_eval_time_s:.6f}" if avg_eval_time_s is not None else "n/a",
            "forward_time_s", f"{forward_time_s:.6f}" if forward_time_s is not None else "n/a",
            "backward_time_s", f"{backward_time_s:.6f}" if backward_time_s is not None else "n/a",
            "optimizer", str(optimizer) if optimizer is not None else "n/a",
            "switch_reason", str(switch_reason).replace("\n", " ") if switch_reason is not None else "n/a",
            flush=True,
        )


def _cuda_peak_memory_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return float("nan"), float("nan")
    torch.cuda.synchronize()
    return (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        torch.cuda.max_memory_reserved() / (1024.0 * 1024.0),
    )


def _records_from_helper_history(
    *,
    history: list[dict[str, Any]],
    strategy: str,
    target_rates: torch.Tensor,
    target_nll: float,
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    target = target_rates.to(dtype=torch.float64)
    for idx, item in enumerate(history, start=1):
        rates = torch.as_tensor(item["rates"], dtype=torch.float64, device="cpu").reshape(-1)
        if torch.isfinite(target).all():
            rel = torch.max(torch.abs(rates[:3] - target) / torch.clamp(target.abs(), min=1e-30)).item()
        else:
            rel = float("nan")
        nll = float(item.get("negative_log_likelihood", item["nll"]))
        phase = str(item.get("phase", "unknown"))
        dtype = str(item.get("dtype", _dtype_from_phase(phase)))
        records.append(
            EvalRecord(
                strategy=strategy,
                eval_idx=int(item.get("eval", idx)),
                elapsed_s=float(item.get("elapsed_s", 0.0)),
                nll=nll,
                grad_inf=float(item.get("grad_infinity_norm", float("nan"))),
                d_rate=float(rates[0]),
                l_rate=float(rates[1]),
                t_rate=float(rates[2]),
                max_rate_rel_err=float(rel),
                target_nll_gap=nll - target_nll,
                phase=phase,
                dtype=dtype,
                phase_eval_idx=int(item["phase_eval"]) if item.get("phase_eval") is not None else None,
                eval_time_s=_optional_float(item.get("eval_time_s")),
                forward_time_s=_optional_float(item.get("forward_time_s")),
                backward_time_s=_optional_float(item.get("backward_time_s")),
                relative_rate_step=_optional_float(item.get("relative_rate_step")),
                nll_change=_optional_float(item.get("nll_change")),
            )
        )
    return records


def _run_production_helper_lbfgs(
    *,
    helper: ProductionHelper,
    root: Path,
    dtype: torch.dtype,
    init_rates: tuple[float, float, float],
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> tuple[list[EvalRecord], object]:
    name = f"recommended-fp32-helper"
    if dtype != torch.float32:
        raise ValueError(f"{name}: production helper benchmark currently expects fp32, got {_dtype_name(dtype)}")
    _validate_interior_init_rates(
        init_rates,
        min_rate=args.min_rate,
        strategy=name,
        allow_floor_init=args.allow_floor_init,
    )
    model = _make_model(root, dtype=dtype, init_rates=init_rates, args=args)
    records: list[EvalRecord] = []
    start = time.perf_counter()
    eval_count = 0

    def helper_callback(*callback_args: object, **callback_kwargs: object) -> None:
        nonlocal eval_count
        nll = _as_float(callback_kwargs.get("nll"))
        if nll is None:
            for value in callback_args:
                nll = _as_float(value)
                if nll is not None:
                    break
        if nll is None:
            return

        grad_value = callback_kwargs.get("grad")
        if isinstance(grad_value, torch.Tensor):
            grad_inf = float(torch.max(torch.abs(grad_value)).detach().cpu())
        elif model.theta.grad is not None:
            grad_inf = float(torch.max(torch.abs(model.theta.grad)).detach().cpu())
        else:
            grad_inf = float("nan")

        eval_count += 1
        metrics = _rate_metrics(model, target_rates, target_nll)
        _record(records, name, eval_count, start, nll, grad_inf, metrics, target_nll)
        if args.print_evals:
            _print_record(records[-1])

    signature = inspect.signature(helper.func)
    parameters = signature.parameters
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    candidate_kwargs: dict[str, object] = {
        "init_rates": init_rates,
        "min_rate": args.min_rate,
        "override_floor_init": args.allow_floor_init,
        "steps": args.torch_lbfgs_max_iter,
        "lr": 1.0,
        "max_eval": args.maxfun,
        "max_evals": args.maxfun,
        "maxfun": args.maxfun,
        "history_size": 10,
        "tolerance_grad": args.gtol,
        "gtol": args.gtol,
        "tolerance_change": args.ftol,
        "ftol": args.ftol,
        "line_search_fn": "strong_wolfe",
        "dtype": dtype,
        "fp64_polish": False,
        "verbose": False,
        "bf16_start_lr": args.bf16_start_lr,
        "bf16_switch_rate_rtol": args.bf16_switch_rate_rtol,
        "bf16_switch_nll_abs_tol": args.bf16_switch_nll_abs_tol,
        "bf16_switch_min_steps": args.bf16_threshold_min_steps,
        "bf16_switch_max_steps": args.bf16_threshold_max_steps,
        "bf16_switch_criteria": args.bf16_switch_criteria,
    }
    kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if has_var_kwargs or key in parameters
    }
    if has_var_kwargs or "callback" in parameters:
        kwargs["callback"] = helper_callback
    elif "eval_callback" in parameters:
        kwargs["eval_callback"] = helper_callback

    print(
        "helper_status",
        "strategy", "recommended-fp32",
        "mode", "production_helper",
        "source", helper.source,
        "callback", int("callback" in kwargs or "eval_callback" in kwargs),
        flush=True,
    )
    raw_result = helper.func(model, **kwargs)
    model.clamp_theta_(min_rate=args.min_rate)

    if isinstance(raw_result, dict) and isinstance(raw_result.get("history"), list):
        records = _records_from_helper_history(
            history=raw_result["history"],
            strategy=name,
            target_rates=target_rates,
            target_nll=target_nll,
        )
        if args.print_evals:
            for record in records:
                _print_record(record)

    if not records:
        nll, _grad, grad_inf, metrics = _evaluate_with_grad(
            model,
            target_rates=target_rates,
            target_nll=target_nll,
            min_rate=args.min_rate,
        )
        _record(records, name, 1, start, nll, grad_inf, metrics, target_nll)
        if args.print_evals:
            _print_record(records[-1])
        message = f"helper={helper.source}; final_eval_only; raw_message={_result_message(raw_result)}"
    else:
        timing = raw_result.get("timing", {}) if isinstance(raw_result, dict) else {}
        total_s = timing.get("total_s", "n/a") if isinstance(timing, dict) else "n/a"
        initialization = raw_result.get("initialization", "n/a") if isinstance(raw_result, dict) else "n/a"
        message = (
            f"helper={helper.source}; initialization={initialization}; "
            f"helper_total_s={total_s}; raw_message={_result_message(raw_result)}"
        )

    helper_timing = raw_result.get("timing", {}) if isinstance(raw_result, dict) else {}
    helper_total_s = helper_timing.get("total_s") if isinstance(helper_timing, dict) else None
    return records, SimpleNamespace(
        success=_result_success(raw_result),
        message=message,
        build_time_s=float(getattr(model, "_bench_build_time_s", float("nan"))),
        optimizer_time_s=float(helper_total_s) if helper_total_s is not None else (records[-1].elapsed_s if records else 0.0),
        phase_summaries=_phase_summaries_from_helper(
            raw_result=raw_result,
            records=records,
            strategy=name,
        ),
    )


def _run_recommended_lbfgs(
    *,
    root: Path,
    init_rates: tuple[float, float, float],
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> tuple[list[EvalRecord], object]:
    if args.helper_mode != "direct":
        helper = _resolve_production_helper()
        if helper is not None:
            try:
                return _run_production_helper_lbfgs(
                    helper=helper,
                    root=root,
                    dtype=torch.float32,
                    init_rates=init_rates,
                    target_rates=target_rates,
                    target_nll=target_nll,
                    args=args,
                )
            except TypeError as exc:
                if args.helper_mode == "require":
                    raise
                print(
                    "helper_status",
                    "strategy", "recommended-fp32",
                    "mode", "direct_fallback",
                    "reason", f"incompatible_helper:{type(exc).__name__}:{str(exc).replace(chr(10), ' ')}",
                    flush=True,
                )
        elif args.helper_mode == "require":
            raise RuntimeError(
                "recommended-fp32 requires optimize_global_rates_lbfgs, but no "
                f"candidate was found in {PRODUCTION_HELPER_CANDIDATES!r}"
            )

    print(
        "helper_status",
        "strategy", "recommended-fp32",
        "mode", "direct_fallback",
        "reason", "helper_unavailable" if args.helper_mode != "direct" else "helper_mode_direct",
        flush=True,
    )
    return _run_torch_lbfgs(
        name="recommended-fp32-direct",
        root=root,
        dtype=torch.float32,
        init_rates=init_rates,
        target_rates=target_rates,
        target_nll=target_nll,
        args=args,
    )


def _run_scipy_lbfgsb_fp64_polish(
    *,
    root: Path,
    init_rates: tuple[float, float, float],
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> tuple[list[EvalRecord], object]:
    seed_records, _seed_result = _run_torch_lbfgs(
        name="fp64-polish-seed-fp32",
        root=root,
        dtype=torch.float32,
        init_rates=init_rates,
        target_rates=target_rates,
        target_nll=target_nll,
        args=args,
        print_evals=False,
    )
    if not seed_records:
        raise RuntimeError("fp64 polish could not run because the fp32 seed produced no records")
    seed = seed_records[-1]
    seed_rates = (seed.d_rate, seed.l_rate, seed.t_rate)
    print(
        "polish_seed",
        "strategy", "scipy-lbfgsb-fp64-polish",
        "seed_strategy", "fp64-polish-seed-fp32",
        "seed_evals", len(seed_records),
        "seed_time_s", f"{seed.elapsed_s:.6f}",
        "seed_gap", f"{seed.target_nll_gap:.8e}",
        "seed_rate_rel", f"{seed.max_rate_rel_err:.8e}",
        "D", f"{seed.d_rate:.10e}",
        "L", f"{seed.l_rate:.10e}",
        "T", f"{seed.t_rate:.10e}",
        flush=True,
    )
    return _run_scipy_lbfgsb(
        name="scipy-lbfgsb-fp64-polish",
        root=root,
        dtype=torch.float64,
        init_rates=seed_rates,
        target_rates=target_rates,
        target_nll=target_nll,
        args=args,
    )


def _helper_kwargs(
    helper: ProductionHelper,
    *,
    init_rates: tuple[float, float, float],
    dtype: torch.dtype,
    steps: int,
    max_eval: int | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    signature = inspect.signature(helper.func)
    parameters = signature.parameters
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    candidate_kwargs: dict[str, object] = {
        "init_rates": init_rates,
        "min_rate": args.min_rate,
        "override_floor_init": args.allow_floor_init,
        "steps": steps,
        "lr": 1.0,
        "max_eval": max_eval,
        "max_evals": max_eval,
        "maxfun": max_eval,
        "history_size": 10,
        "tolerance_grad": args.gtol,
        "gtol": args.gtol,
        "tolerance_change": args.ftol,
        "ftol": args.ftol,
        "line_search_fn": "strong_wolfe",
        "dtype": dtype,
        "fp64_polish": False,
        "verbose": False,
        "bf16_start_lr": args.bf16_start_lr,
        "bf16_switch_rate_rtol": args.bf16_switch_rate_rtol,
        "bf16_switch_nll_abs_tol": args.bf16_switch_nll_abs_tol,
        "bf16_switch_min_steps": args.bf16_threshold_min_steps,
        "bf16_switch_max_steps": args.bf16_threshold_max_steps,
        "bf16_switch_criteria": args.bf16_switch_criteria,
    }
    return {
        key: value
        for key, value in candidate_kwargs.items()
        if value is not None and (has_var_kwargs or key in parameters)
    }


def _run_helper_phase(
    *,
    helper: ProductionHelper,
    phase_name: str,
    root: Path,
    dtype: torch.dtype,
    init_rates: tuple[float, float, float],
    steps: int,
    max_eval: int | None,
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> tuple[list[EvalRecord], object]:
    _validate_interior_init_rates(
        init_rates,
        min_rate=args.min_rate,
        strategy=phase_name,
        allow_floor_init=args.allow_floor_init,
    )
    model = _make_model(root, dtype=dtype, init_rates=init_rates, args=args)
    kwargs = _helper_kwargs(
        helper,
        init_rates=init_rates,
        dtype=dtype,
        steps=steps,
        max_eval=max_eval,
        args=args,
    )
    raw_result = helper.func(model, **kwargs)
    model.clamp_theta_(min_rate=args.min_rate)
    records: list[EvalRecord] = []
    if isinstance(raw_result, dict) and isinstance(raw_result.get("history"), list):
        records = _records_from_helper_history(
            history=raw_result["history"],
            strategy=phase_name,
            target_rates=target_rates,
            target_nll=target_nll,
        )
    if not records:
        eval_start = time.perf_counter()
        nll, _grad, grad_inf, metrics = _evaluate_with_grad(
            model,
            target_rates=target_rates,
            target_nll=target_nll,
            min_rate=args.min_rate,
        )
        _record(records, phase_name, 1, eval_start, nll, grad_inf, metrics, target_nll)
    timing = raw_result.get("timing", {}) if isinstance(raw_result, dict) else {}
    optimizer_time_s = timing.get("total_s") if isinstance(timing, dict) else None
    return records, SimpleNamespace(
        success=_result_success(raw_result),
        message=f"helper={helper.source}; raw_message={_result_message(raw_result)}",
        build_time_s=float(getattr(model, "_bench_build_time_s", float("nan"))),
        optimizer_time_s=float(optimizer_time_s) if optimizer_time_s is not None else (records[-1].elapsed_s if records else 0.0),
        phase_summaries=_phase_summaries_from_helper(
            raw_result=raw_result,
            records=records,
            strategy=phase_name,
        ),
    )


def _run_bf16_start_fp32_polish(
    *,
    root: Path,
    init_rates: tuple[float, float, float],
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> tuple[list[EvalRecord], object]:
    helper = _resolve_production_helper()
    if helper is None:
        return [], SimpleNamespace(
            success=False,
            message=(
                "bf16-start/fp32-polish requires optimize_global_rates_lbfgs, "
                f"but no candidate was found in {PRODUCTION_HELPER_CANDIDATES!r}"
            ),
            build_time_s=0.0,
            optimizer_time_s=0.0,
        )

    print(
        "helper_status",
        "strategy", "bf16-start-fp32-polish",
        "mode", "production_helper_bf16_start",
        "source", helper.source,
        "bf16_start_steps", args.bf16_start_steps,
        "fp32_polish_steps", args.fp32_polish_steps,
        flush=True,
    )

    _validate_interior_init_rates(
        init_rates,
        min_rate=args.min_rate,
        strategy="bf16-start-fp32-polish",
        allow_floor_init=args.allow_floor_init,
    )
    model = _make_model(root, dtype=torch.float32, init_rates=init_rates, args=args)
    signature = inspect.signature(helper.func)
    parameters = signature.parameters
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    candidate_kwargs: dict[str, object] = {
        "init_rates": init_rates,
        "min_rate": args.min_rate,
        "override_floor_init": args.allow_floor_init,
        "steps": args.fp32_polish_steps,
        "lr": 1.0,
        "max_eval": args.maxfun,
        "history_size": 10,
        "tolerance_grad": args.gtol,
        "tolerance_change": args.ftol,
        "line_search_fn": "strong_wolfe",
        "dtype": torch.float32,
        "bf16_start_steps": args.bf16_start_steps,
        "bf16_start_lr": args.bf16_start_lr,
        "bf16_switch_rate_rtol": args.bf16_switch_rate_rtol,
        "bf16_switch_nll_abs_tol": args.bf16_switch_nll_abs_tol,
        "bf16_switch_min_steps": args.bf16_threshold_min_steps,
        "bf16_switch_max_steps": args.bf16_threshold_max_steps,
        "bf16_switch_criteria": args.bf16_switch_criteria,
        "fp64_polish": False,
        "verbose": False,
    }
    kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if has_var_kwargs or key in parameters
    }

    try:
        raw_result = helper.func(model, **kwargs)
    except Exception as exc:
        return [], SimpleNamespace(
            success=False,
            message=f"bf16-start/fp32-polish failed: {type(exc).__name__}: {exc}",
            build_time_s=float(getattr(model, "_bench_build_time_s", float("nan"))),
            optimizer_time_s=0.0,
        )

    records: list[EvalRecord] = []
    if isinstance(raw_result, dict) and isinstance(raw_result.get("history"), list):
        records = _records_from_helper_history(
            history=raw_result["history"],
            strategy="bf16-start-fp32-polish",
            target_rates=target_rates,
            target_nll=target_nll,
        )
    timing = raw_result.get("timing", {}) if isinstance(raw_result, dict) else {}
    optimizer_time_s = timing.get("total_s") if isinstance(timing, dict) else None
    initialization = raw_result.get("initialization", "n/a") if isinstance(raw_result, dict) else "n/a"
    phases = []
    if isinstance(timing, dict):
        phases = [
            str(phase.get("phase", ""))
            for phase in timing.get("phases", [])
            if isinstance(phase, dict)
        ]
    message = (
        f"helper={helper.source}; initialization={initialization}; "
        f"phases={','.join(phases)}; raw_message={_result_message(raw_result)}"
    )
    return records, SimpleNamespace(
        success=_result_success(raw_result),
        message=message,
        build_time_s=float(getattr(model, "_bench_build_time_s", float("nan"))),
        optimizer_time_s=(
            float(optimizer_time_s)
            if optimizer_time_s is not None
            else (records[-1].elapsed_s if records else 0.0)
        ),
        phase_summaries=_phase_summaries_from_helper(
            raw_result=raw_result,
            records=records,
            strategy="bf16-start-fp32-polish",
        ),
    )


def _threshold_enabled(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _handoff_reason(
    *,
    step_idx: int,
    min_steps: int,
    max_steps: int,
    relative_rate_step: float | None,
    nll_change: float | None,
    args: argparse.Namespace,
) -> str | None:
    if step_idx >= max_steps:
        return f"max_steps:{max_steps}"
    if step_idx < min_steps:
        return None

    checks: list[tuple[str, bool]] = []
    if _threshold_enabled(args.bf16_switch_rate_rtol):
        checks.append(
            (
                f"relative_rate_step<={args.bf16_switch_rate_rtol:.3e}",
                relative_rate_step is not None and relative_rate_step <= args.bf16_switch_rate_rtol,
            )
        )
    if _threshold_enabled(args.bf16_switch_nll_abs_tol):
        improvement = None if nll_change is None else abs(nll_change)
        checks.append(
            (
                f"abs_nll_change<={args.bf16_switch_nll_abs_tol:.3e}",
                improvement is not None and improvement <= args.bf16_switch_nll_abs_tol,
            )
        )
    if not checks:
        return None
    if args.bf16_switch_criteria == "all":
        if all(ok for _name, ok in checks):
            return "all:" + "+".join(name for name, _ok in checks)
    elif any(ok for _name, ok in checks):
        return "any:" + "+".join(name for name, ok in checks if ok)
    return None


def _run_resident_bf16_start(
    *,
    name: str,
    root: Path,
    init_rates: tuple[float, float, float],
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
    threshold: bool,
) -> tuple[list[EvalRecord], tuple[float, float, float] | None, object]:
    _validate_interior_init_rates(
        init_rates,
        min_rate=args.min_rate,
        strategy=name,
        allow_floor_init=args.allow_floor_init,
    )
    if not torch.cuda.is_bf16_supported():
        return [], None, SimpleNamespace(
            success=False,
            message="resident bf16 phase requires CUDA bf16 support",
            build_time_s=0.0,
            optimizer_time_s=0.0,
            phase_summaries=[],
        )

    max_steps = args.bf16_threshold_max_steps if threshold else args.bf16_start_steps
    min_steps = args.bf16_threshold_min_steps if threshold else args.bf16_start_steps
    if max_steps < 1:
        raise ValueError(f"{name}: bf16 phase max steps must be >= 1")
    if min_steps < 1:
        raise ValueError(f"{name}: bf16 phase min steps must be >= 1")
    if min_steps > max_steps:
        raise ValueError(f"{name}: bf16 phase min steps cannot exceed max steps")
    if args.bf16_start_lr <= 0:
        raise ValueError("--bf16-start-lr must be strictly positive")

    model = _make_model(root, dtype=torch.bfloat16, init_rates=init_rates, args=args)
    device = model.theta.device
    theta_min = math.log2(args.min_rate)
    records: list[EvalRecord] = []
    start = time.perf_counter()
    phase_start = start
    previous_rates: torch.Tensor | None = None
    previous_nll: float | None = None
    switch_reason = "not_started"

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = torch.zeros_like(model.theta, dtype=torch.float32, device=device)
    v = torch.zeros_like(model.theta, dtype=torch.float32, device=device)

    for step_idx in range(1, max_steps + 1):
        eval_start = time.perf_counter()
        with torch.no_grad():
            model.theta.clamp_(min=theta_min)

        model.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        loss = model()
        torch.cuda.synchronize(device)
        forward_time = time.perf_counter() - forward_start
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"Non-finite NLL in {name} phase=bf16_start")

        backward_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_time = time.perf_counter() - backward_start

        grad = model.theta.grad.detach().float()
        if not torch.isfinite(grad).all():
            raise FloatingPointError(f"Non-finite theta gradient in {name} phase=bf16_start")

        nll = float(loss.detach().float().cpu())
        rates_cpu = model.rates.detach().float().cpu()
        grad_inf = float(grad.detach().cpu().abs().max().item())
        rate_step = None
        if previous_rates is not None:
            denom = torch.clamp(previous_rates.abs(), min=1e-30)
            rate_step = float(torch.max(torch.abs(rates_cpu - previous_rates) / denom).item())
        nll_change = None if previous_nll is None else nll - previous_nll
        metrics = _rate_metrics(model, target_rates, target_nll)
        _record(
            records,
            name,
            len(records) + 1,
            start,
            nll,
            grad_inf,
            metrics,
            target_nll,
            phase="bf16_start",
            dtype=torch.bfloat16,
            phase_eval_idx=step_idx,
            eval_time_s=time.perf_counter() - eval_start,
            forward_time_s=forward_time,
            backward_time_s=backward_time,
            relative_rate_step=rate_step,
            nll_change=nll_change,
        )
        if args.print_evals:
            _print_record(records[-1])

        switch_reason = (
            _handoff_reason(
                step_idx=step_idx,
                min_steps=min_steps,
                max_steps=max_steps,
                relative_rate_step=rate_step,
                nll_change=nll_change,
                args=args,
            )
            if threshold
            else (f"fixed_steps:{max_steps}" if step_idx >= max_steps else "not_yet")
        )
        previous_rates = rates_cpu
        previous_nll = nll
        if threshold and switch_reason:
            break

        with torch.no_grad():
            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            m_hat = m / (1.0 - beta1**step_idx)
            v_hat = v / (1.0 - beta2**step_idx)
            theta_fp32 = model.theta.detach().float()
            theta_fp32.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-args.bf16_start_lr)
            theta_fp32.clamp_(min=theta_min)
            model.theta.copy_(theta_fp32.to(dtype=model.theta.dtype))
        model.static.warm_E = None
        if not threshold and switch_reason and switch_reason != "not_yet":
            break

    torch.cuda.synchronize(device)
    seed_rates = tuple(float(x) for x in model.rates.detach().float().cpu())
    eval_times = [record.eval_time_s for record in records if record.eval_time_s is not None]
    forward_times = [record.forward_time_s for record in records if record.forward_time_s is not None]
    backward_times = [record.backward_time_s for record in records if record.backward_time_s is not None]
    phase_time_s = time.perf_counter() - phase_start
    return records, seed_rates, SimpleNamespace(
        success=True,
        message=f"resident_bf16_start={switch_reason}; seed_rates={seed_rates!r}",
        build_time_s=float(getattr(model, "_bench_build_time_s", float("nan"))),
        optimizer_time_s=phase_time_s,
        phase_summaries=[
            {
                "strategy": name,
                "phase": "bf16_start",
                "dtype": "bf16",
                "evaluations": len(records),
                "time_s": phase_time_s,
                "avg_eval_time_s": sum(eval_times) / len(eval_times) if eval_times else None,
                "forward_time_s": sum(forward_times) if forward_times else None,
                "backward_time_s": sum(backward_times) if backward_times else None,
                "optimizer": "adam_fp32_accum",
                "switch_reason": switch_reason,
            }
        ],
    )


def _run_resident_bf16_handoff(
    *,
    name: str,
    root: Path,
    init_rates: tuple[float, float, float],
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
    threshold: bool,
) -> tuple[list[EvalRecord], object]:
    try:
        bf16_records, seed_rates, bf16_result = _run_resident_bf16_start(
            name=name,
            root=root,
            init_rates=init_rates,
            target_rates=target_rates,
            target_nll=target_nll,
            args=args,
            threshold=threshold,
        )
    except Exception as exc:
        return [], SimpleNamespace(
            success=False,
            message=f"{name} failed during resident bf16 phase: {type(exc).__name__}: {exc}",
            build_time_s=0.0,
            optimizer_time_s=0.0,
            phase_summaries=[],
        )
    if not _result_success(bf16_result) or seed_rates is None:
        return bf16_records, bf16_result

    polish_args = argparse.Namespace(**vars(args))
    polish_args.torch_lbfgs_max_iter = args.fp32_polish_steps
    polish_records, polish_result = _run_torch_lbfgs(
        name=name,
        root=root,
        dtype=torch.float32,
        init_rates=seed_rates,
        target_rates=target_rates,
        target_nll=target_nll,
        args=polish_args,
        print_evals=args.print_evals,
        phase="fp32_polish",
    )
    if polish_records:
        offset_eval = len(bf16_records)
        offset_elapsed = bf16_records[-1].elapsed_s if bf16_records else 0.0
        first_polish_elapsed = polish_records[0].elapsed_s
        for record in polish_records:
            record.eval_idx += offset_eval
            record.elapsed_s = offset_elapsed + max(0.0, record.elapsed_s - first_polish_elapsed)
        bf16_records.extend(polish_records)

    phase_summaries = []
    phase_summaries.extend(_result_phase_summaries(bf16_result))
    phase_summaries.extend(_result_phase_summaries(polish_result))
    build_time_s = (_result_metric(bf16_result, "build_time_s") or 0.0) + (
        _result_metric(polish_result, "build_time_s") or 0.0
    )
    optimizer_time_s = (_result_metric(bf16_result, "optimizer_time_s") or 0.0) + (
        _result_metric(polish_result, "optimizer_time_s") or 0.0
    )
    return bf16_records, SimpleNamespace(
        success=_result_success(polish_result),
        message=f"{_result_message(bf16_result)}; fp32_polish={_result_message(polish_result)}",
        build_time_s=build_time_s,
        optimizer_time_s=optimizer_time_s,
        phase_summaries=phase_summaries,
    )


def _run_bad_floor_init_guard(args: argparse.Namespace) -> tuple[list[EvalRecord], object]:
    init_rates = (args.min_rate, args.min_rate, args.min_rate)
    try:
        _validate_interior_init_rates(
            init_rates,
            min_rate=args.min_rate,
            strategy="bad-floor-init-guard",
            allow_floor_init=False,
        )
    except ValueError as exc:
        print(
            "guard",
            "strategy", "bad-floor-init-guard",
            "status", "rejected",
            "message", str(exc).replace("\n", " "),
            flush=True,
        )
        return [], SimpleNamespace(success=True, message=str(exc))
    return [], SimpleNamespace(success=False, message="floor initialization was not rejected")


def _run_adam_then_scipy(
    *,
    root: Path,
    dtype: torch.dtype,
    init_rates: tuple[float, float, float],
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> tuple[list[EvalRecord], object]:
    name = f"adam{args.adam_steps}-scipy-{_dtype_name(dtype)}"
    _validate_interior_init_rates(
        init_rates,
        min_rate=args.min_rate,
        strategy=name,
        allow_floor_init=args.allow_floor_init,
    )
    model = _make_model(root, dtype=dtype, init_rates=init_rates, args=args)
    opt = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    records: list[EvalRecord] = []
    start = time.perf_counter()

    for idx in range(1, args.adam_steps + 1):
        nll, grad, grad_inf, metrics = _evaluate_with_grad(
            model,
            target_rates=target_rates,
            target_nll=target_nll,
            min_rate=args.min_rate,
        )
        opt.zero_grad(set_to_none=True)
        model.theta.grad = grad
        opt.step()
        model.clamp_theta_(min_rate=args.min_rate)
        _record(records, name, idx, start, nll, grad_inf, metrics, target_nll)
        if args.print_evals:
            _print_record(records[-1])

    init_after_adam = tuple(float(x) for x in model.rates.detach().cpu().to(dtype=torch.float64))
    scipy_records, scipy_result = _run_scipy_lbfgsb(
        name=name,
        root=root,
        dtype=dtype,
        init_rates=init_after_adam,
        target_rates=target_rates,
        target_nll=target_nll,
        args=args,
    )
    # Rebase elapsed/eval indices so the combined history is readable.
    if scipy_records:
        offset_eval = len(records)
        offset_elapsed = records[-1].elapsed_s if records else 0.0
        first_scipy_elapsed = scipy_records[0].elapsed_s
        for r in scipy_records:
            r.eval_idx += offset_eval
            r.elapsed_s = offset_elapsed + max(0.0, r.elapsed_s - first_scipy_elapsed)
        records.extend(scipy_records)
    return records, scipy_result


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.float64:
        return "fp64"
    if dtype == torch.bfloat16:
        return "bf16"
    return str(dtype).replace("torch.", "")


def _summarize(
    strategy: str,
    records: list[EvalRecord],
    result: object,
    *,
    args: argparse.Namespace,
) -> None:
    peak_alloc_mb, peak_reserved_mb = _cuda_peak_memory_mb()
    build_time_s = _result_metric(result, "build_time_s")
    optimizer_time_s = _result_metric(result, "optimizer_time_s")
    _print_phase_summaries(strategy, result)
    if not records:
        print(
            "summary",
            "strategy", strategy,
            "evals", 0,
            "success", int(_result_success(result)),
            "build_time_s", f"{build_time_s:.6f}" if build_time_s is not None else "n/a",
            "avg_eval_time_s", "n/a",
            "hit_eval", "n/a",
            "hit_time_s", "n/a",
            "best_nll", "n/a",
            "best_gap", "n/a",
            "best_rate_rel", "n/a",
            "last_nll", "n/a",
            "final_nll", "n/a",
            "last_gap", "n/a",
            "last_rate_rel", "n/a",
            "last_D", "n/a",
            "last_L", "n/a",
            "last_T", "n/a",
            "final_D", "n/a",
            "final_L", "n/a",
            "final_T", "n/a",
            "last_grad_inf", "n/a",
            "optimizer_time_s", f"{optimizer_time_s:.6f}" if optimizer_time_s is not None else "0.000000",
            "peak_alloc_mb", f"{peak_alloc_mb:.2f}",
            "peak_reserved_mb", f"{peak_reserved_mb:.2f}",
            "message", _result_message(result).replace("\n", " "),
            flush=True,
        )
        return
    best = min(records, key=lambda r: r.nll)
    last = records[-1]
    hit = _first_hit(records, nll_tol=args.nll_tol, rate_rtol=args.rate_rtol)
    status = _result_message(result)
    success = int(_result_success(result))
    if optimizer_time_s is None:
        optimizer_time_s = last.elapsed_s
    avg_eval_time_s = optimizer_time_s / len(records) if records else float("nan")
    print(
        "summary",
        "strategy", strategy,
        "evals", len(records),
        "success", success,
        "build_time_s", f"{build_time_s:.6f}" if build_time_s is not None else "n/a",
        "avg_eval_time_s", f"{avg_eval_time_s:.6f}",
        "hit_eval", hit.eval_idx if hit is not None else "none",
        "hit_time_s", f"{hit.elapsed_s:.6f}" if hit is not None else "none",
        "best_nll", f"{best.nll:.8f}",
        "best_gap", f"{best.target_nll_gap:.8e}",
        "best_rate_rel", f"{best.max_rate_rel_err:.8e}",
        "last_nll", f"{last.nll:.8f}",
        "final_nll", f"{last.nll:.8f}",
        "last_gap", f"{last.target_nll_gap:.8e}",
        "last_rate_rel", f"{last.max_rate_rel_err:.8e}",
        "last_D", f"{last.d_rate:.10e}",
        "last_L", f"{last.l_rate:.10e}",
        "last_T", f"{last.t_rate:.10e}",
        "final_D", f"{last.d_rate:.10e}",
        "final_L", f"{last.l_rate:.10e}",
        "final_T", f"{last.t_rate:.10e}",
        "last_grad_inf", f"{last.grad_inf:.8e}",
        "optimizer_time_s", f"{optimizer_time_s:.6f}",
        "peak_alloc_mb", f"{peak_alloc_mb:.2f}",
        "peak_reserved_mb", f"{peak_reserved_mb:.2f}",
        "message", str(status).replace("\n", " "),
        flush=True,
    )


def _eval_target(
    *,
    root: Path,
    dtype: torch.dtype,
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> list[EvalRecord]:
    rates = tuple(float(x) for x in target_rates.tolist())
    model = _make_model(root, dtype=dtype, init_rates=rates, args=args)
    records: list[EvalRecord] = []
    start = time.perf_counter()
    eval_start = time.perf_counter()
    nll, _grad, grad_inf, metrics = _evaluate_with_grad(
        model,
        target_rates=target_rates,
        target_nll=target_nll,
        min_rate=args.min_rate,
    )
    _record(
        records,
        f"eval-target-{_dtype_name(dtype)}",
        1,
        start,
        nll,
        grad_inf,
        metrics,
        target_nll,
        phase="eval_target",
        dtype=dtype,
        phase_eval_idx=1,
        eval_time_s=time.perf_counter() - eval_start,
    )
    if args.print_evals:
        _print_record(records[-1])
    return records


def _bf16_smoke(
    *,
    root: Path,
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> tuple[list[EvalRecord], object]:
    start = time.perf_counter()
    try:
        records = _eval_target(
            root=root,
            dtype=torch.bfloat16,
            target_rates=target_rates,
            target_nll=target_nll,
            args=args,
        )
        return records, SimpleNamespace(success=True, message="ok", optimizer_time_s=records[-1].elapsed_s if records else 0.0)
    except Exception as exc:
        print("strategy_error", "strategy", "bf16-smoke", "type", type(exc).__name__, "message", str(exc).replace("\n", " "))
        return [], SimpleNamespace(
            success=False,
            message=f"{type(exc).__name__}: {exc}",
            optimizer_time_s=time.perf_counter() - start,
        )


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    root = Path(args.dataset)
    target_rates = _reference_rates(root, allow_missing=args.allow_missing_target)
    target_nll = _reference_nll(root, allow_missing=args.allow_missing_target)
    init_rates = (args.init_rate, args.init_rate, args.init_rate)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    print(
        "target",
        "dataset", root,
        "families", len(_gene_paths(root, args.max_families)),
        "max_families", args.max_families if args.max_families is not None else "all",
        "target_nll_bits_from_per_family", f"{target_nll:.8f}",
        "target_nll_nats_from_per_family", f"{target_nll * math.log(2.0):.8f}",
        "target_D", f"{float(target_rates[0]):.10e}",
        "target_L", f"{float(target_rates[1]):.10e}",
        "target_T", f"{float(target_rates[2]):.10e}",
        "init_rate", f"{args.init_rate:.10e}",
        "min_rate", f"{args.min_rate:.10e}",
        "nll_tol", f"{args.nll_tol:.3e}",
        "rate_rtol", f"{args.rate_rtol:.3e}",
        "helper_mode", args.helper_mode,
        "allow_floor_init", int(args.allow_floor_init),
        "allow_missing_target", int(args.allow_missing_target),
        "bf16_start_lr", f"{args.bf16_start_lr:.3e}",
        "bf16_switch_rate_rtol", f"{args.bf16_switch_rate_rtol:.3e}",
        "bf16_switch_nll_abs_tol", f"{args.bf16_switch_nll_abs_tol:.3e}",
        "bf16_switch_criteria", args.bf16_switch_criteria,
        flush=True,
    )

    for strategy in strategies:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if strategy == "eval-target":
            records = _eval_target(
                root=root,
                dtype=torch.float32,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            )
            _summarize("eval-target-fp32", records, object(), args=args)
        elif strategy == "eval-target-fp64":
            records = _eval_target(
                root=root,
                dtype=torch.float64,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            )
            _summarize("eval-target-fp64", records, object(), args=args)
        elif strategy == "recommended-fp32":
            records, result = _run_recommended_lbfgs(
                root=root,
                init_rates=init_rates,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            )
            _summarize(strategy, records, result, args=args)
        elif strategy == "scipy-lbfgsb-fp32":
            records, result = _run_scipy_lbfgsb(
                name=strategy,
                root=root,
                dtype=torch.float32,
                init_rates=init_rates,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            )
            _summarize(strategy, records, result, args=args)
        elif strategy == "scipy-lbfgsb-fp64":
            records, result = _run_scipy_lbfgsb(
                name=strategy,
                root=root,
                dtype=torch.float64,
                init_rates=init_rates,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            )
            _summarize(strategy, records, result, args=args)
        elif strategy == "scipy-lbfgsb-fp64-polish":
            try:
                records, result = _run_scipy_lbfgsb_fp64_polish(
                    root=root,
                    init_rates=init_rates,
                    target_rates=target_rates,
                    target_nll=target_nll,
                    args=args,
                )
            except Exception as exc:
                print(
                    "strategy_error",
                    "strategy", strategy,
                    "type", type(exc).__name__,
                    "message", str(exc).replace("\n", " "),
                    flush=True,
                )
                records = []
                result = SimpleNamespace(success=False, message=f"{type(exc).__name__}: {exc}")
            _summarize(strategy, records, result, args=args)
        elif strategy == "torch-lbfgs-fp32":
            records, result = _run_torch_lbfgs(
                root=root,
                dtype=torch.float32,
                init_rates=init_rates,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            )
            _summarize(strategy, records, result, args=args)
        elif strategy == "adam3-scipy-fp32":
            records, result = _run_adam_then_scipy(
                root=root,
                dtype=torch.float32,
                init_rates=init_rates,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            )
            _summarize(strategy, records, result, args=args)
        elif strategy == "bf16-start-fp32-polish":
            records, result = _run_bf16_start_fp32_polish(
                root=root,
                init_rates=init_rates,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            )
            _summarize(strategy, records, result, args=args)
        elif strategy in ("bf16-resident-fixed-fp32-polish", "resident-bf16-fixed-fp32-polish"):
            records, result = _run_resident_bf16_handoff(
                name="bf16-resident-fixed-fp32-polish",
                root=root,
                init_rates=init_rates,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
                threshold=False,
            )
            _summarize("bf16-resident-fixed-fp32-polish", records, result, args=args)
        elif strategy in (
            "bf16-resident-threshold-fp32-polish",
            "resident-bf16-threshold-fp32-polish",
            "bf16-threshold-fp32-polish",
        ):
            records, result = _run_resident_bf16_handoff(
                name="bf16-resident-threshold-fp32-polish",
                root=root,
                init_rates=init_rates,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
                threshold=True,
            )
            _summarize("bf16-resident-threshold-fp32-polish", records, result, args=args)
        elif strategy == "bf16-smoke":
            records, status = _bf16_smoke(
                root=root,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            )
            _summarize(strategy, records, status, args=args)
        elif strategy == "bad-floor-init-guard":
            records, result = _run_bad_floor_init_guard(args)
            _summarize(strategy, records, result, args=args)
        else:
            raise ValueError(f"unknown strategy: {strategy}")


if __name__ == "__main__":
    main()
