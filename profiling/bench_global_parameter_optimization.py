#!/usr/bin/env python3
"""Benchmark global/uniform DTL-rate optimization strategies.

The script targets the 3-parameter global mode and uses the public
``GeneReconModel`` autograd bridge. ``recommended-fp32`` uses the production
``optimize_global_rates_lbfgs`` helper when available, and can fall back to the
direct experimental ``GeneReconModel`` LBFGS path with ``--helper-mode direct``.
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
        default="eval-target,recommended-fp32,scipy-lbfgsb-fp32,scipy-lbfgsb-fp64-polish,bad-floor-init-guard",
    )
    parser.add_argument("--init-rate", type=float, default=0.05)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--maxfun", type=int, default=60)
    parser.add_argument("--torch-lbfgs-max-iter", type=int, default=12)
    parser.add_argument("--adam-steps", type=int, default=3)
    parser.add_argument("--adam-lr", type=float, default=0.35)
    parser.add_argument("--fixed-iters-Pi", type=int, default=6)
    parser.add_argument("--max-wave-size", type=int, default=32768)
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


def _gene_paths(root: Path) -> list[str]:
    return [str(p) for p in sorted(root.glob("g_*.nwk"))]


def _reference_rates(root: Path) -> torch.Tensor:
    path = root / "output_global" / "model_parameters" / "model_parameters.txt"
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4:
            return torch.tensor([float(parts[1]), float(parts[2]), float(parts[3])])
    raise RuntimeError(f"could not parse rates from {path}")


def _reference_nll(root: Path) -> float:
    path = root / "output_global" / "per_fam_likelihoods.txt"
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
    return GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=_gene_paths(root),
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


def _rate_metrics(model: GeneReconModel, target_rates: torch.Tensor, target_nll: float) -> tuple[float, float, float, float, float]:
    rates = model.rates.detach().to(dtype=torch.float64, device="cpu")
    target = target_rates.to(dtype=torch.float64)
    rel = torch.max(torch.abs(rates - target) / torch.clamp(target.abs(), min=1e-30)).item()
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
) -> None:
    d_rate, l_rate, t_rate, max_rate_rel_err, _ = metrics
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
        )
    )


def _print_record(r: EvalRecord) -> None:
    print(
        "eval",
        "strategy", r.strategy,
        "idx", r.eval_idx,
        "elapsed_s", f"{r.elapsed_s:.6f}",
        "nll", f"{r.nll:.8f}",
        "target_gap", f"{r.target_nll_gap:.8e}",
        "grad_inf", f"{r.grad_inf:.8e}",
        "D", f"{r.d_rate:.10e}",
        "L", f"{r.l_rate:.10e}",
        "T", f"{r.t_rate:.10e}",
        "rate_rel", f"{r.max_rate_rel_err:.8e}",
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
        _record(records, name, eval_count, start, nll, grad_inf, metrics, target_nll)
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
) -> tuple[list[EvalRecord], object]:
    name = name or f"torch-lbfgs-{_dtype_name(dtype)}"
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
        nll, _grad, grad_inf, metrics = _evaluate_with_grad(
            model,
            target_rates=target_rates,
            target_nll=target_nll,
            min_rate=args.min_rate,
        )
        eval_count += 1
        _record(records, name, eval_count, start, nll, grad_inf, metrics, target_nll)
        if should_print_evals:
            _print_record(records[-1])
        return model.theta.grad.new_tensor(nll)

    result = opt.step(closure)
    model.clamp_theta_(min_rate=args.min_rate)
    return records, result


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


def _result_success(result: object) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success", True))
    return bool(getattr(result, "success", True))


def _result_message(result: object) -> str:
    if isinstance(result, dict):
        return str(result.get("message", "ok"))
    return str(getattr(result, "message", "ok"))


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
        rel = torch.max(torch.abs(rates[:3] - target) / torch.clamp(target.abs(), min=1e-30)).item()
        nll = float(item.get("negative_log_likelihood", item["nll"]))
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

    return records, SimpleNamespace(success=_result_success(raw_result), message=message)


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
    if not records:
        print(
            "summary",
            "strategy", strategy,
            "evals", 0,
            "success", int(_result_success(result)),
            "hit_eval", "n/a",
            "hit_time_s", "n/a",
            "best_nll", "n/a",
            "best_gap", "n/a",
            "best_rate_rel", "n/a",
            "last_nll", "n/a",
            "last_gap", "n/a",
            "last_rate_rel", "n/a",
            "last_grad_inf", "n/a",
            "total_time_s", "0.000000",
            "message", _result_message(result).replace("\n", " "),
            flush=True,
        )
        return
    best = min(records, key=lambda r: r.nll)
    last = records[-1]
    hit = _first_hit(records, nll_tol=args.nll_tol, rate_rtol=args.rate_rtol)
    status = _result_message(result)
    success = int(_result_success(result))
    print(
        "summary",
        "strategy", strategy,
        "evals", len(records),
        "success", success,
        "hit_eval", hit.eval_idx if hit is not None else "none",
        "hit_time_s", f"{hit.elapsed_s:.6f}" if hit is not None else "none",
        "best_nll", f"{best.nll:.8f}",
        "best_gap", f"{best.target_nll_gap:.8e}",
        "best_rate_rel", f"{best.max_rate_rel_err:.8e}",
        "last_nll", f"{last.nll:.8f}",
        "last_gap", f"{last.target_nll_gap:.8e}",
        "last_rate_rel", f"{last.max_rate_rel_err:.8e}",
        "last_grad_inf", f"{last.grad_inf:.8e}",
        "total_time_s", f"{last.elapsed_s:.6f}",
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
    nll, _grad, grad_inf, metrics = _evaluate_with_grad(
        model,
        target_rates=target_rates,
        target_nll=target_nll,
        min_rate=args.min_rate,
    )
    _record(records, f"eval-target-{_dtype_name(dtype)}", 1, start, nll, grad_inf, metrics, target_nll)
    if args.print_evals:
        _print_record(records[-1])
    return records


def _bf16_smoke(
    *,
    root: Path,
    target_rates: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
) -> tuple[list[EvalRecord], str]:
    try:
        return (
            _eval_target(
                root=root,
                dtype=torch.bfloat16,
                target_rates=target_rates,
                target_nll=target_nll,
                args=args,
            ),
            "ok",
        )
    except Exception as exc:
        print("strategy_error", "strategy", "bf16-smoke", "type", type(exc).__name__, "message", str(exc).replace("\n", " "))
        return [], f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    root = Path(args.dataset)
    target_rates = _reference_rates(root)
    target_nll = _reference_nll(root)
    init_rates = (args.init_rate, args.init_rate, args.init_rate)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    print(
        "target",
        "dataset", root,
        "families", len(_gene_paths(root)),
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
        flush=True,
    )

    for strategy in strategies:
        torch.cuda.empty_cache()
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
