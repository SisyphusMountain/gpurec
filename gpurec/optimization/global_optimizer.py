"""Production helper for global uniform DTL-rate optimization."""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from gpurec.api.model import GeneReconModel


_DEFAULT_INTERIOR_INIT_RATES = (0.05, 0.05, 0.05)
GlobalLBFGSRecord = dict[str, Any]
GlobalLBFGSResult = dict[str, Any]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_rate_tensor(
    rates: Sequence[float] | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    rate_t = torch.as_tensor(rates, device=device, dtype=dtype)
    if tuple(rate_t.shape) != (3,):
        raise ValueError(f"{name} must contain exactly three rates (D, L, T)")
    if not torch.isfinite(rate_t).all():
        raise ValueError(f"{name} must be finite")
    if (rate_t <= 0).any():
        raise ValueError(f"{name} must be strictly positive")
    return rate_t


def _max_relative_rate_step(
    rates: torch.Tensor,
    previous_rates: torch.Tensor | None,
) -> float | None:
    if previous_rates is None:
        return None
    denom = torch.clamp(previous_rates.abs(), min=1e-30)
    return float(torch.max(torch.abs(rates - previous_rates) / denom).item())


def _copy_rates_to_model(model: "GeneReconModel", rates: torch.Tensor) -> None:
    with torch.no_grad():
        model.theta.copy_(torch.log2(rates).to(device=model.theta.device, dtype=model.theta.dtype))
    model.static.warm_E = None


def _cast_theta_data_(
    model: "GeneReconModel",
    dtype: torch.dtype,
) -> None:
    if model.theta.dtype == dtype:
        return
    with torch.no_grad():
        model.theta.data = model.theta.data.to(dtype=dtype)
        if model.theta.grad is not None:
            model.theta.grad = model.theta.grad.to(dtype=dtype)


def _cast_model_dtype_(
    model: "GeneReconModel",
    dtype: torch.dtype,
) -> None:
    """Cast resident model state and clear dtype-sensitive warm starts."""
    model.to(dtype=dtype)
    model.static.warm_E = None


def _make_temporary_static_dtype(model: "GeneReconModel", dtype: torch.dtype):
    """Return a cast static state without mutating the model's current static."""
    from gpurec.api.autograd import _apply_to_static

    return _apply_to_static(
        model.static,
        lambda tensor: tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor,
    )


def _validate_global_uniform_model(model: "GeneReconModel") -> None:
    mode = getattr(model, "mode", None)
    if mode != "global":
        raise ValueError(f"optimize_global_rates_lbfgs requires mode='global', got {mode!r}")

    static = getattr(model, "static", None)
    pibar_mode = getattr(static, "pibar_mode", None)
    if pibar_mode != "uniform":
        raise ValueError(
            "optimize_global_rates_lbfgs requires a GeneReconModel built with "
            f"pibar_mode='uniform', got {pibar_mode!r}"
        )

    theta = getattr(model, "theta", None)
    if not torch.is_tensor(theta) or tuple(theta.shape) != (3,):
        shape = None if theta is None else tuple(theta.shape)
        raise ValueError(
            "optimize_global_rates_lbfgs requires a global three-parameter "
            f"theta tensor, got shape {shape}"
        )


def _prepare_initial_theta(
    model: "GeneReconModel",
    *,
    init_rates: Sequence[float] | torch.Tensor | None,
    interior_init_rates: Sequence[float] | torch.Tensor,
    min_rate: float,
    override_floor_init: bool,
) -> str:
    device = model.theta.device
    dtype = model.theta.dtype
    min_theta = math.log2(min_rate)
    floor_tol = max(1e-12, 16.0 * torch.finfo(dtype).eps * max(1.0, abs(min_theta)))

    interior_rates = _as_rate_tensor(
        interior_init_rates,
        device=device,
        dtype=dtype,
        name="interior_init_rates",
    )
    if (interior_rates <= min_rate).any():
        raise ValueError("interior_init_rates must be strictly above min_rate")

    if init_rates is not None:
        requested_rates = _as_rate_tensor(
            init_rates,
            device=device,
            dtype=dtype,
            name="init_rates",
        )
        if (requested_rates <= min_rate).any():
            if not override_floor_init:
                raise ValueError(
                    "init_rates contains a value at or below min_rate; use an "
                    "interior initialization or enable override_floor_init"
                )
            _copy_rates_to_model(model, interior_rates)
            return "overrode_requested_floor_init"
        _copy_rates_to_model(model, requested_rates)
        return "used_requested_init_rates"

    theta = model.theta.detach()
    if not torch.isfinite(theta).all():
        raise ValueError("model.theta contains non-finite values")

    if (theta <= min_theta + floor_tol).any():
        if not override_floor_init:
            raise ValueError(
                "model.theta is initialized at the lower rate floor; use an "
                "interior initialization or enable override_floor_init"
            )
        _copy_rates_to_model(model, interior_rates)
        return "overrode_model_floor_init"

    with torch.no_grad():
        model.theta.clamp_(min=min_theta)
    return "used_model_theta"


def _run_lbfgs_phase(
    model: "GeneReconModel",
    *,
    phase: str,
    min_rate: float,
    steps: int,
    lr: float,
    max_eval: int | None,
    history_size: int,
    tolerance_grad: float,
    tolerance_change: float,
    line_search_fn: str | None,
    history: list[dict[str, Any]],
    total_start: float,
    verbose: bool,
) -> dict[str, Any]:
    if steps < 1:
        return {"phase": phase, "time_s": 0.0, "evaluations": 0}

    opt_kwargs: dict[str, Any] = {
        "lr": lr,
        "max_iter": int(steps),
        "history_size": int(history_size),
        "tolerance_grad": float(tolerance_grad),
        "tolerance_change": float(tolerance_change),
        "line_search_fn": line_search_fn,
    }
    if max_eval is not None:
        opt_kwargs["max_eval"] = int(max_eval)

    opt = torch.optim.LBFGS(model.parameters(), **opt_kwargs)
    device = model.theta.device
    theta_min = math.log2(min_rate)
    phase_start = time.perf_counter()
    eval_count = 0
    previous_rates: torch.Tensor | None = None
    previous_nll: float | None = None

    def closure() -> torch.Tensor:
        nonlocal eval_count, previous_rates, previous_nll
        eval_start = time.perf_counter()
        with torch.no_grad():
            model.theta.clamp_(min=theta_min)

        opt.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        loss = model()
        _synchronize(device)
        forward_time = time.perf_counter() - forward_start
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(
                f"Non-finite NLL in optimize_global_rates_lbfgs phase={phase}"
            )
        backward_start = time.perf_counter()
        loss.backward()
        _synchronize(device)
        backward_time = time.perf_counter() - backward_start

        grad = model.theta.grad.detach()
        if not torch.isfinite(grad).all():
            raise FloatingPointError(
                f"Non-finite theta gradient in optimize_global_rates_lbfgs phase={phase}"
            )

        nll = float(loss.detach().cpu())
        theta_cpu = model.theta.detach().cpu().clone()
        rates_cpu = torch.exp2(theta_cpu)
        grad_cpu = grad.detach().cpu().clone()
        grad_inf = float(grad_cpu.abs().max().item())
        rate_step = _max_relative_rate_step(rates_cpu, previous_rates)
        nll_change = None if previous_nll is None else nll - previous_nll
        eval_time = time.perf_counter() - eval_start
        eval_count += 1

        record = {
            "phase": phase,
            "eval": len(history) + 1,
            "phase_eval": eval_count,
            "elapsed_s": time.perf_counter() - total_start,
            "eval_time_s": eval_time,
            "theta": theta_cpu,
            "rates": rates_cpu,
            "nll": nll,
            "negative_log_likelihood": nll,
            "log_likelihood": -nll,
            "grad_infinity_norm": grad_inf,
            "gradient": grad_cpu,
            "relative_rate_step": rate_step,
            "nll_change": nll_change,
            "forward_time_s": forward_time,
            "backward_time_s": backward_time,
            "dtype": str(model.theta.dtype).replace("torch.", ""),
            "theta_dtype": model.theta.dtype,
            "static_dtype": model.static.dtype,
            "loss_dtype": loss.dtype,
            "grad_dtype": grad.dtype,
        }
        history.append(record)

        if verbose:
            step_s = "n/a" if rate_step is None else f"{rate_step:.3e}"
            delta_s = "n/a" if nll_change is None else f"{nll_change:.3e}"
            rates_s = ", ".join(f"{float(x):.6e}" for x in rates_cpu)
            print(
                f"  {phase} eval {eval_count:3d}  NLL={nll:.6f}  "
                f"|g|={grad_inf:.3e}  dNLL={delta_s}  rel_rate_step={step_s}  "
                f"rates=({rates_s})  t={eval_time:.2f}s",
                flush=True,
            )

        previous_rates = rates_cpu
        previous_nll = nll
        return loss

    result_loss = opt.step(closure)
    model.clamp_theta_(min_rate=min_rate)
    _synchronize(device)

    result_value = None
    if torch.is_tensor(result_loss):
        result_value = float(result_loss.detach().cpu())
    return {
        "phase": phase,
        "time_s": time.perf_counter() - phase_start,
        "evaluations": eval_count,
        "optimizer_return": result_value,
    }


def _run_bf16_start_phase(
    model: "GeneReconModel",
    *,
    min_rate: float,
    steps: int,
    lr: float,
    switch_rate_rtol: float | None,
    switch_nll_abs_tol: float | None,
    switch_min_steps: int,
    switch_max_steps: int | None,
    switch_criteria: str,
    history: list[dict[str, Any]],
    total_start: float,
    verbose: bool,
) -> dict[str, Any]:
    if steps < 1:
        return {"phase": "bf16_start", "time_s": 0.0, "evaluations": 0}

    if model.theta.device.type != "cuda":
        raise ValueError("bf16_start_steps requires a CUDA resident model")
    if not torch.cuda.is_bf16_supported(model.theta.device):
        raise ValueError("bf16_start_steps requires CUDA bf16 support")
    if switch_min_steps < 1:
        raise ValueError("bf16_switch_min_steps must be >= 1")
    if switch_max_steps is not None and switch_max_steps < switch_min_steps:
        raise ValueError("bf16_switch_max_steps must be >= bf16_switch_min_steps")
    if switch_criteria not in ("any", "all"):
        raise ValueError("bf16_switch_criteria must be 'any' or 'all'")

    device = model.theta.device
    theta_min = math.log2(min_rate)
    max_steps = int(switch_max_steps) if switch_max_steps is not None else int(steps)
    phase_start = time.perf_counter()
    previous_rates: torch.Tensor | None = None
    previous_nll: float | None = None
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = torch.zeros_like(model.theta, dtype=torch.float32, device=device)
    v = torch.zeros_like(model.theta, dtype=torch.float32, device=device)
    stop_reason = f"fixed_steps:{max_steps}"
    updates = 0

    def _threshold_enabled(value: float | None) -> bool:
        return value is not None and math.isfinite(float(value)) and float(value) > 0.0

    def _switch_reason(step_idx: int, rate_step: float | None, nll_change: float | None) -> str | None:
        if step_idx < switch_min_steps:
            return None
        checks: list[tuple[str, bool]] = []
        if _threshold_enabled(switch_rate_rtol):
            checks.append(
                (
                    f"relative_rate_step<={float(switch_rate_rtol):.3e}",
                    rate_step is not None and rate_step <= float(switch_rate_rtol),
                )
            )
        if _threshold_enabled(switch_nll_abs_tol):
            checks.append(
                (
                    f"abs_nll_change<={float(switch_nll_abs_tol):.3e}",
                    nll_change is not None and abs(nll_change) <= float(switch_nll_abs_tol),
                )
            )
        if not checks:
            return None
        if switch_criteria == "all":
            return (
                "all:" + "+".join(name for name, _ok in checks)
                if all(ok for _name, ok in checks)
                else None
            )
        fired = [name for name, ok in checks if ok]
        return "any:" + "+".join(fired) if fired else None

    for step_idx in range(1, max_steps + 1):
        eval_start = time.perf_counter()
        with torch.no_grad():
            model.theta.clamp_(min=theta_min)

        model.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        loss = model()
        _synchronize(device)
        forward_time = time.perf_counter() - forward_start
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(
                "Non-finite NLL in optimize_global_rates_lbfgs phase=bf16_start"
            )

        backward_start = time.perf_counter()
        loss.backward()
        _synchronize(device)
        backward_time = time.perf_counter() - backward_start

        grad = model.theta.grad.detach().float()
        if not torch.isfinite(grad).all():
            raise FloatingPointError(
                "Non-finite theta gradient in optimize_global_rates_lbfgs "
                "phase=bf16_start"
            )

        nll = float(loss.detach().float().cpu())
        theta_cpu = model.theta.detach().cpu().clone()
        rates_cpu = torch.exp2(theta_cpu.float())
        grad_cpu = grad.detach().cpu().clone()
        grad_inf = float(grad_cpu.abs().max().item())
        rate_step = _max_relative_rate_step(rates_cpu, previous_rates)
        nll_change = None if previous_nll is None else nll - previous_nll
        eval_time = time.perf_counter() - eval_start

        record = {
            "phase": "bf16_start",
            "eval": len(history) + 1,
            "phase_eval": step_idx,
            "elapsed_s": time.perf_counter() - total_start,
            "eval_time_s": eval_time,
            "theta": theta_cpu,
            "rates": rates_cpu,
            "nll": nll,
            "negative_log_likelihood": nll,
            "log_likelihood": -nll,
            "grad_infinity_norm": grad_inf,
            "gradient": grad_cpu,
            "relative_rate_step": rate_step,
            "nll_change": nll_change,
            "forward_time_s": forward_time,
            "backward_time_s": backward_time,
            "dtype": "bfloat16",
            "theta_dtype": model.theta.dtype,
            "static_dtype": model.static.dtype,
            "loss_dtype": loss.dtype,
            "grad_dtype": model.theta.grad.dtype if model.theta.grad is not None else None,
        }
        history.append(record)

        if verbose:
            step_s = "n/a" if rate_step is None else f"{rate_step:.3e}"
            delta_s = "n/a" if nll_change is None else f"{nll_change:.3e}"
            rates_s = ", ".join(f"{float(x):.6e}" for x in rates_cpu)
            print(
                f"  bf16_start eval {step_idx:3d}  NLL={nll:.6f}  "
                f"|g|={grad_inf:.3e}  dNLL={delta_s}  rel_rate_step={step_s}  "
                f"rates=({rates_s})  t={eval_time:.2f}s",
                flush=True,
            )

        reason = _switch_reason(step_idx, rate_step, nll_change)
        if reason is not None:
            stop_reason = reason
            previous_rates = rates_cpu
            previous_nll = nll
            break

        with torch.no_grad():
            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            m_hat = m / (1.0 - beta1**step_idx)
            v_hat = v / (1.0 - beta2**step_idx)
            theta_fp32 = model.theta.detach().float()
            theta_fp32.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
            theta_fp32.clamp_(min=theta_min)
            model.theta.copy_(theta_fp32.to(dtype=model.theta.dtype))
        model.static.warm_E = None
        updates += 1

        previous_rates = rates_cpu
        previous_nll = nll

    _synchronize(device)
    return {
        "phase": "bf16_start",
        "time_s": time.perf_counter() - phase_start,
        "evaluations": len([r for r in history if r.get("phase") == "bf16_start"]),
        "updates": updates,
        "optimizer": "adam_fp32_accum",
        "lr": lr,
        "dtype": "bf16",
        "switch_reason": stop_reason,
        "switch_rate_rtol": switch_rate_rtol,
        "switch_nll_abs_tol": switch_nll_abs_tol,
    }


@torch.no_grad()
def _final_nll(model: "GeneReconModel") -> tuple[float, float]:
    start = time.perf_counter()
    nll = float(model().detach().cpu())
    _synchronize(model.theta.device)
    return nll, time.perf_counter() - start


def optimize_global_rates_lbfgs(
    model: "GeneReconModel",
    *,
    init_rates: Sequence[float] | torch.Tensor | None = None,
    min_rate: float = 1e-10,
    interior_init_rates: Sequence[float] | torch.Tensor = _DEFAULT_INTERIOR_INIT_RATES,
    override_floor_init: bool = True,
    steps: int = 12,
    lr: float = 1.0,
    max_eval: int | None = None,
    history_size: int = 10,
    tolerance_grad: float = 1e-3,
    tolerance_change: float = 1e-7,
    line_search_fn: str | None = "strong_wolfe",
    dtype: torch.dtype | None = torch.float32,
    bf16_start_steps: int = 0,
    bf16_start_lr: float = 0.05,
    bf16_switch_rate_rtol: float | None = None,
    bf16_switch_nll_abs_tol: float | None = None,
    bf16_switch_min_steps: int = 1,
    bf16_switch_max_steps: int | None = None,
    bf16_switch_criteria: str = "any",
    fp64_polish: bool = False,
    fp64_polish_steps: int = 4,
    fp64_polish_max_eval: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Optimize global DTL rates for a resident uniform ``GeneReconModel``.

    The helper intentionally keeps a small surface around the public model API:
    callers build ``GeneReconModel.from_trees(..., mode="global",
    pibar_mode="uniform")`` once, then this function only updates
    ``model.theta``. The default path is projected fp32 PyTorch L-BFGS with a
    strong-Wolfe line search. If ``bf16_start_steps`` is positive, the helper
    first evaluates and updates the global parameters with a temporary bf16
    static state and bf16 ``theta``, then restores the original static state
    for fp32 LBFGS. The fp32 static tensors are not rounded in-place by the
    bf16 phase. The bf16 Adam update keeps its three-parameter moment vectors
    in fp32 but stores every evaluated ``theta`` in bf16.

    Returns a dict with ``theta``, ``rates``, ``nll``,
    ``negative_log_likelihood``, ``log_likelihood``, ``history`` and
    ``timing``.
    """
    _validate_global_uniform_model(model)
    if min_rate <= 0:
        raise ValueError("min_rate must be strictly positive")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if fp64_polish_steps < 1:
        raise ValueError("fp64_polish_steps must be >= 1")
    if bf16_start_steps < 0:
        raise ValueError("bf16_start_steps must be >= 0")
    if bf16_start_lr <= 0:
        raise ValueError("bf16_start_lr must be strictly positive")
    if bf16_switch_min_steps < 1:
        raise ValueError("bf16_switch_min_steps must be >= 1")
    if bf16_switch_max_steps is not None and bf16_switch_max_steps < bf16_switch_min_steps:
        raise ValueError("bf16_switch_max_steps must be >= bf16_switch_min_steps")
    if bf16_switch_criteria not in ("any", "all"):
        raise ValueError("bf16_switch_criteria must be 'any' or 'all'")
    if bf16_start_steps > 0 and dtype == torch.bfloat16:
        raise ValueError(
            "bf16_start_steps hands off to fp32 LBFGS; use dtype=torch.float32 "
            "or leave dtype at the default"
        )

    original_static = None
    if bf16_start_steps > 0:
        if model.theta.device.type != "cuda":
            raise ValueError("bf16_start_steps requires a CUDA resident model")
        if not torch.cuda.is_bf16_supported(model.theta.device):
            raise ValueError("bf16_start_steps requires CUDA bf16 support")
        original_static = model.static
        model._static = _make_temporary_static_dtype(model, torch.bfloat16)
        _cast_theta_data_(model, torch.bfloat16)
        model.static.warm_E = None
    elif dtype is not None and model.theta.dtype != dtype:
        _cast_model_dtype_(model, dtype)

    initialization = _prepare_initial_theta(
        model,
        init_rates=init_rates,
        interior_init_rates=interior_init_rates,
        min_rate=min_rate,
        override_floor_init=override_floor_init,
    )

    total_start = time.perf_counter()
    history: list[dict[str, Any]] = []
    phase_timings = []

    if bf16_start_steps > 0:
        phase_timings.append(
            _run_bf16_start_phase(
                model,
                min_rate=min_rate,
                steps=bf16_start_steps,
                lr=bf16_start_lr,
                switch_rate_rtol=bf16_switch_rate_rtol,
                switch_nll_abs_tol=bf16_switch_nll_abs_tol,
                switch_min_steps=bf16_switch_min_steps,
                switch_max_steps=bf16_switch_max_steps,
                switch_criteria=bf16_switch_criteria,
                history=history,
                total_start=total_start,
                verbose=verbose,
            )
        )
        if original_static is None:
            raise RuntimeError("internal error: missing original static after bf16 start")
        model._static = original_static
        model.static.warm_E = None
        lbfgs_dtype = torch.float32 if dtype is None else dtype
        if model.static.dtype != lbfgs_dtype:
            _cast_model_dtype_(model, lbfgs_dtype)
        else:
            _cast_theta_data_(model, lbfgs_dtype)

    phase_timings.append(
        _run_lbfgs_phase(
            model,
            phase="fp32_lbfgs" if model.theta.dtype == torch.float32 else "lbfgs",
            min_rate=min_rate,
            steps=steps,
            lr=lr,
            max_eval=max_eval,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
            history=history,
            total_start=total_start,
            verbose=verbose,
        )
    )

    if fp64_polish and model.theta.dtype != torch.float64:
        model.to(dtype=torch.float64)
        model.static.warm_E = None
        phase_timings.append(
            _run_lbfgs_phase(
                model,
                phase="fp64_polish",
                min_rate=min_rate,
                steps=fp64_polish_steps,
                lr=lr,
                max_eval=fp64_polish_max_eval,
                history_size=history_size,
                tolerance_grad=tolerance_grad,
                tolerance_change=tolerance_change,
                line_search_fn=line_search_fn,
                history=history,
                total_start=total_start,
                verbose=verbose,
            )
        )

    model.clamp_theta_(min_rate=min_rate)
    final_nll, final_eval_s = _final_nll(model)
    total_s = time.perf_counter() - total_start
    theta = model.theta.detach().cpu().clone()
    rates = torch.exp2(theta)

    timing = {
        "total_s": total_s,
        "final_eval_s": final_eval_s,
        "evaluations": len(history),
        "phases": phase_timings,
    }

    return {
        "theta": theta,
        "rates": rates,
        "nll": final_nll,
        "negative_log_likelihood": final_nll,
        "log_likelihood": -final_nll,
        "history": history,
        "timing": timing,
        "initialization": initialization,
        "min_rate": min_rate,
    }
