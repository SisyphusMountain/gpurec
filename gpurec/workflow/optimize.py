from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch

from gpurec.api.model import GeneReconModel

from .checkpoint import (
    load_checkpoint,
    restore_model_theta,
    save_checkpoint,
    validate_checkpoint_model_compatibility,
)
from .config import RunConfig
from .diagnostics import (
    append_jsonl,
    parameter_stats,
    rates_and_survival_probability,
    solver_stats,
    tensor_stats,
    write_csv,
)
from .model_factory import build_alerax_workflow_model


@dataclass
class OptimizationResult:
    out_dir: Path
    status: str
    reason: str
    final_nll_bits: float
    final_grad_inf: float
    best_nll_bits: float | None
    best_step: int | None
    steps_completed: int
    sampling_checkpoint: Path | None = None


_MISSING = object()


def _is_finite_tensor(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and bool(torch.isfinite(tensor).all().item())


def _invalid_resume_field(path: Path, key: str) -> RuntimeError:
    return RuntimeError(f"checkpoint {path} has invalid {key}")


def _resume_int(
    path: Path,
    key: str,
    value: Any,
    *,
    default: int | object = _MISSING,
    allow_none: bool = False,
    nonnegative: bool = False,
) -> int | None:
    if value is _MISSING:
        if default is not _MISSING:
            return int(default)
        raise _invalid_resume_field(path, key)
    if value is None:
        if allow_none:
            return None
        raise _invalid_resume_field(path, key)
    if isinstance(value, bool):
        raise _invalid_resume_field(path, key)
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        raw = float(value)
        if not math.isfinite(raw) or not raw.is_integer():
            raise _invalid_resume_field(path, key)
        number = int(raw)
    else:
        raise _invalid_resume_field(path, key)
    if nonnegative and number < 0:
        raise _invalid_resume_field(path, key)
    return number


def _resume_float(
    path: Path,
    key: str,
    value: Any,
    *,
    allow_none: bool = False,
) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise _invalid_resume_field(path, key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _invalid_resume_field(path, key)
    number = float(value)
    if not math.isfinite(number):
        raise _invalid_resume_field(path, key)
    return number


def _family_names(model: GeneReconModel) -> list[str]:
    return model.family_names


def _parameter_labels(model: GeneReconModel, mode: str) -> list[str]:
    theta_rows = int(model.theta.detach().reshape(-1, 3).shape[0])
    if mode == "genewise":
        return _family_names(model)
    if mode == "specieswise":
        return model.species_names[:theta_rows]
    return ["global"]


def _write_rate_table(path: Path, model: GeneReconModel, mode: str) -> None:
    labels = _parameter_labels(model, mode)
    theta = model.theta.detach().reshape(-1, 3).to(device="cpu", dtype=torch.float64)
    rates, p_s = rates_and_survival_probability(theta)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("row\tname\tD\tT\tL\tpS\ttheta_D\ttheta_T\ttheta_L\n")
        for row, label in enumerate(labels):
            theta_row = 0 if theta.shape[0] == 1 else row
            handle.write(
                "\t".join(
                    str(value)
                    for value in (
                        row,
                        label,
                        float(rates[theta_row, 0]),
                        float(rates[theta_row, 2]),
                        float(rates[theta_row, 1]),
                        float(p_s[theta_row]),
                        float(theta[theta_row, 0]),
                        float(theta[theta_row, 2]),
                        float(theta[theta_row, 1]),
                    )
                )
                + "\n"
            )


@torch.no_grad()
def _per_family_nll(model: GeneReconModel) -> list[tuple[str, float]]:
    values = model.full_nll_per_family().detach().cpu().reshape(-1).tolist()
    return list(zip(_family_names(model), values))


def _write_per_family_likelihoods(path: Path, model: GeneReconModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("family\tnll_bits\tlog_likelihood_bits\n")
        for family, nll in _per_family_nll(model):
            handle.write(f"{family}\t{nll:.12g}\t{-nll:.12g}\n")


class OptimizationRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.history: list[dict[str, Any]] = []
        self.history_jsonl = config.out_dir / "history.jsonl"

    def build_model(self) -> GeneReconModel:
        config = self.config
        return build_alerax_workflow_model(config, prefetch_batches="all")

    def _make_optimizer(self, model: GeneReconModel, phase: str) -> torch.optim.Optimizer:
        config = self.config
        if phase == "adam":
            return torch.optim.Adam([model.theta], lr=config.lr)
        if phase == "adagrad":
            return torch.optim.Adagrad([model.theta], lr=config.lr, eps=1e-10)
        if phase == "lbfgs":
            return torch.optim.LBFGS(
                [model.theta],
                lr=config.lbfgs_lr,
                max_iter=config.lbfgs_max_iter,
                history_size=config.lbfgs_history_size,
                line_search_fn=(
                    None if config.lbfgs_line_search == "none" else config.lbfgs_line_search
                ),
            )
        raise ValueError(f"unknown optimizer phase {phase!r}")

    def _phase_for_step(self, step: int) -> str:
        if self.config.optimizer == "adam-lbfgs":
            return "adam" if step < self.config.adam_warmup_steps else "lbfgs"
        return self.config.optimizer

    def _evaluate_and_backward(self, model: GeneReconModel) -> tuple[torch.Tensor, dict[str, Any]]:
        model.theta.grad = None
        loss = model.full_loss()
        loss.backward()
        if model.theta.grad is None:
            raise RuntimeError("optimizer evaluation did not produce theta gradients")
        row: dict[str, Any] = {
            "likelihood/data_nll_bits": float(loss.detach().cpu()),
            "likelihood/log_likelihood_bits": float(-loss.detach().cpu()),
        }
        row.update(tensor_stats("grad", model.theta.grad))
        row.update(parameter_stats(model.theta))
        row.update(solver_stats(model))
        return loss, row

    def _record(self, row: dict[str, Any]) -> None:
        self.history.append(row)
        append_jsonl(self.history_jsonl, row)

    def _restore_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        state: Any,
        *,
        current_phase: str | None = None,
        checkpoint_phase: Any = None,
    ) -> dict[str, Any]:
        if state is None:
            return {"resume_optimizer_state": "missing"}
        if checkpoint_phase is not None and not isinstance(checkpoint_phase, str):
            return {
                "resume_optimizer_state": "discarded",
                "resume_optimizer_reason": "invalid_phase",
            }
        if (
            current_phase is not None
            and checkpoint_phase is not None
            and checkpoint_phase != current_phase
        ):
            return {
                "resume_optimizer_state": "discarded",
                "resume_optimizer_reason": "phase_mismatch",
                "resume_optimizer_checkpoint_phase": checkpoint_phase,
                "resume_optimizer_current_phase": current_phase,
            }
        try:
            optimizer.load_state_dict(state)
        except ValueError as exc:
            return {
                "resume_optimizer_state": "discarded",
                "resume_optimizer_error": str(exc),
            }
        return {"resume_optimizer_state": "restored"}

    def _save_status(
        self,
        path: Path,
        *,
        model: GeneReconModel,
        optimizer: torch.optim.Optimizer | None,
        step: int,
        status: dict[str, Any],
        row: dict[str, Any] | None,
        next_step: int | None = None,
        optimizer_phase: str | None = None,
    ) -> None:
        save_checkpoint(
            path,
            config=self.config,
            model=model,
            optimizer=optimizer,
            optimizer_phase=optimizer_phase,
            step=step,
            next_step=next_step,
            status=status,
            row=row,
        )

    def run(self) -> OptimizationResult:
        config = self.config
        config.out_dir.mkdir(parents=True, exist_ok=True)
        config.write_json(config.out_dir / "run_config.json")
        if self.history_jsonl.exists() and config.resume_from is None:
            self.history_jsonl.unlink()

        model = self.build_model()
        optimizer: torch.optim.Optimizer | None = None
        started = time.perf_counter()
        best_nll: float | None = None
        best_step: int | None = None
        previous_objective: float | None = None
        stable_loss_steps = 0
        start_step = 0
        status = {"status": "running", "reason": "running"}
        final_row: dict[str, Any] = {}
        resume_info: dict[str, Any] = {}
        resume_payload: dict[str, Any] | None = None
        best_checkpoint = config.out_dir / "checkpoints" / "best.pt"
        latest_checkpoint = config.out_dir / "checkpoints" / "latest.pt"
        sampling_checkpoint: Path | None = None

        try:
            if config.resume_from is not None:
                resume_payload = load_checkpoint(
                    config.resume_from,
                    map_location=config.device,
                )
                validate_checkpoint_model_compatibility(
                    path=config.resume_from,
                    config=config,
                    model=model,
                    payload=resume_payload,
                )
                restore_model_theta(model, resume_payload)
                start_step = int(
                    _resume_int(
                        config.resume_from,
                        "next_step",
                        resume_payload.get("next_step", _MISSING),
                        default=0,
                        nonnegative=True,
                    )
                )
                ckpt_status = resume_payload.get("status")
                if ckpt_status is None:
                    ckpt_status = {}
                elif not isinstance(ckpt_status, dict):
                    raise RuntimeError(
                        f"checkpoint {config.resume_from} has invalid status metadata"
                    )
                best_nll = _resume_float(
                    config.resume_from,
                    "status.best_nll_bits",
                    ckpt_status.get("best_nll_bits"),
                    allow_none=True,
                )
                best_step = _resume_int(
                    config.resume_from,
                    "status.best_step",
                    ckpt_status.get("best_step"),
                    allow_none=True,
                    nonnegative=True,
                )
                previous_objective = _resume_float(
                    config.resume_from,
                    "status.previous_objective",
                    ckpt_status.get("previous_objective"),
                    allow_none=True,
                )
                stable_loss_steps = int(
                    _resume_int(
                        config.resume_from,
                        "status.stable_loss_steps",
                        ckpt_status.get("stable_loss_steps", _MISSING),
                        default=0,
                        nonnegative=True,
                    )
                )

            current_phase = self._phase_for_step(start_step)
            optimizer = self._make_optimizer(model, current_phase)
            if config.resume_from is not None:
                resume_info = self._restore_optimizer_state(
                    optimizer,
                    (
                        None
                        if resume_payload is None
                        else resume_payload.get("optimizer_state")
                    ),
                    current_phase=current_phase,
                    checkpoint_phase=(
                        None
                        if resume_payload is None
                        else resume_payload.get("optimizer_phase")
                    ),
                )

            for step in range(start_step, config.steps):
                phase = self._phase_for_step(step)
                if optimizer is None or phase != current_phase:
                    current_phase = phase
                    optimizer = self._make_optimizer(model, phase)

                t0 = time.perf_counter()
                theta_before = model.theta.detach().clone()
                closure_evals = 0
                metrics: dict[str, Any] = {}

                def closure() -> torch.Tensor:
                    nonlocal closure_evals, metrics
                    with torch.no_grad():
                        model.clamp_theta_(config.min_rate, config.max_rate)
                    if optimizer is None:
                        raise RuntimeError("missing optimizer")
                    optimizer.zero_grad(set_to_none=True)
                    loss_i, metrics_i = self._evaluate_and_backward(model)
                    metrics = metrics_i
                    closure_evals += 1
                    return loss_i

                stop_after_row = False
                save_best_after_row = False
                if phase == "lbfgs":
                    try:
                        optimizer.step(closure)
                    except RuntimeError:
                        status = {"status": "failed", "reason": "lbfgs_runtime_error"}
                        break
                    with torch.no_grad():
                        model.clamp_theta_(config.min_rate, config.max_rate)
                    theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
                    model.clear()
                    # LBFGS may evaluate trial points during line search. Record
                    # one current-theta evaluation so checkpoints and diagnostics
                    # refer to the saved parameters.
                    model.theta.grad = None
                    _loss_current, metrics = self._evaluate_and_backward(model)
                    closure_evals += 1
                    model.clear()
                else:
                    loss = closure()
                    if not torch.isfinite(loss).item() or not _is_finite_tensor(model.theta.grad):
                        status = {
                            "status": "failed",
                            "reason": "nonfinite_objective_or_gradient",
                        }
                        break
                    theta_step = 0.0
                    skip_step_for_gradient = (
                        float(metrics.get("grad/inf", math.inf)) <= config.grad_inf_tol
                    )
                    if skip_step_for_gradient:
                        stop_after_row = True
                    else:
                        optimizer.step()
                        with torch.no_grad():
                            model.clamp_theta_(config.min_rate, config.max_rate)
                        theta_step = float(
                            (model.theta.detach() - theta_before).abs().amax().cpu()
                        )
                        model.clear()
                        # Ordinary optimizers evaluate gradients before the
                        # parameter update.  Record one current-theta
                        # evaluation so rows and checkpoints describe the
                        # weights that are actually being saved.
                        model.theta.grad = None
                        loss, metrics = self._evaluate_and_backward(model)
                        closure_evals += 1
                        if (
                            not torch.isfinite(loss).item()
                            or not _is_finite_tensor(model.theta.grad)
                        ):
                            status = {
                                "status": "failed",
                                "reason": "nonfinite_objective_or_gradient",
                            }
                            break
                    model.clear()

                objective = float(metrics["likelihood/data_nll_bits"])
                delta = None if previous_objective is None else previous_objective - objective
                if delta is not None and abs(delta) <= config.loss_change_tol:
                    stable_loss_steps += 1
                else:
                    stable_loss_steps = 0
                previous_objective = objective

                improved = (
                    best_nll is None
                    or objective < best_nll - config.best_likelihood_min_delta
                )
                if improved:
                    best_nll = objective
                    best_step = step
                    save_best_after_row = True

                if phase != "lbfgs":
                    row_grad = float(metrics.get("grad/inf", math.inf))
                    if row_grad <= config.grad_inf_tol:
                        status = {
                            "status": "converged",
                            "reason": "gradient_tolerance",
                        }
                        stop_after_row = True
                    if config.loss_patience and stable_loss_steps >= config.loss_patience:
                        status = {"status": "stalled", "reason": "loss_change_patience"}
                        stop_after_row = True
                    if (
                        config.best_likelihood_patience
                        and best_step is not None
                        and step - int(best_step) >= config.best_likelihood_patience
                    ):
                        status = {"status": "stalled", "reason": "best_likelihood_patience"}
                        stop_after_row = True

                row = {
                    "step": step,
                    "optimizer/phase": phase,
                    "closure_evals": closure_evals,
                    "theta_step_inf": theta_step,
                    "delta_likelihood_bits": delta,
                    "stable_loss_steps": stable_loss_steps,
                    "best_nll_bits": best_nll,
                    "best_step": best_step,
                    **resume_info,
                    "step_s": time.perf_counter() - t0,
                    **metrics,
                }
                final_row = row
                self._record(row)

                checkpoint_status = {
                    "status": "running",
                    "reason": "running",
                    **resume_info,
                    "best_nll_bits": best_nll,
                    "best_step": best_step,
                    "previous_objective": previous_objective,
                    "stable_loss_steps": stable_loss_steps,
                }
                if save_best_after_row:
                    self._save_status(
                        best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        status=checkpoint_status,
                        row=row,
                        optimizer_phase=phase,
                    )
                    sampling_checkpoint = best_checkpoint
                if config.checkpoint_every and step % config.checkpoint_every == 0:
                    self._save_status(
                        latest_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        status=checkpoint_status,
                        row=row,
                        optimizer_phase=phase,
                    )

                if step % config.log_every == 0:
                    print(
                        f"step={step} phase={phase} "
                        f"nll_bits={objective:.6f} "
                        f"grad_inf={row.get('grad/inf', float('nan')):.6g} "
                        f"delta={float('nan') if delta is None else delta:.6g} "
                        f"best={float('nan') if best_nll is None else best_nll:.6f} "
                        f"step_s={row['step_s']:.3f}",
                        flush=True,
                    )

                if stop_after_row:
                    break
                if phase == "lbfgs" and row.get("grad/inf", math.inf) <= config.grad_inf_tol:
                    status = {"status": "converged", "reason": "gradient_tolerance"}
                    break
                if phase == "lbfgs" and config.loss_patience and stable_loss_steps >= config.loss_patience:
                    status = {"status": "stalled", "reason": "loss_change_patience"}
                    break
                if (
                    phase == "lbfgs"
                    and config.best_likelihood_patience
                    and best_step is not None
                    and step - int(best_step) >= config.best_likelihood_patience
                ):
                    status = {"status": "stalled", "reason": "best_likelihood_patience"}
                    break
            else:
                status = {"status": "not_converged", "reason": "max_steps"}

            model.theta.grad = None
            final_loss, final_metrics = self._evaluate_and_backward(model)
            final_step = max(start_step, min(config.steps, int(final_row.get("step", -1)) + 1))
            final_objective = float(final_loss.detach().cpu())
            final_improved = (
                best_nll is None
                or final_objective < best_nll - config.best_likelihood_min_delta
            )
            if final_improved:
                best_nll = final_objective
                best_step = final_step
            final_row = {
                "step": final_step,
                "optimizer/phase": "final_eval",
                "closure_evals": 1,
                "theta_step_inf": 0.0,
                "delta_likelihood_bits": None,
                "stable_loss_steps": stable_loss_steps,
                "best_nll_bits": best_nll,
                "best_step": best_step,
                **resume_info,
                "step_s": 0.0,
                **final_metrics,
            }
            self._record(final_row)

            final_status = {
                **status,
                **resume_info,
                "elapsed_s": time.perf_counter() - started,
                "best_nll_bits": best_nll,
                "best_step": best_step,
                "previous_objective": float(final_loss.detach().cpu()),
                "stable_loss_steps": stable_loss_steps,
            }
            if final_improved:
                self._save_status(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    step=int(final_row["step"]),
                    next_step=final_step,
                    status=final_status,
                    row=final_row,
                    optimizer_phase=current_phase,
                )
                sampling_checkpoint = best_checkpoint
            self._save_status(
                latest_checkpoint,
                model=model,
                optimizer=optimizer,
                step=int(final_row["step"]),
                next_step=final_step,
                status=final_status,
                row=final_row,
                optimizer_phase=current_phase,
            )
            if sampling_checkpoint is None:
                sampling_checkpoint = latest_checkpoint
            _write_rate_table(config.out_dir / "rates_final.tsv", model, config.mode)
            if config.mode == "genewise":
                _write_per_family_likelihoods(
                    config.out_dir / "per_fam_likelihoods.tsv",
                    model,
                )
            torch.save(model.theta.detach().cpu(), config.out_dir / "theta_final.pt")
            write_csv(config.out_dir / "optimization_history.csv", self.history)
            summary = {
                **final_status,
                "families": model.n_families,
                "species": int(model.n_species),
                "batches": len(model.batch_metadata),
                "final_nll_bits": float(final_loss.detach().cpu()),
                "final_grad_inf": float(final_row.get("grad/inf", math.inf)),
            }
            (config.out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return OptimizationResult(
                out_dir=config.out_dir,
                status=str(status["status"]),
                reason=str(status["reason"]),
                final_nll_bits=float(final_loss.detach().cpu()),
                final_grad_inf=float(final_row.get("grad/inf", math.inf)),
                best_nll_bits=None if best_nll is None else float(best_nll),
                best_step=None if best_step is None else int(best_step),
                steps_completed=int(final_row["step"]),
                sampling_checkpoint=sampling_checkpoint,
            )
        finally:
            model.close()


def optimize(config: RunConfig) -> OptimizationResult:
    return OptimizationRunner(config).run()
