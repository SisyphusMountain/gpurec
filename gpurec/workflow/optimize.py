from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gpurec.api.model import GeneReconModel

from ._artifact_publish import (
    StagedArtifact,
    create_artifact_temp_dir,
    publish_staged_artifacts,
)
from ._cleanup import cleanup_stage, cleanup_stage_after_error, close_model_after_error
from ._metadata import (
    MISSING,
    checkpoint_finite_float,
    checkpoint_nonnegative_int,
    checkpoint_progress,
    checkpoint_status_dict,
    model_family_names,
)
from .checkpoint import (
    load_checkpoint,
    restore_model_theta,
    save_checkpoint,
    validate_checkpoint_model_compatibility,
)
from .config import RunConfig
from .diagnostics import (
    append_jsonl,
    json_dumps_strict,
    parameter_stats,
    rates_and_survival_probability,
    solver_stats,
    tensor_stats,
    write_csv,
    write_json_strict,
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


@dataclass(frozen=True)
class _ResumeState:
    start_step: int = 0
    best_nll: float | None = None
    best_step: int | None = None
    previous_objective: float | None = None
    stable_loss_steps: int = 0
    active_batch_index: int = 0
    active_solver_stage: str = "full"
    active_batch_local_step: int = 0
    converged_family_indices: tuple[int, ...] = ()
    batch_plan_generation: int = 0


_FINAL_ARTIFACT_FILES = (
    "history.jsonl",
    "rates_final.tsv",
    "per_fam_likelihoods.tsv",
    "theta_final.pt",
    "optimization_history.csv",
    "summary.json",
)
_ACTIVE_BATCH_LBFGS_STALL_PATIENCE = 3
_ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES = 64
_FD_NEWTON_LARGE_BATCH_MAX_LS = 8
_FD_NEWTON_EXTENDED_LINE_SEARCH_MAX_FAMILIES = 256
_FD_NEWTON_FALLBACK_LINE_SEARCH_MAX_STEPS = 2
_FD_NEWTON_CURVATURE_EPS = 1e-12
_HESSIAN_SGD_LINE_SEARCH_MAX_STEPS = 8
_HESSIAN_SGD_LINE_SEARCH_ACCEPT_FRACTION = 0.6
_HESSIAN_SGD_LINE_SEARCH_LOW_ACCEPT_PATIENCE = 2
_HESSIAN_SGD_NO_LINE_REFRESH_STEPS = 64
_HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES = 400_000
_HESSIAN_SGD_LARGE_BATCH_WARMUP_ITERS = 2
_BATCHWISE_ACTIVE_OPTIMIZERS = frozenset(
    {"batched-lbfgs", "adam-fd-newton", "hessian-sgd"}
)
_HESSIAN_CONDITIONED_OPTIMIZERS = frozenset({"adam-fd-newton", "hessian-sgd"})
_POST_STEP_OPTIMIZERS = frozenset(
    {"lbfgs", "batched-lbfgs", "adam-fd-newton", "hessian-sgd"}
)


@dataclass
class _FDNewtonHessianState:
    batch_index: int
    solver_stage: str
    family_indices: tuple[int, ...]
    hessian: torch.Tensor
    active_theta: torch.Tensor
    active_grad: torch.Tensor
    active_loss: torch.Tensor
    updates_since_refresh: int = 0


def _is_finite_tensor(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and bool(torch.isfinite(tensor).all().item())


def _clear_cuda_allocator_cache_if_needed(model: GeneReconModel) -> None:
    theta = getattr(model, "theta", None)
    if bool(getattr(theta, "is_cuda", False)) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _drop_cached_static_states_if_needed(model: GeneReconModel) -> None:
    drop_cached_static_states = getattr(
        model,
        "drop_cached_static_states",
        None,
    )
    if callable(drop_cached_static_states):
        drop_cached_static_states()
    else:
        model.clear()
    _clear_cuda_allocator_cache_if_needed(model)


def _is_memory_retryable_runtime_error(exc: RuntimeError) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "memory budget" in message
        or "estimated scratch" in message
        or ("scratch" in message and "budget" in message)
    )


def _checkpoint_index_tuple(
    path: Path,
    name: str,
    value: Any,
) -> tuple[int, ...]:
    if value is MISSING or value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"checkpoint {path} has invalid status.{name}")
    out: list[int] = []
    seen: set[int] = set()
    for position, item in enumerate(value):
        index = int(
            checkpoint_nonnegative_int(
                path,
                f"status.{name}[{position}]",
                item,
            )
        )
        if index in seen:
            raise RuntimeError(
                f"checkpoint {path} has duplicate family index {index} in status.{name}"
            )
        seen.add(index)
        out.append(index)
    return tuple(out)


def _resume_state_from_payload(path: Path, payload: dict[str, Any]) -> _ResumeState:
    _, start_step = checkpoint_progress(path, payload)
    ckpt_status = checkpoint_status_dict(path, payload)

    return _ResumeState(
        start_step=start_step,
        best_nll=checkpoint_finite_float(
            path,
            "status.best_nll_bits",
            ckpt_status.get("best_nll_bits"),
            allow_none=True,
        ),
        best_step=checkpoint_nonnegative_int(
            path,
            "status.best_step",
            ckpt_status.get("best_step"),
            allow_none=True,
        ),
        previous_objective=checkpoint_finite_float(
            path,
            "status.previous_objective",
            ckpt_status.get("previous_objective"),
            allow_none=True,
        ),
        stable_loss_steps=int(
            checkpoint_nonnegative_int(
                path,
                "status.stable_loss_steps",
                ckpt_status.get("stable_loss_steps", MISSING),
                default=0,
            )
        ),
        active_batch_index=int(
            checkpoint_nonnegative_int(
                path,
                "status.active_batch_index",
                ckpt_status.get("active_batch_index", MISSING),
                default=0,
            )
        ),
        active_solver_stage=str(ckpt_status.get("active_solver_stage", "full")),
        active_batch_local_step=int(
            checkpoint_nonnegative_int(
                path,
                "status.active_batch_local_step",
                ckpt_status.get("active_batch_local_step", MISSING),
                default=0,
            )
        ),
        converged_family_indices=_checkpoint_index_tuple(
            path,
            "converged_family_indices",
            ckpt_status.get("converged_family_indices", MISSING),
        ),
        batch_plan_generation=int(
            checkpoint_nonnegative_int(
                path,
                "status.batch_plan_generation",
                ckpt_status.get("batch_plan_generation", MISSING),
                default=0,
            )
        ),
    )


def _validate_resume_progress(
    path: Path,
    state: _ResumeState,
    *,
    configured_steps: int,
) -> None:
    if state.start_step > configured_steps:
        raise RuntimeError(
            f"checkpoint {path} has next_step {state.start_step}, which exceeds "
            f"configured steps {configured_steps}"
        )


def _parameter_labels(model: GeneReconModel, mode: str) -> list[str]:
    theta_rows = int(model.theta.detach().reshape(-1, 3).shape[0])
    if mode == "genewise":
        return model_family_names(model)
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
def _per_family_nll(
    model: GeneReconModel,
    values: torch.Tensor | None = None,
) -> list[tuple[str, float]]:
    if values is None:
        values = model.full_nll_per_family()
    values = values.detach().cpu().reshape(-1)
    family_names = model_family_names(model)
    if values.numel() != len(family_names):
        raise RuntimeError(
            "per-family likelihood vector has "
            f"{values.numel()} rows for {len(family_names)} families"
        )
    return list(zip(family_names, values.tolist()))


def _write_per_family_likelihoods(
    path: Path,
    model: GeneReconModel,
    values: torch.Tensor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("family\tnll_bits\tlog_likelihood_bits\n")
        for family, nll in _per_family_nll(model, values):
            handle.write(f"{family}\t{nll:.12g}\t{-nll:.12g}\n")


def _write_history_jsonl_with_final_row(
    path: Path,
    current_history_path: Path,
    final_row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_handle:
        if current_history_path.is_file():
            existing = current_history_path.read_text(encoding="utf-8")
            out_handle.write(existing)
            if existing and not existing.endswith("\n"):
                out_handle.write("\n")
        out_handle.write(json_dumps_strict(final_row, sort_keys=True) + "\n")


def _final_artifact_paths(out_dir: Path) -> list[Path]:
    return [
        path
        for name in _FINAL_ARTIFACT_FILES
        if (path := out_dir / name).is_file()
    ]


def _clear_final_artifacts(out_dir: Path) -> None:
    for path in _final_artifact_paths(out_dir):
        path.unlink()


def _publish_final_artifacts(
    out_dir: Path,
    staged_outputs: list[StagedArtifact],
) -> None:
    publish_staged_artifacts(
        base_dir=out_dir,
        staged_outputs=staged_outputs,
        current_paths=_final_artifact_paths(out_dir),
        backup_prefix=".gpurec-optimization-backup-",
        clear_current=lambda: _clear_final_artifacts(out_dir),
    )


def _write_final_artifacts(
    config: RunConfig,
    *,
    model: GeneReconModel,
    history: list[dict[str, Any]],
    final_row: dict[str, Any],
    summary: dict[str, Any],
    history_jsonl: Path,
    per_family_nll: torch.Tensor | None = None,
) -> None:
    stage_dir: Path | None = None
    try:
        stage_dir = create_artifact_temp_dir(
            config.out_dir,
            prefix=".gpurec-optimization-stage-",
        )
        staged_outputs: list[StagedArtifact] = []

        history_stage_path = stage_dir / "history.jsonl"
        _write_history_jsonl_with_final_row(
            history_stage_path,
            history_jsonl,
            final_row,
        )
        history_jsonl_output = (history_stage_path, history_jsonl)

        rates_stage_path = stage_dir / "rates_final.tsv"
        _write_rate_table(rates_stage_path, model, config.mode)
        staged_outputs.append((rates_stage_path, config.out_dir / "rates_final.tsv"))

        if config.mode == "genewise":
            per_family_stage_path = stage_dir / "per_fam_likelihoods.tsv"
            _write_per_family_likelihoods(
                per_family_stage_path,
                model,
                per_family_nll,
            )
            staged_outputs.append(
                (
                    per_family_stage_path,
                    config.out_dir / "per_fam_likelihoods.tsv",
                )
            )

        theta_stage_path = stage_dir / "theta_final.pt"
        torch.save(model.theta.detach().cpu(), theta_stage_path)
        staged_outputs.append((theta_stage_path, config.out_dir / "theta_final.pt"))

        history_csv_stage_path = stage_dir / "optimization_history.csv"
        write_csv(history_csv_stage_path, history)
        staged_outputs.append(
            (history_csv_stage_path, config.out_dir / "optimization_history.csv")
        )

        summary_stage_path = stage_dir / "summary.json"
        write_json_strict(summary_stage_path, summary)
        staged_outputs.append(history_jsonl_output)
        staged_outputs.append((summary_stage_path, config.out_dir / "summary.json"))

        _publish_final_artifacts(config.out_dir, staged_outputs)
    except BaseException as exc:
        cleanup_stage_after_error(stage_dir, exc)
        raise
    else:
        cleanup_stage(stage_dir)


def _step_stopping_status(
    config: RunConfig,
    *,
    step: int,
    stable_loss_steps: int,
    best_step: int | None,
    loss_patience: int | None = None,
    best_likelihood_patience: int | None = None,
) -> dict[str, str] | None:
    loss_patience = config.loss_patience if loss_patience is None else loss_patience
    best_likelihood_patience = (
        config.best_likelihood_patience
        if best_likelihood_patience is None
        else best_likelihood_patience
    )
    if loss_patience and stable_loss_steps >= loss_patience:
        return {"status": "converged", "reason": "loss_change_patience"}
    if (
        best_likelihood_patience
        and best_step is not None
        and step - int(best_step) >= best_likelihood_patience
    ):
        return {"status": "converged", "reason": "best_likelihood_patience"}
    return None


def _active_batch_patience(configured_patience: int) -> int:
    if configured_patience <= 0:
        return configured_patience
    return min(configured_patience, _ACTIVE_BATCH_LBFGS_STALL_PATIENCE)


class OptimizationRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.history: list[dict[str, Any]] = []
        self.history_jsonl = config.out_dir / "history.jsonl"

    def build_model(self) -> GeneReconModel:
        config = self.config
        prefetch_batches: int | str = (
            1
            if config.mode == "genewise"
            and config.optimizer in _BATCHWISE_ACTIVE_OPTIMIZERS
            else "all"
        )
        return build_alerax_workflow_model(config, prefetch_batches=prefetch_batches)

    def _make_optimizer(self, model: GeneReconModel, phase: str) -> torch.optim.Optimizer:
        config = self.config
        if phase == "adam":
            return torch.optim.Adam([model.theta], lr=config.lr)
        if phase == "adagrad":
            return torch.optim.Adagrad([model.theta], lr=config.lr, eps=1e-10)
        if phase == "batched-lbfgs":
            from gpurec.optimization import BatchedLBFGS

            return BatchedLBFGS(
                [model.theta],
                lr=config.lbfgs_lr,
                max_iter=config.lbfgs_max_iter,
                history_size=config.lbfgs_history_size,
                max_ls=config.lbfgs_max_ls,
                tolerance_grad=0.0,
                line_search_fn=(
                    "strong_wolfe"
                    if config.lbfgs_line_search == "strong_wolfe"
                    else "armijo"
                ),
                lower_bound=math.log2(config.min_rate),
                upper_bound=math.log2(config.max_rate),
            )
        if phase == "adam-fd-newton":
            return torch.optim.Adam([model.theta], lr=config.lr)
        if phase == "hessian-sgd":
            return torch.optim.SGD([model.theta], lr=config.lr)
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

    def _evaluate_genewise_vector_and_grad(
        self,
        model: GeneReconModel,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        model.theta.grad = None
        loss_vec, grad = model.full_genewise_nll_and_grad(need_grad=True)
        if grad is None:
            raise RuntimeError("genewise optimizer evaluation did not produce gradients")
        model.theta.grad = grad.detach().to(
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        loss = loss_vec.sum()
        row: dict[str, Any] = {
            "likelihood/data_nll_bits": float(loss.detach().cpu()),
            "likelihood/log_likelihood_bits": float(-loss.detach().cpu()),
        }
        row.update(tensor_stats("grad", model.theta.grad))
        row.update(parameter_stats(model.theta))
        row.update(solver_stats(model))
        return loss_vec.detach(), row

    def _evaluate_genewise_loss_vector(self, model: GeneReconModel) -> torch.Tensor:
        loss_vec, _grad = model.full_genewise_nll_and_grad(need_grad=False)
        return loss_vec.detach()

    def _final_eval_fallback_clade_budgets(self) -> list[int]:
        current = self.config.clade_budget
        candidates: list[int] = []
        if current is not None:
            candidates.extend(
                max(1, int(current) // divisor)
                for divisor in (2, 5, 10, 20)
                if int(current) // divisor > 0
            )
        candidates.extend([100_000, 50_000, 25_000])
        seen: set[int] = set()
        out: list[int] = []
        for budget in candidates:
            if current is not None and budget >= int(current):
                continue
            if budget in seen:
                continue
            seen.add(budget)
            out.append(budget)
        return out

    def _evaluate_genewise_vector_and_grad_with_memory_fallback(
        self,
        model: GeneReconModel,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        try:
            return self._evaluate_genewise_vector_and_grad(model)
        except RuntimeError as original_exc:
            if not _is_memory_retryable_runtime_error(original_exc):
                raise
            _drop_cached_static_states_if_needed(model)
            try:
                loss_vec, metrics = self._evaluate_genewise_vector_and_grad(model)
                metrics = dict(metrics)
                metrics["optimizer/final_eval_source"] = (
                    "recomputed_after_cache_drop"
                )
                metrics["optimizer/final_eval_fallback_reason"] = (
                    f"{type(original_exc).__name__}: {original_exc}"
                )
                return loss_vec, metrics
            except RuntimeError as retry_exc:
                if not _is_memory_retryable_runtime_error(retry_exc):
                    raise
                _drop_cached_static_states_if_needed(model)
            budgets = self._final_eval_fallback_clade_budgets()
            if not budgets:
                raise
            fallback_errors: list[str] = []
            for budget in budgets:
                fallback_model: GeneReconModel | None = None
                try:
                    fallback_data = self.config.to_dict()
                    fallback_data["clade_budget"] = budget
                    fallback_config = RunConfig.from_dict(fallback_data)
                    fallback_model = build_alerax_workflow_model(
                        fallback_config,
                        prefetch_batches=1,
                    )
                    with torch.no_grad():
                        fallback_model.theta.copy_(
                            model.theta.detach().to(
                                device=fallback_model.theta.device,
                                dtype=fallback_model.theta.dtype,
                            )
                        )
                    self._configure_solver_stage(fallback_model, "full")
                    loss_vec, metrics = self._evaluate_genewise_vector_and_grad(
                        fallback_model
                    )
                    if fallback_model.theta.grad is None:
                        raise RuntimeError("fallback final eval did not produce gradients")
                    model.theta.grad = fallback_model.theta.grad.detach().to(
                        device=model.theta.device,
                        dtype=model.theta.dtype,
                    )
                    metrics = dict(metrics)
                    metrics["optimizer/final_eval_source"] = (
                        "fallback_clade_budget"
                    )
                    metrics["optimizer/final_eval_fallback_clade_budget"] = float(
                        budget
                    )
                    metrics["optimizer/final_eval_fallback_reason"] = (
                        f"{type(original_exc).__name__}: {original_exc}"
                    )
                    metrics.update(tensor_stats("grad", model.theta.grad))
                    metrics.update(parameter_stats(model.theta))
                    return loss_vec.detach().to(
                        device=model.theta.device,
                        dtype=model.theta.dtype,
                    ), metrics
                except RuntimeError as fallback_exc:
                    if not _is_memory_retryable_runtime_error(fallback_exc):
                        raise
                    fallback_errors.append(
                        f"clade_budget={budget}: "
                        f"{type(fallback_exc).__name__}: {fallback_exc}"
                    )
                except Exception:
                    raise
                finally:
                    if fallback_model is not None:
                        fallback_model.close()
                    _clear_cuda_allocator_cache_if_needed(model)
            raise RuntimeError(
                "final genewise evaluation failed in the resident layout and all "
                "smaller-clade fallbacks failed; original error: "
                f"{type(original_exc).__name__}: {original_exc}; fallbacks: "
                + "; ".join(fallback_errors)
            ) from original_exc

    def _evaluate_final_check_genewise_with_memory_fallback(
        self,
        model: GeneReconModel,
        *,
        check_iters: int,
        original_exc: RuntimeError,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        budgets = self._final_eval_fallback_clade_budgets()
        if not budgets:
            raise original_exc
        _clear_cuda_allocator_cache_if_needed(model)
        fallback_errors: list[str] = []
        for budget in budgets:
            fallback_model: GeneReconModel | None = None
            try:
                fallback_data = self.config.to_dict()
                fallback_data["clade_budget"] = budget
                fallback_config = RunConfig.from_dict(fallback_data)
                fallback_model = build_alerax_workflow_model(
                    fallback_config,
                    prefetch_batches=1,
                )
                with torch.no_grad():
                    fallback_model.theta.copy_(
                        model.theta.detach().to(
                            device=fallback_model.theta.device,
                            dtype=fallback_model.theta.dtype,
                        )
                    )
                fallback_model.configure_solver_iterations(
                    fixed_iters_E=self.config.fixed_iters_e,
                    fixed_iters_Pi=check_iters,
                    neumann_terms=check_iters,
                )
                loss_vec, _metrics = self._evaluate_genewise_vector_and_grad(
                    fallback_model
                )
                if fallback_model.theta.grad is None:
                    raise RuntimeError("fallback final check did not produce gradients")
                grad = fallback_model.theta.grad.detach().to(
                    device=model.theta.device,
                    dtype=model.theta.dtype,
                )
                metrics = {
                    "optimizer/final_check_source": "fallback_clade_budget",
                    "optimizer/final_check_fallback_clade_budget": float(budget),
                    "optimizer/final_check_fallback_reason": (
                        f"{type(original_exc).__name__}: {original_exc}"
                    ),
                }
                return (
                    loss_vec.detach().to(
                        device=model.theta.device,
                        dtype=model.theta.dtype,
                    ),
                    grad,
                    metrics,
                )
            except RuntimeError as fallback_exc:
                if not _is_memory_retryable_runtime_error(fallback_exc):
                    raise
                fallback_errors.append(
                    f"clade_budget={budget}: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                )
            except Exception:
                raise
            finally:
                if fallback_model is not None:
                    fallback_model.close()
                _clear_cuda_allocator_cache_if_needed(model)
        raise RuntimeError(
            "final iteration check failed in the resident layout and all "
            "smaller-clade fallbacks failed; original error: "
            f"{type(original_exc).__name__}: {original_exc}; fallbacks: "
            + "; ".join(fallback_errors)
        ) from original_exc

    def _uses_solver_warmup(self) -> bool:
        return (
            self.config.mode == "genewise"
            and self.config.optimizer in _BATCHWISE_ACTIVE_OPTIMIZERS
            and self.config.solver_warmup_iters > 0
        )

    def _configure_solver_stage(
        self,
        model: GeneReconModel,
        stage: str,
    ) -> None:
        config = self.config
        if stage == "warmup":
            iters = int(config.solver_warmup_iters)
            if config.optimizer == "hessian-sgd":
                active_clade_count = int(
                    getattr(model.current_batch_metadata, "clade_count", 0) or 0
                )
                if active_clade_count >= _HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES:
                    iters = min(iters, _HESSIAN_SGD_LARGE_BATCH_WARMUP_ITERS)
            fixed_iters_E = (
                config.fixed_iters_e
                if config.optimizer == "hessian-sgd"
                else iters
            )
            model.configure_solver_iterations(
                fixed_iters_E=fixed_iters_E,
                fixed_iters_Pi=iters,
                neumann_terms=iters,
            )
            return
        if stage == "full":
            model.configure_solver_iterations(
                fixed_iters_E=config.fixed_iters_e,
                fixed_iters_Pi=config.fixed_iters_pi,
                neumann_terms=config.neumann_terms,
            )
            return
        raise ValueError(f"unknown solver stage {stage!r}")

    def _configure_active_solver_stage(
        self,
        model: GeneReconModel,
        stage: str,
    ) -> None:
        config = self.config
        if config.optimizer == "hessian-sgd" and stage == "full":
            model.configure_solver_iterations(
                fixed_iters_E=config.fixed_iters_e,
                fixed_iters_Pi=(
                    config.hessian_sgd_normal_fixed_iters_pi
                    if config.hessian_sgd_normal_fixed_iters_pi is not None
                    else config.fixed_iters_pi
                ),
                neumann_terms=(
                    config.hessian_sgd_normal_neumann_terms
                    if config.hessian_sgd_normal_neumann_terms is not None
                    else config.neumann_terms
                ),
            )
            return
        self._configure_solver_stage(model, stage)

    def _active_batch_result_is_canonical_full_solver(
        self,
        *,
        phase: str,
        solver_stage: str,
    ) -> bool:
        config = self.config
        if solver_stage != "full":
            return False
        if phase != "hessian-sgd":
            return True
        normal_pi = (
            config.hessian_sgd_normal_fixed_iters_pi
            if config.hessian_sgd_normal_fixed_iters_pi is not None
            else config.fixed_iters_pi
        )
        normal_neumann = (
            config.hessian_sgd_normal_neumann_terms
            if config.hessian_sgd_normal_neumann_terms is not None
            else config.neumann_terms
        )
        return (
            normal_pi == config.fixed_iters_pi
            and normal_neumann == config.neumann_terms
        )

    def _evaluate_final_iteration_check(
        self,
        model: GeneReconModel,
        *,
        baseline_loss: torch.Tensor,
        baseline_grad: torch.Tensor,
    ) -> dict[str, Any]:
        config = self.config
        check_iters = int(config.final_check_iters)
        configure_solver = getattr(model, "configure_solver_iterations", None)
        if config.mode != "genewise" or not callable(configure_solver):
            return {
                "optimizer/final_check_status": "skipped",
                "optimizer/final_check_iters": check_iters,
            }
        if check_iters <= 0:
            return {
                "optimizer/final_check_status": "disabled",
                "optimizer/final_check_iters": 0,
            }

        baseline_loss_bits = float(baseline_loss.detach().cpu())
        baseline_grad = baseline_grad.detach().clone()
        baseline_grad_inf = (
            float(baseline_grad.detach().abs().amax().cpu())
            if baseline_grad.numel()
            else 0.0
        )

        metrics: dict[str, Any] = {
            "optimizer/final_check_status": "failed",
            "optimizer/final_check_iters": check_iters,
            "optimizer/final_check_evals": 1,
        }
        try:
            _drop_cached_static_states_if_needed(model)
            configure_solver(
                fixed_iters_E=config.fixed_iters_e,
                fixed_iters_Pi=check_iters,
                neumann_terms=check_iters,
            )
            if config.mode == "genewise" and callable(
                getattr(model, "full_genewise_nll_and_grad", None)
            ):
                try:
                    check_loss_vec, _check_metrics = (
                        self._evaluate_genewise_vector_and_grad(model)
                    )
                    check_grad = model.theta.grad
                except RuntimeError as check_exc:
                    if not _is_memory_retryable_runtime_error(check_exc):
                        raise
                    check_loss_vec, check_grad, fallback_metrics = (
                        self._evaluate_final_check_genewise_with_memory_fallback(
                            model,
                            check_iters=check_iters,
                            original_exc=check_exc,
                        )
                    )
                    metrics.update(fallback_metrics)
                check_loss = check_loss_vec.sum()
            else:
                check_loss, _check_metrics = self._evaluate_and_backward(model)
                check_grad = model.theta.grad
            check_failed = (
                not torch.isfinite(check_loss).item()
                or not _is_finite_tensor(check_grad)
            )
            if check_failed or check_grad is None:
                metrics["optimizer/final_check_reason"] = (
                    "nonfinite_objective_or_gradient"
                )
                return metrics

            check_grad = check_grad.detach()
            check_loss_bits = float(check_loss.detach().cpu())
            grad_delta = (check_grad - baseline_grad).detach()
            grad_delta_inf = (
                float(grad_delta.abs().amax().cpu()) if grad_delta.numel() else 0.0
            )
            check_grad_inf = (
                float(check_grad.abs().amax().cpu()) if check_grad.numel() else 0.0
            )
            grad_scale = max(baseline_grad_inf, check_grad_inf, 1.0)
            metrics.update(
                {
                    "optimizer/final_check_status": "ok",
                    "optimizer/final_check_loss_bits": check_loss_bits,
                    "optimizer/final_check_loss_delta_bits": (
                        check_loss_bits - baseline_loss_bits
                    ),
                    "optimizer/final_check_loss_abs_delta_bits": abs(
                        check_loss_bits - baseline_loss_bits
                    ),
                    "optimizer/final_check_grad_inf": check_grad_inf,
                    "optimizer/final_check_grad_baseline_inf": baseline_grad_inf,
                    "optimizer/final_check_grad_max_abs_delta": grad_delta_inf,
                    "optimizer/final_check_grad_rel_inf_delta": (
                        grad_delta_inf / grad_scale
                    ),
                }
            )
            return metrics
        except Exception as exc:  # pragma: no cover - defensive diagnostic path
            metrics["optimizer/final_check_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return metrics
        finally:
            self._configure_solver_stage(model, "full")
            model.theta.grad = baseline_grad
            model.clear()

    def _should_switch_solver_warmup(
        self,
        *,
        stable_loss_steps: int,
    ) -> bool:
        config = self.config
        return (
            config.solver_warmup_loss_patience > 0
            and stable_loss_steps >= config.solver_warmup_loss_patience
        )

    def _active_batch_indices(self, model: GeneReconModel) -> torch.Tensor:
        indices = getattr(model.current_batch_metadata, "family_indices")
        return torch.as_tensor(
            indices,
            dtype=torch.long,
            device=model.theta.device,
        )

    def _full_vector_from_active_batch(
        self,
        model: GeneReconModel,
        active_values: torch.Tensor,
    ) -> torch.Tensor:
        idx = self._active_batch_indices(model)
        values = active_values.detach().reshape(-1).to(
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        if values.numel() != idx.numel():
            raise RuntimeError(
                "active genewise objective returned "
                f"{values.numel()} values for {idx.numel()} batch families"
            )
        full = torch.zeros(
            (int(model.n_families),),
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        full.index_copy_(0, idx, values)
        return full

    def _zero_inactive_batch_grad(
        self,
        model: GeneReconModel,
        idx: torch.Tensor,
    ) -> None:
        grad = model.theta.grad
        if grad is None:
            raise RuntimeError("active genewise optimizer evaluation did not produce gradients")
        mask = torch.zeros(
            (int(model.n_families),),
            device=grad.device,
            dtype=torch.bool,
        )
        mask.index_fill_(0, idx.to(device=grad.device), True)
        grad = grad.detach().clone()
        grad[~mask] = 0
        model.theta.grad = grad

    def _active_batch_metrics(
        self,
        model: GeneReconModel,
        *,
        loss_vec: torch.Tensor,
        solver_stage: str,
    ) -> dict[str, Any]:
        metadata = model.current_batch_metadata
        family_indices = tuple(int(idx) for idx in metadata.family_indices)
        loss = loss_vec.sum()
        row: dict[str, Any] = {
            "likelihood/data_nll_bits": float(loss.detach().cpu()),
            "likelihood/log_likelihood_bits": float(-loss.detach().cpu()),
            "optimizer/objective_scope": "active_batch",
            "optimizer/batch_index": int(model.current_batch_index),
            "optimizer/batch_family_count": int(len(family_indices)),
            "optimizer/solver_stage": solver_stage,
        }
        if family_indices:
            row["optimizer/batch_family_first"] = int(min(family_indices))
            row["optimizer/batch_family_last"] = int(max(family_indices))
        row.update(tensor_stats("grad", model.theta.grad))
        row.update(parameter_stats(model.theta))
        row.update(solver_stats(model))
        return row

    def _evaluate_active_genewise_vector_and_grad(
        self,
        model: GeneReconModel,
        *,
        solver_stage: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        model.theta.grad = None
        local_loss_vec = model.nll_per_family()
        local_loss_vec.sum().backward()
        idx = self._active_batch_indices(model)
        self._zero_inactive_batch_grad(model, idx)
        loss_vec = self._full_vector_from_active_batch(model, local_loss_vec)
        return loss_vec, self._active_batch_metrics(
            model,
            loss_vec=loss_vec,
            solver_stage=solver_stage,
        )

    def _evaluate_active_genewise_loss_vector(self, model: GeneReconModel) -> torch.Tensor:
        local_loss_vec = model.nll_per_family()
        return self._full_vector_from_active_batch(model, local_loss_vec)

    def _projected_grad_inf(
        self,
        model: GeneReconModel,
        *,
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[torch.Tensor, float]:
        grad = model.theta.grad
        if grad is None:
            raise RuntimeError("projected gradient requested before gradient evaluation")
        theta = model.theta.detach()
        projected = grad.detach().clone()
        projected[(theta <= lower_bound) & (projected > 0)] = 0
        projected[(theta >= upper_bound) & (projected < 0)] = 0
        projected_inf = float(projected.detach().abs().amax().cpu()) if projected.numel() else 0.0
        return projected, projected_inf

    def _evaluate_active_genewise_vector_grad_at_current_theta(
        self,
        model: GeneReconModel,
        *,
        solver_stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        model.clear()
        loss_vec, metrics = self._evaluate_active_genewise_vector_and_grad(
            model,
            solver_stage=solver_stage,
        )
        if model.theta.grad is None:
            raise RuntimeError("active genewise evaluation did not produce gradients")
        return loss_vec.detach(), model.theta.grad.detach().clone(), metrics

    def _active_adam_step(
        self,
        model: GeneReconModel,
        optimizer: torch.optim.Optimizer,
        *,
        solver_stage: str,
    ) -> tuple[torch.Tensor, dict[str, Any], int]:
        optimizer.zero_grad(set_to_none=True)
        _pre_loss, _pre_grad, _pre_metrics = (
            self._evaluate_active_genewise_vector_grad_at_current_theta(
                model,
                solver_stage=solver_stage,
            )
        )
        optimizer.step()
        with torch.no_grad():
            model.clamp_theta_(self.config.min_rate, self.config.max_rate)
        loss_vec, _grad, metrics = self._evaluate_active_genewise_vector_grad_at_current_theta(
            model,
            solver_stage=solver_stage,
        )
        _projected_grad, projected_grad_inf = self._projected_grad_inf(
            model,
            lower_bound=math.log2(self.config.min_rate),
            upper_bound=math.log2(self.config.max_rate),
        )
        metrics["grad/projected_inf"] = projected_grad_inf
        return loss_vec, metrics, 2

    def _active_projected_grad_and_free(
        self,
        active_theta: torch.Tensor,
        active_grad: torch.Tensor,
        *,
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = active_grad.clone()
        projected[(active_theta <= lower_bound) & (projected > 0)] = 0
        projected[(active_theta >= upper_bound) & (projected < 0)] = 0
        return projected, projected.abs() > 0

    def _fd_newton_state_matches(
        self,
        model: GeneReconModel,
        state: _FDNewtonHessianState | None,
        *,
        solver_stage: str,
    ) -> bool:
        if state is None:
            return False
        if state.batch_index != int(model.current_batch_index):
            return False
        if state.solver_stage != solver_stage:
            return False
        family_indices = tuple(int(idx) for idx in model.current_batch_metadata.family_indices)
        if state.family_indices != family_indices:
            return False
        idx = self._active_batch_indices(model)
        active_theta = model.theta.detach().index_select(0, idx)
        return torch.equal(active_theta, state.active_theta)

    def _refresh_fd_newton_hessian_state(
        self,
        model: GeneReconModel,
        *,
        solver_stage: str,
        baseline_state: _FDNewtonHessianState | None = None,
    ) -> tuple[_FDNewtonHessianState, dict[str, Any], int]:
        config = self.config
        idx = self._active_batch_indices(model)
        theta0 = model.theta.detach().clone()
        lower_bound = math.log2(config.min_rate)
        upper_bound = math.log2(config.max_rate)
        eps = float(config.fd_hessian_epsilon)

        if baseline_state is None:
            loss0, grad0, metrics0 = (
                self._evaluate_active_genewise_vector_grad_at_current_theta(
                    model,
                    solver_stage=solver_stage,
                )
            )
            grad_evals = 1
            active_grad0 = grad0.index_select(0, idx)
            active_loss0 = loss0.index_select(0, idx)
        else:
            active_grad0 = baseline_state.active_grad.detach().clone()
            active_loss0 = baseline_state.active_loss.detach().clone()
            metrics0 = {
                "grad/inf": (
                    float(active_grad0.detach().abs().amax().cpu())
                    if active_grad0.numel()
                    else 0.0
                ),
            }
            grad_evals = 0
        loss_evals = 0
        active_theta0 = theta0.index_select(0, idx)
        rows, cols = active_grad0.shape
        if cols != 3:
            raise RuntimeError(
                "Hessian-conditioned genewise optimization expects three D/T/L "
                "parameters per family; "
                f"got {cols}"
            )

        projected_grad, free = self._active_projected_grad_and_free(
            active_theta0,
            active_grad0,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        hessian = torch.zeros(
            (rows, cols, cols),
            device=model.theta.device,
            dtype=model.theta.dtype,
        )
        for col in range(cols):
            plus = theta0.clone()
            minus = theta0.clone()
            plus_active = torch.clamp(
                active_theta0[:, col] + eps,
                min=lower_bound,
                max=upper_bound,
            )
            minus_active = torch.clamp(
                active_theta0[:, col] - eps,
                min=lower_bound,
                max=upper_bound,
            )
            plus_rows = active_theta0.clone()
            minus_rows = active_theta0.clone()
            plus_rows[:, col] = plus_active
            minus_rows[:, col] = minus_active
            plus.index_copy_(0, idx, plus_rows)
            minus.index_copy_(0, idx, minus_rows)
            self._set_model_theta(model, plus)
            _plus_loss, plus_grad, _plus_metrics = (
                self._evaluate_active_genewise_vector_grad_at_current_theta(
                    model,
                    solver_stage=solver_stage,
                )
            )
            self._set_model_theta(model, minus)
            _minus_loss, minus_grad, _minus_metrics = (
                self._evaluate_active_genewise_vector_grad_at_current_theta(
                    model,
                    solver_stage=solver_stage,
                )
            )
            grad_evals += 2
            denom = (plus_active - minus_active).abs().clamp_min(1e-12)
            hessian[:, :, col] = (
                plus_grad.index_select(0, idx) - minus_grad.index_select(0, idx)
            ) / denom[:, None]

        self._set_model_theta(model, theta0)
        hessian = 0.5 * (hessian + hessian.transpose(1, 2))
        state = _FDNewtonHessianState(
            batch_index=int(model.current_batch_index),
            solver_stage=solver_stage,
            family_indices=tuple(
                int(index) for index in model.current_batch_metadata.family_indices
            ),
            hessian=hessian.detach().clone(),
            active_theta=active_theta0.detach().clone(),
            active_grad=active_grad0.detach().clone(),
            active_loss=active_loss0.detach().clone(),
            updates_since_refresh=0,
        )
        metrics0 = dict(metrics0)
        metrics0["grad/projected_inf"] = (
            float(projected_grad.detach().abs().amax().cpu())
            if projected_grad.numel()
            else 0.0
        )
        return state, metrics0, grad_evals

    def _bfgs_update_fd_newton_hessian(
        self,
        *,
        state: _FDNewtonHessianState,
        active_theta: torch.Tensor,
        active_grad: torch.Tensor,
        active_loss: torch.Tensor,
        accepted: torch.Tensor,
        free_before: torch.Tensor,
        free_after: torch.Tensor,
    ) -> tuple[_FDNewtonHessianState, torch.Tensor]:
        old_hessian = state.hessian.detach()
        s = active_theta - state.active_theta
        y = active_grad - state.active_grad
        bs = torch.bmm(old_hessian, s.unsqueeze(-1)).squeeze(-1)
        sbs = (s * bs).sum(dim=1)
        ys = (y * s).sum(dim=1)
        finite = (
            torch.isfinite(s).all(dim=1)
            & torch.isfinite(y).all(dim=1)
            & torch.isfinite(bs).all(dim=1)
            & torch.isfinite(sbs)
            & torch.isfinite(ys)
        )
        active_set_same = (free_before == free_after).all(dim=1)
        moved = s.abs().amax(dim=1) > _FD_NEWTON_CURVATURE_EPS
        valid_update = (
            accepted
            & moved
            & finite
            & active_set_same
            & (ys > _FD_NEWTON_CURVATURE_EPS)
            & (sbs > _FD_NEWTON_CURVATURE_EPS)
        )
        safe_sbs = sbs.abs().clamp_min(_FD_NEWTON_CURVATURE_EPS)
        safe_ys = ys.abs().clamp_min(_FD_NEWTON_CURVATURE_EPS)
        bfgs_hessian = (
            old_hessian
            - torch.einsum("bi,bj->bij", bs, bs) / safe_sbs[:, None, None]
            + torch.einsum("bi,bj->bij", y, y) / safe_ys[:, None, None]
        )
        bfgs_hessian = 0.5 * (bfgs_hessian + bfgs_hessian.transpose(1, 2))
        hessian = torch.where(
            valid_update[:, None, None],
            bfgs_hessian,
            old_hessian,
        )
        new_state = _FDNewtonHessianState(
            batch_index=state.batch_index,
            solver_stage=state.solver_stage,
            family_indices=state.family_indices,
            hessian=hessian.detach().clone(),
            active_theta=active_theta.detach().clone(),
            active_grad=active_grad.detach().clone(),
            active_loss=active_loss.detach().clone(),
            updates_since_refresh=state.updates_since_refresh + 1,
        )
        return new_state, valid_update

    def _active_fd_newton_step(
        self,
        model: GeneReconModel,
        *,
        solver_stage: str,
        hessian_state: _FDNewtonHessianState | None = None,
        update_hessian_with_bfgs: bool = True,
        step_scale: float = 1.0,
        use_line_search: bool = True,
        reject_loss_increases_after_step: bool = False,
        hessian_refresh_steps: int | None = None,
        line_search_max_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any], int, _FDNewtonHessianState]:
        config = self.config
        hessian_refresh_steps = (
            config.fd_hessian_refresh_steps
            if hessian_refresh_steps is None
            else int(hessian_refresh_steps)
        )
        if hessian_refresh_steps < 1:
            raise ValueError("hessian_refresh_steps must be positive")
        idx = self._active_batch_indices(model)
        lower_bound = math.log2(config.min_rate)
        upper_bound = math.log2(config.max_rate)
        damping = float(config.fd_newton_damping)
        grad_evals = 0
        loss_evals = 0
        hessian_state_matches = self._fd_newton_state_matches(
            model,
            hessian_state,
            solver_stage=solver_stage,
        )
        refreshed_hessian = (
            not hessian_state_matches
            or hessian_state is None
            or hessian_state.updates_since_refresh >= hessian_refresh_steps
        )
        if refreshed_hessian:
            hessian_state, metrics0, refresh_grad_evals = (
                self._refresh_fd_newton_hessian_state(
                    model,
                    solver_stage=solver_stage,
                    baseline_state=(
                        hessian_state if hessian_state_matches else None
                    ),
                )
            )
            grad_evals += refresh_grad_evals
        else:
            metrics0 = {}

        theta0 = model.theta.detach().clone()
        active_theta0 = hessian_state.active_theta.detach()
        active_grad0 = hessian_state.active_grad.detach()
        active_loss0 = hessian_state.active_loss.detach()
        hessian = hessian_state.hessian.detach()
        rows, cols = active_grad0.shape
        if cols != 3:
            raise RuntimeError(
                "Hessian-conditioned genewise optimization expects three D/T/L "
                "parameters per family; "
                f"got {cols}"
            )

        projected_grad, free = self._active_projected_grad_and_free(
            active_theta0,
            active_grad0,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        projected_grad_inf = (
            float(projected_grad.detach().abs().amax().cpu())
            if projected_grad.numel()
            else 0.0
        )
        row_active = free.any(dim=1)
        eye = torch.eye(cols, device=hessian.device, dtype=hessian.dtype).expand(rows, cols, cols)
        free_matrix = free[:, :, None] & free[:, None, :]
        hessian_solve = torch.where(free_matrix, hessian, torch.zeros_like(hessian))
        hessian_solve = hessian_solve + damping * eye
        diag = torch.diagonal(hessian_solve, dim1=1, dim2=2)
        diag.copy_(torch.where(free, diag, torch.ones_like(diag)))
        rhs = -torch.where(free, projected_grad, torch.zeros_like(projected_grad))
        solution, solve_info = torch.linalg.solve_ex(hessian_solve, rhs.unsqueeze(-1))
        step = solution.squeeze(-1)
        solve_ok = (solve_info == 0) & torch.isfinite(step).all(dim=1)
        descent = (projected_grad * step).sum(dim=1) < -1e-12
        fallback = row_active & (~solve_ok | ~descent)
        step = torch.where(fallback[:, None], -projected_grad, step)
        step = torch.where(row_active[:, None], step, torch.zeros_like(step))
        step = step * float(step_scale)
        raw_step_inf = float(step.detach().abs().amax().cpu()) if step.numel() else 0.0
        bounded_step = (
            torch.clamp(active_theta0 + step, min=lower_bound, max=upper_bound)
            - active_theta0
        )
        bounded_step_inf = (
            float(bounded_step.detach().abs().amax().cpu())
            if bounded_step.numel()
            else 0.0
        )
        gtd = (projected_grad * bounded_step).sum(dim=1)
        valid_projected_step = row_active & torch.isfinite(gtd) & (gtd < -1e-12)
        searching = valid_projected_step
        accepted = torch.zeros(rows, device=model.theta.device, dtype=torch.bool)
        accepted_active = active_theta0.clone()
        alpha = torch.ones(rows, device=model.theta.device, dtype=model.theta.dtype)
        line_search_fallback_attempted = torch.zeros_like(accepted)
        line_search_fallback_accepted = torch.zeros_like(accepted)
        max_line_search_steps = 0
        if use_line_search:
            max_line_search_steps = (
                int(config.lbfgs_max_ls)
                if line_search_max_steps is None
                else int(line_search_max_steps)
            )
            if max_line_search_steps < 1:
                raise ValueError("line_search_max_steps must be positive")
            if rows > _FD_NEWTON_EXTENDED_LINE_SEARCH_MAX_FAMILIES:
                max_line_search_steps = min(
                    max_line_search_steps,
                    _FD_NEWTON_LARGE_BATCH_MAX_LS,
                )

            for _ in range(max_line_search_steps):
                if not bool(searching.any()):
                    break
                trial_active = torch.clamp(
                    active_theta0 + alpha[:, None] * step,
                    min=lower_bound,
                    max=upper_bound,
                )
                candidate_active = torch.where(
                    searching[:, None],
                    trial_active,
                    accepted_active,
                )
                candidate = theta0.clone()
                candidate.index_copy_(0, idx, candidate_active)
                self._set_model_theta(model, candidate)
                model.clear()
                with torch.no_grad():
                    trial_loss_vec = self._evaluate_active_genewise_loss_vector(model)
                loss_evals += 1
                trial_active_loss = trial_loss_vec.index_select(0, idx)
                trial_delta = trial_active - active_theta0
                trial_gtd = (projected_grad * trial_delta).sum(dim=1)
                trial_searching = searching & torch.isfinite(trial_gtd) & (
                    trial_gtd < -1e-12
                )
                armijo_rhs = active_loss0 + 1e-4 * trial_gtd
                ok = trial_searching & torch.isfinite(trial_active_loss) & (
                    trial_active_loss <= armijo_rhs
                )
                if bool(ok.any()):
                    accepted = accepted | ok
                    accepted_active = torch.where(
                        ok[:, None],
                        trial_active,
                        accepted_active,
                    )
                searching = trial_searching & ~accepted
                alpha = torch.where(searching, alpha * 0.5, alpha)
            fallback_searching = row_active & ~accepted & torch.isfinite(
                projected_grad,
            ).all(dim=1)
            fallback_step = -torch.where(
                free,
                projected_grad,
                torch.zeros_like(projected_grad),
            ) * float(step_scale)
            fallback_bounded_step = (
                torch.clamp(
                    active_theta0 + fallback_step,
                    min=lower_bound,
                    max=upper_bound,
                )
                - active_theta0
            )
            fallback_gtd = (projected_grad * fallback_bounded_step).sum(dim=1)
            fallback_searching = fallback_searching & torch.isfinite(
                fallback_gtd,
            ) & (fallback_gtd < -1e-12)
            line_search_fallback_attempted = fallback_searching.clone()
            fallback_alpha = torch.ones(
                rows,
                device=model.theta.device,
                dtype=model.theta.dtype,
            )
            fallback_line_search_steps = min(
                max_line_search_steps,
                _FD_NEWTON_FALLBACK_LINE_SEARCH_MAX_STEPS,
            )
            for _ in range(fallback_line_search_steps):
                if not bool(fallback_searching.any()):
                    break
                trial_active = torch.clamp(
                    active_theta0 + fallback_alpha[:, None] * fallback_step,
                    min=lower_bound,
                    max=upper_bound,
                )
                candidate_active = torch.where(
                    fallback_searching[:, None],
                    trial_active,
                    accepted_active,
                )
                candidate = theta0.clone()
                candidate.index_copy_(0, idx, candidate_active)
                self._set_model_theta(model, candidate)
                model.clear()
                with torch.no_grad():
                    trial_loss_vec = self._evaluate_active_genewise_loss_vector(model)
                loss_evals += 1
                trial_active_loss = trial_loss_vec.index_select(0, idx)
                trial_delta = trial_active - active_theta0
                trial_gtd = (projected_grad * trial_delta).sum(dim=1)
                trial_searching = fallback_searching & torch.isfinite(trial_gtd) & (
                    trial_gtd < -1e-12
                )
                armijo_rhs = active_loss0 + 1e-4 * trial_gtd
                ok = trial_searching & torch.isfinite(trial_active_loss) & (
                    trial_active_loss <= armijo_rhs
                )
                if bool(ok.any()):
                    accepted = accepted | ok
                    line_search_fallback_accepted = (
                        line_search_fallback_accepted | ok
                    )
                    accepted_active = torch.where(
                        ok[:, None],
                        trial_active,
                        accepted_active,
                    )
                fallback_searching = trial_searching & ~accepted
                fallback_alpha = torch.where(
                    fallback_searching,
                    fallback_alpha * 0.5,
                    fallback_alpha,
                )
        else:
            trial_active = active_theta0 + bounded_step
            accepted = valid_projected_step
            accepted_active = torch.where(
                accepted[:, None],
                trial_active,
                accepted_active,
            )

        final_theta = theta0.clone()
        final_theta.index_copy_(0, idx, accepted_active)
        self._set_model_theta(model, final_theta)
        loss_vec, _grad, metrics = self._evaluate_active_genewise_vector_grad_at_current_theta(
            model,
            solver_stage=solver_stage,
        )
        grad_evals += 1
        active_theta1 = final_theta.index_select(0, idx).detach()
        active_loss1 = loss_vec.detach().index_select(0, idx)
        active_grad1 = model.theta.grad.detach().index_select(0, idx)
        loss_rejected = torch.zeros_like(accepted)
        if reject_loss_increases_after_step:
            finite_loss = torch.isfinite(active_loss1)
            accepted_after_loss = accepted & finite_loss & (active_loss1 <= active_loss0)
            loss_rejected = accepted & ~accepted_after_loss
            if bool(loss_rejected.any().detach().cpu()):
                accepted = accepted_after_loss
                active_theta1 = torch.where(
                    accepted[:, None],
                    active_theta1,
                    active_theta0,
                )
                active_loss1 = torch.where(accepted, active_loss1, active_loss0)
                active_grad1 = torch.where(
                    accepted[:, None],
                    active_grad1,
                    active_grad0,
                )
                final_theta = final_theta.clone()
                final_theta.index_copy_(0, idx, active_theta1)
                self._set_model_theta(model, final_theta)
                loss_vec = loss_vec.detach().clone()
                loss_vec.index_copy_(0, idx, active_loss1)
                grad = model.theta.grad.detach().clone()
                grad.index_copy_(0, idx, active_grad1)
                model.theta.grad = grad
                corrected_loss = loss_vec.sum()
                metrics = dict(metrics)
                metrics["likelihood/data_nll_bits"] = float(
                    corrected_loss.detach().cpu()
                )
                metrics["likelihood/log_likelihood_bits"] = float(
                    -corrected_loss.detach().cpu()
                )
                metrics.update(tensor_stats("grad", model.theta.grad))
                metrics.update(parameter_stats(model.theta))
        final_projected_grad, final_projected_grad_inf = self._projected_grad_inf(
            model,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        active_projected_grad1 = final_projected_grad.index_select(0, idx)
        free_after = active_projected_grad1.abs() > 0
        if update_hessian_with_bfgs:
            next_state, bfgs_updated = self._bfgs_update_fd_newton_hessian(
                state=hessian_state,
                active_theta=active_theta1,
                active_grad=active_grad1,
                active_loss=active_loss1,
                accepted=accepted,
                free_before=free,
                free_after=free_after,
            )
            hessian_update = "bfgs"
        else:
            next_state = _FDNewtonHessianState(
                batch_index=hessian_state.batch_index,
                solver_stage=hessian_state.solver_stage,
                family_indices=hessian_state.family_indices,
                hessian=hessian_state.hessian.detach().clone(),
                active_theta=active_theta1.detach().clone(),
                active_grad=active_grad1.detach().clone(),
                active_loss=active_loss1.detach().clone(),
                updates_since_refresh=hessian_state.updates_since_refresh + 1,
            )
            bfgs_updated = torch.zeros_like(accepted)
            hessian_update = "fixed"
        bfgs_skipped = accepted & ~bfgs_updated
        if refreshed_hessian:
            refresh_grad_inf = metrics0.get("grad/inf")
            refresh_projected_inf = metrics0.get("grad/projected_inf")
            if refresh_grad_inf is not None:
                metrics["optimizer/fd_newton_refresh_grad_inf"] = refresh_grad_inf
            if refresh_projected_inf is not None:
                metrics["optimizer/fd_newton_refresh_projected_inf"] = (
                    refresh_projected_inf
                )
        metrics["grad/projected_inf"] = final_projected_grad_inf
        metrics["optimizer/fd_newton_grad_evals"] = float(grad_evals)
        metrics["optimizer/fd_newton_loss_evals"] = float(loss_evals)
        metrics["optimizer/fd_newton_line_search"] = bool(use_line_search)
        metrics["optimizer/fd_newton_post_step_loss_filter"] = bool(
            reject_loss_increases_after_step
        )
        metrics["optimizer/fd_newton_loss_rejected_rows"] = float(
            loss_rejected.sum().detach().cpu()
        )
        metrics["optimizer/fd_newton_max_ls"] = float(max_line_search_steps)
        metrics["optimizer/fd_newton_fallback_max_ls"] = float(
            min(max_line_search_steps, _FD_NEWTON_FALLBACK_LINE_SEARCH_MAX_STEPS)
            if use_line_search
            else 0
        )
        metrics["optimizer/fd_newton_accepted_rows"] = float(accepted.sum().detach().cpu())
        metrics["optimizer/fd_newton_accepted_fraction"] = float(
            accepted.to(dtype=torch.float32).mean().detach().cpu()
        )
        metrics["optimizer/fd_newton_fallback_rows"] = float(fallback.sum().detach().cpu())
        metrics["optimizer/fd_newton_line_search_fallback_attempted_rows"] = float(
            line_search_fallback_attempted.sum().detach().cpu()
        )
        metrics["optimizer/fd_newton_line_search_fallback_rows"] = float(
            line_search_fallback_accepted.sum().detach().cpu()
        )
        metrics["optimizer/fd_newton_hessian_source"] = (
            "finite_difference"
            if refreshed_hessian
            else ("bfgs_update" if update_hessian_with_bfgs else "fixed_hessian")
        )
        metrics["optimizer/fd_newton_hessian_update"] = hessian_update
        metrics["optimizer/fd_newton_hessian_refreshed"] = bool(refreshed_hessian)
        metrics["optimizer/fd_newton_hessian_updates_since_refresh"] = float(
            next_state.updates_since_refresh
        )
        metrics["optimizer/fd_newton_hessian_refresh_steps"] = float(
            hessian_refresh_steps
        )
        metrics["optimizer/fd_newton_bfgs_updated_rows"] = float(
            bfgs_updated.sum().detach().cpu()
        )
        metrics["optimizer/fd_newton_bfgs_skipped_rows"] = float(
            bfgs_skipped.sum().detach().cpu()
        )
        metrics["optimizer/fd_newton_baseline_projected_inf"] = projected_grad_inf
        metrics["optimizer/fd_newton_step_scale"] = float(step_scale)
        metrics["optimizer/fd_newton_raw_step_inf"] = raw_step_inf
        metrics["optimizer/fd_newton_bound_projected_step_inf"] = bounded_step_inf
        return loss_vec, metrics, grad_evals + loss_evals, next_state

    def _set_model_theta(self, model: GeneReconModel, theta: torch.Tensor) -> None:
        with torch.no_grad():
            model.theta.copy_(theta)

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
        except (RuntimeError, TypeError, ValueError) as exc:
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
        active_batch_index = 0
        active_batch_local_step = 0
        active_optimizer_batch_index: int | None = None
        batchwise_active_optimizer = (
            config.mode == "genewise"
            and config.optimizer in _BATCHWISE_ACTIVE_OPTIMIZERS
        )
        batchwise_batched_lbfgs = (
            config.mode == "genewise" and config.optimizer == "batched-lbfgs"
        )
        batchwise_fd_newton = (
            config.mode == "genewise" and config.optimizer == "adam-fd-newton"
        )
        batchwise_hessian_sgd = (
            config.mode == "genewise" and config.optimizer == "hessian-sgd"
        )
        adaptive_rebatch_enabled = bool(
            config.adaptive_rebatch
            and batchwise_active_optimizer
        )
        solver_warmup_enabled = self._uses_solver_warmup()
        active_solver_stage = "warmup" if solver_warmup_enabled else "full"
        batch_best_nll: float | None = None
        batch_best_step: int | None = None
        batch_final_loss_cache: torch.Tensor | None = None
        batch_final_grad_cache: torch.Tensor | None = None
        batch_final_cache_ready: torch.Tensor | None = None
        fd_newton_hessian_state: _FDNewtonHessianState | None = None
        hessian_sgd_line_search_active = False
        hessian_sgd_low_accept_steps = 0
        converged_family_mask: torch.Tensor | None = None
        adaptive_family_best_nll: torch.Tensor | None = None
        adaptive_family_stable_steps: torch.Tensor | None = None
        if adaptive_rebatch_enabled:
            converged_family_mask = torch.zeros(
                (int(model.n_families),),
                device=model.theta.device,
                dtype=torch.bool,
            )
            adaptive_family_best_nll = torch.full(
                (int(model.n_families),),
                math.inf,
                device=model.theta.device,
                dtype=model.theta.dtype,
            )
            adaptive_family_stable_steps = torch.zeros(
                (int(model.n_families),),
                device=model.theta.device,
                dtype=torch.long,
            )
        batch_plan_generation = 0
        active_batch_last_checked_converged_count = 0
        status = {"status": "running", "reason": "running"}
        final_row: dict[str, Any] = {}
        resume_info: dict[str, Any] = {}
        resume_payload: dict[str, Any] | None = None
        best_checkpoint = config.out_dir / "checkpoints" / "best.pt"
        latest_checkpoint = config.out_dir / "checkpoints" / "latest.pt"
        sampling_checkpoint: Path | None = None

        def _adaptive_family_indices(
            mask: torch.Tensor,
            *,
            converged: bool,
        ) -> list[int]:
            selected = mask if converged else ~mask
            return [
                int(index)
                for index in torch.nonzero(
                    selected,
                    as_tuple=False,
                ).flatten().detach().cpu().tolist()
            ]

        def _adaptive_remaining_current_plan_indices(mask: torch.Tensor) -> list[int]:
            plan_indices: list[int] = []
            for metadata in model.batch_metadata[active_batch_index:]:
                plan_indices.extend(int(index) for index in metadata.family_indices)
            if not plan_indices:
                return []
            idx = torch.as_tensor(
                plan_indices,
                dtype=torch.long,
                device=model.theta.device,
            )
            keep = ~mask.index_select(0, idx)
            if not bool(keep.any().detach().cpu()):
                return []
            return [
                int(index)
                for index in idx.index_select(
                    0,
                    torch.nonzero(keep, as_tuple=False).flatten(),
                ).detach().cpu().tolist()
            ]

        def _adaptive_mask_with_prior_plan_families(mask: torch.Tensor) -> torch.Tensor:
            if active_batch_index <= 0:
                return mask
            prior_indices: list[int] = []
            for metadata in model.batch_metadata[:active_batch_index]:
                prior_indices.extend(int(index) for index in metadata.family_indices)
            if not prior_indices:
                return mask
            out = mask.clone()
            out.index_fill_(
                0,
                torch.as_tensor(
                    prior_indices,
                    dtype=torch.long,
                    device=model.theta.device,
                ),
                True,
            )
            return out

        def _adaptive_checkpoint_status(base: dict[str, Any]) -> dict[str, Any]:
            if not adaptive_rebatch_enabled or converged_family_mask is None:
                return base
            enriched = dict(base)
            enriched["converged_family_indices"] = _adaptive_family_indices(
                converged_family_mask,
                converged=True,
            )
            enriched["batch_plan_generation"] = batch_plan_generation
            return enriched

        try:
            if config.resume_from is not None:
                resume_payload = load_checkpoint(
                    config.resume_from,
                    map_location="cpu",
                )
                validate_checkpoint_model_compatibility(
                    path=config.resume_from,
                    config=config,
                    model=model,
                    payload=resume_payload,
                )
                resume_state = _resume_state_from_payload(
                    config.resume_from,
                    resume_payload,
                )
                _validate_resume_progress(
                    config.resume_from,
                    resume_state,
                    configured_steps=config.steps,
                )
                restore_model_theta(model, resume_payload)
                start_step = resume_state.start_step
                best_nll = resume_state.best_nll
                best_step = resume_state.best_step
                previous_objective = resume_state.previous_objective
                stable_loss_steps = resume_state.stable_loss_steps
                active_batch_index = resume_state.active_batch_index
                active_solver_stage = resume_state.active_solver_stage
                active_batch_local_step = resume_state.active_batch_local_step
                batch_plan_generation = resume_state.batch_plan_generation
                if adaptive_rebatch_enabled and converged_family_mask is not None:
                    if resume_state.converged_family_indices:
                        max_index = max(resume_state.converged_family_indices)
                        if max_index >= int(model.n_families):
                            raise RuntimeError(
                                f"checkpoint {config.resume_from} has "
                                "out-of-range converged family indices"
                            )
                        converged_family_mask.index_fill_(
                            0,
                            torch.as_tensor(
                                resume_state.converged_family_indices,
                                dtype=torch.long,
                                device=model.theta.device,
                            ),
                            True,
                        )
                    if batch_plan_generation > 0:
                        remaining_indices = _adaptive_family_indices(
                            converged_family_mask,
                            converged=False,
                        )
                        if remaining_indices:
                            model.replan_resident_batches(remaining_indices)
                if active_solver_stage not in {"warmup", "full"}:
                    raise RuntimeError(
                        f"checkpoint {config.resume_from} has invalid active_solver_stage"
                    )
                if active_solver_stage == "warmup" and not solver_warmup_enabled:
                    active_solver_stage = "full"
                if batchwise_active_optimizer:
                    batch_best_nll = best_nll
                    batch_best_step = best_step
                    best_nll = None
                    best_step = None

            if batchwise_active_optimizer:
                if active_batch_index >= len(model.batch_metadata):
                    raise RuntimeError(
                        f"checkpoint active batch {active_batch_index} exceeds "
                        f"{len(model.batch_metadata)} model batches"
                    )
                batch_final_loss_cache = torch.empty(
                    (int(model.n_families),),
                    device=model.theta.device,
                    dtype=model.theta.dtype,
                )
                batch_final_grad_cache = torch.empty_like(model.theta)
                batch_final_cache_ready = torch.zeros(
                    (int(model.n_families),),
                    device=model.theta.device,
                    dtype=torch.bool,
                )
                if model.current_batch_index != active_batch_index:
                    _drop_cached_static_states_if_needed(model)
                model.select_batch(active_batch_index)
                self._configure_active_solver_stage(
                    model,
                    active_solver_stage,
                )

            current_phase = self._phase_for_step(start_step)
            optimizer = self._make_optimizer(model, current_phase)
            if (
                current_phase in _BATCHWISE_ACTIVE_OPTIMIZERS
                and batchwise_active_optimizer
            ):
                active_optimizer_batch_index = active_batch_index
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
                if batchwise_batched_lbfgs and phase == "batched-lbfgs":
                    if model.current_batch_index != active_batch_index:
                        _drop_cached_static_states_if_needed(model)
                        model.select_batch(active_batch_index)
                    if (
                        optimizer is None
                        or phase != current_phase
                        or active_optimizer_batch_index != active_batch_index
                    ):
                        current_phase = phase
                        optimizer = self._make_optimizer(model, phase)
                        active_optimizer_batch_index = active_batch_index
                elif (
                    (batchwise_fd_newton or batchwise_hessian_sgd)
                    and phase in _HESSIAN_CONDITIONED_OPTIMIZERS
                ):
                    if model.current_batch_index != active_batch_index:
                        _drop_cached_static_states_if_needed(model)
                        model.select_batch(active_batch_index)
                    if (
                        optimizer is None
                        or phase != current_phase
                        or active_optimizer_batch_index != active_batch_index
                    ):
                        current_phase = phase
                        optimizer = self._make_optimizer(model, phase)
                        active_optimizer_batch_index = active_batch_index
                elif optimizer is None or phase != current_phase:
                    current_phase = phase
                    optimizer = self._make_optimizer(model, phase)
                    active_optimizer_batch_index = None

                t0 = time.perf_counter()
                theta_before = model.theta.detach().clone()
                closure_evals = 0
                batched_grad_evals = 0
                batched_loss_evals = 0
                metrics: dict[str, Any] = {}
                adaptive_rebatch_pending_indices: list[int] | None = None
                adaptive_rebatch_stop = False

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

                def batched_closure() -> torch.Tensor:
                    nonlocal batched_grad_evals, metrics
                    batched_grad_evals += 1
                    with torch.no_grad():
                        model.clamp_theta_(config.min_rate, config.max_rate)
                    if optimizer is None:
                        raise RuntimeError("missing optimizer")
                    optimizer.zero_grad(set_to_none=True)
                    if batchwise_active_optimizer:
                        loss_vec_i, metrics_i = (
                            self._evaluate_active_genewise_vector_and_grad(
                                model,
                                solver_stage=active_solver_stage,
                            )
                        )
                    else:
                        loss_vec_i, metrics_i = self._evaluate_genewise_vector_and_grad(
                            model
                        )
                    metrics = metrics_i
                    return loss_vec_i

                def batched_loss_closure() -> torch.Tensor:
                    nonlocal batched_loss_evals
                    batched_loss_evals += 1
                    with torch.no_grad():
                        model.clamp_theta_(config.min_rate, config.max_rate)
                    if batchwise_active_optimizer:
                        return self._evaluate_active_genewise_loss_vector(model)
                    return self._evaluate_genewise_loss_vector(model)

                save_best_after_row = False
                first_order_pending_step = False
                eval_position = (
                    "post_step"
                    if phase in _POST_STEP_OPTIMIZERS
                    else "pre_step"
                )
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
                    loss_current, metrics = self._evaluate_and_backward(model)
                    closure_evals += 1
                    if (
                        not torch.isfinite(loss_current).item()
                        or not _is_finite_tensor(model.theta.grad)
                    ):
                        status = {
                            "status": "failed",
                            "reason": "nonfinite_objective_or_gradient",
                        }
                        model.clear()
                        break
                    model.clear()
                elif phase == "batched-lbfgs":
                    try:
                        optimizer.step(
                            batched_closure,
                            loss_closure=batched_loss_closure,
                        )
                    except RuntimeError:
                        status = {
                            "status": "failed",
                            "reason": "batched_lbfgs_runtime_error",
                        }
                        break
                    opt_state = optimizer.state.get(model.theta, {})
                    closure_evals = batched_grad_evals + batched_loss_evals
                    with torch.no_grad():
                        model.clamp_theta_(config.min_rate, config.max_rate)
                    theta_step = float((model.theta.detach() - theta_before).abs().amax().cpu())
                    model.clear()
                    reused_optimizer_gradient = False
                    loss_vec_current = opt_state.get("last_loss")
                    grad_current = opt_state.get("last_grad")
                    if (
                        config.lbfgs_line_search == "none"
                        and torch.is_tensor(loss_vec_current)
                        and torch.is_tensor(grad_current)
                        and loss_vec_current.numel() == int(model.n_families)
                        and grad_current.numel() == model.theta.numel()
                    ):
                        model.theta.grad = grad_current.detach().reshape_as(model.theta).to(
                            device=model.theta.device,
                            dtype=model.theta.dtype,
                        )
                        loss_vec_current = loss_vec_current.detach().to(
                            device=model.theta.device,
                            dtype=model.theta.dtype,
                        ).reshape(int(model.n_families))
                        metrics = dict(metrics)
                        metrics["likelihood/data_nll_bits"] = float(
                            loss_vec_current.sum().detach().cpu()
                        )
                        metrics["likelihood/log_likelihood_bits"] = float(
                            -loss_vec_current.sum().detach().cpu()
                        )
                        metrics.update(tensor_stats("grad", model.theta.grad))
                        metrics.update(parameter_stats(model.theta))
                        reused_optimizer_gradient = True
                    else:
                        model.theta.grad = None
                        if batchwise_batched_lbfgs:
                            loss_vec_current, metrics = (
                                self._evaluate_active_genewise_vector_and_grad(
                                    model,
                                    solver_stage=active_solver_stage,
                                )
                            )
                        else:
                            loss_vec_current, metrics = self._evaluate_genewise_vector_and_grad(
                                model
                            )
                        closure_evals += 1
                    metrics["optimizer/batched_lbfgs_grad_evals"] = float(batched_grad_evals)
                    metrics["optimizer/batched_lbfgs_loss_evals"] = float(batched_loss_evals)
                    metrics["optimizer/batched_lbfgs_reused_gradient"] = (
                        reused_optimizer_gradient
                    )
                    metrics["optimizer/batched_lbfgs_inner_iters"] = float(
                        int(opt_state.get("last_n_iter", 0))
                    )
                    accepted = opt_state.get("last_accepted")
                    if torch.is_tensor(accepted):
                        accepted_f = accepted.detach().to(dtype=torch.float32)
                        metrics["optimizer/batched_lbfgs_accepted_rows"] = float(
                            accepted_f.sum().cpu()
                        )
                        metrics["optimizer/batched_lbfgs_accepted_fraction"] = float(
                            accepted_f.mean().cpu()
                        )
                    alpha = opt_state.get("last_alpha")
                    if torch.is_tensor(alpha):
                        alpha_cpu = alpha.detach().cpu()
                        metrics["optimizer/batched_lbfgs_alpha_mean"] = float(
                            alpha_cpu.mean()
                        )
                        metrics["optimizer/batched_lbfgs_alpha_max"] = float(
                            alpha_cpu.max()
                        )
                    if (
                        not bool(torch.isfinite(loss_vec_current).all().item())
                        or not _is_finite_tensor(model.theta.grad)
                    ):
                        status = {
                            "status": "failed",
                            "reason": "nonfinite_objective_or_gradient",
                        }
                        model.clear()
                        break
                    if (
                        batchwise_batched_lbfgs
                        and active_solver_stage == "full"
                        and batch_final_loss_cache is not None
                        and batch_final_grad_cache is not None
                        and batch_final_cache_ready is not None
                    ):
                        idx = self._active_batch_indices(model)
                        batch_final_loss_cache.index_copy_(
                            0,
                            idx,
                            loss_vec_current.detach().index_select(0, idx),
                        )
                        batch_final_grad_cache.index_copy_(
                            0,
                            idx,
                            model.theta.grad.detach().index_select(0, idx),
                        )
                        batch_final_cache_ready.index_fill_(0, idx, True)
                    model.clear()
                elif phase in _HESSIAN_CONDITIONED_OPTIMIZERS:
                    if optimizer is None:
                        raise RuntimeError("missing optimizer")
                    if (
                        phase == "adam-fd-newton"
                        and active_batch_local_step < config.fd_adam_warmup_steps
                    ):
                        fd_newton_hessian_state = None
                        loss_vec_current, metrics, closure_evals = self._active_adam_step(
                            model,
                            optimizer,
                            solver_stage=active_solver_stage,
                        )
                        metrics["optimizer/fd_newton_subphase"] = "adam_warmup"
                    else:
                        hessian_refresh_steps = config.fd_hessian_refresh_steps
                        active_clade_count = int(
                            getattr(
                                model.current_batch_metadata,
                                "clade_count",
                                0,
                            )
                            or 0
                        )
                        if (
                            phase == "hessian-sgd"
                            and not hessian_sgd_line_search_active
                            and active_clade_count
                            >= _HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
                        ):
                            hessian_refresh_steps = max(
                                hessian_refresh_steps,
                                _HESSIAN_SGD_NO_LINE_REFRESH_STEPS,
                            )
                        (
                            loss_vec_current,
                            metrics,
                            closure_evals,
                            fd_newton_hessian_state,
                        ) = (
                            self._active_fd_newton_step(
                                model,
                                solver_stage=active_solver_stage,
                                hessian_state=fd_newton_hessian_state,
                                update_hessian_with_bfgs=phase
                                in {"adam-fd-newton", "hessian-sgd"},
                                step_scale=(
                                    1.0
                                    if phase == "adam-fd-newton"
                                    else config.lr
                                ),
                                use_line_search=(
                                    phase == "adam-fd-newton"
                                    or (
                                        phase == "hessian-sgd"
                                        and hessian_sgd_line_search_active
                                    )
                                ),
                                reject_loss_increases_after_step=(
                                    phase == "hessian-sgd"
                                    and not hessian_sgd_line_search_active
                                ),
                                hessian_refresh_steps=hessian_refresh_steps,
                                line_search_max_steps=(
                                    _HESSIAN_SGD_LINE_SEARCH_MAX_STEPS
                                    if (
                                        phase == "hessian-sgd"
                                        and hessian_sgd_line_search_active
                                    )
                                    else None
                                ),
                            )
                        )
                        metrics["optimizer/fd_newton_subphase"] = (
                            "fd_newton"
                            if phase == "adam-fd-newton"
                            else "hessian_sgd"
                        )
                    theta_step = float(
                        (model.theta.detach() - theta_before).abs().amax().cpu()
                    )
                    if (
                        not bool(torch.isfinite(loss_vec_current).all().item())
                        or not _is_finite_tensor(model.theta.grad)
                    ):
                        status = {
                            "status": "failed",
                            "reason": "nonfinite_objective_or_gradient",
                        }
                        model.clear()
                        break
                    if (
                        self._active_batch_result_is_canonical_full_solver(
                            phase=phase,
                            solver_stage=active_solver_stage,
                        )
                        and batch_final_loss_cache is not None
                        and batch_final_grad_cache is not None
                        and batch_final_cache_ready is not None
                    ):
                        idx = self._active_batch_indices(model)
                        batch_final_loss_cache.index_copy_(
                            0,
                            idx,
                            loss_vec_current.detach().index_select(0, idx),
                        )
                        batch_final_grad_cache.index_copy_(
                            0,
                            idx,
                            model.theta.grad.detach().index_select(0, idx),
                        )
                        batch_final_cache_ready.index_fill_(0, idx, True)
                    active_batch_local_step += 1
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
                    first_order_pending_step = True

                if adaptive_rebatch_enabled and phase in _BATCHWISE_ACTIVE_OPTIMIZERS:
                    metrics = dict(metrics)
                    metrics["optimizer/adaptive_rebatch_enabled"] = True
                    metrics["optimizer/rebatch_generation"] = float(
                        batch_plan_generation
                    )
                    metrics["optimizer/rebatch_triggered"] = False
                    idx = self._active_batch_indices(model)
                    batch_family_count = int(idx.numel())
                    metrics["optimizer/rebatch_active_family_count"] = float(
                        batch_family_count
                    )
                    active_batch_large_enough = (
                        batch_family_count >= _ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES
                    )
                    should_check_rebatch = (
                        active_solver_stage == "full"
                        and active_batch_large_enough
                        and (step + 1) % config.adaptive_rebatch_check_interval == 0
                    )
                    metrics["optimizer/rebatch_checked"] = should_check_rebatch
                    if not active_batch_large_enough:
                        metrics["optimizer/rebatch_reason"] = "small_active_batch"
                    if (
                        should_check_rebatch
                        and converged_family_mask is not None
                        and model.theta.grad is not None
                        and adaptive_family_best_nll is not None
                        and adaptive_family_stable_steps is not None
                    ):
                        active_loss = loss_vec_current.detach().index_select(0, idx)
                        active_best = adaptive_family_best_nll.index_select(0, idx)
                        active_stable = adaptive_family_stable_steps.index_select(
                            0,
                            idx,
                        )
                        finite_loss = torch.isfinite(active_loss)
                        improved_family = finite_loss & (
                            active_loss < active_best - config.best_likelihood_min_delta
                        )
                        next_best = torch.where(
                            improved_family,
                            active_loss,
                            active_best,
                        )
                        next_stable = torch.where(
                            improved_family,
                            torch.zeros_like(active_stable),
                            active_stable + 1,
                        )
                        adaptive_family_best_nll.index_copy_(0, idx, next_best)
                        adaptive_family_stable_steps.index_copy_(
                            0,
                            idx,
                            next_stable,
                        )
                        family_patience = _active_batch_patience(
                            config.best_likelihood_patience
                        )
                        if family_patience > 0:
                            row_converged = next_stable >= family_patience
                        else:
                            row_converged = torch.zeros_like(
                                next_stable,
                                dtype=torch.bool,
                            )
                        row_converged = (
                            row_converged
                            & ~converged_family_mask.index_select(0, idx)
                        )
                        threshold_count = max(
                            1,
                            math.ceil(
                                config.adaptive_rebatch_fraction * batch_family_count
                            ),
                        )
                        converged_count = int(row_converged.sum().detach().cpu())
                        crossed_threshold = (
                            active_batch_last_checked_converged_count
                            < threshold_count
                            <= converged_count
                        )
                        active_batch_last_checked_converged_count = converged_count
                        metrics.update(
                            {
                                "optimizer/rebatch_active_converged_families": float(
                                    converged_count
                                ),
                                "optimizer/rebatch_convergence_criterion": (
                                    "best_likelihood_patience"
                                ),
                                "optimizer/rebatch_family_stable_steps_max": float(
                                    next_stable.max().detach().cpu()
                                ),
                                "optimizer/rebatch_threshold_families": float(
                                    threshold_count
                                ),
                            }
                        )
                        if crossed_threshold:
                            candidate_mask = converged_family_mask.clone()
                            if bool(row_converged.any().detach().cpu()):
                                candidate_mask.index_fill_(
                                    0,
                                    idx.index_select(
                                        0,
                                        torch.nonzero(
                                            row_converged,
                                            as_tuple=False,
                                        ).flatten(),
                                    ),
                                    True,
                                )
                            candidate_mask = _adaptive_mask_with_prior_plan_families(
                                candidate_mask
                            )
                            remaining_indices = _adaptive_remaining_current_plan_indices(
                                candidate_mask
                            )
                            remaining_count = len(remaining_indices)
                            metrics["optimizer/rebatch_remaining_families"] = float(
                                remaining_count
                            )
                            if remaining_count == 0:
                                converged_family_mask.copy_(candidate_mask)
                                adaptive_rebatch_stop = True
                                metrics["optimizer/rebatch_reason"] = "all_converged"
                            elif (
                                remaining_count
                                >= config.adaptive_rebatch_min_remaining_families
                            ):
                                converged_family_mask.copy_(candidate_mask)
                                adaptive_rebatch_pending_indices = remaining_indices
                                metrics["optimizer/rebatch_triggered"] = True
                            else:
                                metrics["optimizer/rebatch_reason"] = (
                                    "below_min_remaining"
                                )

                active_objective_scope = (
                    batchwise_active_optimizer
                    and phase in _BATCHWISE_ACTIVE_OPTIMIZERS
                )
                active_family_count = (
                    max(1, int(metrics.get("optimizer/batch_family_count", 1)))
                    if active_objective_scope
                    else 1
                )
                loss_change_tol_bits = config.loss_change_tol * active_family_count
                best_likelihood_min_delta_bits = (
                    config.best_likelihood_min_delta * active_family_count
                )
                objective = float(metrics["likelihood/data_nll_bits"])
                delta = None if previous_objective is None else previous_objective - objective
                if delta is not None and delta <= loss_change_tol_bits:
                    stable_loss_steps += 1
                else:
                    stable_loss_steps = 0
                previous_objective = objective

                if active_objective_scope:
                    improved = (
                        batch_best_nll is None
                        or objective < batch_best_nll - best_likelihood_min_delta_bits
                    )
                    if improved:
                        batch_best_nll = objective
                        batch_best_step = step
                    row_best_nll = batch_best_nll
                    row_best_step = batch_best_step
                else:
                    improved = (
                        best_nll is None
                        or objective < best_nll - best_likelihood_min_delta_bits
                    )
                    if improved:
                        best_nll = objective
                        best_step = step
                        save_best_after_row = True
                    row_best_nll = best_nll
                    row_best_step = best_step

                row = {
                    "step": step,
                    "optimizer/phase": phase,
                    "optimizer/eval_position": eval_position,
                    "closure_evals": closure_evals,
                    "theta_step_inf": theta_step,
                    "delta_likelihood_bits": delta,
                    "loss_change_tol_bits": loss_change_tol_bits,
                    "best_likelihood_min_delta_bits": best_likelihood_min_delta_bits,
                    "stable_loss_steps": stable_loss_steps,
                    "best_nll_bits": row_best_nll,
                    "best_step": row_best_step,
                    **resume_info,
                    "step_s": time.perf_counter() - t0,
                    **metrics,
                }
                checkpoint_status = {
                    "status": "running",
                    "reason": "running",
                    **resume_info,
                    "best_nll_bits": row_best_nll,
                    "best_step": row_best_step,
                    "previous_objective": previous_objective,
                    "stable_loss_steps": stable_loss_steps,
                }
                if active_objective_scope:
                    checkpoint_status["active_batch_index"] = active_batch_index
                    checkpoint_status["active_solver_stage"] = active_solver_stage
                    checkpoint_status["active_batch_local_step"] = active_batch_local_step
                if save_best_after_row and phase not in _POST_STEP_OPTIMIZERS:
                    best_row = dict(row)
                    best_row["optimizer/step_applied"] = False
                    best_row["step_s"] = time.perf_counter() - t0
                    self._save_status(
                        best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        next_step=step,
                        status=_adaptive_checkpoint_status(checkpoint_status),
                        row=best_row,
                        optimizer_phase=phase,
                    )
                    sampling_checkpoint = best_checkpoint
                    save_best_after_row = False

                if first_order_pending_step:
                    optimizer.step()
                    with torch.no_grad():
                        model.clamp_theta_(config.min_rate, config.max_rate)
                    theta_step = float(
                        (model.theta.detach() - theta_before).abs().amax().cpu()
                    )
                model.clear()
                row["theta_step_inf"] = theta_step
                row["optimizer/step_applied"] = bool(
                    first_order_pending_step
                    or phase in _POST_STEP_OPTIMIZERS
                )
                row["step_s"] = time.perf_counter() - t0

                final_row = row
                self._record(row)

                if adaptive_rebatch_stop:
                    status = {"status": "converged", "reason": "best_likelihood_patience"}
                    break
                if adaptive_rebatch_pending_indices is not None:
                    _drop_cached_static_states_if_needed(model)
                    model.replan_resident_batches(adaptive_rebatch_pending_indices)
                    batch_plan_generation += 1
                    active_batch_index = 0
                    active_solver_stage = "full"
                    active_batch_local_step = config.fd_adam_warmup_steps
                    fd_newton_hessian_state = None
                    hessian_sgd_line_search_active = False
                    hessian_sgd_low_accept_steps = 0
                    previous_objective = None
                    stable_loss_steps = 0
                    batch_best_nll = None
                    batch_best_step = None
                    optimizer = None
                    active_optimizer_batch_index = None
                    active_batch_last_checked_converged_count = 0
                    if batch_final_cache_ready is not None:
                        batch_final_cache_ready.index_fill_(
                            0,
                            torch.as_tensor(
                                adaptive_rebatch_pending_indices,
                                dtype=torch.long,
                                device=batch_final_cache_ready.device,
                            ),
                            False,
                        )
                    self._configure_active_solver_stage(
                        model,
                        active_solver_stage,
                    )
                    if config.checkpoint_every:
                        transition_status = {
                            **checkpoint_status,
                            "active_batch_index": active_batch_index,
                            "active_solver_stage": active_solver_stage,
                            "active_batch_local_step": active_batch_local_step,
                            "previous_objective": None,
                            "stable_loss_steps": 0,
                            "best_nll_bits": None,
                            "best_step": None,
                        }
                        self._save_status(
                            latest_checkpoint,
                            model=model,
                            optimizer=None,
                            step=step,
                            next_step=step + 1,
                            status=_adaptive_checkpoint_status(transition_status),
                            row=row,
                            optimizer_phase=phase,
                        )
                    resume_info = {}
                    continue

                if save_best_after_row:
                    self._save_status(
                        best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        status=_adaptive_checkpoint_status(checkpoint_status),
                        row=row,
                        optimizer_phase=phase,
                    )
                    sampling_checkpoint = best_checkpoint
                if config.checkpoint_every and (step + 1) % config.checkpoint_every == 0:
                    self._save_status(
                        latest_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        status=_adaptive_checkpoint_status(checkpoint_status),
                        row=row,
                        optimizer_phase=phase,
                    )

                if step % config.log_every == 0:
                    print(
                        f"step={step} phase={phase} "
                        f"solver={row.get('optimizer/solver_stage', 'full')} "
                        f"nll_bits={objective:.6f} "
                        f"grad_inf={row.get('grad/inf', float('nan')):.6g} "
                        f"delta={float('nan') if delta is None else delta:.6g} "
                        f"best={float('nan') if row_best_nll is None else row_best_nll:.6f} "
                        f"step_s={row['step_s']:.3f}",
                        flush=True,
                    )

                step_status = _step_stopping_status(
                    config,
                    step=step,
                    stable_loss_steps=stable_loss_steps,
                    best_step=row_best_step,
                    loss_patience=(
                        _active_batch_patience(config.loss_patience)
                        if active_objective_scope
                        else None
                    ),
                    best_likelihood_patience=(
                        _active_batch_patience(config.best_likelihood_patience)
                        if active_objective_scope
                        else None
                    ),
                )
                full_stage_plateau = (
                    step_status is not None
                    and active_objective_scope
                    and active_solver_stage == "full"
                )
                hessian_sgd_activate_line_search = False
                if (
                    batchwise_hessian_sgd
                    and phase == "hessian-sgd"
                    and active_objective_scope
                    and not hessian_sgd_line_search_active
                    and not full_stage_plateau
                ):
                    accepted_fraction = metrics.get(
                        "optimizer/fd_newton_accepted_fraction"
                    )
                    loss_rejected_rows = metrics.get(
                        "optimizer/fd_newton_loss_rejected_rows",
                        0.0,
                    )
                    low_acceptance = (
                        accepted_fraction is not None
                        and float(accepted_fraction)
                        < _HESSIAN_SGD_LINE_SEARCH_ACCEPT_FRACTION
                        and float(loss_rejected_rows) > 0.0
                    )
                    if low_acceptance:
                        hessian_sgd_low_accept_steps += 1
                    else:
                        hessian_sgd_low_accept_steps = 0
                    hessian_sgd_activate_line_search = (
                        hessian_sgd_low_accept_steps
                        >= _HESSIAN_SGD_LINE_SEARCH_LOW_ACCEPT_PATIENCE
                    )
                    active_clade_count = int(
                        getattr(
                            model.current_batch_metadata,
                            "clade_count",
                            0,
                        )
                        or 0
                    )
                    if (
                        hessian_sgd_activate_line_search
                        and active_solver_stage == "full"
                        and stable_loss_steps > 0
                        and active_clade_count
                        >= _HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
                    ):
                        hessian_sgd_activate_line_search = False
                if hessian_sgd_activate_line_search:
                    hessian_sgd_line_search_active = True
                    hessian_sgd_low_accept_steps = 0
                    fd_newton_hessian_state = None
                    previous_objective = None
                    stable_loss_steps = 0
                    batch_best_nll = None
                    batch_best_step = None
                    optimizer = None
                    active_optimizer_batch_index = None
                    resume_info = {}
                    continue

                warmup_switch = (
                    active_objective_scope
                    and active_solver_stage == "warmup"
                    and self._should_switch_solver_warmup(
                        stable_loss_steps=stable_loss_steps,
                    )
                )
                if (
                    step_status is not None
                    and active_objective_scope
                    and active_solver_stage == "warmup"
                ):
                    warmup_switch = True
                    step_status = None
                if warmup_switch:
                    active_clade_count = int(
                        getattr(
                            model.current_batch_metadata,
                            "clade_count",
                            0,
                        )
                        or 0
                    )
                    carry_warmup_hessian = (
                        batchwise_hessian_sgd
                        and phase == "hessian-sgd"
                        and not hessian_sgd_line_search_active
                        and active_clade_count
                        >= _HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES
                        and fd_newton_hessian_state is not None
                    )
                    warmup_hessian_state = fd_newton_hessian_state
                    active_solver_stage = "full"
                    active_batch_local_step = 0
                    fd_newton_hessian_state = None
                    hessian_sgd_line_search_active = False
                    hessian_sgd_low_accept_steps = 0
                    previous_objective = None
                    stable_loss_steps = 0
                    batch_best_nll = None
                    batch_best_step = None
                    optimizer = None
                    active_optimizer_batch_index = None
                    active_batch_last_checked_converged_count = 0
                    self._configure_active_solver_stage(
                        model,
                        active_solver_stage,
                    )
                    if carry_warmup_hessian and warmup_hessian_state is not None:
                        loss_vec_current, _grad, _metrics = (
                            self._evaluate_active_genewise_vector_grad_at_current_theta(
                                model,
                                solver_stage=active_solver_stage,
                            )
                        )
                        idx = self._active_batch_indices(model)
                        fd_newton_hessian_state = _FDNewtonHessianState(
                            batch_index=int(model.current_batch_index),
                            solver_stage=active_solver_stage,
                            family_indices=tuple(
                                int(index)
                                for index in model.current_batch_metadata.family_indices
                            ),
                            hessian=warmup_hessian_state.hessian.detach().clone(),
                            active_theta=model.theta.detach().index_select(0, idx).clone(),
                            active_grad=model.theta.grad.detach().index_select(0, idx).clone(),
                            active_loss=loss_vec_current.detach().index_select(0, idx).clone(),
                            updates_since_refresh=warmup_hessian_state.updates_since_refresh,
                        )
                        model.clear()
                    if config.checkpoint_every:
                        transition_status = {
                            **checkpoint_status,
                            "active_batch_index": active_batch_index,
                            "active_solver_stage": active_solver_stage,
                            "active_batch_local_step": active_batch_local_step,
                            "previous_objective": None,
                            "stable_loss_steps": 0,
                            "best_nll_bits": None,
                            "best_step": None,
                        }
                        self._save_status(
                            latest_checkpoint,
                            model=model,
                            optimizer=None,
                            step=step,
                            next_step=step + 1,
                            status=_adaptive_checkpoint_status(transition_status),
                            row=row,
                            optimizer_phase=phase,
                        )
                    resume_info = {}
                    continue
                if step_status is not None:
                    if (
                        active_objective_scope
                        and active_batch_index + 1 < len(model.batch_metadata)
                    ):
                        active_batch_index += 1
                        active_solver_stage = "warmup" if solver_warmup_enabled else "full"
                        active_batch_local_step = 0
                        fd_newton_hessian_state = None
                        hessian_sgd_line_search_active = False
                        hessian_sgd_low_accept_steps = 0
                        previous_objective = None
                        stable_loss_steps = 0
                        batch_best_nll = None
                        batch_best_step = None
                        optimizer = None
                        active_optimizer_batch_index = None
                        active_batch_last_checked_converged_count = 0
                        if config.checkpoint_every:
                            transition_status = {
                                **checkpoint_status,
                                "active_batch_index": active_batch_index,
                                "active_solver_stage": active_solver_stage,
                                "active_batch_local_step": active_batch_local_step,
                                "previous_objective": None,
                                "stable_loss_steps": 0,
                                "best_nll_bits": None,
                                "best_step": None,
                            }
                            self._save_status(
                                latest_checkpoint,
                                model=model,
                                optimizer=None,
                                step=step,
                                next_step=step + 1,
                                status=_adaptive_checkpoint_status(transition_status),
                                row=row,
                                optimizer_phase=phase,
                            )
                        _drop_cached_static_states_if_needed(model)
                        model.select_batch(active_batch_index)
                        self._configure_active_solver_stage(
                            model,
                            active_solver_stage,
                        )
                        resume_info = {}
                        continue
                    status = step_status
                    break
            else:
                status = {"status": "not_converged", "reason": "max_steps"}

            if batchwise_active_optimizer:
                self._configure_solver_stage(model, "full")
            model.theta.grad = None
            final_per_family_nll: torch.Tensor | None = None
            final_closure_evals = 1
            if (
                batchwise_active_optimizer
                and batch_final_loss_cache is not None
                and batch_final_grad_cache is not None
                and batch_final_cache_ready is not None
                and bool(batch_final_cache_ready.all().item())
            ):
                final_per_family_nll = batch_final_loss_cache.detach().clone()
                model.theta.grad = batch_final_grad_cache.detach().clone()
                final_loss = final_per_family_nll.sum()
                final_metrics = {
                    "likelihood/data_nll_bits": float(final_loss.detach().cpu()),
                    "likelihood/log_likelihood_bits": float(-final_loss.detach().cpu()),
                    "optimizer/final_eval_source": "cached_active_batches",
                }
                final_metrics.update(tensor_stats("grad", model.theta.grad))
                final_metrics.update(parameter_stats(model.theta))
                final_metrics.update(solver_stats(model))
                final_closure_evals = 0
            elif config.mode == "genewise" and callable(
                getattr(model, "full_genewise_nll_and_grad", None)
            ):
                final_loss_vec, final_metrics = (
                    self._evaluate_genewise_vector_and_grad_with_memory_fallback(
                        model
                    )
                )
                final_loss = final_loss_vec.sum()
                final_per_family_nll = final_loss_vec.detach()
            else:
                final_loss, final_metrics = self._evaluate_and_backward(model)
            final_step = max(start_step, min(config.steps, int(final_row.get("step", -1)) + 1))
            final_eval_failed = (
                not torch.isfinite(final_loss).item()
                or not _is_finite_tensor(model.theta.grad)
            )
            if final_eval_failed:
                status = {
                    "status": "failed",
                    "reason": "nonfinite_objective_or_gradient",
                }
                final_improved = False
                final_nll_bits = (
                    math.nan
                    if previous_objective is None
                    else float(previous_objective)
                )
                final_grad_inf = math.inf
                final_row = {
                    "step": final_step,
                    "optimizer/phase": "final_eval",
                    "optimizer/eval_position": "final",
                    "optimizer/step_applied": False,
                    "optimizer/final_eval_status": "failed",
                    "optimizer/final_eval_reason": (
                        "nonfinite_objective_or_gradient"
                    ),
                    "closure_evals": final_closure_evals,
                    "theta_step_inf": 0.0,
                    "delta_likelihood_bits": None,
                    "stable_loss_steps": stable_loss_steps,
                    "best_nll_bits": best_nll,
                    "best_step": best_step,
                    **resume_info,
                    "step_s": 0.0,
                }
                model.theta.grad = None
                model.clear()
            else:
                final_metrics.update(
                    self._evaluate_final_iteration_check(
                        model,
                        baseline_loss=final_loss,
                        baseline_grad=model.theta.grad,
                    )
                )
                final_nll_bits = float(final_loss.detach().cpu())
                final_grad_inf = float(final_metrics.get("grad/inf", math.inf))
                final_improved = (
                    best_nll is None
                    or final_nll_bits < best_nll - config.best_likelihood_min_delta
                )
                if final_improved:
                    best_nll = final_nll_bits
                    best_step = final_step
                final_row = {
                    "step": final_step,
                    "optimizer/phase": "final_eval",
                    "optimizer/eval_position": "final",
                    "optimizer/step_applied": False,
                    "closure_evals": final_closure_evals,
                    "theta_step_inf": 0.0,
                    "delta_likelihood_bits": None,
                    "stable_loss_steps": stable_loss_steps,
                    "best_nll_bits": best_nll,
                    "best_step": best_step,
                    **resume_info,
                    "step_s": 0.0,
                    **final_metrics,
                }
            self.history.append(final_row)

            final_status = {
                **status,
                **resume_info,
                "elapsed_s": time.perf_counter() - started,
                "best_nll_bits": best_nll,
                "best_step": best_step,
                "previous_objective": (
                    None if final_eval_failed else final_nll_bits
                ),
                "stable_loss_steps": stable_loss_steps,
            }
            if final_improved:
                self._save_status(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    step=int(final_row["step"]),
                    next_step=final_step,
                    status=_adaptive_checkpoint_status(final_status),
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
                status=_adaptive_checkpoint_status(final_status),
                row=final_row,
                optimizer_phase=current_phase,
            )
            if sampling_checkpoint is None:
                sampling_checkpoint = latest_checkpoint
            summary = {
                **final_status,
                "families": model.n_families,
                "species": int(model.n_species),
                "batches": len(model.batch_metadata),
                "final_nll_bits": final_nll_bits,
                "final_grad_inf": final_grad_inf,
            }
            _write_final_artifacts(
                config,
                model=model,
                history=self.history,
                final_row=final_row,
                summary=summary,
                history_jsonl=self.history_jsonl,
                per_family_nll=final_per_family_nll,
            )
            result = OptimizationResult(
                out_dir=config.out_dir,
                status=str(status["status"]),
                reason=str(status["reason"]),
                final_nll_bits=final_nll_bits,
                final_grad_inf=final_grad_inf,
                best_nll_bits=None if best_nll is None else float(best_nll),
                best_step=None if best_step is None else int(best_step),
                steps_completed=int(final_row["step"]),
                sampling_checkpoint=sampling_checkpoint,
            )
        except BaseException as exc:
            close_model_after_error(model, exc)
            raise
        else:
            model.close()
            return result


def optimize(config: RunConfig) -> OptimizationResult:
    return OptimizationRunner(config).run()
