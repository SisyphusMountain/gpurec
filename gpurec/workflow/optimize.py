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


_FINAL_ARTIFACT_FILES = (
    "history.jsonl",
    "rates_final.tsv",
    "per_fam_likelihoods.tsv",
    "theta_final.pt",
    "optimization_history.csv",
    "summary.json",
)


def _is_finite_tensor(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and bool(torch.isfinite(tensor).all().item())


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
def _per_family_nll(model: GeneReconModel) -> list[tuple[str, float]]:
    values = model.full_nll_per_family().detach().cpu().reshape(-1).tolist()
    return list(zip(model_family_names(model), values))


def _write_per_family_likelihoods(path: Path, model: GeneReconModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("family\tnll_bits\tlog_likelihood_bits\n")
        for family, nll in _per_family_nll(model):
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
            _write_per_family_likelihoods(per_family_stage_path, model)
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
    grad_inf: float,
    stable_loss_steps: int,
    best_step: int | None,
) -> dict[str, str] | None:
    if grad_inf <= config.grad_inf_tol:
        return {"status": "converged", "reason": "gradient_tolerance"}
    if config.loss_patience and stable_loss_steps >= config.loss_patience:
        return {"status": "stalled", "reason": "loss_change_patience"}
    if (
        config.best_likelihood_patience
        and best_step is not None
        and step - int(best_step) >= config.best_likelihood_patience
    ):
        return {"status": "stalled", "reason": "best_likelihood_patience"}
    return None


class OptimizationRunner:
    def __init__(self, config: RunConfig):
        self.config = config
        self.history: list[dict[str, Any]] = []
        self.history_jsonl = config.out_dir / "history.jsonl"

    def build_model(self) -> GeneReconModel:
        config = self.config
        prefetch_batches: int | str = (
            1
            if config.mode == "genewise" and config.optimizer == "batched-lbfgs"
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
                tolerance_grad=config.grad_inf_tol,
                line_search_fn=(
                    "strong_wolfe"
                    if config.lbfgs_line_search == "strong_wolfe"
                    else "armijo"
                ),
                lower_bound=math.log2(config.min_rate),
                upper_bound=math.log2(config.max_rate),
            )
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

    def _uses_solver_warmup(self) -> bool:
        return (
            self.config.mode == "genewise"
            and self.config.optimizer == "batched-lbfgs"
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
            model.configure_solver_iterations(
                fixed_iters_E=iters,
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

    def _should_switch_solver_warmup(
        self,
        *,
        grad_inf: float,
        stable_loss_steps: int,
    ) -> bool:
        config = self.config
        if grad_inf <= config.solver_warmup_grad_inf_tol:
            return True
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
        active_optimizer_batch_index: int | None = None
        batchwise_batched_lbfgs = (
            config.mode == "genewise" and config.optimizer == "batched-lbfgs"
        )
        solver_warmup_enabled = self._uses_solver_warmup()
        active_solver_stage = "warmup" if solver_warmup_enabled else "full"
        batch_best_nll: float | None = None
        batch_best_step: int | None = None
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
                if active_solver_stage not in {"warmup", "full"}:
                    raise RuntimeError(
                        f"checkpoint {config.resume_from} has invalid active_solver_stage"
                    )
                if active_solver_stage == "warmup" and not solver_warmup_enabled:
                    active_solver_stage = "full"
                if batchwise_batched_lbfgs:
                    batch_best_nll = best_nll
                    batch_best_step = best_step
                    best_nll = None
                    best_step = None

            if batchwise_batched_lbfgs:
                if active_batch_index >= len(model.batch_metadata):
                    raise RuntimeError(
                        f"checkpoint active batch {active_batch_index} exceeds "
                        f"{len(model.batch_metadata)} model batches"
                    )
                model.select_batch(active_batch_index)
                self._configure_solver_stage(model, active_solver_stage)

            current_phase = self._phase_for_step(start_step)
            optimizer = self._make_optimizer(model, current_phase)
            if current_phase == "batched-lbfgs" and batchwise_batched_lbfgs:
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
                    if batchwise_batched_lbfgs:
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
                    if batchwise_batched_lbfgs:
                        return self._evaluate_active_genewise_loss_vector(model)
                    return self._evaluate_genewise_loss_vector(model)

                save_best_after_row = False
                first_order_pending_step = False
                eval_position = (
                    "post_step"
                    if phase in {"lbfgs", "batched-lbfgs"}
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
                    first_order_pending_step = not skip_step_for_gradient

                objective = float(metrics["likelihood/data_nll_bits"])
                delta = None if previous_objective is None else previous_objective - objective
                if delta is not None and abs(delta) <= config.loss_change_tol:
                    stable_loss_steps += 1
                else:
                    stable_loss_steps = 0
                previous_objective = objective

                active_objective_scope = (
                    batchwise_batched_lbfgs and phase == "batched-lbfgs"
                )
                if active_objective_scope:
                    improved = (
                        batch_best_nll is None
                        or objective < batch_best_nll - config.best_likelihood_min_delta
                    )
                    if improved:
                        batch_best_nll = objective
                        batch_best_step = step
                    row_best_nll = batch_best_nll
                    row_best_step = batch_best_step
                else:
                    improved = (
                        best_nll is None
                        or objective < best_nll - config.best_likelihood_min_delta
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
                if save_best_after_row and phase not in {"lbfgs", "batched-lbfgs"}:
                    best_row = dict(row)
                    best_row["optimizer/step_applied"] = False
                    best_row["step_s"] = time.perf_counter() - t0
                    self._save_status(
                        best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        next_step=step,
                        status=checkpoint_status,
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
                    first_order_pending_step or phase in {"lbfgs", "batched-lbfgs"}
                )
                row["step_s"] = time.perf_counter() - t0

                final_row = row
                self._record(row)

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
                if config.checkpoint_every and (step + 1) % config.checkpoint_every == 0:
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
                        f"solver={row.get('optimizer/solver_stage', 'full')} "
                        f"nll_bits={objective:.6f} "
                        f"grad_inf={row.get('grad/inf', float('nan')):.6g} "
                        f"delta={float('nan') if delta is None else delta:.6g} "
                        f"best={float('nan') if row_best_nll is None else row_best_nll:.6f} "
                        f"step_s={row['step_s']:.3f}",
                        flush=True,
                    )

                warmup_switch = (
                    active_objective_scope
                    and active_solver_stage == "warmup"
                    and self._should_switch_solver_warmup(
                        grad_inf=float(row.get("grad/inf", math.inf)),
                        stable_loss_steps=stable_loss_steps,
                    )
                )
                step_status = _step_stopping_status(
                    config,
                    step=step,
                    grad_inf=float(row.get("grad/inf", math.inf)),
                    stable_loss_steps=stable_loss_steps,
                    best_step=row_best_step,
                )
                if warmup_switch:
                    active_solver_stage = "full"
                    previous_objective = None
                    stable_loss_steps = 0
                    batch_best_nll = None
                    batch_best_step = None
                    optimizer = None
                    active_optimizer_batch_index = None
                    self._configure_solver_stage(model, active_solver_stage)
                    if config.checkpoint_every:
                        transition_status = {
                            **checkpoint_status,
                            "active_batch_index": active_batch_index,
                            "active_solver_stage": active_solver_stage,
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
                            status=transition_status,
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
                        previous_objective = None
                        stable_loss_steps = 0
                        batch_best_nll = None
                        batch_best_step = None
                        optimizer = None
                        active_optimizer_batch_index = None
                        if config.checkpoint_every:
                            transition_status = {
                                **checkpoint_status,
                                "active_batch_index": active_batch_index,
                                "active_solver_stage": active_solver_stage,
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
                                status=transition_status,
                                row=row,
                                optimizer_phase=phase,
                            )
                        model.select_batch(active_batch_index)
                        self._configure_solver_stage(model, active_solver_stage)
                        resume_info = {}
                        continue
                    status = step_status
                    break
            else:
                status = {"status": "not_converged", "reason": "max_steps"}

            if batchwise_batched_lbfgs:
                self._configure_solver_stage(model, "full")
            model.theta.grad = None
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
                    "closure_evals": 1,
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
