from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gpurec.api.model import GeneReconModel

from .checkpoint import load_checkpoint, restore_model_theta, save_checkpoint
from .config import RunConfig
from .diagnostics import (
    append_jsonl,
    parameter_stats,
    rates_and_survival_probability,
    solver_stats,
    tensor_stats,
    write_csv,
)


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


def _is_finite_tensor(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and bool(torch.isfinite(tensor).all().item())


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
        if not str(config.device).startswith("cuda"):
            raise RuntimeError("gpurec production optimization currently requires CUDA")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return GeneReconModel.from_alerax_families(
            str(config.species_tree),
            config.families_file,
            mode=config.mode,
            start=config.start,
            max_families=config.max_families,
            device=config.device,
            dtype=config.torch_dtype,
            theta_init_rates=config.theta_init_rates,
            preprocess_cache_dir=config.preprocess_cache,
            refresh_preprocess_cache=config.refresh_preprocess_cache,
            fixed_iters_E=config.fixed_iters_e,
            max_iters_E=config.max_iters_e,
            tol_E=config.tol_e,
            fixed_iters_Pi=config.fixed_iters_pi,
            neumann_terms=config.neumann_terms,
            adaptive_iters=config.adaptive_iters,
            convergence_check_interval=config.convergence_check_interval,
            e_logsumexp_tol=config.e_logsumexp_tol,
            pi_max_diff_tol=config.pi_max_diff_tol,
            gradient_change_tol=config.gradient_change_tol,
            gradient_change_rtol=config.gradient_change_rtol,
            family_chunk_size=config.family_chunk_size,
            clade_budget=config.clade_budget,
            batch_packing=config.batch_packing,
            max_wave_size=config.max_wave_size,
            lazy_preprocess=True,
            prefetch_batches="all",
        )

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

    def _save_status(
        self,
        path: Path,
        *,
        model: GeneReconModel,
        optimizer: torch.optim.Optimizer | None,
        step: int,
        status: dict[str, Any],
        row: dict[str, Any] | None,
    ) -> None:
        save_checkpoint(
            path,
            config=self.config,
            model=model,
            optimizer=optimizer,
            step=step,
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

        try:
            if config.resume_from is not None:
                payload = load_checkpoint(config.resume_from, map_location=config.device)
                restore_model_theta(model, payload)
                start_step = int(payload.get("next_step", 0))
                ckpt_status = payload.get("status") or {}
                best_nll = ckpt_status.get("best_nll_bits")
                best_step = ckpt_status.get("best_step")
                previous_objective = ckpt_status.get("previous_objective")
                stable_loss_steps = int(ckpt_status.get("stable_loss_steps", 0))

            current_phase = self._phase_for_step(start_step)
            optimizer = self._make_optimizer(model, current_phase)
            if config.resume_from is not None:
                payload = load_checkpoint(config.resume_from, map_location=config.device)
                state = payload.get("optimizer_state")
                if state is not None:
                    try:
                        optimizer.load_state_dict(state)
                    except ValueError:
                        pass

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
                    save_best_after_row = phase == "lbfgs"

                if phase != "lbfgs":
                    checkpoint_status = {
                        "status": "running",
                        "reason": "running",
                        "best_nll_bits": best_nll,
                        "best_step": best_step,
                        "previous_objective": previous_objective,
                        "stable_loss_steps": stable_loss_steps,
                    }
                    if improved:
                        self._save_status(
                            config.out_dir / "checkpoints" / "best.pt",
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            status=checkpoint_status,
                            row={
                                "step": step,
                                "optimizer/phase": phase,
                                "theta_step_inf": 0.0,
                                "delta_likelihood_bits": delta,
                                **metrics,
                            },
                        )
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
                    if not stop_after_row:
                        optimizer.step()
                        with torch.no_grad():
                            model.clamp_theta_(config.min_rate, config.max_rate)
                        theta_step = float(
                            (model.theta.detach() - theta_before).abs().amax().cpu()
                        )
                    model.clear()

                row = {
                    "step": step,
                    "optimizer/phase": phase,
                    "closure_evals": closure_evals,
                    "theta_step_inf": theta_step,
                    "delta_likelihood_bits": delta,
                    "stable_loss_steps": stable_loss_steps,
                    "best_nll_bits": best_nll,
                    "best_step": best_step,
                    "step_s": time.perf_counter() - t0,
                    **metrics,
                }
                final_row = row
                self._record(row)

                checkpoint_status = {
                    "status": "running",
                    "reason": "running",
                    "best_nll_bits": best_nll,
                    "best_step": best_step,
                    "previous_objective": previous_objective,
                    "stable_loss_steps": stable_loss_steps,
                }
                if save_best_after_row:
                    self._save_status(
                        config.out_dir / "checkpoints" / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        status=checkpoint_status,
                        row=row,
                    )
                if config.checkpoint_every and step % config.checkpoint_every == 0:
                    self._save_status(
                        config.out_dir / "checkpoints" / "latest.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        status=checkpoint_status,
                        row=row,
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
                "step_s": 0.0,
                **final_metrics,
            }
            self._record(final_row)

            final_status = {
                **status,
                "elapsed_s": time.perf_counter() - started,
                "best_nll_bits": best_nll,
                "best_step": best_step,
                "previous_objective": float(final_loss.detach().cpu()),
                "stable_loss_steps": stable_loss_steps,
            }
            if final_improved:
                self._save_status(
                    config.out_dir / "checkpoints" / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=int(final_row["step"]),
                    status=final_status,
                    row=final_row,
                )
            self._save_status(
                config.out_dir / "checkpoints" / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=int(final_row["step"]),
                status=final_status,
                row=final_row,
            )
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
            )
        finally:
            model.close()


def optimize(config: RunConfig) -> OptimizationResult:
    return OptimizationRunner(config).run()
