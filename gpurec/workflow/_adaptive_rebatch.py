from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from gpurec.api.model import GeneReconModel

from ._runtime_state import _ResumeState

_ACTIVE_BATCH_REBATCH_PATIENCE = 3
_ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES = 64


def _active_batch_patience(configured_patience: int) -> int:
    if configured_patience <= 0:
        return configured_patience
    return min(configured_patience, _ACTIVE_BATCH_REBATCH_PATIENCE)


@dataclass
class _AdaptiveRebatchDecision:
    metrics: dict[str, Any]
    pending_indices: list[int] | None = None
    stop: bool = False


@dataclass
class _AdaptiveRebatchState:
    enabled: bool
    converged_family_mask: torch.Tensor | None
    adaptive_family_best_nll: torch.Tensor | None
    adaptive_family_stable_steps: torch.Tensor | None
    batch_plan_generation: int = 0
    last_checked_converged_count: int = 0
    min_active_families: int = _ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        model: GeneReconModel,
        min_active_families: int | None = None,
    ) -> "_AdaptiveRebatchState":
        if min_active_families is None:
            min_active_families = _ADAPTIVE_REBATCH_MIN_ACTIVE_FAMILIES
        if not enabled:
            return cls(
                enabled=False,
                converged_family_mask=None,
                adaptive_family_best_nll=None,
                adaptive_family_stable_steps=None,
                min_active_families=int(min_active_families),
            )
        family_count = int(model.n_families)
        return cls(
            enabled=True,
            converged_family_mask=torch.zeros(
                (family_count,),
                device=model.theta.device,
                dtype=torch.bool,
            ),
            adaptive_family_best_nll=torch.full(
                (family_count,),
                math.inf,
                device=model.theta.device,
                dtype=model.theta.dtype,
            ),
            adaptive_family_stable_steps=torch.zeros(
                (family_count,),
                device=model.theta.device,
                dtype=torch.long,
            ),
            min_active_families=int(min_active_families),
        )

    def to_converged_family_indices(self) -> list[int]:
        if not self.enabled or self.converged_family_mask is None:
            return []
        return [
            int(index)
            for index in torch.nonzero(
                self.converged_family_mask,
                as_tuple=False,
            ).flatten().detach().cpu().tolist()
        ]

    def _current_plan_indices(
        self,
        model: GeneReconModel,
        active_batch_index: int,
    ) -> list[int]:
        if not self.enabled:
            return []
        plan_indices: list[int] = []
        for metadata in model.batch_metadata[active_batch_index:]:
            plan_indices.extend(int(index) for index in metadata.family_indices)
        return plan_indices

    def remaining_current_plan_indices(
        self,
        model: GeneReconModel,
        *,
        active_batch_index: int,
        converged_mask: torch.Tensor | None = None,
    ) -> list[int]:
        if self.converged_family_mask is None:
            return []
        plan_indices = self._current_plan_indices(model, active_batch_index)
        if not plan_indices:
            return []
        idx = torch.as_tensor(
            plan_indices,
            dtype=torch.long,
            device=model.theta.device,
        )
        mask = (
            self.converged_family_mask
            if converged_mask is None
            else converged_mask
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

    def _mask_with_prior_plan_families(
        self,
        model: GeneReconModel,
        *,
        active_batch_index: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.converged_family_mask is None:
            return mask
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

    def checkpoint_status(self, base: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or self.converged_family_mask is None:
            return base
        enriched = dict(base)
        enriched["converged_family_indices"] = self.to_converged_family_indices()
        enriched["batch_plan_generation"] = float(self.batch_plan_generation)
        return enriched

    def restore_from_resume(
        self,
        *,
        model: GeneReconModel,
        resume_state: _ResumeState,
        active_batch_index: int,
        checkpoint_path: str | None = None,
    ) -> list[int] | None:
        if not self.enabled or self.converged_family_mask is None:
            return None
        self.batch_plan_generation = int(resume_state.batch_plan_generation)
        if resume_state.converged_family_indices:
            max_index = max(resume_state.converged_family_indices)
            if max_index >= int(model.n_families):
                checkpoint_name = checkpoint_path or "checkpoint"
                raise RuntimeError(
                    f"checkpoint {checkpoint_name} has out-of-range converged family indices"
                )
            self.converged_family_mask.index_fill_(
                0,
                torch.as_tensor(
                    resume_state.converged_family_indices,
                    dtype=torch.long,
                    device=model.theta.device,
                ),
                True,
            )

        if self.batch_plan_generation > 0:
            return self.remaining_current_plan_indices(
                model,
                active_batch_index=active_batch_index,
            )
        return None

    def evaluate(
        self,
        *,
        config: Any,
        model: GeneReconModel,
        active_solver_stage: str,
        step: int,
        loss_vec_current: torch.Tensor,
    ) -> _AdaptiveRebatchDecision:
        if not self.enabled:
            return _AdaptiveRebatchDecision({})
        if self.converged_family_mask is None:
            return _AdaptiveRebatchDecision({})
        if self.adaptive_family_best_nll is None or self.adaptive_family_stable_steps is None:
            return _AdaptiveRebatchDecision({})

        metrics: dict[str, Any] = {
            "optimizer/adaptive_rebatch_enabled": True,
            "optimizer/rebatch_generation": float(self.batch_plan_generation),
            "optimizer/rebatch_triggered": False,
        }
        idx = torch.as_tensor(
            model.current_batch_metadata.family_indices,
            dtype=torch.long,
            device=model.theta.device,
        )
        batch_family_count = int(idx.numel())
        metrics["optimizer/rebatch_active_family_count"] = float(batch_family_count)

        active_batch_large_enough = (
            batch_family_count >= self.min_active_families
        )
        should_check_rebatch = (
            active_solver_stage == "full"
            and active_batch_large_enough
            and (step + 1) % int(config.adaptive_rebatch_check_interval) == 0
        )
        metrics["optimizer/rebatch_checked"] = should_check_rebatch
        if not active_batch_large_enough:
            metrics["optimizer/rebatch_reason"] = "small_active_batch"
            return _AdaptiveRebatchDecision(metrics)
        if (
            not should_check_rebatch
            or model.theta.grad is None
        ):
            return _AdaptiveRebatchDecision(metrics)

        active_loss = loss_vec_current.detach().index_select(0, idx)
        active_best = self.adaptive_family_best_nll.index_select(0, idx)
        active_stable = self.adaptive_family_stable_steps.index_select(0, idx)

        finite_loss = torch.isfinite(active_loss)
        improved_family = finite_loss & (
            active_loss < active_best - float(config.best_likelihood_min_delta)
        )
        next_best = torch.where(improved_family, active_loss, active_best)
        next_stable = torch.where(improved_family, torch.zeros_like(active_stable), active_stable + 1)

        self.adaptive_family_best_nll.index_copy_(0, idx, next_best)
        self.adaptive_family_stable_steps.index_copy_(0, idx, next_stable)

        family_patience = _active_batch_patience(int(config.best_likelihood_patience))
        if family_patience > 0:
            row_converged = next_stable >= family_patience
        else:
            row_converged = torch.zeros_like(next_stable, dtype=torch.bool)
        row_converged = row_converged & ~self.converged_family_mask.index_select(0, idx)

        threshold_count = max(
            1,
            math.ceil(float(config.adaptive_rebatch_fraction) * batch_family_count),
        )
        converged_count = int(row_converged.sum().detach().cpu())
        crossed_threshold = (
            self.last_checked_converged_count < threshold_count
            and threshold_count <= converged_count
        )
        self.last_checked_converged_count = converged_count

        metrics.update(
            {
                "optimizer/rebatch_active_converged_families": float(converged_count),
                "optimizer/rebatch_convergence_criterion": "best_likelihood_patience",
                "optimizer/rebatch_family_stable_steps_max": float(next_stable.max().detach().cpu()),
                "optimizer/rebatch_threshold_families": float(threshold_count),
            }
        )

        if not crossed_threshold:
            return _AdaptiveRebatchDecision(metrics)

        candidate_mask = self.converged_family_mask.clone()
        active_rows = torch.nonzero(row_converged, as_tuple=False).flatten()
        if bool(active_rows.any().detach().cpu()):
            candidate_mask.index_fill_(0, idx.index_select(0, active_rows), True)

        candidate_mask = self._mask_with_prior_plan_families(
            model,
            active_batch_index=int(model.current_batch_index),
            mask=candidate_mask,
        )
        remaining_indices = self.remaining_current_plan_indices(
            model,
            active_batch_index=int(model.current_batch_index),
            converged_mask=candidate_mask,
        )
        remaining_count = len(remaining_indices)
        metrics["optimizer/rebatch_remaining_families"] = float(remaining_count)

        if remaining_count == 0:
            self.converged_family_mask.copy_(candidate_mask)
            return _AdaptiveRebatchDecision(
                metrics={**metrics, "optimizer/rebatch_reason": "all_converged"},
                stop=True,
            )

        if remaining_count >= int(config.adaptive_rebatch_min_remaining_families):
            self.converged_family_mask.copy_(candidate_mask)
            return _AdaptiveRebatchDecision(
                metrics={**metrics, "optimizer/rebatch_triggered": True},
                pending_indices=remaining_indices,
            )

        return _AdaptiveRebatchDecision(
            metrics={**metrics, "optimizer/rebatch_reason": "below_min_remaining"},
        )
