"""Configuration dataclasses for :class:`GeneReconModel`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SolverSettings:
    fixed_iters_E: int | None
    max_iters_E: int
    tol_E: float
    fixed_iters_Pi: int
    neumann_terms: int
    adaptive_iters: bool
    adaptive_neumann_terms: bool
    convergence_check_interval: int
    e_logsumexp_tol: float
    pi_max_diff_tol: float
    gradient_change_tol: float
    gradient_change_rtol: float
    use_pruning: bool
    pruning_threshold: float
    pi_adjoint_warmstart: bool
    pi_adjoint_cache_update_mode: str
    pi_fixed_point_relaxation: float

    def static_kwargs(self) -> dict[str, Any]:
        return {
            "fixed_iters_E": self.fixed_iters_E,
            "max_iters_E": self.max_iters_E,
            "tol_E": self.tol_E,
            "fixed_iters_Pi": self.fixed_iters_Pi,
            "neumann_terms": self.neumann_terms,
            "adaptive_iters": self.adaptive_iters,
            "adaptive_neumann_terms": self.adaptive_neumann_terms,
            "convergence_check_interval": self.convergence_check_interval,
            "e_logsumexp_tol": self.e_logsumexp_tol,
            "pi_max_diff_tol": self.pi_max_diff_tol,
            "gradient_change_tol": self.gradient_change_tol,
            "gradient_change_rtol": self.gradient_change_rtol,
            "use_pruning": self.use_pruning,
            "pruning_threshold": self.pruning_threshold,
            "pi_fixed_point_relaxation": self.pi_fixed_point_relaxation,
        }


@dataclass(frozen=True)
class ModelBatchSettings:
    family_chunk_size: int
    clade_budget: int | None
    batch_packing: str
    small_family_max_leaves: int
    lazy_preprocess: bool
    prefetch_batches: int | str
    shared_loss_batch_streams: int
    max_wave_size: int | None
    max_root_wave_size: int | None
    max_dts_partial_rows: int | None
