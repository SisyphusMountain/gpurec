"""Constructor setup helpers for :class:`GeneReconModel`.

This module is internal support for ``gpurec.api`` model construction, not a
public import surface. It owns constructor validation and default model state
only; resident cache lifecycle stays in ``_resident_runtime``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from gpurec._validation import disabled_adaptive_neumann_terms_value
from gpurec.core.model import GeneDataset
from gpurec.core.origination import (
    OriginationPrior,
    PreparedOriginationPrior,
    prepare_origination_prior,
)

from ._batch_specs import (
    _normalize_batch_packing,
    _normalize_clade_budget,
    _normalize_family_chunk_size,
    _normalize_pi_adjoint_cache_update_mode,
    _normalize_prefetch_batches,
)
from ._model_config import ModelBatchSettings, SolverSettings
from ._static_builder import ResidentCommonState
from ._theta_init import _default_theta_init, _mode_to_flags, _normalize_mode
from ._theta_init import _validate_gene_dtype
from ._validation import (
    bool_value,
    nonnegative_float,
    nonnegative_int,
    optional_positive_int,
    positive_even_int,
    positive_float,
    positive_int,
    require_cuda_device,
    require_default_objective,
    validate_theta_shape,
)


@dataclass(frozen=True)
class ModelInitState:
    dataset: GeneDataset
    mode: str
    settings: SolverSettings
    batch_settings: ModelBatchSettings
    batched_resident: bool
    theta_init: torch.Tensor
    origination_prior: PreparedOriginationPrior
    resident_common_state: ResidentCommonState


def prepare_model_init(
    *,
    dataset: GeneDataset,
    mode: str,
    fixed_iters_E: Optional[int],
    max_iters_E: int,
    tol_E: float,
    fixed_iters_Pi: int,
    neumann_terms: int,
    adaptive_iters: bool,
    adaptive_neumann_terms: bool,
    convergence_check_interval: int,
    e_logsumexp_tol: float,
    pi_max_diff_tol: float,
    gradient_change_tol: float,
    gradient_change_rtol: float,
    use_pruning: bool,
    pruning_threshold: float,
    theta_init: Optional[torch.Tensor],
    max_wave_size: Optional[int],
    max_root_wave_size: Optional[int],
    max_dts_partial_rows: Optional[int],
    family_chunk_size: int | str | None,
    clade_budget: int | None,
    batch_packing: str | None,
    small_family_max_leaves: int | None,
    lazy_preprocess: bool,
    prefetch_batches: int | str | None,
    pi_adjoint_warmstart: bool,
    pi_adjoint_cache_update_mode: str,
    pi_fixed_point_relaxation: float,
    shared_loss_batch_streams: int,
    origination_probs: (
        torch.Tensor
        | Sequence[float]
        | OriginationPrior
        | PreparedOriginationPrior
        | None
    ),
) -> ModelInitState:
    require_default_objective("GeneReconModel")
    mode = _normalize_mode(mode)
    _mode_to_flags(mode)
    if fixed_iters_E is not None:
        fixed_iters_E = positive_int("fixed_iters_E", fixed_iters_E)
    fixed_iters_Pi = positive_even_int("fixed_iters_Pi", fixed_iters_Pi)
    neumann_terms = positive_int("neumann_terms", neumann_terms)
    convergence_check_interval = positive_int(
        "convergence_check_interval",
        convergence_check_interval,
    )
    adaptive_iters = bool_value("adaptive_iters", adaptive_iters)
    adaptive_neumann_terms = disabled_adaptive_neumann_terms_value(
        adaptive_neumann_terms
    )
    if adaptive_iters and convergence_check_interval % 2 != 0:
        raise ValueError("adaptive_iters requires an even convergence_check_interval")
    max_iters_E = positive_int("max_iters_E", max_iters_E)
    tol_E = nonnegative_float("tol_E", tol_E)
    e_logsumexp_tol = nonnegative_float("e_logsumexp_tol", e_logsumexp_tol)
    pi_max_diff_tol = nonnegative_float("pi_max_diff_tol", pi_max_diff_tol)
    gradient_change_tol = nonnegative_float(
        "gradient_change_tol",
        gradient_change_tol,
    )
    gradient_change_rtol = nonnegative_float(
        "gradient_change_rtol",
        gradient_change_rtol,
    )
    pruning_threshold = nonnegative_float("pruning_threshold", pruning_threshold)
    use_pruning = bool_value("use_pruning", use_pruning)
    family_chunk_requested = family_chunk_size is not None
    family_chunk_size = _normalize_family_chunk_size(family_chunk_size)
    clade_budget = _normalize_clade_budget(clade_budget)
    max_wave_size = optional_positive_int("max_wave_size", max_wave_size)
    max_root_wave_size = optional_positive_int(
        "max_root_wave_size",
        max_root_wave_size,
    )
    max_dts_partial_rows = optional_positive_int(
        "max_dts_partial_rows",
        max_dts_partial_rows,
    )
    small_family_max_leaves = (
        0
        if small_family_max_leaves is None
        else nonnegative_int("small_family_max_leaves", small_family_max_leaves)
    )
    batch_packing = _normalize_batch_packing(batch_packing)
    lazy_preprocess = bool_value("lazy_preprocess", lazy_preprocess)
    prefetch_batches = _normalize_prefetch_batches(
        prefetch_batches,
        lazy=lazy_preprocess,
    )
    pi_adjoint_warmstart = bool_value(
        "pi_adjoint_warmstart",
        pi_adjoint_warmstart,
    )
    pi_adjoint_cache_update_mode = _normalize_pi_adjoint_cache_update_mode(
        pi_adjoint_cache_update_mode
    )
    pi_fixed_point_relaxation = positive_float(
        "pi_fixed_point_relaxation",
        pi_fixed_point_relaxation,
    )
    shared_loss_batch_streams = positive_int(
        "shared_loss_batch_streams",
        shared_loss_batch_streams,
    )
    _validate_gene_dtype(dataset.dtype)

    ds_g, ds_sw = (dataset.genewise, dataset.specieswise)
    expected_g, expected_sw = _mode_to_flags(mode)
    if (ds_g, ds_sw) != (expected_g, expected_sw):
        raise ValueError(
            f"Dataset flags (genewise={ds_g}, specieswise={ds_sw}) do not "
            f"match requested mode {mode!r} "
            f"(expected genewise={expected_g}, specieswise={expected_sw}). "
            "Construct GeneDataset with matching flags or use "
            "GeneReconModel.from_trees()."
        )
    if theta_init is not None:
        theta_init = validate_theta_shape(
            "theta_init",
            theta_init,
            mode=mode,
            species_count=int(dataset.S),
            family_count=len(dataset.families),
            device=dataset.device,
            dtype=dataset.dtype,
        )

    require_cuda_device(dataset.device, owner="GeneReconModel")

    origination_prior = prepare_origination_prior(
        origination_probs,
        S=int(dataset.S),
        device=dataset.device,
        dtype=dataset.dtype,
        family_count=len(dataset.families) if origination_probs is not None else None,
    )
    settings = SolverSettings(
        fixed_iters_E=fixed_iters_E,
        max_iters_E=max_iters_E,
        tol_E=tol_E,
        fixed_iters_Pi=fixed_iters_Pi,
        neumann_terms=neumann_terms,
        adaptive_iters=adaptive_iters,
        adaptive_neumann_terms=adaptive_neumann_terms,
        convergence_check_interval=convergence_check_interval,
        e_logsumexp_tol=e_logsumexp_tol,
        pi_max_diff_tol=pi_max_diff_tol,
        gradient_change_tol=gradient_change_tol,
        gradient_change_rtol=gradient_change_rtol,
        use_pruning=use_pruning,
        pruning_threshold=pruning_threshold,
        pi_adjoint_warmstart=pi_adjoint_warmstart,
        pi_adjoint_cache_update_mode=pi_adjoint_cache_update_mode,
        pi_fixed_point_relaxation=pi_fixed_point_relaxation,
    )
    batch_settings = ModelBatchSettings(
        family_chunk_size=family_chunk_size,
        family_chunk_requested=family_chunk_requested,
        clade_budget=clade_budget,
        batch_packing=batch_packing,
        small_family_max_leaves=small_family_max_leaves,
        lazy_preprocess=lazy_preprocess,
        prefetch_batches=prefetch_batches,
        shared_loss_batch_streams=shared_loss_batch_streams,
        max_wave_size=max_wave_size,
        max_root_wave_size=max_root_wave_size,
        max_dts_partial_rows=max_dts_partial_rows,
    )
    batched_resident = bool(
        batch_settings.lazy_preprocess
        or batch_settings.family_chunk_requested
        or batch_settings.clade_budget is not None
    )
    if theta_init is None:
        theta_init = _default_theta_init(dataset, mode)

    species_helpers, ancestors_T = dataset._species_helpers_for_mode(
        device=dataset.device,
        dtype=dataset.dtype,
    )
    resident_common_state = ResidentCommonState(
        species_helpers=species_helpers,
        ancestors_T=ancestors_T,
        unnorm_row_max=dataset.unnorm_row_max.to(
            device=dataset.device,
            dtype=dataset.dtype,
        ),
    )
    return ModelInitState(
        dataset=dataset,
        mode=mode,
        settings=settings,
        batch_settings=batch_settings,
        batched_resident=batched_resident,
        theta_init=theta_init,
        origination_prior=origination_prior,
        resident_common_state=resident_common_state,
    )


def apply_model_init_state(model: Any, state: ModelInitState) -> None:
    model._mode = state.mode
    model._dataset = state.dataset
    model._origination_prior = state.origination_prior
    model._settings = state.settings
    model.register_buffer("origination_probs", state.origination_prior.probs)

    batch_settings = state.batch_settings
    model.family_chunk_size = batch_settings.family_chunk_size
    model.clade_budget = batch_settings.clade_budget
    model.batch_packing = batch_settings.batch_packing
    model.small_family_max_leaves = batch_settings.small_family_max_leaves
    model.lazy_preprocess = batch_settings.lazy_preprocess
    model.prefetch_batches = batch_settings.prefetch_batches
    model.shared_loss_batch_streams = batch_settings.shared_loss_batch_streams
    model._batched_resident = state.batched_resident
    model.theta = torch.nn.Parameter(state.theta_init.clone())

    _mirror_solver_settings(model, state.settings)
    model.max_wave_size = batch_settings.max_wave_size
    model.max_root_wave_size = batch_settings.max_root_wave_size
    model.max_dts_partial_rows = batch_settings.max_dts_partial_rows
    model._resident_common_state = state.resident_common_state


def _mirror_solver_settings(model: Any, settings: SolverSettings) -> None:
    model._fixed_iters_E = settings.fixed_iters_E
    model._max_iters_E = settings.max_iters_E
    model._tol_E = float(settings.tol_E)
    model._fixed_iters_Pi = settings.fixed_iters_Pi
    model._neumann_terms = settings.neumann_terms
    model._adaptive_iters = settings.adaptive_iters
    model._adaptive_neumann_terms = settings.adaptive_neumann_terms
    model._convergence_check_interval = settings.convergence_check_interval
    model._e_logsumexp_tol = float(settings.e_logsumexp_tol)
    model._pi_max_diff_tol = float(settings.pi_max_diff_tol)
    model._gradient_change_tol = float(settings.gradient_change_tol)
    model._gradient_change_rtol = float(settings.gradient_change_rtol)
    model._use_pruning = settings.use_pruning
    model._pruning_threshold = settings.pruning_threshold
    model._pi_adjoint_warmstart = settings.pi_adjoint_warmstart
    model._pi_adjoint_cache_update_mode = settings.pi_adjoint_cache_update_mode
    model._pi_fixed_point_relaxation = settings.pi_fixed_point_relaxation
