"""High-level ``nn.Module`` for phylogenetic reconciliation.

Wraps a :class:`gpurec.core.model.GeneDataset` (used purely for preprocessing)
and exposes ``theta`` as an ``nn.Parameter`` so notebook users can use any
``torch.optim`` optimizer with the standard pattern::

    model = GeneReconModel.from_trees(
        species_tree="sp.nwk", gene_trees=["g1.nwk"], mode="global", device="cuda",
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = model()              # NLL
        loss.backward()
        opt.step()
        model.clamp_theta_()
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
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
    _normalize_family_chunk_size as _normalize_family_chunk_size_impl,
    _normalize_clade_budget as _normalize_clade_budget_impl,
    _normalize_batch_packing as _normalize_batch_packing_impl,
    _normalize_prefetch_batches as _normalize_prefetch_batches_impl,
    _normalize_gene_solver_kwargs as _normalize_gene_solver_kwargs_impl,
    _normalize_pi_adjoint_cache_update_mode as _normalize_pi_adjoint_cache_update_mode_impl,
    _build_batch_specs as _build_batch_specs_impl,
    _build_batch_specs_from_retained_rust as _build_batch_specs_from_retained_rust_impl,
    _start_batch_specs_from_retained_rust as _start_batch_specs_from_retained_rust_impl,
    _finish_batch_specs_from_retained_rust as _finish_batch_specs_from_retained_rust_impl,
    _cancel_batch_specs_from_retained_rust as _cancel_batch_specs_from_retained_rust_impl,
    _should_use_compact_retained_preprocess as _should_use_compact_retained_preprocess_impl,
    _build_family_schedule_stats as _build_family_schedule_stats_impl,
)
from ._model_config import SolverSettings
from ._model_builders import (
    build_from_alerax_families_inputs,
    build_from_trees_inputs,
)
from ._model_types import (
    ActiveFamilyBatch,
    BatchMetadata,
    FamilyInput,
    ReconciliationState,
    _FamilyScheduleStats,
    _ResidentBatchSpec,
    _public_family_value,
)
from ._resident_cache import ResidentBatchCache, _RESIDENT_PREFETCH_WORKERS
from ._static_builder import (
    ResidentCommonState,
    _build_batch_static_state as _build_batch_static_state_impl,
    _build_static_state as _build_static_state_impl,
    _metadata_for_full_static as _metadata_for_full_static_impl,
)
from ._streaming import stream_full_batches as _stream_full_batches_impl
from ._tensor_validation import (
    _validate_genewise_loss_vector as _validate_genewise_loss_vector_impl,
    _validate_genewise_gradient_matrix as _validate_genewise_gradient_matrix_impl,
)
from ._theta_init import (
    _default_theta_init as _default_theta_init_impl,
    _mode_to_flags as _mode_to_flags_impl,
    _normalize_mode as _normalize_mode_impl,
    _validate_gene_dtype,
)
from ._uniform_evaluator import (
    evaluate_resident_export_state,
    evaluate_resident_no_grad,
    evaluate_resident_static_state as _evaluate_static_state_impl,
)
from ._warmup import (
    _finish_cuda_context_warmup as _finish_cuda_context_warmup_impl,
    _start_resident_uniform_kernel_warmup as _start_resident_uniform_kernel_warmup_impl,
)
from .autograd import _GeneReconFunction, ReconStaticState, _clear_pi_adjoint_runtime_cache
from ._validation import (
    bool_value,
    integer_value,
    nonnegative_int,
    nonnegative_float,
    optional_positive_int,
    positive_even_int,
    positive_float,
    positive_int,
    require_cuda_device,
    require_default_objective,
    validate_theta_shape,
)

_UNSET = object()


# Canonical helper implementation aliases from the split modules.
_normalize_mode = _normalize_mode_impl
_mode_to_flags = _mode_to_flags_impl
_normalize_gene_solver_kwargs = _normalize_gene_solver_kwargs_impl
_normalize_family_chunk_size = _normalize_family_chunk_size_impl
_normalize_clade_budget = _normalize_clade_budget_impl
_normalize_batch_packing = _normalize_batch_packing_impl
_normalize_prefetch_batches = _normalize_prefetch_batches_impl
_normalize_pi_adjoint_cache_update_mode = _normalize_pi_adjoint_cache_update_mode_impl
_build_batch_specs = _build_batch_specs_impl
_build_batch_specs_from_retained_rust = _build_batch_specs_from_retained_rust_impl
_start_batch_specs_from_retained_rust = _start_batch_specs_from_retained_rust_impl
_finish_batch_specs_from_retained_rust = _finish_batch_specs_from_retained_rust_impl
_cancel_batch_specs_from_retained_rust = _cancel_batch_specs_from_retained_rust_impl
_should_use_compact_retained_preprocess = _should_use_compact_retained_preprocess_impl
_build_family_schedule_stats = _build_family_schedule_stats_impl
_validate_genewise_loss_vector = _validate_genewise_loss_vector_impl
_validate_genewise_gradient_matrix = _validate_genewise_gradient_matrix_impl
_default_theta_init = _default_theta_init_impl

# Keep canonical tensor validators and static-build helpers authoritative.
_evaluate_static_state = _evaluate_static_state_impl
_metadata_for_full_static = _metadata_for_full_static_impl


class _GeneReconFullLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, model: "GeneReconModel"):
        with torch.no_grad():
            loss, grad_theta = model._stream_full_batches(theta, need_grad=True)
        if grad_theta is None:
            raise RuntimeError("internal error: missing full-loss gradient")
        ctx.save_for_backward(grad_theta.detach().to(device=theta.device, dtype=theta.dtype))
        return loss.to(device=theta.device, dtype=theta.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (grad_theta,) = ctx.saved_tensors
        return grad_theta * grad_output.to(device=grad_theta.device, dtype=grad_theta.dtype), None


class GeneReconModel(torch.nn.Module):
    """A ``nn.Module`` view over a :class:`GeneDataset`.

    ``forward()`` returns the negative log-likelihood as a differentiable
    scalar. ``theta`` is registered as an ``nn.Parameter`` so any
    ``torch.optim`` optimizer can be used directly.
    """

    def __init__(
        self,
        *,
        dataset: GeneDataset,
        mode: str,
        fixed_iters_E: Optional[int] = None,
        max_iters_E: int = 2000,
        tol_E: float = 1e-8,
        fixed_iters_Pi: int = 6,
        neumann_terms: int = 3,
        adaptive_iters: bool = False,
        adaptive_neumann_terms: bool = False,
        convergence_check_interval: int = 4,
        e_logsumexp_tol: float = 1e-5,
        pi_max_diff_tol: float = 1e-5,
        gradient_change_tol: float = 1e-4,
        gradient_change_rtol: float = 1e-4,
        use_pruning: bool = True,
        pruning_threshold: float = 1e-6,
        theta_init: Optional[torch.Tensor] = None,
        max_wave_size: Optional[int] = 8192,
        max_root_wave_size: Optional[int] = None,
        max_dts_partial_rows: Optional[int] = None,
        family_chunk_size: int | str | None = None,
        clade_budget: int | None = None,
        batch_packing: str | None = None,
        small_family_max_leaves: int | None = None,
        lazy_preprocess: bool = False,
        prefetch_batches: int | str | None = None,
        pi_adjoint_warmstart: bool = False,
        pi_adjoint_cache_update_mode: str = "immediate",
        pi_fixed_point_relaxation: float = 1.0,
        shared_loss_batch_streams: int = 1,
        origination_probs: (
            torch.Tensor
            | Sequence[float]
            | OriginationPrior
            | PreparedOriginationPrior
            | None
        ) = None,
    ):
        super().__init__()
        require_default_objective("GeneReconModel")
        # Validate mode early
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
            raise ValueError(
                "adaptive_iters requires an even convergence_check_interval"
            )
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

        # Sanity check: dataset flags must be consistent with mode
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

        self._mode = mode
        self._dataset = dataset
        self._origination_prior = prepare_origination_prior(
            origination_probs,
            S=int(dataset.S),
            device=dataset.device,
            dtype=dataset.dtype,
            family_count=len(dataset.families) if origination_probs is not None else None,
        )
        self._settings = SolverSettings(
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
        self.register_buffer("origination_probs", self._origination_prior.probs)
        self.family_chunk_size = family_chunk_size
        self.clade_budget = clade_budget
        self.batch_packing = batch_packing
        self.small_family_max_leaves = small_family_max_leaves
        self.lazy_preprocess = lazy_preprocess
        self.prefetch_batches = prefetch_batches
        self.shared_loss_batch_streams = shared_loss_batch_streams
        self._batched_resident = bool(
            self.lazy_preprocess
            or family_chunk_requested
            or self.clade_budget is not None
        )

        if theta_init is None:
            theta_init = _default_theta_init(dataset, mode)
        self.theta = torch.nn.Parameter(theta_init.clone())

        self._fixed_iters_E = self._settings.fixed_iters_E
        self._max_iters_E = self._settings.max_iters_E
        self._tol_E = float(self._settings.tol_E)
        self._fixed_iters_Pi = self._settings.fixed_iters_Pi
        self._neumann_terms = self._settings.neumann_terms
        self._adaptive_iters = self._settings.adaptive_iters
        self._adaptive_neumann_terms = self._settings.adaptive_neumann_terms
        self._convergence_check_interval = self._settings.convergence_check_interval
        self._e_logsumexp_tol = float(self._settings.e_logsumexp_tol)
        self._pi_max_diff_tol = float(self._settings.pi_max_diff_tol)
        self._gradient_change_tol = float(self._settings.gradient_change_tol)
        self._gradient_change_rtol = float(self._settings.gradient_change_rtol)
        self._use_pruning = self._settings.use_pruning
        self._pruning_threshold = self._settings.pruning_threshold
        self._pi_adjoint_warmstart = self._settings.pi_adjoint_warmstart
        self._pi_adjoint_cache_update_mode = (
            self._settings.pi_adjoint_cache_update_mode
        )
        self._pi_fixed_point_relaxation = self._settings.pi_fixed_point_relaxation
        self.max_wave_size = max_wave_size
        self.max_root_wave_size = max_root_wave_size
        self.max_dts_partial_rows = max_dts_partial_rows

        self._static: ReconStaticState | None = None
        self._batch_specs: list[_ResidentBatchSpec] = []
        self._current_batch_index = 0
        self._family_schedule_stats: _FamilyScheduleStats | None = None
        species_helpers, ancestors_T = dataset._species_helpers_for_mode(
            device=dataset.device,
            dtype=dataset.dtype,
        )
        self._resident_common_state = ResidentCommonState(
            species_helpers=species_helpers,
            ancestors_T=ancestors_T,
            unnorm_row_max=dataset.unnorm_row_max.to(
                device=dataset.device,
                dtype=dataset.dtype,
            ),
        )

        if self._batched_resident:
            rust_specs_kwargs = dict(
                mode=mode,
                family_chunk_size=self.family_chunk_size,
                clade_budget=self.clade_budget,
                batch_packing=self.batch_packing,
                max_wave_size=max_wave_size,
                max_root_wave_size=max_root_wave_size,
                max_dts_partial_rows=max_dts_partial_rows,
                small_family_max_leaves=self.small_family_max_leaves,
            )
            rust_specs_handle = _start_batch_specs_from_retained_rust(
                dataset,
                **rust_specs_kwargs,
            )
            resident_warmup = None
            try:
                if (
                    self.lazy_preprocess
                    and self.prefetch_batches != 0
                    and rust_specs_handle is not None
                ):
                    resident_warmup = _start_resident_uniform_kernel_warmup_impl(
                        species_helpers,
                        ancestors_T,
                        dtype=dataset.dtype,
                        device=dataset.device,
                    )
                rust_specs = _finish_batch_specs_from_retained_rust(
                    rust_specs_handle,
                )
                rust_specs_handle = None
            finally:
                _finish_cuda_context_warmup_impl(resident_warmup)
                _cancel_batch_specs_from_retained_rust(rust_specs_handle)
            if rust_specs is None:
                self._family_schedule_stats = _build_family_schedule_stats(
                    dataset,
                    batch_packing=self.batch_packing,
                    small_family_max_leaves=self.small_family_max_leaves,
                )
                self._batch_specs = _build_batch_specs(
                    dataset,
                    mode=mode,
                    family_chunk_size=self.family_chunk_size,
                    clade_budget=self.clade_budget,
                    batch_packing=self.batch_packing,
                    max_wave_size=max_wave_size,
                    max_root_wave_size=max_root_wave_size,
                    max_dts_partial_rows=max_dts_partial_rows,
                    small_family_max_leaves=self.small_family_max_leaves,
                    schedule_stats=self._family_schedule_stats,
                )
            else:
                self._batch_specs = rust_specs
            self.batch_metadata = [spec.metadata for spec in self._batch_specs]
            if not self._batch_specs:
                raise ValueError("GeneReconModel requires at least one family")
            self._resident_cache = ResidentBatchCache(
                specs=self._batch_specs,
                build_static=self._build_batch_static,
                prefetch_batches=self.prefetch_batches,
            )
            self._resident_cache.ensure(0)
            if self.lazy_preprocess:
                self._resident_cache.schedule_prefetch()
            else:
                for batch_idx in range(1, len(self._batch_specs)):
                    self._resident_cache.ensure(batch_idx)
        else:
            self._resident_cache = None
            self._static = _build_static_state_impl(
                dataset,
                origination_prior=self._origination_prior,
                common_state=self._resident_common_state,
                settings=self._settings,
                max_wave_size=max_wave_size,
                max_root_wave_size=max_root_wave_size,
                max_dts_partial_rows=max_dts_partial_rows,
            )
            self.batch_metadata = [
                _metadata_for_full_static(dataset, mode=mode, static=self._static)
            ]
            self._apply_pi_adjoint_warmstart_config(self._static, clear_cache=False)

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_trees(
        cls,
        species_tree: str | os.PathLike[str],
        gene_trees: Sequence[
            str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
        ],
        *,
        mode: str = "global",
        device: Any = "cuda",
        dtype: torch.dtype = torch.float32,
        theta_init_rates: Optional[tuple[float, float, float]] = None,
        preprocess_cpu_cores: int | None = None,
        **solver_kwargs,
    ) -> "GeneReconModel":
        """One-liner: Newick paths → ready-to-optimize model.

        Parameters
        ----------
        species_tree : str
            Path to one rooted binary species tree in the supported simple
            Newick subset.
        gene_trees : list[str]
            Paths to gene trees in the supported simple Newick subset.  Each
            file may contain one or more semicolon-delimited records, and the
            final record may omit its terminal semicolon.  Records supplied for
            one family are amalgamated into that family's CCP.
            Branch lengths are ignored, and gene multifurcations are
            right-binarized by preprocessing.
            Leaf labels use the legacy species-prefix fallback: ``Species_gene``
            maps to species ``Species``, and labels without ``_`` map to the
            full label.  Use ``from_alerax_families()`` or lower-level dataset
            construction with explicit ``leaf_species_maps`` for labels that do
            not follow this convention; the README documents this as a narrow
            low-level ``GeneDataset`` exception rather than a general
            ``gpurec.core`` stability promise.
        mode : str
            "global" | "specieswise" | "genewise".
        device : str | torch.device
            Target device. Defaults to ``"cuda"``.
        dtype : torch.dtype
            Floating-point dtype. ``torch.float32`` is the default; switch to
            ``torch.float64`` if optimization stalls due to precision.
        theta_init_rates : (D, L, T) | None
            Optional natural-space initial rates. If ``None``, the dataset
            default of ``log2(1e-10)`` is used (matching GeneDataset).
        preprocess_cpu_cores : int | None
            Optional worker thread count for CPU preprocessing.
        """
        inputs = build_from_trees_inputs(
            species_tree=species_tree,
            gene_trees=gene_trees,
            mode=mode,
            device=device,
            dtype=dtype,
            theta_init_rates=theta_init_rates,
            preprocess_cpu_cores=preprocess_cpu_cores,
            solver_kwargs=solver_kwargs,
        )
        return cls(
            dataset=inputs.dataset,
            mode=inputs.mode,
            theta_init=inputs.theta_init,
            **inputs.solver_kwargs,
        )

    @classmethod
    def from_alerax_families(
        cls,
        species_tree: str,
        families_file: str | os.PathLike,
        *,
        mode: str = "global",
        start: int = 0,
        max_families: int | None = None,
        device: Any = "cuda",
        dtype: torch.dtype = torch.float32,
        theta_init_rates: Optional[tuple[float, float, float]] = None,
        preprocess_cpu_cores: int | None = None,
        **solver_kwargs,
    ) -> "GeneReconModel":
        """Build from an AleRax ``[FAMILIES]`` file with CCP/tree samples.

        Referenced species and gene tree files use the same supported simple
        Newick subset as :meth:`from_trees`. ``preprocess_cpu_cores``
        optionally fixes the worker thread count for CPU preprocessing.
        """
        inputs = build_from_alerax_families_inputs(
            species_tree=species_tree,
            families_file=families_file,
            mode=mode,
            start=start,
            max_families=max_families,
            device=device,
            dtype=dtype,
            theta_init_rates=theta_init_rates,
            preprocess_cpu_cores=preprocess_cpu_cores,
            solver_kwargs=solver_kwargs,
        )
        return cls(
            dataset=inputs.dataset,
            mode=inputs.mode,
            theta_init=inputs.theta_init,
            **inputs.solver_kwargs,
        )

    # ──────────────────────────────────────────────────────────────────
    # Resident batch management
    # ──────────────────────────────────────────────────────────────────
    def _build_batch_static(self, batch_idx: int) -> ReconStaticState:
        static = _build_batch_static_state_impl(
            self._batch_specs[batch_idx],
            dataset=self._dataset,
            common_state=self._resident_common_state,
            origination_prior=self._origination_prior.select_families(
                self._batch_specs[batch_idx].family_indices,
            ),
            settings=self._settings,
        )
        self._apply_pi_adjoint_warmstart_config(static, clear_cache=False)
        return static

    def _shutdown_prefetch_executor_for_replan(self) -> None:
        cache = getattr(self, "_resident_cache", None)
        if cache is not None:
            cache.close()
        self._resident_cache = None

    def replan_resident_batches(
        self,
        family_indices: Sequence[int],
    ) -> list[BatchMetadata]:
        """Rebuild resident batch specs for selected genewise family rows.

        This is an internal workflow hook for adaptive genewise optimization.
        It reuses preprocessed family payloads and cached per-family scheduler
        stats, then asks the Rust planner/scheduler to regroup and regenerate
        waves for the selected original family indices.
        """
        if not self._batched_resident or self._mode != "genewise":
            raise RuntimeError(
                "replan_resident_batches() requires genewise resident-batch mode"
            )
        indices = [integer_value("family_indices entries", value) for value in family_indices]
        if not indices:
            raise ValueError("family_indices must not be empty")
        seen: set[int] = set()
        for index in indices:
            if index < 0 or index >= self.n_families:
                raise IndexError(
                    f"family index {index} out of range for {self.n_families} families"
                )
            if index in seen:
                raise ValueError(f"duplicate family index {index}")
            seen.add(index)
        if self._family_schedule_stats is None:
            raise RuntimeError("family scheduler stats are not available")

        self._shutdown_prefetch_executor_for_replan()
        specs = _build_batch_specs(
            self._dataset,
            mode=self._mode,
            family_chunk_size=self.family_chunk_size,
            clade_budget=self.clade_budget,
            batch_packing=self.batch_packing,
            max_wave_size=self.max_wave_size,
            max_root_wave_size=self.max_root_wave_size,
            max_dts_partial_rows=self.max_dts_partial_rows,
            small_family_max_leaves=self.small_family_max_leaves,
            family_indices=indices,
            schedule_stats=self._family_schedule_stats,
        )
        if not specs:
            raise ValueError("replanned resident batches must not be empty")
        self._batch_specs = specs
        self.batch_metadata = [spec.metadata for spec in specs]
        self._current_batch_index = 0
        self._resident_cache = ResidentBatchCache(
            specs=self._batch_specs,
            build_static=self._build_batch_static,
            prefetch_batches=self.prefetch_batches,
        )
        self._resident_cache.ensure(0)
        self._resident_cache.schedule_prefetch()
        return list(self.batch_metadata)

    def _ensure_batch_static(self, batch_idx: int) -> ReconStaticState:
        if not self._batched_resident:
            if self._static is None:
                raise RuntimeError("resident static state has not been built")
            return self._static
        if batch_idx < 0 or batch_idx >= len(self._batch_specs):
            raise IndexError(
                f"batch index {batch_idx} out of range for {len(self._batch_specs)} batches"
            )
        cache = getattr(self, "_resident_cache", None)
        if cache is not None:
            return cache.ensure(batch_idx)
        raise RuntimeError("resident cache is not initialized")

    def _submit_prefetch(self, batch_idx: int) -> None:
        if batch_idx < 0 or batch_idx >= len(self._batch_specs):
            return
        cache = getattr(self, "_resident_cache", None)
        if cache is not None:
            cache.submit_prefetch(batch_idx)
            return
        if hasattr(self, "_batch_statics") or getattr(self, "_prefetch_closed", False):
            self._submit_legacy_prefetch(batch_idx)
            return
        raise RuntimeError("resident cache is not initialized")

    def _schedule_prefetch(self) -> None:
        if not self._batched_resident or self.prefetch_batches == 0:
            return
        cache = getattr(self, "_resident_cache", None)
        if cache is not None:
            cache.schedule_prefetch()
            return
        if hasattr(self, "_batch_statics") or getattr(self, "_prefetch_closed", False):
            self._schedule_legacy_prefetch()
            return
        raise RuntimeError("resident cache is not initialized")

    def _legacy_prefetch_guard(self):
        lock = getattr(self, "_batch_lock", None)
        return lock if lock is not None else nullcontext()

    def _submit_legacy_prefetch(self, batch_idx: int) -> None:
        if getattr(self, "_prefetch_closed", False):
            return
        statics = getattr(self, "_batch_statics", None)
        futures = getattr(self, "_batch_futures", None)
        if statics is None:
            raise RuntimeError("resident cache is not initialized")
        if futures is None:
            futures = {}
            self._batch_futures = futures
        if batch_idx < 0 or batch_idx >= len(statics):
            return
        with self._legacy_prefetch_guard():
            if statics[batch_idx] is not None or batch_idx in futures:
                return
            executor = getattr(self, "_prefetch_executor", None)
            if executor is None:
                executor = ThreadPoolExecutor(
                    max_workers=_RESIDENT_PREFETCH_WORKERS,
                    thread_name_prefix="gpurec-preprocess",
                )
                self._prefetch_executor = executor
            futures[batch_idx] = executor.submit(self._build_batch_static, batch_idx)

    def _schedule_legacy_prefetch(self) -> None:
        if getattr(self, "_prefetch_closed", False):
            return
        statics = getattr(self, "_batch_statics", None)
        if statics is None:
            raise RuntimeError("resident cache is not initialized")
        start = self._current_batch_index + 1
        if self.prefetch_batches == "all":
            stop = len(statics)
        else:
            stop = min(len(statics), start + int(self.prefetch_batches))
        for batch_idx in range(start, stop):
            self._submit_legacy_prefetch(batch_idx)

    def _active_static(self) -> ReconStaticState:
        if self._batched_resident:
            return self._ensure_batch_static(self._current_batch_index)
        if self._static is None:
            raise RuntimeError("resident static state has not been built")
        return self._static

    def _theta_for_batch_index(
        self,
        batch_idx: int,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        if not self._batched_resident or self._mode != "genewise":
            return theta
        idx = torch.as_tensor(
            self._batch_specs[batch_idx].family_indices,
            dtype=torch.long,
            device=theta.device,
        )
        return theta.index_select(0, idx)

    def _active_theta(self, theta: torch.Tensor | None = None) -> torch.Tensor:
        return self._theta_for_batch_index(
            self._current_batch_index,
            self.theta if theta is None else theta,
        )

    def _stream_full_batches(
        self,
        theta: torch.Tensor,
        *,
        need_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return _stream_full_batches_impl(self, theta, need_grad=need_grad)

    @property
    def current_batch_metadata(self) -> BatchMetadata:
        """Metadata for the resident batch currently selected by the model."""
        return self.batch_metadata[self._current_batch_index]

    @property
    def current_batch_index(self) -> int:
        """Index of the resident batch currently selected for evaluation."""
        return self._current_batch_index

    @property
    def mode(self) -> str:
        """Parameter-sharing mode: ``global``, ``specieswise``, or ``genewise``."""
        return self._mode

    @property
    def family_names(self) -> list[str]:
        """Family names in dataset order."""
        return list(self._dataset.family_names)

    @property
    def species_tree_path(self) -> Path:
        """Path to the species tree used to construct the model."""
        return Path(self._dataset.species_tree_path)

    @property
    def n_families(self) -> int:
        """Number of gene families in the model dataset."""
        return len(self._dataset.families)

    @property
    def species_names(self) -> list[str]:
        """Species names in the internal species-index order."""
        return [str(name) for name in self._dataset.species_helpers["names"]]

    @property
    def cached_static_states(self) -> list[ReconStaticState]:
        """Static states that are currently built and available for diagnostics."""
        if self._batched_resident:
            cache = getattr(self, "_resident_cache", None)
            if cache is not None:
                return cache.cached()
            return []
        return [] if self._static is None else [self._static]

    def drop_cached_static_states(self) -> None:
        """Release built resident batch static states while keeping batch metadata."""
        if not self._batched_resident:
            return
        cache = getattr(self, "_resident_cache", None)
        if cache is not None:
            cache.drop_statics()
            return

    def materialize_batches(self) -> list[BatchMetadata]:
        """Build all resident batch static states and return metadata copies.

        In resident-batch mode this forces every batch static state to be built
        before returning.  The returned list is a copy of ``batch_metadata``, so
        callers can inspect batch ownership without mutating model bookkeeping.
        """
        if self._batched_resident:
            cache = getattr(self, "_resident_cache", None)
            if cache is not None:
                for batch_idx in range(len(self._batch_specs)):
                    cache.ensure(batch_idx)
                return list(self.batch_metadata)
            for batch_idx in range(len(self._batch_specs)):
                self._ensure_batch_static(batch_idx)
        elif self._static is None:
            raise RuntimeError("resident static state has not been built")
        return list(self.batch_metadata)

    def _apply_pi_adjoint_warmstart_config(
        self,
        static: ReconStaticState,
        *,
        clear_cache: bool,
    ) -> None:
        static.pi_adjoint_warmstart = bool(self._pi_adjoint_warmstart)
        static.pi_adjoint_cache_update_mode = self._pi_adjoint_cache_update_mode
        static.pi_fixed_point_relaxation = self._pi_fixed_point_relaxation
        if clear_cache:
            _clear_pi_adjoint_runtime_cache(static)

    def configure_pi_adjoint_warmstart(
        self,
        *,
        enabled: bool,
        cache_update_mode: str = "immediate",
        pi_fixed_point_relaxation: float | None = None,
    ) -> None:
        """Update Pi-adjoint warm-start policy on defaults and built batches."""
        warmstart = bool_value("pi_adjoint_warmstart", enabled)
        cache_mode = _normalize_pi_adjoint_cache_update_mode(cache_update_mode)
        if pi_fixed_point_relaxation is not None:
            pi_fixed_point_relaxation = positive_float(
                "pi_fixed_point_relaxation",
                pi_fixed_point_relaxation,
            )
        self._pi_adjoint_warmstart = warmstart
        self._pi_adjoint_cache_update_mode = cache_mode
        if pi_fixed_point_relaxation is not None:
            self._pi_fixed_point_relaxation = pi_fixed_point_relaxation
        for static in self.cached_static_states:
            self._apply_pi_adjoint_warmstart_config(static, clear_cache=True)

    def configure_solver_iterations(
        self,
        *,
        fixed_iters_E: int | None | object = _UNSET,
        fixed_iters_Pi: int | None = None,
        neumann_terms: int | None = None,
        pi_max_diff_tol: float | None = None,
        gradient_change_tol: float | None = None,
        adaptive_neumann_terms: bool | None = None,
    ) -> None:
        """Update solver iteration controls on the model and built batches.

        The method updates model defaults and resident batch static states that
        are already built.  It does not cancel or rewrite pending background
        prefetch work; configure before scheduling lazy prefetch, or materialize
        resident batches and configure again when all batches should share the
        new controls.
        """
        if fixed_iters_E is not _UNSET:
            if fixed_iters_E is not None:
                fixed_iters_E = positive_int("fixed_iters_E", fixed_iters_E)
            self._fixed_iters_E = fixed_iters_E
        if fixed_iters_Pi is not None:
            fixed_iters_Pi = positive_even_int("fixed_iters_Pi", fixed_iters_Pi)
            self._fixed_iters_Pi = fixed_iters_Pi
        if neumann_terms is not None:
            neumann_terms = positive_int("neumann_terms", neumann_terms)
            self._neumann_terms = neumann_terms
        if pi_max_diff_tol is not None:
            pi_max_diff_tol = nonnegative_float("pi_max_diff_tol", pi_max_diff_tol)
            self._pi_max_diff_tol = pi_max_diff_tol
        if gradient_change_tol is not None:
            gradient_change_tol = nonnegative_float(
                "gradient_change_tol",
                gradient_change_tol,
            )
            self._gradient_change_tol = gradient_change_tol
        if adaptive_neumann_terms is not None:
            adaptive_neumann_terms = disabled_adaptive_neumann_terms_value(
                adaptive_neumann_terms
            )
            self._adaptive_neumann_terms = adaptive_neumann_terms

        for static in self.cached_static_states:
            if fixed_iters_E is not _UNSET:
                static.fixed_iters_E = fixed_iters_E
            if fixed_iters_Pi is not None:
                static.fixed_iters_Pi = fixed_iters_Pi
            if neumann_terms is not None:
                static.neumann_terms = neumann_terms
            if pi_max_diff_tol is not None:
                static.pi_max_diff_tol = pi_max_diff_tol
            if gradient_change_tol is not None:
                static.gradient_change_tol = gradient_change_tol
            if adaptive_neumann_terms is not None:
                static.adaptive_neumann_terms = adaptive_neumann_terms

    def solver_stat_records(self) -> list[dict[str, Any]]:
        """Copies of solver stats from already-built static states."""
        records: list[dict[str, Any]] = []
        for static in self.cached_static_states:
            stats = static.last_solver_stats
            if stats is not None:
                records.append(dict(stats))
        return records

    def family_input(self, family_index: int) -> FamilyInput:
        family_index = integer_value("family_index", family_index)
        if family_index < 0 or family_index >= self.n_families:
            raise IndexError(
                f"family_index {family_index} outside 0..{self.n_families}"
            )
        ensure_full = getattr(self._dataset, "_ensure_full_families", None)
        if callable(ensure_full):
            ensure_full()
        family = self._dataset.families[family_index]
        leaf_map = self._dataset.leaf_species_maps[family_index] or {}
        return FamilyInput(
            index=family_index,
            name=self._dataset.family_names[family_index],
            gene_tree_paths=list(self._dataset.gene_tree_paths[family_index]),
            leaf_species_map=dict(leaf_map),
            clade_count=int(family["C"]),
            split_count=int(family["N_splits"]),
            root_clade_id=int(family["root_clade_id"]),
            ccp_helpers=_public_family_value(family["ccp_helpers"]),
            leaf_row_index=_public_family_value(family["leaf_row_index"]),
            leaf_col_index=_public_family_value(family["leaf_col_index"]),
            clade_leaf_labels=list(family.get("clade_leaf_labels", [])),
        )

    def active_theta(self, theta: torch.Tensor | None = None) -> torch.Tensor:
        """Return theta as addressed by the currently selected resident batch."""
        return self._active_theta(theta)

    def select_batch(self, batch_index: int) -> BatchMetadata:
        """Select a resident batch and return its metadata.

        In non-batched mode only batch ``0`` exists.  Selecting a new batch
        clears warm runtime state from the previous active batch.
        """
        batch_index = integer_value("batch_index", batch_index)
        if batch_index < 0 or batch_index >= len(self.batch_metadata):
            raise IndexError(
                f"batch index {batch_index} out of range for {len(self.batch_metadata)} batches"
            )
        if batch_index != self._current_batch_index:
            self.clear()
            self._current_batch_index = batch_index
        cache = getattr(self, "_resident_cache", None)
        if cache is not None and self._batched_resident:
            cache.select(batch_index)
        else:
            self._ensure_batch_static(batch_index)
        self._schedule_prefetch()
        return self.current_batch_metadata

    def activate_family(self, family_index: int) -> ActiveFamilyBatch:
        """Select the resident batch containing ``family_index``.

        Returns the family offset inside the active Pi matrix plus the local
        family index used by batch-local parameter tensors.
        """
        family_index = integer_value("family_index", family_index)
        if family_index < 0 or family_index >= self.n_families:
            raise IndexError(
                f"family_index {family_index} outside 0..{self.n_families}"
            )

        if not self._batched_resident:
            offset = 0
            for idx in range(family_index):
                offset += int(self._dataset.families[idx]["C"])
            metadata = self.select_batch(0)
            return ActiveFamilyBatch(
                family_index=family_index,
                batch_index=0,
                local_family_index=family_index,
                clade_offset=offset,
                metadata=metadata,
            )

        for batch_idx, metadata in enumerate(self.batch_metadata):
            family_indices = [int(idx) for idx in metadata.family_indices]
            if family_index not in family_indices:
                continue
            offset = 0
            for local_idx, idx in enumerate(family_indices):
                if idx == family_index:
                    metadata = self.select_batch(batch_idx)
                    return ActiveFamilyBatch(
                        family_index=family_index,
                        batch_index=batch_idx,
                        local_family_index=local_idx,
                        clade_offset=offset,
                        metadata=metadata,
                    )
                offset += int(self._dataset.families[idx]["C"])
        raise IndexError(f"family_index {family_index} is not present in any resident batch")

    def next(self) -> BatchMetadata:
        """Advance to the next resident batch and return its metadata."""
        if self._current_batch_index + 1 >= len(self.batch_metadata):
            raise StopIteration("already at the final resident batch")
        return self.select_batch(self._current_batch_index + 1)

    def clear(self) -> None:
        """Release active runtime caches held by the model."""
        if self._batched_resident:
            cache = getattr(self, "_resident_cache", None)
            if cache is not None:
                cache.clear_active_runtime()
                static = cache.statics[cache.current_index]
                if static is not None:
                    _clear_pi_adjoint_runtime_cache(static)
                return
            statics = getattr(self, "_batch_statics", None)
            if statics is not None:
                batch_idx = getattr(self, "_current_batch_index", 0)
                if 0 <= batch_idx < len(statics):
                    static = statics[batch_idx]
                    if static is not None:
                        static.warm_E = None
                        _clear_pi_adjoint_runtime_cache(static)
                return
            raise RuntimeError("resident cache is not initialized")
            return
        static = self._active_static()
        static.warm_E = None
        _clear_pi_adjoint_runtime_cache(static)

    def close(self) -> None:
        """Stop background batch preprocessing and drop pending futures."""
        self._prefetch_closed = True
        cache = getattr(self, "_resident_cache", None)
        if cache is not None:
            cache.close()
            self._resident_cache = None
        self._close_legacy_prefetch_executor()

    def _close_legacy_prefetch_executor(self) -> None:
        executor = getattr(self, "_prefetch_executor", None)
        lock = getattr(self, "_batch_lock", None)
        if lock is None:
            self._prefetch_executor = None
            futures = getattr(self, "_batch_futures", None)
            if futures is not None:
                futures.clear()
        else:
            with lock:
                self._prefetch_executor = None
                futures = getattr(self, "_batch_futures", None)
                if futures is not None:
                    futures.clear()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────
    # Likelihood / loss
    # ──────────────────────────────────────────────────────────────────
    def forward(self, reduce: str = "sum") -> torch.Tensor:
        """Returns negative log-likelihood (a loss).

        ``reduce="sum"`` (default) returns a scalar. ``reduce="per_family"``
        returns differentiable per-family losses in genewise mode and a
        no-grad diagnostic vector in shared-theta modes.
        """
        if reduce not in ("sum", "per_family"):
            raise ValueError(f"reduce must be 'sum' or 'per_family', got {reduce!r}")
        if reduce == "per_family" and self._mode != "genewise":
            if torch.is_grad_enabled():
                raise ValueError(
                    "reduce='per_family' is only differentiable in genewise mode."
                )
            return self._forward_per_family_inference()
        theta = self._active_theta()
        if not torch.is_grad_enabled() or not theta.requires_grad:
            return self._forward_no_grad(per_family=(reduce == "per_family"))
        static = self._active_static()
        return _GeneReconFunction.apply(theta, static, reduce)

    @torch.no_grad()
    def _forward_no_grad(self, *, per_family: bool) -> torch.Tensor:
        theta = self._active_theta()
        out = evaluate_resident_no_grad(
            self._active_static(),
            theta,
            per_family=per_family,
        )
        return out.to(device=theta.device, dtype=theta.dtype)

    @torch.no_grad()
    def _forward_per_family_inference(self) -> torch.Tensor:
        """Per-family diagnostic NLL for shared-theta modes."""
        return self._forward_no_grad(per_family=True)

    def full_loss(self) -> torch.Tensor:
        """Stream every resident batch and return the exact full NLL."""
        if not self._batched_resident:
            return self.forward(reduce="sum")
        if not torch.is_grad_enabled() or not self.theta.requires_grad:
            loss, _grad = self._stream_full_batches(self.theta, need_grad=False)
            return loss.to(device=self.theta.device, dtype=self.theta.dtype)
        return _GeneReconFullLossFunction.apply(self.theta, self)

    def full_loss_for_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Stream every resident batch using an explicit theta tensor.

        When gradients are enabled and ``theta`` requires gradients, the method
        uses the gradient-producing full-batch streaming path.  Under
        ``torch.no_grad()`` or with a non-differentiable tensor, it uses the
        loss-only streaming path.
        """
        theta = validate_theta_shape(
            "theta",
            theta,
            mode=self._mode,
            species_count=int(self._dataset.S),
            family_count=len(self._dataset.families),
            device=self._dataset.device,
            dtype=self._dataset.dtype,
        )
        if torch.is_grad_enabled() and theta.requires_grad:
            return _GeneReconFullLossFunction.apply(theta, self)
        with torch.no_grad():
            loss, _grad = self._stream_full_batches(theta, need_grad=False)
        return loss.to(device=theta.device, dtype=theta.dtype)

    def nll_per_family(self) -> torch.Tensor:
        """Per-family NLL ``[G]``. Only valid in genewise mode."""
        if self._mode != "genewise":
            raise ValueError(
                "nll_per_family() is only valid in genewise mode; in global / "
                "specieswise mode all families share theta, so independent "
                "per-family gradients are not defined."
            )
        return self.forward(reduce="per_family")

    @torch.no_grad()
    def full_genewise_nll_and_grad(
        self,
        *,
        need_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stream genewise per-family NLLs and optional independent gradients.

        This is the model-owned public surface for row-wise optimizers that need
        one loss and one gradient vector per gene family.
        """
        if self._mode != "genewise":
            raise ValueError(
                "full_genewise_nll_and_grad() is only valid in genewise mode"
            )

        values = torch.empty(
            (self.n_families,),
            device=self.theta.device,
            dtype=self.theta.dtype,
        )
        grad_total = torch.zeros_like(self.theta) if need_grad else None

        if not self._batched_resident:
            loss, grad = _evaluate_static_state(
                self._active_static(),
                self.theta,
                need_grad=need_grad,
                per_family=True,
            )
            loss = _validate_genewise_loss_vector(
                "genewise per-family NLL",
                loss,
                family_count=self.n_families,
            )
            values.copy_(loss.to(device=values.device, dtype=values.dtype))
            if need_grad:
                if grad is None or grad_total is None:
                    raise RuntimeError("internal error: missing genewise gradient")
                grad = _validate_genewise_gradient_matrix(
                    "genewise gradient",
                    grad,
                    expected_shape=tuple(int(dim) for dim in grad_total.shape),
                )
                grad_total.copy_(grad.to(device=grad_total.device, dtype=grad_total.dtype))
            return values, grad_total

        previous_batch = self.current_batch_index
        try:
            for batch_idx, metadata in enumerate(self.batch_metadata):
                self.select_batch(batch_idx)
                static = self._active_static()
                theta_batch = self._active_theta()
                batch_values, batch_grad = _evaluate_static_state(
                    static,
                    theta_batch,
                    need_grad=need_grad,
                    per_family=True,
                )
                batch_values = _validate_genewise_loss_vector(
                    "genewise batch per-family NLL",
                    batch_values,
                    family_count=len(metadata.family_indices),
                )
                idx = torch.as_tensor(
                    metadata.family_indices,
                    dtype=torch.long,
                    device=values.device,
                )
                values.index_copy_(
                    0,
                    idx,
                    batch_values.to(device=values.device, dtype=values.dtype),
                )
                if need_grad:
                    if batch_grad is None or grad_total is None:
                        raise RuntimeError("internal error: missing genewise batch gradient")
                    batch_grad = _validate_genewise_gradient_matrix(
                        "genewise batch gradient",
                        batch_grad,
                        expected_shape=tuple(int(dim) for dim in theta_batch.shape),
                    )
                    grad_total.index_copy_(
                        0,
                        idx.to(device=grad_total.device),
                        batch_grad.to(device=grad_total.device, dtype=grad_total.dtype),
                    )
        finally:
            self.select_batch(previous_batch)
        return values, grad_total

    @torch.no_grad()
    def full_nll_per_family(self) -> torch.Tensor:
        """Per-family NLL for every family in genewise mode.

        This is the no-gradient companion to ``full_genewise_nll_and_grad()``.
        In global or specieswise mode, use ``forward(reduce="per_family")``
        under ``torch.no_grad()`` for diagnostic shared-theta values; those
        modes do not have independent per-family gradients.
        """
        if self._mode != "genewise":
            raise ValueError(
                "full_nll_per_family() is only valid in genewise mode; use "
                "forward(reduce='per_family') under torch.no_grad() for "
                "shared-theta diagnostic values."
            )
        values, _grad = self.full_genewise_nll_and_grad(need_grad=False)
        return values

    @torch.no_grad()
    def reconciliation_state(self, *, original_order: bool = True) -> ReconciliationState:
        """Solve E/Pi for the currently selected batch and return export state.

        The model mode controls parameter sharing:
        ``global`` uses one theta vector, ``specieswise`` uses ``[S, 3]``
        theta, and ``genewise`` uses ``[G, 3]`` theta addressed by the cached
        family index.  In memory-safe resident-batch mode, this returns tensors
        for the active batch; otherwise it returns the full resident state.
        """
        static = self._active_static()
        theta = self._active_theta()
        export_state = evaluate_resident_export_state(
            static,
            theta,
            original_order=original_order,
        )
        solve = export_state.solve
        return ReconciliationState(
            e=solve.e_out["E"],
            pi=export_state.pi,
            pibar=export_state.pibar,
            ebar=solve.e_out["E_bar"],
            log_p_s=solve.log_p_s,
            log_p_d=solve.log_p_d,
            log_p_l=solve.log_p_l,
            max_transfer=solve.max_transfer,
            origination_prior=static.origination_prior,
            origination_probs=static.origination_probs,
        )

    @torch.no_grad()
    def pi_matrix(self, *, original_order: bool = True) -> torch.Tensor:
        """Return converged Pi rows for the retained uniform-transfer path."""
        return self.reconciliation_state(original_order=original_order).pi

    # ──────────────────────────────────────────────────────────────────
    # Parameter management
    # ──────────────────────────────────────────────────────────────────
    def clamp_theta_(
        self,
        min_rate: float = 1e-10,
        max_rate: Optional[float] = None,
    ) -> None:
        """In-place safety floor on theta to prevent rate underflow.

        Useful after ordinary PyTorch optimizer steps to keep rates in a
        numerically valid range.
        """
        min_rate = positive_float("min_rate", min_rate)
        if max_rate is not None:
            max_rate = positive_float("max_rate", max_rate)
        if max_rate is not None and max_rate < min_rate:
            raise ValueError("max_rate must be greater than or equal to min_rate")
        with torch.no_grad():
            self.theta.clamp_(
                min=math.log2(min_rate),
                max=None if max_rate is None else math.log2(max_rate),
            )

    @property
    def n_species(self) -> int:
        """Number of species in the model species tree."""
        return int(self._dataset.S)

    @property
    def static(self) -> ReconStaticState:
        """Read-only access to the cached static state (for advanced use)."""
        return self._active_static()
