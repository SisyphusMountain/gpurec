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

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from gpurec.core.model import GeneDataset
from gpurec.core.origination import (
    OriginationPrior,
    PreparedOriginationPrior,
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
    _family_index_chunks as _family_index_chunks_impl,
)
from ._model_controls import _GeneReconModelControlsMixin
from ._model_resident_batches import _GeneReconModelResidentBatchMixin
from ._model_builders import (
    build_from_alerax_families_inputs,
    build_from_trees_inputs,
)
from ._model_init import apply_model_init_state, prepare_model_init
from ._model_types import (
    ActiveFamilyBatch as ActiveFamilyBatch,
    BatchMetadata as BatchMetadata,
    FamilyInput as FamilyInput,
    ReconciliationState,
)
from ._resident_runtime import initialize_resident_state as _initialize_resident_state_impl
from ._static_builder import _metadata_for_full_static as _metadata_for_full_static_impl
from ._theta_init import (
    _default_theta_init as _default_theta_init_impl,
    _mode_to_flags as _mode_to_flags_impl,
    _normalize_mode as _normalize_mode_impl,
    _validate_gene_dtype as _validate_gene_dtype_impl,
)
from ._theta_constraints import clamp_theta_rates_
from ._uniform_evaluator import (
    evaluate_resident_export_state,
    evaluate_resident_no_grad,
    evaluate_resident_static_state as _evaluate_static_state_impl,
)
from ._genewise_streaming import (
    full_genewise_nll_and_grad as _full_genewise_nll_and_grad_impl,
    full_nll_per_family as _full_nll_per_family_impl,
)
from .autograd import _GeneReconFunction, ReconStaticState
from ._validation import validate_theta_shape


# Canonical helper implementation aliases from the split modules.
_normalize_mode = _normalize_mode_impl
_mode_to_flags = _mode_to_flags_impl
_validate_gene_dtype = _validate_gene_dtype_impl
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
_family_index_chunks = _family_index_chunks_impl
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


class GeneReconModel(
    _GeneReconModelControlsMixin,
    _GeneReconModelResidentBatchMixin,
    torch.nn.Module,
):
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
        init_state = prepare_model_init(
            dataset=dataset,
            mode=mode,
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
            theta_init=theta_init,
            max_wave_size=max_wave_size,
            max_root_wave_size=max_root_wave_size,
            max_dts_partial_rows=max_dts_partial_rows,
            family_chunk_size=family_chunk_size,
            clade_budget=clade_budget,
            batch_packing=batch_packing,
            small_family_max_leaves=small_family_max_leaves,
            lazy_preprocess=lazy_preprocess,
            prefetch_batches=prefetch_batches,
            pi_adjoint_warmstart=pi_adjoint_warmstart,
            pi_adjoint_cache_update_mode=pi_adjoint_cache_update_mode,
            pi_fixed_point_relaxation=pi_fixed_point_relaxation,
            shared_loss_batch_streams=shared_loss_batch_streams,
            origination_probs=origination_probs,
        )
        apply_model_init_state(self, init_state)
        _initialize_resident_state_impl(self)

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
        return _full_genewise_nll_and_grad_impl(self, need_grad=need_grad)

    @torch.no_grad()
    def full_nll_per_family(self) -> torch.Tensor:
        """Per-family NLL for every family in genewise mode.

        This is the no-gradient companion to ``full_genewise_nll_and_grad()``.
        In global or specieswise mode, use ``forward(reduce="per_family")``
        under ``torch.no_grad()`` for diagnostic shared-theta values; those
        modes do not have independent per-family gradients.
        """
        return _full_nll_per_family_impl(self)

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
        clamp_theta_rates_(self.theta, min_rate=min_rate, max_rate=max_rate)

    @property
    def n_species(self) -> int:
        """Number of species in the model species tree."""
        return int(self._dataset.S)

    @property
    def static(self) -> ReconStaticState:
        """Read-only access to the cached static state (for advanced use)."""
        return self._active_static()
