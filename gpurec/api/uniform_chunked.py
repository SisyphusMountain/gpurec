"""Chunked PyTorch interface for global/uniform reconciliation.

This module exposes the optimized uniform forward/backward pipeline as a
``torch.nn.Module`` that can be used with ordinary PyTorch optimizers.  Unlike
``GeneReconModel``, it does not build one resident wave layout for all gene
families.  It streams families through fixed resident chunks, accumulates the
shared global adjoints, and returns a scalar NLL whose backward pass supplies
the precomputed gradient with respect to the three global DTL rates.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from gpurec.core.batch_planning import (
    normalize_batch_packing,
    normalize_clade_budget,
)
from gpurec.core.memory_policy import UniformPipelinePolicy, choose_uniform_pipeline_policy
from gpurec.core.model import (
    GeneDataset,
    normalize_family_inputs,
    normalize_family_selection,
    parse_alerax_family_file,
)
from gpurec.core.origination import (
    OriginationPrior,
    PreparedOriginationPrior,
    prepare_origination_prior,
)

from ._uniform_chunked_layout import (
    _UniformBuiltChunk as _UniformBuiltChunk,
    _UniformChunkedState as _UniformChunkedState,
    _UniformChunkSpec as _UniformChunkSpec,
    _built_chunks_from_rust as _built_chunks_from_rust,
    _dtype_name_for_rust as _dtype_name_for_rust,
    _move_wave_layout_to_device as _move_wave_layout_to_device,
)
from ._uniform_chunked_inputs import (
    _normalize_uniform_solver_kwargs as _normalize_uniform_solver_kwargs,
    _selected_gene_paths as _selected_gene_paths,
    _validate_uniform_dtype as _validate_uniform_dtype,
)
from ._uniform_chunked_eval import (
    _PI_BACKWARD_COUNTER_KEYS as _PI_BACKWARD_COUNTER_KEYS,
    _PI_BACKWARD_TENSOR_KEYS as _PI_BACKWARD_TENSOR_KEYS,
    _UniformChunkedEvaluation as _UniformChunkedEvaluation,
    _UniformChunkedReadOnlyEvaluation as _UniformChunkedReadOnlyEvaluation,
    _UniformChunkStatsRow as _UniformChunkStatsRow,
    _chunk_stats_row as _chunk_stats_row,
    _e_adjoint_stats_fields as _e_adjoint_stats_fields,
    _evaluate_chunked_uniform as _evaluate_chunked_uniform,
    _evaluate_chunked_uniform_read_only as _evaluate_chunked_uniform_read_only,
    _evaluate_chunked_uniform_result as _evaluate_chunked_uniform_result,
    _new_pi_backward_accumulator as _new_pi_backward_accumulator,
    _require_chunked_gradient_dtype as _require_chunked_gradient_dtype,
    _root_count_tensor as _root_count_tensor,
    _selected_chunks as _selected_chunks,
    _time_cuda_ms as _time_cuda_ms,
)
from ._validation import (
    auto_int as _auto_int,
    auto_nonnegative_int as _auto_nonnegative_int,
    auto_positive_int as _auto_positive_int,
    bool_value,
    nonnegative_float,
    nonnegative_int_sequence as _normalize_nonnegative_int_sequence,
    optional_positive_int,
    positive_even_int,
    positive_int,
    positive_int_sequence as _normalize_positive_int_sequence,
    require_cuda_device,
    require_default_objective,
    theta_init_base_from_rates,
)
from ._theta_constraints import clamp_theta_rates_

_as_auto_int = _auto_int


@dataclass(frozen=True)
class UniformChunkMetadata:
    """Public metadata for one resident uniform chunk."""

    chunk_index: int
    family_indices: tuple[int, ...]
    family_names: tuple[str, ...]
    gene_tree_paths: tuple[tuple[str, ...], ...]
    family_count: int
    clade_count: int
    split_count: int
    wave_count: int
    max_wave_size: int
    split_rows: int
    max_wave_split_rows: int


class _UniformChunkedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, state: _UniformChunkedState):
        require_default_objective("UniformChunkedReconModel")
        with torch.no_grad():
            loss, grad_theta, stats = _evaluate_chunked_uniform(
                state,
                theta,
                need_grad=True,
            )
        if grad_theta is None:
            raise RuntimeError("internal error: missing chunked uniform gradient")
        ctx.save_for_backward(grad_theta.detach().to(device=theta.device, dtype=theta.dtype))
        return loss.to(device=theta.device, dtype=theta.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (grad_theta,) = ctx.saved_tensors
        return grad_theta * grad_output.to(device=grad_theta.device, dtype=grad_theta.dtype), None


class UniformChunkedReconModel(torch.nn.Module):
    """Global/uniform DTL model backed by the optimized chunked pipeline.

    The module has one trainable parameter, ``theta`` with shape ``[3]`` in
    log2-rate space.  ``forward()`` returns the summed negative log-likelihood
    in bits.  The forward call computes both the objective and the analytical
    gradient internally; the subsequent ``loss.backward()`` call simply returns
    that cached gradient to PyTorch.
    ``loss_and_grad()`` returns the same direct gradient with a stats dictionary
    that includes selected chunk/family counts, timing fields, gradient norm,
    and E-adjoint solve telemetry.
    ``nll_per_family(chunk_indices=...)`` is a no-grad global/uniform
    diagnostic that returns one shared-theta NLL per selected family after
    chunk filtering; it does not define independent per-family gradients.

    Tree inputs use the retained preprocessing parser's simple Newick subset:
    one rooted binary species tree, unquoted labels, ignored numeric branch
    lengths, and gene-tree files that may contain multiple semicolon-delimited
    records for CCP amalgamation.

    ``torch.float32`` and ``torch.float64`` are the supported production dtypes.
    ``torch.bfloat16`` is accepted only on this direct API as an experimental
    CUDA memory-saving path for forward/NLL probes; workflow configuration and
    CLI runs intentionally expose only fp32/fp64, and the retained Pi
    backward/gradient path does not support bf16.
    """

    def __init__(
        self,
        *,
        species_tree: str | os.PathLike[str],
        gene_trees: Sequence[str | os.PathLike[str] | Sequence[str | os.PathLike[str]]],
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        theta_init_rates: tuple[float, float, float] = (0.05, 0.05, 0.05),
        family_names: Sequence[str] | None = None,
        leaf_species_maps: Sequence[dict[str, str]] | None = None,
        preprocess_cpu_cores: int | None = None,
        family_chunk_size: int | str = "auto",
        max_wave_size: int | str | None = "auto",
        max_root_wave_size: int | None = None,
        clade_budget: int | None = None,
        batch_packing: str = "sequential",
        family_chunk_candidates: Sequence[int] = (25, 50, 10, 75, 100),
        max_wave_candidates: Sequence[int] = (8192, 16384, 4096, 32768),
        fixed_iters_Pi: int = 6,
        fixed_iters_E: int | None = None,
        max_iters_E: int = 2000,
        tol_E: float = 1e-8,
        neumann_terms: int = 3,
        use_pruning: bool = True,
        pruning_threshold: float = 1e-6,
        warm_start_E: bool = True,
        profile: bool = False,
        origination_probs: (
            torch.Tensor
            | Sequence[float]
            | OriginationPrior
            | PreparedOriginationPrior
            | None
        ) = None,
    ) -> None:
        super().__init__()
        require_default_objective("UniformChunkedReconModel")
        dtype = _validate_uniform_dtype(dtype)
        theta_init = theta_init_base_from_rates(
            theta_init_rates,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        if theta_init is None:
            raise ValueError("theta_init_rates must be provided")
        if fixed_iters_E is not None:
            fixed_iters_E = positive_int("fixed_iters_E", fixed_iters_E)
        fixed_iters_Pi = positive_even_int("fixed_iters_Pi", fixed_iters_Pi)
        max_iters_E = positive_int("max_iters_E", max_iters_E)
        neumann_terms = positive_int("neumann_terms", neumann_terms)
        tol_E = nonnegative_float("tol_E", tol_E)
        pruning_threshold = nonnegative_float("pruning_threshold", pruning_threshold)
        use_pruning = bool_value("use_pruning", use_pruning)
        warm_start_E = bool_value("warm_start_E", warm_start_E)
        profile = bool_value("profile", profile)
        preprocess_cpu_cores = optional_positive_int(
            "preprocess_cpu_cores",
            preprocess_cpu_cores,
        )
        chunk_value = _auto_nonnegative_int("family_chunk_size", family_chunk_size)
        wave_value = _auto_positive_int("max_wave_size", max_wave_size)
        max_root_wave_size = optional_positive_int(
            "max_root_wave_size",
            max_root_wave_size,
        )
        clade_budget = normalize_clade_budget(clade_budget)
        normalized_packing = normalize_batch_packing(batch_packing)
        family_chunk_candidates = _normalize_nonnegative_int_sequence(
            "family_chunk_candidates",
            family_chunk_candidates,
        )
        max_wave_candidates = _normalize_positive_int_sequence(
            "max_wave_candidates",
            max_wave_candidates,
        )
        family_tree_paths, normalized_family_names, normalized_leaf_maps = (
            normalize_family_inputs(
                gene_trees,
                family_names,
                leaf_species_maps,
            )
        )

        device = require_cuda_device(device, owner="UniformChunkedReconModel")
        theta_init = theta_init.to(device=device)

        from gpurec.core.preprocess_rust import RustPreprocessExtension

        families_input = {
            name: paths
            for name, paths in zip(normalized_family_names, family_tree_paths)
        }
        leaf_maps_input = {
            name: mapping
            for name, mapping in zip(normalized_family_names, normalized_leaf_maps)
            if mapping
        }
        rust_preprocessed = RustPreprocessExtension().preprocess_dataset(
            str(species_tree),
            families_input,
            leaf_species_maps=leaf_maps_input,
            include_species_matrices=False,
            num_threads=0 if preprocess_cpu_cores is None else preprocess_cpu_cores,
        )
        dataset = GeneDataset._from_preprocessed_raw(
            raw=rust_preprocessed.to_torch(),
            species_tree_path=str(species_tree),
            gene_tree_paths=family_tree_paths,
            genewise=False,
            specieswise=False,
            dtype=dtype,
            device=device,
            family_names=normalized_family_names,
            leaf_species_maps=normalized_leaf_maps,
        )
        species_helpers, ancestors_T = dataset._species_helpers_for_mode(
            device=device,
            dtype=dtype,
        )
        unnorm_row_max = dataset.unnorm_row_max.to(device=device, dtype=dtype)
        prepared_origination_prior = prepare_origination_prior(
            origination_probs,
            S=int(dataset.S),
            device=device,
            dtype=dtype,
            family_count=len(dataset.families) if origination_probs is not None else None,
        )

        rust_counts = (
            rust_preprocessed.family_counts()
            if normalized_packing == "depth_first_fit"
            else rust_preprocessed.family_basic_counts()
        )
        clade_counts = [int(value) for value in rust_counts["clade_counts"]]
        leaf_counts: list[int] | None = None
        nonleaf_counts: list[int] | None = None
        schedule_depths: list[int] | None = None
        if normalized_packing == "depth_first_fit":
            leaf_counts = [int(value) for value in rust_counts["leaf_counts"]]
            nonleaf_counts = [int(value) for value in rust_counts["nonleaf_counts"]]
            schedule_depths = [int(value) for value in rust_counts["schedule_depths"]]
        memory_policy: UniformPipelinePolicy | None = None
        if chunk_value == "auto" or wave_value == "auto":
            chunk_candidates = (
                family_chunk_candidates
                if chunk_value == "auto"
                else (int(chunk_value),)
            )
            wave_candidates = (
                max_wave_candidates
                if wave_value == "auto"
                else (sum(clade_counts) if wave_value is None else int(wave_value),)
            )
            memory_policy = choose_uniform_pipeline_policy(
                clade_counts,
                int(dataset.S),
                dtype,
                device=device,
                family_chunk_candidates=chunk_candidates,
                max_wave_candidates=wave_candidates,
                clade_budget=clade_budget,
                batch_packing=normalized_packing,
                leaf_counts=leaf_counts,
                nonleaf_counts=nonleaf_counts,
                schedule_depths=schedule_depths,
            )
            if chunk_value == "auto":
                chunk_value = memory_policy.family_chunk_size
            if wave_value == "auto":
                wave_value = memory_policy.max_wave_size

        family_chunk_n = 0 if chunk_value is None else int(chunk_value)
        max_wave_n = None if wave_value is None else int(wave_value)
        built_chunks = _built_chunks_from_rust(
            rust_preprocessed.build_chunked_layouts(
                family_chunk_size=family_chunk_n,
                clade_budget=clade_budget,
                batch_packing=normalized_packing,
                max_wave_size=max_wave_n,
                max_root_wave_size=max_root_wave_size,
                dtype=_dtype_name_for_rust(dtype),
                num_threads=0 if preprocess_cpu_cores is None else preprocess_cpu_cores,
                include_family_idx=False,
            ),
            device=device,
            dtype=dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        self.theta = torch.nn.Parameter(theta_init)
        self._origination_prior = prepared_origination_prior
        self.register_buffer("origination_probs", self._origination_prior.probs)
        self._state = _UniformChunkedState(
            dataset=dataset,
            species_helpers=species_helpers,
            ancestors_T=ancestors_T,
            unnorm_row_max=unnorm_row_max,
            built_chunks=built_chunks,
            device=device,
            dtype=dtype,
            origination_prior=self._origination_prior,
            origination_probs=self.origination_probs,
            fixed_iters_Pi=fixed_iters_Pi,
            fixed_iters_E=fixed_iters_E,
            max_iters_E=max_iters_E,
            tol_E=tol_E,
            neumann_terms=neumann_terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            warm_start_E=warm_start_E,
            profile=profile,
        )
        self.family_chunk_size = family_chunk_n
        self.max_wave_size = max_wave_n
        self.max_root_wave_size = max_root_wave_size
        self.clade_budget = clade_budget
        self.batch_packing = normalized_packing
        self.memory_policy = memory_policy
        self.gene_trees = dataset.gene_tree_paths
        self.family_names = dataset.family_names
        self.species_tree = str(species_tree)

    @property
    def n_families(self) -> int:
        """Number of gene families available to the chunked model."""
        return len(self._state.dataset.families)

    @property
    def family_count(self) -> int:
        """Alias for :attr:`n_families` used by workflow diagnostics."""
        return self.n_families

    @property
    def chunk_count(self) -> int:
        """Number of built uniform chunks in the resident chunk plan."""
        return len(self._state.built_chunks)

    @property
    def fixed_iters_Pi(self) -> int:
        """Fixed Pi iteration count used by each chunk evaluation."""
        return self._state.fixed_iters_Pi

    @property
    def fixed_iters_E(self) -> int | None:
        """Fixed E iteration count, or ``None`` when adaptive E is active."""
        return self._state.fixed_iters_E

    @property
    def chunk_metadata(self) -> tuple[UniformChunkMetadata, ...]:
        """Immutable per-chunk metadata for diagnostics and audit logs."""
        return tuple(
            UniformChunkMetadata(
                chunk_index=chunk_idx,
                family_indices=tuple(built.spec.indices),
                family_names=tuple(
                    self.family_names[family_idx]
                    for family_idx in built.spec.indices
                ),
                gene_tree_paths=tuple(
                    tuple(self.gene_trees[family_idx])
                    for family_idx in built.spec.indices
                ),
                family_count=len(built.spec.indices),
                clade_count=int(built.spec.clades),
                split_count=int(built.spec.splits),
                wave_count=int(built.waves),
                max_wave_size=int(built.max_wave),
                split_rows=int(built.split_rows),
                max_wave_split_rows=int(built.max_wave_split_rows),
            )
            for chunk_idx, built in enumerate(self._state.built_chunks)
        )

    @classmethod
    def from_trees(
        cls,
        species_tree: str | os.PathLike[str],
        gene_trees: Sequence[str | os.PathLike[str] | Sequence[str | os.PathLike[str]]],
        *,
        mode: str = "global",
        **kwargs: Any,
    ) -> "UniformChunkedReconModel":
        """Build the chunked global/uniform model from explicit tree paths.

        This mirrors :meth:`gpurec.GeneReconModel.from_trees` for constructor
        homogeneity.  The chunked implementation is intentionally global-only;
        use :class:`gpurec.GeneReconModel` for specieswise or genewise rates.
        Tree files use the supported simple Newick subset documented by
        :meth:`gpurec.GeneReconModel.from_trees`.
        """
        normalized = str(mode).strip().lower()
        if normalized not in {"global", "uniform"}:
            raise ValueError(
                "UniformChunkedReconModel.from_trees only supports mode='global' "
                f"or mode='uniform', got {mode!r}"
            )
        require_default_objective("UniformChunkedReconModel")
        kwargs = _normalize_uniform_solver_kwargs(kwargs)
        return cls(species_tree=species_tree, gene_trees=gene_trees, **kwargs)

    @classmethod
    def from_folder(
        cls,
        folder: str | os.PathLike[str],
        *,
        species_tree_name: str = "sp.nwk",
        gene_glob: str = "g_*.nwk",
        start: int = 0,
        max_families: int | None = None,
        **kwargs: Any,
    ) -> "UniformChunkedReconModel":
        """Build a model from a folder containing ``sp.nwk`` and gene trees.

        The folder contents use the supported simple Newick subset documented
        by :meth:`gpurec.GeneReconModel.from_trees`.
        """
        require_default_objective("UniformChunkedReconModel")
        start, max_families = normalize_family_selection(start, max_families)
        kwargs = _normalize_uniform_solver_kwargs(kwargs)
        root = Path(folder)
        species_tree = root / species_tree_name
        if not species_tree.exists():
            raise FileNotFoundError(f"missing species tree: {species_tree}")
        genes = _selected_gene_paths(
            root,
            gene_glob=gene_glob,
            start=start,
            max_families=max_families,
        )
        return cls(species_tree=species_tree, gene_trees=genes, **kwargs)

    @classmethod
    def from_alerax_families(
        cls,
        species_tree: str | os.PathLike[str],
        families_file: str | os.PathLike[str],
        *,
        mode: str = "global",
        start: int = 0,
        max_families: int | None = None,
        **kwargs: Any,
    ) -> "UniformChunkedReconModel":
        """Build the uniform model from an AleRax family/CCP list.

        Referenced tree files use the supported simple Newick subset documented
        by :meth:`gpurec.GeneReconModel.from_trees`.
        """
        normalized = str(mode).strip().lower()
        if normalized not in {"global", "uniform"}:
            raise ValueError(
                "UniformChunkedReconModel.from_alerax_families only supports "
                f"mode='global' or mode='uniform', got {mode!r}"
            )
        require_default_objective("UniformChunkedReconModel")
        start, max_families = normalize_family_selection(start, max_families)
        kwargs = _normalize_uniform_solver_kwargs(kwargs)
        theta_init_base_from_rates(
            kwargs.get("theta_init_rates", (0.05, 0.05, 0.05)),
            dtype=kwargs.get("dtype", torch.float32),
            device=torch.device("cpu"),
        )
        require_cuda_device(
            kwargs.get("device", "cuda"),
            owner="UniformChunkedReconModel",
        )
        family_names, tree_paths, leaf_maps = parse_alerax_family_file(
            families_file,
            start=start,
            max_families=max_families,
        )
        return cls(
            species_tree=species_tree,
            gene_trees=tree_paths,
            family_names=family_names,
            leaf_species_maps=leaf_maps,
            **kwargs,
        )

    def forward(self) -> torch.Tensor:
        if not torch.is_grad_enabled() or not self.theta.requires_grad:
            return self.nll()
        return _UniformChunkedFunction.apply(self.theta, self._state)

    @torch.no_grad()
    def nll(
        self,
        chunk_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = _evaluate_chunked_uniform_read_only(
            self._state,
            self.theta,
            chunk_indices=chunk_indices,
        )
        return result.loss

    @torch.no_grad()
    def nll_per_family(
        self,
        chunk_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return no-grad per-family NLL diagnostics for selected chunks.

        The output has one value per selected family in the selected chunk
        order.  This is a global/uniform shared-theta diagnostic, not an
        independent per-family gradient surface.
        """
        result = _evaluate_chunked_uniform_read_only(
            self._state,
            self.theta,
            collect_per_family=True,
            chunk_indices=chunk_indices,
        )
        if result.per_family_nll is None:
            raise RuntimeError("internal error: missing chunked per-family NLL")
        return result.per_family_nll

    @torch.no_grad()
    def loss_and_grad(
        self,
        *,
        chunk_indices: Sequence[int] | torch.Tensor | None = None,
        reduction: str = "sum",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Evaluate selected chunks and return ``(loss, grad, stats)``.

        This bypasses PyTorch autograd and exposes the custom chunked gradient
        directly, which is useful for stochastic optimizers that sample chunks.
        The returned stats dictionary includes ``e_adjoint_method``,
        ``e_adjoint_iterations``, ``e_adjoint_rel_res``, and
        ``e_adjoint_success`` for the retained E-adjoint solve.

        ``reduction`` controls the returned loss and gradient scale:

        - ``"sum"``: selected-family NLL sum.
        - ``"mean"``: selected-family mean NLL.
        - ``"full_sum_estimate"``: selected sum scaled by
          ``total_families / selected_families``.
        """
        if reduction not in ("sum", "mean", "full_sum_estimate"):
            raise ValueError(
                "reduction must be 'sum', 'mean', or 'full_sum_estimate', "
                f"got {reduction!r}"
            )
        loss, grad, stats = _evaluate_chunked_uniform(
            self._state,
            self.theta,
            need_grad=True,
            chunk_indices=chunk_indices,
        )
        if grad is None:
            raise RuntimeError("internal error: missing chunked uniform gradient")
        selected_families = int(stats["selected_families"])
        total_families = int(stats["total_families"])
        if reduction == "mean":
            scale = 1.0 / float(selected_families)
        elif reduction == "full_sum_estimate":
            scale = float(total_families) / float(selected_families)
        else:
            scale = 1.0
        if scale != 1.0:
            scale_t = torch.as_tensor(scale, device=loss.device, dtype=loss.dtype)
            loss = loss * scale_t
            grad = grad * scale_t.to(device=grad.device, dtype=grad.dtype)
        stats = dict(stats)
        stats["reduction"] = reduction
        stats["scale"] = float(scale)
        stats["reduced_loss"] = float(loss.detach().cpu())
        stats["reduced_grad_norm"] = float(torch.linalg.vector_norm(grad).detach().cpu())
        return (
            loss.to(device=self.theta.device, dtype=self.theta.dtype),
            grad.to(device=self.theta.device, dtype=self.theta.dtype),
            stats,
        )

    def clamp_theta_(
        self,
        min_rate: float = 1e-10,
        max_rate: float | None = None,
    ) -> None:
        clamp_theta_rates_(self.theta, min_rate=min_rate, max_rate=max_rate)


__all__ = [
    "UniformChunkMetadata",
    "UniformChunkedReconModel",
]
