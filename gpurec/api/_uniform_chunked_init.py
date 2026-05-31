"""Private constructor setup for :mod:`gpurec.api.uniform_chunked`."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from gpurec.core.batch_planning import (
    normalize_batch_packing,
    normalize_clade_budget,
)
from gpurec.core.memory_policy import (
    UniformPipelinePolicy,
)
from gpurec.core.model import normalize_family_inputs
from gpurec.core.origination import (
    OriginationPrior,
    PreparedOriginationPrior,
    prepare_origination_prior,
)

from ._uniform_chunked_inputs import _validate_uniform_dtype
from ._uniform_chunked_layout import (
    _UniformBuiltChunk,
    _UniformChunkedState,
    _built_chunks_from_rust,
    _dtype_name_for_rust,
)
from ._validation import (
    auto_nonnegative_int as _auto_nonnegative_int,
    auto_positive_int as _auto_positive_int,
    bool_value,
    nonnegative_float,
    nonnegative_int_sequence as _normalize_nonnegative_int_sequence,
    optional_positive_int,
    positive_even_int,
    positive_int,
    positive_int_sequence as _normalize_positive_int_sequence,
    require_default_objective,
    theta_init_base_from_rates,
)


class _CudaDeviceResolver(Protocol):
    def __call__(self, device: Any, *, owner: str) -> torch.device:
        """Resolve and validate the requested CUDA device."""


class _UniformPipelinePolicyChooser(Protocol):
    def __call__(
        self,
        clade_counts: Sequence[int],
        species_count: int,
        dtype: torch.dtype,
        **kwargs: Any,
    ) -> UniformPipelinePolicy:
        """Choose resident chunk and wave dimensions for uniform evaluation."""


@dataclass(frozen=True)
class UniformChunkedInitDependencies:
    """Constructor dependencies supplied by the public compatibility module."""

    gene_dataset_cls: Any
    require_cuda_device: _CudaDeviceResolver
    choose_uniform_pipeline_policy: _UniformPipelinePolicyChooser


@dataclass(frozen=True)
class UniformChunkedInitState:
    """Prepared constructor state before registration on the public module."""

    theta_init: torch.Tensor
    dataset: Any
    species_helpers: dict[str, Any]
    ancestors_T: torch.Tensor | None
    unnorm_row_max: torch.Tensor
    built_chunks: list[_UniformBuiltChunk]
    device: torch.device
    dtype: torch.dtype
    origination_prior: PreparedOriginationPrior
    fixed_iters_Pi: int
    fixed_iters_E: int | None
    max_iters_E: int
    tol_E: float
    neumann_terms: int
    use_pruning: bool
    pruning_threshold: float
    warm_start_E: bool
    profile: bool
    family_chunk_size: int
    max_wave_size: int | None
    max_root_wave_size: int | None
    clade_budget: int | None
    batch_packing: str
    memory_policy: UniformPipelinePolicy | None
    species_tree: str


def prepare_uniform_chunked_init(
    *,
    dependencies: UniformChunkedInitDependencies,
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
) -> UniformChunkedInitState:
    """Validate inputs, preprocess trees, and build the static chunked state."""
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

    device = dependencies.require_cuda_device(
        device,
        owner="UniformChunkedReconModel",
    )
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
    dataset = dependencies.gene_dataset_cls._from_preprocessed_raw(
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
        memory_policy = dependencies.choose_uniform_pipeline_policy(
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

    return UniformChunkedInitState(
        theta_init=theta_init,
        dataset=dataset,
        species_helpers=species_helpers,
        ancestors_T=ancestors_T,
        unnorm_row_max=unnorm_row_max,
        built_chunks=built_chunks,
        device=device,
        dtype=dtype,
        origination_prior=prepared_origination_prior,
        fixed_iters_Pi=fixed_iters_Pi,
        fixed_iters_E=fixed_iters_E,
        max_iters_E=max_iters_E,
        tol_E=tol_E,
        neumann_terms=neumann_terms,
        use_pruning=use_pruning,
        pruning_threshold=pruning_threshold,
        warm_start_E=warm_start_E,
        profile=profile,
        family_chunk_size=family_chunk_n,
        max_wave_size=max_wave_n,
        max_root_wave_size=max_root_wave_size,
        clade_budget=clade_budget,
        batch_packing=normalized_packing,
        memory_policy=memory_policy,
        species_tree=str(species_tree),
    )


def apply_uniform_chunked_init(
    model: torch.nn.Module,
    init_state: UniformChunkedInitState,
) -> None:
    """Register prepared constructor state on ``UniformChunkedReconModel``."""
    model.theta = torch.nn.Parameter(init_state.theta_init)
    model._origination_prior = init_state.origination_prior
    model.register_buffer("origination_probs", init_state.origination_prior.probs)
    model._state = _UniformChunkedState(
        dataset=init_state.dataset,
        species_helpers=init_state.species_helpers,
        ancestors_T=init_state.ancestors_T,
        unnorm_row_max=init_state.unnorm_row_max,
        built_chunks=init_state.built_chunks,
        device=init_state.device,
        dtype=init_state.dtype,
        origination_prior=model._origination_prior,
        origination_probs=model.origination_probs,
        fixed_iters_Pi=init_state.fixed_iters_Pi,
        fixed_iters_E=init_state.fixed_iters_E,
        max_iters_E=init_state.max_iters_E,
        tol_E=init_state.tol_E,
        neumann_terms=init_state.neumann_terms,
        use_pruning=init_state.use_pruning,
        pruning_threshold=init_state.pruning_threshold,
        warm_start_E=init_state.warm_start_E,
        profile=init_state.profile,
    )
    model.family_chunk_size = init_state.family_chunk_size
    model.max_wave_size = init_state.max_wave_size
    model.max_root_wave_size = init_state.max_root_wave_size
    model.clade_budget = init_state.clade_budget
    model.batch_packing = init_state.batch_packing
    model.memory_policy = init_state.memory_policy
    model.gene_trees = init_state.dataset.gene_tree_paths
    model.family_names = init_state.dataset.family_names
    model.species_tree = init_state.species_tree
