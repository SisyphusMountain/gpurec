"""Private construction helpers for :class:`gpurec.api.model.GeneReconModel`."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch

from gpurec.core.model import (
    GeneDataset,
    normalize_family_selection,
    normalize_family_tree_paths,
    parse_alerax_family_file,
)

from ._batch_specs import (
    _normalize_gene_solver_kwargs,
    _should_use_compact_retained_preprocess,
)
from ._theta_init import _expand_theta_base, _mode_to_flags, _normalize_mode
from ._theta_init import _validate_gene_dtype
from ._validation import (
    optional_positive_int,
    require_cuda_device,
    require_default_objective,
    theta_init_base_from_rates,
)
from ._warmup import (
    _finish_cuda_context_warmup,
    _start_cuda_context_warmup,
)


@dataclass(frozen=True)
class _ValidatedFactoryMode:
    mode: str
    genewise: bool
    specieswise: bool
    dtype: torch.dtype


@dataclass(frozen=True)
class _RetainedDatasetRequest:
    species_tree_path: str | os.PathLike[str]
    gene_tree_paths: Sequence[Sequence[str | os.PathLike[str]]]
    mode: str
    genewise: bool
    specieswise: bool
    dtype: torch.dtype
    device: torch.device
    preprocess_cpu_cores: int | None
    solver_kwargs: dict[str, Any]
    family_count: int
    family_names: Sequence[str] | None = None
    leaf_species_maps: Sequence[dict[str, str]] | None = None


@dataclass(frozen=True)
class GeneReconModelBuildInputs:
    """Validated inputs needed to instantiate ``GeneReconModel``."""

    dataset: GeneDataset
    mode: str
    theta_init: torch.Tensor | None
    solver_kwargs: dict[str, Any]


def _validate_mode_dtype_objective(
    *,
    mode: str,
    dtype: Any,
) -> _ValidatedFactoryMode:
    mode = _normalize_mode(mode)
    genewise, specieswise = _mode_to_flags(mode)
    require_default_objective("GeneReconModel")
    dtype = _validate_gene_dtype(dtype)
    return _ValidatedFactoryMode(
        mode=mode,
        genewise=genewise,
        specieswise=specieswise,
        dtype=dtype,
    )


def _expand_factory_theta(
    theta_base: torch.Tensor | None,
    *,
    mode: str,
    dataset: GeneDataset,
    family_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    if theta_base is not None:
        theta_base = theta_base.to(device=device)
    return _expand_theta_base(
        theta_base,
        mode=mode,
        species_count=int(dataset.S),
        family_count=family_count,
        device=device,
    )


def _build_retained_dataset(request: _RetainedDatasetRequest) -> GeneDataset:
    return GeneDataset.from_retained_preprocess(
        species_tree_path=request.species_tree_path,
        gene_tree_paths=request.gene_tree_paths,
        genewise=request.genewise,
        specieswise=request.specieswise,
        dtype=request.dtype,
        device=request.device,
        preprocess_cpu_cores=request.preprocess_cpu_cores,
        family_names=request.family_names,
        leaf_species_maps=request.leaf_species_maps,
        compact_families=_should_use_compact_retained_preprocess(
            request.mode,
            request.solver_kwargs,
        ),
    )


def _prepare_retained_dataset_with_cuda_warmup(
    device: torch.device,
    request_factory: Callable[[], _RetainedDatasetRequest],
) -> tuple[GeneDataset, _RetainedDatasetRequest]:
    cuda_warmup = _start_cuda_context_warmup(device)
    try:
        request = request_factory()
        return _build_retained_dataset(request), request
    finally:
        _finish_cuda_context_warmup(cuda_warmup)


def _trees_request(
    *,
    species_tree: str | os.PathLike[str],
    gene_tree_paths: Sequence[Sequence[str | os.PathLike[str]]],
    validated: _ValidatedFactoryMode,
    device: torch.device,
    preprocess_cpu_cores: int | None,
    solver_kwargs: dict[str, Any],
) -> _RetainedDatasetRequest:
    return _RetainedDatasetRequest(
        species_tree_path=species_tree,
        gene_tree_paths=gene_tree_paths,
        mode=validated.mode,
        genewise=validated.genewise,
        specieswise=validated.specieswise,
        dtype=validated.dtype,
        device=device,
        preprocess_cpu_cores=preprocess_cpu_cores,
        solver_kwargs=solver_kwargs,
        family_count=len(gene_tree_paths),
    )


def _alerax_request(
    *,
    species_tree: str,
    families_file: str | os.PathLike,
    start: int,
    max_families: int | None,
    validated: _ValidatedFactoryMode,
    device: torch.device,
    preprocess_cpu_cores: int | None,
    solver_kwargs: dict[str, Any],
) -> _RetainedDatasetRequest:
    family_names, tree_paths, leaf_maps = parse_alerax_family_file(
        families_file,
        start=start,
        max_families=max_families,
    )
    return _RetainedDatasetRequest(
        species_tree_path=species_tree,
        gene_tree_paths=tree_paths,
        mode=validated.mode,
        genewise=validated.genewise,
        specieswise=validated.specieswise,
        dtype=validated.dtype,
        device=device,
        preprocess_cpu_cores=preprocess_cpu_cores,
        solver_kwargs=solver_kwargs,
        family_count=len(family_names),
        family_names=family_names,
        leaf_species_maps=leaf_maps,
    )


def build_from_trees_inputs(
    *,
    species_tree: str | os.PathLike[str],
    gene_trees: Sequence[
        str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
    ],
    mode: str,
    device: Any,
    dtype: torch.dtype,
    theta_init_rates: Optional[tuple[float, float, float]],
    preprocess_cpu_cores: int | None,
    solver_kwargs: dict[str, Any],
) -> GeneReconModelBuildInputs:
    validated = _validate_mode_dtype_objective(mode=mode, dtype=dtype)
    solver_kwargs = _normalize_gene_solver_kwargs(solver_kwargs)
    preprocess_cpu_cores = optional_positive_int(
        "preprocess_cpu_cores",
        preprocess_cpu_cores,
    )
    theta_base = theta_init_base_from_rates(
        theta_init_rates,
        dtype=validated.dtype,
        device=torch.device("cpu"),
    )
    gene_tree_paths = normalize_family_tree_paths(gene_trees)
    device = require_cuda_device(device, owner="GeneReconModel")
    dataset, request = _prepare_retained_dataset_with_cuda_warmup(
        device,
        lambda: _trees_request(
            species_tree=species_tree,
            gene_tree_paths=gene_tree_paths,
            validated=validated,
            device=device,
            preprocess_cpu_cores=preprocess_cpu_cores,
            solver_kwargs=solver_kwargs,
        ),
    )
    theta_init = _expand_factory_theta(
        theta_base,
        mode=validated.mode,
        dataset=dataset,
        family_count=request.family_count,
        device=device,
    )
    return GeneReconModelBuildInputs(
        dataset=dataset,
        mode=validated.mode,
        theta_init=theta_init,
        solver_kwargs=solver_kwargs,
    )


def build_from_alerax_families_inputs(
    *,
    species_tree: str,
    families_file: str | os.PathLike,
    mode: str,
    start: int,
    max_families: int | None,
    device: Any,
    dtype: torch.dtype,
    theta_init_rates: Optional[tuple[float, float, float]],
    preprocess_cpu_cores: int | None,
    solver_kwargs: dict[str, Any],
) -> GeneReconModelBuildInputs:
    validated = _validate_mode_dtype_objective(mode=mode, dtype=dtype)
    start, max_families = normalize_family_selection(start, max_families)
    solver_kwargs = _normalize_gene_solver_kwargs(solver_kwargs)
    preprocess_cpu_cores = optional_positive_int(
        "preprocess_cpu_cores",
        preprocess_cpu_cores,
    )
    theta_base = theta_init_base_from_rates(
        theta_init_rates,
        dtype=validated.dtype,
        device=torch.device("cpu"),
    )
    device = require_cuda_device(device, owner="GeneReconModel")
    dataset, request = _prepare_retained_dataset_with_cuda_warmup(
        device,
        lambda: _alerax_request(
            species_tree=species_tree,
            families_file=families_file,
            start=start,
            max_families=max_families,
            validated=validated,
            device=device,
            preprocess_cpu_cores=preprocess_cpu_cores,
            solver_kwargs=solver_kwargs,
        ),
    )
    theta_init = _expand_factory_theta(
        theta_base,
        mode=validated.mode,
        dataset=dataset,
        family_count=request.family_count,
        device=device,
    )
    return GeneReconModelBuildInputs(
        dataset=dataset,
        mode=validated.mode,
        theta_init=theta_init,
        solver_kwargs=solver_kwargs,
    )
