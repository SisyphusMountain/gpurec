"""Static-state construction helpers for :class:`GeneReconModel`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gpurec.core.model import GeneDataset
from gpurec.core.origination import PreparedOriginationPrior

from ._model_types import BatchMetadata, _ResidentBatchSpec
from ._batch_specs import _parameter_mapping
from ._family_layout import build_family_wave_layout, family_wave_inputs
from .autograd import ReconStaticState


@dataclass(frozen=True)
class ResidentCommonState:
    species_helpers: dict[str, Any]
    ancestors_T: torch.Tensor | None
    unnorm_row_max: torch.Tensor


def _move_wave_layout_to_device(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    if torch.is_tensor(value):
        if value.dtype.is_floating_point:
            return value.to(device=device, dtype=dtype).contiguous()
        return value.to(device=device).contiguous()
    if isinstance(value, list):
        return [
            _move_wave_layout_to_device(item, device=device, dtype=dtype)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _move_wave_layout_to_device(item, device=device, dtype=dtype)
            for key, item in value.items()
        }
    return value


def _build_static_state(
    dataset: GeneDataset,
    *,
    origination_prior: PreparedOriginationPrior,
    common_state: ResidentCommonState,
    settings: Any,
    max_wave_size: int | None = 8192,
    max_root_wave_size: int | None = None,
    max_dts_partial_rows: int | None = None,
    clear_runtime_after_backward: bool = False,
) -> ReconStaticState:
    device = dataset.device
    dtype = dataset.dtype

    family_layout = build_family_wave_layout(
        family_wave_inputs(dataset, range(len(dataset.families))),
        device=device,
        dtype=dtype,
        max_wave_size=max_wave_size,
        max_root_wave_size=max_root_wave_size,
        max_dts_partial_rows=max_dts_partial_rows,
    )
    return ReconStaticState(
        device=device,
        dtype=dtype,
        wave_layout=family_layout.wave_layout,
        species_helpers=common_state.species_helpers,
        unnorm_row_max=common_state.unnorm_row_max,
        ancestors_T=common_state.ancestors_T,
        genewise=bool(dataset.genewise),
        specieswise=bool(dataset.specieswise),
        origination_prior=origination_prior,
        origination_probs=origination_prior.probs,
        clear_runtime_after_backward=clear_runtime_after_backward,
        **settings.static_kwargs(),
    )


def _build_batch_static_state(
    spec: _ResidentBatchSpec,
    *,
    dataset: GeneDataset,
    common_state: ResidentCommonState,
    origination_prior: PreparedOriginationPrior,
    settings: Any,
) -> ReconStaticState:
    device = dataset.device
    dtype = dataset.dtype
    if spec.wave_layout is None:
        if spec.layout_inputs is None:
            raise RuntimeError("resident batch spec is missing layout inputs")
        family_layout = build_family_wave_layout(
            spec.layout_inputs,
            device=device,
            dtype=dtype,
            waves=spec.waves,
            phases=spec.phases,
        )
        wave_layout = family_layout.wave_layout
    else:
        wave_layout = _move_wave_layout_to_device(
            spec.wave_layout,
            device=device,
            dtype=dtype,
        )

    return ReconStaticState(
        device=device,
        dtype=dtype,
        wave_layout=wave_layout,
        species_helpers=common_state.species_helpers,
        unnorm_row_max=common_state.unnorm_row_max,
        ancestors_T=common_state.ancestors_T,
        genewise=bool(dataset.genewise),
        specieswise=bool(dataset.specieswise),
        origination_prior=origination_prior,
        origination_probs=origination_prior.probs,
        clear_runtime_after_backward=True,
        **settings.static_kwargs(),
    )


def _metadata_for_full_static(
    dataset: GeneDataset,
    *,
    mode: str,
    static: ReconStaticState,
) -> BatchMetadata:
    layout_inputs = family_wave_inputs(dataset, range(len(dataset.families)))
    wave_metas = static.wave_layout["wave_metas"]
    return BatchMetadata(
        batch_index=0,
        family_indices=list(range(len(dataset.families))),
        family_names=list(dataset.family_names),
        gene_tree_paths=[list(paths) for paths in dataset.gene_tree_paths],
        family_count=len(dataset.families),
        clade_count=int(static.wave_layout["C"]),
        split_count=layout_inputs.split_count,
        wave_count=len(wave_metas),
        max_wave_size=max((int(meta["W"]) for meta in wave_metas), default=0),
        root_clade_rows=layout_inputs.root_clade_rows,
        parameter_mapping=_parameter_mapping(
            mode=mode,
            dataset=dataset,
            family_indices=range(len(dataset.families)),
        ),
    )
