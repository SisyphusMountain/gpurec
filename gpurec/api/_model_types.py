"""Public and internal dataclasses used by :mod:`gpurec.api.model`."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import torch


_MODE_MAP: dict[str, tuple[bool, bool]] = {
    "global": (False, False),
    "specieswise": (False, True),
    "genewise": (True, False),
}


@dataclass(frozen=True)
class BatchMetadata:
    """Public metadata for one resident batch."""

    batch_index: int
    family_indices: tuple[int, ...]
    family_names: tuple[str, ...]
    gene_tree_paths: tuple[tuple[str, ...], ...]
    family_count: int
    clade_count: int
    split_count: int
    wave_count: int
    max_wave_size: int
    root_clade_rows: tuple[int, ...]
    parameter_mapping: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family_indices",
            tuple(int(index) for index in self.family_indices),
        )
        object.__setattr__(
            self,
            "family_names",
            tuple(str(name) for name in self.family_names),
        )
        object.__setattr__(
            self,
            "gene_tree_paths",
            tuple(
                tuple(str(path) for path in paths)
                for paths in self.gene_tree_paths
            ),
        )
        object.__setattr__(
            self,
            "root_clade_rows",
            tuple(int(row) for row in self.root_clade_rows),
        )
        object.__setattr__(
            self,
            "parameter_mapping",
            _immutable_public_value(self.parameter_mapping),
        )


@dataclass(frozen=True)
class ActiveFamilyBatch:
    """Location of one family in the currently selected resident batch."""

    family_index: int
    batch_index: int
    local_family_index: int
    clade_offset: int
    metadata: BatchMetadata


@dataclass(frozen=True)
class FamilyInput:
    """Read-only family metadata needed by workflow/export utilities."""

    index: int
    name: str
    gene_tree_paths: tuple[str, ...]
    leaf_species_map: Mapping[str, str]
    clade_count: int
    split_count: int
    root_clade_id: int
    ccp_helpers: Mapping[str, Any]
    leaf_row_index: Any
    leaf_col_index: Any
    clade_leaf_labels: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gene_tree_paths",
            tuple(str(path) for path in self.gene_tree_paths),
        )
        object.__setattr__(
            self,
            "leaf_species_map",
            _immutable_public_value(
                {
                    str(gene): str(species)
                    for gene, species in self.leaf_species_map.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "ccp_helpers",
            _immutable_public_value(self.ccp_helpers),
        )
        object.__setattr__(
            self,
            "clade_leaf_labels",
            tuple(str(label) for label in self.clade_leaf_labels),
        )


@dataclass(frozen=True)
class ReconciliationState:
    """Solved reconciliation tensors for the currently selected model batch."""

    e: Any
    pi: Any
    pibar: Any | None
    ebar: Any | None
    log_p_s: Any
    log_p_d: Any
    log_p_l: Any
    max_transfer: Any
    origination_probs: Any | None
    origination_prior: Any | None = None


@dataclass(frozen=True)
class _ResidentBatchSpec:
    index: int
    family_indices: list[int]
    layout_inputs: Any | None
    waves: list[list[int]]
    phases: list[int]
    metadata: BatchMetadata
    wave_layout: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _FamilyScheduleStats:
    clade_counts: list[int]
    split_counts: list[int]
    leaf_counts: list[int] | None = None
    nonleaf_counts: list[int] | None = None
    schedule_depths: list[int] | None = None


def _public_family_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _public_family_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_family_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_public_family_value(item) for item in value)
    return value


def _immutable_public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _immutable_public_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_public_value(item) for item in value)
    return value
