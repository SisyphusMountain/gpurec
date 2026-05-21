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

from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Optional, Sequence

import torch

from gpurec.core.batching import family_schedule_summary
from gpurec.core.batch_planning import (
    normalize_batch_packing,
    normalize_clade_budget,
    normalize_family_chunk_size,
    plan_family_batches,
)
from gpurec.core.model import (
    GeneDataset,
    normalize_family_selection,
    normalize_family_tree_paths,
    parse_alerax_family_file,
)
from gpurec.core.likelihood import (
    compute_nll,
    prepare_origination_probs,
)
from gpurec.optimization.implicit_grad import implicit_grad_loglik_vjp_wave

from .autograd import (
    ReconStaticState,
    _GeneReconFunction,
    _record_backward_solver_stats,
)
from ._family_layout import (
    FamilyWaveInputs,
    build_family_wave_layout,
    family_wave_inputs,
    origination_probs_for_family_indices,
    schedule_family_waves,
)
from ._uniform_evaluator import (
    _record_forward_solver_stats,
    evaluate_resident_no_grad,
    solve_resident_e_pi,
)
from ._validation import (
    bool_value,
    integer_value,
    nonnegative_float,
    optional_positive_int,
    positive_even_int,
    positive_float,
    positive_int,
    require_cuda_device,
    require_default_objective,
    theta_init_base_from_rates,
    validate_theta_shape,
)

_MODE_MAP: dict[str, tuple[bool, bool]] = {
    "global": (False, False),
    "specieswise": (False, True),
    "genewise": (True, False),
}


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _MODE_MAP:
        raise ValueError(f"Unknown mode {mode!r}. Valid: {sorted(_MODE_MAP)}")
    return normalized


def _mode_to_flags(mode: str) -> tuple[bool, bool]:
    return _MODE_MAP[_normalize_mode(mode)]


def _validate_gene_dtype(dtype: Any) -> torch.dtype:
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"dtype must be torch.float32 or torch.float64, got {dtype!r}"
        )
    return dtype


def _default_theta_init(dataset: GeneDataset, mode: str) -> torch.Tensor:
    base = math.log2(1e-10)
    genewise, specieswise = _mode_to_flags(mode)
    if genewise:
        shape = (len(dataset.families), 3)
    elif specieswise:
        shape = (int(dataset.S), 3)
    else:
        shape = (3,)
    return torch.full(shape, base, dtype=dataset.dtype, device=dataset.device)


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

    e: torch.Tensor
    pi: torch.Tensor
    log_p_s: torch.Tensor
    log_p_d: torch.Tensor
    log_p_l: torch.Tensor
    max_transfer: torch.Tensor
    origination_probs: torch.Tensor | None


@dataclass(frozen=True)
class _ResidentBatchSpec:
    index: int
    family_indices: list[int]
    layout_inputs: FamilyWaveInputs
    waves: list[list[int]]
    phases: list[int]
    metadata: BatchMetadata


def _normalize_family_chunk_size(value: int | str | None) -> int:
    return int(normalize_family_chunk_size(value))


def _normalize_clade_budget(value: int | None) -> int | None:
    return normalize_clade_budget(value)


def _normalize_batch_packing(value: str | None) -> str:
    return normalize_batch_packing(value)


def _normalize_prefetch_batches(value: int | str | None, *, lazy: bool) -> int | str:
    if value is None:
        return "all" if lazy else 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "all"):
            return "all"
        if text in ("0", "none", "null", "false"):
            return 0
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError("prefetch_batches must be non-negative or 'all'") from exc
    if isinstance(value, bool):
        raise ValueError("prefetch_batches must be an integer or 'all'")
    if isinstance(value, int):
        count = int(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("prefetch_batches must be an integer or 'all'")
        count = int(value)
    else:
        raise ValueError("prefetch_batches must be an integer or 'all'")
    if count < 0:
        raise ValueError("prefetch_batches must be non-negative or 'all'")
    return count


def _normalize_gene_solver_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate public solver kwargs before CUDA setup or tree parsing."""
    normalized = dict(kwargs)
    if normalized.get("fixed_iters_E") is not None:
        normalized["fixed_iters_E"] = positive_int(
            "fixed_iters_E",
            normalized["fixed_iters_E"],
        )
    if "fixed_iters_Pi" in normalized:
        normalized["fixed_iters_Pi"] = positive_even_int(
            "fixed_iters_Pi",
            normalized["fixed_iters_Pi"],
        )
    if "neumann_terms" in normalized:
        normalized["neumann_terms"] = positive_int(
            "neumann_terms",
            normalized["neumann_terms"],
        )
    if "convergence_check_interval" in normalized:
        convergence_check_interval = positive_int(
            "convergence_check_interval",
            normalized["convergence_check_interval"],
        )
        normalized["convergence_check_interval"] = convergence_check_interval
    if "max_iters_E" in normalized:
        normalized["max_iters_E"] = positive_int(
            "max_iters_E",
            normalized["max_iters_E"],
        )
    for name in (
        "tol_E",
        "e_logsumexp_tol",
        "pi_max_diff_tol",
        "gradient_change_tol",
        "gradient_change_rtol",
        "pruning_threshold",
    ):
        if name in normalized:
            normalized[name] = nonnegative_float(name, normalized[name])
    if "adaptive_iters" in normalized:
        normalized["adaptive_iters"] = bool_value(
            "adaptive_iters",
            normalized["adaptive_iters"],
        )
    if "use_pruning" in normalized:
        normalized["use_pruning"] = bool_value(
            "use_pruning",
            normalized["use_pruning"],
        )
    if "family_chunk_size" in normalized:
        normalized["family_chunk_size"] = _normalize_family_chunk_size(
            normalized["family_chunk_size"]
        )
    if "clade_budget" in normalized:
        normalized["clade_budget"] = _normalize_clade_budget(
            normalized["clade_budget"]
        )
    for name in ("max_wave_size", "max_root_wave_size", "max_dts_partial_rows"):
        if name in normalized:
            normalized[name] = optional_positive_int(name, normalized[name])
    if "batch_packing" in normalized:
        normalized["batch_packing"] = _normalize_batch_packing(
            normalized["batch_packing"]
        )
    if "lazy_preprocess" in normalized:
        normalized["lazy_preprocess"] = bool_value(
            "lazy_preprocess",
            normalized["lazy_preprocess"],
        )
    lazy_preprocess = normalized.get("lazy_preprocess", False)
    if "prefetch_batches" in normalized:
        normalized["prefetch_batches"] = _normalize_prefetch_batches(
            normalized["prefetch_batches"],
            lazy=lazy_preprocess,
        )
    adaptive_iters = normalized.get("adaptive_iters", False)
    convergence_check_interval = int(
        normalized.get("convergence_check_interval", 4)
    )
    if adaptive_iters and convergence_check_interval % 2 != 0:
        raise ValueError("adaptive_iters requires an even convergence_check_interval")
    return normalized


def _parameter_mapping(
    *,
    mode: str,
    dataset: GeneDataset,
    family_indices: Sequence[int],
) -> dict[str, Any]:
    if mode == "global":
        return {
            "mode": "global",
            "theta_shape": [3],
            "shared": True,
            "batch_theta_rows": [],
        }
    if mode == "specieswise":
        return {
            "mode": "specieswise",
            "theta_shape": [int(dataset.S), 3],
            "shared": True,
            "batch_theta_rows": list(range(int(dataset.S))),
        }
    return {
        "mode": "genewise",
        "theta_shape": [len(dataset.families), 3],
        "shared": False,
        "batch_theta_rows": [int(i) for i in family_indices],
    }


def _family_index_chunks(
    *,
    total: int,
    clade_counts: Sequence[int],
    family_chunk_size: int,
    clade_budget: int | None,
    batch_packing: str = "sequential",
    leaf_counts: Sequence[int] | None = None,
    nonleaf_counts: Sequence[int] | None = None,
    schedule_depths: Sequence[int] | None = None,
    max_wave_size: int | None = None,
) -> list[list[int]]:
    return [
        plan.indices
        for plan in plan_family_batches(
            total=total,
            clade_counts=clade_counts,
            family_chunk_size=family_chunk_size,
            clade_budget=clade_budget,
            batch_packing=batch_packing,
            leaf_counts=leaf_counts,
            nonleaf_counts=nonleaf_counts,
            schedule_depths=schedule_depths,
            max_wave_size=max_wave_size,
        )
    ]


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
            {
                key: _immutable_public_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_public_value(item) for item in value)
    return value


def _build_batch_specs(
    dataset: GeneDataset,
    *,
    mode: str,
    family_chunk_size: int,
    clade_budget: int | None,
    batch_packing: str,
    max_wave_size: int | None,
    max_root_wave_size: int | None,
    max_dts_partial_rows: int | None,
) -> list[_ResidentBatchSpec]:
    clade_counts = [int(fam["C"]) for fam in dataset.families]
    leaf_counts: list[int] | None = None
    nonleaf_counts: list[int] | None = None
    schedule_depths: list[int] | None = None
    if _normalize_batch_packing(batch_packing) == "depth_first_fit":
        summaries = [
            family_schedule_summary(fam["ccp_helpers"])
            for fam in dataset.families
        ]
        leaf_counts = [int(summary["leaf_count"]) for summary in summaries]
        nonleaf_counts = [int(summary["nonleaf_count"]) for summary in summaries]
        schedule_depths = [int(summary["max_level"]) for summary in summaries]
    chunks = _family_index_chunks(
        total=len(dataset.families),
        clade_counts=clade_counts,
        family_chunk_size=family_chunk_size,
        clade_budget=clade_budget,
        batch_packing=batch_packing,
        leaf_counts=leaf_counts,
        nonleaf_counts=nonleaf_counts,
        schedule_depths=schedule_depths,
        max_wave_size=max_wave_size,
    )
    specs: list[_ResidentBatchSpec] = []
    for batch_index, family_indices in enumerate(chunks):
        layout_inputs = family_wave_inputs(dataset, family_indices)
        cross_waves, cross_phases = schedule_family_waves(
            layout_inputs,
            max_wave_size=max_wave_size,
            max_root_wave_size=max_root_wave_size,
            max_dts_partial_rows=max_dts_partial_rows,
        )

        metadata = BatchMetadata(
            batch_index=batch_index,
            family_indices=[int(i) for i in family_indices],
            family_names=[dataset.family_names[i] for i in family_indices],
            gene_tree_paths=[list(dataset.gene_tree_paths[i]) for i in family_indices],
            family_count=len(family_indices),
            clade_count=layout_inputs.clade_count,
            split_count=layout_inputs.split_count,
            wave_count=len(cross_waves),
            max_wave_size=max((len(w) for w in cross_waves), default=0),
            root_clade_rows=layout_inputs.root_clade_rows,
            parameter_mapping=_parameter_mapping(
                mode=mode,
                dataset=dataset,
                family_indices=family_indices,
            ),
        )
        specs.append(
            _ResidentBatchSpec(
                index=batch_index,
                family_indices=[int(i) for i in family_indices],
                layout_inputs=layout_inputs,
                waves=cross_waves,
                phases=cross_phases,
                metadata=metadata,
            )
        )
    return specs


def _build_static_state(
    dataset: GeneDataset,
    *,
    fixed_iters_E: Optional[int],
    max_iters_E: int,
    tol_E: float,
    fixed_iters_Pi: int,
    neumann_terms: int,
    adaptive_iters: bool,
    convergence_check_interval: int,
    e_logsumexp_tol: float,
    pi_max_diff_tol: float,
    gradient_change_tol: float,
    gradient_change_rtol: float,
    use_pruning: bool,
    pruning_threshold: float,
    origination_probs: torch.Tensor | None,
    max_wave_size: Optional[int] = 8192,
    max_root_wave_size: Optional[int] = None,
    max_dts_partial_rows: Optional[int] = None,
) -> ReconStaticState:
    """Absorb the wave-layout boilerplate that lives in
    ``experiments/validate_three_modes.py:100-149``.

    Builds a single cross-family wave layout for the entire dataset and
    moves species helpers (and ``ancestors_T`` for uniform mode) onto the
    target device. The result is cached on the model and reused across
    every ``forward()`` call.
    """
    device = dataset.device
    dtype = dataset.dtype

    # 1. Cross-family wave layout
    family_layout = build_family_wave_layout(
        family_wave_inputs(dataset, range(len(dataset.families))),
        device=device,
        dtype=dtype,
        max_wave_size=max_wave_size,
        max_root_wave_size=max_root_wave_size,
        max_dts_partial_rows=max_dts_partial_rows,
    )
    wave_layout = family_layout.wave_layout

    # 2. Species helpers on device.
    species_helpers, ancestors_T = dataset._species_helpers_for_mode(
        device=device, dtype=dtype,
    )

    # 3. Other static tensors
    unnorm_row_max = dataset.unnorm_row_max.to(device=device, dtype=dtype)

    return ReconStaticState(
        device=device,
        dtype=dtype,
        wave_layout=wave_layout,
        species_helpers=species_helpers,
        unnorm_row_max=unnorm_row_max,
        ancestors_T=ancestors_T,
        genewise=bool(dataset.genewise),
        specieswise=bool(dataset.specieswise),
        origination_probs=origination_probs,
        fixed_iters_E=fixed_iters_E,
        max_iters_E=max_iters_E,
        tol_E=tol_E,
        fixed_iters_Pi=fixed_iters_Pi,
        neumann_terms=neumann_terms,
        adaptive_iters=adaptive_iters,
        convergence_check_interval=convergence_check_interval,
        e_logsumexp_tol=e_logsumexp_tol,
        pi_max_diff_tol=pi_max_diff_tol,
        gradient_change_tol=gradient_change_tol,
        gradient_change_rtol=gradient_change_rtol,
        use_pruning=use_pruning,
        pruning_threshold=pruning_threshold,
    )


def _build_batch_static_state(
    spec: _ResidentBatchSpec,
    *,
    dataset: GeneDataset,
    species_helpers: dict[str, Any],
    ancestors_T: torch.Tensor | None,
    unnorm_row_max: torch.Tensor,
    fixed_iters_E: Optional[int],
    max_iters_E: int,
    tol_E: float,
    fixed_iters_Pi: int,
    neumann_terms: int,
    adaptive_iters: bool,
    convergence_check_interval: int,
    e_logsumexp_tol: float,
    pi_max_diff_tol: float,
    gradient_change_tol: float,
    gradient_change_rtol: float,
    use_pruning: bool,
    pruning_threshold: float,
    origination_probs: torch.Tensor | None,
) -> ReconStaticState:
    device = dataset.device
    dtype = dataset.dtype
    family_layout = build_family_wave_layout(
        spec.layout_inputs,
        device=device,
        dtype=dtype,
        waves=spec.waves,
        phases=spec.phases,
    )
    wave_layout = family_layout.wave_layout
    return ReconStaticState(
        device=device,
        dtype=dtype,
        wave_layout=wave_layout,
        species_helpers=species_helpers,
        unnorm_row_max=unnorm_row_max,
        ancestors_T=ancestors_T,
        genewise=bool(dataset.genewise),
        specieswise=bool(dataset.specieswise),
        origination_probs=origination_probs,
        fixed_iters_E=fixed_iters_E,
        max_iters_E=max_iters_E,
        tol_E=tol_E,
        fixed_iters_Pi=fixed_iters_Pi,
        neumann_terms=neumann_terms,
        adaptive_iters=adaptive_iters,
        convergence_check_interval=convergence_check_interval,
        e_logsumexp_tol=e_logsumexp_tol,
        pi_max_diff_tol=pi_max_diff_tol,
        gradient_change_tol=gradient_change_tol,
        gradient_change_rtol=gradient_change_rtol,
        use_pruning=use_pruning,
        pruning_threshold=pruning_threshold,
        clear_runtime_after_backward=True,
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


def _evaluate_static_state(
    static: ReconStaticState,
    theta: torch.Tensor,
    *,
    need_grad: bool,
    per_family: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    require_default_objective("GeneReconModel")
    if not need_grad:
        return evaluate_resident_no_grad(static, theta, per_family=per_family), None
    if need_grad and per_family and not static.genewise:
        raise ValueError("per-family gradients are only independent in genewise mode")

    solve = solve_resident_e_pi(
        static,
        theta,
        return_original=False,
        return_root_rows=False,
    )
    E_out = solve.e_out
    pi_out = solve.pi_out
    log_pS = solve.log_p_s
    log_pD = solve.log_p_d
    log_pL = solve.log_p_l
    max_transfer_vec = solve.max_transfer
    theta_eval = solve.theta
    _record_forward_solver_stats(static, E_out, pi_out)
    if need_grad:
        loss_vec = compute_nll(
            pi_out["Pi_wave_ordered"],
            E_out["E"],
            static.wave_layout["root_clade_ids"],
            static.origination_probs,
            origination_probs_prepared=True,
        )
        grad_theta, _stats = implicit_grad_loglik_vjp_wave(
            static.wave_layout,
            static.species_helpers,
            Pi_star_wave=pi_out["Pi_wave_ordered"],
            Pibar_star_wave=pi_out["Pibar_wave_ordered"],
            E_star=E_out["E"],
            Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"],
            E_s2=E_out["E_s2"],
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            max_transfer_mat=max_transfer_vec,
            root_clade_ids_perm=static.wave_layout["root_clade_ids"],
            theta=theta_eval,
            unnorm_row_max=static.unnorm_row_max,
            specieswise=static.specieswise,
            device=static.device,
            dtype=static.dtype,
            neumann_terms=static.neumann_terms,
            use_pruning=static.use_pruning,
            pruning_threshold=static.pruning_threshold,
            ancestors_T=static.ancestors_T,
            family_idx=(
                static.wave_layout["family_idx"] if static.genewise else None
            ),
            uniform_pibar_row_max=pi_out.get("uniform_pibar_row_max"),
            origination_probs=static.origination_probs,
            origination_probs_prepared=True,
            genewise=static.genewise,
            gradient_convergence_tol=(
                static.gradient_change_tol if static.adaptive_iters else -1.0
            ),
            gradient_convergence_rtol=static.gradient_change_rtol,
            gradient_convergence_check_interval=static.convergence_check_interval,
        )
        _record_backward_solver_stats(static, _stats)
        static.warm_E = None
        return (loss_vec.detach() if per_family else loss_vec.sum().detach()), grad_theta.detach()
    raise RuntimeError("internal error: unreachable no-grad resident evaluation path")

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
        lazy_preprocess: bool = False,
        prefetch_batches: int | str | None = None,
        origination_probs: torch.Tensor | Sequence[float] | None = None,
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
        batch_packing = _normalize_batch_packing(batch_packing)
        lazy_preprocess = bool_value("lazy_preprocess", lazy_preprocess)
        prefetch_batches = _normalize_prefetch_batches(
            prefetch_batches,
            lazy=lazy_preprocess,
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
            )

        require_cuda_device(dataset.device, owner="GeneReconModel")

        self._mode = mode
        self._dataset = dataset
        prepared_origination_probs = prepare_origination_probs(
            origination_probs,
            S=int(dataset.S),
            device=dataset.device,
            dtype=dataset.dtype,
            family_count=len(dataset.families) if origination_probs is not None else None,
        )
        self.register_buffer("origination_probs", prepared_origination_probs)
        self.family_chunk_size = family_chunk_size
        self.clade_budget = clade_budget
        self.batch_packing = batch_packing
        self.lazy_preprocess = lazy_preprocess
        self.prefetch_batches = prefetch_batches
        self._batched_resident = bool(
            self.lazy_preprocess
            or family_chunk_requested
            or self.clade_budget is not None
        )

        if theta_init is None:
            theta_init = _default_theta_init(dataset, mode)
        self.theta = torch.nn.Parameter(theta_init.clone())

        self._fixed_iters_E = fixed_iters_E
        self._max_iters_E = max_iters_E
        self._tol_E = tol_E
        self._fixed_iters_Pi = fixed_iters_Pi
        self._neumann_terms = neumann_terms
        self._adaptive_iters = adaptive_iters
        self._convergence_check_interval = convergence_check_interval
        self._e_logsumexp_tol = float(e_logsumexp_tol)
        self._pi_max_diff_tol = float(pi_max_diff_tol)
        self._gradient_change_tol = float(gradient_change_tol)
        self._gradient_change_rtol = float(gradient_change_rtol)
        self._use_pruning = use_pruning
        self._pruning_threshold = pruning_threshold
        self.max_wave_size = max_wave_size
        self.max_root_wave_size = max_root_wave_size
        self.max_dts_partial_rows = max_dts_partial_rows

        self._static: ReconStaticState | None = None
        self._batch_specs: list[_ResidentBatchSpec] = []
        self._batch_statics: list[ReconStaticState | None] = []
        self._batch_futures: dict[int, Future[ReconStaticState]] = {}
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_closed = False
        self._batch_lock = Lock()
        self._current_batch_index = 0

        if self._batched_resident:
            species_helpers, ancestors_T = dataset._species_helpers_for_mode(
                device=dataset.device,
                dtype=dataset.dtype,
            )
            self._resident_species_helpers = species_helpers
            self._resident_ancestors_T = ancestors_T
            self._resident_unnorm_row_max = dataset.unnorm_row_max.to(
                device=dataset.device,
                dtype=dataset.dtype,
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
            )
            self._batch_statics = [None for _ in self._batch_specs]
            self.batch_metadata = [spec.metadata for spec in self._batch_specs]
            if not self._batch_specs:
                raise ValueError("GeneReconModel requires at least one family")
            self._ensure_batch_static(0)
            if self.lazy_preprocess:
                self._schedule_prefetch()
            else:
                for batch_idx in range(1, len(self._batch_specs)):
                    self._ensure_batch_static(batch_idx)
        else:
            self._static = _build_static_state(
                dataset,
                fixed_iters_E=fixed_iters_E,
                max_iters_E=max_iters_E,
                tol_E=tol_E,
                fixed_iters_Pi=fixed_iters_Pi,
                neumann_terms=neumann_terms,
                adaptive_iters=self._adaptive_iters,
                convergence_check_interval=self._convergence_check_interval,
                e_logsumexp_tol=self._e_logsumexp_tol,
                pi_max_diff_tol=self._pi_max_diff_tol,
                gradient_change_tol=self._gradient_change_tol,
                gradient_change_rtol=self._gradient_change_rtol,
                use_pruning=use_pruning,
                pruning_threshold=pruning_threshold,
                max_wave_size=max_wave_size,
                max_root_wave_size=max_root_wave_size,
                max_dts_partial_rows=max_dts_partial_rows,
                origination_probs=self.origination_probs,
            )
            self.batch_metadata = [
                _metadata_for_full_static(dataset, mode=mode, static=self._static)
            ]

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
        preprocess_cache_dir: str | os.PathLike | None = None,
        refresh_preprocess_cache: bool = False,
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
        preprocess_cache_dir : str | os.PathLike | None
            Optional directory for CPU preprocessing cache files. Reusing the
            same cache avoids reparsing/rebuilding unchanged gene trees.
        refresh_preprocess_cache : bool
            Ignore existing preprocessing cache entries and overwrite them.
        """
        mode = _normalize_mode(mode)
        genewise, specieswise = _mode_to_flags(mode)
        require_default_objective("GeneReconModel")
        refresh_preprocess_cache = bool_value(
            "refresh_preprocess_cache",
            refresh_preprocess_cache,
        )
        dtype = _validate_gene_dtype(dtype)
        solver_kwargs = _normalize_gene_solver_kwargs(solver_kwargs)
        theta_base = theta_init_base_from_rates(
            theta_init_rates,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        gene_tree_paths = normalize_family_tree_paths(gene_trees)
        device = require_cuda_device(device, owner="GeneReconModel")
        if theta_base is not None:
            theta_base = theta_base.to(device=device)
        ds = GeneDataset(
            species_tree_path=species_tree,
            gene_tree_paths=gene_tree_paths,
            genewise=genewise,
            specieswise=specieswise,
            dtype=dtype,
            device=device,
            preprocess_cache_dir=preprocess_cache_dir,
            refresh_preprocess_cache=refresh_preprocess_cache,
        )
        theta_init = None
        if theta_base is not None:
            if mode == "specieswise":
                theta_init = theta_base.unsqueeze(0).expand(int(ds.S), -1).clone()
            elif mode == "genewise":
                theta_init = (
                    theta_base.unsqueeze(0).expand(len(gene_tree_paths), -1).clone()
                )
            else:
                theta_init = theta_base
        return cls(
            dataset=ds,
            mode=mode,
            theta_init=theta_init,
            **solver_kwargs,
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
        preprocess_cache_dir: str | os.PathLike | None = None,
        refresh_preprocess_cache: bool = False,
        **solver_kwargs,
    ) -> "GeneReconModel":
        """Build from an AleRax ``[FAMILIES]`` file with CCP/tree samples.

        Referenced species and gene tree files use the same supported simple
        Newick subset as :meth:`from_trees`.
        """
        mode = _normalize_mode(mode)
        genewise, specieswise = _mode_to_flags(mode)
        require_default_objective("GeneReconModel")
        refresh_preprocess_cache = bool_value(
            "refresh_preprocess_cache",
            refresh_preprocess_cache,
        )
        dtype = _validate_gene_dtype(dtype)
        start, max_families = normalize_family_selection(start, max_families)
        solver_kwargs = _normalize_gene_solver_kwargs(solver_kwargs)
        theta_base = theta_init_base_from_rates(
            theta_init_rates,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        device = require_cuda_device(device, owner="GeneReconModel")
        if theta_base is not None:
            theta_base = theta_base.to(device=device)
        family_names, tree_paths, leaf_maps = parse_alerax_family_file(
            families_file,
            start=start,
            max_families=max_families,
        )
        ds = GeneDataset(
            species_tree_path=species_tree,
            gene_tree_paths=tree_paths,
            genewise=genewise,
            specieswise=specieswise,
            dtype=dtype,
            device=device,
            preprocess_cache_dir=preprocess_cache_dir,
            refresh_preprocess_cache=refresh_preprocess_cache,
            family_names=family_names,
            leaf_species_maps=leaf_maps,
        )
        theta_init = None
        if theta_base is not None:
            if mode == "specieswise":
                theta_init = theta_base.unsqueeze(0).expand(int(ds.S), -1).clone()
            elif mode == "genewise":
                theta_init = theta_base.unsqueeze(0).expand(len(family_names), -1).clone()
            else:
                theta_init = theta_base
        return cls(
            dataset=ds,
            mode=mode,
            theta_init=theta_init,
            **solver_kwargs,
        )

    # ──────────────────────────────────────────────────────────────────
    # Resident batch management
    # ──────────────────────────────────────────────────────────────────
    def _build_batch_static(self, batch_idx: int) -> ReconStaticState:
        return _build_batch_static_state(
            self._batch_specs[batch_idx],
            dataset=self._dataset,
            species_helpers=self._resident_species_helpers,
            ancestors_T=self._resident_ancestors_T,
            unnorm_row_max=self._resident_unnorm_row_max,
            fixed_iters_E=self._fixed_iters_E,
            max_iters_E=self._max_iters_E,
            tol_E=self._tol_E,
            fixed_iters_Pi=self._fixed_iters_Pi,
            neumann_terms=self._neumann_terms,
            adaptive_iters=self._adaptive_iters,
            convergence_check_interval=self._convergence_check_interval,
            e_logsumexp_tol=self._e_logsumexp_tol,
            pi_max_diff_tol=self._pi_max_diff_tol,
            gradient_change_tol=self._gradient_change_tol,
            gradient_change_rtol=self._gradient_change_rtol,
            use_pruning=self._use_pruning,
            pruning_threshold=self._pruning_threshold,
            origination_probs=origination_probs_for_family_indices(
                self.origination_probs,
                self._batch_specs[batch_idx].family_indices,
            ),
        )

    def _ensure_batch_static(self, batch_idx: int) -> ReconStaticState:
        if not self._batched_resident:
            if self._static is None:
                raise RuntimeError("resident static state has not been built")
            return self._static
        if batch_idx < 0 or batch_idx >= len(self._batch_specs):
            raise IndexError(
                f"batch index {batch_idx} out of range for {len(self._batch_specs)} batches"
            )
        with self._batch_lock:
            static = self._batch_statics[batch_idx]
            future = self._batch_futures.pop(batch_idx, None)
        if static is not None:
            return static
        if future is not None:
            static = future.result()
        else:
            static = self._build_batch_static(batch_idx)
        with self._batch_lock:
            existing = self._batch_statics[batch_idx]
            if existing is None:
                self._batch_statics[batch_idx] = static
                return static
            return existing

    def _submit_prefetch(self, batch_idx: int) -> None:
        if self._prefetch_closed:
            return
        if batch_idx < 0 or batch_idx >= len(self._batch_specs):
            return
        with self._batch_lock:
            if (
                self._batch_statics[batch_idx] is not None
                or batch_idx in self._batch_futures
            ):
                return
            if self._prefetch_executor is None:
                self._prefetch_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="gpurec-preprocess",
                )
            self._batch_futures[batch_idx] = self._prefetch_executor.submit(
                self._build_batch_static,
                batch_idx,
            )

    def _schedule_prefetch(self) -> None:
        if (
            self._prefetch_closed
            or not self._batched_resident
            or self.prefetch_batches == 0
        ):
            return
        start = self._current_batch_index + 1
        if self.prefetch_batches == "all":
            stop = len(self._batch_specs)
        else:
            stop = min(len(self._batch_specs), start + int(self.prefetch_batches))
        for batch_idx in range(start, stop):
            self._submit_prefetch(batch_idx)

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
        if not self._batched_resident:
            loss, grad = _evaluate_static_state(
                self._active_static(),
                theta,
                need_grad=need_grad,
            )
            return loss, grad

        total_loss = torch.zeros((), device=self._dataset.device, dtype=self._dataset.dtype)
        grad_total = torch.zeros_like(theta.detach()) if need_grad else None
        for batch_idx in range(len(self._batch_specs)):
            static = self._ensure_batch_static(batch_idx)
            theta_batch = self._theta_for_batch_index(batch_idx, theta)
            loss_i, grad_i = _evaluate_static_state(
                static,
                theta_batch,
                need_grad=need_grad,
            )
            total_loss = total_loss + loss_i.to(device=total_loss.device, dtype=total_loss.dtype)
            if need_grad:
                if grad_i is None or grad_total is None:
                    raise RuntimeError("internal error: missing batch gradient")
                grad_i = grad_i.to(device=grad_total.device, dtype=grad_total.dtype)
                if self._mode == "genewise":
                    idx = torch.as_tensor(
                        self._batch_specs[batch_idx].family_indices,
                        dtype=torch.long,
                        device=grad_total.device,
                    )
                    grad_total.index_add_(0, idx, grad_i)
                else:
                    grad_total.add_(grad_i)
        return total_loss.detach(), None if grad_total is None else grad_total.detach()

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
            return [static for static in self._batch_statics if static is not None]
        return [] if self._static is None else [self._static]

    def materialize_batches(self) -> list[BatchMetadata]:
        """Build all resident batch static states and return metadata copies.

        In resident-batch mode this forces every batch static state to be built
        before returning.  The returned list is a copy of ``batch_metadata``, so
        callers can inspect batch ownership without mutating model bookkeeping.
        """
        if self._batched_resident:
            for batch_idx in range(len(self._batch_specs)):
                self._ensure_batch_static(batch_idx)
        elif self._static is None:
            raise RuntimeError("resident static state has not been built")
        return list(self.batch_metadata)

    def configure_solver_iterations(
        self,
        *,
        fixed_iters_Pi: int | None = None,
        neumann_terms: int | None = None,
        pi_max_diff_tol: float | None = None,
        gradient_change_tol: float | None = None,
    ) -> None:
        """Update solver iteration controls on the model and built batches.

        The method updates model defaults and resident batch static states that
        are already built.  It does not cancel or rewrite pending background
        prefetch work; configure before scheduling lazy prefetch, or materialize
        resident batches and configure again when all batches should share the
        new controls.
        """
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

        for static in self.cached_static_states:
            if fixed_iters_Pi is not None:
                static.fixed_iters_Pi = fixed_iters_Pi
            if neumann_terms is not None:
                static.neumann_terms = neumann_terms
            if pi_max_diff_tol is not None:
                static.pi_max_diff_tol = pi_max_diff_tol
            if gradient_change_tol is not None:
                static.gradient_change_tol = gradient_change_tol

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
        static = self._active_static()
        static.warm_E = None

    def close(self) -> None:
        """Stop background batch preprocessing and drop pending futures."""
        executor = getattr(self, "_prefetch_executor", None)
        batch_futures = getattr(self, "_batch_futures", None)
        batch_lock = getattr(self, "_batch_lock", None)
        if batch_lock is None:
            self._prefetch_closed = True
            self._prefetch_executor = None
            if batch_futures is not None:
                batch_futures.clear()
        else:
            with batch_lock:
                self._prefetch_closed = True
                executor = getattr(self, "_prefetch_executor", None)
                self._prefetch_executor = None
                batch_futures = getattr(self, "_batch_futures", None)
                if batch_futures is not None:
                    batch_futures.clear()
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
            values.copy_(loss.to(device=values.device, dtype=values.dtype).reshape(-1))
            if need_grad:
                if grad is None or grad_total is None:
                    raise RuntimeError("internal error: missing genewise gradient")
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
                idx = torch.as_tensor(
                    metadata.family_indices,
                    dtype=torch.long,
                    device=values.device,
                )
                values.index_copy_(
                    0,
                    idx,
                    batch_values.to(device=values.device, dtype=values.dtype).reshape(-1),
                )
                if need_grad:
                    if batch_grad is None or grad_total is None:
                        raise RuntimeError("internal error: missing genewise batch gradient")
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
        solve = solve_resident_e_pi(
            static,
            theta,
            return_original=original_order,
            return_root_rows=False,
        )
        pi = (
            solve.pi_out["Pi"]
            if original_order
            else solve.pi_out["Pi_wave_ordered"]
        )
        return ReconciliationState(
            e=solve.e_out["E"],
            pi=pi,
            log_p_s=solve.log_p_s,
            log_p_d=solve.log_p_d,
            log_p_l=solve.log_p_l,
            max_transfer=solve.max_transfer,
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
