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

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from threading import Lock
from typing import Any, Optional, Sequence

import torch

from gpurec.core.backward import Pi_wave_backward
from gpurec.core.batching import (
    build_wave_layout,
    collate_gene_families,
    family_schedule_summary,
    schedule_global_phased_waves,
)
from gpurec.core.model import GeneDataset, parse_alerax_family_file
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import (
    E_fixed_point,
    compute_log_likelihood,
    compute_log_likelihood_root_rows,
    prepare_origination_probs,
)
from gpurec.optimization.implicit_grad import _e_adjoint_and_theta_vjp

from .autograd import (
    ReconStaticState,
    _GeneReconFunction,
    _extract_parameters,
)

_MODE_MAP: dict[str, tuple[bool, bool]] = {
    "global": (False, False),
    "specieswise": (False, True),
    "genewise": (True, False),
}


def _mode_to_flags(mode: str) -> tuple[bool, bool]:
    if mode not in _MODE_MAP:
        raise ValueError(f"Unknown mode {mode!r}. Valid: {sorted(_MODE_MAP)}")
    return _MODE_MAP[mode]


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
    family_indices: list[int]
    family_names: list[str]
    gene_tree_paths: list[list[str]]
    family_count: int
    clade_count: int
    split_count: int
    wave_count: int
    max_wave_size: int
    root_clade_rows: list[int]
    parameter_mapping: dict[str, Any]


@dataclass(frozen=True)
class _ResidentBatchSpec:
    index: int
    family_indices: list[int]
    items: tuple[dict[str, Any], ...]
    waves: list[list[int]]
    phases: list[int]
    family_clade_counts: list[int]
    family_clade_offsets: list[int]
    metadata: BatchMetadata


def _normalize_family_chunk_size(value: int | str | None) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "0", "all", "none", "null"):
            return 0
        value = int(text)
    size = int(value)
    if size < 0:
        raise ValueError("family_chunk_size must be non-negative")
    return size


def _normalize_clade_budget(value: int | None) -> int | None:
    if value is None:
        return None
    budget = int(value)
    if budget <= 0:
        raise ValueError("clade_budget must be positive when provided")
    return budget


def _normalize_batch_packing(value: str | None) -> str:
    if value is None:
        return "sequential"
    text = str(value).strip().lower().replace("-", "_")
    if text in ("", "sequential", "contiguous", "input_order"):
        return "sequential"
    if text in ("clade_first_fit", "first_fit_decreasing", "ffd", "clade_ffd"):
        return "clade_first_fit"
    if text in (
        "depth_first_fit",
        "depth_ffd",
        "critical_path_first_fit",
        "critical_first_fit",
        "wave_first_fit",
    ):
        return "depth_first_fit"
    raise ValueError(
        "batch_packing must be 'sequential', 'clade_first_fit', or "
        "'depth_first_fit', "
        f"got {value!r}"
    )


def _normalize_prefetch_batches(value: int | str | None, *, lazy: bool) -> int | str:
    if value is None:
        return "all" if lazy else 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "all"):
            return "all"
        if text in ("0", "none", "null", "false"):
            return 0
        value = int(text)
    count = int(value)
    if count < 0:
        raise ValueError("prefetch_batches must be non-negative or 'all'")
    return count


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
    batch_packing = _normalize_batch_packing(batch_packing)
    if batch_packing == "clade_first_fit":
        if clade_budget is None:
            raise ValueError("batch_packing='clade_first_fit' requires clade_budget")
        chunks: list[list[int]] = []
        chunk_clades: list[int] = []
        order = sorted(range(total), key=lambda idx: int(clade_counts[idx]), reverse=True)
        for idx in order:
            n_clades = int(clade_counts[idx])
            best_j: int | None = None
            best_remaining: int | None = None
            for j, current_clades in enumerate(chunk_clades):
                if family_chunk_size > 0 and len(chunks[j]) >= family_chunk_size:
                    continue
                remaining = clade_budget - current_clades - n_clades
                if remaining < 0:
                    continue
                if best_remaining is None or remaining < best_remaining:
                    best_j = j
                    best_remaining = remaining
            if best_j is None:
                chunks.append([idx])
                chunk_clades.append(n_clades)
            else:
                chunks[best_j].append(idx)
                chunk_clades[best_j] += n_clades
        return chunks
    if batch_packing == "depth_first_fit":
        if clade_budget is None:
            raise ValueError("batch_packing='depth_first_fit' requires clade_budget")
        if leaf_counts is None or nonleaf_counts is None or schedule_depths is None:
            raise ValueError(
                "batch_packing='depth_first_fit' requires leaf_counts, "
                "nonleaf_counts, and schedule_depths"
            )
        if (
            len(leaf_counts) != total
            or len(nonleaf_counts) != total
            or len(schedule_depths) != total
        ):
            raise ValueError("depth_first_fit scheduling stats must match total families")

        wave_cap = (
            sum(int(c) for c in clade_counts)
            if max_wave_size is None
            else int(max_wave_size)
        )
        if wave_cap <= 0:
            raise ValueError("max_wave_size must be positive")

        def lower_bound(leaves: int, nonleaves: int, depth: int) -> int:
            leaf_waves = math.ceil(int(leaves) / wave_cap)
            work_waves = math.ceil(int(nonleaves) / wave_cap)
            return leaf_waves + max(int(depth), work_waves)

        chunks: list[list[int]] = []
        chunk_clades: list[int] = []
        chunk_leaves: list[int] = []
        chunk_nonleaves: list[int] = []
        chunk_depths: list[int] = []
        order = sorted(
            range(total),
            key=lambda idx: (int(schedule_depths[idx]), int(clade_counts[idx])),
            reverse=True,
        )
        for idx in order:
            n_clades = int(clade_counts[idx])
            n_leaves = int(leaf_counts[idx])
            n_nonleaves = int(nonleaf_counts[idx])
            depth = int(schedule_depths[idx])
            best_j: int | None = None
            best_key: tuple[int, int, int] | None = None
            for j, current_clades in enumerate(chunk_clades):
                if family_chunk_size > 0 and len(chunks[j]) >= family_chunk_size:
                    continue
                new_clades = current_clades + n_clades
                if new_clades > clade_budget:
                    continue
                before = lower_bound(
                    chunk_leaves[j],
                    chunk_nonleaves[j],
                    chunk_depths[j],
                )
                after = lower_bound(
                    chunk_leaves[j] + n_leaves,
                    chunk_nonleaves[j] + n_nonleaves,
                    max(chunk_depths[j], depth),
                )
                remaining = clade_budget - new_clades
                key = (after - before, after, remaining)
                if best_key is None or key < best_key:
                    best_j = j
                    best_key = key
            if best_j is None:
                chunks.append([idx])
                chunk_clades.append(n_clades)
                chunk_leaves.append(n_leaves)
                chunk_nonleaves.append(n_nonleaves)
                chunk_depths.append(depth)
            else:
                chunks[best_j].append(idx)
                chunk_clades[best_j] += n_clades
                chunk_leaves[best_j] += n_leaves
                chunk_nonleaves[best_j] += n_nonleaves
                chunk_depths[best_j] = max(chunk_depths[best_j], depth)
        return chunks

    chunks: list[list[int]] = []
    current: list[int] = []
    current_clades = 0

    def flush() -> None:
        nonlocal current, current_clades
        if current:
            chunks.append(list(current))
            current = []
            current_clades = 0

    for idx in range(total):
        n_clades = int(clade_counts[idx])
        family_cap_hit = family_chunk_size > 0 and len(current) >= family_chunk_size
        clade_cap_hit = (
            clade_budget is not None
            and current
            and current_clades + n_clades > clade_budget
        )
        if family_cap_hit or clade_cap_hit:
            flush()
        current.append(idx)
        current_clades += n_clades
    flush()
    return chunks


def _origination_probs_for_family_indices(
    origination_probs: torch.Tensor | None,
    family_indices: Sequence[int],
) -> torch.Tensor | None:
    if origination_probs is None or origination_probs.ndim == 1:
        return origination_probs
    idx = torch.as_tensor(
        family_indices,
        dtype=torch.long,
        device=origination_probs.device,
    )
    return origination_probs.index_select(0, idx)


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
        items: list[dict[str, Any]] = []
        family_clade_counts: list[int] = []
        family_clade_offsets: list[int] = []
        root_clade_rows: list[int] = []
        clade_offset = 0
        split_count = 0

        for family_idx in family_indices:
            fam = dataset.families[family_idx]
            items.append(
                {
                    "ccp": fam["ccp_helpers"],
                    "leaf_row_index": fam["leaf_row_index"],
                    "leaf_col_index": fam["leaf_col_index"],
                    "root_clade_id": int(fam["root_clade_id"]),
                }
            )
            C_i = int(fam["C"])
            family_clade_counts.append(C_i)
            family_clade_offsets.append(clade_offset)
            root_clade_rows.append(int(fam["root_clade_id"]) + clade_offset)
            clade_offset += C_i
            split_count += int(fam["N_splits"])

        cross_waves, cross_phases = schedule_global_phased_waves(
            items,
            family_clade_offsets,
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
            clade_count=sum(family_clade_counts),
            split_count=split_count,
            wave_count=len(cross_waves),
            max_wave_size=max((len(w) for w in cross_waves), default=0),
            root_clade_rows=root_clade_rows,
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
                items=tuple(items),
                waves=cross_waves,
                phases=cross_phases,
                family_clade_counts=family_clade_counts,
                family_clade_offsets=family_clade_offsets,
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
    items = [
        {
            "ccp": fam["ccp_helpers"],
            "leaf_row_index": fam["leaf_row_index"],
            "leaf_col_index": fam["leaf_col_index"],
            "root_clade_id": int(fam["root_clade_id"]),
        }
        for fam in dataset.families
    ]
    batched = collate_gene_families(items, dtype=dtype, device=device)

    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves, cross_phases = schedule_global_phased_waves(
        items,
        offsets,
        max_wave_size=max_wave_size,
        max_root_wave_size=max_root_wave_size,
        max_dts_partial_rows=max_dts_partial_rows,
    )

    family_clade_counts = [m["C"] for m in batched["family_meta"]]
    family_clade_offsets = [m["clade_offset"] for m in batched["family_meta"]]

    wave_layout = build_wave_layout(
        waves=cross_waves,
        phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device,
        dtype=dtype,
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
    )

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
    use_pruning: bool,
    pruning_threshold: float,
    origination_probs: torch.Tensor | None,
) -> ReconStaticState:
    device = dataset.device
    dtype = dataset.dtype
    batched = collate_gene_families(list(spec.items), dtype=dtype, device=device)
    wave_layout = build_wave_layout(
        waves=spec.waves,
        phases=spec.phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device,
        dtype=dtype,
        family_clade_counts=spec.family_clade_counts,
        family_clade_offsets=spec.family_clade_offsets,
    )
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
    root_rows: list[int] = []
    offset = 0
    split_count = 0
    for fam in dataset.families:
        root_rows.append(int(fam["root_clade_id"]) + offset)
        offset += int(fam["C"])
        split_count += int(fam["N_splits"])
    wave_metas = static.wave_layout["wave_metas"]
    return BatchMetadata(
        batch_index=0,
        family_indices=list(range(len(dataset.families))),
        family_names=list(dataset.family_names),
        gene_tree_paths=[list(paths) for paths in dataset.gene_tree_paths],
        family_count=len(dataset.families),
        clade_count=int(static.wave_layout["C"]),
        split_count=split_count,
        wave_count=len(wave_metas),
        max_wave_size=max((int(meta["W"]) for meta in wave_metas), default=0),
        root_clade_rows=root_rows,
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
    if os.environ.get("GPUREC_ALERAX_COMPAT", "0") != "0":
        raise NotImplementedError(
            "GPUREC_ALERAX_COMPAT changes the forward objective to "
            "AleRax's fixed four-pass evaluator. The current custom "
            "backward differentiates GPUREC's converged fixed-point "
            "objective, so gradient-based optimization is disabled in "
            "this mode."
        )
    if need_grad and per_family:
        raise ValueError("per-family streaming evaluation is no-grad only")

    theta_eval = theta.detach().to(device=static.device, dtype=static.dtype)
    log_pS, log_pD, log_pL, max_transfer_vec = _extract_parameters(theta_eval, static)
    e_max_iters = (
        static.fixed_iters_E
        if static.fixed_iters_E is not None
        else static.max_iters_E
    )
    e_tolerance = -1.0 if static.fixed_iters_E is not None else static.tol_E
    E_out = E_fixed_point(
        species_helpers=static.species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        max_transfer_mat=max_transfer_vec,
        max_iters=e_max_iters,
        tolerance=e_tolerance,
        warm_start_E=None,
        dtype=static.dtype,
        device=static.device,
        ancestors_T=static.ancestors_T,
    )

    pi_out = Pi_wave_forward(
        wave_layout=static.wave_layout,
        species_helpers=static.species_helpers,
        E=E_out["E"],
        Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"],
        E_s2=E_out["E_s2"],
        log_pS=log_pS,
        log_pD=log_pD,
        max_transfer_mat=max_transfer_vec,
        device=static.device,
        dtype=static.dtype,
        fixed_iters=static.fixed_iters_Pi,
        return_original=False,
        return_root_rows=not need_grad,
        family_idx=(
            static.wave_layout.get("family_idx") if static.genewise else None
        ),
    )
    if need_grad:
        loss_vec = compute_log_likelihood(
            pi_out["Pi_wave_ordered"],
            E_out["E"],
            static.wave_layout["root_clade_ids"],
            static.origination_probs,
        )
        pi_bwd = Pi_wave_backward(
            wave_layout=static.wave_layout,
            Pi_star_wave=pi_out["Pi_wave_ordered"],
            Pibar_star_wave=pi_out["Pibar_wave_ordered"],
            E=E_out["E"],
            Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"],
            E_s2=E_out["E_s2"],
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            max_transfer_mat=max_transfer_vec,
            species_helpers=static.species_helpers,
            root_clade_ids_perm=static.wave_layout["root_clade_ids"],
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
        )
        grad_theta, _stats = _e_adjoint_and_theta_vjp(
            pi_bwd,
            E_out["E"],
            E_out["E_bar"],
            E_out["E_s1"],
            E_out["E_s2"],
            log_pS,
            log_pD,
            log_pL,
            max_transfer_vec,
            static.species_helpers,
            static.wave_layout["root_clade_ids"],
            theta_eval,
            static.unnorm_row_max,
            static.specieswise,
            static.device,
            static.dtype,
            genewise=static.genewise,
            ancestors_T=static.ancestors_T,
            origination_probs=static.origination_probs,
        )
        static.warm_E = None
        return loss_vec.sum().detach(), grad_theta.detach()

    loss_vec = compute_log_likelihood_root_rows(
        pi_out["Pi_root_rows"],
        E_out["E"],
        static.origination_probs,
    )
    static.warm_E = None
    return (loss_vec.detach() if per_family else loss_vec.sum().detach()), None


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
        # Validate mode early
        _mode_to_flags(mode)
        if fixed_iters_E is not None:
            fixed_iters_E = int(fixed_iters_E)
            if fixed_iters_E < 1:
                raise ValueError("fixed_iters_E must be >= 1 when provided")
        fixed_iters_Pi = int(fixed_iters_Pi)
        if fixed_iters_Pi < 1 or fixed_iters_Pi % 2 != 0:
            raise ValueError("fixed_iters_Pi must be a positive even integer")

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
        self.family_chunk_size = _normalize_family_chunk_size(family_chunk_size)
        self.clade_budget = _normalize_clade_budget(clade_budget)
        self.batch_packing = _normalize_batch_packing(batch_packing)
        self.lazy_preprocess = bool(lazy_preprocess)
        self.prefetch_batches = _normalize_prefetch_batches(
            prefetch_batches,
            lazy=self.lazy_preprocess,
        )
        self._batched_resident = bool(
            self.lazy_preprocess
            or family_chunk_size is not None
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
        species_tree: str,
        gene_trees: list[str],
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
            Path to the species tree (Newick).
        gene_trees : list[str]
            Paths to gene trees (Newick).
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
        genewise, specieswise = _mode_to_flags(mode)
        if isinstance(device, str):
            device = torch.device(device)
        ds = GeneDataset(
            species_tree_path=species_tree,
            gene_tree_paths=gene_trees,
            genewise=genewise,
            specieswise=specieswise,
            dtype=dtype,
            device=device,
            preprocess_cache_dir=preprocess_cache_dir,
            refresh_preprocess_cache=refresh_preprocess_cache,
        )
        theta_init = None
        if theta_init_rates is not None:
            D, L, T = theta_init_rates
            base = torch.log2(
                torch.tensor([D, L, T], dtype=dtype, device=device)
            )
            if mode == "specieswise":
                theta_init = base.unsqueeze(0).expand(int(ds.S), -1).clone()
            elif mode == "genewise":
                theta_init = (
                    base.unsqueeze(0).expand(len(gene_trees), -1).clone()
                )
            else:
                theta_init = base
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
        """Build from an AleRax ``[FAMILIES]`` file with CCP/tree samples."""
        genewise, specieswise = _mode_to_flags(mode)
        if isinstance(device, str):
            device = torch.device(device)
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
        if theta_init_rates is not None:
            D, L, T = theta_init_rates
            base = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))
            if mode == "specieswise":
                theta_init = base.unsqueeze(0).expand(int(ds.S), -1).clone()
            elif mode == "genewise":
                theta_init = base.unsqueeze(0).expand(len(family_names), -1).clone()
            else:
                theta_init = base
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
            use_pruning=self._use_pruning,
            pruning_threshold=self._pruning_threshold,
            origination_probs=_origination_probs_for_family_indices(
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
        if not self._batched_resident or self.prefetch_batches == 0:
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
        return self.batch_metadata[self._current_batch_index]

    @property
    def current_batch_index(self) -> int:
        return self._current_batch_index

    def next(self) -> BatchMetadata:
        """Advance to the next resident batch and return its metadata."""
        if self._current_batch_index + 1 >= len(self.batch_metadata):
            raise StopIteration("already at the final resident batch")
        self.clear()
        self._current_batch_index += 1
        self._ensure_batch_static(self._current_batch_index)
        self._schedule_prefetch()
        return self.current_batch_metadata

    def clear(self) -> None:
        """Release active runtime caches held by the model."""
        static = self._active_static()
        static.warm_E = None

    def close(self) -> None:
        """Stop background batch preprocessing and drop pending futures."""
        with self._batch_lock:
            executor = self._prefetch_executor
            self._prefetch_executor = None
            self._batch_futures.clear()
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
        static = self._active_static()
        theta = self._active_theta()
        if not torch.is_grad_enabled() or not theta.requires_grad:
            out, _grad = _evaluate_static_state(
                static,
                theta,
                need_grad=False,
                per_family=(reduce == "per_family"),
            )
            return out.to(device=theta.device, dtype=theta.dtype)
        return _GeneReconFunction.apply(theta, static, reduce)

    @torch.no_grad()
    def _forward_per_family_inference(self) -> torch.Tensor:
        """Per-family diagnostic NLL for shared-theta modes."""
        theta = self._active_theta()
        out, _grad = _evaluate_static_state(
            self._active_static(),
            theta,
            need_grad=False,
            per_family=True,
        )
        return out.to(device=theta.device, dtype=theta.dtype)

    def full_loss(self) -> torch.Tensor:
        """Stream every resident batch and return the exact full NLL."""
        if not self._batched_resident:
            return self.forward(reduce="sum")
        if not torch.is_grad_enabled() or not self.theta.requires_grad:
            loss, _grad = self._stream_full_batches(self.theta, need_grad=False)
            return loss.to(device=self.theta.device, dtype=self.theta.dtype)
        return _GeneReconFullLossFunction.apply(self.theta, self)

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
    def pi_matrix(self, *, original_order: bool = True) -> torch.Tensor:
        """Return converged Pi rows for the retained uniform-transfer path.

        The model mode controls parameter sharing:
        ``global`` uses one theta vector, ``specieswise`` uses ``[S, 3]``
        theta, and ``genewise`` uses ``[G, 3]`` theta addressed by the cached
        family index.  In memory-safe resident-batch mode, this returns the
        active batch matrix; otherwise it returns the full resident matrix.
        """
        static = self._active_static()
        theta = self._active_theta()
        log_pS, log_pD, log_pL, max_transfer_vec = (
            _extract_parameters(theta.detach(), static)
        )
        e_max_iters = (
            static.fixed_iters_E
            if static.fixed_iters_E is not None
            else static.max_iters_E
        )
        e_tolerance = -1.0 if static.fixed_iters_E is not None else static.tol_E
        E_out = E_fixed_point(
            species_helpers=static.species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            max_transfer_mat=max_transfer_vec,
            max_iters=e_max_iters,
            tolerance=e_tolerance,
            warm_start_E=None,
            dtype=static.dtype,
            device=static.device,
            ancestors_T=static.ancestors_T,
        )
        Pi_out = Pi_wave_forward(
            wave_layout=static.wave_layout,
            species_helpers=static.species_helpers,
            E=E_out["E"],
            Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"],
            E_s2=E_out["E_s2"],
            log_pS=log_pS,
            log_pD=log_pD,
            max_transfer_mat=max_transfer_vec,
            device=static.device,
            dtype=static.dtype,
            fixed_iters=static.fixed_iters_Pi,
            return_original=original_order,
            return_root_rows=False,
            family_idx=(
                static.wave_layout.get("family_idx") if static.genewise else None
            ),
        )
        return Pi_out["Pi"] if original_order else Pi_out["Pi_wave_ordered"]

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
        if min_rate <= 0:
            raise ValueError("min_rate must be strictly positive")
        if max_rate is not None and max_rate < min_rate:
            raise ValueError("max_rate must be greater than or equal to min_rate")
        with torch.no_grad():
            self.theta.clamp_(
                min=math.log2(min_rate),
                max=None if max_rate is None else math.log2(max_rate),
            )

    @property
    def n_species(self) -> int:
        return int(self._dataset.S)

    @property
    def static(self) -> ReconStaticState:
        """Read-only access to the cached static state (for advanced use)."""
        return self._active_static()
