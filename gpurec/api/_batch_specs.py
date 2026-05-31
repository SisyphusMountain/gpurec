"""Batch planning and retained-Rust layout helpers for :class:`GeneReconModel`."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Mapping

import torch

from gpurec._validation import disabled_adaptive_neumann_terms_value
from gpurec._validation import (
    bool_value,
    nonnegative_int,
    nonnegative_float,
    optional_positive_int,
    positive_even_int,
    positive_float,
    positive_int,
)
from gpurec.core.batch_planning import (
    normalize_batch_packing,
    normalize_clade_budget,
    normalize_family_chunk_size,
    plan_family_batches,
)
from gpurec.core.model import GeneDataset
from ._family_layout import (
    FamilyWaveInputs,
    build_family_wave_layout,
    family_schedule_summary,
    family_wave_inputs,
    schedule_family_waves,
)
from ._model_types import (
    BatchMetadata,
    _FamilyScheduleStats,
    _ResidentBatchSpec,
)


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
    try:
        return nonnegative_int("prefetch_batches", value)
    except ValueError as exc:
        if "non-negative" in str(exc):
            raise ValueError("prefetch_batches must be non-negative or 'all'") from exc
        raise ValueError("prefetch_batches must be an integer or 'all'") from exc


def _normalize_gene_solver_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate public solver kwargs before CUDA setup or tree parsing."""
    normalized = dict(kwargs)
    removed_cache_kwargs = sorted(
        set(normalized).intersection(
            {"preprocess_cache_dir", "refresh_preprocess_cache"}
        )
    )
    if removed_cache_kwargs:
        names = ", ".join(removed_cache_kwargs)
        raise TypeError(f"preprocess caching has been removed; unsupported: {names}")
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
    if "adaptive_neumann_terms" in normalized:
        normalized["adaptive_neumann_terms"] = disabled_adaptive_neumann_terms_value(
            normalized["adaptive_neumann_terms"]
        )
    if "pi_adjoint_warmstart" in normalized:
        normalized["pi_adjoint_warmstart"] = bool_value(
            "pi_adjoint_warmstart",
            normalized["pi_adjoint_warmstart"],
        )
    if "pi_adjoint_cache_update_mode" in normalized:
        normalized["pi_adjoint_cache_update_mode"] = (
            _normalize_pi_adjoint_cache_update_mode(
                normalized["pi_adjoint_cache_update_mode"]
            )
        )
    if "pi_fixed_point_relaxation" in normalized:
        normalized["pi_fixed_point_relaxation"] = positive_float(
            "pi_fixed_point_relaxation",
            normalized["pi_fixed_point_relaxation"],
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
    if "small_family_max_leaves" in normalized:
        normalized["small_family_max_leaves"] = nonnegative_int(
            "small_family_max_leaves",
            normalized["small_family_max_leaves"],
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
    if "shared_loss_batch_streams" in normalized:
        normalized["shared_loss_batch_streams"] = positive_int(
            "shared_loss_batch_streams",
            normalized["shared_loss_batch_streams"],
        )
    adaptive_iters = normalized.get("adaptive_iters", False)
    convergence_check_interval = int(
        normalized.get("convergence_check_interval", 4)
    )
    if adaptive_iters and convergence_check_interval % 2 != 0:
        raise ValueError("adaptive_iters requires an even convergence_check_interval")
    return normalized


def _normalize_pi_adjoint_cache_update_mode(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            "pi_adjoint_cache_update_mode must be 'immediate' or 'stage'"
        )
    mode = value.strip().lower().replace("_", "-")
    if mode not in {"immediate", "stage"}:
        raise ValueError(
            "pi_adjoint_cache_update_mode must be 'immediate' or 'stage'"
        )
    return mode


def _parameter_mapping(
    *,
    mode: str,
    dataset: GeneDataset,
    family_indices: Any,
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
    clade_counts: list[int],
    family_chunk_size: int,
    clade_budget: int | None,
    batch_packing: str = "sequential",
    indices: list[int] | None = None,
    split_counts: list[int] | None = None,
    leaf_counts: list[int] | None = None,
    small_family_max_leaves: int | None = None,
    nonleaf_counts: list[int] | None = None,
    schedule_depths: list[int] | None = None,
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
            indices=indices,
            leaf_counts=leaf_counts,
            small_family_max_leaves=small_family_max_leaves,
            nonleaf_counts=nonleaf_counts,
            schedule_depths=schedule_depths,
            split_counts=split_counts,
            max_wave_size=max_wave_size,
        )
    ]


def _build_family_schedule_stats(
    dataset: GeneDataset,
    *,
    batch_packing: str,
    small_family_max_leaves: int,
) -> _FamilyScheduleStats:
    ensure_full = getattr(dataset, "_ensure_full_families", None)
    if callable(ensure_full):
        ensure_full()
    clade_counts = [int(fam["C"]) for fam in dataset.families]
    split_counts = [int(fam["N_splits"]) for fam in dataset.families]
    needs_depth_stats = _normalize_batch_packing(batch_packing) == "depth_first_fit"
    needs_leaf_stats = needs_depth_stats or small_family_max_leaves > 0
    if not needs_leaf_stats:
        return _FamilyScheduleStats(
            clade_counts=clade_counts,
            split_counts=split_counts,
        )

    summaries = [
        family_schedule_summary(fam["ccp_helpers"])
        for fam in dataset.families
    ]
    return _FamilyScheduleStats(
        clade_counts=clade_counts,
        split_counts=split_counts,
        leaf_counts=[int(summary["leaf_count"]) for summary in summaries],
        nonleaf_counts=(
            [int(summary["nonleaf_count"]) for summary in summaries]
            if needs_depth_stats
            else None
        ),
        schedule_depths=(
            [int(summary["max_level"]) for summary in summaries]
            if needs_depth_stats
            else None
        ),
    )


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
    small_family_max_leaves: int = 0,
    family_indices: list[int] | None = None,
    schedule_stats: _FamilyScheduleStats | None = None,
) -> list[_ResidentBatchSpec]:
    if schedule_stats is None:
        schedule_stats = _build_family_schedule_stats(
            dataset,
            batch_packing=batch_packing,
            small_family_max_leaves=small_family_max_leaves,
        )
    chunks = _family_index_chunks(
        total=len(dataset.families),
        clade_counts=schedule_stats.clade_counts,
        family_chunk_size=family_chunk_size,
        clade_budget=clade_budget,
        batch_packing=batch_packing,
        indices=family_indices,
        split_counts=schedule_stats.split_counts,
        leaf_counts=schedule_stats.leaf_counts,
        small_family_max_leaves=(
            small_family_max_leaves if small_family_max_leaves > 0 else None
        ),
        nonleaf_counts=schedule_stats.nonleaf_counts,
        schedule_depths=schedule_stats.schedule_depths,
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


def _dtype_name_for_rust(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    raise RuntimeError(f"unsupported resident Rust layout dtype {dtype}")


def _build_batch_specs_from_retained_rust(
    dataset: GeneDataset,
    *,
    mode: str,
    family_chunk_size: int,
    clade_budget: int | None,
    batch_packing: str,
    max_wave_size: int | None,
    max_root_wave_size: int | None,
    max_dts_partial_rows: int | None,
    small_family_max_leaves: int,
) -> list[_ResidentBatchSpec] | None:
    rust_preprocessed = getattr(dataset, "_rust_preprocessed", None)
    if rust_preprocessed is None or mode == "genewise" or small_family_max_leaves:
        return None
    build_chunked_layouts = getattr(rust_preprocessed, "build_chunked_layouts", None)
    if not callable(build_chunked_layouts):
        return None

    preprocess_cpu_cores = getattr(dataset, "_preprocess_cpu_cores", None)
    payloads = build_chunked_layouts(
        family_chunk_size=family_chunk_size,
        clade_budget=clade_budget,
        batch_packing=batch_packing,
        max_wave_size=max_wave_size,
        max_root_wave_size=max_root_wave_size,
        max_dts_partial_rows=max_dts_partial_rows,
        dtype=_dtype_name_for_rust(dataset.dtype),
        num_threads=0 if preprocess_cpu_cores is None else int(preprocess_cpu_cores),
        nonleaf_schedule_policy=(
            "forward" if batch_packing == "clade_first_fit" else "auto"
        ),
        include_family_idx=(mode == "genewise"),
    )
    specs: list[_ResidentBatchSpec] = []
    for batch_index, payload in enumerate(payloads):
        family_indices = [int(index) for index in payload["indices"]]
        wave_layout = dict(payload["wave_layout"])
        root_clade_rows: list[int] = []
        clade_offset = 0
        for family_index in family_indices:
            family = dataset.families[family_index]
            root_clade_rows.append(int(family["root_clade_id"]) + clade_offset)
            clade_offset += int(family["C"])
        metadata = BatchMetadata(
            batch_index=batch_index,
            family_indices=family_indices,
            family_names=[dataset.family_names[i] for i in family_indices],
            gene_tree_paths=[list(dataset.gene_tree_paths[i]) for i in family_indices],
            family_count=len(family_indices),
            clade_count=int(payload["clades"]),
            split_count=int(payload["splits"]),
            wave_count=int(payload["waves"]),
            max_wave_size=int(payload["max_wave"]),
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
                family_indices=family_indices,
                layout_inputs=None,
                waves=[],
                phases=[],
                metadata=metadata,
                wave_layout=wave_layout,
            )
        )
    return specs


def _retained_rust_layout_available(
    dataset: GeneDataset,
    *,
    mode: str,
    small_family_max_leaves: int,
) -> bool:
    rust_preprocessed = getattr(dataset, "_rust_preprocessed", None)
    if rust_preprocessed is None or mode == "genewise" or small_family_max_leaves:
        return False
    return callable(getattr(rust_preprocessed, "build_chunked_layouts", None))


def _start_batch_specs_from_retained_rust(
    dataset: GeneDataset,
    **kwargs: Any,
) -> tuple[ThreadPoolExecutor, Future] | None:
    if not _retained_rust_layout_available(
        dataset,
        mode=kwargs["mode"],
        small_family_max_leaves=kwargs["small_family_max_leaves"],
    ):
        return None
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="gpurec-rust-layout",
    )
    return executor, executor.submit(_build_batch_specs_from_retained_rust, dataset, **kwargs)


def _finish_batch_specs_from_retained_rust(
    handle: tuple[ThreadPoolExecutor, Future] | None,
) -> list[_ResidentBatchSpec] | None:
    if handle is None:
        return None
    executor, future = handle
    try:
        return future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _cancel_batch_specs_from_retained_rust(
    handle: tuple[ThreadPoolExecutor, Future] | None,
) -> None:
    if handle is None:
        return
    executor, _future = handle
    executor.shutdown(wait=True, cancel_futures=True)


def _should_use_compact_retained_preprocess(
    mode: str,
    solver_kwargs: Mapping[str, Any],
) -> bool:
    if mode == "genewise":
        return False
    small_family_max_leaves = solver_kwargs.get("small_family_max_leaves")
    if (
        small_family_max_leaves is not None
        and int(small_family_max_leaves) != 0
    ):
        return False
    return (
        bool_value("lazy_preprocess", solver_kwargs.get("lazy_preprocess", False))
        or solver_kwargs.get("family_chunk_size") is not None
        or solver_kwargs.get("clade_budget") is not None
    )
