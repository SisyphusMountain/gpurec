"""Resident runtime and batch lifecycle helpers for ``GeneReconModel``.

This module is internal support for ``gpurec.api`` model methods, not a public
import surface.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Sequence

import torch

from ._batch_specs import _build_batch_specs as _build_batch_specs_impl
from ._model_types import ActiveFamilyBatch, BatchMetadata
from ._resident_cache import ResidentBatchCache, _RESIDENT_PREFETCH_WORKERS
from ._static_builder import _build_batch_static_state as _build_batch_static_state_impl
from ._validation import integer_value
from .autograd import ReconStaticState, _clear_pi_adjoint_runtime_cache


def _build_batch_static(model: Any, batch_idx: int) -> ReconStaticState:
    static = _build_batch_static_state_impl(
        model._batch_specs[batch_idx],
        dataset=model._dataset,
        common_state=model._resident_common_state,
        origination_prior=model._origination_prior.select_families(
            model._batch_specs[batch_idx].family_indices,
        ),
        settings=model._settings,
    )
    model._apply_pi_adjoint_warmstart_config(static, clear_cache=False)
    return static


def _shutdown_prefetch_executor_for_replan(model: Any) -> None:
    cache = getattr(model, "_resident_cache", None)
    if cache is not None:
        cache.close()
    model._resident_cache = None


def replan_resident_batches(
    model: Any,
    family_indices: Sequence[int],
) -> list[BatchMetadata]:
    if not model._batched_resident or model._mode != "genewise":
        raise RuntimeError(
            "replan_resident_batches() requires genewise resident-batch mode"
        )
    indices = [integer_value("family_indices entries", value) for value in family_indices]
    if not indices:
        raise ValueError("family_indices must not be empty")
    seen: set[int] = set()
    for index in indices:
        if index < 0 or index >= model.n_families:
            raise IndexError(
                f"family index {index} out of range for {model.n_families} families"
            )
        if index in seen:
            raise ValueError(f"duplicate family index {index}")
        seen.add(index)
    if model._family_schedule_stats is None:
        raise RuntimeError("family scheduler stats are not available")

    model._shutdown_prefetch_executor_for_replan()
    specs = _build_batch_specs_impl(
        model._dataset,
        mode=model._mode,
        family_chunk_size=model.family_chunk_size,
        clade_budget=model.clade_budget,
        batch_packing=model.batch_packing,
        max_wave_size=model.max_wave_size,
        max_root_wave_size=model.max_root_wave_size,
        max_dts_partial_rows=model.max_dts_partial_rows,
        small_family_max_leaves=model.small_family_max_leaves,
        family_indices=indices,
        schedule_stats=model._family_schedule_stats,
    )
    if not specs:
        raise ValueError("replanned resident batches must not be empty")
    model._batch_specs = specs
    model.batch_metadata = [spec.metadata for spec in specs]
    model._current_batch_index = 0
    model._resident_cache = ResidentBatchCache(
        specs=model._batch_specs,
        build_static=model._build_batch_static,
        prefetch_batches=model.prefetch_batches,
    )
    model._resident_cache.ensure(0)
    model._resident_cache.schedule_prefetch()
    return list(model.batch_metadata)


def _ensure_batch_static(model: Any, batch_idx: int) -> ReconStaticState:
    if not model._batched_resident:
        if model._static is None:
            raise RuntimeError("resident static state has not been built")
        return model._static
    if batch_idx < 0 or batch_idx >= len(model._batch_specs):
        raise IndexError(
            f"batch index {batch_idx} out of range for {len(model._batch_specs)} batches"
        )
    cache = getattr(model, "_resident_cache", None)
    if cache is not None:
        return cache.ensure(batch_idx)
    raise RuntimeError("resident cache is not initialized")


def _submit_prefetch(model: Any, batch_idx: int) -> None:
    if batch_idx < 0 or batch_idx >= len(model._batch_specs):
        return
    cache = getattr(model, "_resident_cache", None)
    if cache is not None:
        cache.submit_prefetch(batch_idx)
        return
    if hasattr(model, "_batch_statics") or getattr(model, "_prefetch_closed", False):
        model._submit_legacy_prefetch(batch_idx)
        return
    raise RuntimeError("resident cache is not initialized")


def _schedule_prefetch(model: Any) -> None:
    if not model._batched_resident or model.prefetch_batches == 0:
        return
    cache = getattr(model, "_resident_cache", None)
    if cache is not None:
        cache.schedule_prefetch()
        return
    if hasattr(model, "_batch_statics") or getattr(model, "_prefetch_closed", False):
        model._schedule_legacy_prefetch()
        return
    raise RuntimeError("resident cache is not initialized")


def _legacy_prefetch_guard(model: Any):
    lock = getattr(model, "_batch_lock", None)
    return lock if lock is not None else nullcontext()


def _submit_legacy_prefetch(model: Any, batch_idx: int) -> None:
    if getattr(model, "_prefetch_closed", False):
        return
    statics = getattr(model, "_batch_statics", None)
    futures = getattr(model, "_batch_futures", None)
    if statics is None:
        raise RuntimeError("resident cache is not initialized")
    if futures is None:
        futures = {}
        model._batch_futures = futures
    if batch_idx < 0 or batch_idx >= len(statics):
        return
    with model._legacy_prefetch_guard():
        if statics[batch_idx] is not None or batch_idx in futures:
            return
        executor = getattr(model, "_prefetch_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=_RESIDENT_PREFETCH_WORKERS,
                thread_name_prefix="gpurec-preprocess",
            )
            model._prefetch_executor = executor
        futures[batch_idx] = executor.submit(model._build_batch_static, batch_idx)


def _schedule_legacy_prefetch(model: Any) -> None:
    if getattr(model, "_prefetch_closed", False):
        return
    statics = getattr(model, "_batch_statics", None)
    if statics is None:
        raise RuntimeError("resident cache is not initialized")
    start = model._current_batch_index + 1
    if model.prefetch_batches == "all":
        stop = len(statics)
    else:
        stop = min(len(statics), start + int(model.prefetch_batches))
    for batch_idx in range(start, stop):
        model._submit_legacy_prefetch(batch_idx)


def _active_static(model: Any) -> ReconStaticState:
    if model._batched_resident:
        return model._ensure_batch_static(model._current_batch_index)
    if model._static is None:
        raise RuntimeError("resident static state has not been built")
    return model._static


def _theta_for_batch_index(
    model: Any,
    batch_idx: int,
    theta: torch.Tensor,
) -> torch.Tensor:
    if not model._batched_resident or model._mode != "genewise":
        return theta
    idx = torch.as_tensor(
        model._batch_specs[batch_idx].family_indices,
        dtype=torch.long,
        device=theta.device,
    )
    return theta.index_select(0, idx)


def _active_theta(model: Any, theta: torch.Tensor | None = None) -> torch.Tensor:
    return model._theta_for_batch_index(
        model._current_batch_index,
        model.theta if theta is None else theta,
    )


def cached_static_states(model: Any) -> list[ReconStaticState]:
    if model._batched_resident:
        cache = getattr(model, "_resident_cache", None)
        if cache is not None:
            return cache.cached()
        return []
    return [] if model._static is None else [model._static]


def drop_cached_static_states(model: Any) -> None:
    if not model._batched_resident:
        return
    cache = getattr(model, "_resident_cache", None)
    if cache is not None:
        cache.drop_statics()
        return


def materialize_batches(model: Any) -> list[BatchMetadata]:
    if model._batched_resident:
        cache = getattr(model, "_resident_cache", None)
        if cache is not None:
            for batch_idx in range(len(model._batch_specs)):
                cache.ensure(batch_idx)
            return list(model.batch_metadata)
        for batch_idx in range(len(model._batch_specs)):
            model._ensure_batch_static(batch_idx)
    elif model._static is None:
        raise RuntimeError("resident static state has not been built")
    return list(model.batch_metadata)


def select_batch(model: Any, batch_index: int) -> BatchMetadata:
    batch_index = integer_value("batch_index", batch_index)
    if batch_index < 0 or batch_index >= len(model.batch_metadata):
        raise IndexError(
            f"batch index {batch_index} out of range for {len(model.batch_metadata)} batches"
        )
    if batch_index != model._current_batch_index:
        model.clear()
        model._current_batch_index = batch_index
    cache = getattr(model, "_resident_cache", None)
    if cache is not None and model._batched_resident:
        cache.select(batch_index)
    else:
        model._ensure_batch_static(batch_index)
    model._schedule_prefetch()
    return model.current_batch_metadata


def activate_family(model: Any, family_index: int) -> ActiveFamilyBatch:
    family_index = integer_value("family_index", family_index)
    if family_index < 0 or family_index >= model.n_families:
        raise IndexError(f"family_index {family_index} outside 0..{model.n_families}")

    if not model._batched_resident:
        offset = 0
        for idx in range(family_index):
            offset += int(model._dataset.families[idx]["C"])
        metadata = model.select_batch(0)
        return ActiveFamilyBatch(
            family_index=family_index,
            batch_index=0,
            local_family_index=family_index,
            clade_offset=offset,
            metadata=metadata,
        )

    for batch_idx, metadata in enumerate(model.batch_metadata):
        family_indices = [int(idx) for idx in metadata.family_indices]
        if family_index not in family_indices:
            continue
        offset = 0
        for local_idx, idx in enumerate(family_indices):
            if idx == family_index:
                metadata = model.select_batch(batch_idx)
                return ActiveFamilyBatch(
                    family_index=family_index,
                    batch_index=batch_idx,
                    local_family_index=local_idx,
                    clade_offset=offset,
                    metadata=metadata,
                )
            offset += int(model._dataset.families[idx]["C"])
    raise IndexError(f"family_index {family_index} is not present in any resident batch")


def next_batch(model: Any) -> BatchMetadata:
    if model._current_batch_index + 1 >= len(model.batch_metadata):
        raise StopIteration("already at the final resident batch")
    return model.select_batch(model._current_batch_index + 1)


def clear(model: Any) -> None:
    if model._batched_resident:
        cache = getattr(model, "_resident_cache", None)
        if cache is not None:
            cache.clear_active_runtime()
            static = cache.statics[cache.current_index]
            if static is not None:
                _clear_pi_adjoint_runtime_cache(static)
            return
        statics = getattr(model, "_batch_statics", None)
        if statics is not None:
            batch_idx = getattr(model, "_current_batch_index", 0)
            if 0 <= batch_idx < len(statics):
                static = statics[batch_idx]
                if static is not None:
                    static.warm_E = None
                    _clear_pi_adjoint_runtime_cache(static)
            return
        raise RuntimeError("resident cache is not initialized")
    static = model._active_static()
    static.warm_E = None
    _clear_pi_adjoint_runtime_cache(static)


def close(model: Any) -> None:
    model._prefetch_closed = True
    cache = getattr(model, "_resident_cache", None)
    if cache is not None:
        cache.close()
        model._resident_cache = None
    model._close_legacy_prefetch_executor()


def _close_legacy_prefetch_executor(model: Any) -> None:
    executor = getattr(model, "_prefetch_executor", None)
    lock = getattr(model, "_batch_lock", None)
    if lock is None:
        model._prefetch_executor = None
        futures = getattr(model, "_batch_futures", None)
        if futures is not None:
            futures.clear()
    else:
        with lock:
            model._prefetch_executor = None
            futures = getattr(model, "_batch_futures", None)
            if futures is not None:
                futures.clear()
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
