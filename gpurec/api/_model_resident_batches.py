"""Resident batch lifecycle wrappers for ``GeneReconModel``.

This module is internal support for ``gpurec.api`` model methods, not a public
import surface. It owns resident-batch selection, cache, prefetch, and streaming
wrappers only; likelihood and autograd stay in ``model`` while constructor setup
stays in private init/runtime helpers.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from . import _resident_runtime
from ._model_types import ActiveFamilyBatch, BatchMetadata
from ._streaming import stream_full_batches as _stream_full_batches_impl
from .autograd import ReconStaticState


class _GeneReconModelResidentBatchMixin:
    """Private mixin for ``GeneReconModel`` resident batch lifecycle methods."""

    def _build_batch_static(self, batch_idx: int) -> ReconStaticState:
        return _resident_runtime._build_batch_static(self, batch_idx)

    def _shutdown_prefetch_executor_for_replan(self) -> None:
        _resident_runtime._shutdown_prefetch_executor_for_replan(self)

    def replan_resident_batches(
        self,
        family_indices: Sequence[int],
    ) -> list[BatchMetadata]:
        """Rebuild resident batch specs for selected genewise family rows.

        This is an internal workflow hook for adaptive genewise optimization.
        It reuses preprocessed family payloads and cached per-family scheduler
        stats, then asks the Rust planner/scheduler to regroup and regenerate
        waves for the selected original family indices.
        """
        return _resident_runtime.replan_resident_batches(self, family_indices)

    def _ensure_batch_static(self, batch_idx: int) -> ReconStaticState:
        return _resident_runtime._ensure_batch_static(self, batch_idx)

    def _submit_prefetch(self, batch_idx: int) -> None:
        _resident_runtime._submit_prefetch(self, batch_idx)

    def _schedule_prefetch(self) -> None:
        _resident_runtime._schedule_prefetch(self)

    def _legacy_prefetch_guard(self):
        return _resident_runtime._legacy_prefetch_guard(self)

    def _submit_legacy_prefetch(self, batch_idx: int) -> None:
        _resident_runtime._submit_legacy_prefetch(self, batch_idx)

    def _schedule_legacy_prefetch(self) -> None:
        _resident_runtime._schedule_legacy_prefetch(self)

    def _active_static(self) -> ReconStaticState:
        return _resident_runtime._active_static(self)

    def _theta_for_batch_index(
        self,
        batch_idx: int,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        return _resident_runtime._theta_for_batch_index(self, batch_idx, theta)

    def _active_theta(self, theta: torch.Tensor | None = None) -> torch.Tensor:
        return _resident_runtime._active_theta(self, theta)

    def _stream_full_batches(
        self,
        theta: torch.Tensor,
        *,
        need_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return _stream_full_batches_impl(self, theta, need_grad=need_grad)

    @property
    def current_batch_metadata(self) -> BatchMetadata:
        """Metadata for the resident batch currently selected by the model."""
        return self.batch_metadata[self._current_batch_index]

    @property
    def current_batch_index(self) -> int:
        """Index of the resident batch currently selected for evaluation."""
        return self._current_batch_index

    @property
    def cached_static_states(self) -> list[ReconStaticState]:
        """Static states that are currently built and available for diagnostics."""
        return _resident_runtime.cached_static_states(self)

    def drop_cached_static_states(self) -> None:
        """Release built resident batch static states while keeping batch metadata."""
        _resident_runtime.drop_cached_static_states(self)

    def materialize_batches(self) -> list[BatchMetadata]:
        """Build all resident batch static states and return metadata copies.

        In resident-batch mode this forces every batch static state to be built
        before returning.  The returned list is a copy of ``batch_metadata``, so
        callers can inspect batch ownership without mutating model bookkeeping.
        """
        return _resident_runtime.materialize_batches(self)

    def active_theta(self, theta: torch.Tensor | None = None) -> torch.Tensor:
        """Return theta as addressed by the currently selected resident batch."""
        return self._active_theta(theta)

    def select_batch(self, batch_index: int) -> BatchMetadata:
        """Select a resident batch and return its metadata.

        In non-batched mode only batch ``0`` exists.  Selecting a new batch
        clears warm runtime state from the previous active batch.
        """
        return _resident_runtime.select_batch(self, batch_index)

    def activate_family(self, family_index: int) -> ActiveFamilyBatch:
        """Select the resident batch containing ``family_index``.

        Returns the family offset inside the active Pi matrix plus the local
        family index used by batch-local parameter tensors.
        """
        return _resident_runtime.activate_family(self, family_index)

    def next(self) -> BatchMetadata:
        """Advance to the next resident batch and return its metadata."""
        return _resident_runtime.next_batch(self)

    def clear(self) -> None:
        """Release active runtime caches held by the model."""
        _resident_runtime.clear(self)

    def close(self) -> None:
        """Stop background batch preprocessing and drop pending futures."""
        _resident_runtime.close(self)

    def _close_legacy_prefetch_executor(self) -> None:
        _resident_runtime._close_legacy_prefetch_executor(self)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
