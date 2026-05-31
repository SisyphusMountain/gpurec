"""Resident-batch static cache and prefetch management."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable

from ._model_types import _ResidentBatchSpec
from .autograd import ReconStaticState


_RESIDENT_PREFETCH_WORKERS = 3


class ResidentBatchCache:
    def __init__(
        self,
        specs: list[_ResidentBatchSpec],
        build_static: Callable[[int], ReconStaticState],
        *,
        prefetch_batches: int | str,
    ):
        self.specs = specs
        self.statics: list[ReconStaticState | None] = [None for _ in specs]
        self.futures: dict[int, Future[ReconStaticState]] = {}
        self.executor: ThreadPoolExecutor | None = None
        self.lock = Lock()
        self.current_index = 0
        self.prefetch_batches = prefetch_batches
        self._build_static = build_static
        self.closed = False

    def ensure(self, index: int) -> ReconStaticState:
        if index < 0 or index >= len(self.specs):
            raise IndexError(
                f"batch index {index} out of range for {len(self.specs)} batches"
            )
        with self.lock:
            static = self.statics[index]
            future = self.futures.pop(index, None)
        if static is not None:
            return static
        if future is not None:
            static = future.result()
        else:
            static = self._build_static(index)
        with self.lock:
            existing = self.statics[index]
            if existing is None:
                self.statics[index] = static
                return static
            return existing

    def select(self, index: int) -> ReconStaticState:
        if index < 0 or index >= len(self.specs):
            raise IndexError(
                f"batch index {index} out of range for {len(self.specs)} batches"
            )
        self.current_index = index
        return self.ensure(index)

    def schedule_prefetch(self) -> None:
        if self.closed or self.prefetch_batches == 0:
            return
        start = self.current_index + 1
        if self.prefetch_batches == "all":
            stop = len(self.specs)
        else:
            stop = min(len(self.specs), start + int(self.prefetch_batches))
        for batch_idx in range(start, stop):
            self._submit_prefetch(batch_idx)

    def _submit_prefetch(self, batch_idx: int) -> None:
        if self.closed:
            return
        if batch_idx < 0 or batch_idx >= len(self.specs):
            return
        with self.lock:
            if self.statics[batch_idx] is not None or batch_idx in self.futures:
                return
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=_RESIDENT_PREFETCH_WORKERS,
                    thread_name_prefix="gpurec-preprocess",
                )
            self.futures[batch_idx] = self.executor.submit(
                self._build_static,
                batch_idx,
            )

    def submit_prefetch(self, batch_idx: int) -> None:
        self._submit_prefetch(batch_idx)

    def drop_statics(self) -> None:
        self.close_executor()
        with self.lock:
            self.statics = [None for _ in self.specs]

    def clear_active_runtime(self) -> None:
        with self.lock:
            static = self.statics[self.current_index]
        if static is not None:
            static.warm_E = None

    def close(self) -> None:
        self.closed = True
        self.close_executor()
        with self.lock:
            self.futures.clear()

    def close_executor(self) -> None:
        with self.lock:
            executor = self.executor
            self.executor = None
            self.futures.clear()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def cached(self) -> list[ReconStaticState]:
        return [static for static in self.statics if static is not None]
