"""Memory estimates and GPU policy helpers for uniform wave kernels."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch

from .batch_planning import plan_family_batches


GIB = 1024 ** 3


def _positive_int(name: str, value: int) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_int(name: str, value: int) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def cuda_memory_budget_bytes(
    device: torch.device | int | None = None,
    *,
    fraction_env: str = "GPUREC_MEMORY_POLICY_FRACTION",
    reserve_gib_env: str = "GPUREC_MEMORY_POLICY_RESERVE_GIB",
    default_fraction: float = 0.85,
    default_reserve_gib: float = 1.0,
) -> int | None:
    """Return a conservative usable-memory budget for the current GPU.

    The budget is based on both total VRAM and currently free VRAM. The tensor
    formulas below are exact payload-byte formulas, but allocator rounding,
    cached blocks, profiler state, and transient PyTorch temporaries require a
    margin. That margin is controlled by the fraction and reserve env vars.
    """
    if not torch.cuda.is_available():
        return None
    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
    else:
        idx = int(device)
    free_b, total_b = torch.cuda.mem_get_info(idx)
    fraction = float(os.environ.get(fraction_env, str(default_fraction)))
    reserve_b = int(float(os.environ.get(reserve_gib_env, str(default_reserve_gib))) * GIB)
    total_budget = int(total_b * fraction)
    free_budget = max(0, int(free_b) - reserve_b)
    return max(0, min(total_budget, free_budget))


def proposal0_wave_scratch_bytes(
    W: int,
    S: int,
    dtype: torch.dtype,
    *,
    scratch_tensors: int = 10,
) -> int:
    """Exact payload bytes for Proposal 0's current `[W, S]` scratch set."""
    rows = _nonnegative_int("W", W)
    species = _positive_int("S", S)
    tensors = _nonnegative_int("scratch_tensors", scratch_tensors)
    return tensors * rows * species * dtype_nbytes(dtype)


def uniform_training_dense_state_bytes(
    C: int,
    S: int,
    dtype: torch.dtype,
    *,
    dense_state_tensors: int = 3,
) -> int:
    """Payload bytes for resident dense training state.

    The current uniform training path keeps `Pi`, `Pibar`, and accumulated RHS
    state live for a chunk, hence the default factor of 3.
    """
    clades = _nonnegative_int("C", C)
    species = _positive_int("S", S)
    tensors = _nonnegative_int("dense_state_tensors", dense_state_tensors)
    return tensors * clades * species * dtype_nbytes(dtype)


def uniform_training_payload_bytes(
    C: int,
    S: int,
    dtype: torch.dtype,
    *,
    max_wave_rows: int,
    dense_state_tensors: int = 3,
) -> int:
    clades = _nonnegative_int("C", C)
    max_wave_rows = _positive_int("max_wave_rows", max_wave_rows)
    dense = uniform_training_dense_state_bytes(
        clades, S, dtype, dense_state_tensors=dense_state_tensors
    )
    wave = proposal0_wave_scratch_bytes(max_wave_rows, S, dtype)
    # Row maxima, root outputs, E/theta adjoints, active masks, and compact
    # topology arrays are small relative to dense `[C, S]` state but not zero.
    small_state = clades * dtype_nbytes(dtype) * 8
    return dense + wave + small_state


def estimate_chunk_payload_bytes(
    clade_counts: Sequence[int],
    S: int,
    dtype: torch.dtype,
    *,
    family_chunk_size: int,
    max_wave_size: int,
    clade_budget: int | None = None,
    batch_packing: str = "sequential",
    leaf_counts: Sequence[int] | None = None,
    nonleaf_counts: Sequence[int] | None = None,
    schedule_depths: Sequence[int] | None = None,
) -> int:
    species = _positive_int("S", S)
    max_wave = _positive_int("max_wave_size", max_wave_size)
    plans = plan_family_batches(
        clade_counts=clade_counts,
        family_chunk_size=int(family_chunk_size),
        clade_budget=clade_budget,
        batch_packing=batch_packing,
        leaf_counts=leaf_counts,
        nonleaf_counts=nonleaf_counts,
        schedule_depths=schedule_depths,
        max_wave_size=max_wave,
    )
    chunk_clades = max((int(plan.clades) for plan in plans), default=0)
    max_wave_rows = min(max_wave, max(1, int(chunk_clades)))
    return uniform_training_payload_bytes(
        chunk_clades,
        species,
        dtype,
        max_wave_rows=max_wave_rows,
    )


@dataclass(frozen=True)
class UniformPipelinePolicy:
    family_chunk_size: int
    max_wave_size: int
    estimated_payload_bytes: int
    budget_bytes: int | None
    reason: str


def choose_uniform_pipeline_policy(
    clade_counts: Sequence[int],
    S: int,
    dtype: torch.dtype,
    *,
    device: torch.device | int | None = None,
    family_chunk_candidates: Sequence[int] = (25, 50, 10, 75, 100),
    max_wave_candidates: Sequence[int] = (8192, 16384, 4096, 32768),
    clade_budget: int | None = None,
    batch_packing: str = "sequential",
    leaf_counts: Sequence[int] | None = None,
    nonleaf_counts: Sequence[int] | None = None,
    schedule_depths: Sequence[int] | None = None,
) -> UniformPipelinePolicy:
    """Choose a practical uniform training policy from measured-safe candidates.

    Candidate order encodes profiling knowledge; the memory formula decides
    whether a candidate is feasible on the current GPU and workload.
    """
    chunk_candidates = [
        _nonnegative_int("family_chunk_candidates entries", candidate)
        for candidate in family_chunk_candidates
    ]
    wave_candidates = [
        _positive_int("max_wave_candidates entries", candidate)
        for candidate in max_wave_candidates
    ]
    budget = cuda_memory_budget_bytes(device)
    for chunk_size in chunk_candidates:
        for max_wave in wave_candidates:
            est = estimate_chunk_payload_bytes(
                clade_counts,
                S,
                dtype,
                family_chunk_size=chunk_size,
                max_wave_size=max_wave,
                clade_budget=clade_budget,
                batch_packing=batch_packing,
                leaf_counts=leaf_counts,
                nonleaf_counts=nonleaf_counts,
                schedule_depths=schedule_depths,
            )
            if budget is None or est <= budget:
                return UniformPipelinePolicy(
                    family_chunk_size=int(chunk_size),
                    max_wave_size=int(max_wave),
                    estimated_payload_bytes=est,
                    budget_bytes=budget,
                    reason="first_profiled_candidate_with_estimated_payload_within_budget",
                )

    raise RuntimeError(
        "no retained uniform pipeline policy fits the memory budget "
        f"({(budget or 0) / (1024 ** 3):.2f} GiB)"
    )


def proposal0_memory_gate(
    W: int,
    S: int,
    dtype: torch.dtype,
    *,
    device: torch.device | int | None = None,
    already_live_bytes: int = 0,
) -> tuple[bool, int, int | None]:
    required = _nonnegative_int(
        "already_live_bytes",
        already_live_bytes,
    ) + proposal0_wave_scratch_bytes(W, S, dtype)
    budget = cuda_memory_budget_bytes(device)
    if budget is None:
        return True, required, budget
    return required <= budget, required, budget
