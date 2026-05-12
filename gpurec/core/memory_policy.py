"""Memory estimates and GPU policy helpers for uniform wave kernels."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch


GIB = 1024 ** 3


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
    return int(scratch_tensors) * int(W) * int(S) * dtype_nbytes(dtype)


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
    return int(dense_state_tensors) * int(C) * int(S) * dtype_nbytes(dtype)


def uniform_training_payload_bytes(
    C: int,
    S: int,
    dtype: torch.dtype,
    *,
    max_wave_rows: int,
    dense_state_tensors: int = 3,
) -> int:
    dense = uniform_training_dense_state_bytes(
        C, S, dtype, dense_state_tensors=dense_state_tensors
    )
    wave = proposal0_wave_scratch_bytes(max_wave_rows, S, dtype)
    # Row maxima, root outputs, E/theta adjoints, active masks, and compact
    # topology arrays are small relative to dense `[C, S]` state but not zero.
    small_state = int(C) * dtype_nbytes(dtype) * 8
    return dense + wave + small_state


def estimate_chunk_payload_bytes(
    clade_counts: Sequence[int],
    S: int,
    dtype: torch.dtype,
    *,
    family_chunk_size: int,
    max_wave_size: int,
) -> int:
    if family_chunk_size <= 0:
        chunk_clades = sum(int(c) for c in clade_counts)
    else:
        chunk_clades = 0
        for start in range(0, len(clade_counts), family_chunk_size):
            chunk_clades = max(
                chunk_clades,
                sum(int(c) for c in clade_counts[start:start + family_chunk_size]),
            )
    max_wave_rows = min(int(max_wave_size), max(1, int(chunk_clades)))
    return uniform_training_payload_bytes(
        chunk_clades,
        S,
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
) -> UniformPipelinePolicy:
    """Choose a practical uniform training policy from measured-safe candidates.

    Candidate order encodes profiling knowledge; the memory formula decides
    whether a candidate is feasible on the current GPU and workload.
    """
    budget = cuda_memory_budget_bytes(device)
    for chunk_size in family_chunk_candidates:
        for max_wave in max_wave_candidates:
            est = estimate_chunk_payload_bytes(
                clade_counts,
                S,
                dtype,
                family_chunk_size=int(chunk_size),
                max_wave_size=int(max_wave),
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
    required = int(already_live_bytes) + proposal0_wave_scratch_bytes(W, S, dtype)
    budget = cuda_memory_budget_bytes(device)
    if budget is None:
        return True, required, budget
    return required <= budget, required, budget
