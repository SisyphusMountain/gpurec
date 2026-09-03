from __future__ import annotations

import os
from numbers import Integral, Real

import torch

from gpurec.config.memory import MemoryOptions

GIB = 1024 ** 3


def _int_dimension(name: str, value: int | float) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(number)
    raise ValueError(f"{name} must be an integer")


def _nonnegative_int(name: str, value: int | float) -> int:
    number = _int_dimension(name, value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_int(name: str, value: int | float) -> int:
    number = _int_dimension(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def cuda_memory_budget_bytes(
    device: torch.device | int | None = None,
    *,
    default_fraction: float = MemoryOptions().fraction,
    default_reserve_gib: float = MemoryOptions().reserve_gib,
) -> int | None:
    if not torch.cuda.is_available():
        return None
    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
    else:
        idx = int(device)
    free_b, total_b = torch.cuda.mem_get_info(idx)
    # mem_get_info's free_b is the DRIVER-visible free memory, which counts PyTorch's
    # reserved-but-unused caching-allocator pool as "used". A new torch allocation (e.g. the
    # transient self-loop scratch) can be served from that reclaimable pool WITHOUT touching the
    # driver, so the memory truly available to the next allocation is free_b + (reserved -
    # allocated). Omitting it makes the budget collapse to ~0 mid-run (when torch holds a large
    # reserved pool, e.g. the fp64 final_eval backward) and falsely rejects a scratch that fits.
    # At build time reserved ~= allocated (empty pool) so this adds nothing and the warm-adjoint
    # fit decision is unchanged; it only credits genuinely-reusable memory when a pool exists.
    reclaimable_b = max(0, int(torch.cuda.memory_reserved(idx)) - int(torch.cuda.memory_allocated(idx)))
    available_b = int(free_b) + reclaimable_b
    fraction = float(os.environ.get("GPUREC_MEMORY_POLICY_FRACTION", str(default_fraction)))
    reserve_b = int(float(os.environ.get("GPUREC_MEMORY_POLICY_RESERVE_GIB", str(default_reserve_gib))) * GIB)
    return max(0, min(int(total_b * fraction), max(0, available_b - reserve_b)))


def proposal0_wave_scratch_bytes(
    W: int, S: int, dtype: torch.dtype, *, scratch_tensors: int = MemoryOptions().scratch_tensors
) -> int:
    return (
        _nonnegative_int("W", W)
        * _positive_int("S", S)
        * _nonnegative_int("scratch_tensors", scratch_tensors)
        * dtype_nbytes(dtype)
    )


def proposal0_memory_gate(
    W: int,
    S: int,
    dtype: torch.dtype,
    *,
    device: torch.device | int | None = None,
    already_live_bytes: int = 0,
    reserved_scratch_bytes: int | None = None,
) -> tuple[bool, int, int | None]:
    """Gate a transient per-wave self-loop scratch allocation of ``W*S`` against the memory budget.

    ``reserved_scratch_bytes`` (default ``None``): when the warm-adjoint fit decision has ALREADY
    reserved transient scratch headroom for this backward (``warm_adjoint_fits`` verified
    ``cache + max_batch_scratch <= budget`` at build time, BEFORE the resident cache depleted free
    memory), pass that reservation here. A single wave's scratch (``W <= max_batch_clades``) is a
    subset of the reserved ``max_batch_scratch``, so it provably fits -- we must NOT re-read the now
    depleted ``cuda_memory_budget_bytes`` (which excludes the resident cache and would falsely reject
    a scratch the build already accounted for). The gate then validates the wave scratch against the
    reservation instead, so a scratch that GENUINELY exceeds what was reserved is still rejected.
    When ``None`` (cold path, no resident warm cache) the gate reads current free memory as before.
    """
    scratch = proposal0_wave_scratch_bytes(W, S, dtype)
    required = _nonnegative_int("already_live_bytes", already_live_bytes) + scratch
    if reserved_scratch_bytes is not None:
        reserved = _nonnegative_int("reserved_scratch_bytes", reserved_scratch_bytes)
        return scratch <= reserved, required, reserved
    budget = cuda_memory_budget_bytes(device)
    if budget is None:
        return True, required, budget
    return required <= budget, required, budget


def warm_adjoint_cache_bytes(total_clades: int, S: int, dtype: torch.dtype) -> int:
    """Resident bytes of the GPUREC_WARM_ADJOINT per-wave adjoint cache (``warm_v``).

    The cache stores one adjoint vector per clade-row per species node, accumulated over every wave of
    every batch and held resident for the whole run, so it scales as ``total_clades * S * dtype`` -- the
    SAME clades x species product as the per-batch backward scratch, but resident rather than transient.
    On large family sets this is the dominant consumer (full Hogenom: ~3.6M clades x 1331 species x 4 B
    ~ 19 GiB), far larger than the static base (~0.25 GiB) or one batch's scratch (~1.8 GiB).
    """
    return _nonnegative_int("total_clades", total_clades) * _positive_int("S", S) * dtype_nbytes(dtype)


def warm_adjoint_fits(
    total_clades: int,
    S: int,
    dtype: torch.dtype,
    *,
    device: torch.device | int | None = None,
    max_batch_clades: int = 0,
    resident_caches: int,
) -> tuple[bool, int, int, int | None]:
    """Decide whether the warm-adjoint cache(s) + one batch's backward scratch fit the memory budget.

    Returns ``(ok, cache_bytes, scratch_bytes, budget_bytes)``. ``ok`` is True (warm allowed) when the
    resident warm-adjoint caches PLUS the largest single batch's transient backward scratch fit within
    ``cuda_memory_budget_bytes`` (free - reserve, capped at a fraction of total). When it does not fit,
    the caller should run COLD -- the cache, not the clade batcher, is what exhausts memory on large
    family sets. This gates GPUREC_WARM_ADJOINT by the right quantity (clades x species x dtype) instead
    of a family count.

    ``resident_caches`` is how many ``[total_clades, S]`` caches stay resident: 1 for the gradient
    adjoint (``static.warm_v``) alone, plus one per Hessian-vector-product probe direction when the
    HVP warm start (``static.warm_v_tangent[probe_id]``) is enabled -- each probe keeps its own cache.
    Counting only the gradient cache let a 500-family / 13-batch run on a 94 GiB H100 pass the gate
    (32 GiB) and then OOM at 90 GiB inside the 3-probe Hessian.
    """
    cache = warm_adjoint_cache_bytes(total_clades, S, dtype) * _positive_int("resident_caches", resident_caches)
    scratch = proposal0_wave_scratch_bytes(max_batch_clades, S, dtype) if max_batch_clades else 0
    budget = cuda_memory_budget_bytes(device)
    if budget is None:
        return True, cache, scratch, budget
    return (cache + scratch) <= budget, cache, scratch, budget


# ---------------------------------------------------------------------------------------------
# Sizing one fit's GPU footprint, so the per-batch clade budget can be derived from the card.
#
# A genewise fit's allocated high-water mark has two parts, and only the second one depends on how
# the clades are split into batches:
#
#   1. the per-batch STATIC tensors -- the index and split-probability arrays
#      gpurec/core/scheduling/batching.py builds for every wave of every batch and keeps resident
#      for the whole fit. They scale with the dataset's TOTAL clades and splits, not with the batch
#      size: re-packing the same clades into more, smaller batches does not change their total.
#   2. one batch's transient WORKING SET -- the [clades x species] forward, adjoint and
#      curvature-probe buffers of whichever batch is being processed. This scales with the batch,
#      so the clade budget is the knob that sizes it.
#
# Both constants below were measured on the full Coleman set (job 57680456, H100 NVL: 5123
# families, 22,926,670 clades, 60,422,454 splits, S=2013, fp32, 79 batches of <=315,000 clades,
# peak 23.85 GiB of which 2.22 GiB static).
# ---------------------------------------------------------------------------------------------

# Resident static bytes per clade and per CCP split, summed over every batch of a fit. Per clade:
# ``perm`` (int64) + ``family_idx`` (int64) + ``leaf_species_index`` (int32) = 20 B. Per split:
# ``sl`` + ``sr`` + ``reduce_idx`` + ``eq1_reduce_idx`` (int32 each) + the model-dtype
# ``log_split_probs`` (4 B in fp32) = 20 B, plus the ``ge2_ptr`` / ``ge2_parent_ids`` fan-out arrays,
# which the measurement puts at about 4 B per split more. These are a fact of that module's tensor
# list rather than a tuning knob, so they are hard-wired here: change the tensors there and change
# these with them. Check: 20 B x 22.93M + 24 B x 60.42M = 1.78 GiB, against 1.77 GiB measured (the
# 2.22 GiB above minus the 0.45 GiB fp64 split-probability copy that batching.py no longer
# materializes for an fp32 model).
_STATIC_BYTES_PER_CLADE = 20
_STATIC_BYTES_PER_SPLIT = 24

# The caching allocator RESERVES more than it hands out. At the Coleman peak it held 28.82 GiB
# reserved against 23.85 GiB allocated, a factor of 1.21. A clade budget derived from the free
# memory must therefore aim the ALLOCATED peak below the free budget by that factor, or the
# reservation the allocator needs in order to serve it will not fit on the card. Not a setting: a
# measured property of PyTorch's allocator on this workload, rounded up from 1.21 for margin.
_RESERVED_PER_ALLOCATED = 1.25


def batch_static_bytes(total_clades: int, total_splits: int) -> int:
    """Resident bytes of the per-batch static tensors of a WHOLE fit (all batches together).

    Independent of the clade budget: the same clades and splits are described whether they are
    packed into few large batches or many small ones. See ``_STATIC_BYTES_PER_CLADE``.
    """
    return (
        _nonnegative_int("total_clades", total_clades) * _STATIC_BYTES_PER_CLADE
        + _nonnegative_int("total_splits", total_splits) * _STATIC_BYTES_PER_SPLIT
    )


def batch_working_set_bytes(clade_budget: int, S: int, dtype: torch.dtype, *, scratch_tensors: int) -> int:
    """Transient high-water bytes of ONE batch of ``clade_budget`` clades.

    The same ``scratch_tensors`` x clades x species x dtype shape as
    ``proposal0_wave_scratch_bytes``, applied to a whole batch rather than one wave: the forward
    Pi/Pibar, the adjoint, and the curvature probes' buffers are all [clades x species]. At the
    Coleman measurement the peak phase (the exact 3-probe analytic Hessian) sat at 21.63 GiB of
    transient over 315,000 clades x 2013 species x 4 B, i.e. 9.16 such tensors, so the default
    ``MemoryOptions.scratch_tensors`` of 10 is the measured multiplier with ~9 % of margin.
    """
    return proposal0_wave_scratch_bytes(clade_budget, S, dtype, scratch_tensors=scratch_tensors)


def clade_budget_for_device(
    *,
    total_clades: int,
    total_splits: int,
    S: int,
    dtype: torch.dtype,
    device: torch.device | int | None,
    fixed_clade_budget: int,
    scratch_tensors: int,
) -> tuple[int, dict]:
    """The largest per-batch clade budget whose predicted peak fits this device's memory budget.

    Returns ``(clade_budget, detail)``. ``clade_budget`` is never ABOVE ``fixed_clade_budget``: a
    card with memory to spare keeps the tuned batch size and the run is bit-for-bit the one it
    always was. It is only lowered when the fixed budget's predicted peak does not fit -- which is
    what lets the same fit that needs 23.85 GiB on a 94 GiB H100 also run on a 24 GB card.

    ``detail`` carries the numbers behind the decision (budget, static and working-set bytes, the
    predicted peak) so the caller can print why it chose what it chose.

    Raises ``ValueError`` when the dataset's resident statics alone exceed the device budget: no
    batch size can rescue that, and failing here says so far more clearly than an OOM 10 minutes in.
    """
    fixed = _positive_int("fixed_clade_budget", fixed_clade_budget)
    static_b = batch_static_bytes(total_clades, total_splits)
    budget_b = cuda_memory_budget_bytes(device)
    detail = {
        "device_budget_bytes": budget_b,
        "static_bytes": static_b,
        "fixed_clade_budget": fixed,
        "automatic": False,
    }
    if budget_b is None:   # no CUDA device to size against: keep the tuned batch size
        detail["working_set_bytes"] = batch_working_set_bytes(
            fixed, S, dtype, scratch_tensors=scratch_tensors)
        detail["predicted_peak_bytes"] = static_b + detail["working_set_bytes"]
        return fixed, detail
    # Aim the ALLOCATED peak low enough that the reservation needed to serve it still fits.
    allocatable_b = int(budget_b / _RESERVED_PER_ALLOCATED)
    usable_b = allocatable_b - static_b
    if usable_b <= 0:
        raise ValueError(
            f"the resident per-batch statics of this dataset ({static_b / GIB:.2f} GiB for "
            f"{total_clades:,} clades and {total_splits:,} splits) do not fit the device memory "
            f"budget ({budget_b / GIB:.2f} GiB, of which {allocatable_b / GIB:.2f} GiB is "
            f"allocatable); no per-batch clade budget can make this fit"
        )
    per_clade_b = _positive_int("scratch_tensors", scratch_tensors) * _positive_int("S", S)         * dtype_nbytes(dtype)
    affordable = usable_b // per_clade_b
    chosen = int(min(fixed, affordable))
    detail["automatic"] = chosen < fixed
    detail["allocatable_bytes"] = allocatable_b
    detail["affordable_clade_budget"] = int(affordable)
    detail["working_set_bytes"] = batch_working_set_bytes(
        chosen, S, dtype, scratch_tensors=scratch_tensors)
    detail["predicted_peak_bytes"] = static_b + detail["working_set_bytes"]
    return chosen, detail
