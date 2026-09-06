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
    # At build time reserved ~= allocated (empty pool), so this only credits genuinely reusable
    # memory once a caching-allocator pool exists.
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

    ``reserved_scratch_bytes`` can provide a sweep-wide budget already read by the caller. The gate
    then avoids another blocking free-memory query for every wave. When it is ``None``, the gate
    reads the current device budget directly.
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


def wave_scratch_budget_bytes(reserved_scratch_bytes: int | None, *, device) -> int | None:
    """Resolve, ONCE for a whole reverse sweep, the byte budget every wave's scratch is gated on.

    ``proposal0_memory_gate`` above compares one wave's scratch against either an already resolved
    sweep-wide budget (``reserved_scratch_bytes``) or the free memory it reads on the spot. Reading
    it on the spot costs one blocking ``cudaMemGetInfo`` plus two
    ``torch.cuda.memory_stats()`` nested-dict builds, so a 1977-wave gradient once paid for that
    query 1977 times: 88.7 ms of GPU idle and 65.3 ms of host time, about 6 % of one 200-family
    Coleman gradient.

    Every wave of one sweep gets the same answer anyway -- the scratch is transient (allocated and
    freed inside the wave), so nothing a wave does moves the budget the next wave would read. Read
    it once here and hand the number down as the reservation; the gate then makes the identical
    comparison (``scratch <= budget``, since the callers pass no ``already_live_bytes``) without the
    driver round trip. A wave whose scratch genuinely exceeds the budget is still rejected.

    Returns ``None`` off CUDA, which leaves the gate exactly as it was (it returns "fits" when there
    is no budget to read).
    """
    if reserved_scratch_bytes is not None:
        return _nonnegative_int("reserved_scratch_bytes", reserved_scratch_bytes)
    return cuda_memory_budget_bytes(device)


def forward_gene_split_cache_fits(
    max_batch_clades: int,
    S: int,
    dtype: torch.dtype,
    *,
    device: torch.device | int | None,
    scratch_tensors: int,
) -> tuple[bool, int, int, int | None]:
    """Decide whether the backward may read the forward's gene-split (DTS) rows instead of redoing them.

    The forward computes one ``[wave clades x species]`` gene-split row block per split wave and
    throws it away; the backward then recomputes exactly the same rows (three Triton reduction
    launches per wave, 11 % of one gradient on the 200-family Coleman batch). Keeping them costs
    ONE more ``[batch clades x species]`` tensor per batch -- the wave blocks tile that shape --
    on top of the ``scratch_tensors`` that ``batch_working_set_bytes`` already budgets for a
    batch's working set. Its ``[batch clades]`` row-offset sidecar adds another 1/S of that, which
    at S in the thousands is below the rounding of this estimate.

    Returns ``(ok, cache_bytes, working_set_bytes, budget_bytes)``. ``ok`` False means the caller
    should let the backward recompute, which is what it always did.
    """
    cache = proposal0_wave_scratch_bytes(max_batch_clades, S, dtype, scratch_tensors=1)
    working = batch_working_set_bytes(
        max_batch_clades, S, dtype, scratch_tensors=scratch_tensors
    )
    budget = cuda_memory_budget_bytes(device)
    if budget is None:
        return True, cache, working, budget
    return (working + cache) <= budget, cache, working, budget


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
# families, 22,931,793 clades, 71,885,789 splits as the parser counts them, S=2013, fp32, 79
# batches of <=315,000 clades, peak 23.85 GiB of which 2.22 GiB static).
# ---------------------------------------------------------------------------------------------

# Resident static bytes per clade and per CCP split, summed over every batch of a fit. Per clade:
# ``perm`` (int64) + ``family_idx`` (int64) + ``leaf_species_index`` (int32) = 20 B. Per split:
# ``sl`` + ``sr`` + ``reduce_idx`` (int32 each) = 12 B, the model-dtype ``log_split_probs`` (4 B in
# fp32), and 4 B more for the ``eq1_reduce_idx`` / ``ge2_ptr`` / ``ge2_parent_ids`` arrays, which
# only cover part of the splits each -- 20 B in total. These are a fact of that module's tensor list
# rather than a tuning knob, so they are hard-wired here: change the tensors there and change these
# with them.
#
# Check against the measurement: the 2.22 GiB above includes the fp64 split-probability copy that
# batching.py no longer materializes for an fp32 model (71,885,789 x 8 B = 0.54 GiB), so the static
# footprint this predicts is 2.22 - 0.54 = 1.68 GiB. The constants give 20 B x 22.93M + 20 B x
# 71.89M = 1.77 GiB, i.e. 5 % high -- the conservative direction, since over-estimating the
# statics can only make the derived clade budget smaller.
_STATIC_BYTES_PER_CLADE = 20
_STATIC_BYTES_PER_SPLIT = 20

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
    per_clade_b = (
        _positive_int("scratch_tensors", scratch_tensors)
        * _positive_int("S", S)
        * dtype_nbytes(dtype)
    )
    affordable = usable_b // per_clade_b
    if affordable <= 0:
        raise ValueError(
            f"after the resident statics ({static_b / GIB:.2f} GiB) this device's memory budget "
            f"({budget_b / GIB:.2f} GiB, {allocatable_b / GIB:.2f} GiB allocatable) leaves room "
            f"for fewer than one clade per batch at {S} species; this fit does not fit this card"
        )
    chosen = int(min(fixed, affordable))
    detail["automatic"] = chosen < fixed
    detail["allocatable_bytes"] = allocatable_b
    detail["affordable_clade_budget"] = int(affordable)
    detail["working_set_bytes"] = batch_working_set_bytes(
        chosen, S, dtype, scratch_tensors=scratch_tensors)
    detail["predicted_peak_bytes"] = static_b + detail["working_set_bytes"]
    return chosen, detail
