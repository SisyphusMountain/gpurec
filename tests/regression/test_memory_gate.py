"""Forward self-loop memory-budget accounting."""

import torch

from gpurec.core import memory_policy
from gpurec.core.memory_policy import (
    proposal0_memory_gate,
    proposal0_wave_scratch_bytes,
)

DTYPE = torch.float32


def test_cold_path_rejects_when_current_budget_too_small(monkeypatch):
    """reserved_scratch_bytes=None (cold): the gate reads current free memory. A tiny budget rejects
    -- this is the exact double-gate condition that falsely fired with the resident warm cache."""
    W, S = 8192, 999
    scratch = proposal0_wave_scratch_bytes(W, S, DTYPE)
    # Emulate the depleted post-cache free memory: budget ~ 0.
    monkeypatch.setattr(memory_policy, "cuda_memory_budget_bytes", lambda device=None: 0)
    ok, required, budget = proposal0_memory_gate(W, S, DTYPE, reserved_scratch_bytes=None)
    assert not ok
    assert required == scratch
    assert budget == 0


def test_reserved_path_trusts_build_reservation_over_depleted_budget(monkeypatch):
    """The fix: when the warm-adjoint fit already reserved >= this wave's scratch, the gate PASSES
    even though the re-read current budget is ~0 (it must not consult the depleted free memory)."""
    W, S = 8192, 999
    scratch = proposal0_wave_scratch_bytes(W, S, DTYPE)
    reserved = scratch  # build reserved exactly (or more) than this wave needs

    # If the gate wrongly re-read the budget it would reject; make that failure loud if it happens.
    def _boom(device=None):
        raise AssertionError("gate must NOT read cuda_memory_budget_bytes on the reserved path")

    monkeypatch.setattr(memory_policy, "cuda_memory_budget_bytes", _boom)
    ok, required, budget = proposal0_memory_gate(W, S, DTYPE, reserved_scratch_bytes=reserved)
    assert ok
    assert budget == reserved


def test_reserved_path_still_rejects_genuinely_oversized_scratch():
    """Safety preserved: a wave scratch that GENUINELY exceeds the reserved headroom is rejected --
    the fix corrects the accounting, it does not disable the gate."""
    W, S = 8192, 999
    scratch = proposal0_wave_scratch_bytes(W, S, DTYPE)
    reserved = scratch - 1  # one byte short of what this wave needs
    ok, required, budget = proposal0_memory_gate(W, S, DTYPE, reserved_scratch_bytes=reserved)
    assert not ok
    assert required == scratch
    assert budget == reserved


def test_budget_counts_reclaimable_cache(monkeypatch):
    """cuda_memory_budget_bytes must credit PyTorch's reclaimable caching-allocator pool
    (reserved - allocated) as available. mem_get_info's driver free_b counts that reusable pool as
    'used', so mid-run (e.g. the fp64 final_eval backward holds a large reserved pool) the raw
    free_b collapses to ~0 and the gate falsely rejects a scratch that the allocator could serve
    from its own pool. The budget must be free_b + (reserved - allocated) - reserve, capped by
    total*fraction. Regression for the 500-leaf final_eval 'budget 0.09 GiB' false rejection."""
    GIB = 1024 ** 3
    total = 24 * GIB
    free_b = 1 * GIB          # driver sees only 1 GiB free...
    reserved = 18 * GIB       # ...but torch holds an 18 GiB pool,
    allocated = 8 * GIB       # 8 GiB live -> 10 GiB reclaimable.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda idx=0: (free_b, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx=0: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda idx=0: allocated)
    # fraction=0.85 cap = 20.4 GiB (not binding); reserve=1 GiB.
    budget = memory_policy.cuda_memory_budget_bytes(0, default_fraction=0.85, default_reserve_gib=1.0)
    reclaimable = reserved - allocated
    expected = min(int(total * 0.85), (free_b + reclaimable) - 1 * GIB)
    assert budget == expected
    # Concretely: 1 + 10 - 1 = 10 GiB available, far above a fresh raw-free budget of ~0.
    assert budget == 10 * GIB
    # And a genuinely tight state (no reclaimable pool, low free) still yields a near-zero budget.
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx=0: allocated)  # reclaimable = 0
    tight = memory_policy.cuda_memory_budget_bytes(0, default_fraction=0.85, default_reserve_gib=1.0)
    assert tight == 0  # free_b(1) - reserve(1) = 0, no pool to credit -> genuine OOM still caught
