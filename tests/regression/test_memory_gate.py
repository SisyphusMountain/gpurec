"""Forward self-loop memory-gate accounting (Task C).

Regression guard for the ``budget 0.00 GiB`` false-rejection: with warm-adjoint ON, the model's
build-time ``warm_adjoint_fits`` decision already reserves transient per-wave scratch headroom
(``cache + max_batch_scratch <= budget`` BEFORE the ~GiB warm cache is allocated). The forward
``proposal0_memory_gate`` used to RE-read ``cuda_memory_budget_bytes`` -- now depleted by the
resident cache -- and falsely reject a per-wave scratch the build already accounted for. The fix
threads that reservation as ``reserved_scratch_bytes`` so the gate validates the wave scratch against
the reservation (a scratch that GENUINELY exceeds it is still rejected) instead of the moving budget.

These tests are synthetic (W/S/budget arithmetic) -- fast, no CUDA / 500-leaf hardware needed. The
@slow 500-leaf end-to-end lives in ``test_memory_gate_e2e`` below.
"""
import math

import pytest

import torch

from gpurec.core import memory_policy
from gpurec.core.memory_policy import (
    proposal0_memory_gate,
    proposal0_wave_scratch_bytes,
    warm_adjoint_fits,
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


def test_wave_scratch_is_subset_of_warm_adjoint_reservation(monkeypatch):
    """Invariant the fix relies on: a single wave's W is a subset of the batch's clades, so its
    scratch never exceeds the ``max_batch_clades`` scratch that ``warm_adjoint_fits`` reserved.
    Thus, whenever warm fit at build, the reserved-path gate provably passes every wave."""
    S = 999
    max_batch_clades = 314_998            # e.g. clade_budget cap
    # Pretend a comfortable build-time budget so warm 'fits'.
    reserved = proposal0_wave_scratch_bytes(max_batch_clades, S, DTYPE)
    monkeypatch.setattr(
        memory_policy, "cuda_memory_budget_bytes", lambda device=None: reserved * 100
    )
    ok_build, cache, scratch_reserved, budget = warm_adjoint_fits(
        1_000_000, S, DTYPE, max_batch_clades=max_batch_clades
    )
    assert ok_build
    assert scratch_reserved == reserved
    # Every physically-realizable wave (W <= max_batch_clades) passes the reserved-path gate, even
    # when the (mocked) current budget is ~0, whereas the cold path would reject it.
    monkeypatch.setattr(memory_policy, "cuda_memory_budget_bytes", lambda device=None: 0)
    for W in (1, 4096, 8192, max_batch_clades):
        ok_reserved, _, _ = proposal0_memory_gate(
            W, S, DTYPE, reserved_scratch_bytes=scratch_reserved
        )
        assert ok_reserved, W
        ok_cold, _, _ = proposal0_memory_gate(W, S, DTYPE, reserved_scratch_bytes=None)
        assert not ok_cold, W  # cold path re-reads the depleted budget and (wrongly, pre-fix) rejects


@pytest.mark.gpu
@pytest.mark.slow
def test_500_leaf_warm_adjoint_forward_gate_e2e(tmp_path):
    """500 extant leaves (S=999), warm-adjoint ON, moderate family count so warm FITS at build:
    the model must build + run a short fit WITHOUT the forward self-loop 'budget 0.00 GiB' false
    rejection, with warm-adjoint staying ON and peak memory within the GPU."""
    rustree = pytest.importorskip("rustree")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import os

    from gpurec.bench.simulate import simulate_dataset
    from gpurec.api.model import GeneReconModel
    from gpurec.fit.optimize import optimize

    os.environ["GPUREC_WARM_ADJOINT"] = "1"
    sp, genes = simulate_dataset("global", tmp_path, n_species=500, n_families=700,
                                 dtl=0.05, seed=20260709)
    model = GeneReconModel(sp, genes, mode="global", device="cuda", dtype=torch.float32,
                           clade_budget=40_000)
    if not model.warm_adjoint_ok:
        pytest.skip("warm-adjoint did not fit at build on this GPU (cold regime, not the target)")
    model.configure_solver(e_adjoint_solver="neumann")
    torch.cuda.reset_peak_memory_stats()
    theta_hat, hist = optimize(model.batch_statics, model.theta.detach(),
                               model.receiver_weights.detach(),
                               max_steps=5, polish_mode="none", verbose=False)
    assert model.warm_adjoint_ok
    assert math.isfinite(float(hist[-1]["loss"]))
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    _, total = torch.cuda.mem_get_info()
    assert peak_gib < total / (1024 ** 3)
