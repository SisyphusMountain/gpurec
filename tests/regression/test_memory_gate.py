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
def test_hvp_path_threads_reserved_scratch_bytes_when_warm_active(monkeypatch, tmp_path):
    """Task C follow-up: the memory-gate fix (97f00559) threaded ``reserved_scratch_bytes``
    through the GRADIENT path (``_execution.py`` -> ``implicit_grad_loglik_vjp_wave``) but
    missed the HVP path -- ``build_point_cache`` (via ``vjp_root_to_theta``) and the
    tangent-adjoint solve in ``make_exact_hvp_single`` both called ``wave_backward_uniform_fused``
    without it, so a resident warm-adjoint cache could still trip the false ``budget 0.00 GiB``
    rejection there. Regression guard: spy on both call sites and assert the reservation is
    threaded exactly like the gradient path -- ``static.warm_scratch_reserved_bytes`` when
    warm-adjoint is active for this static (env set + ``warm_adjoint_ok``), else ``None``.

    Tiny fixture (12 species / 4 families, a couple of HVP evals) -- fast, no 500-leaf hardware.
    """
    rustree = pytest.importorskip("rustree")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import tempfile

    import gpurec.api._implicit_grad as _implicit_grad_mod
    import gpurec.solver.hvp_exact as _hvp_mod
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.bench.simulate import simulate_dataset
    from gpurec.solver.hvp_exact import build_point_cache, make_exact_hvp_single
    from gpurec.solver.value_and_grad import forward_solve

    so = SolverOptions(neumann_terms=16, pi_iters=32, self_loop_solver="neumann",
                        bicgstab_max_iter=200, bicgstab_tol=1e-8, bicgstab_breakdown_tol=1e-30,
                        adjoint_pruning_threshold=1e-6, use_adjoint_pruning=True,
                        pibar_side_threshold=0.0)
    so.validate()
    d = tempfile.mkdtemp(dir=tmp_path)
    sp, genes = simulate_dataset("specieswise", d, n_species=12, n_families=4, dtl=0.05, seed=7)
    model = GeneReconModel(sp, genes, mode="specieswise", device="cuda", dtype=torch.float64,
                           solver_options=so, family_chunk_size=None)
    assert len(model.batch_statics) == 1
    st = model.batch_statics[0]
    theta = model.theta.detach().clone()
    rw = model.receiver_weights.detach().clone()

    real_impl_fn = _implicit_grad_mod.wave_backward_uniform_fused
    real_hvp_fn = _hvp_mod.wave_backward_uniform_fused
    captured = {"implicit_grad": [], "hvp_exact": []}

    def spy_impl(*a, **kw):
        captured["implicit_grad"].append(kw.get("reserved_scratch_bytes"))
        return real_impl_fn(*a, **kw)

    def spy_hvp(*a, **kw):
        captured["hvp_exact"].append(kw.get("reserved_scratch_bytes"))
        return real_hvp_fn(*a, **kw)

    monkeypatch.setattr(_implicit_grad_mod, "wave_backward_uniform_fused", spy_impl)
    monkeypatch.setattr(_hvp_mod, "wave_backward_uniform_fused", spy_hvp)

    _loss, sv = forward_solve([st], theta, rw)

    # --- warm-adjoint ACTIVE for this static: reservation must be threaded (not None). ---
    monkeypatch.setenv("GPUREC_WARM_ADJOINT", "1")
    st.warm_adjoint_ok = True
    sentinel_reserved = 123_456_789
    st.warm_scratch_reserved_bytes = sentinel_reserved

    _gt, _gc, cache = build_point_cache(st, theta, rw, sv)
    assert captured["implicit_grad"], "build_point_cache never reached the gated fast path"
    assert all(v == sentinel_reserved for v in captured["implicit_grad"]), captured["implicit_grad"]

    hvp = make_exact_hvp_single(st, theta, rw, sv, cache=cache, tangent_self_iters=8)
    torch.manual_seed(0)
    u = torch.randn(theta.numel(), device="cuda", dtype=theta.dtype)
    hvp(u)
    assert captured["hvp_exact"], "hvp() never reached the gated fast path"
    assert all(v == sentinel_reserved for v in captured["hvp_exact"]), captured["hvp_exact"]

    # --- cold path unchanged: warm-adjoint inactive -> reserved_scratch_bytes stays None. ---
    captured["implicit_grad"].clear()
    captured["hvp_exact"].clear()
    monkeypatch.delenv("GPUREC_WARM_ADJOINT", raising=False)

    _gt2, _gc2, cache2 = build_point_cache(st, theta, rw, sv)
    assert captured["implicit_grad"]
    assert all(v is None for v in captured["implicit_grad"]), captured["implicit_grad"]

    hvp2 = make_exact_hvp_single(st, theta, rw, sv, cache=cache2, tangent_self_iters=8)
    hvp2(torch.randn(theta.numel(), device="cuda", dtype=theta.dtype))
    assert captured["hvp_exact"]
    assert all(v is None for v in captured["hvp_exact"]), captured["hvp_exact"]


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
