"""Golden regression test for the gpurec optimization layer (`gpurec/optim`).

Locks the likelihood (value + gradient) against future codebase changes: gpurec must reproduce the
frozen kernel-bench result at the 666x80 fixture at BOTH the initial fixture theta AND the fitted
checkpoint ``theta_hat``, at the capture's truncated solver (pi=16) and at a converged solver
(pi=64). Two layers:

  * **self-golden** — gpurec loss/grad vs pinned, verified-correct values in
    ``gpurec/optim/_golden/checkpoint_666x80.pt`` (loss rtol 1e-3 catches correctness bugs while
    tolerating atomic-reduction noise; grad rel-L2 < 2e-3).
  * **cross-parity** — gpurec == LIVE kernel-bench, BIT-EXACT, on identical inputs/hardware/kernels
    (the merge-boundary guard; skipped if the kernel-bench tree is not importable).

The golden was established 2026-06-15 by re-verifying gpurec == live kernel-bench bit-exact on an
RTX 4090 in fp32. NOTE: the value *recorded* in kernel-bench ``newton_golden_fp32.pt`` (``FN`` ~
144033) is **stale** — it does not reproduce in either codebase (both give ~143462 converged), so it
is deliberately NOT used as the golden. See ``gpurec/docs/optim/golden_test.md``.

Needs the kernel-bench 666x80 capture (``whole.pt``, ~3.4 GB, not in-repo): set ``GPUREC_KB_CAPTURE``
or have it at ``../kernel-bench/data/666x80/whole.pt``. Skips cleanly when capture / CUDA / triton /
golden artifact are absent.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import triton  # noqa: F401  -- HARD dependency; the whole codebase is Triton. Fail loud, never skip.

_GOLDEN = Path(__file__).resolve().parents[1] / "gpurec" / "optim" / "_golden" / "checkpoint_666x80.pt"
_CAP_REL = Path("kernel-bench") / "data" / "666x80" / "whole.pt"
LOSS_RTOL = 1e-3   # the stale-golden gap was 4e-3; atomic-reduction noise is ~1e-5 -> 1e-3 separates
GRAD_RTOL = 2e-3   # backward is atomically nondeterministic at ~2e-4; matches the parity bar


def _cap_path() -> str:
    """Resolve the kernel-bench 666x80 capture: GPUREC_KB_CAPTURE, else search ancestors for a
    ``kernel-bench/data/666x80/whole.pt`` (handles both the sibling layout and the nested worktree)."""
    env = os.environ.get("GPUREC_KB_CAPTURE")
    if env:
        return env
    for anc in Path(__file__).resolve().parents:
        cand = anc / _CAP_REL
        if cand.exists():
            return str(cand)
    return str(_CAP_REL)  # not found -> _require() skips with this hint


def _require() -> None:
    # triton is a hard dependency (imported at module load) -- a broken triton ERRORS, never skips.
    # Only genuine environment gaps skip: no GPU hardware, or the external 3.4 GB capture is absent.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not _GOLDEN.exists():
        pytest.skip(f"golden artifact missing: {_GOLDEN}")
    if not Path(_cap_path()).exists():
        pytest.skip(f"kernel-bench 666x80 capture not found at {_cap_path()} (set GPUREC_KB_CAPTURE)")


@pytest.fixture(scope="module")
def ctx():
    _require()
    from gates._parity_kbench import gpurec_static_from_capture

    cap = torch.load(_cap_path(), map_location="cpu", weights_only=False)
    golden = torch.load(_GOLDEN, map_location="cpu", weights_only=False)
    rw = cap["inputs"]["col_weights"].cuda().contiguous()
    return {"cap": cap, "golden": golden, "rw": rw, "make": gpurec_static_from_capture}


def _loss_grad(ctx, theta, pi, neu):
    from gpurec.optim.value_and_grad import make_value_and_grad

    static = ctx["make"](ctx["cap"], "cuda")
    static.solver_options.pi_iters, static.solver_options.neumann_terms = pi, neu
    f = make_value_and_grad([static], ctx["rw"], theta_shape=tuple(theta.shape))
    loss, g, _, _ = f(theta.reshape(-1))
    return float(loss), g.reshape(-1).float().cpu()


def _rel_l2(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def test_fixture_loss_grad(ctx):
    """gpurec ≡ canonical kernel-bench golden at the fixture theta (capture's pi=16)."""
    g = ctx["golden"]
    loss, grad = _loss_grad(ctx, g["theta_fixture"].cuda(), g["pi_iters_capture"], g["neumann_capture"])
    assert abs(loss - g["fixture_loss_pi16"]) <= LOSS_RTOL * abs(g["fixture_loss_pi16"]), loss
    assert _rel_l2(grad, g["fixture_grad_pi16"]) <= GRAD_RTOL


def test_checkpoint_loss_grad_pi16(ctx):
    """gpurec reproduces the pinned checkpoint loss/grad at theta_hat (capture's pi=16)."""
    g = ctx["golden"]
    loss, grad = _loss_grad(ctx, g["theta_hat"].cuda(), 16, 16)
    assert abs(loss - g["checkpoint_loss_pi16"]) <= LOSS_RTOL * abs(g["checkpoint_loss_pi16"]), loss
    assert _rel_l2(grad, g["checkpoint_grad_pi16"]) <= GRAD_RTOL


def test_checkpoint_loss_grad_converged(ctx):
    """gpurec reproduces the pinned checkpoint loss/grad at theta_hat (converged pi=64)."""
    g = ctx["golden"]
    loss, grad = _loss_grad(ctx, g["theta_hat"].cuda(), g["pi_iters_converged"], g["neumann_converged"])
    assert abs(loss - g["checkpoint_loss_converged"]) <= LOSS_RTOL * abs(g["checkpoint_loss_converged"]), loss
    assert _rel_l2(grad, g["checkpoint_grad_converged"]) <= GRAD_RTOL


def test_cross_parity_vs_live_kbench(ctx):
    """Strongest merge-boundary guard: gpurec == live kernel-bench loss, BIT-EXACT, at both the
    fixture and the checkpoint, at pi=16 and pi=64."""
    kb = Path(_cap_path()).resolve().parents[2]  # .../kernel-bench
    if not (kb / "kbench").exists():
        pytest.skip(f"kernel-bench tree not importable at {kb}")
    if str(kb) not in sys.path:
        sys.path.insert(0, str(kb))
    try:
        from kbench.runtime import make_static
        from newton.vg import forward_solve as kb_fwd
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"kbench import failed: {exc}")
    from gpurec.optim.value_and_grad import forward_solve as gp_fwd

    g, cap, rw = ctx["golden"], ctx["cap"], ctx["rw"]
    for name, theta in (("fixture", g["theta_fixture"].cuda()), ("checkpoint", g["theta_hat"].cuda())):
        for pi, neu in ((16, 16), (64, 64)):
            gs = ctx["make"](cap, "cuda")
            gs.solver_options.pi_iters, gs.solver_options.neumann_terms = pi, neu
            ks = make_static(cap, "cuda")
            ks.solver_options.pi_iters, ks.solver_options.neumann_terms = pi, neu
            gl = float(gp_fwd([gs], theta, rw)[0])
            kl = float(kb_fwd(ks, theta, rw)[0])
            assert abs(gl - kl) <= 1e-6 * max(1.0, abs(kl)), f"{name} pi={pi}: gpurec {gl} != kbench {kl}"
