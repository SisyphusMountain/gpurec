"""How many gradients does one exact per-family Hessian cost?

The genewise Newton fit refreshes its curvature with ``_analytic_hessian`` (gpurec/fit/
genewise_fit.py): per batch it runs one forward solve, builds the adjoint point cache once, and
then fires three Hessian-vector probes (the three unit rate directions D, L, T). This script times
that whole refresh against one plain gradient over the same families and prints the ratio, which is
the number the recipe sweep trades off ("exact Newton every iteration costs N Hessians").

Three measurements, because a shared GPU makes some of them unreliable on their own:

* WALL: gradient and Hessian timed alternately, one sample of each per round, so a neighbour's load
  lands on both sides of the ratio rather than only one. Reported as medians.
* GPU KERNEL TIME (torch.profiler, CUDA activities): both sides are profiled inside ONE session,
  back to back, which puts them in the same contention window. GPU time is NOT contention-proof on
  its own -- sharing the streaming multiprocessors slows every kernel -- so never compare a GPU
  time from one run against a GPU time from another run minutes later.
* LAUNCH COUNT: how many CUDA kernels each side asks for. Exact, and unaffected by anything else on
  the card, so this is the number to trust when comparing code versions.

The Hessian side is measured on the FIRST batch only (one forward solve + one point cache + three
probes) and the gradient side over every batch, then divided by the batch count so the two are
directly comparable.

With ``--save-hessian PATH`` the [G,3,3] curvature is written out, so a before/after pair of runs
can be compared entry by entry. Note that this Hessian is NOT bit-reproducible: the reduction
kernels accumulate with float32 atomics in whatever order the card schedules them, so two runs of
the SAME code on the 200-family Coleman set differ by up to ~1.3e-1 in absolute value (~5e-3 rms)
on entries whose mean magnitude is 8.6. A before/after difference inside that band is noise, not a
change of value.

Usage: hessian_cost.py --species S --families LIST --limit 200 --clade-budget 100000
                       --pi-iters 16 --repeats 3 [--save-hessian out.pt]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def _launches(fn):
    """CUDA kernel launches in one call to ``fn`` (exact; a busy card does not change it)."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0)


def _gpu_ms_same_window(named_fns):
    """GPU time (ms) of each ``(name, fn)``, all measured inside ONE profiler session."""
    for _name, fn in named_fns:
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for name, fn in named_fns:
            with record_function(name):
                fn()
            torch.cuda.synchronize()
    rows = {e.key: e for e in prof.key_averages()}
    return {name: rows[name].device_time_total / 1e3 for name, _fn in named_fns}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int)
    ap.add_argument("--clade-budget", required=True, type=int)
    ap.add_argument("--pi-iters", required=True, type=int,
                    help="tangent self-loop iteration count (the fit passes its current pi tier)")
    ap.add_argument("--repeats", required=True, type=int,
                    help="how many alternating gradient/Hessian rounds to take the median of")
    ap.add_argument("--save-hessian", required=False, default=None,
                    help="write the [G,3,3] curvature here for a before/after comparison")
    args = ap.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER, _analytic_hessian
    from gpurec.solver.hvp.exact import make_exact_hvp_single
    from gpurec.solver.value_and_grad import forward_solve

    paths = [ln.strip() for ln in open(args.families)
             if ln.strip() and not ln.startswith("#")][: args.limit]
    so = SolverOptions(**{**_BASE_SOLVER, "forward_self_loop": "exact", "adjoint_self_loop": "exact"})
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
                       solver_options=so, clade_budget=args.clade_budget)
    m.receiver_weights.requires_grad_(False)
    rw = m.receiver_weights.detach()
    theta = torch.tensor([-6.0, -3.0, -6.0], dtype=torch.float32,
                         device="cuda").repeat(len(paths), 1).contiguous()
    n_batches = len(m.batch_statics)
    print(f"[cost] {len(paths)} families, {n_batches} batches, clade budget {args.clade_budget}, "
          f"pi_iters {args.pi_iters}", flush=True)

    def grad():
        m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)

    def hess():
        return _analytic_hessian(m, theta, args.pi_iters, args.species, paths)

    static = m.batch_statics[0]
    fam = static.family_index_tensor.to(theta.device)
    theta_b = theta.index_select(0, fam).contiguous()

    def one_batch_hessian():
        """What _analytic_hessian does for ONE batch: forward solve, point cache, three probes."""
        _l, sv = forward_solve([static], theta, rw)
        hvp = make_exact_hvp_single(static, theta_b, rw, sv, tangent_self_iters=args.pi_iters)
        for j in range(3):
            u = torch.zeros(int(fam.numel()), 3, device=theta.device, dtype=theta.dtype)
            u[:, j] = 1.0
            hvp(u.reshape(-1), probe_id=j)

    # --- wall, alternating so a busy card loads both sides equally ---
    grad(); hess(); torch.cuda.synchronize()
    grad_samples, hess_samples = [], []
    for _ in range(args.repeats):
        for fn, out in ((grad, grad_samples), (hess, hess_samples)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append(time.perf_counter() - t0)
    g_wall = statistics.median(grad_samples)
    h_wall = statistics.median(hess_samples)
    print(f"[cost] WALL   one gradient {g_wall*1e3:.0f} ms, one 3-probe Hessian {h_wall*1e3:.0f} ms"
          f"  ->  Hessian = {h_wall/g_wall:.2f} gradients "
          f"(median of {args.repeats} alternating rounds)", flush=True)

    ms = _gpu_ms_same_window([("gradient_all_batches", grad),
                              ("hessian_one_batch", one_batch_hessian)])
    g_ms = ms["gradient_all_batches"] / n_batches
    h_ms = ms["hessian_one_batch"]
    print(f"[cost] GPU    one gradient {g_ms:.1f} ms per batch, one 3-probe Hessian {h_ms:.1f} ms "
          f"per batch  ->  Hessian = {h_ms/g_ms:.2f} gradients (both in one profiler window)",
          flush=True)

    g_launches = _launches(grad) / n_batches
    h_launches = _launches(one_batch_hessian)
    print(f"[cost] LAUNCH one gradient {g_launches:.0f} kernels per batch, one 3-probe Hessian "
          f"{h_launches} kernels per batch  ->  Hessian = {h_launches/g_launches:.2f} gradients",
          flush=True)

    if args.save_hessian is not None:
        H = hess()
        torch.save(H.cpu(), args.save_hessian)
        print(f"[cost] wrote [{tuple(H.shape)}] curvature to {args.save_hessian}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
