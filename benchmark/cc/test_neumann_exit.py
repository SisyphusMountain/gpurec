"""Validate the fused Neumann-series self-loop kernel against the per-term reference loop.

The wave adjoint used to launch ``_apply_reconciliation_self_loop_transpose_kernel`` once per
Neumann term (16 in the fit tier, 64 in the certificate tier). It now runs the whole series
inside one launch of ``_reconciliation_self_loop_transpose_series_kernel``, with a per-row-block
early exit once the block's largest remaining term is at or below
``neumann_term_tol * (block max |adjoint|)``.

This script reports, in one job:
  A  tol = 0, fused vs the reference loop: are the gradients bit-identical, and what is the
     run-to-run noise of two identical reference calls (the backward accumulates with atomics)?
  B  tol from --tol (production 1e-7) vs the reference loop: largest gradient difference.
  C  wall time of loss+gradient (mean of 3 warm calls) and of one 3-probe analytic Hessian,
     reference vs fused.
  D  mean number of Neumann terms each row block actually ran before its early exit.

Usage:
  python benchmark/cc/test_neumann_exit.py --species S.nwk --families LIST.txt \
      --limit 100 --clade-budget 315000 --timing-limit 500 --tol 1e-7
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def _build(species, paths, clade_budget, neumann_terms, pi_iters, tol):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    so = SolverOptions(**{
        **_BASE_SOLVER,
        "pi_iters": pi_iters,
        "neumann_terms": neumann_terms,
        "neumann_term_tol": tol,
    })
    m = GeneReconModel(species, paths, mode="genewise", device="cuda", dtype=torch.float32,
                       solver_options=so, clade_budget=clade_budget)
    m.receiver_weights.requires_grad_(False)
    return m


def _grad(m, theta):
    out = m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    grad = out[1]
    torch.cuda.synchronize()
    return grad.detach().clone()


def _set_tol(m, tol):
    # batch_statics share the model's SolverOptions object, so one setattr reaches all.
    m.configure_solver(neumann_term_tol=float(tol))


# The only arguments the self-loop solve writes into are these accumulators; every
# other argument is read-only, so a replay only has to give each run its own copy
# of these (plus its own copy of rhs, which later waves keep adding into).
_MUTATED_KWARGS = ("grad_receiver_log_probs", "self_loop_grad_targets")
_RHS_POSITION = 6


def _copy_of(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, (tuple, list)):
        return type(x)(_copy_of(v) for v in x)
    return x


def _capture_largest_wave(m, theta, wb):
    """Record the arguments of the biggest single wave self-loop solve of one gradient pass."""
    orig = wb._solve_reconciliation_wave_vjp_2d
    best = {}

    def spy(*a, **k):
        W = int(k["W"]) if "W" in k else int(a[3])
        if W > best.get("W", -1):
            args = list(a)
            args[_RHS_POSITION] = _copy_of(args[_RHS_POSITION])
            best.clear()
            best.update(W=W, args=tuple(args), kwargs=dict(k))
        return orig(*a, **k)

    wb._solve_reconciliation_wave_vjp_2d = spy
    try:
        m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        torch.cuda.synchronize()
    finally:
        wb._solve_reconciliation_wave_vjp_2d = orig
    return orig, best


def _replay(orig, cap, wb, fused, tol):
    wb.set_fused_neumann_series(fused)
    k = dict(cap["kwargs"])
    for name in _MUTATED_KWARGS:
        if k.get(name) is not None:
            k[name] = _copy_of(k[name])
    k["neumann_term_tol"] = float(tol)
    out = orig(*cap["args"], **k)
    torch.cuda.synchronize()
    return [t.clone() for t in out if torch.is_tensor(t)]


def _time_grad(m, theta, reps):
    m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int, help="families for the correctness checks")
    ap.add_argument("--clade-budget", required=True, type=int)
    ap.add_argument("--timing-limit", required=True, type=int, help="families for the timing checks")
    ap.add_argument("--tol", required=True, type=float, help="neumann_term_tol under test")
    ap.add_argument("--neumann-terms", required=True, type=int)
    ap.add_argument("--pi-iters", required=True, type=int)
    args = ap.parse_args()

    all_paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    paths = all_paths[: args.limit]

    from gpurec.core.kernels import wave_backward as wb

    print(f"[env] torch {torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[env] families={len(paths)} clade_budget={args.clade_budget} "
          f"neumann_terms={args.neumann_terms} pi_iters={args.pi_iters} tol={args.tol}", flush=True)

    # ---------------------------------------------------------------- A and B
    t0 = time.perf_counter()
    m = _build(args.species, paths, args.clade_budget, args.neumann_terms, args.pi_iters, 0.0)
    theta = torch.full((len(paths), 3), -6.0, device="cuda")
    print(f"[A] build {time.perf_counter() - t0:.1f}s  batches={len(m.batch_statics)}", flush=True)

    wb.set_fused_neumann_series(False)
    ref1 = _grad(m, theta)
    ref2 = _grad(m, theta)
    noise = (ref1 - ref2).abs().max().item()
    print(f"[A] reference vs reference (run-to-run noise): equal={torch.equal(ref1, ref2)} "
          f"max|diff|={noise:.3e}", flush=True)

    wb.set_fused_neumann_series(True)
    _set_tol(m, 0.0)
    fused0 = _grad(m, theta)
    eq0 = torch.equal(ref1, fused0)
    d0 = (ref1 - fused0).abs().max().item()
    print(f"[A] fused(tol=0) vs reference: bitwise_equal={eq0} max|diff|={d0:.3e}", flush=True)

    # Bitwise check on the self-loop solve itself. The FULL gradient cannot be
    # compared bit for bit because its scatter-accumulation uses atomics (see the
    # noise line above), so replay one captured wave -- identical inputs, only the
    # Neumann loop differs -- and compare its outputs exactly.
    wb.set_fused_neumann_series(False)
    orig, cap = _capture_largest_wave(m, theta, wb)
    if cap:
        ref_out = _replay(orig, cap, wb, False, 0.0)
        ref_out2 = _replay(orig, cap, wb, False, 0.0)
        f0_out = _replay(orig, cap, wb, True, 0.0)
        ftol_out = _replay(orig, cap, wb, True, args.tol)
        same_ref = all(torch.equal(x, y) for x, y in zip(ref_out, ref_out2))
        same_f0 = all(torch.equal(x, y) for x, y in zip(ref_out, f0_out))
        dmax = max((x - y).abs().max().item() for x, y in zip(ref_out, ftol_out))
        vmax = ref_out[0].abs().max().item()
        print(f"[A] wave replay (W={cap['W']}, {len(ref_out)} output tensors): "
              f"reference reproducible={same_ref}  fused(tol=0) bitwise_equal={same_f0}", flush=True)
        print(f"[B] wave replay fused(tol={args.tol}) vs reference: max|diff|={dmax:.3e} "
              f"over max|adjoint|={vmax:.3e} (rel={dmax / max(vmax, 1e-30):.3e})", flush=True)
    wb.set_fused_neumann_series(True)

    _set_tol(m, args.tol)
    fused_tol = _grad(m, theta)
    dt = (ref1 - fused_tol).abs().max().item()
    denom = ref1.abs().max().item()
    print(f"[B] fused(tol={args.tol}) vs reference: max|diff|={dt:.3e} "
          f"(noise={noise:.3e}, max|grad|={denom:.3e}, rel={dt / max(denom, 1e-30):.3e})", flush=True)

    # ------------------------------------------------------------------- D
    wb.set_neumann_term_stat_collection(True)
    _grad(m, theta)
    counts = wb.take_neumann_term_counts()
    wb.set_neumann_term_stat_collection(False)
    if counts:
        flat = torch.cat([c.reshape(-1) for c in counts]).float()
        print(f"[D] Neumann terms per row block at tol={args.tol}: "
              f"mean={flat.mean().item():.2f} min={int(flat.min())} max={int(flat.max())} "
              f"budget={args.neumann_terms} row_blocks={flat.numel()}", flush=True)
        hist = torch.bincount(flat.int(), minlength=args.neumann_terms + 1)
        print(f"[D] histogram (index = terms taken): {hist.tolist()}", flush=True)
    else:
        print("[D] no term counts collected", flush=True)

    del m, theta, ref1, ref2, fused0, fused_tol
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------- C
    tpaths = all_paths[: args.timing_limit]
    t0 = time.perf_counter()
    mt = _build(args.species, tpaths, args.clade_budget, args.neumann_terms, args.pi_iters, args.tol)
    theta_t = torch.full((len(tpaths), 3), -6.0, device="cuda")
    print(f"[C] build {time.perf_counter() - t0:.1f}s  families={len(tpaths)} "
          f"batches={len(mt.batch_statics)}", flush=True)

    from gpurec.fit.genewise_fit import _analytic_hessian

    hessians = {}
    for label, fused in (("reference", False), ("reference2", False), ("fused", True)):
        wb.set_fused_neumann_series(fused)
        _set_tol(mt, args.tol if fused else 0.0)
        times = _time_grad(mt, theta_t, 3)
        t = time.perf_counter()
        H = _analytic_hessian(mt, theta_t, 16)
        torch.cuda.synchronize()
        hess_s = time.perf_counter() - t
        hessians[label] = H
        print(f"[C] {label:10s}: loss+grad mean of 3 = {statistics.mean(times):.3f}s "
              f"(each {[round(x, 3) for x in times]}), analytic Hessian(16) = {hess_s:.2f}s, "
              f"H trace = {H.diagonal(dim1=1, dim2=2).sum().item():.6e}", flush=True)
    # The Hessian is built from the same atomically-accumulated gradients, so it has
    # its own run-to-run noise; quote the fused-vs-reference gap against it.
    hnoise = (hessians["reference"] - hessians["reference2"]).abs().max().item()
    hdiff = (hessians["reference"] - hessians["fused"]).abs().max().item()
    hscale = hessians["reference"].abs().max().item()
    print(f"[C] Hessian entries: reference run-to-run noise max|diff|={hnoise:.3e}, "
          f"fused vs reference max|diff|={hdiff:.3e}, max|H|={hscale:.3e}", flush=True)

    wb.set_fused_neumann_series(True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
