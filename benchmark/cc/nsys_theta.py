"""Why is one genewise gradient slower at the FITTED theta than at a flat theta?

Runs ONE loss+gradient at the fitted per-family rates (loaded from a ``run_genewise`` ``.pt``, matched
to the family list by path) and ONE at a flat theta of -6, in the same process, with the same solver
settings as the production recipe's first tier (pi_iters=16, neumann_terms=16, cold adjoint --
``GPUREC_WARM_ADJOINT`` removed, receiver weights frozen), and reports where the difference goes.

Two modes, because the measurements interfere with each other:

  --mode nsys   Both gradients run untouched, each inside its own NVTX range ("grad_fitted",
                "grad_flat") between cudaProfilerStart/Stop, for an Nsight Systems kernel breakdown.
                Both thetas are warmed up BEFORE the capture so no Triton module loading lands
                inside a range.

  --mode count  Counts the iterations of the two fixed-point loops that carry a device->host sync per
                iteration, by monkey-patching (no library file is modified):
                  * forward E solve -- ``gpurec.core.kernels.e_step.e_fixed_point_triton``'s loop,
                    counted through its per-iteration ``_launch_e_step_forward_2d`` call;
                  * E-adjoint Neumann series -- ``gpurec.api._implicit_grad._neumann_e_adjoint``,
                    counted through its per-iteration ``Av`` call.
                Both are capped at ``e_max_iter`` / ``e_adjoint_max_iter`` (128), and each E solve
                covers a WHOLE batch at once, so one stiff family can hold the whole batch at the cap.
                It also collects the per-row Neumann-term counts of the fused BACKWARD self-loop
                kernel ``_reconciliation_self_loop_transpose_series_kernel``, whose loop runs inside
                the kernel with a per-row early exit (so more terms show up as a longer kernel, not
                as more launches), through the library's own ``set_neumann_term_stat_collection`` /
                ``take_neumann_term_counts`` debug hook. The forward self-loop has no such count:
                the exact solve eliminates the fixed point rather than iterating it.

Usage:
  python benchmark/cc/nsys_theta.py --species S.nwk --families LIST.txt --limit 500 \
      --clade-budget 315000 --theta-pt /path/full_v3.pt --mode nsys
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch


def _load_thetas(theta_pt, paths, device):
    """(fitted, flat) [F,3] log2-rate tensors, fitted rows reordered onto ``paths``."""
    saved = torch.load(theta_pt, map_location="cpu")
    pos = {p: i for i, p in enumerate(saved["paths"])}
    fitted = torch.stack([saved["theta"][pos[p]] for p in paths]).float().to(device)
    flat = torch.full_like(fitted, -6.0)
    return fitted, flat


def _build(species, paths, clade_budget, device, forward_self_loop):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    os.environ.pop("GPUREC_WARM_ADJOINT", None)   # cold adjoint, as in test_grad_theta.py
    so = SolverOptions(**{**dict(_BASE_SOLVER), "pi_iters": 16, "neumann_terms": 16,
                          "forward_self_loop": forward_self_loop})
    m = GeneReconModel(str(species), [str(p) for p in paths], mode="genewise", device=device,
                       dtype=torch.float32, solver_options=so, clade_budget=clade_budget)
    m.receiver_weights.requires_grad_(False)
    return m, so


def _run_nsys(m, thetas):
    """One gradient per theta, each inside its own NVTX range, between cudaProfilerStart/Stop."""
    for label, th in thetas:   # warm up BOTH thetas first: Triton module loads stay out of the capture
        t0 = time.perf_counter()
        m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
        torch.cuda.synchronize()
        print(f"[warmup] {label}: {time.perf_counter() - t0:.2f} s", flush=True)

    torch.cuda.cudart().cudaProfilerStart()
    out = {}
    for label, th in thetas:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"grad_{label}")
        t0 = time.perf_counter()
        lv, g, _ = m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
        torch.cuda.synchronize()
        out[label] = time.perf_counter() - t0
        torch.cuda.nvtx.range_pop()
        print(f"[capture] grad_{label}: {out[label]:.3f} s  loss_sum={float(lv.sum()):.4f} "
              f"grad_absmax={float(g.abs().max()):.4f}", flush=True)
        del lv, g
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    return out


def _install_counters():
    """Monkey-patch the two syncing fixed-point loops so their iterations can be counted.

    Returns (forward_records, adjoint_records, restore) where each record list is filled with one
    (iterations, seconds) tuple per solve. Nothing on disk is changed; ``restore()`` puts the
    original functions back.
    """
    from gpurec.api import _implicit_grad
    from gpurec.core.inference import solver as inference_solver
    from gpurec.core.kernels import e_step
    from gpurec.core.kernels.wave_backward import set_neumann_term_stat_collection

    forward_records = []
    adjoint_records = []

    orig_launch = e_step._launch_e_step_forward_2d
    orig_fixed_point = inference_solver.e_fixed_point_triton
    orig_neumann = _implicit_grad._neumann_e_adjoint
    launch_calls = [0]

    def counting_launch(*a, **k):
        launch_calls[0] += 1
        return orig_launch(*a, **k)

    def counting_fixed_point(*a, **k):
        launch_calls[0] = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_fixed_point(*a, **k)
        torch.cuda.synchronize()
        # The loop launches once per iteration; e_fixed_point_triton then launches once more after
        # the loop to materialise E_s1/E_s2/Ebar. Iterations = calls - 1.
        forward_records.append((launch_calls[0] - 1, time.perf_counter() - t0))
        return out

    def counting_neumann(Av, b, **k):
        n = [0]

        def counting_av(x):
            n[0] += 1
            return Av(x)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_neumann(counting_av, b, **k)
        torch.cuda.synchronize()
        adjoint_records.append((n[0], time.perf_counter() - t0))
        return out

    e_step._launch_e_step_forward_2d = counting_launch
    inference_solver.e_fixed_point_triton = counting_fixed_point
    _implicit_grad._neumann_e_adjoint = counting_neumann
    set_neumann_term_stat_collection(True)

    def restore():
        e_step._launch_e_step_forward_2d = orig_launch
        inference_solver.e_fixed_point_triton = orig_fixed_point
        _implicit_grad._neumann_e_adjoint = orig_neumann
        set_neumann_term_stat_collection(False)

    return forward_records, adjoint_records, restore


def _summarize_rows(tensors, cap):
    """Per-row iteration counts of a fused self-loop kernel, pooled over every wave."""
    if not tensors:
        return dict(rows=0)
    allv = torch.cat([t.reshape(-1).float().cpu() for t in tensors])
    nz = allv[allv > 0]
    if nz.numel() == 0:
        return dict(rows=int(allv.numel()), rows_run=0)
    return dict(
        launches=len(tensors), rows=int(allv.numel()), rows_run=int(nz.numel()),
        total_iters=int(nz.sum()), mean_iters=float(nz.mean()),
        median_iters=float(nz.median()), max_iters=int(nz.max()),
        rows_at_cap=int((nz >= cap).sum()), cap=cap,
    )


def _summarize(records, cap):
    iters = [n for n, _ in records]
    secs = [s for _, s in records]
    if not iters:
        return dict(solves=0)
    return dict(
        solves=len(iters), total_iters=sum(iters), total_s=sum(secs),
        min_iters=min(iters), median_iters=statistics.median(iters), max_iters=max(iters),
        at_cap=sum(1 for n in iters if n >= cap), cap=cap,
    )


def _run_count(m, thetas, solver_options):
    forward_records, adjoint_records, restore = _install_counters()
    result = {}
    try:
        for label, th in thetas:
            m.genewise_loss_vector_and_grad(theta=th, need_grad=True)   # warm-up, not counted
            torch.cuda.synchronize()
            from gpurec.core.kernels.wave_backward import take_neumann_term_counts
            take_neumann_term_counts()
            forward_records.clear()
            adjoint_records.clear()
            t0 = time.perf_counter()
            m.genewise_loss_vector_and_grad(theta=th, need_grad=True)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            fwd = _summarize(forward_records, int(solver_options.e_max_iter))
            adj = _summarize(adjoint_records, int(solver_options.e_adjoint_max_iter))
            neu_loop = _summarize_rows(take_neumann_term_counts(), int(solver_options.neumann_terms))
            result[label] = dict(grad_wall_s=wall, forward_E=fwd, e_adjoint_neumann=adj,
                                 fused_backward_neumann=neu_loop)
            print(f"\n[count] theta={label}  gradient wall {wall:.2f} s", flush=True)
            print(f"[count]   forward E solve : {fwd['solves']} solves, {fwd['total_iters']} iterations "
                  f"total, {fwd['total_s']:.2f} s total; per solve min/median/max = "
                  f"{fwd['min_iters']}/{fwd['median_iters']}/{fwd['max_iters']}, "
                  f"{fwd['at_cap']} solves at the {fwd['cap']} cap", flush=True)
            print(f"[count]   E-adjoint Neumann: {adj['solves']} solves, {adj['total_iters']} iterations "
                  f"total, {adj['total_s']:.2f} s total; per solve min/median/max = "
                  f"{adj['min_iters']}/{adj['median_iters']}/{adj['max_iters']}, "
                  f"{adj['at_cap']} solves at the {adj['cap']} cap", flush=True)
            for tag, d in (("backward fused Neumann", neu_loop),):
                if d.get("rows_run"):
                    print(f"[count]   {tag}: {d['launches']} launches, {d['rows_run']:,} rows ran, "
                          f"{d['total_iters']:,} iterations total; per row mean/median/max = "
                          f"{d['mean_iters']:.2f}/{d['median_iters']:.0f}/{d['max_iters']}, "
                          f"{d['rows_at_cap']:,} rows at the {d['cap']} cap", flush=True)
                else:
                    print(f"[count]   {tag}: no rows recorded ({d})", flush=True)
    finally:
        restore()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True, help="listfile: one .ale path per line")
    ap.add_argument("--limit", required=True, type=int, help="0 = all families; N = first N of the list")
    ap.add_argument("--clade-budget", required=True, type=int, help="clades per batch (model clade_budget)")
    ap.add_argument("--theta-pt", required=True, help="run_genewise .pt with {'theta','paths'}")
    ap.add_argument("--mode", required=True, choices=("nsys", "count"))
    ap.add_argument("--forward-self-loop", required=True, choices=("log", "exact"))
    ap.add_argument("--out", required=True, help="path of the JSON summary to write")
    args = ap.parse_args()

    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]
    dev = torch.device("cuda")

    t0 = time.perf_counter()
    m, so = _build(args.species, paths, args.clade_budget, dev, args.forward_self_loop)
    torch.cuda.synchronize()
    print(f"[build] {time.perf_counter() - t0:.1f} s  families={len(m.families)} "
          f"batches={len(m.batch_statics)} gpu={torch.cuda.get_device_name(0)}", flush=True)

    fitted, flat = _load_thetas(args.theta_pt, paths, dev)
    frac_hot = float((fitted.max(dim=1).values > -2).float().mean())
    print(f"[theta] fitted min={float(fitted.min()):.2f} max={float(fitted.max()):.2f}; "
          f"fraction of families with some rate above 2^-2 = {frac_hot:.3f}", flush=True)
    thetas = [("fitted", fitted), ("flat", flat)]

    rec = dict(mode=args.mode, n_families=len(paths), clade_budget=args.clade_budget,
               n_batches=len(m.batch_statics), theta_pt=args.theta_pt,
               fitted_min=float(fitted.min()), fitted_max=float(fitted.max()), frac_hot=frac_hot,
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    if args.mode == "nsys":
        rec["range_wall_s"] = _run_nsys(m, thetas)
    else:
        rec["counts"] = _run_count(m, thetas, so)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"[out] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
