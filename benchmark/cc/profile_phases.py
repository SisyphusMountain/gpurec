"""Phase-by-phase timing of the genewise fit's expensive pieces on one GPU.

Measures, with the SAME solver settings the production recipe uses
(``gpurec.fit.genewise_fit._BASE_SOLVER`` + pi_iters=16 / neumann_terms=16, the recipe's
first tier, warm adjoint on, receiver weights frozen):

  build       -- ``GeneReconModel(...)`` construction, and separately a bare
                 ``preprocess_dataset(...)`` call on the same inputs (the Rust preprocessor).
  grad        -- ``genewise_loss_vector_and_grad(need_grad=True)``: 1 warm-up (Triton compiles) + 3 timed.
  fwd_resid   -- one ``solve_forward_residual`` pass over every batch.
  hessian     -- one ``gpurec.fit.genewise_fit._analytic_hessian`` call (3 analytic-HVP probes),
                 and -- because that call can run out of GPU memory on a many-batch model -- a
                 per-batch streamed variant of the same computation implemented here.
  rebuild     -- a second ``GeneReconModel(...)`` over the same paths (warm Python / warm page cache).

Every phase that can run out of GPU memory is caught, recorded and skipped, so one failure does not
lose the other phases' numbers. Prints every number with a label and writes a JSON summary to ``--out``.

Usage:
  python benchmark/cc/profile_phases.py --species S.nwk --families LIST.txt \
      --limit 500 --clade-budget 315000 --out /path/out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def _sync():
    torch.cuda.synchronize()


def _peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30


def _live_gib() -> float:
    return torch.cuda.memory_allocated() / 2**30


def _timed(fn):
    """Run ``fn`` with CUDA synchronised on both sides; return (result, seconds)."""
    _sync()
    t0 = time.perf_counter()
    out = fn()
    _sync()
    return out, time.perf_counter() - t0


def _streamed_hessian(m, theta, pi_cur):
    """Per-batch streamed version of ``_analytic_hessian``: build the forward solve + adjoint point
    cache ONCE per batch and reuse it for the 3 probes, freeing between batches. Same [G,3,3] result
    as the library call, but it never holds more than one batch's HVP state at a time."""
    from gpurec.solver.hvp.exact import make_exact_hvp_single
    from gpurec.solver.value_and_grad import forward_solve, free_cuda_cache_if_tight

    G = theta.shape[0]
    dev, dtype = theta.device, theta.dtype
    rw = m.receiver_weights.detach()
    cols = [torch.zeros(G, 3, device=dev, dtype=dtype) for _ in range(3)]
    for static in m.batch_statics:
        fam = static.family_index_tensor.to(dev)
        G_b = int(fam.numel())
        theta_b = theta.index_select(0, fam).contiguous()
        _l, sv = forward_solve([static], theta, rw)
        hvp = make_exact_hvp_single(static, theta_b, rw, sv, tangent_self_iters=pi_cur)
        for j in range(3):
            u_b = torch.zeros(G_b, 3, device=dev, dtype=dtype)
            u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1), probe_id=j)[: G_b * 3].reshape(G_b, 3)
            cols[j].index_add_(0, fam, out_b)
        del hvp, sv
        # Drop this batch's tangent-adjoint warm-start cache (3 probes x clades x species floats,
        # ~7.6 GiB for a 315k-clade batch): it is only reused across repeated HVP calls on the SAME
        # batch, and keeping all batches' copies alive is what exhausts the GPU.
        static.warm_v_tangent = None
        free_cuda_cache_if_tight()
    H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True, help="listfile: one .ale path per line")
    ap.add_argument("--limit", required=True, type=int, help="0 = all families; N = first N of the list")
    ap.add_argument("--clade-budget", required=True, type=int, help="clades per batch (model clade_budget)")
    ap.add_argument("--out", required=True, help="path of the JSON summary to write")
    args = ap.parse_args()

    # Same environment knob the recipe sets (the library memory-gate may still disable it).
    os.environ["GPUREC_WARM_ADJOINT"] = "1"

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.core.inference.solver import solve_forward_residual
    from gpurec.core.scheduling.batching import preprocess_dataset
    from gpurec.fit.genewise_fit import _BASE_SOLVER, _analytic_hessian

    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]

    # The recipe's first tier: pi_iters=16, neumann_terms=16 on top of the genewise base solver.
    pi_cur = 16
    solver_options = SolverOptions(**{**dict(_BASE_SOLVER), "pi_iters": pi_cur, "neumann_terms": 16})
    dtype = torch.float32
    dev = torch.device("cuda")

    rec: dict = {
        "species": args.species,
        "families_list": args.families,
        "n_families": len(paths),
        "clade_budget": args.clade_budget,
        "pi_iters": pi_cur,
        "neumann_terms": 16,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
    }

    def _dump():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rec, fh, indent=2)

    print(f"[cfg] families={len(paths)} clade_budget={args.clade_budget} pi_iters={pi_cur} "
          f"neumann_terms=16 dtype={dtype} gpu={rec['gpu']} torch={torch.__version__}", flush=True)

    torch.cuda.reset_peak_memory_stats()

    # ---- phase 1a: bare preprocess_dataset (Rust preprocessor + JSON decode, CPU only) ----
    def _pre():
        return preprocess_dataset(
            str(args.species), paths,
            accumulator_dtype=torch.float64,
            family_chunk_size=300,
            clade_budget=args.clade_budget,
            batch_packing="depth_first_fit",
            max_wave_size=8192,
        )

    raw, t_pre = _timed(_pre)
    n_batches_raw = len(raw.get("batches") or [])
    del raw
    print(f"[build] preprocess_dataset      : {t_pre:8.2f} s   ({n_batches_raw} batches planned)", flush=True)
    rec["preprocess_dataset_s"] = t_pre
    _dump()

    # ---- phase 1b: full model build ----
    def _build():
        m = GeneReconModel(
            str(args.species), [str(p) for p in paths], mode="genewise",
            device=dev, dtype=dtype, solver_options=solver_options,
            clade_budget=args.clade_budget,
        )
        m.receiver_weights.requires_grad_(False)
        return m

    m, t_build = _timed(_build)
    print(f"[build] GeneReconModel(...)     : {t_build:8.2f} s", flush=True)
    rec["build_s"] = t_build

    n_batches = len(m.batch_statics)
    per_batch_clades = [int(s.wave_layout["leaf_species_index"].numel()) for s in m.batch_statics]
    per_batch_waves = [len(s.wave_layout["wave_metas"]) for s in m.batch_statics]
    per_batch_families = [len(s.family_indices) for s in m.batch_statics]
    S = int(m.species_helpers["S"])
    rec.update(
        n_batches=n_batches, S=S,
        per_batch_clades=per_batch_clades, per_batch_waves=per_batch_waves,
        per_batch_families=per_batch_families,
        total_clades=sum(per_batch_clades), total_waves=sum(per_batch_waves),
        warm_adjoint_ok=bool(m.warm_adjoint_ok),
    )
    print(f"[build] batches={n_batches}  species_nodes={S}  total_clades={sum(per_batch_clades):,}  "
          f"total_waves={sum(per_batch_waves)}  warm_adjoint_ok={m.warm_adjoint_ok}", flush=True)
    for i, (c, w, f) in enumerate(zip(per_batch_clades, per_batch_waves, per_batch_families)):
        print(f"[build]   batch {i:3d}: families={f:5d} clades={c:9,d} waves={w:4d}", flush=True)
    rec["peak_gib_after_build"] = _peak_gib()
    rec["live_gib_after_build"] = _live_gib()
    print(f"[mem]  after build   peak={rec['peak_gib_after_build']:7.2f} GiB "
          f"resident={rec['live_gib_after_build']:7.2f} GiB", flush=True)
    _dump()

    theta = torch.zeros(len(m.families), 3, device=dev, dtype=dtype)

    # ---- phase 2: loss + gradient ----
    def _lg():
        lv, g, _ = m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        return lv, g

    (lv, g), t_warm = _timed(_lg)
    print(f"[grad] warm-up call (compiles)  : {t_warm:8.2f} s   loss_sum={float(lv.sum()):.4f}", flush=True)
    rec["grad_warmup_s"] = t_warm
    rec["loss_sum_bits"] = float(lv.sum())
    rec["grad_absmax"] = float(g.abs().max())
    del lv, g
    grad_times = []
    for k in range(3):
        (lv, g), t = _timed(_lg)
        del lv, g
        grad_times.append(t)
        print(f"[grad] timed call {k + 1}             : {t:8.2f} s", flush=True)
    rec["grad_s"] = grad_times
    rec["grad_mean_s"] = sum(grad_times) / len(grad_times)
    rec["peak_gib_after_grad"] = _peak_gib()
    rec["live_gib_after_grad"] = _live_gib()
    print(f"[grad] mean of 3                : {rec['grad_mean_s']:8.2f} s", flush=True)
    print(f"[mem]  after grad    peak={rec['peak_gib_after_grad']:7.2f} GiB "
          f"resident={rec['live_gib_after_grad']:7.2f} GiB", flush=True)
    _dump()

    # ---- phase 3: forward residual over all batches ----
    def _fwd():
        out = torch.zeros(len(m.families), device=dev, dtype=dtype)
        rw = m.receiver_weights.detach()
        with torch.no_grad():
            for static in m.batch_statics:
                r = solve_forward_residual(static, m._theta_for_static(static, theta), rw, pi_iters=pi_cur)
                out[static.family_index_tensor.to(dev)] = r.to(dev)
        return out

    try:
        resid, t_fwd = _timed(_fwd)
        print(f"[fwd]  solve_forward_residual   : {t_fwd:8.2f} s   max_resid={float(resid.max()):.3e}", flush=True)
        rec["forward_residual_s"] = t_fwd
        rec["forward_residual_max"] = float(resid.max())
        del resid
    except torch.OutOfMemoryError as exc:
        rec["forward_residual_s"] = None
        rec["forward_residual_oom"] = str(exc).splitlines()[0]
        print(f"[fwd]  solve_forward_residual   : OUT OF MEMORY -- {rec['forward_residual_oom']}", flush=True)
    torch.cuda.empty_cache()
    rec["peak_gib_after_forward_residual"] = _peak_gib()
    rec["live_gib_after_forward_residual"] = _live_gib()
    print(f"[mem]  after fwd     peak={rec['peak_gib_after_forward_residual']:7.2f} GiB "
          f"resident={rec['live_gib_after_forward_residual']:7.2f} GiB", flush=True)
    _dump()

    # ---- phase 4a: the library's _analytic_hessian (3 HVP probes) ----
    try:
        H, t_hess = _timed(lambda: _analytic_hessian(m, theta, pi_cur))
        print(f"[hess] _analytic_hessian        : {t_hess:8.2f} s   H[0,0,0]={float(H[0, 0, 0]):.4f}", flush=True)
        rec["hessian_s"] = t_hess
        rec["hessian_h000"] = float(H[0, 0, 0])
        del H
    except torch.OutOfMemoryError as exc:
        rec["hessian_s"] = None
        rec["hessian_oom"] = str(exc).splitlines()[0]
        print(f"[hess] _analytic_hessian        : OUT OF MEMORY -- {rec['hessian_oom']}", flush=True)
    torch.cuda.empty_cache()
    rec["peak_gib_after_hessian"] = _peak_gib()
    print(f"[mem]  after hessian peak={rec['peak_gib_after_hessian']:7.2f} GiB "
          f"resident={_live_gib():7.2f} GiB", flush=True)
    _dump()

    # ---- phase 4b: per-batch streamed Hessian (same result, one batch of state at a time) ----
    try:
        H, t_hess_s = _timed(lambda: _streamed_hessian(m, theta, pi_cur))
        print(f"[hess] streamed per-batch       : {t_hess_s:8.2f} s   H[0,0,0]={float(H[0, 0, 0]):.4f}", flush=True)
        rec["hessian_streamed_s"] = t_hess_s
        rec["hessian_streamed_h000"] = float(H[0, 0, 0])
        del H
    except torch.OutOfMemoryError as exc:
        rec["hessian_streamed_s"] = None
        rec["hessian_streamed_oom"] = str(exc).splitlines()[0]
        print(f"[hess] streamed per-batch       : OUT OF MEMORY -- {rec['hessian_streamed_oom']}", flush=True)
    torch.cuda.empty_cache()
    rec["peak_gib_after_hessian_streamed"] = _peak_gib()
    print(f"[mem]  after streamed peak={rec['peak_gib_after_hessian_streamed']:7.2f} GiB "
          f"resident={_live_gib():7.2f} GiB", flush=True)
    _dump()

    # ---- phase 5: rebuild (warm Python, warm page cache) ----
    del m, theta
    torch.cuda.empty_cache()
    m2, t_rebuild = _timed(_build)
    print(f"[build] rebuild GeneReconModel  : {t_rebuild:8.2f} s", flush=True)
    rec["rebuild_s"] = t_rebuild
    rec["peak_gib_after_rebuild"] = _peak_gib()
    print(f"[mem]  after rebuild peak={rec['peak_gib_after_rebuild']:7.2f} GiB", flush=True)
    del m2
    torch.cuda.empty_cache()

    rec["peak_gib_total"] = _peak_gib()
    print(f"[mem]  peak overall             : {rec['peak_gib_total']:8.2f} GiB", flush=True)

    _dump()
    print(f"[out]  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
