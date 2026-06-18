"""Feasibility probe for the Sanderson-style CV run on the 1055-family hogenom set.

Times the per-iteration solve cost (value+grad and value-only) at the CV solver settings
(pi=64/neumann=64) and reports peak GPU memory, so we can (a) estimate total CV wall-clock and
(b) decide whether the exact-HVP PD certificate fits in 24 GB. Pure measurement, no fit.

    GPUREC_PREPROCESS_PATH=<.../libgpurec_preprocess.so> python experiments/sanderson_cv/probe.py
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.optim.value_and_grad import make_value_and_grad

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_hogenom_core/hogenom")
SP_TREE = (ROOT / "runs/MFP/true_start_ufboot1000/"
           "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/"
           "starting_species_tree.newick")

_CV_SO = dict(
    e_max_iter=2000, e_tol=1e-8, pi_iters=64, neumann_terms=64,
    self_loop_solver="neumann", bicgstab_max_iter=500, bicgstab_tol=1e-7,
    bicgstab_breakdown_tol=1e-30, adjoint_pruning_threshold=1e-6,
    use_adjoint_pruning=True, pibar_side_threshold=0.0,
)


def _family_paths(n=None):
    fams = [ln.strip() for ln in open(HERE / "families_1055.txt") if ln.strip()]
    if n is not None:
        fams = fams[:n]
    return [str(ROOT / "families" / f / "gene_trees" / "ufboot1000.MFP.geneTree.newick") for f in fams]


def main(n_families=None, warmup=2, reps=5):
    so = SolverOptions(**_CV_SO)
    so.validate()
    paths = _family_paths(n_families)
    print(f"[probe] building model: {len(paths)} families, pi=64/neumann=64 ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = GeneReconModel(str(SP_TREE), paths, mode="specieswise", device="cuda", solver_options=so)
    torch.cuda.synchronize()
    t_build = time.perf_counter() - t0
    S = int(model.species_helpers["S"])
    nbatches = len(model.batch_statics)
    mem_build = torch.cuda.max_memory_allocated() / 1e9
    print(f"[probe] build: {t_build:.1f}s  S={S}  batches={nbatches}  peak_mem(build)={mem_build:.2f} GB")

    rw = model.receiver_weights.detach().clone()
    theta = torch.zeros((S, 3), device="cuda", dtype=torch.float32)  # init theta=0 -> all DTL probs 0.25

    f = make_value_and_grad(model.batch_statics, rw, theta_shape=(S, 3))

    # value+grad timing
    torch.cuda.reset_peak_memory_stats()
    warm = None
    for _ in range(warmup):
        loss, g, _sv, warm = f(theta.reshape(-1), warm_E=warm)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        loss, g, _sv, warm = f(theta.reshape(-1), warm_E=warm)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    mem_vg = torch.cuda.max_memory_allocated() / 1e9
    vg = sorted(ts)[len(ts) // 2]
    print(f"[probe] value+grad: median={vg*1e3:.0f} ms  (min={min(ts)*1e3:.0f} max={max(ts)*1e3:.0f})  "
          f"peak_mem={mem_vg:.2f} GB  loss={float(loss):.2f}  |g|={float(g.norm()):.3e}")

    # value-only timing (held-out NLL path)
    from gpurec.api._execution import stream_batches
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        l, _, _ = stream_batches(model.batch_statics, theta, rw, genewise=False, need_grad=False)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        l, _, _ = stream_batches(model.batch_statics, theta, rw, genewise=False, need_grad=False)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    vo = sorted(ts)[len(ts) // 2]
    print(f"[probe] value-only: median={vo*1e3:.0f} ms  peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # rough CV cost estimate
    K, NLAM, ITERS = 5, 5, 100  # folds, |lambda grid|, ~adam+lbfgs solves per (fold,lambda)
    est = K * NLAM * ITERS * vg
    print(f"\n[probe] rough CV estimate: {K} folds x {NLAM} lambdas x ~{ITERS} solves x {vg*1e3:.0f} ms "
          f"~= {est/60:.0f} min ({est/3600:.1f} h)  + folds rebuild models")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n_families=n)
