"""Trace per-family NLL and gradient at every evaluation of the genewise recipe, then summarize:
for each family, the Newton iteration at which its NLL last improved by more than --nll-eps bits,
versus the iteration at which its projected gradient first fell below --tol.

Usage: recipe_trace.py --species S --families LIST --limit 200 --clade-budget 100000 --out trace.pt \
          --nll-eps 1e-4 --tol 1e-3 --curvature-update bfgs --dtype float32
"""
from __future__ import annotations

import argparse
import math
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True); ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int); ap.add_argument("--clade-budget", required=True, type=int)
    ap.add_argument("--out", required=True); ap.add_argument("--nll-eps", required=True, type=float)
    ap.add_argument("--tol", required=True, type=float)
    ap.add_argument("--curvature-update", required=True, choices=("bfgs", "sr1", "multisecant"))
    ap.add_argument("--dtype", required=True, choices=("float32", "float64"))
    ap.add_argument("--step-extrapolation", required=True, type=float)
    ap.add_argument("--step-model", required=True, choices=("quadratic", "rate_affine"))
    ap.add_argument("--stop-nll-bits", required=True, type=float)
    ap.add_argument("--approach-pruning-threshold", required=True, type=float)
    # Round-6 experiments; off = 0 / 0.0 / 0 / 0 and the ratio test's own 0.25 / 0.75 / 0.5 / 0.05.
    ap.add_argument("--stuck-from", required=True, type=int)
    ap.add_argument("--stuck-max-frac", required=True, type=float)
    ap.add_argument("--stage-freeze-t", required=True, type=int)
    ap.add_argument("--stage-d-only", required=True, type=int)
    ap.add_argument("--trust-shrink", required=True, type=float)
    ap.add_argument("--trust-grow-ratio", required=True, type=float)
    ap.add_argument("--trust-radius-min", required=True, type=float)
    ap.add_argument("--trust-min-predicted-bits", required=True, type=float)
    args = ap.parse_args()
    from gpurec.api import model as model_module
    from gpurec.fit.genewise_fit import fit_genewise, _GENEWISE_RATE_BOUNDS
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")][: args.limit]
    index_of = {p: i for i, p in enumerate(paths)}
    records = []   # (call, family index tensor, theta [n,3], nll [n], grad [n,3])
    orig = model_module.GeneReconModel.genewise_loss_vector_and_grad
    orig_init = model_module.GeneReconModel.__init__

    def traced_init(self, species_tree, gene_trees, *a, **k):
        self._trace_paths = [str(p) for p in gene_trees]
        orig_init(self, species_tree, gene_trees, *a, **k)

    model_module.GeneReconModel.__init__ = traced_init

    def traced(self, *a, **k):
        out = orig(self, *a, **k)
        theta = k.get("theta", a[0] if a else None)
        fam = torch.tensor([index_of[p] for p in self._trace_paths], dtype=torch.long)
        loss = out[0].detach().double().cpu()
        grad = out[1].detach().double().cpu() if (k.get("need_grad", False) and out[1] is not None) else None
        records.append((len(records), fam, theta.detach().double().cpu().clone(), loss, grad))
        return out

    model_module.GeneReconModel.genewise_loss_vector_and_grad = traced
    fit_genewise(
        args.species, paths, device="cuda", dtype=args.dtype, certify=True, certify_curvature=False,
        min_drop=32, rebuild_frac=0.25, hessian_refresh=15, init_curvature="adam_bfgs",
        curvature_update=args.curvature_update,
        solver_options=None, config=None, verbose=False,
        init_log2_rates=(math.log2(0.01), math.log2(0.1), math.log2(0.01)),
        clade_budget=(None if args.clade_budget == 0 else args.clade_budget), stall_patience=120, trust_max=8.0, adam_steps=3, mu=1e-4,
        step_extrapolation=args.step_extrapolation, step_model=args.step_model,
        stop_nll_bits=args.stop_nll_bits, approach_pruning_threshold=args.approach_pruning_threshold,
        targeted_hessian=(args.stuck_from, args.stuck_max_frac),
        coordinate_staging=(args.stage_freeze_t, args.stage_d_only),
        trust_test=(args.trust_shrink, args.trust_grow_ratio,
                    args.trust_radius_min, args.trust_min_predicted_bits),
    )
    model_module.GeneReconModel.genewise_loss_vector_and_grad = orig
    model_module.GeneReconModel.__init__ = orig_init
    torch.save(records, args.out)

    lo, hi = math.log2(_GENEWISE_RATE_BOUNDS.min_rate), math.log2(_GENEWISE_RATE_BOUNDS.max_rate)
    eps = 1e-3
    n = len(paths)
    # per family: list of (call, nll, |Pg|) over gradient evaluations
    hist = [[] for _ in range(n)]
    for call, fam, theta, loss, grad in records:
        if grad is None:
            continue
        fixed = ((theta >= hi - eps) & (grad < 0)) | ((theta <= lo + eps) & (grad > 0))
        pg = (grad * (~fixed)).abs().amax(dim=1)
        for j in range(fam.numel()):
            hist[int(fam[j])].append((call, float(loss[j]), float(pg[j])))
    it_conv, it_last_improve, n_evals, wasted = [], [], [], 0
    for h in hist:
        h.sort()
        nlls = [x[1] for x in h]; pgs = [x[2] for x in h]
        best = nlls[0]; last_imp = 0
        for i, v in enumerate(nlls):
            if v < best - args.nll_eps:
                best = v; last_imp = i
        conv = next((i for i, p in enumerate(pgs) if p < args.tol), len(h) - 1)
        it_conv.append(conv); it_last_improve.append(last_imp); n_evals.append(len(h))
        wasted += max(0, conv - last_imp)
    import statistics as st
    print(f"[trace] families {n}; gradient evaluations per family: median {st.median(n_evals):.0f}, max {max(n_evals)}", flush=True)
    print(f"[trace] evaluation index of last NLL improvement > {args.nll_eps:g} bits: median {st.median(it_last_improve):.0f}, p90 {sorted(it_last_improve)[int(0.9*n)]}", flush=True)
    print(f"[trace] evaluation index of first |Pg| < {args.tol:g}: median {st.median(it_conv):.0f}, p90 {sorted(it_conv)[int(0.9*n)]}", flush=True)
    print(f"[trace] evaluations spent after the NLL stopped improving, summed over families: {wasted} of {sum(n_evals)} ({100*wasted/sum(n_evals):.0f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
