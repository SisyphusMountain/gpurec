"""Coarse-to-fine genewise fit: solve on truncated clade tables first, then hand the answer over.

The idea under test.  ``fit_genewise`` spends nearly all of its time on Newton gradients, and one
gradient costs roughly the total number of clades of the families that are still being stepped.  A
truncated copy of the same families (built by ``benchmark/cc/truncate_ccp.py``, in either of its two
modes: ``top`` keeps each clade's most probable splits up to a cumulative mass ``tau``, ``thin``
sub-samples the counts as if the table had been built from fewer bootstrap trees) has fewer clades,
so a gradient on it is cheaper.  If the truncated ("coarse") problem has its optimum close
to the real ("fine") one, we can pay many cheap coarse Newton steps, then start the real problem at
the coarse answer with the coarse curvature and hope it finishes in about three full-population
passes instead of the ~25 the baseline needs.

The script runs exactly three phases and times each of them:

  1. COARSE -- ``fit_genewise`` on the truncated .ale files with the production recipe except
     ``tol=--coarse-tol``, ``certify=False`` and ``check_every=1``.  Keeps its ``theta`` (the coarse
     optimum) and its ``curvature`` (the last per-family 3x3 the coarse fit carried).
  2. HANDOVER -- which curvature the fine phase starts from.  ``--handover bfgs`` reuses the 3x3s
     the coarse fit returned as they are.  ``--handover exact`` throws them away and computes the
     analytic 3-probe Hessian of the COARSE model at the coarse optimum instead (about 7 coarse
     gradients), which is a more faithful curvature but is still the curvature of the truncated
     problem, not of the real one.
  3. FINE -- ``fit_genewise`` on the ORIGINAL .ale files with the production recipe, no Adam warm-up
     (``adam_steps=0``), started at the coarse theta and the handover curvature, ``check_every=1``,
     ``tol=1e-3`` and ``certify=True``.

House rule: every command-line argument is required (no defaults, no environment variables), and
every function parameter is passed explicitly by every caller.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import torch

from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.config.memory import MemoryOptions
from gpurec.core.memory_policy import clade_budget_for_device
from gpurec.core.scheduling.batching import DEFAULT_CLADE_BUDGET, parse_families
from gpurec.fit.genewise_fit import _BASE_SOLVER, TRUST_TEST_OFF, _analytic_hessian, fit_genewise

# --- constants of the production recipe (see benchmark/cc/recipe_experiment.py and the values the
# --- baseline run `noverify200` used).  These are facts about the accepted recipe, not knobs of
# --- this experiment, so they are named module constants rather than command-line arguments.
PROD_ADAM_STEPS = 3            # Adam warm-up steps, coarse phase only (the fine phase uses 0)
PROD_ADAM_LR = 1.0
PROD_MIN_DROP = 32             # families that must look converged before a verification round
PROD_REBUILD_FRAC = 0.25       # share of a model's clades that must be frozen before re-planning
PROD_HESSIAN_REFRESH = 15      # Newton iterations between exact analytic Hessians
PROD_MU = 1e-4                 # sign guard on the 3x3 curvature
PROD_TRUST = 2.0               # starting per-family trust radius, in log2 rate units
PROD_TRUST_MAX = 8.0           # largest per-family trust radius
PROD_STALL_PATIENCE = 120
PROD_CURVATURE_UPDATE = "bfgs"  # how the 3x3 is carried between exact Hessians (bfgs/sr1/multisecant)
FINE_TOL = 1e-3                # the real problem's convergence threshold on |projected gradient|
# Starting rates of the production recipe: D = 0.01, L = 0.1, T = 0.01 relative to speciation.
PROD_INIT_LOG2_RATES = (math.log2(0.01), math.log2(0.1), math.log2(0.01))
# The exact-only recipe uses one tier. These counts bound the rare row-local numerical fallbacks,
# so the handover Hessian uses the same budgets as the fit itself.
RECIPE_PI = 16
RECIPE_NEUMANN = 16


class Tee:
    """Write to the real stdout and keep every line, so the fit's own log can be re-read afterwards."""

    def __init__(self, stream, sink):
        self.stream = stream
        self.sink = sink
        self.partial = ""

    def write(self, text):
        self.stream.write(text)
        self.partial += text
        while "\n" in self.partial:
            line, self.partial = self.partial.split("\n", 1)
            self.sink.append(line)
        return len(text)

    def flush(self):
        self.stream.flush()


def read_paths(list_file, limit):
    """The first ``limit`` non-comment lines of a families list file."""
    out = []
    for line in open(list_file):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out[:limit]


def per_family_distance(a, b):
    """Per-family distance in log2 rate units: for each family the largest of the 3 rate gaps.

    Returns (median, 90th percentile, max) over families as plain floats.  The 90th percentile is
    what says whether a handful of families or most of them are far from the target.
    """
    d = sorted((a - b).abs().amax(dim=1).detach().cpu().tolist())
    p90 = d[min(len(d) - 1, int(math.ceil(0.9 * len(d))) - 1)]
    return statistics.median(d), p90, max(d)


def derive_clade_budget(species_tree, paths, device):
    """The clade budget fit_genewise would derive for these families on this card (its lines are mirrored
    here; float32 because every model this script builds directly is built with the default dtype)."""
    parsed = parse_families(species_tree, [str(p) for p in paths])
    meta = parsed.families(list(range(len(paths))))
    budget, _detail = clade_budget_for_device(
        total_clades=sum(int(f["C"]) for f in meta),
        total_splits=sum(int(f["N_splits"]) for f in meta),
        S=int(parsed.species()["S"]),
        dtype=torch.float32,
        device=device,
        fixed_clade_budget=DEFAULT_CLADE_BUDGET,
        scratch_tensors=MemoryOptions().scratch_tensors,
    )
    return budget


def coarse_exact_hessian(species_tree, coarse_paths, theta_coarse, clade_budget, device, dtype):
    """The analytic 3-probe Hessian of the COARSE model at the coarse optimum, as a [F,3,3] tensor.

    Builds the model the same way ``fit_genewise``'s inner ``build`` does: genewise mode, the
    reference recipe's solver options with the recipe's pi / Neumann counts, and the transfer
    receiver weights frozen (the UndatedDTL default of uniform recipients).
    """
    solver = SolverOptions(**{**_BASE_SOLVER, "pi_iters": RECIPE_PI, "neumann_terms": RECIPE_NEUMANN})
    m = GeneReconModel(
        str(species_tree), [str(p) for p in coarse_paths], mode="genewise",
        device=device, dtype=dtype, config=None, solver_options=solver,
        clade_budget=clade_budget,
    )
    m.receiver_weights.requires_grad_(False)
    H = _analytic_hessian(m, theta_coarse, RECIPE_PI, species_tree, list(coarse_paths))
    del m
    torch.cuda.empty_cache()
    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True, help="species tree newick")
    ap.add_argument("--families", required=True, help="list file of the ORIGINAL .ale paths")
    ap.add_argument("--coarse-families", required=True, help="list file of the TRUNCATED .ale paths, same order")
    ap.add_argument("--limit", required=True, type=int, help="how many families to take from each list")
    ap.add_argument("--clade-budget", required=True, type=int, help="clades per batch (0 means: derive from the card)")
    ap.add_argument("--coarse-tol", required=True, type=float, help="|projected gradient| the coarse phase stops at")
    ap.add_argument("--handover", required=True, choices=("bfgs", "exact"),
                    help="curvature the fine phase starts from: the coarse fit's own 3x3s, or the "
                         "analytic Hessian of the coarse model at the coarse optimum")
    ap.add_argument("--baseline", required=True, help=".pt of the baseline fit (dict with 'theta' and 'paths')")
    ap.add_argument("--baseline-nll", required=True, type=float,
                    help="the baseline fit's certified NLL in bits, for the headline comparison")
    ap.add_argument("--fine-adam-steps", required=True, type=int,
                    help="Adam warm-up steps the FINE phase takes from the coarse theta. 0 is the "
                         "plain handover. 1 spends one extra full-population gradient on an Adam "
                         "move before Newton starts; the handed-over curvature is kept either way "
                         "(one Adam step produces no (step, gradient-change) pair to fold in).")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    log_lines: list[str] = []
    sys.stdout = Tee(sys.__stdout__, log_lines)

    paths = read_paths(args.families, args.limit)
    coarse_paths = read_paths(args.coarse_families, args.limit)
    if len(paths) != len(coarse_paths):
        raise ValueError(f"{len(paths)} original families but {len(coarse_paths)} truncated ones")
    # 0 means "size the batches to the card", the way fit_genewise does it.  The budget must be a
    # number here because this script also builds models directly (the coarse Hessian and the final
    # comparison model), and a GeneReconModel built with clade_budget=None puts every family in ONE
    # batch (measured: 200 families in one 11 GiB batch on the RTX 4090, out of memory).
    budget = (derive_clade_budget(args.species, paths, torch.device("cuda"))
              if args.clade_budget == 0 else args.clade_budget)

    # Everything both phases share.  `tol`, `certify`, `adam_steps`, `init_curvature` and
    # `init_log2_rates` differ between the two phases and are passed at each call site.
    common = dict(
        device="cuda", dtype=None, certify_curvature=False,
        min_drop=PROD_MIN_DROP, rebuild_frac=PROD_REBUILD_FRAC, hessian_refresh=PROD_HESSIAN_REFRESH,
        mu=PROD_MU, trust=PROD_TRUST, trust_max=PROD_TRUST_MAX, stall_patience=PROD_STALL_PATIENCE,
        adam_lr=PROD_ADAM_LR, check_every=1, clade_budget=budget,
        solver_options=None, config=None, verbose=True,
        curvature_update=PROD_CURVATURE_UPDATE,
        # Round-5 step experiments, all off here: this driver measures the coarse-to-fine handover,
        # not the step rule (see fit_genewise's documentation for what each value turns on).
        step_extrapolation=1.0, step_model="quadratic", stop_nll_bits=0.0,
        approach_pruning_threshold=0.0,
        # Round-6 experiments, off here for the same reason.
        targeted_hessian=(0, 0.0), coordinate_staging=(0, 0), trust_test=TRUST_TEST_OFF,
    )

    # ---------------- 1. COARSE ----------------
    print(f"[c2f] ===== coarse phase: {len(coarse_paths)} truncated families, tol={args.coarse_tol} =====")
    mark = len(log_lines)
    t0 = time.perf_counter()
    coarse = fit_genewise(
        args.species, coarse_paths,
        adam_steps=PROD_ADAM_STEPS, init_curvature="adam_bfgs", init_log2_rates=PROD_INIT_LOG2_RATES,
        tol=args.coarse_tol, certify=False, **common,
    )
    coarse_s = time.perf_counter() - t0
    coarse_lines = log_lines[mark:]
    theta_c = coarse["theta"].detach().clone()
    print(f"[c2f] coarse done: wall {coarse_s:.1f}s, {coarse['n_steps']} Newton steps, "
          f"{coarse['n_hessians']} exact Hessians, newton_grad {coarse['newton_grad_seconds']:.1f}s, "
          f"adam {coarse['adam_seconds']:.1f}s")

    # ---------------- 2. HANDOVER ----------------
    t0 = time.perf_counter()
    if args.handover == "bfgs":
        handover_curvature = coarse["curvature"].detach().clone()
    else:
        handover_curvature = coarse_exact_hessian(
            species_tree=args.species, coarse_paths=coarse_paths, theta_coarse=theta_c,
            clade_budget=budget, device=torch.device("cuda"), dtype=theta_c.dtype,
        )
    handover_s = time.perf_counter() - t0
    print(f"[c2f] handover ({args.handover}) took {handover_s:.1f}s")

    # ---------------- 3. FINE ----------------
    print(f"[c2f] ===== fine phase: {len(paths)} original families, "
          f"{args.fine_adam_steps} Adam warm-up steps =====")
    mark = len(log_lines)
    t0 = time.perf_counter()
    fine = fit_genewise(
        args.species, paths,
        adam_steps=args.fine_adam_steps, init_curvature=handover_curvature, init_log2_rates=theta_c,
        tol=FINE_TOL, certify=True, **common,
    )
    fine_s = time.perf_counter() - t0
    fine_lines = log_lines[mark:]
    theta_f = fine["theta"].detach().clone()

    # ---------------- 4. REPORT ----------------
    # Line up the baseline's rows with ours BY PATH, so a baseline fitted on a superset (or in
    # another order) still gives the right per-family comparison.
    base = torch.load(args.baseline, map_location="cpu", weights_only=False)
    base_row = {p: i for i, p in enumerate(base["paths"])}
    missing = [p for p in paths if p not in base_row]
    if missing:
        raise ValueError(f"{len(missing)} of our families are absent from the baseline, e.g. {missing[0]}")
    rows = torch.tensor([base_row[p] for p in paths], dtype=torch.long)
    theta_b = base["theta"].index_select(0, rows).to(device=theta_f.device, dtype=theta_f.dtype)
    base_nll = args.baseline_nll

    fine_vs_base_med, fine_vs_base_p90, fine_vs_base_max = per_family_distance(theta_f, theta_b)
    coarse_offset_med, coarse_offset_p90, coarse_offset_max = per_family_distance(theta_c, theta_f)
    # Where the coarse phase actually left us relative to the TRUE optimum (the baseline fit), which
    # is the quantity the warm-start hypothesis is about; `coarse_offset_*` above measures it
    # against this run's own fine answer instead.
    coarse_vs_base_med, coarse_vs_base_p90, coarse_vs_base_max = per_family_distance(theta_c, theta_b)

    # How many families were ALREADY converged when the fine phase first looked (its iteration 0)?
    # With check_every=1 the fit prints one "live=.. conv=.." line per Newton iteration; `conv` on
    # the it0 line is the count that the coarse optimum already satisfies at the real problem's tol.
    conv_at_zero = None
    for line in fine_lines:
        if "it0]" in line and "conv=" in line:
            conv_at_zero = int(line.split("conv=")[1].split()[0])
            break

    # The honest cost measure is the number of gradients over the whole live population, which is
    # one per "live=" log line.  ``n_steps`` is smaller than that: it counts only the iterations
    # that actually ended in a Newton step, and skips the ones whose gradient was thrown away by a
    # model re-plan or by the last family freezing.
    coarse_grad_passes = sum(1 for line in coarse_lines if "live=" in line)
    fine_grad_passes = sum(1 for line in fine_lines if "live=" in line)

    print("")
    print("[c2f] ---- fine phase per-iteration log (one line = one full-population gradient) ----")
    for line in fine_lines:
        if "live=" in line or "re-planned" in line or "stalled" in line:
            print("[c2f]   " + line.strip())
    print("")

    # Basin changes: evaluate the ORIGINAL files once at our theta and once at the baseline theta,
    # so the two per-family NLL vectors come from the same model and the same solver tier and their
    # difference is only the rates.  A family whose NLL differs by more than 0.1 bits ended in a
    # different local optimum, not in float noise (the fit's own tolerance is 1e-3 on the gradient).
    m_cmp = GeneReconModel(str(args.species), [str(p) for p in paths], mode="genewise",
                           device="cuda", dtype=None, config=None, solver_options=None,
                           clade_budget=budget)
    with torch.no_grad():
        nll_ours = m_cmp.genewise_loss_vector(theta=theta_f.to(m_cmp.theta.dtype)).double().cpu()
        nll_base = m_cmp.genewise_loss_vector(theta=theta_b.to(m_cmp.theta.dtype)).double().cpu()
    del m_cmp
    torch.cuda.empty_cache()
    d_nll = nll_ours - nll_base
    changed = [(paths[i].split("/")[-1].split(".")[0], float(d_nll[i]))
               for i in range(len(paths)) if abs(float(d_nll[i])) > 0.1]
    changed.sort(key=lambda row: -abs(row[1]))
    print(f"[c2f] families whose NLL differs from the baseline by more than 0.1 bits: {len(changed)}")
    for name, delta in changed[:15]:
        print(f"[c2f]   {name} {delta:+.3f} bits")

    summary = dict(
        tag=args.tag, handover=args.handover, coarse_tol=args.coarse_tol,
        coarse_families=args.coarse_families, n_families=len(paths),
        coarse_s=coarse_s, handover_s=handover_s, fine_s=fine_s,
        total_s=coarse_s + handover_s + fine_s,
        coarse_steps=int(coarse["n_steps"]), fine_steps=int(fine["n_steps"]),
        coarse_grad_passes=coarse_grad_passes, fine_grad_passes=fine_grad_passes,
        fine_adam_s=fine["adam_seconds"], fine_hessian_s=fine["hessian_seconds"],
        fine_n_hessians=int(fine["n_hessians"]), fine_newton_grad_s=fine["newton_grad_seconds"],
        fine_verify_s=fine["verify_seconds"], fine_rebuild_s=fine["rebuild_seconds"],
        fine_certify_s=fine["certify_seconds"],
        nll_bits=float(fine["loss_bits"]), nll_minus_baseline=float(fine["loss_bits"]) - base_nll,
        certified=int(fine["converged"]), unconverged=int(fine["unconverged"]),
        bound_active=int(fine["bound_active"]), pg_max=float(fine["pg_max"]),
        certified_at_iteration_0=conv_at_zero,
        fine_vs_baseline_median=fine_vs_base_med, fine_vs_baseline_p90=fine_vs_base_p90,
        fine_vs_baseline_max=fine_vs_base_max,
        coarse_offset_median=coarse_offset_med, coarse_offset_p90=coarse_offset_p90,
        coarse_offset_max=coarse_offset_max,
        coarse_vs_baseline_median=coarse_vs_base_med, coarse_vs_baseline_p90=coarse_vs_base_p90,
        coarse_vs_baseline_max=coarse_vs_base_max,
        fine_adam_steps=args.fine_adam_steps,
        n_families_nll_changed=len(changed), nll_changed=changed[:15],
        nll_sum_ours=float(nll_ours.sum()), nll_sum_baseline_theta=float(nll_base.sum()),
    )
    print("[c2f] " + json.dumps(summary, default=str))

    torch.save({"theta": theta_f.cpu(), "theta_coarse": theta_c.cpu(), "paths": paths,
                "summary": summary}, f"{args.out_dir}/c2f_{args.tag}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
