"""Certify a TRUE local minimum for every lambda of a Sanderson-CV run.

Builds the model ONCE, then for each lambda's all-data refit checkpoint runs the saddle-escape +
Newton polish (saddle_escape.run): escape if the refit is a saddle, always Newton-polish to |g|->0,
re-certify the Hessian. Prints a per-lambda table and saves per-lambda result checkpoints.

  DATASET=archaea FAMILIES=256 CKPT_DIR=<archaea_cv/ckpt> OUT_DIR=<out> \
  LAMBDAS="10 3 1 0.3 0.1 0.03 0" python certify_all_lambdas.py
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import saddle_escape as se
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions

DEV = "cuda"
ds = os.environ.get("DATASET", "archaea")
n = int(os.environ.get("FAMILIES", "256"))
ckpt_dir = os.environ["CKPT_DIR"]
out_dir = os.environ.get("OUT_DIR", ckpt_dir + "/certified")
lambdas = [float(x) for x in os.environ.get("LAMBDAS", "10 3 1 0.3 0.1 0.03 0").split()]
os.makedirs(out_dir, exist_ok=True)

so = SolverOptions(**_CV_SO); so.validate()
t0 = time.time()
m = GeneReconModel(str(DATASETS[ds]["species_tree"]), [str(x) for x in DATASETS[ds]["families"](n)],
                   mode="specieswise", device=DEV, solver_options=so)
bs = m.batch_statics; rw = m.receiver_weights.detach(); sp = m.species_helpers["sp_parent"]
S = int(m.species_helpers["S"])
print(f"[certify_all] dataset={ds} n={n} S={S} p={3*S} batches={len(bs)} lambdas={lambdas} "
      f"(model built {time.time()-t0:.0f}s)", flush=True)

rows = []
for i, lam in enumerate(lambdas):
    theta_path = f"{ckpt_dir}/refit_lam{i}.pt"
    if not os.path.exists(theta_path):
        print(f"[lam={lam}] MISSING {theta_path}; skip", flush=True); continue
    print(f"\n########## lambda = {lam}  (refit_lam{i}.pt) ##########", flush=True)
    res = se.run(bs, rw, sp, S, theta_path, lam, full=None,
                 out_path=f"{out_dir}/{ds}_lam{lam}_certified.pt", meta=dict(dataset=ds, families=n, lam=lam))
    rows.append((lam, res))

print("\n================= CERTIFIED-MINIMUM SUMMARY =================", flush=True)
print(f"{'lambda':>8} {'refit_saddle?':>13} {'lam_min(refit)':>15} {'lam_min(final)':>15} "
      f"{'|g|(final)':>12} {'F(refit)':>12} {'F(final)':>12} {'certified':>10}")
for lam, r in rows:
    print(f"{lam:>8g} {str(r['is_saddle']):>13} {r['lam_min_saddle']:>+15.5e} {r['lam_min_newton']:>+15.6e} "
          f"{r['gnorm_newton']:>12.3e} {r['loss_saddle']:>12.4f} {r['loss_newton']:>12.4f} {str(r['certified']):>10}",
          flush=True)
print(f"\n[certify_all] DONE  ({time.time()-t0:.0f}s)", flush=True)
