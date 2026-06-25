"""Exact-vs-weak D-L non-identifiability test (the mathematician's diagnostics, executed).

Separates DATA curvature from PRIOR curvature on the TURNOVER subspace, prior-free, using the
Laplacian-null trick: the GLOBAL turnover mode (every species delta_thetaD=delta_thetaL=+1, delta_thetaT=0)
lies in the null space of the tree Laplacian L (it is constant across species), so lam*L contributes
EXACTLY 0 to its curvature. Whatever curvature it has is pure data information.

Tests at a converged loose-box theta*:
  (1) q_data / q_F on v_glob_turn  -- must be ~equal (prior null on the constant mode). The value is the
      prior-free DATA curvature of global turnover. ~0 => exact non-id; >0 => weak id.
  (2) q_data on v_glob_net (D=+1,L=-1)  -- the stiff reference (net growth). Anisotropy = net/turn.
  (3) family SCALING of q_data(v_glob_turn) at m=64,128,256 families (same theta*). Linear in m =>
      intrinsic (Fisher info adds over families); flat/noise => exact non-id.
  (4) full TURNOVER-subspace spectrum: lam_min of U^T H_data U vs U^T H_F U (U = per-species turnover
      basis, S cols). Does the prior do the lifting, or the data?

fp64 for the small curvatures. 256-fam archaea (cheap HVP).
"""
import os, sys, glob, math
os.environ["SADDLE_DTYPE"] = "float64"
os.environ.setdefault("GPUREC_MEMORY_POLICY_RESERVE_GIB", "0")
from pathlib import Path
import numpy as np
import torch

WT = Path("/home/enzo/Documents/git/gpurec/agent-worktrees/kernel-bench-mapcv-merge")
sys.path.insert(0, str(WT / "experiments/sanderson_cv"))
import saddle_escape as se
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions

DEV = "cuda"; DT = torch.float64
ARCH = "archaea"; LAM = 0.03
CONV = WT / "experiments/sanderson_cv/runs/cv_archaea_n256_box_1e-4_16/refit_lam0.03_fp64_converged.pt"


def build(n):
    so = SolverOptions(**{**_CV_SO, "pi_iters": 128, "neumann_terms": 64}); so.validate()
    paths = DATASETS[ARCH]["families"](n)
    m = GeneReconModel(str(DATASETS[ARCH]["species_tree"]), [str(x) for x in paths],
                       mode="specieswise", device=DEV, solver_options=so,
                       clade_budget=80000, family_chunk_size=300)
    S = int(m.species_helpers["S"]); sp = m.species_helpers["sp_parent"].detach().reshape(-1).long()
    child = (sp >= 0).nonzero(as_tuple=True)[0].to(DEV); par = sp[child].to(DEV)
    return dict(bs=m.batch_statics, rw=m.receiver_weights.detach(), S=S, p=3 * S,
                child=child, par=par, n=len(paths))


def modes(S, p):
    """Global turnover (D=L=+1 all species) and global net (D=+1,L=-1), normalized; T=0."""
    vt = torch.zeros(S, 3, device=DEV, dtype=DT); vt[:, 0] = 1.0; vt[:, 1] = 1.0   # turnover
    vn = torch.zeros(S, 3, device=DEV, dtype=DT); vn[:, 0] = 1.0; vn[:, 1] = -1.0  # net
    vt = vt.reshape(-1) / vt.norm(); vn = vn.reshape(-1) / vn.norm()
    return vt, vn


M = build(256)
S, p = M["S"], M["p"]
th = torch.load(CONV, weights_only=False)["theta"].to(DEV).to(DT).reshape(S, 3)
lap0 = se.make_lap(M["child"], M["par"], 0.0)      # H_data only
lapF = se.make_lap(M["child"], M["par"], LAM)      # H_data + lam L
Hd = se.build_hvp_once(M["bs"], th, M["rw"], lap0, p)
HF = se.build_hvp_once(M["bs"], th, M["rw"], lapF, p)
vt, vn = modes(S, p)


def quad(Av, v): return float(torch.dot(v, Av(v)))


print(f"=== D-L identifiability @ converged 256-fam loose-box lam={LAM} (fp64) ===", flush=True)
qt_d, qt_F = quad(Hd, vt), quad(HF, vt)
qn_d, qn_F = quad(Hd, vn), quad(HF, vn)
print(f"[1] GLOBAL TURNOVER mode (D=L=+1):  data q={qt_d:+.6e}   data+prior q={qt_F:+.6e}   "
      f"prior contrib={qt_F-qt_d:+.2e}", flush=True)
print(f"    -> prior contributes ~0 (Laplacian-null on constant mode): {abs(qt_F-qt_d):.2e}", flush=True)
print(f"[2] GLOBAL NET mode (D=+1,L=-1):    data q={qn_d:+.6e}   data+prior q={qn_F:+.6e}", flush=True)
print(f"    -> data anisotropy net/turnover = {qn_d/qt_d:.1f}x", flush=True)

# [3] family scaling of the prior-free turnover curvature (same theta*)
print("[3] family scaling of DATA turnover curvature q_data(v_glob_turn):", flush=True)
print(f"      m=256 : {qt_d:+.6e}", flush=True)
for n in (64, 128):
    Mn = build(n)
    Hdn = se.build_hvp_once(Mn["bs"], th, Mn["rw"], se.make_lap(Mn["child"], Mn["par"], 0.0), p)
    vtn, _ = modes(Mn["S"], Mn["p"])
    q = quad(Hdn, vtn)
    print(f"      m={Mn['n']:>3d} : {q:+.6e}   (per-family {q/Mn['n']:+.3e})", flush=True)
print(f"      per-family @256: {qt_d/256:+.3e}", flush=True)

# [4] full turnover-subspace spectrum: lam_min of U^T H_data U vs U^T H_F U
print("[4] turnover-subspace spectrum (U = per-species D=L basis, S cols):", flush=True)
U = torch.zeros(p, S, device=DEV, dtype=DT)
for s in range(S):
    U[3 * s + 0, s] = 1.0; U[3 * s + 1, s] = 1.0
U = U / U[:, 0].norm()  # each column same norm
Md = torch.stack([Hd(U[:, j]) for j in range(S)], dim=1)   # H_data U  (S HVPs)
MF = torch.stack([HF(U[:, j]) for j in range(S)], dim=1)   # H_F U
Ad = (U.T @ Md); Ad = 0.5 * (Ad + Ad.T)
AF = (U.T @ MF); AF = 0.5 * (AF + AF.T)
wd = torch.linalg.eigvalsh(Ad); wF = torch.linalg.eigvalsh(AF)
print(f"    U^T H_data U : lam_min={float(wd[0]):+.4e}  lam_max={float(wd[-1]):+.4e}  "
      f"#(<1e-3)={int((wd<1e-3).sum())}/{S}", flush=True)
print(f"    U^T H_F    U : lam_min={float(wF[0]):+.4e}  lam_max={float(wF[-1]):+.4e}  "
      f"#(<1e-3)={int((wF<1e-3).sum())}/{S}", flush=True)
print(f"    prior lift of subspace lam_min: {float(wd[0]):+.4e} -> {float(wF[0]):+.4e}", flush=True)
print("=== done ===", flush=True)
