"""Marginal (Schur) recipient-weight standard errors (GPT 5.5 Pro red flag #3).

The paper's se_w invert only the receiver block H_aa (conditional on theta). The MARGINAL SE accounts
for rate uncertainty: it is the alpha-alpha block of the inverse of the FULL free-subspace joint Hessian,
i.e. (H_aa - H_a,thf H_thf,thf^-1 H_thf,a)^-1 after gauge + active-set projection. Cheapest exact route:
form the dense free-subspace joint Hessian over [free theta (inactive); beta], invert (gauge-fixed
pseudo-inverse), take the beta block, delta-method to w. Compare to the conditional se_w. The dense block
also yields the exact free-subspace spectrum (lam_min) as a bonus.

Env: SADDLE_DTYPE(float32). ~ (free_theta + S) HVP applies (~1.5-2h on the 4090).
"""
import os, sys, time
os.environ["SADDLE_DTYPE"] = "float32"; os.environ.setdefault("GPUREC_MEMORY_POLICY_RESERVE_GIB", "0")
os.environ.setdefault("GPUREC_MEMORY_POLICY_FRACTION", "0.9")
RW = "/home/enzo/Documents/git/gpurec/agent-worktrees/receiver-weights-hvp"; sys.path.insert(0, RW)
import torch
import converge_bounded_joint_archaea as drv
from gpurec import GeneReconModel, SolverOptions
DEV = drv.DEV; lam = 0.03; t0 = time.time()

so = SolverOptions(**drv._CV_SO); so.validate()
paths = drv._archaea_families(0)
model = GeneReconModel(drv._SP_TREE, [str(p) for p in paths], mode="specieswise", device=DEV, solver_options=so)
bs = model.batch_statics; S = int(model.species_helpers["S"]); tn = 3 * S
sp_parent = model.species_helpers["sp_parent"].to(DEV).long().reshape(-1)
child = (sp_parent >= 0).nonzero(as_tuple=True)[0].contiguous(); parent = sp_parent[child].contiguous()
lap = drv.make_tree_lap(child, parent, lam)
rp = drv.make_reparam(S, floor_frac=1e-2)
lo, hi = drv.log2_rate_bounds(0.05, 2.0)
print(f"[build] {len(paths)} fam S={S} ({time.time()-t0:.0f}s)", flush=True)

B = torch.load(f"{RW}/experiments/sanderson_cv/runs/bounded_joint_archaea_full_fp32.pt", weights_only=False, map_location=DEV)
theta = B["theta"].to(DEV).double(); beta = B["beta"].to(DEV).double(); alpha = rp["alpha_of_beta"](beta)

vg = drv.make_value_and_grad(bs, beta, theta_shape=(S, 3), optimize_receiver=True, tree_penalty=(lam, sp_parent))
g_z = vg(torch.cat([theta.reshape(-1), alpha]))[1].double()
free_mask = (~drv.binding_theta_mask(theta.reshape(-1), g_z[:tn], lo, hi))
free_theta = free_mask.nonzero(as_tuple=True)[0].tolist()
nft = len(free_theta)
print(f"[free] {nft} free theta + {S} beta = {nft+S} cols; active theta {tn-nft}/{tn}", flush=True)

Hb = drv.make_Hbeta(bs, theta.reshape(S, 3), beta, rp, lap, tn, S)   # joint HVP in [theta; beta]
p = tn + S
# free-subspace basis columns: free-theta unit vectors, then beta unit vectors (gauge handled by pinv)
cols = []
for j in free_theta:
    e = torch.zeros(p, device=DEV, dtype=torch.float64); e[j] = 1.0; cols.append(e)
for k in range(S):
    e = torch.zeros(p, device=DEV, dtype=torch.float64); e[tn + k] = 1.0; cols.append(e)
Bmat = torch.stack(cols, dim=1)                                      # [p, ncol]
ncol = Bmat.shape[1]
print(f"[hvp] forming dense {ncol}x{ncol} free Hessian ({ncol} HVP applies)...", flush=True)
HB = torch.zeros(p, ncol, device=DEV, dtype=torch.float64)
for i in range(ncol):
    HB[:, i] = Hb(cols[i]).double()
    if (i + 1) % 40 == 0:
        torch.cuda.empty_cache()
        print(f"    {i+1}/{ncol}  ({time.time()-t0:.0f}s)", flush=True)
M = Bmat.T @ HB; M = 0.5 * (M + M.T)

evals, evecs = torch.linalg.eigh(M)
print(f"\n[spectrum] free-subspace lam_min={float(evals[0]):.3e}  (2nd {float(evals[1]):.3e})  "
      f"max {float(evals[-1]):.3e}; n_neg={int((evals< -1e-6*evals[-1]).sum())}", flush=True)
thr = evals[-1] * 1e-7
keep = evals > thr
inv = torch.zeros_like(evals); inv[keep] = 1.0 / evals[keep]
Sigma = (evecs * inv) @ evecs.T                                     # gauge-fixed pseudo-inverse [ncol,ncol]
Sig_bb = Sigma[nft:, nft:]                                          # marginal beta covariance [S,S]
J = rp["dw_jac"](beta)                                              # dw/dbeta [S,S]
Sig_w = J @ Sig_bb @ J.T
se_w_marg = Sig_w.diagonal().clamp_min(0.0).sqrt()

cond = torch.load(f"{RW}/experiments/sanderson_cv/runs/receiver_se_full_fp32.pt", weights_only=False, map_location=DEV)
se_w_cond = cond["se_w"].to(DEV).double(); wB = B["w"].to(DEV).double().flatten(); uni = 1.0 / S
ratio = (se_w_marg / se_w_cond.clamp_min(1e-12))
print(f"\n=== MARGINAL vs CONDITIONAL recipient SE ===", flush=True)
print(f"  marginal se_w range [{float(se_w_marg.min()):.2e},{float(se_w_marg.max()):.2e}]; "
      f"conditional [{float(se_w_cond.min()):.2e},{float(se_w_cond.max()):.2e}]", flush=True)
print(f"  marginal/conditional ratio: median {float(ratio.median()):.2f}, max {float(ratio.max()):.2f}", flush=True)
order = torch.argsort(wB, descending=True)
print(f"  top sinks: w, se_cond, se_marg, z_marg=(w-uni)/se_marg", flush=True)
for i in order[:6].tolist():
    z = (float(wB[i]) - uni) / max(float(se_w_marg[i]), 1e-12)
    print(f"    #{i:<3d} w={float(wB[i]):.4f} se_cond={float(se_w_cond[i]):.4f} se_marg={float(se_w_marg[i]):.4f} z_marg={z:+.1f}", flush=True)
n_sig = int(((wB - uni) / se_w_marg.clamp_min(1e-12) > 2).sum())
print(f"  sinks still significant (z_marg>2): {n_sig}/{S} above uniform", flush=True)
torch.save(dict(se_w_marg=se_w_marg.cpu(), se_w_cond=se_w_cond.cpu(), Sigma_bb=Sig_bb.cpu(),
                lam_min_free=float(evals[0]), evals=evals.cpu(), nft=nft, S=S),
           f"{RW}/experiments/sanderson_cv/runs/marginal_se.pt")
print(f"[saved] runs/marginal_se.pt  ({time.time()-t0:.0f}s)", flush=True)
