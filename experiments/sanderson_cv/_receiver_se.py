import os, sys, math, time
os.environ["SADDLE_DTYPE"]="float32"; os.environ.setdefault("GPUREC_MEMORY_POLICY_RESERVE_GIB","0")
RW="/home/enzo/Documents/git/gpurec/agent-worktrees/receiver-weights-hvp"; sys.path.insert(0,RW)
import torch
import converge_bounded_joint_archaea as drv
from gpurec import GeneReconModel, SolverOptions
DEV=drv.DEV; lam=0.03; t0=time.time()

so=SolverOptions(**drv._CV_SO); so.validate()
paths=drv._archaea_families(0)
model=GeneReconModel(drv._SP_TREE,[str(p) for p in paths],mode="specieswise",device=DEV,solver_options=so)
bs=model.batch_statics; S=int(model.species_helpers["S"]); theta_numel=3*S
sp_parent=model.species_helpers["sp_parent"].to(DEV).long().reshape(-1)
child=(sp_parent>=0).nonzero(as_tuple=True)[0].contiguous(); parent=sp_parent[child].contiguous()
lap=drv.make_tree_lap(child,parent,lam)
rp=drv.make_reparam(S, floor_frac=1e-2)
print(f"[build] {len(paths)} families S={S} ({time.time()-t0:.0f}s)", flush=True)

B=torch.load(f"{RW}/experiments/sanderson_cv/runs/bounded_joint_archaea_full_fp32.pt",weights_only=False,map_location=DEV)
theta=B["theta"].to(DEV).double(); beta=B["beta"].to(DEV).double(); wB=B["w"].to(DEV).double().flatten()
Hb=drv.make_Hbeta(bs, theta.reshape(S,3), beta, rp, lap, theta_numel, S)

# Conditional receiver-weight block H_bb (theta fixed) in BETA space, gauge-projected:
#   column k = proj_alpha( Hb([0; proj_alpha(e_k)])_[beta-block] )
print(f"[se] building dense {S}x{S} conditional weight Hessian via {S} joint HVPs...", flush=True)
M=torch.zeros(S,S,device=DEV,dtype=torch.float64)
for k in range(S):
    ek=torch.zeros(S,device=DEV,dtype=torch.float64); ek[k]=1.0
    v=torch.cat([torch.zeros(theta_numel,device=DEV,dtype=torch.float64), drv.proj_alpha(ek)])
    Hv=Hb(v)
    M[:,k]=drv.proj_alpha(Hv[theta_numel:].double())
    if (k+1)%20==0: print(f"    {k+1}/{S}  ({time.time()-t0:.0f}s)", flush=True)
M=0.5*(M+M.T)

evals,evecs=torch.linalg.eigh(M)                         # ascending; one ~0 = gauge null
tol=evals.abs().max()*1e-8
print(f"\n[se] weight-block spectrum: min {float(evals[0]):.3e}  (gauge~0: {float(evals[1]):.3e})  "
      f"max {float(evals[-1]):.3e}", flush=True)
n_neg=int((evals < -tol).sum())
if n_neg: print(f"  WARNING: {n_neg} negative eigenvalue(s) (not exactly stationary, |Pg|=6.4) -> clamped", flush=True)
# gauge-fixed pseudo-inverse: invert eigenvalues clearly > 0 (drop the gauge null + any tiny/neg)
thr=evals[-1]*1e-6
keep=evals > max(thr, tol)
inv=torch.zeros_like(evals); inv[keep]=1.0/evals[keep]
Sigma_beta=(evecs*inv)@evecs.T                           # gauge-fixed covariance of beta
J=rp["dw_jac"](beta)                                     # dw/dbeta [S,S]
Sigma_w=J@Sigma_beta@J.T
se_w=Sigma_w.diagonal().clamp_min(0.0).sqrt()
se_beta=Sigma_beta.diagonal().clamp_min(0.0).sqrt()
uni=1.0/S

rec=dict(theta=theta.cpu(), beta=beta.cpu(), w=wB.cpu(), se_w=se_w.cpu(), se_beta=se_beta.cpu(),
         H_beta_block=M.cpu(), eval_min=float(evals[0]), eval_gauge=float(evals[1]),
         n_neg=n_neg, conditional=True, lam=lam, box=(0.05,2.0), S=S, n_families=len(paths),
         note="conditional (theta-fixed) receiver-weight SEs; fp32; |Pg|_fit=6.4")
drv.__dict__  # keep import
out=f"{RW}/experiments/sanderson_cv/runs/receiver_se_full_fp32.pt"
torch.save(rec, out)

print(f"\n=== Per-recipient weights with uncertainty (uniform={uni:.4f}) ===")
order=torch.argsort(wB, descending=True)
print(f"  {'rank':>4} {'species':>8} {'w':>9} {'se_w':>9} {'w/uni':>6} {'z=(w-uni)/se':>12}")
for r,i in enumerate(order[:10].tolist()):
    z=(float(wB[i])-uni)/max(float(se_w[i]),1e-12)
    print(f"  {r+1:>4} {('#'+str(i)):>8} {float(wB[i]):>9.4f} {float(se_w[i]):>9.4f} {float(wB[i])/uni:>5.1f}x {z:>+11.1f}")
print("  ... floor species (smallest w):")
for i in order[-4:].tolist():
    z=(float(wB[i])-uni)/max(float(se_w[i]),1e-12)
    print(f"       {('#'+str(i)):>8} {float(wB[i]):>9.4f} {float(se_w[i]):>9.4f} {float(wB[i])/uni:>5.2f}x {z:>+11.1f}")
n_sig=int(((wB-uni)/se_w.clamp_min(1e-12) > 2).sum()); n_sig_lo=int(((uni-wB)/se_w.clamp_min(1e-12) > 2).sum())
print(f"\n  species significantly ABOVE uniform (z>2): {n_sig}/{S};  significantly BELOW: {n_sig_lo}/{S}")
print(f"  se_w range [{float(se_w.min()):.2e}, {float(se_w.max()):.2e}]")
print(f"[saved] {out}  ({time.time()-t0:.0f}s)", flush=True)
