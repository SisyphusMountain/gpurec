import os, sys, time
os.environ["SADDLE_DTYPE"]="float32"; os.environ.setdefault("GPUREC_MEMORY_POLICY_RESERVE_GIB","0")
RW="/home/enzo/Documents/git/gpurec/agent-worktrees/receiver-weights-hvp"; sys.path.insert(0,RW)
import torch
import converge_bounded_joint_archaea as drv
from gpurec import GeneReconModel, SolverOptions
from gpurec.optim.receiver_curvature import certify_joint_min
DEV=drv.DEV; lam=0.03; t0=time.time()

so=SolverOptions(**drv._CV_SO); so.validate()
paths=drv._archaea_families(0)
model=GeneReconModel(drv._SP_TREE,[str(p) for p in paths],mode="specieswise",device=DEV,solver_options=so)
bs=model.batch_statics; S=int(model.species_helpers["S"]); theta_numel=3*S
sp_parent=model.species_helpers["sp_parent"].to(DEV).long().reshape(-1)
child=(sp_parent>=0).nonzero(as_tuple=True)[0].contiguous(); parent=sp_parent[child].contiguous()
lap=drv.make_tree_lap(child,parent,lam)
rp=drv.make_reparam(S, floor_frac=1e-2)
lo,hi=drv.log2_rate_bounds(0.05,2.0)
print(f"[build] {len(paths)} fam S={S} ({time.time()-t0:.0f}s)", flush=True)

B=torch.load(f"{RW}/experiments/sanderson_cv/runs/bounded_joint_archaea_full_fp32.pt",weights_only=False,map_location=DEV)
theta=B["theta"].to(DEV).double(); beta=B["beta"].to(DEV).double(); alpha=rp["alpha_of_beta"](beta)

# active set from the gradient (which theta coords are pinned at the box)
vg=drv.make_value_and_grad(bs, beta, theta_shape=(S,3), optimize_receiver=True, tree_penalty=(lam, sp_parent))
g_z=vg(torch.cat([theta.reshape(-1), alpha]))[1].double()
free_mask=(~drv.binding_theta_mask(theta.reshape(-1), g_z[:theta_numel], lo, hi)).double()
n_active=int((free_mask<0.5).sum())
P=drv.make_P_free(free_mask, theta_numel)
print(f"[cert] active theta {n_active}/{theta_numel}; free joint dim {theta_numel-n_active}+{S} "
      f"= {theta_numel-n_active+S}  ({time.time()-t0:.0f}s)", flush=True)

# EXACT multibatch joint HVP (NOT the GN make_Hbeta) -> a real saddle would show as lam_min<0
Ha=drv.build_joint_hvp_multibatch(bs, theta.reshape(S,3), alpha, lap, theta_numel, S)
print(f"[cert] running deflated-gauge Lanczos m=200 on the EXACT joint HVP...", flush=True)
cert=certify_joint_min(bs, theta, alpha, hvp=Ha, theta_numel=theta_numel, S=S, proj=P, m=200, verbose=True)
print(f"\n[RESULT] lam_min_free={cert['lam_min_gauge']:+.6e}  ritz_resid={cert['ritz_resid']:.2e}  "
      f"leak={cert['leak']:.2e}  PD={cert['pd']}  ({time.time()-t0:.0f}s)", flush=True)
torch.save(dict(lam_min_free=cert['lam_min_gauge'], ritz_resid=cert['ritz_resid'], leak=cert['leak'],
                pd=bool(cert['pd']), n_active=n_active, m=200, dtype="float32",
                note="EXACT joint HVP, deflated-gauge Lanczos, at |Pg|_fit=6.4"),
           f"{RW}/experiments/sanderson_cv/runs/joint_cert_full_fp32.pt")
print("[saved] runs/joint_cert_full_fp32.pt", flush=True)
