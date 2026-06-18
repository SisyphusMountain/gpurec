"""Instrument the per-step bounded-Newton trajectory of the WORST-gradient hogenom families."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds
DEV="cuda"; DT=torch.float32; PI=int(os.environ.get("PI","64")); K=int(os.environ.get("K","6"))
MINR,MAXR=1e-6,2.0; FD=1e-2; MU=1e-2; TRUST=2.0
LO,HI=log2_rate_bounds(MINR,MAXR)

d=torch.load("/tmp/bench_genewise_hogenom1055_recert_pi64_theta.pt",map_location=DEV,weights_only=False)
theta_full=d["theta"].to(DEV).to(DT); gpf=d["gpf"].to(DEV)
paths=DATASETS["hogenom"]["families"](None)
worst=gpf.argsort(descending=True)[:K]                        # the K worst-gradient families
print("worst families (global idx, |g|@pi64, theta):")
for i in worst.tolist(): print(f"  {i:5d}: |g|={float(gpf[i]):.2f} theta={[round(float(x),2) for x in theta_full[i]]}")
sub_paths=[paths[i] for i in worst.tolist()]
m=GeneReconModel(str(DATASETS["hogenom"]["species_tree"]),[str(x) for x in sub_paths],mode="genewise",
                 device=DEV,solver_options=SolverOptions(**{**_CV_SO,"pi_iters":PI,"neumann_terms":PI}),clade_budget=80000)
m.receiver_weights.requires_grad_(False)
def lg(th):
    lv,g,_=m.genewise_loss_vector_and_grad(theta=th,need_grad=True); return lv.to(DT),g.to(DT)
theta=theta_full.index_select(0,worst).clone(); clamp_log_rate_(theta,min_rate=MINR,max_rate=MAXR)
print(f"\nbounded projected Newton, pi={PI}, theta box [{LO:.2f},{HI:.2f}], TRUST={TRUST}, MU={MU}\n")
for it in range(16):
    lv,g=lg(theta)
    H=torch.zeros(K,3,3,device=DEV,dtype=DT)
    for j in range(3):
        tp=theta.clone();tp[:,j]+=FD;_,gp=lg(tp); tm=theta.clone();tm[:,j]-=FD;_,gm=lg(tm)
        H[:,:,j]=(gp-gm)/(2*FD)
    H=0.5*(H+H.transpose(1,2)); e,V=torch.linalg.eigh(H)
    Hd=V@torch.diag_embed(e.clamp(min=MU))@V.transpose(1,2)
    # ACTIVE-SET reduced Hessian: a coord is FIXED if at a bound with the gradient pushing further out
    # (KKT-binding). Solve Newton only on the FREE coords (zero fixed rows/cols, identity on fixed diag).
    fixed=((theta>=HI-1e-6)&(g<0))|((theta<=LO+1e-6)&(g>0))   # [K,3]
    free=(~fixed).float()
    g_red=g*free
    Hred=Hd*free.unsqueeze(1)*free.unsqueeze(2)+torch.diag_embed(1.0-free)
    delta=-torch.linalg.solve(Hred,g_red.unsqueeze(-1)).squeeze(-1)
    dn=delta.norm(dim=1,keepdim=True); delta_c=delta*(TRUST/dn.clamp(min=TRUST))
    pg=project_rate_gradient_(theta,g.clone(),min_rate=MINR,max_rate=MAXR)
    if it<8 or it==15:
        # show family 0 (the worst) in detail
        f=0; atb=[("hi" if float(theta[f,j])>=HI-1e-6 else "lo" if float(theta[f,j])<=LO+1e-6 else "-") for j in range(3)]
        print(f"it{it:2d} f0: L={float(lv[f]):.3f} |g|={float(g[f].abs().max()):.2e} |Pg|={float(pg[f].abs().max()):.2e} "
              f"g={[round(float(x),1) for x in g[f]]} eig={[round(float(x),2) for x in e[f]]} "
              f"|raw_delta|={float(dn[f]):.1e} |step|={float(delta_c[f].norm()):.2e} bound={atb} th={[round(float(x),2) for x in theta[f]]}")
    theta=theta+delta_c; clamp_log_rate_(theta,min_rate=MINR,max_rate=MAXR)
_,g=lg(theta); pg=project_rate_gradient_(theta,g.clone(),min_rate=MINR,max_rate=MAXR)
print(f"\nfinal |Pg|max over {K} families = {float(pg.abs().amax(dim=1).max()):.2e}")
