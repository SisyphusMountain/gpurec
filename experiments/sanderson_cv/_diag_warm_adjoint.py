"""Verify + measure adjoint warm-starting. (1) CORRECTNESS: warm-started gradient must match cold at
high neumann. (2) BENEFIT: warm-starting from the PREVIOUS optimizer step's adjoint -- does it reach the
same gradient accuracy with fewer neumann terms? Realistic: theta_new = theta_prev + a Newton-sized step."""
import os, sys, torch
os.environ["GPUREC_WARM_ADJOINT"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.optimization import clamp_log_rate_
DEV = "cuda"; DT = torch.float32; PI = 64
d = torch.load("/tmp/bench_genewise_hogenom_adaptive_countstop_theta.pt", map_location=DEV, weights_only=False)
theta_all = d["theta"].to(DEV)
sel = torch.arange(24, device=DEV)
sub = [DATASETS["hogenom"]["families"](None)[i] for i in sel.tolist()]
m = GeneReconModel(str(DATASETS["hogenom"]["species_tree"]), [str(x) for x in sub], mode="genewise", device=DEV,
                   solver_options=SolverOptions(**{**_CV_SO, "pi_iters": PI, "neumann_terms": PI}), clade_budget=80000)
m.receiver_weights.requires_grad_(False)
print(f"{sel.numel()} families, {len(m.batch_statics)} batch(es)", flush=True)

def set_neu(neu): m.solver_options = SolverOptions(**{**_CV_SO, "pi_iters": PI, "neumann_terms": neu})
def clear():      [setattr(s, "warm_v", {}) for s in m.batch_statics]
def save():       return [{k: v.clone() for k, v in (s.warm_v or {}).items()} for s in m.batch_statics]
def load(w):      [setattr(s, "warm_v", {k: v.clone() for k, v in wi.items()}) for s, wi in zip(m.batch_statics, w)]
def grad(th):     return m.genewise_loss_vector_and_grad(theta=th, need_grad=True)[1].to(DT)

theta_prev = theta_all.index_select(0, sel).to(DT)
torch.manual_seed(0)
theta_new = theta_prev + 0.05 * torch.randn_like(theta_prev)        # a realistic Newton-sized step
clamp_log_rate_(theta_new, min_rate=1e-6, max_rate=2)               # the optimizer always clamps rates <= 2

# warm up at theta_prev -> converged adjoint cached
clear(); set_neu(128); grad(theta_prev); warm_prev = save()
# reference: true gradient at theta_new (cold, very high neumann)
clear(); set_neu(512); g_ref = grad(theta_new)

def safe_err(g):
    if g is None or bool(torch.isnan(g).any()):
        return float("nan")
    return float((g - g_ref).abs().max())
def try_grad(th):
    try:
        return grad(th)
    except Exception:
        return None

print(f"\n[BENEFIT] gradient error vs g_ref (cold vs warm-from-prev-step), |theta step|inf="
      f"{float((theta_new-theta_prev).abs().max()):.3f}:", flush=True)
print(f"  {'neumann':>8} {'cold_err':>12} {'warm_err':>12}", flush=True)
for neu in [1, 2, 4, 8, 16, 32, 64, 128]:
    clear(); set_neu(neu); ec = safe_err(try_grad(theta_new))
    load(warm_prev); set_neu(neu); ew = safe_err(try_grad(theta_new))
    print(f"  {neu:>8} {ec:>12.2e} {ew:>12.2e}", flush=True)
