"""A100 fp64 polish of a Sanderson-CV checkpoint, off a portable capture (no model build / no Rust
preprocessor needed -- imports only the cluster-safe gpurec.optim.* submodules, NOT GeneReconModel).

Answers the open question: does fp64 drive the penalized gradient below the fp32 precision floor
(~0.36 locally), and is the polished theta a certified local minimum?

Stages, all on the full-data GBM-penalized objective  F = sum_i NLL_i + (lam/2) sum_edges ||t_c-t_p||^2:
  1. continue L-BFGS in fp32 from the checkpoint -> record the fp32 ||g|| floor
  2. continue L-BFGS in fp64 from there         -> record the fp64 ||g|| floor (does it break fp32's?)
  3. exact multi-batch HVP (H = sum_b H_b + lam*L), Lanczos min-eig in fp64 -> PD certificate

Env: CAP, THETA (input .pt paths), LAM (default 1.0), OUT (output .pt), CERT_M (Lanczos m, default 120).
"""
import os, time
import numpy as np
import torch
from scipy.optimize import minimize

from gpurec.optim.value_and_grad import make_value_and_grad, forward_solve
from gpurec.optim.hvp_exact import make_exact_hvp
from gpurec.optim.cg import lanczos_min_eigpair

CAP = os.environ["CAP"]; THETA = os.environ["THETA"]; OUT = os.environ.get("OUT", "polish_out.pt")
LAM = float(os.environ.get("LAM", "1.0")); CERT_M = int(os.environ.get("CERT_M", "120"))

dev = "cuda"
print(f"=== A100 polish: {torch.cuda.get_device_name(0)}  lam={LAM}  cert_m={CERT_M} ===", flush=True)
cap = torch.load(CAP, map_location=dev, weights_only=False)
bs = cap["batch_statics"]; rw = cap["rw"].to(dev); sp = cap["sp_parent"].to(dev).long(); S = int(cap["S"])
theta0 = torch.load(THETA, map_location=dev, weights_only=False)["theta"].to(dev)
p = S * 3
child = (sp >= 0).nonzero(as_tuple=True)[0].contiguous(); par = sp[child].contiguous()
print(f"loaded: S={S}, {len(bs)} batches, theta {tuple(theta0.shape)}", flush=True)


def lbfgs_polish(theta_init, dtype, maxiter=300, tag=""):
    f = make_value_and_grad(bs, rw, theta_shape=(S, 3), tree_penalty=(LAM, sp))
    n = {"k": 0}; t0 = time.time()
    def fun(x):
        n["k"] += 1
        x = torch.tensor(x, device=dev, dtype=dtype)
        l, g, _, _ = f(x)
        return float(l), g.double().cpu().numpy()
    x0 = theta_init.reshape(-1).to(torch.float64).cpu().numpy()
    l0, g0 = fun(x0)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=None,
                   options=dict(maxiter=maxiter, maxfun=maxiter * 2, maxcor=50, ftol=1e-16, gtol=1e-12))
    gN = float(np.linalg.norm(res.jac))
    print(f"  [{tag} {str(dtype).split('.')[-1]}] loss {l0:.4f} -> {res.fun:.4f}   ||g|| {np.linalg.norm(g0):.4e} -> {gN:.4e}"
          f"   nit={res.nit} nfev={n['k']} ({time.time()-t0:.0f}s)  exit: {res.message}", flush=True)
    return torch.tensor(res.x, device=dev, dtype=torch.float32).reshape(S, 3), res.fun, gN


def lap_apply(u, lam):
    u = u.reshape(-1, 3); out = torch.zeros_like(u)
    d = u.index_select(0, child) - u.index_select(0, par)
    out.index_add_(0, child, d); out.index_add_(0, par, -d)
    return (lam * out).reshape(-1)


def certify(theta, m):
    """Smallest eigenvalue of H + lam*L via ANALYTIC exact-HVP summed over batches.
    fp32 HVP + fp64 Lanczos vectors (matches the verified spectrum_min; analytic HVP needs no fp64 --
    no FD cancellation -- and fp32 keeps the 5 resident batch caches well within 80 GB)."""
    t0 = time.time()
    th = theta.float()          # fp32 HVP
    rwf = rw.float()
    hvps = []
    for st in bs:
        _l, sv = forward_solve([st], th, rwf)
        hvps.append(make_exact_hvp([st], th, rwf, sv))
    def Av(v):
        acc = torch.zeros(p, dtype=torch.float64, device=dev)
        vf = v.float()
        for h in hvps:
            acc += h(vf).double()
        return acc + lap_apply(v.double(), LAM)
    lam_min, vmin = lanczos_min_eigpair(Av, p, m=m, seed=0)
    resid = float((Av(vmin) - lam_min * vmin).norm())
    print(f"  [certify] lam_min={lam_min:+.6e}  ritz_resid={resid:.3e}  m={m}  ({time.time()-t0:.0f}s)", flush=True)
    return lam_min, resid


t_all = time.time()
print("\n[1] fp32 L-BFGS polish from checkpoint:", flush=True)
th32, f32, g32 = lbfgs_polish(theta0, torch.float32, tag="fp32")
print("\n[2] fp64 L-BFGS polish (continue from fp32 result):", flush=True)
th64, f64, g64 = lbfgs_polish(th32, torch.float64, tag="fp64")
print(f"\n  fp32 floor ||g||={g32:.4e}  ->  fp64 floor ||g||={g64:.4e}   "
      f"(fp64 {'BROKE the fp32 floor' if g64 < 0.5*g32 else 'did not improve much'})  "
      f"loss {f32:.4f} -> {f64:.4f} (gain {f32-f64:.4f})", flush=True)
print("\n[3] PD certificate (analytic exact-HVP, fp64) at the fp64-polished theta:", flush=True)
lam_min, resid = certify(th64, CERT_M)
pd = lam_min > 0 and resid < 0.1 * max(1.0, abs(lam_min))
torch.save({"theta": th64.cpu(), "lam": LAM, "fp32_gnorm": g32, "fp64_gnorm": g64,
            "loss_fp32": f32, "loss_fp64": f64, "lam_min": lam_min, "ritz_resid": resid,
            "certified_pd": bool(pd)}, OUT)
print(f"\n=== DONE t={time.time()-t_all:.0f}s  PD={pd}  saved {OUT} ===", flush=True)
