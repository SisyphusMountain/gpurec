"""Subspace-deflation PD certificate for a polished theta, on the MULTI-BATCH GBM-penalized operator.

The plain Lanczos min-eig is residual-dominated here (bottom of H+lam*L is a tight near-degenerate
cluster -- fp64 m=160 gave Ritz resid 0.19 >> the eigenvalue). Adapts the prior-art subspace
deflation (a100_subspace_deflate.py): Lanczos captures the bottom INVARIANT SUBSPACE V (it can't
resolve individual vectors in a cluster, but it spans them); then an EXACT Rayleigh-Ritz on span(V)
-- H_k = V^T (H+lam L) V via exact HVPs -- resolves the cluster and yields a TRUSTWORTHY certified
bottom eigenvalue with a real (small) residual.

Operator: H = sum_b exact_hvp_b + lam*L  (multi-batch NLL Hessian + tree-Laplacian penalty), fp64.
Env: CAP, THETA, LAM, DEFLATE_M (Lanczos depth), DEFLATE_K (subspace dim), NEWTON_TANGENT_SELF_ITERS.
"""
import os, time
os.environ.setdefault("NEWTON_TANGENT_SELF_ITERS", "64")
import numpy as np
import torch
from scipy.linalg import eigh_tridiagonal
from gpurec.optim.value_and_grad import forward_solve
from gpurec.optim.hvp_exact import make_exact_hvp

DEV = "cuda"
CAP = os.environ["CAP"]; THETA = os.environ["THETA"]; LAM = float(os.environ.get("LAM", "1.0"))
M = int(os.environ.get("DEFLATE_M", "200")); K = int(os.environ.get("DEFLATE_K", "16"))

print(f"=== subspace-deflation cert: {torch.cuda.get_device_name(0)}  lam={LAM}  M={M} K={K} "
      f"tangent={os.environ['NEWTON_TANGENT_SELF_ITERS']}  fp64 ===", flush=True)
cap = torch.load(CAP, map_location=DEV, weights_only=False)
bs = cap["batch_statics"]; rw = cap["rw"].to(DEV).double(); sp = cap["sp_parent"].to(DEV).long(); S = int(cap["S"])
theta = torch.load(THETA, map_location=DEV, weights_only=False)["theta"].to(DEV).double()
p = 3 * S
child = (sp >= 0).nonzero(as_tuple=True)[0].contiguous(); par = sp[child].contiguous()

def lap(v):
    v = v.reshape(-1, 3); out = torch.zeros_like(v)
    d = v.index_select(0, child) - v.index_select(0, par)
    out.index_add_(0, child, d); out.index_add_(0, par, -d)
    return (LAM * out).reshape(-1)

t0 = time.time()
hvps = []
for i, st in enumerate(bs):
    _l, sv = forward_solve([st], theta, rw)
    hvps.append(make_exact_hvp([st], theta, rw, sv))
    print(f"  built batch {i} hvp  peakmem={torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

def Av(v):                                   # H = sum_b H_b + lam*L,  all fp64
    acc = torch.zeros(p, dtype=torch.float64, device=DEV)
    for h in hvps:
        acc += h(v).double()
    return acc + lap(v)

# ---- 1) Lanczos -> bottom-K invariant subspace V (full reorthogonalization, fp64) ----
gen = torch.Generator(device=DEV).manual_seed(0)
q = torch.randn(p, generator=gen, device=DEV, dtype=torch.float64); q /= q.norm()
Q, al, be = [], [], []
beta, qp = 0.0, torch.zeros_like(q)
for it in range(M):
    w = Av(q) - beta * qp
    a = float(torch.dot(w, q)); w -= a * q
    for qq in Q:
        w -= torch.dot(w, qq) * qq
    Q.append(q.clone()); al.append(a)
    b = float(w.norm())
    if b < 1e-12:
        break
    qp, q, beta = q, w / b, b; be.append(b)
    if (it + 1) % 25 == 0:
        print(f"  lanczos {it+1}/{M}  t={time.time()-t0:.0f}s", flush=True)
ev, Sm = eigh_tridiagonal(np.array(al), np.array(be[: len(al) - 1]), eigvals_only=False)
kk = min(K, len(al))
Qm = torch.stack(Q, dim=1)
V = Qm @ torch.tensor(Sm[:, :kk], device=DEV, dtype=torch.float64)
print(f"  Lanczos done: {len(al)} steps, tridiag bottom evals {np.sort(ev)[:6]}  t={time.time()-t0:.0f}s", flush=True)

# ---- 2) exact Rayleigh-Ritz on span(V): resolves the cluster ----
V, _ = torch.linalg.qr(V)
HV = torch.stack([Av(V[:, j].contiguous()) for j in range(V.shape[1])], dim=1)
Hk = V.T @ HV; Hk = 0.5 * (Hk + Hk.T)
mu, W = torch.linalg.eigh(Hk)
U = V @ W; HU = HV @ W
resid = (HU - mu.unsqueeze(0) * U).norm(dim=0)

print("\n=== certified bottom spectrum of H + lam*L (exact Rayleigh-Ritz) ===", flush=True)
for j in range(min(8, len(mu))):
    print(f"  mu[{j}] = {float(mu[j]):+.6e}   residual = {float(resid[j]):.3e}", flush=True)
lam_min = float(mu[0]); r0 = float(resid[0])
certified = r0 < 0.1 * max(1e-3, abs(lam_min))
verdict = ("CERTIFIED PD (true local min)" if (lam_min > 0 and certified) else
           "CERTIFIED INDEFINITE (saddle / not a min)" if (lam_min < 0 and certified) else
           "STILL UNRESOLVED")
print(f"\nDEFLATE CERT  lam_min={lam_min:+.6e}  resid={r0:.3e}  -> {verdict}  "
      f"peakmem={torch.cuda.max_memory_allocated()/1e9:.1f} GB  t={time.time()-t0:.0f}s", flush=True)
