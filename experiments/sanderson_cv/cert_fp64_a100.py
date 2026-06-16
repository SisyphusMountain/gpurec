"""Resolved fp64 PD certificate for a polished theta: smallest eigenvalue of H + lam*L via the
ANALYTIC exact-HVP summed over batches, run in fp64 (the fp32 HVP cannot resolve the near-zero
bottom -- residual ~0.5%*||H|| ~ 0.25). Prints peak memory after each batch HVP build so an OOM is
diagnosable. Env: CAP, THETA, LAM, CERT_M."""
import os, time
import torch
from gpurec.optim.value_and_grad import forward_solve
from gpurec.optim.hvp_exact import make_exact_hvp
from gpurec.optim.cg import lanczos_min_eigpair

CAP = os.environ["CAP"]; THETA = os.environ["THETA"]
LAM = float(os.environ.get("LAM", "1.0")); M = int(os.environ.get("CERT_M", "120"))
dev = "cuda"
print(f"=== fp64 cert: {torch.cuda.get_device_name(0)}  lam={LAM}  m={M} ===", flush=True)
cap = torch.load(CAP, map_location=dev, weights_only=False)
bs = cap["batch_statics"]; rw = cap["rw"].to(dev).double(); sp = cap["sp_parent"].to(dev).long(); S = int(cap["S"])
theta = torch.load(THETA, map_location=dev, weights_only=False)["theta"].to(dev).double()
p = S * 3
child = (sp >= 0).nonzero(as_tuple=True)[0].contiguous(); par = sp[child].contiguous()

def lap(u, lam):
    u = u.reshape(-1, 3); out = torch.zeros_like(u)
    d = u.index_select(0, child) - u.index_select(0, par)
    out.index_add_(0, child, d); out.index_add_(0, par, -d)
    return (lam * out).reshape(-1)

t0 = time.time()
hvps = []
for i, st in enumerate(bs):
    _l, sv = forward_solve([st], theta, rw)               # fp64 forward (theta/rw fp64)
    hvps.append(make_exact_hvp([st], theta, rw, sv))       # fp64 HVP
    print(f"  built batch {i} hvp  peakmem={torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

def Av(v):
    acc = torch.zeros(p, dtype=torch.float64, device=dev)
    for h in hvps:
        acc += h(v.double()).double()
    return acc + lap(v.double(), LAM)

lam_min, vmin = lanczos_min_eigpair(Av, p, m=M, seed=0)
resid = float((Av(vmin) - lam_min * vmin).norm())
pd = lam_min > 0 and resid < 0.1 * max(1.0, abs(lam_min))
print(f"\nFP64 CERT  lam_min={lam_min:+.6e}  ritz_resid={resid:.3e}  m={M}  PD={pd}  "
      f"peakmem={torch.cuda.max_memory_allocated()/1e9:.1f} GB  t={time.time()-t0:.0f}s", flush=True)
