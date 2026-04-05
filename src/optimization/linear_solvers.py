"""Matrix-free Krylov solvers (CG + GMRES fallback) for implicit gradient."""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor

from .types import LinearSolveStats


@torch.no_grad()
def _cg(
    Av: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    M: Optional[Callable[[Tensor], Tensor]] = None,
    tol: float = 1e-8,
    maxiter: int = 500,
    x0: Optional[Tensor] = None,
) -> Tuple[Tensor, LinearSolveStats, bool]:
    """
    Conjugate Gradients for SPD A. Uses only Av (no A^T).
    Returns (x, stats, success). If breakdown detected, success=False.
    """
    device, dtype = b.device, b.dtype
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - Av(x)
    z = r if M is None else M(r)
    p = z.clone()
    rz_old = float(torch.dot(r, z))
    bnorm = float(b.norm()) if b.numel() > 0 else 1.0
    bnorm = max(bnorm, 1.0)

    success = True
    iters = 0
    for k in range(1, maxiter + 1):
        Ap = Av(p)
        pAp = float(torch.dot(p, Ap))
        if pAp <= 0.0 or not math.isfinite(pAp):  # non-SPD or numerical breakdown
            success = False
            iters = k - 1
            break
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel_res = float(r.norm()) / bnorm
        if rel_res <= tol:
            iters = k
            return x, LinearSolveStats("CG", iters, rel_res, fallback_used=False), True
        z = r if M is None else M(r)
        rz_new = float(torch.dot(r, z))
        beta = rz_new / max(rz_old, 1e-30)
        p = z + beta * p
        rz_old = rz_new
        iters = k

    rel_res = float(r.norm()) / bnorm
    return x, LinearSolveStats("CG", iters, rel_res, fallback_used=False), success


@torch.no_grad()
def _gmres(
    Av: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    M: Optional[Callable[[Tensor], Tensor]] = None,  # right preconditioner
    tol: float = 1e-8,
    restart: int = 40,
    maxiter: int = 500,
    x0: Optional[Tensor] = None,
) -> Tuple[Tensor, LinearSolveStats]:
    """
    Restarted GMRES with right preconditioning. Uses only Av.
    """
    device, dtype = b.device, b.dtype
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    Id = (lambda v: v)
    M = Id if M is None else M

    iters = 0
    bnorm = float(b.norm())
    bnorm = max(bnorm, 1.0)

    while iters < maxiter:
        r = b - Av(x)
        beta = float(r.norm())
        if beta / bnorm <= tol:
            return x, LinearSolveStats("GMRES", iters, beta / bnorm, fallback_used=False)

        m = min(restart, maxiter - iters)
        V: List[Tensor] = []
        Z: List[Tensor] = []
        H = torch.zeros((m + 1, m), device=device, dtype=dtype)

        v1 = r / (beta + 1e-30)
        V.append(v1)
        happy = False
        used = 0

        for j in range(m):
            z_j = M(V[j]); Z.append(z_j)
            w = Av(z_j)
            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = w.norm()
            if float(H[j + 1, j]) < 1e-14:
                happy = True
                used = j + 1
                break
            V.append(w / (H[j + 1, j] + 1e-30))
            used = j + 1

        e1 = torch.zeros(used + 1, device=device, dtype=dtype); e1[0] = 1.0
        y = torch.linalg.lstsq(H[:used + 1, :used], beta * e1[:used + 1]).solution
        x = x + sum(y[i] * Z[i] for i in range(used))
        iters += used
        if happy:
            r = b - Av(x)
            rel = float(r.norm()) / bnorm
            return x, LinearSolveStats("GMRES", iters, rel, fallback_used=False)

    r = b - Av(x)
    rel = float(r.norm()) / bnorm
    return x, LinearSolveStats("GMRES", iters, rel, fallback_used=False)
