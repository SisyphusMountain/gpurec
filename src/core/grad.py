# ---------- Krylov & preconditioners (matrix-free) ----------
import torch
from typing import Callable, Optional

@torch.no_grad()
def estimate_diag_hutchinson(Av: Callable[[torch.Tensor], torch.Tensor],
                             n: int, k: int, device, dtype,
                             eps: float = 1e-8) -> torch.Tensor:
    """Estimate diag(A) from matvecs using Hutchinson probes; A is matrix-free."""
    acc = torch.zeros(n, device=device, dtype=dtype)
    for _ in range(k):
        z = torch.empty(n, device=device, dtype=dtype).bernoulli_(0.5).mul_(2).add_(-1)  # ±1
        acc += z * Av(z)
    d = (acc / float(k)).abs().clamp_min(eps)
    # optional normalization to avoid extreme scales
    med = d.median()
    if med > 0: d = d / med
    return d

@torch.no_grad()
def fgmres(Av: Callable[[torch.Tensor], torch.Tensor],
           b: torch.Tensor,
           M_apply: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
           tol: float = 1e-6,
           restart: int = 40,
           maxit: int = 200,
           x0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Flexible GMRES (right preconditioning). Minimizes ||r||_2; uses only matvecs.
    Av: x -> A x     ;  M_apply: right preconditioner z -> M^{-1} z
    """
    device, dtype = b.device, b.dtype
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    Id = (lambda v: v)
    if M_apply is None: M_apply = Id

    it = 0
    while it < maxit:
        r = b - Av(x)
        beta = r.norm()
        if beta <= tol * b.norm().clamp_min(1.0): return x
        m = min(restart, maxit - it)
        V, Z = [], []
        H = torch.zeros((m+1, m), device=device, dtype=dtype)
        v1 = r / beta
        V.append(v1)

        happy = False
        for j in range(m):
            z_j = M_apply(V[j])
            Z.append(z_j)
            w = Av(z_j)  # one VJP inside if Av = I - J^T
            # Modified Gram-Schmidt
            for i in range(j+1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j+1, j] = w.norm()
            if H[j+1, j] < 1e-14:
                happy = True
                m = j + 1
                break
            V.append(w / H[j+1, j])

        e1 = torch.zeros(m+1, device=device, dtype=dtype); e1[0] = 1.0
        y = torch.linalg.lstsq(H[:m+1, :m], beta * e1[:m+1]).solution
        # Update x with right preconditioner: x += Σ y_i Z_i
        x = x + sum(y[i] * Z[i] for i in range(m))
        it += m
        if happy:  # exact in reduced subspace
            return x
    return x

# ---------- Adjoint solves (transpose systems) via FGMRES ----------
@torch.no_grad()
def solve_adjoint_krylov(vjp_apply: Callable[[torch.Tensor], torch.Tensor],
                         rhs: torch.Tensor,
                         *,
                         tol: float = 1e-6,
                         restart: int = 40,
                         maxit: int = 200,
                         prec: str = "jacobi",
                         prec_hutch_probes: int = 8,
                         warm_start: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Solve (I - J^T) x = rhs. vjp_apply(v) returns J^T v.
    Right preconditioning:
      - 'jacobi' (default): diag estimated via Hutchinson; ~0 VJPs per Krylov step.
      - 'none': no preconditioner.
    """
    device, dtype = rhs.device, rhs.dtype

    def Av(x):  # A x = (I - J^T) x
        return x - vjp_apply(x)

    M_apply = None
    if prec == "jacobi":
        n = rhs.numel()
        d = estimate_diag_hutchinson(Av, n, k=prec_hutch_probes, device=device, dtype=dtype)
        M_apply = lambda v: v / d

    x = fgmres(Av, rhs, M_apply=M_apply, tol=tol, restart=restart, maxit=maxit, x0=warm_start)
    return x


v = solve_adjoint_krylov(lambda x: vjpF(x)[0], phi, tol=1e-6, restart=40, prec="jacobi")

w = solve_adjoint_krylov(lambda x: vjpG(x)[0], q, tol=1e-6, restart=40, prec="jacobi")
