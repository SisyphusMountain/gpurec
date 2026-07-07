import torch
from gpurec.optim.hvp_exact import make_exact_hvp


def genewise_hessian_blocks(static, theta, receiver_weights, sv, *, omega=None, active=("theta",)):
    """Structured curvature for the genewise arrowhead Newton solve. P0: theta-only.

    Returns per-family blocks; omega/alpha entries are added in Tasks 4-5. The HVP itself
    is one joint operator (make_exact_hvp) -- these blocks are materialized ONLY for the
    Newton solve, never to compute H@u.
    """
    assert tuple(active) == ("theta",), "P0: theta-only; omega/alpha added in Tasks 4-5"
    G = int(theta.shape[0])
    tsi = int(static.solver_options.pi_iters)
    hvp = make_exact_hvp([static], theta, receiver_weights, sv, tangent_self_iters=tsi)
    cols = []
    for j in range(3):  # broadcast e_j across all families -> column j of every 3x3 block at once
        u = torch.zeros(G, 3, device=theta.device, dtype=theta.dtype); u[:, j] = 1.0
        cols.append(hvp(u.reshape(-1))[: G * 3].reshape(G, 3))
    H = torch.stack(cols, dim=-1)  # [G,3,3], H[:,:,j] = col j
    return {"H_tt": 0.5 * (H + H.transpose(1, 2))}


def _assemble_dense_arrowhead(blocks, g_theta, g_omega, g_alpha, mu):
    """TEST-ONLY oracle: materialize the full arrowhead Hessian + mu*I and the stacked gradient.

    Block/flat order matches ``newton_step_joint``'s output packing and the test's
    ``cat([dth.reshape(-1), dom.reshape(-1), dal.reshape(-1)])``:
        [ theta_0..theta_{G-1} (3 each) | omega_0..omega_{G-1} (S each) | alpha (S) ].
    ``H_oo_g = diag(H_oo_diag_g) + H_oo_lr_g @ H_oo_lr_g^T`` is materialized densely here (only here).
    Returns ``(Hd, gd)`` so the test can compare against ``torch.linalg.solve(Hd, -gd)``.
    """
    H_tt = blocks["H_tt"]            # [G,3,3]
    H_to = blocks["H_to"]            # [G,3,S]
    H_oo_diag = blocks["H_oo_diag"]  # [G,S]
    H_oo_lr = blocks["H_oo_lr"]      # [G,S,r]
    H_aa = blocks["H_aa"]            # [S,S]
    H_za = blocks["H_za"]            # [G,3+S,S]
    G = int(H_tt.shape[0])
    S = int(H_aa.shape[0])
    dev, dt = H_aa.device, H_aa.dtype
    N = 3 * G + G * S + S
    Hd = torch.zeros(N, N, device=dev, dtype=dt)

    al = slice(3 * G + G * S, 3 * G + G * S + S)
    for g in range(G):
        ts = slice(3 * g, 3 * g + 3)
        os_ = slice(3 * G + g * S, 3 * G + g * S + S)
        H_oo = torch.diag(H_oo_diag[g]) + H_oo_lr[g] @ H_oo_lr[g].T  # [S,S]
        # per-family core B_g = [[H_tt, H_to],[H_to^T, H_oo]]
        Hd[ts, ts] = H_tt[g]
        Hd[ts, os_] = H_to[g]
        Hd[os_, ts] = H_to[g].T
        Hd[os_, os_] = H_oo
        # couplings z_g = [theta_g; omega_g] <-> alpha
        H_ta = H_za[g][:3, :]   # [3,S]  theta_g <-> alpha
        H_oa = H_za[g][3:, :]   # [S,S]  omega_g <-> alpha
        Hd[ts, al] = H_ta
        Hd[al, ts] = H_ta.T
        Hd[os_, al] = H_oa
        Hd[al, os_] = H_oa.T
    Hd[al, al] = H_aa
    Hd = Hd + mu * torch.eye(N, device=dev, dtype=dt)

    gd = torch.cat([g_theta.reshape(-1), g_omega.reshape(-1), g_alpha.reshape(-1)])
    return Hd, gd


def newton_step_joint(blocks, g_theta, g_omega, g_alpha, mu):
    """Arrowhead Newton solve of ``(H + mu*I) delta = -g`` via Schur complement + Woodbury.

    ``H`` is block-diagonal per-family cores ``B_g = [[H_tt_g, H_to_g],[H_to_g^T, H_oo_g]]``
    (each ``(3+S)``) plus a global ``alpha`` arrow (dense ``H_aa`` ``S x S`` + couplings
    ``H_za_g`` ``(3+S) x S``). ``H_oo_g = diag(H_oo_diag_g) + H_oo_lr_g @ H_oo_lr_g^T`` is applied
    matrix-free via Woodbury -- the dense ``S x S`` ``H_oo`` is NEVER formed.

    Args:
        blocks: dict with H_tt[G,3,3], H_to[G,3,S], H_oo_diag[G,S], H_oo_lr[G,S,r],
            H_aa[S,S], H_za[G,3+S,S].
        g_theta[G,3], g_omega[G,S], g_alpha[S]: gradient pieces.
        mu: Levenberg damping (added to the whole diagonal).
    Returns:
        (dtheta[G,3], domega[G,S], dalpha[S]).
    """
    H_tt = blocks["H_tt"]            # [G,3,3]
    H_to = blocks["H_to"]            # [G,3,S]
    H_oo_diag = blocks["H_oo_diag"]  # [G,S]
    H_oo_lr = blocks["H_oo_lr"]      # [G,S,r]
    H_aa = blocks["H_aa"]            # [S,S]
    H_za = blocks["H_za"]            # [G,3+S,S]
    G = int(H_tt.shape[0])
    S = int(H_aa.shape[0])
    r = int(H_oo_lr.shape[-1])
    dev, dt = H_aa.device, H_aa.dtype

    A = H_tt + mu * torch.eye(3, device=dev, dtype=dt)   # [G,3,3]  H_tt + mu I_3
    Bmat = H_to                                          # [G,3,S]
    Bt = H_to.transpose(1, 2)                            # [G,S,3]
    d = H_oo_diag + mu                                   # [G,S]    diag of H_oo + mu
    U = H_oo_lr                                          # [G,S,r]
    dinv = (1.0 / d).unsqueeze(-1)                       # [G,S,1]

    # Woodbury factor for D = diag(d) + U U^T:  M = I_r + U^T D_d^{-1} U  (SPD).
    Ir = torch.eye(r, device=dev, dtype=dt)
    M = Ir + U.transpose(1, 2) @ (dinv * U)              # [G,r,r]
    Lm = torch.linalg.cholesky(M)

    def Dinv(X):  # X:[G,S,k] -> (diag(d)+UU^T)^{-1} X  via Woodbury
        DiX = dinv * X
        t = U.transpose(1, 2) @ DiX                      # [G,r,k]
        w = torch.cholesky_solve(t, Lm)                  # [G,r,k]
        return DiX - dinv * (U @ w)

    # Schur complement of D inside B_g:  S_A = A - Bmat D^{-1} Bmat^T  (3x3).
    DinvBt = Dinv(Bt)                                    # [G,S,3]  D^{-1} Bmat^T
    S_A = A - Bmat @ DinvBt                              # [G,3,3]
    lu_A, piv_A = torch.linalg.lu_factor(S_A)

    def Binv(RHS):  # RHS:[G,3+S,k] -> B_g^{-1} RHS  (2x2 block solve, 3 vs S)
        r1 = RHS[:, :3, :]                               # [G,3,k]
        r2 = RHS[:, 3:, :]                               # [G,S,k]
        Dinv_r2 = Dinv(r2)
        w1 = r1 - Bmat @ Dinv_r2                         # [G,3,k]
        x1 = torch.linalg.lu_solve(lu_A, piv_A, w1)      # [G,3,k]  S_A^{-1} w1
        x2 = Dinv(r2 - Bt @ x1)                          # [G,S,k]
        return torch.cat([x1, x2], dim=1)               # [G,3+S,k]

    g_z = torch.cat([g_theta, g_omega], dim=1)          # [G,3+S]
    H_az = H_za.transpose(1, 2)                          # [G,S,3+S]

    BinvHza = Binv(H_za)                                 # [G,3+S,S]  B^{-1} H_za
    Binv_gz = Binv(g_z.unsqueeze(-1))                    # [G,3+S,1]  B^{-1} g_z

    # Dense S x S alpha system: (H_aa + mu I - sum_g H_az B^{-1} H_za) d_alpha = -(g_a - sum_g H_az B^{-1} g_z)
    S_alpha = H_aa + mu * torch.eye(S, device=dev, dtype=dt) - (H_az @ BinvHza).sum(0)
    rhs_alpha = -g_alpha + (H_az @ Binv_gz).squeeze(-1).sum(0)
    d_alpha = torch.linalg.solve(S_alpha, rhs_alpha)     # [S]

    # Back-substitute: d_z_g = B_g^{-1}(-g_z_g - H_za_g d_alpha)
    rhs_z = -g_z - (H_za @ d_alpha)                      # [G,3+S]
    d_z = Binv(rhs_z.unsqueeze(-1)).squeeze(-1)          # [G,3+S]
    return d_z[:, :3].contiguous(), d_z[:, 3:].contiguous(), d_alpha
