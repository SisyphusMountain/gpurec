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
