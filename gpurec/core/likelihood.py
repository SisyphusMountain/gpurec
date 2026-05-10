"""Likelihood computation: E solver and log-likelihood.

The heavy Pi forward/backward code lives in forward.py and backward.py;
this module owns E_step, E_fixed_point, and compute_log_likelihood.
"""
import torch
import math
import os

from .terms import gather_E_children
from .log2_utils import logsumexp2, logaddexp2, _safe_log2_internal as _safe_log2
from ._helpers import _nvtx_range

NEG_INF = float("-inf")


# =========================================================================
# E solver
# =========================================================================

def _uniform_ancestor_sum(expE_2d, ancestors_T):
    """Compute ``expE_2d @ ancestors_T`` with a CUDA bf16 sparse fallback."""
    if (
        expE_2d.device.type == "cuda"
        and (expE_2d.dtype == torch.bfloat16 or ancestors_T.dtype == torch.bfloat16)
    ):
        with torch.amp.autocast("cuda", enabled=False):
            return (expE_2d.float() @ ancestors_T.float()).contiguous()
    return (expE_2d @ ancestors_T).contiguous()

def E_step(E, sp_P_idx, sp_child12_idx, log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat, pibar_mode='dense',
           ancestors_T=None):
    """E can either have shape [S] or [N_genes, S]. Likewise, transfer_mat can have shape [S, S] or [N_genes, S, S]."""
    E_stack = torch.empty((4, *E.shape), dtype=E.dtype, device=E.device)
    # S
    E_s12 = gather_E_children(E, sp_P_idx, sp_child12_idx)
    E_s1, E_s2 = torch.chunk(E_s12, 2, dim=-1)  # Each [N_genes*S]
    E_s1 = E_s1.view(E.shape) # should broadcast correctly when E has shape [S] or [N_genes, S]
    E_s2 = E_s2.view(E.shape) # should broadcast correctly when E has shape [S] or [N_genes, S]
    # should broadcast correctly when log_pS is [S] and log_pD is [S]
    # or if log_pS is [N_genes, S] and log_pD is [N_genes, S]
    # Align parameter tensors to broadcast with E
    def _align_param(p: torch.Tensor, E_ref: torch.Tensor):
        if not isinstance(p, torch.Tensor):
            return p
        if p.ndim == 0:
            return p
        if E_ref.ndim == 1:
            # E is [S]; allow p in [] or [S]
            return p
        # E is [N, S]
        N, S = E_ref.shape
        if p.ndim == 1 and p.shape[0] == N:
            return p.unsqueeze(-1)  # [N,1] -> broadcasts on S
        return p

    pS = _align_param(log_pS, E)
    pD = _align_param(log_pD, E)
    pL = _align_param(log_pL, E)

    # S
    E_stack[0] = pS + E_s1 + E_s2
    # D
    E_stack[1] = pD + 2 * E
    # T: compute Ebar
    if pibar_mode == 'uniform':
        # Exact Ebar = row_sum(E) - ancestor_sum(E), in log-space
        max_E = E.max(dim=-1, keepdim=True).values
        expE = torch.exp2(E - max_E)                     # [S] or [N, S]
        expE_2d = expE.unsqueeze(0) if expE.ndim == 1 else expE
        if expE_2d.device.type == "cuda" and expE_2d.dtype == torch.bfloat16:
            expE_2d_acc = expE_2d.float()
            max_E_acc = max_E.float()
            max_transfer_acc = max_transfer_mat.float()
        else:
            expE_2d_acc = expE_2d
            max_E_acc = max_E
            max_transfer_acc = max_transfer_mat
        row_sum = expE_2d_acc.sum(dim=-1, keepdim=True)  # [1, 1] or [N, 1]
        # Sparse COO matmul returns a column-major (non-contiguous) result when LHS
        # is 2D (batched case). Make it contiguous before subsequent arithmetic so
        # downstream Triton kernels that assume C-contiguous layout don't read garbage.
        ancestor_sum = _uniform_ancestor_sum(expE_2d, ancestors_T)  # [1, S] or [N, S]
        Ebar_linear = (row_sum - ancestor_sum).squeeze(0) if expE.ndim == 1 else (row_sum - ancestor_sum)
        # Use safe log2: float32 cancellation can make row_sum - ancestor_sum
        # slightly negative for species with many ancestors; safe log2 returns
        # -inf (zero transfer contribution) instead of NaN.
        Ebar = _safe_log2(Ebar_linear) + max_E_acc + max_transfer_acc.squeeze(-1)
        if Ebar.dtype != E.dtype:
            Ebar = Ebar.to(dtype=E.dtype)
    else:
        # Dense/topk: full matvec with [S,S] transfer matrix
        # (topk uses dense for E since E is [S] — cheap)
        max_E = E.max(dim=-1, keepdim=True).values
        expE = torch.exp2(E - max_E)
        Ebar_linear = torch.einsum("...ij, ...j-> ...i", transfer_mat, expE)
        Ebar = torch.log2(Ebar_linear) + max_E + max_transfer_mat.squeeze(-1)
    E_stack[2] = E + Ebar
    # L
    # should broadcast correctly if log_pL is [S] and if log_pL is [N_genes, S]
    E_stack[3] = pL

    E_new = logsumexp2(E_stack, dim=0)
    return E_new, E_s1, E_s2, Ebar


def _E_fixed_point_alerax_gs(
    *,
    species_helpers,
    log_pS,
    log_pD,
    log_pL,
    max_transfer_mat,
    ancestors_T,
    max_iters,
    dtype,
    device,
):
    """AleRax-compatible extinction update for uniform-transfer GLOBAL runs."""
    S = int(species_helpers["S"])

    def _prob_param(p: torch.Tensor, name: str) -> torch.Tensor:
        p = p.to(device=device, dtype=dtype)
        if p.ndim == 0:
            return torch.exp2(p).expand(S)
        if p.ndim == 1 and p.shape[0] == S:
            return torch.exp2(p)
        raise NotImplementedError(
            f"GPUREC_E_ALERAX_GS currently supports scalar or [S] {name}, "
            f"got shape {tuple(p.shape)}"
        )

    pS = _prob_param(log_pS, "log_pS")
    pD = _prob_param(log_pD, "log_pD")
    pL = _prob_param(log_pL, "log_pL")
    transfer_scale = torch.exp2(max_transfer_mat.to(device=device, dtype=dtype).squeeze(-1))
    if transfer_scale.ndim == 0:
        transfer_scale = transfer_scale.expand(S)
    if transfer_scale.ndim != 1 or transfer_scale.shape[0] != S:
        raise NotImplementedError(
            "GPUREC_E_ALERAX_GS currently supports scalar or [S] max_transfer_mat, "
            f"got shape {tuple(transfer_scale.shape)}"
        )

    sp_P_idx = species_helpers["s_P_indexes"].to(device=device).long()
    sp_child12_idx = species_helpers["s_C12_indexes"].to(device=device).long()
    child1 = torch.full((S,), S, dtype=torch.long, device=device)
    child2 = torch.full((S,), S, dtype=torch.long, device=device)
    first_child = sp_P_idx < S
    if bool(first_child.any()):
        p = sp_P_idx[first_child]
        c = sp_child12_idx[first_child]
        child1[p] = c
    if bool((~first_child).any()):
        p = sp_P_idx[~first_child] - S
        c = sp_child12_idx[~first_child]
        child2[p] = c

    child1_cpu = child1.detach().cpu().tolist()
    child2_cpu = child2.detach().cpu().tolist()
    depths = [0] * S
    for s_idx in range(S):
        c1 = child1_cpu[s_idx]
        if c1 < S:
            c2 = child2_cpu[s_idx]
            depths[s_idx] = max(depths[c1], depths[c2]) + 1
    species_levels = [
        torch.tensor(
            [s_idx for s_idx, depth in enumerate(depths) if depth == level],
            dtype=torch.long,
            device=device,
        )
        for level in range(max(depths) + 1)
    ]

    E_prob = torch.zeros((S,), dtype=dtype, device=device)
    Ebar_prob = torch.zeros((S,), dtype=dtype, device=device)
    ancestors_T = ancestors_T.to(device=device, dtype=dtype)

    for iteration in range(max_iters):
        for level_idx in species_levels:
            if level_idx.numel() == 0:
                continue
            value = pL[level_idx] + pD[level_idx] * E_prob[level_idx] * E_prob[level_idx]
            value = value + E_prob[level_idx] * Ebar_prob[level_idx]
            c1 = child1[level_idx]
            valid = c1 < S
            if bool(valid.any()):
                internal_idx = level_idx[valid]
                value = value.clone()
                value[valid] = value[valid] + (
                    pS[internal_idx]
                    * E_prob[c1[valid]]
                    * E_prob[child2[internal_idx]]
                )
            E_prob[level_idx] = value

        row_sum = E_prob.sum().unsqueeze(0)
        ancestor_sum = (E_prob.unsqueeze(0) @ ancestors_T).squeeze(0)
        Ebar_prob = (row_sum - ancestor_sum) * transfer_scale

    E = _safe_log2(E_prob)
    E_s12 = gather_E_children(E, species_helpers["s_P_indexes"], species_helpers["s_C12_indexes"])
    E_s1, E_s2 = torch.chunk(E_s12, 2, dim=-1)
    E_s1 = E_s1.view(E.shape)
    E_s2 = E_s2.view(E.shape)
    Ebar = _safe_log2(Ebar_prob)
    return {
        "E": E,
        "iterations": max_iters,
        "E_s1": E_s1,
        "E_s2": E_s2,
        "E_bar": Ebar,
    }


def E_fixed_point(species_helpers,
                          log_pS,
                          log_pD,
                          log_pL,
                          transfer_mat,
                          max_transfer_mat,
                          max_iters,
                          tolerance,
                          warm_start_E,
                          dtype,
                          device,
                          pibar_mode='dense',
                          ancestors_T=None):

    S = species_helpers['S']

    alerax_e_compat = (
        os.environ.get("GPUREC_E_ALERAX_GS", "0") != "0"
        or os.environ.get("GPUREC_ALERAX_COMPAT", "0") != "0"
    )
    if alerax_e_compat and (pibar_mode != "uniform" or ancestors_T is None):
        raise NotImplementedError(
            "GPUREC_ALERAX_COMPAT currently supports uniform-transfer "
            "E evaluation with ancestors_T only"
        )
    if (
        alerax_e_compat
        and pibar_mode == "uniform"
        and ancestors_T is not None
    ):
        return _E_fixed_point_alerax_gs(
            species_helpers=species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            max_transfer_mat=max_transfer_mat,
            ancestors_T=ancestors_T,
            max_iters=(
                int(os.environ.get("GPUREC_ALERAX_COMPAT_E_ITERS", "4"))
                if os.environ.get("GPUREC_ALERAX_COMPAT", "0") != "0"
                else max_iters
            ),
            dtype=dtype,
            device=device,
        )

    # Determine batch size from parameters if present
    N = None
    if isinstance(transfer_mat, torch.Tensor) and transfer_mat.ndim == 3:
        N = transfer_mat.shape[0]
    elif isinstance(log_pS, torch.Tensor) and log_pS.ndim == 2:
        N = log_pS.shape[0]
    # If parameters are per-gene scalars (shape [N]) in the genewise/non-specieswise case,
    # infer N from their length when it differs from S.
    if N is None and isinstance(log_pS, torch.Tensor) and log_pS.ndim == 1 and log_pS.shape[0] != S:
        N = log_pS.shape[0]
    if N is None and isinstance(log_pD, torch.Tensor) and log_pD.ndim == 1 and log_pD.shape[0] != S:
        N = log_pD.shape[0]
    if N is None and isinstance(log_pL, torch.Tensor) and log_pL.ndim == 1 and log_pL.shape[0] != S:
        N = log_pL.shape[0]

    # Initialize with log(0.5), or use a warm-start value if available
    if warm_start_E is not None:
        E = warm_start_E.detach()
    else:
        if N is None:
            E = torch.full((S,), -1.0, dtype=dtype, device=device)  # log2(0.5)
        else:
            E = torch.full((N, S), -1.0, dtype=dtype, device=device)  # log2(0.5)
    E_s1 = torch.full_like(E, NEG_INF)
    E_s2 = torch.full_like(E, NEG_INF)

    converged_iter = max_iters
    # Group the entire fixed-point solve under a single NVTX range
    with _nvtx_range("E step"):
        for iteration in range(max_iters):
            with _nvtx_range(f"E iter {iteration}"):
                result = E_step(
                    E=E,
                    sp_P_idx=species_helpers['s_P_indexes'],
                    sp_child12_idx=species_helpers['s_C12_indexes'],
                    log_pS=log_pS,
                    log_pD=log_pD,
                    log_pL=log_pL,
                    transfer_mat=transfer_mat,
                    max_transfer_mat=max_transfer_mat,
                    pibar_mode=pibar_mode,
                    ancestors_T=ancestors_T,
                )
                
                E_new, E_s1, E_s2, E_bar = result
                
                # Check convergence (sup-norm across all dims)
                if torch.abs(E_new - E).max().item() < tolerance:
                    converged_iter = iteration + 1
                    E = E_new
                    break
                
                E = E_new
    
    return {
        'E': E,
        'iterations': converged_iter,
        'E_s1': E_s1,
        'E_s2': E_s2,
        'E_bar': E_bar
        }


# =========================================================================
# Log-likelihood
# =========================================================================

def compute_log_likelihood(Pi, E, root_clade_idx):
    """Computes log-likelihood in a batched way over the number
    of gene families.
    Output has shape len(root_clade_idx).
    Result is in log2 units (bits)."""

    # This will broadcast if root_clade_idx has shape [N_gene_trees]
    root_probs = Pi[root_clade_idx, :]
    # We remove log2(|S|) because we assume a uniform prior over the root species
    numerator = logsumexp2(root_probs, dim=-1) - math.log2(Pi.shape[-1])
    # Will still work if E has shape [N_gene_trees, S]
    denominator = torch.log2((1-torch.exp2(E).mean(dim=-1)))
    return -(numerator - denominator)


def compute_log_likelihood_root_rows(Pi_root_rows, E):
    """Compute per-family NLL from already-gathered root rows.

    ``Pi_root_rows`` is ``[G, S]`` in family order. This is equivalent to
    ``compute_log_likelihood(Pi, E, root_ids)`` when ``Pi_root_rows`` has been
    gathered as ``Pi[root_ids]``, but avoids keeping the full Pi matrix alive in
    root-likelihood-only callers.
    """
    numerator = logsumexp2(Pi_root_rows, dim=-1) - math.log2(Pi_root_rows.shape[-1])
    denominator = torch.log2((1 - torch.exp2(E).mean(dim=-1)))
    return -(numerator - denominator)
