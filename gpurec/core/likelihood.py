"""Likelihood computation: E solver and log-likelihood.

The heavy Pi forward/backward code lives in forward.py and backward.py;
this module owns E_step, E_fixed_point, and compute_log_likelihood.
"""
import torch
import math

from .terms import gather_E_children
from .log2_utils import logsumexp2, _safe_log2_internal as _safe_log2
from ._helpers import _nvtx_range

NEG_INF = float("-inf")


# =========================================================================
# E solver
# =========================================================================

def _uniform_ancestor_sum(expE_2d, ancestors_T):
    """Compute the retained uniform ancestor sum."""
    return (expE_2d @ ancestors_T).contiguous()

def E_step(
    E,
    sp_P_idx,
    sp_child12_idx,
    log_pS,
    log_pD,
    log_pL,
    max_transfer_mat,
    ancestors_T=None,
):
    """One uniform-transfer extinction fixed-point step.

    ``E`` can have shape ``[S]`` or ``[N_genes, S]``.
    """
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
    max_E = E.max(dim=-1, keepdim=True).values
    expE = torch.exp2(E - max_E)
    expE_2d = expE.unsqueeze(0) if expE.ndim == 1 else expE
    if expE_2d.device.type == "cuda" and expE_2d.dtype == torch.bfloat16:
        expE_2d_acc = expE_2d.float()
        max_E_acc = max_E.float()
        max_transfer_acc = max_transfer_mat.float()
    else:
        expE_2d_acc = expE_2d
        max_E_acc = max_E
        max_transfer_acc = max_transfer_mat
    row_sum = expE_2d_acc.sum(dim=-1, keepdim=True)
    ancestor_sum = _uniform_ancestor_sum(expE_2d, ancestors_T)
    Ebar_linear = (row_sum - ancestor_sum).squeeze(0) if expE.ndim == 1 else (row_sum - ancestor_sum)
    Ebar = _safe_log2(Ebar_linear) + max_E_acc + max_transfer_acc.squeeze(-1)
    if Ebar.dtype != E.dtype:
        Ebar = Ebar.to(dtype=E.dtype)
    E_stack[2] = E + Ebar
    # L
    # should broadcast correctly if log_pL is [S] and if log_pL is [N_genes, S]
    E_stack[3] = pL

    E_new = logsumexp2(E_stack, dim=0)
    return E_new, E_s1, E_s2, Ebar


def E_fixed_point(species_helpers,
                          log_pS,
                          log_pD,
                          log_pL,
                          max_transfer_mat,
                          max_iters,
                          tolerance,
                          warm_start_E,
                          dtype,
                          device,
                          ancestors_T=None,
                          progress_callback=None,
                          trace_logsumexp: bool = False):

    S = species_helpers['S']

    # Determine batch size from parameters if present
    N = None
    if isinstance(max_transfer_mat, torch.Tensor) and max_transfer_mat.ndim >= 2:
        N = max_transfer_mat.shape[0]
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
    E_logsumexp_trace = None
    if trace_logsumexp:
        trace_shape = (max_iters,) if E.ndim == 1 else (max_iters, E.shape[0])
        E_logsumexp_trace = torch.empty(trace_shape, dtype=dtype, device=device)

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
                    max_transfer_mat=max_transfer_mat,
                    ancestors_T=ancestors_T,
                )
                
                E_new, E_s1, E_s2, E_bar = result

                max_diff = None
                converged = False
                if tolerance >= 0.0:
                    # Check convergence (sup-norm across all dims). Fixed-pass
                    # callers pass a negative tolerance and avoid this sync.
                    max_diff = torch.abs(E_new - E).max().item()
                    converged = max_diff < tolerance

                if E_logsumexp_trace is not None:
                    E_logsumexp_trace[iteration].copy_(logsumexp2(E_new, dim=-1))
                
                E = E_new

                if progress_callback is not None:
                    progress_callback(iteration + 1, max_iters, max_diff, converged)

                if converged:
                    converged_iter = iteration + 1
                    break
    
    return {
        'E': E,
        'iterations': converged_iter,
        'E_s1': E_s1,
        'E_s2': E_s2,
        'E_bar': E_bar,
        'E_logsumexp_trace': (
            None if E_logsumexp_trace is None
            else E_logsumexp_trace[:converged_iter]
        ),
        }


# =========================================================================
# Log-likelihood
# =========================================================================

def prepare_origination_probs(
    origination_probs,
    *,
    S: int,
    device,
    dtype,
    family_count: int | None = None,
    assume_prepared: bool = False,
):
    """Validate and normalize species origination probabilities.

    ``origination_probs`` may be ``[S]`` for one shared distribution or
    ``[G, S]`` for family-specific distributions.  Rows are normalized so callers
    can pass positive weights without pre-normalizing them.
    """
    if origination_probs is None:
        return None
    probs = torch.as_tensor(origination_probs, device=device, dtype=dtype)
    if probs.ndim == 1:
        if int(probs.shape[0]) != int(S):
            raise ValueError(
                f"origination_probs must have length {S}, got {tuple(probs.shape)}"
            )
    elif probs.ndim == 2:
        if int(probs.shape[1]) != int(S):
            raise ValueError(
                "origination_probs with two dimensions must have shape "
                f"[families, {S}], got {tuple(probs.shape)}"
            )
        if family_count is not None and int(probs.shape[0]) != int(family_count):
            raise ValueError(
                "family-specific origination_probs must have one row per family: "
                f"expected {family_count}, got {int(probs.shape[0])}"
            )
    else:
        raise ValueError(
            "origination_probs must be None, [S], or [families, S], "
            f"got shape {tuple(probs.shape)}"
        )

    if assume_prepared:
        return probs

    if not torch.isfinite(probs).all().item():
        raise ValueError("origination_probs must be finite")
    if (probs < 0).any().item():
        raise ValueError("origination_probs must be non-negative")
    row_sum = probs.sum(dim=-1, keepdim=True)
    if (row_sum <= 0).any().item():
        raise ValueError("each origination probability row must have positive mass")
    return probs / row_sum


def _weighted_logsumexp2(values, origination_probs):
    log_weights = torch.log2(origination_probs)
    return logsumexp2(values + log_weights, dim=-1)


def compute_origination_denominator(
    E,
    origination_probs=None,
    *,
    origination_probs_prepared: bool = False,
):
    """Return the log2 denominator for the root origination distribution."""
    if origination_probs is None:
        return torch.log2(1 - torch.exp2(E).mean(dim=-1))
    probs = prepare_origination_probs(
        origination_probs,
        S=int(E.shape[-1]),
        device=E.device,
        dtype=E.dtype,
        assume_prepared=origination_probs_prepared,
    )
    return torch.log2((probs * (1 - torch.exp2(E))).sum(dim=-1))


def compute_log_likelihood(
    Pi,
    E,
    root_clade_idx,
    origination_probs=None,
    *,
    origination_probs_prepared: bool = False,
):
    """Computes log-likelihood in a batched way over the number
    of gene families.
    Output has shape len(root_clade_idx).
    Result is in log2 units (bits).

    By default the root species origination distribution is uniform.  Passing
    ``origination_probs`` as ``[S]`` or ``[G, S]`` uses those probabilities for
    both the root ``Pi`` mixture and the ``1 - E`` denominator.
    """

    # This will broadcast if root_clade_idx has shape [N_gene_trees]
    root_probs = Pi[root_clade_idx, :]
    if origination_probs is None:
        # We remove log2(|S|) because we assume a uniform prior over the root species
        numerator = logsumexp2(root_probs, dim=-1) - math.log2(Pi.shape[-1])
        # Will still work if E has shape [N_gene_trees, S]
        denominator = torch.log2((1-torch.exp2(E).mean(dim=-1)))
    else:
        probs = prepare_origination_probs(
            origination_probs,
            S=int(Pi.shape[-1]),
            device=Pi.device,
            dtype=Pi.dtype,
            family_count=int(root_probs.shape[0]) if root_probs.ndim == 2 else None,
            assume_prepared=origination_probs_prepared,
        )
        numerator = _weighted_logsumexp2(root_probs, probs)
        denominator = compute_origination_denominator(
            E,
            probs,
            origination_probs_prepared=True,
        )
    return -(numerator - denominator)


def compute_log_likelihood_root_rows(
    Pi_root_rows,
    E,
    origination_probs=None,
    *,
    origination_probs_prepared: bool = False,
):
    """Compute per-family NLL from already-gathered root rows.

    ``Pi_root_rows`` is ``[G, S]`` in family order. This is equivalent to
    ``compute_log_likelihood(Pi, E, root_ids)`` when ``Pi_root_rows`` has been
    gathered as ``Pi[root_ids]``, but avoids keeping the full Pi matrix alive in
    root-likelihood-only callers.
    """
    if origination_probs is None:
        numerator = logsumexp2(Pi_root_rows, dim=-1) - math.log2(Pi_root_rows.shape[-1])
        denominator = torch.log2((1 - torch.exp2(E).mean(dim=-1)))
    else:
        probs = prepare_origination_probs(
            origination_probs,
            S=int(Pi_root_rows.shape[-1]),
            device=Pi_root_rows.device,
            dtype=Pi_root_rows.dtype,
            family_count=int(Pi_root_rows.shape[0]),
            assume_prepared=origination_probs_prepared,
        )
        numerator = _weighted_logsumexp2(Pi_root_rows, probs)
        denominator = compute_origination_denominator(
            E,
            probs,
            origination_probs_prepared=True,
        )
    return -(numerator - denominator)
