"""Likelihood computation: E solver, log-likelihood, and re-exports.

The heavy Pi forward/backward code lives in forward.py and backward.py;
this module keeps the E solver, compute_log_likelihood, shared utilities,
and re-exports everything so that existing ``from .likelihood import X``
statements continue to work unchanged.
"""
import torch
import math
from contextlib import contextmanager
import os
import inspect

from .terms import gather_E_children
from .log2_utils import logsumexp2, logaddexp2
from .log2_utils import _safe_log2_internal as _safe_log2

NEG_INF = float("-inf")


# =========================================================================
# Utilities (shared by forward.py / backward.py)
# =========================================================================

def _safe_exp2_ratio(a, b):
    """2^(a-b) safely: returns 0 when a = -inf."""
    neg_inf_a = (a == NEG_INF)
    a_safe = torch.where(neg_inf_a, torch.zeros_like(a), a)
    b_safe = torch.where(neg_inf_a, torch.zeros_like(b), b)
    return torch.where(neg_inf_a, torch.zeros_like(a), torch.exp2(a_safe - b_safe))


@contextmanager
def _nvtx_range(name: str):
    nvtx = getattr(getattr(torch, "cuda", None), "nvtx", None)
    if nvtx is not None and hasattr(nvtx, "range"):
        try:
            with nvtx.range(name):
                yield
            return
        except Exception:
            pass
    pushed = False
    if nvtx is not None and hasattr(nvtx, "range_push"):
        try:
            nvtx.range_push(name)
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed and hasattr(nvtx, "range_pop"):
            try:
                nvtx.range_pop()
            except Exception:
                pass


@contextmanager
def _nvtx_here(name: str):
    """NVTX range whose label includes caller file:line for easy mapping."""
    try:
        frame = inspect.currentframe().f_back
        info = inspect.getframeinfo(frame, context=0)
        base = os.path.basename(info.filename)
        label = f"{name} [{base}:{info.lineno}]"
    except Exception:
        label = name
    record_function = None
    try:
        record_function = getattr(getattr(torch, 'autograd', None).profiler, 'record_function', None)
    except Exception:
        record_function = None
    if record_function is not None:
        with record_function(label):
            with _nvtx_range(label):
                yield
    else:
        with _nvtx_range(label):
            yield


def _seg_logsumexp_host(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """CPU fallback for segmented logsumexp; uses Triton kernel when CUDA is available."""
    from .kernels.scatter_lse import seg_logsumexp
    if x.is_cuda and ptr.is_cuda:
        return seg_logsumexp(x, ptr)
    num_segs = int(ptr.numel()) - 1
    out = []
    for i in range(num_segs):
        s = int(ptr[i].item())
        e = int(ptr[i + 1].item())
        if e > s:
            out.append(logsumexp2(x[s:e], dim=0))
        else:
            out.append(torch.full_like(x[0], NEG_INF))
    return torch.stack(out, dim=0) if out else torch.empty((0, *x.shape[1:]), device=x.device, dtype=x.dtype)


# =========================================================================
# E solver
# =========================================================================

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
    if pibar_mode == 'uniform_approx':
        # Uniform: Ebar[s] = logsumexp2(E) + mt[s]
        # O(S) instead of O(S^2) matvec
        lse_E = logsumexp2(E, dim=-1, keepdim=True)  # [1] or [N, 1]
        Ebar = lse_E + max_transfer_mat  # broadcasts to [S] or [N, S]
    elif pibar_mode == 'uniform':
        # Exact Ebar = row_sum(E) - ancestor_sum(E), in log-space
        max_E = E.max(dim=-1, keepdim=True).values
        expE = torch.exp2(E - max_E)                     # [S] or [N, S]
        expE_2d = expE.unsqueeze(0) if expE.ndim == 1 else expE
        row_sum = expE_2d.sum(dim=-1, keepdim=True)      # [1, 1] or [N, 1]
        ancestor_sum = expE_2d @ ancestors_T              # [1, S] or [N, S] sparse matmul
        Ebar_linear = (row_sum - ancestor_sum).squeeze(0) if expE.ndim == 1 else (row_sum - ancestor_sum)
        Ebar = torch.log2(Ebar_linear) + max_E + max_transfer_mat.squeeze(-1)
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


# =========================================================================
# Re-exports from forward.py, backward.py and legacy.py
# =========================================================================
# Existing code does ``from .likelihood import Pi_wave_forward`` etc.
# These re-exports keep that working without touching every import site.

from .legacy import (  # noqa: E402, F401
    Pi_step,
    Pi_fixed_point,
)

from .forward import (  # noqa: E402, F401
    Pi_wave_forward,
    _compute_dts_cross,
    _compute_Pibar_inline,
    _compute_DTS_reduced,
    _reconstruct_split_parents,
    compute_gradient_bounds,
)

from .backward import (  # noqa: E402, F401
    _self_loop_differentiable,
    _dts_cross_differentiable,
    _self_loop_vjp_precompute,
    _gmres_self_loop_solve,
    _self_loop_Jt_apply,
    Pi_wave_backward,
)
