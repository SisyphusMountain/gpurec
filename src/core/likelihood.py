import torch
import math
from contextlib import contextmanager
import os
import inspect

from .terms import (
    gather_E_children,
    gather_Pi_children,
    compute_DTS,
    compute_DTS_independent,
    compute_DTS_L,
    compute_DTS_L_independent,
)
from .kernels.scatter_lse import seg_logsumexp
from .kernels.seg_log_matmul import segmented_log_matmul

NEG_INF = float("-inf")


# Lightweight NVTX range helper that is safe on CPU-only setups
@contextmanager
def _nvtx_range(name: str):
    nvtx = getattr(getattr(torch, "cuda", None), "nvtx", None)
    # Try modern context-manager API first
    if nvtx is not None and hasattr(nvtx, "range"):
        try:
            with nvtx.range(name):
                yield
            return
        except Exception:
            # Fall through to push/pop fallback
            pass
    # Fallback: push/pop if available; otherwise no-op
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
    """NVTX range whose label includes caller file:line for easy mapping.

    Example label: "Pi: compute_Pibar [likelihood.py:155]".
    """
    try:
        frame = inspect.currentframe().f_back
        info = inspect.getframeinfo(frame, context=0)
        base = os.path.basename(info.filename)
        label = f"{name} [{base}:{info.lineno}]"
    except Exception:
        label = name
    # Try to also emit a PyTorch profiler user range for Chrome traces
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
    if x.is_cuda and ptr.is_cuda:
        return seg_logsumexp(x, ptr)
    # CPU path: simple loop over segments defined by ptr
    num_segs = int(ptr.numel()) - 1
    out = []
    for i in range(num_segs):
        s = int(ptr[i].item())
        e = int(ptr[i + 1].item())
        if e > s:
            out.append(torch.logsumexp(x[s:e], dim=0))
        else:
            # empty segment; produce -inf row
            out.append(torch.full_like(x[0], NEG_INF))
    return torch.stack(out, dim=0) if out else torch.empty((0, *x.shape[1:]), device=x.device, dtype=x.dtype)


def Pi_step(Pi: torch.Tensor,
            ccp_helpers: dict,
            species_helpers: dict,
            log_pS: torch.Tensor,
            log_pD: torch.Tensor,
            log_pL: torch.Tensor,
            transfer_mat_T: torch.Tensor,
            max_transfer_mat: torch.Tensor,
            clade_species_map: torch.Tensor,
            E: torch.Tensor,
            Ebar: torch.Tensor,
            E_s1: torch.Tensor,
            E_s2: torch.Tensor,
            log_2: torch.Tensor,
            genewise: bool,
            specieswise: bool,
            pairwise: bool,
            clades_per_gene: torch.Tensor,
            batch_info=None,):
    # region helpers
    # Extract helpers (precomputed sorted splits and CSR pointers)
    split_leftrights_sorted = ccp_helpers['split_leftrights_sorted']
    log_split_probs = ccp_helpers['log_split_probs_sorted'].unsqueeze(1).contiguous()  # [N_splits, 1]
    seg_parent_ids = ccp_helpers['seg_parent_ids']
    # Segment partition helpers: contiguous blocks [len>=2][len==1][len==0]
    num_segs_ge2 = ccp_helpers['num_segs_ge2']
    num_segs_eq1 = ccp_helpers['num_segs_eq1']
    # num_segs_eq0 = int(ccp_helpers.get('num_segs_eq0', 0))
    end_rows_ge2 = ccp_helpers['end_rows_ge2']
    ptr_ge2 = ccp_helpers['ptr_ge2']
    N_splits = ccp_helpers["N_splits"]
    sp_P_idx = species_helpers['s_P_indexes'] # index of parent for each internal node
    sp_c12_idx = species_helpers["s_C12_indexes"]

    C, S = Pi.shape
    # endregion helpers
    
    # Computing log event probabilities
    if batch_info is not None:
        seg_ptr = batch_info['seg_ptr']

    with _nvtx_here("Pi: gather_Pi_children"):
        Pi_s12 = gather_Pi_children(Pi, sp_P_idx, sp_c12_idx)  # [C, 2S]


    if batch_info is not None and genewise:
        # In this case, we have independent parameters for each gene tree.
        # Therefore, we need to add different values of log_pD, log_pS, log_pL
        # Compute per-gene transfer contributions via segmented log-matmul
        with _nvtx_here("Pi: segmented_log_matmul"):
            Pibar, _ = segmented_log_matmul(
                Pi,
                transfer_mat_T,
                max_transfer_mat,
                seg_ptr,
            )


        # Broadcast parameters to clade level with correct shape
        # log_pS/log_pD may be [N_genes] or [N_genes, S]. We need [C, S] or [C,1] accordingly.
        C = Pi.shape[0]
        with _nvtx_here("Pi: broadcast_params"):
            rep_log_pS = torch.repeat_interleave(log_pS, clades_per_gene, dim=0, output_size=C)
            if rep_log_pS.ndim == 1:
                rep_log_pS = rep_log_pS.view(C, 1)  # [C,S]
            rep_log_pD = torch.repeat_interleave(log_pD, clades_per_gene, dim=0, output_size=C)
            if rep_log_pD.ndim == 1:
                rep_log_pD = rep_log_pD.view(C, 1)              # [C,1]; broadcasts versus Pi [C,S]
        with _nvtx_here("Pi: Pi_D = Pi + rep_log_pD"):
            Pi_D = Pi + rep_log_pD
            # Split on the species tree children, THEN add log_pS
            Pi_s1, Pi_s2 = torch.chunk(Pi_s12, 2, dim=1)  # Each [C, S]
        with _nvtx_here("Pi: Pi_S_s1 = Pi_s1 + rep_log_pS"):
            Pi_S_s1 = Pi_s1 + rep_log_pS # [C, S]
        with _nvtx_here("Pi: Pi_S_s2 = Pi_s2 + rep_log_pS"):
            Pi_S_s2 = Pi_s2 + rep_log_pS # [C, S]
        with _nvtx_here("Pi: compute_DTS_independent"):
            DTS_term = compute_DTS_independent(Pi, Pibar, Pi_D, split_leftrights_sorted, N_splits, S, Pi_s12, Pi_S_s1, Pi_S_s2, log_split_probs)
        with _nvtx_here("Pi: compute_DTS_L_independent"):
            DTS_L_term = compute_DTS_L_independent(rep_log_pS, Pi, Pibar, E, E_s1, E_s2, Pi_S_s1, Pi_S_s2, Ebar, Pi_D, clade_species_map, log_2, clades_per_gene)
        
    else:
        # Computing Pibar
        with _nvtx_here("Pi: Pi_max = max(Pi, dim=1)"):
            Pi_max = torch.max(Pi, dim=1, keepdim=True).values # [C, 1]
        with _nvtx_here("Pi: Pi - Pi_max"):
            Pi_minus = Pi - Pi_max
        with _nvtx_here("Pi: exp(Pi - Pi_max)"):
            Pi_linear = torch.exp(Pi_minus)  # [C, S]
        with _nvtx_here("Pi: mm(Pi_linear, transfer_T)"):
            Pibar_linear = Pi_linear.mm(transfer_mat_T)  # [C, S]
        with _nvtx_here("Pi: log(Pibar_linear)"):
            Pibar_log = torch.log(Pibar_linear)
        with _nvtx_here("Pi: + Pi_max"):
            Pibar_log = Pibar_log + Pi_max
        with _nvtx_here("Pi: + max_transfer_vec"):
            Pibar = Pibar_log + max_transfer_mat.squeeze(-1)
        with _nvtx_here("Pi: compute_DTS"):
            DTS_term = compute_DTS(log_pD, log_pS, Pi_s12, Pi, Pibar, log_split_probs, split_leftrights_sorted, N_splits, S)
        with _nvtx_here("Pi: compute_DTS_L"):
            DTS_L_term = compute_DTS_L(log_pD, log_pS, Pi, Pibar, Pi_s12, E, Ebar, E_s1, E_s2, clade_species_map, log_2)
    with _nvtx_here("Pi: reduce_DTS"):
        DTS_reduced = torch.full((C, S), NEG_INF, device=Pi.device, dtype=Pi.dtype)
        # clades with >=2 splits
        if num_segs_ge2 > 0:
            with _nvtx_here("Pi: seg_logsumexp_ge2"):
                y_ge2 = _seg_logsumexp_host(DTS_term[:end_rows_ge2], ptr_ge2)
            with _nvtx_here("Pi: index_copy_ge2"):
                DTS_reduced.index_copy_(0, seg_parent_ids[:num_segs_ge2], y_ge2)
        # clades with exactly 1 split
        if num_segs_eq1 > 0:
            with _nvtx_here("Pi: index_copy_eq1"):
                DTS_reduced.index_copy_(0,
                                        seg_parent_ids[num_segs_ge2:num_segs_ge2+num_segs_eq1],
                                        DTS_term[end_rows_ge2:end_rows_ge2+num_segs_eq1])

    with _nvtx_here("Pi: logaddexp(DTS_reduced, DTS_L_term)"):
        return torch.logaddexp(DTS_reduced, DTS_L_term)

def E_step(E, sp_P_idx, sp_child12_idx, log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat):
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
    # T
    # avoid underflow by subtracting max
    # Use per-row max along species dimension for numerical stability
    max_E = E.max(dim=-1, keepdim=True).values
    expE = torch.exp(E - max_E)
    Ebar_linear = torch.einsum("...ij, ...j-> ...i", transfer_mat, expE)
    Ebar = torch.log(Ebar_linear) + max_E + max_transfer_mat.squeeze(-1)
    E_stack[2] = E + Ebar
    # L
    # should broadcast correctly if log_pL is [S] and if log_pL is [N_genes, S]
    E_stack[3] = pL

    E_new = torch.logsumexp(E_stack, dim=0)
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
                          device):

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
            E = torch.full((S,), -0.69, dtype=dtype, device=device)
        else:
            E = torch.full((N, S), -0.69, dtype=dtype, device=device)
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
    

def Pi_fixed_point(
    ccp_helpers,
    species_helpers,
    leaf_row_index,
    leaf_col_index,
    E,
    Ebar,
    E_s1,
    E_s2,
    log_pS,
    log_pD,
    log_pL,
    transfer_mat_T,
    max_transfer_mat,
    max_iters,
    tolerance,
    warm_start_Pi,
    device,
    dtype,
    *,
    genewise: bool = False,
    specieswise: bool = False,
    pairwise: bool = False,
    clades_per_gene: torch.Tensor | None = None,
    batch_info: dict | None = None,
):
    """Fixed-point solver for Pi using leaf mapping indices and current event params.

    This version matches the current Pi_step signature, including transfer_mat_T
    and max_transfer_mat, and supports (optional) genewise batching via clades_per_gene + batch_info.
    """
    C = int(ccp_helpers['C'])
    S = int(species_helpers['S'])

    # Build log clade->species map from compact indices
    clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
    clade_species_map[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0

    # Initialize Pi or use warm start
    if warm_start_Pi is not None:
        Pi = warm_start_Pi
    else:
        Pi = torch.full((C, S), -math.log(10.0), dtype=dtype, device=device)
        Pi[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0

    converged_iter = max_iters
    log_2 = torch.tensor([math.log(2.0)], dtype=dtype, device=device)

    # Default clades_per_gene if not provided (single family)
    if clades_per_gene is None:
        clades_per_gene = torch.tensor([C], dtype=torch.int64, device=device)

    # Group the entire fixed-point solve under a single NVTX range
    with _nvtx_range("Pi step"):
        for iteration in range(max_iters):
            with _nvtx_range(f"Pi iter {iteration}"):
                Pi_new = Pi_step(
                    Pi,
                    ccp_helpers,
                    species_helpers,
                    log_pS,
                    log_pD,
                    log_pL,
                    transfer_mat_T,
                    max_transfer_mat,
                    clade_species_map,
                    E,
                    Ebar,
                    E_s1,
                    E_s2,
                    log_2,
                    genewise,
                    specieswise,
                    pairwise,
                    clades_per_gene,
                    batch_info=batch_info,

                )
                if torch.abs(Pi_new - Pi).max() < tolerance:
                    converged_iter = iteration + 1
                    Pi = Pi_new
                    break
                Pi = Pi_new

    return {'Pi': Pi, 'clade_species_map': clade_species_map, 'iterations': converged_iter}


def compute_log_likelihood(Pi, E, root_clade_idx):
    """Computes log-likelihood in a batched way over the number
    of gene families.
    Output has shape len(root_clade_idx)"""

    # This will broadcast if root_clade_idx has shape [N_gene_trees]
    root_probs = Pi[root_clade_idx, :]
    #print(f"root_probs: {root_probs}")
    # print(f"Pi: {torch.exp(Pi).sum(dim=-1)}")
    # We remove log(|S|) because we assume a uniform prior over the root species
    # The logsumexp will still work if root_probs has shape [N_gene_trees, S]
    numerator = torch.logsumexp(root_probs, dim=-1) - math.log(Pi.shape[-1])
    #print(f"numerator: {numerator}")
    # Will still work if E has shape [N_gene_trees, S]
    denominator = torch.log((1-torch.exp(E).mean(dim=-1)))
    #print(f"denominator: {denominator}")
    # result is either a scalar or has shape [N_gene_trees]
    # We can sum over it or use mean() if we want to jointly optimize multiple gene trees
    #print(f"result: {numerator - denominator}")
    # print("Warning: I don't know why likelihood is different from ALE." \
    # "The ALE loglikelihood if p_S=1 is equal to the log of the number of possible root splits, but" \
    #     "it should be equal to log(n_root_splits) + log(n_sp_branches) according to the formula in the " \
    #     "paper?")
    # To verify.
    return -(numerator - denominator)
