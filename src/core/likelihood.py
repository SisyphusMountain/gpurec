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
from .log2_utils import logsumexp2, logaddexp2
from .kernels.wave_step import wave_step_fused, wave_pibar_step_fused, wave_step_uniform_fused
from .kernels.dts_fused import dts_fused

# Try to import logmatmul for fast dense log-space matmul
try:
    import sys as _sys
    from pathlib import Path as _Path
    _logmatmul_dir = str(_Path(__file__).resolve().parents[2] / 'logmatmul')
    if _logmatmul_dir not in _sys.path:
        _sys.path.insert(0, _logmatmul_dir)
    from src.dense import logspace_matmul as _logspace_matmul
    from src.autograd import LogspaceMatmulFn
    _HAS_LOGMATMUL = True
except ImportError:
    _HAS_LOGMATMUL = False
    _logspace_matmul = None

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
            out.append(logsumexp2(x[s:e], dim=0))
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
        if _HAS_LOGMATMUL and Pi.is_cuda:
            # Fast path: fused log2-space matmul via logmatmul kernel
            # transfer_mat_T is exp2(log_transfer_mat - max_transfer_mat).T
            # We need transfer_mat (not transposed) as M [S, S] in linear space
            with _nvtx_here("Pi: logmatmul Pibar"):
                transfer_mat = transfer_mat_T.T  # [S, S] linear space
                Pi_T = Pi.T.contiguous()          # [S, C] log2-space
                Pibar_T = LogspaceMatmulFn.apply(transfer_mat, Pi_T, "ieee")  # [S, C]
                Pibar = Pibar_T.T + max_transfer_mat.squeeze(-1)  # [C, S]
        else:
            # CPU fallback: manual stabilized matmul
            with _nvtx_here("Pi: Pi_max = max(Pi, dim=1)"):
                Pi_max = torch.max(Pi, dim=1, keepdim=True).values # [C, 1]
            with _nvtx_here("Pi: Pi - Pi_max"):
                Pi_minus = Pi - Pi_max
            with _nvtx_here("Pi: exp2(Pi - Pi_max)"):
                Pi_linear = torch.exp2(Pi_minus)  # [C, S]
            with _nvtx_here("Pi: mm(Pi_linear, transfer_T)"):
                Pibar_linear = Pi_linear.mm(transfer_mat_T)  # [C, S]
            with _nvtx_here("Pi: log2(Pibar_linear)"):
                Pibar_log = torch.log2(Pibar_linear)
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

    with _nvtx_here("Pi: logaddexp2(DTS_reduced, DTS_L_term)"):
        return logaddexp2(DTS_reduced, DTS_L_term)

def E_step(E, sp_P_idx, sp_child12_idx, log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat, pibar_mode='dense'):
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
        # Uniform: Ebar[s] = logsumexp2(E) + mt[s]
        # O(S) instead of O(S^2) matvec
        lse_E = logsumexp2(E, dim=-1, keepdim=True)  # [1] or [N, 1]
        Ebar = lse_E + max_transfer_mat  # broadcasts to [S] or [N, S]
    else:
        # Dense: full matvec with [S,S] transfer matrix
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
                          pibar_mode='dense'):

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
        Pi = torch.full((C, S), -1000.0, dtype=dtype, device=device)
        Pi[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0

    converged_iter = max_iters
    log_2 = torch.tensor([1.0], dtype=dtype, device=device)  # log2(2) = 1

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


def _compute_Pibar_wave(Pi_W, transfer_mat, max_transfer_mat):
    """Compute Pibar for a subset of clades (a wave).

    Args:
        Pi_W: [|W|, S] log2-space, Pi values for this wave's clades
        transfer_mat: [S, S] linear-space transfer matrix (exp2-scaled)
        max_transfer_mat: [S] or [S, 1] log2-space column maxima

    Returns:
        Pibar_W: [|W|, S] log2-space
    """
    if _HAS_LOGMATMUL and Pi_W.is_cuda:
        Pi_W_T = Pi_W.T.contiguous()  # [S, |W|]
        Pibar_W_T = LogspaceMatmulFn.apply(transfer_mat, Pi_W_T, "ieee")  # [S, |W|]
        Pibar_W = Pibar_W_T.T + max_transfer_mat.squeeze(-1)
    else:
        Pi_max = torch.max(Pi_W, dim=1, keepdim=True).values
        Pi_linear = torch.exp2(Pi_W - Pi_max)
        Pibar_linear = Pi_linear @ transfer_mat.T
        Pibar_W = torch.log2(Pibar_linear) + Pi_max + max_transfer_mat.squeeze(-1)
    return Pibar_W


def _compute_Pibar_wave_compressed(topk_idx, topk_vals, transfer_mat_full, max_transfer_mat):
    """Compressed Pibar using top-k sparsification (Phase C).

    Instead of full S×S matmul, uses only k indices per clade.
    Reads k²×|W| entries from transfer_mat instead of S²×|W|.

    Args:
        topk_idx: [k, |W|] int32, row indices of top-k per clade
        topk_vals: [k, |W|] float32, log2-space values at those positions
        transfer_mat_full: [S, S] linear-space full transfer matrix (with max NOT subtracted)
        max_transfer_mat: [S] or [S, 1] log2-space column maxima

    Returns:
        Pibar_compressed: [k, |W|] float32, log2-space Pibar at top-k positions
    """
    if not _HAS_LOGMATMUL:
        raise RuntimeError("logmatmul required for compressed Pibar")
    from src.compressed import logspace_matmul_compressed
    # The compressed kernel computes log2(M @ 2^X) restricted to k indices
    Pibar_compressed = logspace_matmul_compressed(topk_idx, topk_vals, transfer_mat_full)
    # Add back the max_transfer_mat at the top-k row positions
    if max_transfer_mat.ndim == 2:
        max_transfer_mat = max_transfer_mat.squeeze(-1)
    # Gather max_transfer_mat at the topk row indices
    max_at_topk = max_transfer_mat[topk_idx.long()]  # [k, |W|]
    return Pibar_compressed + max_at_topk


def Pi_wave_forward(
    waves,
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
    transfer_mat,
    max_transfer_mat,
    device,
    dtype,
    *,
    phases=None,
    local_iters: int = 50,
    local_tolerance: float = 1e-3,
):
    """Wave-based forward pass for Pi computation (phased).

    Processes clades wave by wave in topological order:
      Phase 1 (leaves): No splits, no cross-clade DTS. Only DTS_L self-loop.
      Phase 2 (internal): DTS cross-clade + DTS_L self-loop iteration.
      Phase 3 (root): Massive DTS reduction, single clade.

    If phases is None, all waves are treated as phase 2 (legacy behavior).
    """
    C = int(ccp_helpers['C'])
    S = int(species_helpers['S'])

    split_leftrights = ccp_helpers['split_leftrights_sorted']
    log_split_probs = ccp_helpers['log_split_probs_sorted'].unsqueeze(1).contiguous()
    N_splits = ccp_helpers['N_splits']
    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers['s_C12_indexes']

    clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
    clade_species_map[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0

    _PI_INIT = torch.finfo(dtype).min  # ~-3.4e38; exp2(this)=0 exactly
    Pi = torch.full((C, S), _PI_INIT, dtype=dtype, device=device)
    Pi[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0
    Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

    split_parents = ccp_helpers.get('split_parents_sorted', None)
    if split_parents is None:
        split_parents = _reconstruct_split_parents(ccp_helpers)

    lefts = split_leftrights[:N_splits]
    rights = split_leftrights[N_splits:]

    # Default phases: treat all as phase 2
    if phases is None:
        phases = [2] * len(waves)

    # --- Precompute per-wave data as tensors (no Python dicts in hot loop) ---
    parent_to_splits = {}
    for i in range(N_splits):
        p = int(split_parents[i].item())
        parent_to_splits.setdefault(p, []).append(i)

    # Precompute constant DTS_L terms (species-level, don't change across iterations)
    DL_const = 1.0 + log_pD + E           # [S] — log_2 + log_pD + E
    SL1_const = log_pS + E_s2             # [S]
    SL2_const = log_pS + E_s1             # [S]
    leaf_term = log_pS + clade_species_map  # [C, S]

    # Precompute species-tree child indices
    sp_child1 = torch.full((S,), S, dtype=torch.long, device=device)
    sp_child2 = torch.full((S,), S, dtype=torch.long, device=device)
    for i in range(len(sp_P_idx)):
        p = int(sp_P_idx[i].item())
        c = int(sp_c12_idx[i].item())
        if p < S:
            sp_child1[p] = c
        else:
            sp_child2[p - S] = c

    # Precompute per-wave: wave_ids tensor, split indices, DTS reduction mapping
    wave_data = []
    for wave_ids in waves:
        if not wave_ids:
            wave_data.append(None)
            continue
        wt = torch.tensor(wave_ids, dtype=torch.long, device=device)
        W = len(wave_ids)

        wsids = []
        for c in wave_ids:
            if c in parent_to_splits:
                wsids.extend(parent_to_splits[c])

        if wsids:
            wst = torch.tensor(wsids, dtype=torch.long, device=device)
            n_ws = len(wsids)
            sl = lefts[wst]
            sr = rights[wst]
            lr = torch.cat([sl, sr])
            wlsp = log_split_probs[wst]
            wsp = split_parents[wst]
            clade_to_wi = torch.empty(C, dtype=torch.long, device=device)
            clade_to_wi[wt] = torch.arange(W, device=device)
            reduce_idx = clade_to_wi[wsp]
            wave_data.append((wt, W, True, wst, n_ws, sl, sr, lr, wlsp, reduce_idx))
        else:
            wave_data.append((wt, W, False, None, 0, None, None, None, None, None))

    total_iters = 0
    mt_squeezed = max_transfer_mat.squeeze(-1)
    transfer_mat_c = transfer_mat.contiguous()
    transfer_mat_T = transfer_mat.T.contiguous()

    # Precompute leaf terms per wave
    wave_leaf_terms = []
    for wd in wave_data:
        if wd is None:
            wave_leaf_terms.append(None)
        else:
            wave_leaf_terms.append(leaf_term[wd[0]].contiguous())

    # Decide strategy based on S:
    # Small S (<=256): per-wave convergence with fused Triton kernel (Pibar in kernel)
    # Large S (>256): per-wave convergence with cuBLAS for Pibar
    use_global_pibar = (S > 256)

    with _nvtx_range("Pi wave forward"):
        if use_global_pibar:
            # --- Large S: per-wave convergence with cuBLAS Pibar ---
            # Process waves in topological order. For each wave:
            #   1. Compute DTS cross-clade terms ONCE (children already converged)
            #   2. Iterate self-loop (Pibar + DTS_L) until convergence
            #   3. Write converged Pi back, move to next wave
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

            for wi, wd in enumerate(wave_data):
                if wd is None:
                    continue
                wt, W, has_splits, wst, n_ws, sl, sr, lr, wlsp, reduce_idx = wd
                leaf_wt = wave_leaf_terms[wi]

                # 1. DTS cross-clade: computed ONCE per wave (children frozen)
                if has_splits:
                    # Use fused kernel reading from full Pi/Pibar
                    dts_term = dts_fused(
                        Pi, Pibar, sl, sr,
                        sp_child1, sp_child2,
                        log_pD, log_pS, wlsp,
                    )  # [n_ws, S]
                    # Reduce splits → per-wave-clade [W, S]
                    reduce_exp = reduce_idx.unsqueeze(1).expand_as(dts_term)
                    dts_r = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
                    dts_r.scatter_reduce_(0, reduce_exp, dts_term, reduce='amax', include_self=True)
                    dts_max = dts_r.clone()
                    dts_sum = torch.zeros((W, S), device=device, dtype=dtype)
                    dts_sum.scatter_add_(0, reduce_exp, torch.exp2(dts_term - dts_max[reduce_idx]))
                    dts_r = torch.log2(dts_sum) + dts_max
                else:
                    dts_r = None

                # 2. Iterate self-loop until convergence
                for local_iter in range(local_iters):
                    total_iters += 1
                    Pi_W = Pi[wt].contiguous()  # [W, S]

                    # Pibar via cuBLAS TF32: [W, S] @ [S, S]
                    Pi_max = Pi_W.max(dim=1, keepdim=True).values
                    Pibar_W = torch.log2(torch.exp2(Pi_W - Pi_max) @ transfer_mat_T) + Pi_max + mt_squeezed

                    # DTS_L update
                    Pi_new = wave_step_fused(
                        Pi_W, Pibar_W,
                        DL_const, Ebar, E, SL1_const, SL2_const,
                        sp_child1, sp_child2, leaf_wt, dts_r,
                    )
                    Pi[wt] = Pi_new
                    Pibar[wt] = Pibar_W

                    # Check convergence every iteration after minimum warmup
                    if local_iter >= 3:
                        significant = Pi_new > -100.0
                        if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < local_tolerance:
                            break

            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        else:
            # --- Small S: per-wave convergence with fully fused Triton kernel ---
            for wi, wd in enumerate(wave_data):
                if wd is None:
                    continue
                wt, W, has_splits, wst, n_ws, sl, sr, lr, wlsp, reduce_idx = wd

                if has_splits:
                    dts_r = _compute_DTS_reduced(
                        Pi, Pibar, lr, n_ws, S, W, log_pD, log_pS,
                        sp_child1, sp_child2, wlsp, reduce_idx, device, dtype)
                else:
                    dts_r = None

                leaf_wt = wave_leaf_terms[wi]
                Pibar_W_buf = torch.empty(W, S, device=device, dtype=dtype)

                for local_iter in range(local_iters):
                    total_iters += 1
                    Pi_W = Pi[wt].contiguous()
                    Pi_new = wave_pibar_step_fused(
                        Pi_W, transfer_mat_c, mt_squeezed,
                        DL_const, Ebar, E, SL1_const, SL2_const,
                        sp_child1, sp_child2, leaf_wt,
                        Pibar_W_buf, dts_r,
                    )
                    Pi[wt] = Pi_new
                    Pibar[wt] = Pibar_W_buf
                    if local_iter >= 3:
                        significant = Pi_new > -100.0
                        if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < local_tolerance:
                            break

    return {'Pi': Pi, 'clade_species_map': clade_species_map, 'iterations': total_iters}


def _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2, log_pD, log_pS,
                       S, device, dtype):
    """Compute DTS cross-clade terms and reduce to [W, S] for one wave.

    Splits are pre-sorted: single-split clades (eq1) first, multi-split (ge2) after.
    eq1: direct permuted copy (no reduction needed).
    ge2: fused seg_logsumexp via CSR pointers.
    """
    sl = meta['sl']
    sr = meta['sr']
    wlsp = meta['log_split_probs']
    W = meta['W']

    dts_term = dts_fused(
        Pi, Pibar, sl, sr,
        sp_child1, sp_child2,
        log_pD, log_pS, wlsp,
    )  # [n_ws, S]

    NEG_INF = float('-inf')
    dts_r = torch.full((W, S), NEG_INF, device=device, dtype=dtype)

    n_eq1 = meta.get('n_eq1', 0)
    n_ge2_clades = meta.get('n_ge2_clades', 0)

    # Single-split clades: direct permuted copy
    if n_eq1 > 0:
        dts_r[meta['eq1_reduce_idx']] = dts_term[:n_eq1]

    # Multi-split clades: fused segmented logsumexp
    if n_ge2_clades > 0:
        ge2_term = dts_term[n_eq1:].contiguous()
        y_ge2 = seg_logsumexp(ge2_term, meta['ge2_ptr'])  # [n_ge2_clades, S]
        dts_r[meta['ge2_parent_ids']] = y_ge2

    return dts_r


def _compute_Pibar_inline(Pi_W, transfer_mat_T, mt_squeezed, pibar_mode):
    """Compute Pibar for a wave, either via dense matmul or uniform approximation."""
    Pi_max = Pi_W.max(dim=1, keepdim=True).values
    if pibar_mode == 'uniform':
        Pi_exp = torch.exp2(Pi_W - Pi_max)
        row_sum = Pi_exp.sum(dim=1, keepdim=True)
        Pibar_W = torch.log2(row_sum - Pi_exp) + Pi_max + mt_squeezed
    else:
        Pibar_W = torch.log2(torch.exp2(Pi_W - Pi_max) @ transfer_mat_T) + Pi_max + mt_squeezed
    return Pibar_W


def Pi_wave_forward_v2(
    wave_layout,
    species_helpers,
    E,
    Ebar,
    E_s1,
    E_s2,
    log_pS,
    log_pD,
    log_pL,
    transfer_mat,
    max_transfer_mat,
    device,
    dtype,
    *,
    local_iters: int = 50,
    local_tolerance: float = 1e-3,
    fixed_iters: int | None = None,
    overlap_streams: bool = False,
    pibar_mode: str = 'dense',
):
    """Wave-based Pi forward pass with wave-ordered layout (v2).

    Clades are permuted so each wave occupies a contiguous block of Pi[ws:we].
    The self-loop uses zero-copy views instead of gather/scatter.

    Args:
        wave_layout: dict from build_wave_layout() containing permuted indices
                     and precomputed per-wave metadata
        species_helpers: species tree helpers dict
        E, Ebar, E_s1, E_s2: converged E vectors [S]
        log_pS, log_pD, log_pL: event probabilities [S]
        transfer_mat: [S, S] linear-space transfer matrix (None when pibar_mode='uniform')
        max_transfer_mat: [S] log2-space column maxima
        device, dtype: target device and float dtype
        local_iters: max iterations per wave self-loop
        local_tolerance: convergence threshold
        fixed_iters: if set, use fixed iteration count (no convergence check / GPU sync)
        overlap_streams: if True, overlap DTS preparation for wave k+1 with
                         self-loop of wave k via a secondary CUDA stream
        pibar_mode: 'dense' (cuBLAS matmul) or 'uniform' (O(W*S) approximation
                    using nearly-uniform transfer matrix structure)

    Returns:
        dict with 'Pi' (in original clade order), 'clade_species_map', 'iterations'
    """
    ccp_helpers = wave_layout['ccp_helpers']
    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']
    wave_metas = wave_layout['wave_metas']
    wave_starts = wave_layout['wave_starts']

    C = int(ccp_helpers['C'])
    S = int(species_helpers['S'])

    _PI_INIT = torch.finfo(dtype).min
    Pi = torch.full((C, S), _PI_INIT, dtype=dtype, device=device)
    Pi[leaf_row_index, leaf_col_index] = 0.0
    Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

    # Precompute species child indices (vectorized to avoid O(S) GPU syncs)
    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers['s_C12_indexes']
    p_cpu = sp_P_idx.cpu().long()
    c_cpu = sp_c12_idx.cpu().long()
    mask_c1 = p_cpu < S
    sp_child1_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child2_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child1_cpu[p_cpu[mask_c1]] = c_cpu[mask_c1]
    sp_child2_cpu[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]
    sp_child1 = sp_child1_cpu.to(device)
    sp_child2 = sp_child2_cpu.to(device)

    # Precompute constant DTS_L terms
    DL_const = 1.0 + log_pD + E           # [S]
    SL1_const = log_pS + E_s2             # [S]
    SL2_const = log_pS + E_s1             # [S]

    if pibar_mode == 'uniform':
        # Avoid [C, S] clade_species_map and leaf_term allocations
        # leaf_wt is computed per-wave instead
        clade_species_map = None
        leaf_term = None
        transfer_mat_T = None
        transfer_mat_c = None
    else:
        clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
        clade_species_map[leaf_row_index, leaf_col_index] = 0.0
        leaf_term = log_pS + clade_species_map  # [C, S]
        transfer_mat_T = transfer_mat.T.contiguous()
        transfer_mat_c = transfer_mat.contiguous()

    mt_squeezed = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 1 else max_transfer_mat

    def _get_leaf_wt(ws, we):
        """Get leaf_wt [W, S] for a wave, either via slice or per-wave construction."""
        if leaf_term is not None:
            return leaf_term[ws:we]
        W = we - ws
        lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
        mask = (leaf_row_index >= ws) & (leaf_row_index < we)
        if mask.any():
            lwt[leaf_row_index[mask] - ws, leaf_col_index[mask]] = 0.0
        return log_pS + lwt

    use_global_pibar = (S > 256)
    n_waves = len(wave_metas)

    # Determine iteration strategy
    use_fixed = fixed_iters is not None
    n_iters = fixed_iters if use_fixed else local_iters
    min_warmup = 0 if use_fixed else 3

    total_iters = 0

    with _nvtx_range("Pi wave forward v2"):
        if use_global_pibar:
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

            if overlap_streams and n_waves > 1:
                # --- Stream-overlapped path ---
                # DTS for wave k+1 is prepared on stream_prep while
                # wave k's self-loop runs on the default stream.
                stream_main = torch.cuda.current_stream(device)
                stream_prep = torch.cuda.Stream(device=device)

                # Prepare DTS for wave 0 on main stream (no overlap yet)
                meta0 = wave_metas[0]
                if meta0['has_splits']:
                    dts_r_current = _compute_dts_cross(
                        Pi, Pibar, meta0, sp_child1, sp_child2,
                        log_pD, log_pS, S, device, dtype)
                else:
                    dts_r_current = None

                # Pending DTS result for the next wave (computed on stream_prep)
                dts_r_next = None
                event_prep_done = None

                for wi in range(n_waves):
                    meta = wave_metas[wi]
                    ws = meta['start']
                    we = meta['end']
                    W = meta['W']

                    # Wait for any pending DTS prep from previous iteration
                    if wi > 0 and event_prep_done is not None:
                        stream_main.wait_event(event_prep_done)
                        dts_r_current = dts_r_next
                        dts_r_next = None

                    dts_r = dts_r_current
                    leaf_wt = _get_leaf_wt(ws, we)

                    # Kick off DTS prep for wave wi+1 on stream_prep
                    if wi + 1 < n_waves:
                        meta_next = wave_metas[wi + 1]
                        if meta_next['has_splits']:
                            # Record event so stream_prep waits for wave wi's
                            # self-loop to write converged Pi/Pibar values
                            # that the next wave's DTS reads from.
                            # We record AFTER the self-loop below.
                            pass  # event recorded after self-loop

                    # Self-loop
                    if pibar_mode == 'uniform':
                        for local_iter in range(n_iters):
                            total_iters += 1
                            Pi_new, max_diff = wave_step_uniform_fused(
                                Pi, Pibar, ws, W, S,
                                mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                                sp_child1, sp_child2, leaf_wt, dts_r,
                            )
                            Pi[ws:we] = Pi_new
                            if not use_fixed and local_iter >= min_warmup:
                                if max_diff < local_tolerance:
                                    break
                    else:
                        for local_iter in range(n_iters):
                            total_iters += 1
                            Pi_W = Pi[ws:we]

                            Pibar_W = _compute_Pibar_inline(Pi_W, transfer_mat_T, mt_squeezed, pibar_mode)

                            Pi_new = wave_step_fused(
                                Pi_W, Pibar_W,
                                DL_const, Ebar, E, SL1_const, SL2_const,
                                sp_child1, sp_child2, leaf_wt, dts_r,
                            )

                            if not use_fixed and local_iter >= min_warmup:
                                significant = Pi_new > -100.0
                                if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < local_tolerance:
                                    Pi[ws:we] = Pi_new
                                    Pibar[ws:we] = Pibar_W
                                    break

                            Pi[ws:we] = Pi_new
                            Pibar[ws:we] = Pibar_W

                    # Now that self-loop is done, launch DTS prep for next wave
                    if wi + 1 < n_waves:
                        meta_next = wave_metas[wi + 1]
                        if meta_next['has_splits']:
                            # Record event on main stream so prep stream
                            # knows Pi/Pibar are ready
                            event_self_done = torch.cuda.Event()
                            event_self_done.record(stream_main)
                            with torch.cuda.stream(stream_prep):
                                stream_prep.wait_event(event_self_done)
                                dts_r_next = _compute_dts_cross(
                                    Pi, Pibar, meta_next, sp_child1, sp_child2,
                                    log_pD, log_pS, S, device, dtype)
                                event_prep_done = torch.cuda.Event()
                                event_prep_done.record(stream_prep)
                        else:
                            dts_r_next = None
                            event_prep_done = None
            else:
                # --- Non-overlapped path ---
                for wi in range(n_waves):
                    meta = wave_metas[wi]
                    ws = meta['start']
                    we = meta['end']
                    W = meta['W']

                    if meta['has_splits']:
                        dts_r = _compute_dts_cross(
                            Pi, Pibar, meta, sp_child1, sp_child2,
                            log_pD, log_pS, S, device, dtype)
                    else:
                        dts_r = None

                    leaf_wt = _get_leaf_wt(ws, we)

                    if pibar_mode == 'uniform':
                        # Fused path: Pibar + wave_step + convergence in one kernel
                        for local_iter in range(n_iters):
                            total_iters += 1
                            Pi_new, max_diff = wave_step_uniform_fused(
                                Pi, Pibar, ws, W, S,
                                mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                                sp_child1, sp_child2, leaf_wt, dts_r,
                            )
                            Pi[ws:we] = Pi_new
                            if not use_fixed and local_iter >= min_warmup:
                                if max_diff < local_tolerance:
                                    break
                    else:
                        for local_iter in range(n_iters):
                            total_iters += 1
                            Pi_W = Pi[ws:we]

                            Pibar_W = _compute_Pibar_inline(Pi_W, transfer_mat_T, mt_squeezed, pibar_mode)

                            Pi_new = wave_step_fused(
                                Pi_W, Pibar_W,
                                DL_const, Ebar, E, SL1_const, SL2_const,
                                sp_child1, sp_child2, leaf_wt, dts_r,
                            )

                            if not use_fixed and local_iter >= min_warmup:
                                significant = Pi_new > -100.0
                                if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < local_tolerance:
                                    Pi[ws:we] = Pi_new
                                    Pibar[ws:we] = Pibar_W
                                    break

                            Pi[ws:we] = Pi_new
                            Pibar[ws:we] = Pibar_W

            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        else:
            # --- Small S: fused Triton kernel ---
            for wi in range(n_waves):
                meta = wave_metas[wi]
                ws = meta['start']
                we = meta['end']
                W = meta['W']

                if meta['has_splits']:
                    sl = meta['sl']
                    sr = meta['sr']
                    lr = torch.cat([sl, sr])
                    dts_r = _compute_DTS_reduced(
                        Pi, Pibar, lr, meta['n_ws'], S, W, log_pD, log_pS,
                        sp_child1, sp_child2, meta['log_split_probs'],
                        meta['reduce_idx'], device, dtype)
                else:
                    dts_r = None

                leaf_wt = _get_leaf_wt(ws, we).contiguous()

                Pibar_W_buf = torch.empty(W, S, device=device, dtype=dtype)

                for local_iter in range(n_iters):
                    total_iters += 1
                    Pi_W = Pi[ws:we]

                    Pi_new = wave_pibar_step_fused(
                        Pi_W, transfer_mat_c, mt_squeezed,
                        DL_const, Ebar, E, SL1_const, SL2_const,
                        sp_child1, sp_child2, leaf_wt,
                        Pibar_W_buf, dts_r,
                    )

                    if not use_fixed and local_iter >= min_warmup:
                        significant = Pi_new > -100.0
                        if not significant.any() or torch.abs(Pi_new - Pi_W)[significant].max().item() < local_tolerance:
                            Pi[ws:we] = Pi_new
                            Pibar[ws:we] = Pibar_W_buf
                            break

                    Pi[ws:we] = Pi_new
                    Pibar[ws:we] = Pibar_W_buf

    # Unpermute Pi back to original clade order
    # Pi is in new (wave-ordered) space; perm[orig] = new, so Pi[perm] reorders to original
    perm = wave_layout['perm']
    Pi_orig = Pi[perm]
    clade_species_map_orig = clade_species_map[perm] if clade_species_map is not None else None

    return {
        'Pi': Pi_orig,
        'clade_species_map': clade_species_map_orig,
        'iterations': total_iters,
        'Pi_wave_ordered': Pi,
        'Pibar_wave_ordered': Pibar,
    }


def _compute_DTS_reduced(Pi, Pibar, lr, n_ws, S, W, log_pD, log_pS,
                          sp_child1, sp_child2, wlsp, reduce_idx, device, dtype):
    """Compute DTS cross-clade terms and reduce to per-wave-clade."""
    Pi_lr = Pi[lr]
    Pibar_lr = Pibar[lr]
    neg_inf_col = torch.full((2 * n_ws, 1), NEG_INF, device=device, dtype=dtype)
    Pi_lr_pad = torch.cat([Pi_lr, neg_inf_col], dim=1)
    DTS = torch.empty((5, n_ws, S), device=device, dtype=dtype)
    DTS[0] = log_pD + Pi_lr[:n_ws] + Pi_lr[n_ws:]
    DTS[1] = Pi_lr[:n_ws] + Pibar_lr[n_ws:]
    DTS[2] = Pi_lr[n_ws:] + Pibar_lr[:n_ws]
    DTS[3] = log_pS + Pi_lr_pad[:n_ws, sp_child1] + Pi_lr_pad[n_ws:, sp_child2]
    DTS[4] = log_pS + Pi_lr_pad[n_ws:, sp_child1] + Pi_lr_pad[:n_ws, sp_child2]
    DTS_term = wlsp + logsumexp2(DTS, dim=0)
    reduce_exp = reduce_idx.unsqueeze(1).expand_as(DTS_term)
    DTS_reduced = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    DTS_reduced.scatter_reduce_(0, reduce_exp, DTS_term, reduce='amax', include_self=True)
    DTS_max = DTS_reduced.clone()
    DTS_sum = torch.zeros((W, S), device=device, dtype=dtype)
    DTS_sum.scatter_add_(0, reduce_exp, torch.exp2(DTS_term - DTS_max[reduce_idx]))
    return torch.log2(DTS_sum) + DTS_max


def _reconstruct_split_parents(ccp_helpers):
    """Reconstruct split_parents_sorted from seg_parent_ids and segment structure."""
    seg_parent_ids = ccp_helpers['seg_parent_ids']
    num_ge2 = int(ccp_helpers['num_segs_ge2'])
    num_eq1 = int(ccp_helpers['num_segs_eq1'])
    end_rows_ge2 = int(ccp_helpers['end_rows_ge2'])
    ptr_ge2 = ccp_helpers['ptr_ge2']
    N_splits = int(ccp_helpers['N_splits'])

    split_parents = torch.empty(N_splits, dtype=torch.long, device=seg_parent_ids.device)

    # >=2 splits per clade
    for i in range(num_ge2):
        start = int(ptr_ge2[i].item())
        end = int(ptr_ge2[i + 1].item())
        parent_id = seg_parent_ids[i]
        split_parents[start:end] = parent_id

    # =1 split per clade
    for i in range(num_eq1):
        row = end_rows_ge2 + i
        parent_id = seg_parent_ids[num_ge2 + i]
        split_parents[row] = parent_id

    return split_parents


def compute_gradient_bounds(
    Pi,
    ccp_helpers,
    root_clade_idx,
    threshold=-20.0,
    wave_metas=None,
):
    """Compute gradient bounds for pruning clades from the backward pass.

    Propagates gradient bounds from root downward. When wave_metas is provided,
    processes waves in reverse order (root→leaves) for correct topological ordering
    and uses vectorized scatter_reduce instead of a Python loop.

    Args:
        Pi: [C, S] log2-space, converged Pi values
        ccp_helpers: dict with split information
        root_clade_idx: int or Long tensor, root clade index(es)
        threshold: log2-space threshold for pruning
        wave_metas: optional list of per-wave metadata dicts (from build_wave_layout)

    Returns:
        grad_bound: [C] float32, upper bound on gradient magnitude per clade
        pruned_mask: [C] bool, True for clades that can be pruned
    """
    C = Pi.shape[0]
    device = Pi.device
    dtype = Pi.dtype

    # Clade scores: max_e Pi[gamma, e]
    s = Pi.max(dim=1).values  # [C]

    # Initialize gradient bounds
    grad_bound = torch.full((C,), NEG_INF, device=device, dtype=dtype)
    if isinstance(root_clade_idx, int):
        grad_bound[root_clade_idx] = 0.0
    else:
        grad_bound[root_clade_idx] = 0.0

    if wave_metas is not None:
        # Vectorized wave-by-wave propagation (root→leaves = reverse wave order)
        for k in range(len(wave_metas) - 1, -1, -1):
            meta = wave_metas[k]
            if not meta['has_splits']:
                continue
            sl = meta['sl']  # [n_ws] left child indices
            sr = meta['sr']  # [n_ws] right child indices
            reduce_idx = meta['reduce_idx']  # [n_ws] wave-local parent index
            ws = meta['start']

            # Get parent gradient bounds for each split
            gb_parents = grad_bound[reduce_idx + ws]  # [n_ws]

            # new bound for left child = gb_parent + s[right sibling]
            new_l = gb_parents + s[sr]
            # new bound for right child = gb_parent + s[left sibling]
            new_r = gb_parents + s[sl]

            # Accumulate via logaddexp into grad_bound at child positions
            # Use scatter_reduce 'amax' as a first pass, then refine with logsumexp
            # For simplicity and correctness, use logaddexp2 element-wise
            grad_bound[sl] = logaddexp2(grad_bound[sl], new_l)
            grad_bound[sr] = logaddexp2(grad_bound[sr], new_r)
    else:
        # Fallback: Python loop (for non-wave callers)
        N_splits = int(ccp_helpers['N_splits'])
        split_leftrights = ccp_helpers['split_leftrights_sorted']
        lefts = split_leftrights[:N_splits]
        rights = split_leftrights[N_splits:]

        split_parents = ccp_helpers.get('split_parents_sorted', None)
        if split_parents is None:
            split_parents = _reconstruct_split_parents(ccp_helpers)

        for i in range(N_splits):
            p = int(split_parents[i].item()) if torch.is_tensor(split_parents[i]) else int(split_parents[i])
            l_idx = int(lefts[i].item()) if torch.is_tensor(lefts[i]) else int(lefts[i])
            r_idx = int(rights[i].item()) if torch.is_tensor(rights[i]) else int(rights[i])

            if grad_bound[p].item() < threshold:
                continue

            gb_p = grad_bound[p]
            new_l = gb_p + s[l_idx]
            new_r = gb_p + s[r_idx]

            grad_bound[l_idx] = logaddexp2(grad_bound[l_idx], new_r)
            grad_bound[r_idx] = logaddexp2(grad_bound[r_idx], new_l)

    pruned_mask = grad_bound < threshold
    return grad_bound, pruned_mask


def _self_loop_differentiable(
    Pi_W, mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_wt, dts_r, S,
):
    """Pure-PyTorch differentiable self-loop step (uniform Pibar + DTS_L).

    Computes one iteration of g_k(Pi_W) = logaddexp2(dts_r, DTS_L(Pi_W, Pibar(Pi_W))).
    Used by the backward pass to build VJP closures via torch.func.vjp.

    Args:
        Pi_W: [W, S] log2-space, requires_grad
        mt_squeezed: [S] max transfer mat
        DL_const, Ebar, E, SL1_const, SL2_const: [S] precomputed constants
        sp_child1, sp_child2: [S] species child indices
        leaf_wt: [W, S] leaf term
        dts_r: [W, S] or None, cross-clade DTS (frozen)
        S: int

    Returns:
        Pi_new: [W, S] log2-space
    """
    W = Pi_W.shape[0]

    # --- Pibar (uniform approximation) ---
    Pi_max = Pi_W.max(dim=1, keepdim=True).values  # [W, 1]
    Pi_exp = torch.exp2(Pi_W - Pi_max)              # [W, S]
    row_sum = Pi_exp.sum(dim=1, keepdim=True)        # [W, 1]
    Pibar_W = torch.log2(row_sum - Pi_exp) + Pi_max + mt_squeezed  # [W, S]

    # --- Species children of Pi_W ---
    # Pi_s1[w, s] = Pi_W[w, child1[s]], Pi_s2[w, s] = Pi_W[w, child2[s]]
    # Sentinel index S → need padding
    Pi_W_pad = torch.cat([Pi_W, torch.full((W, 1), NEG_INF, device=Pi_W.device, dtype=Pi_W.dtype)], dim=1)
    Pi_s1 = Pi_W_pad[:, sp_child1.long()]  # [W, S]
    Pi_s2 = Pi_W_pad[:, sp_child2.long()]  # [W, S]

    # --- DTS_L: 6 terms ---
    DTS_L = torch.stack([
        DL_const.unsqueeze(0) + Pi_W,                    # DL
        Pi_W + Ebar.unsqueeze(0),                         # TL1
        Pibar_W + E.unsqueeze(0),                         # TL2
        SL1_const.unsqueeze(0) + Pi_s1,                   # SL1
        SL2_const.unsqueeze(0) + Pi_s2,                   # SL2
        leaf_wt,                                           # leaf
    ], dim=0)  # [6, W, S]

    DTS_L_term = logsumexp2(DTS_L, dim=0)  # [W, S]

    if dts_r is not None:
        return logaddexp2(dts_r, DTS_L_term)
    else:
        return DTS_L_term


def _dts_cross_differentiable(
    Pi, Pibar, meta, sp_child1, sp_child2, log_pD, log_pS, S, device, dtype,
):
    """Differentiable DTS cross-clade computation for one wave.

    Same as _compute_dts_cross but uses pure PyTorch ops (no Triton)
    so that torch.func.vjp can trace through it.

    Args:
        Pi, Pibar: [C, S] full tensors (Pi requires_grad for children)
        meta: wave metadata dict
        sp_child1, sp_child2: [S] species child indices
        log_pD, log_pS: scalar or [S] event probabilities
        S: int

    Returns:
        dts_r: [W, S] reduced DTS cross-clade terms
    """
    sl = meta['sl']
    sr = meta['sr']
    wlsp = meta['log_split_probs']  # [n_ws, 1]
    W = meta['W']
    n_ws = sl.shape[0]

    # Gather children
    Pi_l = Pi[sl]        # [n_ws, S]
    Pi_r = Pi[sr]        # [n_ws, S]
    Pibar_l = Pibar[sl]  # [n_ws, S]
    Pibar_r = Pibar[sr]  # [n_ws, S]

    # Species children with sentinel padding
    Pi_pad = torch.cat([Pi, torch.full((Pi.shape[0], 1), NEG_INF, device=device, dtype=dtype)], dim=1)
    Pi_l_s1 = Pi_pad[sl][:, sp_child1.long()]  # [n_ws, S]
    Pi_l_s2 = Pi_pad[sl][:, sp_child2.long()]
    Pi_r_s1 = Pi_pad[sr][:, sp_child1.long()]
    Pi_r_s2 = Pi_pad[sr][:, sp_child2.long()]

    # 5 DTS terms
    DTS = torch.stack([
        log_pD + Pi_l + Pi_r,           # D
        Pi_l + Pibar_r,                  # T (l->r)
        Pi_r + Pibar_l,                  # T (r->l)
        log_pS + Pi_l_s1 + Pi_r_s2,     # S1
        log_pS + Pi_r_s1 + Pi_l_s2,     # S2
    ], dim=0)  # [5, n_ws, S]

    dts_term = wlsp + logsumexp2(DTS, dim=0)  # [n_ws, S]

    # Reduce splits → per-wave-clade [W, S] via differentiable scatter
    reduce_idx = meta['reduce_idx']  # [n_ws]

    # Differentiable reduction: group by parent clade and logsumexp
    # Use index_select + logsumexp per unique parent for autograd compatibility
    reduce_expand = reduce_idx.unsqueeze(1).expand(n_ws, S)  # [n_ws, S]

    # Pad dts_term to [n_ws+1, S] with -inf sentinel for safe indexing
    dts_padded = torch.cat([dts_term, torch.full((1, S), NEG_INF, device=device, dtype=dtype)], dim=0)

    # Build a dense [W, max_splits_per_clade, S] tensor and logsumexp over dim=1
    # For simplicity, use scatter + max stabilization without in-place mutation
    # Approach: compute per-clade logsumexp via a loop over unique parents
    # This is small (W clades per wave) so the loop is cheap
    parts = []
    for w in range(W):
        mask_w = reduce_idx == w
        if mask_w.any():
            parts.append(logsumexp2(dts_term[mask_w], dim=0).unsqueeze(0))  # [1, S]
        else:
            parts.append(torch.full((1, S), NEG_INF, device=device, dtype=dtype))
    dts_r = torch.cat(parts, dim=0)  # [W, S]

    return dts_r


@torch.no_grad()
def Pi_wave_backward_v2(
    wave_layout,
    Pi_star_wave,
    Pibar_star_wave,
    E, Ebar, E_s1, E_s2,
    log_pS, log_pD, log_pL,
    max_transfer_mat,
    species_helpers,
    root_clade_ids_perm,
    device, dtype,
    *,
    neumann_terms=3,
    pruning_threshold=-20.0,
    use_pruning=True,
):
    """Wave-decomposed backward pass for implicit gradient computation.

    Computes dL/dPi via Neumann series per wave (root→leaves), then
    accumulates parameter gradients.

    Args:
        wave_layout: dict from build_wave_layout()
        Pi_star_wave: [C, S] converged Pi in wave-ordered space
        Pibar_star_wave: [C, S] converged Pibar in wave-ordered space
        E, Ebar, E_s1, E_s2: [S] species extinction
        log_pS, log_pD, log_pL: event probabilities
        max_transfer_mat: [S] log2-space
        species_helpers: species tree helpers
        root_clade_ids_perm: Long[F] root clade IDs in wave-ordered space
        device, dtype: target device/dtype
        neumann_terms: number of Neumann series terms (default 3)
        pruning_threshold: log2-space threshold for gradient pruning
        use_pruning: whether to prune negligible-gradient clades

    Returns:
        dict with:
            'v_Pi': [C, S] adjoint vector for Pi (wave-ordered)
            'grad_E': [S] accumulated gradient contribution from Pi adjoint to E
            'grad_log_pS': gradient wrt log_pS
            'grad_log_pD': gradient wrt log_pD
            'grad_max_transfer_mat': gradient wrt max_transfer_mat
    """
    wave_metas = wave_layout['wave_metas']
    wave_starts = wave_layout['wave_starts']
    perm = wave_layout['perm']
    C, S = Pi_star_wave.shape
    K = len(wave_metas)

    # Precompute species child indices
    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers['s_C12_indexes']
    p_cpu = sp_P_idx.cpu().long()
    c_cpu = sp_c12_idx.cpu().long()
    mask_c1 = p_cpu < S
    sp_child1_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child2_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child1_cpu[p_cpu[mask_c1]] = c_cpu[mask_c1]
    sp_child2_cpu[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]
    sp_child1 = sp_child1_cpu.to(device)
    sp_child2 = sp_child2_cpu.to(device)

    mt_squeezed = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 1 else max_transfer_mat

    # Precompute constant terms
    DL_const = 1.0 + log_pD + E
    SL1_const = log_pS + E_s2
    SL2_const = log_pS + E_s1

    # Per-wave leaf term helper
    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']

    def _get_leaf_wt(ws, we):
        W = we - ws
        lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
        mask = (leaf_row_index >= ws) & (leaf_row_index < we)
        if mask.any():
            lwt[leaf_row_index[mask] - ws, leaf_col_index[mask]] = 0.0
        return log_pS + lwt

    # Optional pruning
    pruned_perm = None
    if use_pruning:
        grad_bound, pruned_mask = compute_gradient_bounds(
            Pi_star_wave, {'N_splits': 0}, root_clade_ids_perm,
            threshold=pruning_threshold, wave_metas=wave_metas,
        )
        pruned_perm = pruned_mask

    # Initialize RHS: dL/dPi at root clades
    # logL = -(logsumexp2(Pi[root]) - log2(S) - denom)
    # dL/dPi[root,s] = -exp2(Pi[root,s] - logsumexp2(Pi[root]))
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    for r in root_clade_ids_perm:
        r = int(r)
        root_Pi = Pi_star_wave[r]  # [S]
        lse = logsumexp2(root_Pi, dim=0)  # scalar
        accumulated_rhs[r] = -torch.exp2(root_Pi - lse)

    # Accumulate parameter gradients
    grad_log_pD = torch.zeros_like(log_pD)
    grad_log_pS = torch.zeros_like(log_pS)
    grad_mt = torch.zeros_like(mt_squeezed)
    grad_E_acc = torch.zeros(S, device=device, dtype=dtype)
    grad_Ebar_acc = torch.zeros(S, device=device, dtype=dtype)
    grad_E_s1_acc = torch.zeros(S, device=device, dtype=dtype)
    grad_E_s2_acc = torch.zeros(S, device=device, dtype=dtype)

    # Backward sweep: root→leaves
    for k in range(K - 1, -1, -1):
        meta = wave_metas[k]
        ws = meta['start']
        we = meta['end']
        W = meta['W']

        # Skip entirely pruned waves
        if use_pruning and pruned_perm is not None and pruned_perm[ws:we].all():
            continue

        rhs_k = accumulated_rhs[ws:we].clone()  # [W, S]

        # Skip if RHS is all zero (no gradient flows here)
        if rhs_k.abs().max() == 0:
            continue

        # Get frozen inputs for this wave
        Pi_W_star = Pi_star_wave[ws:we].detach()
        leaf_wt = _get_leaf_wt(ws, we)

        # DTS cross-clade for this wave (frozen, from converged values)
        if meta['has_splits']:
            with torch.no_grad():
                dts_r = _compute_dts_cross(
                    Pi_star_wave, Pibar_star_wave, meta,
                    sp_child1, sp_child2, log_pD, log_pS, S, device, dtype,
                )
        else:
            dts_r = None

        # --- Neumann series: v_k = sum_{n=0}^{N} (J_self^T)^n @ rhs_k ---
        v_k = rhs_k.clone()
        term = rhs_k

        for n in range(neumann_terms):
            # Build VJP through one self-loop step
            Pi_W_req = Pi_W_star.clone().requires_grad_(True)
            with torch.enable_grad():
                Pi_new = _self_loop_differentiable(
                    Pi_W_req, mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                    sp_child1, sp_child2, leaf_wt, dts_r, S,
                )
                # VJP: term_new = (dg/dPi_W)^T @ term
                term_new = torch.autograd.grad(Pi_new, Pi_W_req, grad_outputs=term,
                                               retain_graph=False)[0]
            term = term_new.detach()
            v_k = v_k + term

        # --- Accumulate parameter gradients via VJP of g_k wrt params ---
        _log_pD = log_pD.detach().requires_grad_(True)
        _log_pS = log_pS.detach().requires_grad_(True)
        _E = E.detach().requires_grad_(True)
        _Ebar = Ebar.detach().requires_grad_(True)
        _E_s1 = E_s1.detach().requires_grad_(True)
        _E_s2 = E_s2.detach().requires_grad_(True)
        _mt = mt_squeezed.detach().requires_grad_(True)

        # Recompute DL_const etc. with grad-enabled params
        _DL_const = 1.0 + _log_pD + _E
        _SL1_const = _log_pS + _E_s2
        _SL2_const = _log_pS + _E_s1

        Pi_W_fixed = Pi_W_star.detach()  # fixed at converged value
        with torch.enable_grad():
            Pi_new_param = _self_loop_differentiable(
                Pi_W_fixed, _mt, _DL_const, _Ebar, _E, _SL1_const, _SL2_const,
                sp_child1, sp_child2, leaf_wt, dts_r, S,
            )
            param_grads = torch.autograd.grad(
                Pi_new_param, [_log_pD, _log_pS, _E, _Ebar, _E_s1, _E_s2, _mt],
                grad_outputs=v_k,
                retain_graph=False,
                allow_unused=True,
            )

        if param_grads[0] is not None:
            grad_log_pD = grad_log_pD + param_grads[0]
        if param_grads[1] is not None:
            grad_log_pS = grad_log_pS + param_grads[1]
        if param_grads[2] is not None:
            grad_E_acc = grad_E_acc + param_grads[2]
        if param_grads[3] is not None:
            grad_Ebar_acc = grad_Ebar_acc + param_grads[3]
        if param_grads[4] is not None:
            grad_E_s1_acc = grad_E_s1_acc + param_grads[4]
        if param_grads[5] is not None:
            grad_E_s2_acc = grad_E_s2_acc + param_grads[5]
        if param_grads[6] is not None:
            grad_mt = grad_mt + param_grads[6]

        # --- Cross-clade backward: propagate adjoint to earlier waves ---
        if meta['has_splits']:
            sl = meta['sl']
            sr = meta['sr']

            # VJP of dts_cross wrt Pi/Pibar of children
            # Children are at indices sl, sr in the full Pi_star_wave
            Pi_req = Pi_star_wave.detach().clone().requires_grad_(True)
            Pibar_req = Pibar_star_wave.detach().clone().requires_grad_(True)

            with torch.enable_grad():
                dts_r_diff = _dts_cross_differentiable(
                    Pi_req, Pibar_req, meta,
                    sp_child1, sp_child2, log_pD, log_pS, S, device, dtype,
                )
                # The contribution to Pi_new from dts_r enters via logaddexp2(dts_r, DTS_L)
                # We need d(logaddexp2(dts_r, DTS_L))/d(dts_r) * v_k
                # = exp2(dts_r - Pi_new) * v_k  (softmax weight of the dts_r branch)
                if dts_r is not None:
                    Pi_new_at_star = _self_loop_differentiable(
                        Pi_W_star, mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
                        sp_child1, sp_child2, leaf_wt, dts_r, S,
                    )
                    # Weight of dts_r in the logaddexp2
                    dts_weight = torch.exp2(dts_r - Pi_new_at_star)  # [W, S]
                    dts_grad_output = v_k * dts_weight

                    cross_grads = torch.autograd.grad(
                        dts_r_diff, [Pi_req, Pibar_req],
                        grad_outputs=dts_grad_output,
                        retain_graph=False,
                    )

                    # Accumulate into earlier waves' RHS
                    if cross_grads[0] is not None:
                        accumulated_rhs = accumulated_rhs + cross_grads[0]
                    # Pibar depends on Pi, so Pibar gradient needs to chain through
                    # the Pibar computation. For the cross-clade terms, Pibar of children
                    # was computed during the forward pass of earlier waves.
                    # The gradient wrt Pibar_req will accumulate into the Pi adjoint
                    # of the wave that computed that Pibar.
                    if cross_grads[1] is not None:
                        # Pibar[c] = f(Pi[c], mt) — need chain rule through Pibar
                        # For each child c in sl/sr from earlier waves, accumulate
                        # d(dts_cross)/d(Pibar[c]) * d(Pibar[c])/d(Pi[c])
                        pibar_grad = cross_grads[1]  # [C, S]
                        # Pibar[c, s] = log2(sum_f!=s exp2(Pi[c,f]) * M[s,f]) + mt[s]
                        # d(Pibar[c,s])/d(Pi[c,f]) for f != s:
                        #   = exp2(Pi[c,f] - Pibar[c,s] + mt[s]) * M[s,f] / sum_f!=s(...)
                        # For uniform: d(Pibar)/d(Pi) = -exp2(Pi - Pi_max) / (rowsum - exp2(Pi - Pi_max))
                        # This is complex. Use autograd instead.
                        # Find which clades have non-zero pibar_grad
                        nz_clades = pibar_grad.abs().sum(dim=1) > 0
                        if nz_clades.any():
                            nz_idx = nz_clades.nonzero(as_tuple=True)[0]
                            Pi_for_pibar = Pi_star_wave[nz_idx].detach().requires_grad_(True)
                            with torch.enable_grad():
                                Pi_max_p = Pi_for_pibar.max(dim=1, keepdim=True).values
                                Pi_exp_p = torch.exp2(Pi_for_pibar - Pi_max_p)
                                row_sum_p = Pi_exp_p.sum(dim=1, keepdim=True)
                                Pibar_recomp = torch.log2(row_sum_p - Pi_exp_p) + Pi_max_p + mt_squeezed
                                pi_grad_from_pibar = torch.autograd.grad(
                                    Pibar_recomp, Pi_for_pibar,
                                    grad_outputs=pibar_grad[nz_idx],
                                    retain_graph=False,
                                )[0]
                            accumulated_rhs[nz_idx] = accumulated_rhs[nz_idx] + pi_grad_from_pibar

    return {
        'v_Pi': accumulated_rhs,
        'grad_E': grad_E_acc,
        'grad_Ebar': grad_Ebar_acc,
        'grad_E_s1': grad_E_s1_acc,
        'grad_E_s2': grad_E_s2_acc,
        'grad_log_pD': grad_log_pD,
        'grad_log_pS': grad_log_pS,
        'grad_max_transfer_mat': grad_mt,
    }


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
