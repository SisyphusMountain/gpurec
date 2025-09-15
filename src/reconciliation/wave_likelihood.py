import torch
import time
from typing import Dict, List, Optional, Tuple

INF_N = float('-inf')

# -----------------------------
# 0) Event logs & alpha helpers
# -----------------------------

def compute_event_logs(theta: torch.Tensor, S: int, device, dtype):
    exp_theta = torch.exp(theta.to(device=device, dtype=dtype))
    delta, tau, lam = exp_theta
    rates_sum = 1.0 + delta + tau + lam
    log_pS = torch.log(1.0 / rates_sum) * torch.ones(S, device=device, dtype=dtype)
    log_pD = torch.log(delta / rates_sum) * torch.ones(S, device=device, dtype=dtype)
    log_pT = torch.log(tau   / rates_sum) * torch.ones(S, device=device, dtype=dtype)
    return log_pS, log_pD, log_pT

@torch.no_grad()
def compute_log_alpha_strict(log_pD, log_pT, E_log, Ebar_log, out_dtype):
    # alpha = 1 - 2*pD*E - pT*Ebar   (prob. domain)
    pD   = torch.exp(log_pD.to(torch.float64))
    pT   = torch.exp(log_pT.to(torch.float64))
    E    = torch.exp(E_log.to(torch.float64))
    Ebar = torch.exp(Ebar_log.to(torch.float64))
    x = 2.0 * pD * E + pT * Ebar
    x = torch.clamp(x, 0.0, 1.0)  # if ==1 => log_alpha = -inf (correct)
    log_alpha = torch.log1p(-x)
    return log_alpha.to(out_dtype)

# -----------------------------------
# 1) log-stable SpMMs for A and K
# -----------------------------------

def log_spmm_strict(logX: torch.Tensor, csr_mat: torch.Tensor) -> torch.Tensor:
    """
    Exact -inf semantics. Handles sparse CSR[S,S] * [S,B] in the linear domain
    with a row-wise max shift to avoid underflow, then returns to log.
    logX: [B,S]; csr_mat: CSR[S,S].
    """
    B, S = logX.shape
    m = torch.max(logX, dim=1, keepdim=True).values            # [B,1]
    safe_m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    XT_lin = torch.exp(logX.transpose(0,1) - safe_m.T)         # [S,B]
    YT_lin = torch.sparse.mm(csr_mat, XT_lin)                  # [S,B]
    Y_log  = torch.log(YT_lin).transpose(0,1).contiguous() + safe_m  # [B,S]
    return Y_log

def log_A_apply_strict(logX: torch.Tensor, Recipients_mat: torch.Tensor) -> torch.Tensor:
    """
    log(A X) for A row-stochastic. Supports dense or CSR.
    logX: [B,S]
    """
    if Recipients_mat.is_sparse:
        return log_spmm_strict(logX, Recipients_mat)
    # Dense path
    B, S = logX.shape
    m = torch.max(logX, dim=1, keepdim=True).values
    safe_m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    X_lin = torch.exp(logX - safe_m)                           # [B,S]
    Y_lin = X_lin.mm(Recipients_mat.T)                         # [B,S]
    Y_log = torch.log(Y_lin) + safe_m                          # [B,S] (log(0) -> -inf)
    return Y_log

def build_K_csr(
    S: int,
    internal_mask: torch.Tensor,
    s_C1_idx: torch.Tensor, s_C2_idx: torch.Tensor,
    E_s1_log: torch.Tensor, E_s2_log: torch.Tensor
) -> torch.Tensor:
    """
    K[e,f] = exp(E_s2_log[e]), K[e,g] = exp(E_s1_log[e]) for internal e
    """
    device = E_s1_log.device
    val_dtype = E_s1_log.dtype
    rows_internal = torch.nonzero(internal_mask, as_tuple=False).squeeze(1)  # [Nint]
    f = s_C1_idx.to(torch.int64)
    g = s_C2_idx.to(torch.int64)

    vals_fg = torch.stack([
        torch.exp(E_s2_log[rows_internal].to(torch.float64)),
        torch.exp(E_s1_log[rows_internal].to(torch.float64))
    ], dim=1).to(val_dtype).reshape(-1)                                       # [2*Nint]

    cols_fg = torch.stack([f, g], dim=1).reshape(-1).to(torch.int64)         # [2*Nint]

    indptr = torch.zeros(S + 1, dtype=torch.int64, device=device)
    indptr[rows_internal + 1] = 2
    indptr = torch.cumsum(indptr, dim=0)

    K = torch.sparse_csr_tensor(indptr, cols_fg, vals_fg, size=(S, S), device=device)
    return K

# ------------------------------------------------
# 2) Batch RHS builder for non-ubiquitous parents
# ------------------------------------------------

def precompute_single_split_index(C: int, split_parents: torch.Tensor) -> torch.Tensor:
    """
    For each parent clade p in [0..C-1], returns the index of its (single) split,
    or -1 if leaf (no split) or if parent appears multiple times (not used here).
    """
    device = split_parents.device
    idx = torch.full((C,), -1, dtype=torch.long, device=device)
    # First occurrence wins (assumes single split per non-ubiq parent)
    idx.scatter_(0, split_parents, torch.arange(split_parents.numel(), device=device))
    # If a parent had multiple splits, idx will contain the last occurrence; but your premise is single split.
    return idx

def build_rhs_nonubiq_batch(
    parents_batch: torch.Tensor,                    # [B]
    split_parents: torch.Tensor,                    # [N_splits]
    split_lefts: torch.Tensor, split_rights: torch.Tensor,
    log_split_probs: torch.Tensor,                  # [N_splits]
    log_pS: torch.Tensor, log_pD: torch.Tensor, log_pT: torch.Tensor,   # [S]
    Pi: torch.Tensor, Pibar: torch.Tensor,          # [C,S] logs (subclades already solved)
    CH1: Optional[torch.Tensor], CH2: Optional[torch.Tensor],           # CSR selectors (optional)
    clade_species_map: torch.Tensor,                # [C,S] logs (leaf/boundary)
    split_ids_per_parent: Optional[List[torch.Tensor]] = None
) -> torch.Tensor:
    """
    Returns RHS_batch [B,S] for these parents by aggregating both-survive terms
    over all splits of each parent (unordered; includes both speciation orientations),
    plus the leaf/boundary term.
    """
    device, dtype = Pi.device, Pi.dtype
    B = parents_batch.numel()
    S = Pi.shape[1]
    RHS = torch.full((B, S), INF_N, dtype=dtype, device=device)

    if (CH1 is None) or (CH2 is None):
        raise RuntimeError("CH1/CH2 needed for speciation-both on non-ubiquitous clades")

    # Build mapping from global parent id -> batch row
    C_total = Pi.shape[0]
    parent_to_batch = torch.full((C_total,), -1, dtype=torch.long, device=device)
    parent_to_batch[parents_batch] = torch.arange(B, device=device, dtype=torch.long)

    # Select split indices to use and corresponding batch rows
    if split_ids_per_parent is not None:
        idx_list = []
        row_list = []
        for i in range(B):
            idxs = split_ids_per_parent[i]
            if idxs.numel() == 0:
                continue
            idx_list.append(idxs)
            row_list.append(torch.full((idxs.numel(),), i, device=device, dtype=torch.long))
        if len(idx_list) == 0:
            # No splits for these parents; just add leaf/boundary below
            RHS = torch.logaddexp(RHS, log_pS.unsqueeze(0) + clade_species_map.index_select(0, parents_batch))
            return RHS
        split_idx = torch.cat(idx_list, dim=0)            # [Ns]
        batch_rows = torch.cat(row_list, dim=0)           # [Ns]
    else:
        batch_rows_full = parent_to_batch.index_select(0, split_parents)  # [N_splits]
        sel_mask = batch_rows_full >= 0
        if not torch.any(sel_mask):
            RHS = torch.logaddexp(RHS, log_pS.unsqueeze(0) + clade_species_map.index_select(0, parents_batch))
            return RHS
        split_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)    # [Ns]
        batch_rows = batch_rows_full.index_select(0, split_idx)           # [Ns]

    # Gather needed tensors for all selected splits
    L = split_lefts.index_select(0, split_idx)
    R = split_rights.index_select(0, split_idx)
    w = log_split_probs.index_select(0, split_idx).unsqueeze(1)  # [Ns,1]

    PiL    = Pi.index_select(0, L)          # [Ns,S]
    PiR    = Pi.index_select(0, R)
    PibarL = Pibar.index_select(0, L)
    PibarR = Pibar.index_select(0, R)

    PiCH1L = log_spmm_strict(PiL, CH1); PiCH2R = log_spmm_strict(PiR, CH2)
    PiCH1R = log_spmm_strict(PiR, CH1); PiCH2L = log_spmm_strict(PiL, CH2)

    log_dup   = w + log_pD.unsqueeze(0) + PiL + PiR
    log_spec1 = w + log_pS.unsqueeze(0) + PiCH1L + PiCH2R
    log_spec2 = w + log_pS.unsqueeze(0) + PiCH1R + PiCH2L
    log_tr1   = w + log_pT.unsqueeze(0) + PiL + PibarR
    log_tr2   = w + log_pT.unsqueeze(0) + PiR + PibarL

    stacked = torch.stack([log_dup, log_spec1, log_spec2, log_tr1, log_tr2], dim=0)  # [5,Ns,S]
    per_split = torch.logsumexp(stacked, dim=0)                                       # [Ns,S]

    # Reduce across splits to per-parent via stable log-sum-exp using scatter
    batch_rows_exp = batch_rows.unsqueeze(1).expand(-1, S)                            # [Ns,S]
    max_vals = torch.scatter_reduce(
        torch.full((B, S), INF_N, dtype=dtype, device=device),
        0, batch_rows_exp, per_split, reduce='amax'
    )  # [B,S]
    gathered_max = torch.gather(max_vals, 0, batch_rows_exp)
    exp_terms = torch.exp(per_split - gathered_max)
    sum_contribs = torch.scatter_add(
        torch.zeros_like(max_vals),
        0, batch_rows_exp, exp_terms
    )
    RHS_splits = torch.log(sum_contribs) + max_vals

    # Add leaf/boundary term for all parents
    RHS = torch.logaddexp(RHS_splits, log_pS.unsqueeze(0) + clade_species_map.index_select(0, parents_batch))
    return RHS

    # Add leaf/boundary term for all parents
    RHS = torch.logaddexp(RHS, log_pS.unsqueeze(0) + clade_species_map.index_select(0, parents_batch))
    return RHS

# ---------------------------------------
# 3) Per-batch fixed-point (log-space)
# ---------------------------------------

def solve_batch_log_strict(
    log_rhs: torch.Tensor,                   # [B,S]
    Recipients_mat: torch.Tensor,            # A (dense or CSR)
    K_csr: torch.Tensor,                     # CSR K
    log_pS: torch.Tensor, log_pD: torch.Tensor, log_pT: torch.Tensor,  # [S]
    E_log: torch.Tensor, Ebar_log: torch.Tensor,                       # [S]
    *,
    max_iters: int = 60, tol: float = 1e-8,
    warm_start: Optional[torch.Tensor] = None,
    skip_convergence_check: bool = False
) -> torch.Tensor:
    B, S = log_rhs.shape
    dtype, device = log_rhs.dtype, log_rhs.device
    log_alpha = compute_log_alpha_strict(log_pD, log_pT, E_log, Ebar_log, out_dtype=dtype)  # [S]
    log_dT    = (log_pT + E_log).to(dtype)                                                  # [S]
    log_pS    = log_pS.to(dtype)

    logX = warm_start.clone() if warm_start is not None else torch.full((B,S), -1.0, dtype=dtype, device=device)
    for _ in range(max_iters):
        logAX = log_A_apply_strict(logX, Recipients_mat)              # [B,S]
        logKX = log_spmm_strict(logX, K_csr)                          # [B,S]
        tmp = torch.logsumexp(
            torch.stack([
                log_rhs,
                log_dT.unsqueeze(0) + logAX,
                log_pS.unsqueeze(0) + logKX
            ], dim=0), dim=0
        )
        logX_new = tmp - log_alpha.unsqueeze(0)
        if not skip_convergence_check:
            if torch.max(torch.abs(logX_new - logX)) < tol:
                logX = logX_new
                break
        logX = logX_new
    return logX

# -------------------------------------------------
# 4) Ubiquitous clade RHS (single parent, many splits)
# -------------------------------------------------

def build_rhs_ubiquitous(
    parent_idx: int,
    split_parents: torch.Tensor, split_lefts: torch.Tensor, split_rights: torch.Tensor,
    log_split_probs: torch.Tensor,
    Pi: torch.Tensor, Pibar: torch.Tensor,          # [C,S] logs
    CH1: Optional[torch.Tensor], CH2: Optional[torch.Tensor],
    log_pS: torch.Tensor, log_pD: torch.Tensor, log_pT: torch.Tensor,
    split_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Returns [1,S]: logsumexp over all splits (both orientations).
    """
    device, dtype = Pi.device, Pi.dtype
    S = Pi.shape[1]

    if split_indices is None:
        mask = (split_parents == parent_idx)
        L = split_lefts[mask]
        R = split_rights[mask]
        w = log_split_probs[mask].unsqueeze(1)          # [Ns,1]
    else:
        L = split_lefts.index_select(0, split_indices)
        R = split_rights.index_select(0, split_indices)
        w = log_split_probs.index_select(0, split_indices).unsqueeze(1)

    PiL, PiR = Pi.index_select(0, L), Pi.index_select(0, R)          # [Ns,S]
    PibarL, PibarR = Pibar.index_select(0, L), Pibar.index_select(0, R)

    if (CH1 is not None) and (CH2 is not None):
        PiCH1L = log_spmm_strict(PiL, CH1); PiCH2R = log_spmm_strict(PiR, CH2)
        PiCH1R = log_spmm_strict(PiR, CH1); PiCH2L = log_spmm_strict(PiL, CH2)
    else:
        raise RuntimeError("CH1/CH2 needed for speciation-both on ubiquitous clade")

    # Five terms; unordered → include both orientations for speciation & transfer
    log_dup   = w + log_pD.unsqueeze(0) + PiL + PiR                          # [Ns,S]
    log_spec1 = w + log_pS.unsqueeze(0) + PiCH1L + PiCH2R
    log_spec2 = w + log_pS.unsqueeze(0) + PiCH1R + PiCH2L
    log_tr1   = w + log_pT.unsqueeze(0) + PiL + PibarR
    log_tr2   = w + log_pT.unsqueeze(0) + PiR + PibarL

    # Final RHS = logsumexp over Ns of logsumexp(five terms)
    stacked = torch.stack([log_dup, log_spec1, log_spec2, log_tr1, log_tr2], dim=0)  # [5,Ns,S]
    per_split = torch.logsumexp(stacked, dim=0)                                       # [Ns,S]
    rhs = torch.logsumexp(per_split, dim=0, keepdim=True)                             # [1,S]
    return rhs

# -------------------------------------------------
# 5) Main driver: wave-by-size with batching
# -------------------------------------------------

def compute_log_Pi_waves(
    # Inputs
    ccp_helpers: Dict, species_helpers: Dict,
    clade_species_map: torch.Tensor,          # [C,S] logs (leaf/boundary map)
    E_log: torch.Tensor, Ebar_log: torch.Tensor, E_s1_log: torch.Tensor, E_s2_log: torch.Tensor,
    theta: torch.Tensor,
    waves_by_size: List[List[int]],           # increasing size, last entry contains ubiquitous clade
    ubiquitous_clade_idx: int,
    # Options
    B_max: int = 512, max_iters_batch: int = 60, tol_batch: float = 1e-8,
    cache_Pibar: bool = True,
    # Precomputed operators and inputs for CUDA graph capture
    K_csr: Optional[torch.Tensor] = None,
    CH1: Optional[torch.Tensor] = None,
    CH2: Optional[torch.Tensor] = None,
    waves_parent_tensors: Optional[List[torch.Tensor]] = None,
    parent_split_ids_nested: Optional[List[List[torch.Tensor]]] = None,
    log_pS_vec: Optional[torch.Tensor] = None,
    log_pD_vec: Optional[torch.Tensor] = None,
    log_pT_vec: Optional[torch.Tensor] = None,
    ubiquitous_split_indices: Optional[torch.Tensor] = None,
    skip_convergence_check: bool = False,
    # Optional profiling: append (wave_index, wave_size, seconds)
    wave_timing_list: Optional[List[Tuple[int, int, float]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (Pi_log, Pibar_log).
    Assumes: all non-ubiquitous parents have <=1 split (your premise).
    """
    # Unpack
    C = ccp_helpers['C']
    split_parents = ccp_helpers['split_parents']
    split_lefts   = ccp_helpers['split_lefts']
    split_rights  = ccp_helpers['split_rights']
    log_split_probs = torch.log(ccp_helpers['split_probs'])
    Recipients_mat = species_helpers['Recipients_mat']          # A
    internal_mask  = species_helpers['sp_internal_mask']
    s_C1_idx       = species_helpers['s_C1_indexes']
    s_C2_idx       = species_helpers['s_C2_indexes']

    device, dtype = clade_species_map.device, clade_species_map.dtype
    S = clade_species_map.shape[1]

    # Event logs
    if (log_pS_vec is None) or (log_pD_vec is None) or (log_pT_vec is None):
        log_pS, log_pD, log_pT = compute_event_logs(theta, S, device, dtype)
    else:
        log_pS, log_pD, log_pT = log_pS_vec, log_pD_vec, log_pT_vec

    # Operators
    if K_csr is None:
        K_csr = build_K_csr(S, internal_mask, s_C1_idx, s_C2_idx, E_s1_log, E_s2_log)
    # Optional selectors for speciation both (RHS build):
    if CH1 is None:
        CH1 = _build_child_selector_csr(S, internal_mask, s_C1_idx, dtype=dtype)  # (CH1 x)_e = x_f
    if CH2 is None:
        CH2 = _build_child_selector_csr(S, internal_mask, s_C2_idx, dtype=dtype)  # (CH2 x)_e = x_g

    # Storage
    Pi  = torch.full((C, S), INF_N, dtype=dtype, device=device)
    Pibar = torch.full_like(Pi, INF_N)
    # Initialize leaves to their boundary mapping for correctness
    leaf_ids_opt = ccp_helpers.get('leaf_ids', None)
    if leaf_ids_opt is None:
        leaves_mask = ccp_helpers['ccp_leaves_mask']
        leaf_ids_opt = torch.nonzero(leaves_mask, as_tuple=False).squeeze(1)
    if leaf_ids_opt is not None and leaf_ids_opt.numel() > 0:
        Pi[leaf_ids_opt] = clade_species_map.index_select(0, leaf_ids_opt)
        # Cache Pibar for leaves
        Pibar[leaf_ids_opt] = log_A_apply_strict(Pi.index_select(0, leaf_ids_opt), Recipients_mat)

    # Identify which wave contains the ubiquitous clade (for profiling label)
    ub_wave_idx = None
    for _i, _w in enumerate(waves_by_size):
        if ubiquitous_clade_idx in _w:
            ub_wave_idx = _i
            break

    # Process all waves except the final (ubiquitous) wave
    for wi, wave in enumerate(waves_by_size):
        if ubiquitous_clade_idx in wave:  # skip it for the final step
            continue
        # Chunk parents into batches of size <= B_max
        if waves_parent_tensors is None:
            parents = torch.tensor(wave, device=device, dtype=torch.long)
        else:
            parents = waves_parent_tensors[wi]
        # Start per-wave timing
        if wave_timing_list is not None:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
        for b0 in range(0, parents.numel(), B_max):
            batch_parents = parents[b0:b0+B_max]                    # [B]
            # RHS for batch (both-survive + leaf)
            RHS = build_rhs_nonubiq_batch(
                batch_parents,
                split_parents, split_lefts, split_rights, log_split_probs,
                log_pS, log_pD, log_pT,
                Pi, Pibar, CH1, CH2,
                clade_species_map,
                split_ids_per_parent=(parent_split_ids_nested[wi] if parent_split_ids_nested is not None else None)
            )  # [B,S]

            # Solve affine fixed point for these columns
            warm = Pi.index_select(0, batch_parents)                # warm-start
            X = solve_batch_log_strict(
                RHS, Recipients_mat, K_csr, log_pS, log_pD, log_pT, E_log, Ebar_log,
                max_iters=max_iters_batch, tol=tol_batch, warm_start=warm,
                skip_convergence_check=skip_convergence_check
            )

            # Commit and cache
            Pi[batch_parents] = X
            if cache_Pibar:
                Pibar[batch_parents] = log_A_apply_strict(X, Recipients_mat)
        # End per-wave timing
        if wave_timing_list is not None:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            wave_timing_list.append((wi, int(parents.numel()), float(time.perf_counter() - t0)))

    # Final: ubiquitous clade (single parent, many splits, unordered)
    ub = ubiquitous_clade_idx
    # Optional timing for ubiquitous clade step
    if wave_timing_list is not None:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0_ub = time.perf_counter()

    rhs_ub = build_rhs_ubiquitous(
        ub, split_parents, split_lefts, split_rights, log_split_probs,
        Pi, Pibar, CH1, CH2, log_pS, log_pD, log_pT,
        split_indices=ubiquitous_split_indices
    )  # [1,S]
    warm = Pi[ub].unsqueeze(0)
    Xub = solve_batch_log_strict(
        rhs_ub, Recipients_mat, K_csr, log_pS, log_pD, log_pT, E_log, Ebar_log,
        max_iters=max_iters_batch, tol=tol_batch, warm_start=warm,
        skip_convergence_check=skip_convergence_check
    )
    Pi[ub:ub+1] = Xub
    if cache_Pibar:
        Pibar[ub:ub+1] = log_A_apply_strict(Xub, Recipients_mat)

    # Record timing for ubiquitous wave
    if wave_timing_list is not None:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        # Use ub_wave_idx as the wave index label if known
        ub_idx_label = ub_wave_idx if ub_wave_idx is not None else -1
        wave_timing_list.append((int(ub_idx_label), 1, float(time.perf_counter() - t0_ub)))

    return Pi, Pibar

# -------------------------------------------------
# 8) Rolling-window Picard updates (no alpha division)
# -------------------------------------------------

def compute_log_Pi_rolling(
    # Inputs
    ccp_helpers: Dict, species_helpers: Dict,
    clade_species_map: torch.Tensor,          # [C,S] logs (leaf/boundary map)
    E_log: torch.Tensor, Ebar_log: torch.Tensor, E_s1_log: torch.Tensor, E_s2_log: torch.Tensor,
    theta: torch.Tensor,
    waves_by_size: List[List[int]],
    ubiquitous_clade_idx: int,
    # Options
    window_waves: int = 3,                    # number of consecutive waves per window
    passes: int = 2,                          # number of passes over sliding windows
    B_max: int = 512,
    cache_Pibar: bool = True,
    update_root_last: bool = True,
    final_iters: int = 1,                     # iterations for final ubiquitous clade
    final_tol: float = 1e-10,
    # Optional bootstrap using wave solver (few iters) to avoid -inf warm start
    bootstrap_waves_iters: int = 0,
    bootstrap_tol: float = 1e-6,
    # Optional precomputed operators
    CH1: Optional[torch.Tensor] = None,
    CH2: Optional[torch.Tensor] = None,
    # Optional timing collector: append (window_index, window_size, seconds)
    wave_timing_list: Optional[List[Tuple[int, int, float]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rolling-window solver using the direct Picard update (no alpha subtraction).

    Leaves are fixed to the boundary mapping. For internal clades, we slide a
    window of `window_waves` across the clade-size waves and perform one Picard
    update for the union of parents in the window. We repeat this for `passes`
    passes. The ubiquitous clade (root) is updated at the end (optionally with
    a few iterations via `final_iters`).
    """
    # Unpack helpers
    C = ccp_helpers['C']
    split_parents = ccp_helpers['split_parents']
    split_lefts   = ccp_helpers['split_lefts']
    split_rights  = ccp_helpers['split_rights']
    log_split_probs = torch.log(ccp_helpers['split_probs'])

    Recipients_mat = species_helpers['Recipients_mat']
    internal_mask  = species_helpers['sp_internal_mask']
    s_C1_idx       = species_helpers['s_C1_indexes']
    s_C2_idx       = species_helpers['s_C2_indexes']

    device, dtype = clade_species_map.device, clade_species_map.dtype
    S = clade_species_map.shape[1]

    # Event logs
    log_pS, log_pD, log_pT = compute_event_logs(theta, S, device, dtype)
    log_two = torch.log(torch.tensor(2.0, device=device, dtype=dtype))

    # Child selectors
    if CH1 is None:
        CH1 = _build_child_selector_csr(S, internal_mask, s_C1_idx, dtype=dtype)
    if CH2 is None:
        CH2 = _build_child_selector_csr(S, internal_mask, s_C2_idx, dtype=dtype)

    # Storage (optionally bootstrap via wave solver)
    Pi  = torch.full((C, S), INF_N, dtype=dtype, device=device)
    Pibar = torch.full_like(Pi, INF_N)

    # Initialize leaves to boundary mapping
    leaf_ids_opt = ccp_helpers.get('leaf_ids', None)
    if leaf_ids_opt is None:
        leaves_mask = ccp_helpers['ccp_leaves_mask']
        leaf_ids_opt = torch.nonzero(leaves_mask, as_tuple=False).squeeze(1)
    if leaf_ids_opt is not None and leaf_ids_opt.numel() > 0:
        Pi[leaf_ids_opt] = clade_species_map.index_select(0, leaf_ids_opt)
        if cache_Pibar:
            Pibar[leaf_ids_opt] = log_A_apply_strict(Pi.index_select(0, leaf_ids_opt), Recipients_mat)

    # Identify ubiquitous wave index
    ub_wave_idx = None
    for i, w in enumerate(waves_by_size):
        if ubiquitous_clade_idx in w:
            ub_wave_idx = i
            break

    # Helper: one Picard update for a batch of non-ubiquitous parents
    def _update_batch_nonubiq(parents_batch: torch.Tensor) -> torch.Tensor:
        Pi_batch = Pi.index_select(0, parents_batch)                 # [B,S]
        # Both-survive + leaf/boundary
        RHS = build_rhs_nonubiq_batch(
            parents_batch,
            split_parents, split_lefts, split_rights, log_split_probs,
            log_pS, log_pD, log_pT,
            Pi, Pibar, CH1, CH2,
            clade_species_map,
            split_ids_per_parent=None
        )  # [B,S]
        # Children along species tree
        PiCH1 = log_spmm_strict(Pi_batch, CH1)                       # [B,S]
        PiCH2 = log_spmm_strict(Pi_batch, CH2)                       # [B,S]
        # Transfer sums
        Pibar_batch = log_A_apply_strict(Pi_batch, Recipients_mat)   # [B,S]
        # One-copy (loss) terms
        log_D_loss = log_two + log_pD.unsqueeze(0) + Pi_batch + E_log.unsqueeze(0)
        log_S_t1   = log_pS.unsqueeze(0) + PiCH1 + E_s2_log.unsqueeze(0)
        log_S_t2   = log_pS.unsqueeze(0) + PiCH2 + E_s1_log.unsqueeze(0)
        log_T_t1   = log_pT.unsqueeze(0) + Pi_batch + Ebar_log.unsqueeze(0)
        log_T_t2   = log_pT.unsqueeze(0) + Pibar_batch + E_log.unsqueeze(0)

        stacked = torch.stack([RHS, log_D_loss, log_S_t1, log_S_t2, log_T_t1, log_T_t2], dim=0)
        return torch.logsumexp(stacked, dim=0)                       # [B,S]

    # Prepare list of wave indices excluding the ubiquitous wave (if requested)
    wave_indices = list(range(len(waves_by_size)))
    if update_root_last and (ub_wave_idx is not None):
        wave_indices = [wi for wi in wave_indices if wi != ub_wave_idx]

    # Sliding-window passes
    for p in range(max(1, passes)):
        for start in range(0, len(wave_indices)):
            window_wis = wave_indices[start:start + max(1, window_waves)]
            if not window_wis:
                continue
            # Collect unique parents in the window
            parents_list: List[int] = []
            for wi in window_wis:
                parents_list.extend(waves_by_size[wi])
            # Remove duplicates and leaves
            parents_set = sorted(set(parents_list))
            if leaf_ids_opt is not None and leaf_ids_opt.numel() > 0:
                leaf_set = set(int(x) for x in leaf_ids_opt.tolist())
                parents_set = [pid for pid in parents_set if pid not in leaf_set]
            if ubiquitous_clade_idx in parents_set and update_root_last:
                parents_set.remove(ubiquitous_clade_idx)
            if not parents_set:
                continue

            parents = torch.tensor(parents_set, device=device, dtype=torch.long)

            # Timing for the window
            if wave_timing_list is not None:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()

            # Update in batches
            for b0 in range(0, parents.numel(), B_max):
                batch_parents = parents[b0:b0+B_max]
                Xnew = _update_batch_nonubiq(batch_parents)
                Pi[batch_parents] = Xnew
                if cache_Pibar:
                    Pibar[batch_parents] = log_A_apply_strict(Xnew, Recipients_mat)

            if wave_timing_list is not None:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                wave_timing_list.append((start + p * len(wave_indices), int(parents.numel()), float(time.perf_counter() - t0)))

    # Final: ubiquitous clade update
    ub = ubiquitous_clade_idx
    if update_root_last and (ub is not None):
        # Build both-survive RHS for ub
        rhs_ub = build_rhs_ubiquitous(
            ub,
            split_parents, split_lefts, split_rights, log_split_probs,
            Pi, Pibar, CH1, CH2,
            log_pS, log_pD, log_pT,
        )  # [1,S]
        # Add leaf/boundary term
        rhs_ub = torch.logaddexp(rhs_ub, log_pS.unsqueeze(0) + clade_species_map[ub:ub+1])

        # Optional iterative refinement for ub
        Xub = Pi[ub:ub+1]
        for _ in range(max(1, final_iters)):
            Pibar_ub = log_A_apply_strict(Xub, Recipients_mat)
            PiCH1_ub = log_spmm_strict(Xub, CH1)
            PiCH2_ub = log_spmm_strict(Xub, CH2)

            log_D_loss = log_two + log_pD.unsqueeze(0) + Xub + E_log.unsqueeze(0)
            log_S_t1   = log_pS.unsqueeze(0) + PiCH1_ub + E_s2_log.unsqueeze(0)
            log_S_t2   = log_pS.unsqueeze(0) + PiCH2_ub + E_s1_log.unsqueeze(0)
            log_T_t1   = log_pT.unsqueeze(0) + Xub + Ebar_log.unsqueeze(0)
            log_T_t2   = log_pT.unsqueeze(0) + Pibar_ub + E_log.unsqueeze(0)

            stacked = torch.stack([rhs_ub, log_D_loss, log_S_t1, log_S_t2, log_T_t1, log_T_t2], dim=0)
            Xnew = torch.logsumexp(stacked, dim=0)
            if torch.max(torch.abs(Xnew - Xub)).item() < final_tol:
                Xub = Xnew
                break
            Xub = Xnew

        Pi[ub:ub+1] = Xub
        if cache_Pibar:
            Pibar[ub:ub+1] = log_A_apply_strict(Xub, Recipients_mat)

    return Pi, Pibar

# -------------------------------------------------
# 9) Dynamic active-set Picard (simple, dependency-driven)
# -------------------------------------------------

def compute_log_Pi_active(
    ccp_helpers: Dict,
    species_helpers: Dict,
    clade_species_map: torch.Tensor,
    E_log: torch.Tensor,
    Ebar_log: torch.Tensor,
    E_s1_log: torch.Tensor,
    E_s2_log: torch.Tensor,
    theta: torch.Tensor,
    waves_by_size: List[List[int]],
    ubiquitous_clade_idx: int,
    *,
    window_waves: int = 3,
    tol: float = 1e-8,
    max_global_iters: int = 100000,
    B_max: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simpler solver that maintains a dynamic active set of clades.

    - Leaves are fixed to the boundary mapping and marked converged.
    - Active set A starts from clades whose dependencies are all converged.
    - Iterate one Picard update (no alpha) for A in batches; drop rows that meet tol.
    - Whenever a row converges, admit any parent whose dependencies are all converged.
    - Finish by iterating the ubiquitous clade (root) until tol.
    """
    device, dtype = clade_species_map.device, clade_species_map.dtype
    C, S = clade_species_map.shape
    # Use a dtype-aware tolerance for convergence checks to avoid stalling in float32
    finfo = torch.finfo(dtype)
    freeze_tol = max(tol, 1000.0 * float(finfo.eps))

    # Unpack CCP split tensors
    split_parents = ccp_helpers['split_parents']
    split_lefts   = ccp_helpers['split_lefts']
    split_rights  = ccp_helpers['split_rights']
    log_split_probs = torch.log(ccp_helpers['split_probs'])

    # Species helpers
    Recipients_mat = species_helpers['Recipients_mat']
    internal_mask  = species_helpers['sp_internal_mask']
    s_C1_idx       = species_helpers['s_C1_indexes']
    s_C2_idx       = species_helpers['s_C2_indexes']

    # Event logs
    log_pS, log_pD, log_pT = compute_event_logs(theta, S, device, dtype)
    log_two = torch.log(torch.tensor(2.0, device=device, dtype=dtype))

    # Child selector operators
    CH1 = _build_child_selector_csr(S, internal_mask, s_C1_idx, dtype=dtype)
    CH2 = _build_child_selector_csr(S, internal_mask, s_C2_idx, dtype=dtype)

    # Storage
    Pi  = torch.full((C, S), INF_N, device=device, dtype=dtype)
    Pibar = torch.full_like(Pi, INF_N)

    # One Picard update function for a batch of parents (no alpha division)
    def update_batch(batch_parents: torch.Tensor) -> torch.Tensor:
        Pi_batch = Pi.index_select(0, batch_parents)
        RHS = build_rhs_nonubiq_batch(
            batch_parents,
            split_parents, split_lefts, split_rights, log_split_probs,
            log_pS, log_pD, log_pT,
            Pi, Pibar, CH1, CH2,
            clade_species_map,
            split_ids_per_parent=None,
        )
        PiCH1 = log_spmm_strict(Pi_batch, CH1)
        PiCH2 = log_spmm_strict(Pi_batch, CH2)
        Pibar_batch = log_A_apply_strict(Pi_batch, Recipients_mat)

        log_D_loss = log_two + log_pD.unsqueeze(0) + Pi_batch + E_log.unsqueeze(0)
        log_S_t1   = log_pS.unsqueeze(0) + PiCH1 + E_s2_log.unsqueeze(0)
        log_S_t2   = log_pS.unsqueeze(0) + PiCH2 + E_s1_log.unsqueeze(0)
        log_T_t1   = log_pT.unsqueeze(0) + Pi_batch + Ebar_log.unsqueeze(0)
        log_T_t2   = log_pT.unsqueeze(0) + Pibar_batch + E_log.unsqueeze(0)

        stacked = torch.stack([RHS, log_D_loss, log_S_t1, log_S_t2, log_T_t1, log_T_t2], dim=0)
        return torch.logsumexp(stacked, dim=0)

    # Initialize leaves to boundary mapping and mark converged
    leaves_mask = ccp_helpers['ccp_leaves_mask']
    leaf_ids = torch.nonzero(leaves_mask, as_tuple=False).squeeze(1)
    if leaf_ids.numel() > 0:
        Pi[leaf_ids] = clade_species_map.index_select(0, leaf_ids)
        Pibar[leaf_ids] = log_A_apply_strict(Pi.index_select(0, leaf_ids), Recipients_mat)

    # Bootstrap: single bottom-up pass to seed finite values for internal clades
    # Uses one direct Picard update per wave (non-ubiquitous), in increasing size order
    # This avoids all -inf warm starts when the active window begins.
    # Identify ubiquitous wave index for bound
    ub_wave_idx_boot = None
    for wi_b, w_b in enumerate(waves_by_size):
        if ubiquitous_clade_idx in w_b:
            ub_wave_idx_boot = wi_b
            break
    if ub_wave_idx_boot is None:
        ub_wave_idx_boot = len(waves_by_size) - 1
    for wi_b in range(0, ub_wave_idx_boot):
        parents_b = waves_by_size[wi_b]
        if len(parents_b) == 0:
            continue
        parents_b_t = torch.tensor(parents_b, device=device, dtype=torch.long)
        Xseed = update_batch(parents_b_t)
        Pi[parents_b_t] = Xseed
        Pibar[parents_b_t] = log_A_apply_strict(Xseed, Recipients_mat)

    # Build wave index per clade to enable a sliding window over waves
    wave_of = [-1] * C
    ub_wave_idx = None
    for wi, w in enumerate(waves_by_size):
        for cid in w:
            wave_of[int(cid)] = wi
        if ubiquitous_clade_idx in w:
            ub_wave_idx = wi
    if ub_wave_idx is None:
        ub_wave_idx = len(waves_by_size) - 1

    # Track rows that are done (frozen = removed from active set)
    frozen = torch.zeros((C,), dtype=torch.bool, device=device)
    if leaf_ids.numel() > 0:
        frozen[leaf_ids] = True

    # Parent -> split index (single-split premise for non-ubiquitous parents)
    split_idx_per_parent = precompute_single_split_index(C, split_parents)

    # Sliding window bounds over non-ubiquitous waves
    win_start = 0
    win_end = max(0, min(ub_wave_idx, len(waves_by_size) - 1))
    win_end = min(win_end, win_start + max(1, window_waves))

    iters = 0
    # One Picard update function for a batch of current active parents (same as above)

    # Iterate until all non-ubiquitous waves are converged or iteration cap
    while iters < max_global_iters:
        iters += 1

        # Build active set = all non-frozen clades in current window, excluding root and leaves
        window_clades: List[int] = []
        for wi in range(win_start, min(ub_wave_idx, win_end)):
            for cid in waves_by_size[wi]:
                if cid == ubiquitous_clade_idx:
                    continue
                if not frozen[int(cid)] and (leaves_mask[int(cid)].item() is False):
                    window_clades.append(int(cid))
        if not window_clades:
            # Advance window if possible
            if win_end < ub_wave_idx:
                win_start = min(ub_wave_idx, win_start + 1)
                win_end = min(ub_wave_idx, win_start + max(1, window_waves))
                continue
            else:
                break

        active_tensor = torch.tensor(window_clades, device=device, dtype=torch.long)
        drop_mask_local = torch.zeros((active_tensor.numel(),), dtype=torch.bool, device=device)
        # Mark which clades are currently active (to test dependency activity)
        is_active = torch.zeros((C,), dtype=torch.bool, device=device)
        is_active[active_tensor] = True

        # Update in batches
        for b0 in range(0, active_tensor.numel(), B_max):
            batch_ids = active_tensor[b0:b0+B_max]
            Xnew = update_batch(batch_ids)
            prev = Pi.index_select(0, batch_ids)
            diff = torch.max(torch.abs(Xnew - prev), dim=1).values
            Pi[batch_ids] = Xnew
            Pibar[batch_ids] = log_A_apply_strict(Xnew, Recipients_mat)
            # Determine if any dependency is still active in the current window
            # Using single-split premise for non-ubiquitous parents
            parent_split_idx = split_idx_per_parent.index_select(0, batch_ids)
            has_split = parent_split_idx >= 0
            dep_active = torch.zeros_like(diff, dtype=torch.bool)
            if torch.any(has_split):
                idxs = parent_split_idx.masked_select(has_split).to(torch.long)
                lefts = split_lefts.index_select(0, idxs)
                rights = split_rights.index_select(0, idxs)
                any_child_active = is_active.index_select(0, lefts) | is_active.index_select(0, rights)
                dep_active[has_split] = any_child_active
            eligible = (diff < freeze_tol) & (~dep_active)
            drop_mask_local[b0:b0+batch_ids.numel()] = eligible

        # Mark dropped (frozen) and slide window start past waves with no active rows
        frozen[active_tensor] |= drop_mask_local

        # Move win_start forward while its entire wave is converged
        while win_start < ub_wave_idx:
            # A wave is considered done if no non-leaf, non-root clade remains active in it
            any_active = False
            for cid in waves_by_size[win_start]:
                if (cid != ubiquitous_clade_idx) and (not leaves_mask[int(cid)]) and (not frozen[int(cid)]):
                    any_active = True
                    break
            if any_active:
                break
            win_start += 1
            win_end = min(ub_wave_idx, win_start + max(1, window_waves))

    # Optional polish: a few global sweeps over non-root waves to tighten residuals
    for _ in range(3):
        max_delta = 0.0
        for wi_pol in range(0, ub_wave_idx):
            parents_pol = waves_by_size[wi_pol]
            if not parents_pol:
                continue
            parents_pol_t = torch.tensor(parents_pol, device=device, dtype=torch.long)
            Xnew_pol = update_batch(parents_pol_t)
            prev_pol = Pi.index_select(0, parents_pol_t)
            Pi[parents_pol_t] = Xnew_pol
            Pibar[parents_pol_t] = log_A_apply_strict(Xnew_pol, Recipients_mat)
            dval = torch.max(torch.abs(Xnew_pol - prev_pol)).item()
            if dval > max_delta:
                max_delta = dval
        if max_delta < freeze_tol:
            break

    # Final ubiquitous clade: iterate until tol (dtype-aware)
    ub = ubiquitous_clade_idx
    # Both-survive RHS (unordered over splits)
    rhs_ub = build_rhs_ubiquitous(
        ub, split_parents, split_lefts, split_rights, log_split_probs,
        Pi, Pibar, CH1, CH2, log_pS, log_pD, log_pT,
    )  # [1,S]
    rhs_ub = torch.logaddexp(rhs_ub, log_pS.unsqueeze(0) + clade_species_map[ub:ub+1])

    Xub = Pi[ub:ub+1]
    for _ in range(100000):
        Pibar_ub = log_A_apply_strict(Xub, Recipients_mat)
        PiCH1_ub = log_spmm_strict(Xub, CH1)
        PiCH2_ub = log_spmm_strict(Xub, CH2)

        log_D_loss = log_two + log_pD.unsqueeze(0) + Xub + E_log.unsqueeze(0)
        log_S_t1   = log_pS.unsqueeze(0) + PiCH1_ub + E_s2_log.unsqueeze(0)
        log_S_t2   = log_pS.unsqueeze(0) + PiCH2_ub + E_s1_log.unsqueeze(0)
        log_T_t1   = log_pT.unsqueeze(0) + Xub + Ebar_log.unsqueeze(0)
        log_T_t2   = log_pT.unsqueeze(0) + Pibar_ub + E_log.unsqueeze(0)

        stacked = torch.stack([rhs_ub, log_D_loss, log_S_t1, log_S_t2, log_T_t1, log_T_t2], dim=0)
        Xnew = torch.logsumexp(stacked, dim=0)
        if torch.max(torch.abs(Xnew - Xub)) < freeze_tol:
            Xub = Xnew
            break
        Xub = Xnew

    Pi[ub:ub+1] = Xub
    Pibar[ub:ub+1] = log_A_apply_strict(Xub, Recipients_mat)

    return Pi, Pibar

# ---------------------------------------------
# 6) Child selector CSR builders (optional)
# ---------------------------------------------

def _build_child_selector_csr(S, internal_mask, s_child_idx, *, dtype: torch.dtype):
    """
    Returns CSR M such that (M x)_e = x_child for internal e, else 0.
    Values=1.0; this is just a selector.
    """
    device = s_child_idx.device
    rows_internal = torch.nonzero(internal_mask, as_tuple=False).squeeze(1)   # [Nint]
    cols = s_child_idx.to(torch.int64)
    vals = torch.ones_like(cols, dtype=dtype, device=device)
    indptr = torch.zeros(S + 1, dtype=torch.int64, device=device)
    indptr[rows_internal + 1] = 1
    indptr = torch.cumsum(indptr, dim=0)
    return torch.sparse_csr_tensor(indptr, cols, vals, size=(S, S), device=device)
    # Optional bootstrap with a small number of wave iterations to populate Pi
    if bootstrap_waves_iters and bootstrap_waves_iters > 0:
        Pi0, Pibar0 = compute_log_Pi_waves(
            ccp_helpers=ccp_helpers,
            species_helpers=species_helpers,
            clade_species_map=clade_species_map,
            E_log=E_log, Ebar_log=Ebar_log,
            E_s1_log=E_s1_log, E_s2_log=E_s2_log,
            theta=theta,
            waves_by_size=waves_by_size,
            ubiquitous_clade_idx=ubiquitous_clade_idx,
            B_max=B_max,
            max_iters_batch=bootstrap_waves_iters,
            tol_batch=bootstrap_tol,
            cache_Pibar=True,
            CH1=CH1,
            CH2=CH2,
            skip_convergence_check=True,
        )
        Pi.copy_(Pi0)
        Pibar.copy_(Pibar0)
