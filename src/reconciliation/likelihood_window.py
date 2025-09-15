"""
Windowed fixed-point solver for Pi using the exact update from likelihood.py,
applied only to a sliding window of clades.

This module mirrors the math in Pi_step/Pi_fixed_point but restricts updates
to a subset of clades (window), reducing work per iteration while preserving
the same operations and semantics (no new K or A operators).
"""

from typing import Dict, List, Optional
import torch

# Import helpers from the baseline implementation
from .likelihood import (
    dup_both_survive,
    gather_Pi_children,
)


@torch.no_grad()
def _event_logs_from_theta(theta: torch.Tensor, device, dtype):
    exp_theta = torch.exp(theta.to(device=device, dtype=dtype))
    delta, tau, lam = exp_theta
    rates_sum = 1.0 + delta + tau + lam
    log_pS = torch.log(1.0 / rates_sum)
    log_pD = torch.log(delta / rates_sum)
    log_pT = torch.log(tau / rates_sum)
    log_pL = torch.log(lam   / rates_sum)
    return log_pS, log_pD, log_pT, log_pL


def _select_splits_for_parents(split_parents: torch.Tensor,
                               parents_batch: torch.Tensor) -> torch.Tensor:
    """Return indices of splits whose parent is in parents_batch."""
    device = split_parents.device
    C = int(split_parents.max().item()) + 1  # upper bound for mapping length
    parent_to_batch = torch.full((C,), -1, dtype=torch.long, device=device)
    parent_to_batch[parents_batch] = 1  # mark presence
    sel_mask = parent_to_batch.index_select(0, split_parents) >= 0
    return torch.nonzero(sel_mask, as_tuple=False).squeeze(1)  # [Ns]


def Pi_step_window(
    Pi: torch.Tensor,
    window_parents: torch.Tensor,                 # [B]
    ccp_helpers: Dict,
    species_helpers: Dict,
    clade_species_map: torch.Tensor,
    E: torch.Tensor, Ebar: torch.Tensor, E_s1: torch.Tensor, E_s2: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Exact Pi_step update restricted to the rows in window_parents.

    Returns new values [B,S] for those rows; callers assign back into Pi.
    """
    device, dtype = Pi.device, Pi.dtype
    B, S = window_parents.numel(), Pi.shape[1]

    # Event logs
    log_pS, log_pD, log_pT, log_pL = _event_logs_from_theta(theta, device, dtype)

    # Extract helpers
    split_parents = ccp_helpers['split_parents']
    split_lefts   = ccp_helpers['split_lefts']
    split_rights  = ccp_helpers['split_rights']
    split_probs   = ccp_helpers['split_probs']
    log_split_probs = torch.log(split_probs)
    Recipients_mat = species_helpers['Recipients_mat']
    internal_mask  = species_helpers['sp_internal_mask']
    sp_c1_idx      = species_helpers['s_C1_indexes']
    sp_c2_idx      = species_helpers['s_C2_indexes']

    # Build mapping parent -> batch row
    C_total = Pi.shape[0]
    parent_to_batch = torch.full((C_total,), -1, dtype=torch.long, device=device)
    parent_to_batch[window_parents] = torch.arange(B, device=device, dtype=torch.long)

    # Select splits whose parent is in this window
    split_idx = _select_splits_for_parents(split_parents, window_parents)   # [Ns]
    # Short-circuit if no splits for these parents: only leaf/one-copy terms remain
    has_splits = split_idx.numel() > 0
    if has_splits:
        batch_rows = parent_to_batch.index_select(0, split_parents.index_select(0, split_idx))  # [Ns]
        L = split_lefts.index_select(0, split_idx)
        R = split_rights.index_select(0, split_idx)

        Pi_left  = Pi.index_select(0, L)
        Pi_right = Pi.index_select(0, R)

    # Duplication: both survive terms over splits and 1-survivor per parent
    if has_splits:
        log_D_splits = dup_both_survive(Pi_left, Pi_right, log_split_probs.index_select(0, split_idx), log_pD)
    log_2 = torch.log(torch.tensor(2.0, dtype=dtype, device=device))
    Pi_batch = Pi.index_select(0, window_parents)
    log_D_loss = log_2 + log_pD + Pi_batch + E.unsqueeze(0)  # [B,S]

    # Speciation
    Pi_s1 = gather_Pi_children(Pi, sp_c1_idx, internal_mask)  # [C,S]
    Pi_s2 = gather_Pi_children(Pi, sp_c2_idx, internal_mask)  # [C,S]
    Pi_s1_batch = Pi_s1.index_select(0, window_parents)
    Pi_s2_batch = Pi_s2.index_select(0, window_parents)
    if has_splits:
        Pi_s1_left  = Pi_s1.index_select(0, L)
        Pi_s1_right = Pi_s1.index_select(0, R)
        Pi_s2_left  = Pi_s2.index_select(0, L)
        Pi_s2_right = Pi_s2.index_select(0, R)
        log_spec1 = log_split_probs.index_select(0, split_idx).unsqueeze(1) + log_pS + Pi_s1_left + Pi_s2_right
        log_spec2 = log_split_probs.index_select(0, split_idx).unsqueeze(1) + log_pS + Pi_s1_right + Pi_s2_left
    log_S_term1 = log_pS + Pi_s1_batch + E_s2.unsqueeze(0)
    log_S_term2 = log_pS + Pi_s2_batch + E_s1.unsqueeze(0)
    log_leaf_contrib = log_pS + clade_species_map.index_select(0, window_parents)

    # Transfer
    # Compute Pibar only for rows we need: window parents and their L/R children
    if has_splits:
        needed_rows = torch.unique(torch.cat([window_parents, L, R]))
    else:
        needed_rows = window_parents
    Pi_needed = Pi.index_select(0, needed_rows)
    Pi_needed_max = torch.max(Pi_needed, dim=1, keepdim=True).values
    Pi_needed_lin = torch.exp(Pi_needed - Pi_needed_max)
    Pibar_needed_lin = Pi_needed_lin.mm(Recipients_mat.T)
    Pibar_needed = torch.log(Pibar_needed_lin) + Pi_needed_max
    # Map back
    # Build inverse mapping needed_rows -> position
    inv_idx = torch.full((C_total,), -1, dtype=torch.long, device=device)
    inv_idx[needed_rows] = torch.arange(needed_rows.numel(), device=device, dtype=torch.long)
    Pibar_batch = Pibar_needed.index_select(0, inv_idx.index_select(0, window_parents))
    if has_splits:
        Pibar_left  = Pibar_needed.index_select(0, inv_idx.index_select(0, L))
        Pibar_right = Pibar_needed.index_select(0, inv_idx.index_select(0, R))

    if has_splits:
        log_trans1 = log_split_probs.index_select(0, split_idx).unsqueeze(1) + log_pT + Pi_left + Pibar_right
        log_trans2 = log_split_probs.index_select(0, split_idx).unsqueeze(1) + log_pT + Pi_right + Pibar_left
    log_T_term1 = log_pT + Pi_batch + Ebar.unsqueeze(0)
    log_T_term2 = log_pT + Pibar_batch + E.unsqueeze(0)

    # Combine both-survive split terms
    if has_splits:
        no_L_contribs = torch.stack([log_D_splits, log_spec1, log_spec2, log_trans1, log_trans2], dim=0)  # [5,Ns,S]
        log_combined_splits = torch.logsumexp(no_L_contribs, dim=0)  # [Ns,S]
        # Aggregate per parent (window) with stable log-sum-exp via scatter
        batch_rows_exp = batch_rows.unsqueeze(1).expand(-1, S)
        max_vals = torch.scatter_reduce(
            torch.full((B, S), float('-inf'), dtype=dtype, device=device),
            0, batch_rows_exp, log_combined_splits, reduce='amax'
        )
        gathered_max = torch.gather(max_vals, 0, batch_rows_exp)
        exp_terms = torch.exp(log_combined_splits - gathered_max)
        sum_contribs = torch.scatter_add(
            torch.zeros_like(max_vals), 0, batch_rows_exp, exp_terms
        )
        contribs_1 = torch.log(sum_contribs) + max_vals  # [B,S]
    else:
        contribs_1 = torch.full((B, S), float('-inf'), dtype=dtype, device=device)

    # Final reduce for window rows (include all one-copy and leaf terms)
    all_terms = [contribs_1, log_D_loss, log_S_term1, log_S_term2, log_leaf_contrib, log_T_term1, log_T_term2]
    new_Pi_batch = torch.logsumexp(torch.stack(all_terms, dim=0), dim=0)
    return new_Pi_batch


def Pi_fixed_point_window(
    ccp_helpers: Dict,
    species_helpers: Dict,
    clade_species_map: torch.Tensor,
    E: torch.Tensor, Ebar: torch.Tensor, E_s1: torch.Tensor, E_s2: torch.Tensor,
    theta: torch.Tensor,
    waves_by_size: List[List[int]],
    ubiquitous_clade_idx: int,
    *,
    window_waves: int = 4,
    tolerance: float = 1e-12,
    max_iters: int = 3000,
    warm_start_Pi: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Sliding-window fixed-point iteration for Pi, using the exact per-iteration
    update from likelihood.py but applied only to clades in the current window.

    Returns dict { 'Pi': Pi, 'iterations': iters }
    """
    device, dtype = clade_species_map.device, clade_species_map.dtype
    C, S = clade_species_map.shape

    # Initialize Pi like the baseline fixed-point
    if warm_start_Pi is not None:
        Pi = warm_start_Pi.clone()
    else:
        Pi = torch.full((C, S), -torch.log(torch.tensor(100.0, device=device, dtype=dtype)), dtype=dtype, device=device)
        # Set leaf rows to a uniform distribution across mapped species (as in baseline)
        leaves_mask = ccp_helpers['ccp_leaves_mask']
        leaf_ids = torch.nonzero(leaves_mask, as_tuple=False).squeeze(1)
        for c in leaf_ids.tolist():
            finite_mask = torch.isfinite(clade_species_map[c])
            if finite_mask.any():
                k = int(finite_mask.sum().item())
                Pi[c, finite_mask] = -torch.log(torch.tensor(float(k), device=device, dtype=dtype)).clamp(min=-0.001)

    # Track convergence per row
    converged = torch.zeros((C,), dtype=torch.bool, device=device)
    leaves_mask = ccp_helpers['ccp_leaves_mask']
    converged[leaves_mask] = True

    # Window over waves (exclude root until last)
    # Find wave of ubiquitous clade
    ub_wave_idx = None
    for wi, w in enumerate(waves_by_size):
        if ubiquitous_clade_idx in w:
            ub_wave_idx = wi
            break
    if ub_wave_idx is None:
        ub_wave_idx = len(waves_by_size) - 1

    win_start = 0
    win_end = min(ub_wave_idx, win_start + max(1, window_waves))

    iters = 0
    for it in range(max_iters):
        iters = it + 1
        # Build window rows (skip root until last window)
        window_rows: List[int] = []
        for wi in range(win_start, win_end):
            for cid in waves_by_size[wi]:
                if cid == ubiquitous_clade_idx:
                    continue
                if not converged[int(cid)]:
                    window_rows.append(int(cid))
        if not window_rows:
            # Advance the window if possible
            if win_end < ub_wave_idx:
                win_start = min(ub_wave_idx, win_start + 1)
                win_end = min(ub_wave_idx, win_start + max(1, window_waves))
                continue
            else:
                break

        parents_batch = torch.tensor(window_rows, device=device, dtype=torch.long)
        Pi_new_batch = Pi_step_window(Pi, parents_batch, ccp_helpers, species_helpers,
                                      clade_species_map, E, Ebar, E_s1, E_s2, theta)
        prev = Pi.index_select(0, parents_batch)
        Pi[parents_batch] = Pi_new_batch
        # Convergence per row
        row_diff = torch.max(torch.abs(Pi_new_batch - prev), dim=1).values
        converged[parents_batch] |= (row_diff < tolerance)

        # Slide window start if the leading wave is fully converged
        while win_start < ub_wave_idx:
            done = True
            for cid in waves_by_size[win_start]:
                if not converged[int(cid)]:
                    done = False
                    break
            if done:
                win_start += 1
                win_end = min(ub_wave_idx, win_start + max(1, window_waves))
            else:
                break

    # Final window includes the root; iterate root until tolerance inside this loop
    parents_batch = torch.tensor([ubiquitous_clade_idx], device=device, dtype=torch.long)
    for _ in range(max_iters):
        Pi_new_root = Pi_step_window(Pi, parents_batch, ccp_helpers, species_helpers,
                                     clade_species_map, E, Ebar, E_s1, E_s2, theta)
        prev = Pi.index_select(0, parents_batch)
        Pi[parents_batch] = Pi_new_root
        if torch.max(torch.abs(Pi_new_root - prev)) < tolerance:
            break

    return { 'Pi': Pi, 'iterations': iters }

