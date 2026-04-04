"""Legacy code: fixed-point Pi solver and BFS wave scheduler.

These implementations are superseded by the wave-based forward pass
(``Pi_wave_forward``) and the C++ phased scheduler, but are kept for
testing, validation, and as fallback paths.

- ``Pi_step`` / ``Pi_fixed_point``  – full-matrix fixed-point iteration
  (moved from ``forward.py``)
- ``_compute_clade_waves_bfs``      – Python BFS wave scheduler
  (moved from ``scheduling.py``)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple

import torch

from .terms import (
    gather_Pi_children,
    compute_DTS,
    compute_DTS_L,
)
from .kernels.scatter_lse import seg_logsumexp
from .log2_utils import logsumexp2, logaddexp2

NEG_INF = float("-inf")


# ===== try to import logmatmul (same logic as forward.py) =================
try:
    import sys as _sys
    import importlib as _imp
    from pathlib import Path as _Path
    _logmatmul_dir = str(_Path(__file__).resolve().parents[2] / 'logmatmul')
    _saved_src = _sys.modules.get('src')
    if 'src' in _sys.modules:
        del _sys.modules['src']
    if _logmatmul_dir not in _sys.path:
        _sys.path.insert(0, _logmatmul_dir)
    _imp.import_module('src')
    LogspaceMatmulFn = _imp.import_module('src.autograd').LogspaceMatmulFn
    _HAS_LOGMATMUL = True
    if _saved_src is not None:
        _sys.modules['src'] = _saved_src
except (ImportError, FileNotFoundError, ModuleNotFoundError):
    _HAS_LOGMATMUL = False
    LogspaceMatmulFn = None


# ===== Private helpers (copied from forward.py) ===========================

def _seg_logsumexp_host(x: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """CPU fallback for segmented logsumexp; uses Triton kernel on CUDA."""
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
    return torch.stack(out, dim=0) if out else torch.empty(
        (0, *x.shape[1:]), device=x.device, dtype=x.dtype
    )


# ---------------------------------------------------------------------------
# Pi step (legacy full-matrix iteration)
# ---------------------------------------------------------------------------

def Pi_step(
    Pi: torch.Tensor,
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
):
    # region helpers
    split_leftrights_sorted = ccp_helpers['split_leftrights_sorted']
    log_split_probs = ccp_helpers['log_split_probs_sorted'].unsqueeze(1).contiguous()
    seg_parent_ids = ccp_helpers['seg_parent_ids']
    num_segs_ge2 = ccp_helpers['num_segs_ge2']
    num_segs_eq1 = ccp_helpers['num_segs_eq1']
    end_rows_ge2 = ccp_helpers['end_rows_ge2']
    ptr_ge2 = ccp_helpers['ptr_ge2']
    N_splits = ccp_helpers["N_splits"]
    sp_P_idx = species_helpers['s_P_indexes']
    sp_c12_idx = species_helpers["s_C12_indexes"]

    C, S = Pi.shape
    # endregion helpers

    Pi_s12 = gather_Pi_children(Pi, sp_P_idx, sp_c12_idx)

    if _HAS_LOGMATMUL and Pi.is_cuda:
        transfer_mat = transfer_mat_T.T
        Pi_T = Pi.T.contiguous()
        Pibar_T = LogspaceMatmulFn.apply(transfer_mat, Pi_T, "ieee")
        Pibar = Pibar_T.T + max_transfer_mat.squeeze(-1)
    else:
        Pi_max = torch.max(Pi, dim=1, keepdim=True).values
        Pi_minus = Pi - Pi_max
        Pi_linear = torch.exp2(Pi_minus)
        Pibar_linear = Pi_linear.mm(transfer_mat_T)
        Pibar_log = torch.log2(Pibar_linear)
        Pibar_log = Pibar_log + Pi_max
        Pibar = Pibar_log + max_transfer_mat.squeeze(-1)

    DTS_term = compute_DTS(log_pD, log_pS, Pi_s12, Pi, Pibar, log_split_probs, split_leftrights_sorted, N_splits, S)
    DTS_L_term = compute_DTS_L(log_pD, log_pS, Pi, Pibar, Pi_s12, E, Ebar, E_s1, E_s2, clade_species_map, log_2)

    DTS_reduced = torch.full((C, S), NEG_INF, device=Pi.device, dtype=Pi.dtype)
    if num_segs_ge2 > 0:
        y_ge2 = _seg_logsumexp_host(DTS_term[:end_rows_ge2], ptr_ge2)
        DTS_reduced.index_copy_(0, seg_parent_ids[:num_segs_ge2], y_ge2)
    if num_segs_eq1 > 0:
        DTS_reduced.index_copy_(
            0,
            seg_parent_ids[num_segs_ge2:num_segs_ge2 + num_segs_eq1],
            DTS_term[end_rows_ge2:end_rows_ge2 + num_segs_eq1],
        )

    return logaddexp2(DTS_reduced, DTS_L_term)


# ---------------------------------------------------------------------------
# Pi fixed point (legacy)
# ---------------------------------------------------------------------------

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
):
    """Fixed-point solver for Pi using leaf mapping indices and current event params."""
    C = int(ccp_helpers['C'])
    S = int(species_helpers['S'])

    clade_species_map = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
    clade_species_map[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0

    if warm_start_Pi is not None:
        Pi = warm_start_Pi
    else:
        Pi = torch.full((C, S), -1000.0, dtype=dtype, device=device)
        Pi[leaf_row_index.to(device), leaf_col_index.to(device)] = 0.0

    converged_iter = max_iters
    log_2 = torch.tensor([1.0], dtype=dtype, device=device)

    for iteration in range(max_iters):
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
        )
        if torch.abs(Pi_new - Pi).max() < tolerance:
            converged_iter = iteration + 1
            Pi = Pi_new
            break
        Pi = Pi_new

    return {'Pi': Pi, 'clade_species_map': clade_species_map, 'iterations': converged_iter}


# ---------------------------------------------------------------------------
# BFS wave scheduler (legacy)
# ---------------------------------------------------------------------------

def _compute_clade_waves_bfs(
    ccp_helpers: Dict[str, Any],
) -> Tuple[List[List[int]], List[int]]:
    """Legacy BFS scheduler (fallback)."""
    C: int = int(ccp_helpers["C"])
    N_splits: int = int(ccp_helpers["N_splits"])

    split_parents: List[int] = ccp_helpers["split_parents_sorted"].tolist()
    leftrights: List[int] = ccp_helpers["split_leftrights_sorted"].tolist()
    lefts = leftrights[:N_splits]
    rights = leftrights[N_splits:]

    depended_by: List[List[int]] = [[] for _ in range(C)]
    children_sets: List[set] = [set() for _ in range(C)]

    for idx in range(N_splits):
        p = split_parents[idx]
        l = lefts[idx]
        r = rights[idx]
        if l not in children_sets[p]:
            children_sets[p].add(l)
            depended_by[l].append(p)
        if r != l and r not in children_sets[p]:
            children_sets[p].add(r)
            depended_by[r].append(p)

    remaining: List[int] = [len(children_sets[c]) for c in range(C)]
    level: List[int] = [0] * C

    queue: deque[int] = deque()
    for c in range(C):
        if remaining[c] == 0:
            queue.append(c)

    while queue:
        c = queue.popleft()
        for p in depended_by[c]:
            if level[p] <= level[c]:
                level[p] = level[c] + 1
            remaining[p] -= 1
            if remaining[p] == 0:
                queue.append(p)

    max_wave: int = max(level) if C > 0 else 0
    waves: List[List[int]] = [[] for _ in range(max_wave + 1)]
    for c in range(C):
        waves[level[c]].append(c)

    # No phase info — label all as phase 0
    phases = [0] * len(waves)
    return waves, phases
