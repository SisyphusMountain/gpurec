"""
Python wrapper that prepares inputs from the GPU Pi/E pipeline and calls the
pybind11 stochastic backtracking extension.

Build the extension locally with:
  python setup_ale_backtrack.py build_ext --inplace

Then import from this module.
"""
from typing import Dict, List
import torch
import math
import numpy as np

from ..core.likelihood import get_log_params
from .summaries import write_transfer_frequencies, write_events_with_labels

def _to_numpy_log(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _compute_pibar_log(Pi_log: torch.Tensor, Recipients: torch.Tensor) -> torch.Tensor:
    # Pi_log: [C,S], Recipients: [S,S], row-stochastic
    # Pibar = log( exp(Pi) @ Recipients^T ) in a stable way
    Pi_max = torch.max(Pi_log, dim=1, keepdim=True).values
    Pi_lin = torch.exp(Pi_log - Pi_max)           # [C,S]
    Pibar_lin = Pi_lin @ Recipients.T             # [C,S]
    Pibar = torch.log(Pibar_lin) + Pi_max         # [C,S]
    return Pibar

def _build_species_children(species_helpers: Dict) -> Dict[str, np.ndarray]:
    """Build left/right child arrays and leaf mask from compact helpers.

    Expects keys: 'S', 's_P_indexes', 's_C12_indexes'. For i in [0..I-1],
    parent = s_P_indexes[i], left = s_C12_indexes[i], right = s_C12_indexes[i+I].
    Leaves are nodes not present in s_P_indexes.
    """
    S = int(species_helpers['S'])
    left = np.full((S,), -1, dtype=np.int64)
    right = np.full((S,), -1, dtype=np.int64)
    sP = species_helpers['s_P_indexes'].detach().cpu().numpy().astype(np.int64)
    sP = sP[:sP.shape[0]//2] # Only take the first part of the tensor
    sC12 = species_helpers['s_C12_indexes'].detach().cpu().numpy().astype(np.int64)
    I = sP.shape[0]
    assert sC12.shape[0] == 2 * I, "s_C12_indexes must be length 2*I"
    # fill left/right
    for i in range(I):
        p = int(sP[i])
        c1 = int(sC12[i])
        c2 = int(sC12[i + I])
        left[p] = c1
        right[p] = c2
    # leaf mask: nodes not appearing as parents
    is_internal = np.zeros((S,), dtype=np.bool_)
    is_internal[sP] = True
    species_is_leaf = (~is_internal).astype(np.int8)
    return {'left': left, 'right': right, 'species_is_leaf': species_is_leaf}

def _build_clade_leaf_mask(clade_species_map_log: torch.Tensor) -> np.ndarray:
    # leaf clades have any finite entry in mapping matrix rows
    finite = torch.isfinite(clade_species_map_log)
    mask = finite.any(dim=1).to(torch.uint8).detach().cpu().numpy()
    return mask

def sample_reconciliations_cpp(
    fixed_points: Dict,
    samples: int = 1,
    seed: int = 42,
):
    """
    Prepare inputs from setup_fixed_points() result and call the C++ sampler.

    Returns list of scenarios, each a dict { events: [...], origin_species: int, gene_root: int }
    """
    # Try prebuilt; fall back to JIT compile via PyTorch on first use
    try:
        import _ale_backtrack  # type: ignore
    except Exception:
        from .ext_loader import get_ale_backtrack
        _ale_backtrack = get_ale_backtrack(verbose=False)

    Pi_log = fixed_points['Pi']              # [C,S] (log)
    E_log = fixed_points['E']                # [S]
    Ebar_log = fixed_points["Ebar"]
    E_s1_log = fixed_points["E_s1"]
    E_s2_log = fixed_points["E_s2"]
    ccp_helpers = fixed_points['ccp_helpers']
    species_helpers = fixed_points['species_helpers']
    clade_species_map_log = fixed_points['clade_species_map']  # log, -inf where unmapped

    # Build recipients and Pibar
    Recipients = species_helpers['Recipients_mat']            # [S,S] (linear)
    Pibar_log = _compute_pibar_log(Pi_log, Recipients)

    # Event log probabilities (normalized) from theta used by fixed_points
    # setup_fixed_points stores theta or reconstructs it; we can recompute from stored param tensor
    # We need to recover theta; Pi/E were computed with some theta, but fixed_points may not store it.
    # For now, recompute from Pi context is not possible; require user to provide it in fixed_points optionally.
    if 'theta' in fixed_points:
        theta = fixed_points['theta']
    else:
        raise ValueError("fixed_points dict must include 'theta' (log_delta, log_tau, log_lambda)")
    log_pS, log_pD, log_pT, log_pL = get_log_params(theta)

    # Species children/leaf mask
    ch = _build_species_children(species_helpers)
    left = ch['left']
    right = ch['right']
    species_is_leaf = ch['species_is_leaf']
    clade_is_leaf = _build_clade_leaf_mask(clade_species_map_log)

    # CCP arrays (sorted-by-parent)
    _split_lefts_sorted, _split_rights_sorted = torch.chunk(ccp_helpers["split_leftrights_sorted"], 2, dim=0)
    
    split_lefts_sorted = _split_lefts_sorted.detach().cpu().numpy().astype(np.int64)
    split_rights_sorted = _split_rights_sorted.detach().cpu().numpy().astype(np.int64)
    log_split_probs_sorted = ccp_helpers['log_split_probs_sorted'].detach().cpu().numpy().astype(np.float64)
    ptr = ccp_helpers['ptr'].detach().cpu().numpy().astype(np.int64)
    parents_sorted = ccp_helpers['parents_sorted'].detach().cpu().numpy().astype(np.int64)

    # Root clade id
    root_cid = fixed_points["root_clade_id"]
    if root_cid is None:
        raise ValueError("fixed_points must include 'root_clade_id'")

    # Scalars
    log_2 = math.log(2.0)

    scenarios = _ale_backtrack.sample_scenarios(
        _to_numpy_log(Pi_log).astype(np.float64, copy=False),
        _to_numpy_log(Pibar_log).astype(np.float64, copy=False),
        _to_numpy_log(E_log).astype(np.float64, copy=False),
        _to_numpy_log(Ebar_log).astype(np.float64, copy=False),
        _to_numpy_log(E_s1_log).astype(np.float64, copy=False),
        _to_numpy_log(E_s2_log).astype(np.float64, copy=False),
        split_lefts_sorted,
        split_rights_sorted,
        log_split_probs_sorted,
        ptr,
        parents_sorted,
        left,
        right,
        species_is_leaf,
        clade_is_leaf,
        species_helpers['Recipients_mat'].detach().cpu().numpy().astype(np.float64, copy=False),
        _to_numpy_log(clade_species_map_log).astype(np.float64, copy=False),
        float(log_pS.item()),
        float(log_pD.item()),
        float(log_pT.item()),
        float(log_2),
        int(root_cid),
        int(samples),
        int(seed),
    )
    return scenarios


def write_sampler_summaries(
    scenarios: List[Dict],
    species_helpers: Dict,
    ccp_helpers: Dict,
    out_dir: str,
    prefix: str = "",
    normalize_transfers: bool = True,
):
    """Write summary CSVs from sampled scenarios.

    - transfer frequencies: <out_dir>/<prefix>transfer_frequencies.csv
    - events with labels:   <out_dir>/<prefix>events_with_labels.csv
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    tf_path = os.path.join(out_dir, f"{prefix}transfer_frequencies.csv")
    ev_path = os.path.join(out_dir, f"{prefix}events_with_labels.csv")
    write_transfer_frequencies(scenarios, species_helpers, tf_path, normalize=normalize_transfers)
    write_events_with_labels(scenarios, species_helpers, ccp_helpers, ev_path)
    # Donated transfers per branch (parent->child)
    donated_path = os.path.join(out_dir, f"{prefix}donated_transfers_per_branch.txt")
    from .summaries import write_donated_transfers_per_branch
    write_donated_transfers_per_branch(scenarios, species_helpers, donated_path, normalize=normalize_transfers)
    # Per-family mean pairwise transfers (from -> to) in ALE summaries format
    mean_pairs_path = os.path.join(out_dir, f"{prefix}meanTransfers.txt")
    from .summaries import write_total_transfers_txt
    # normalize=True -> expected transfers per sampled reconciliation (mean)
    write_total_transfers_txt(scenarios, species_helpers, mean_pairs_path, normalize=True)
