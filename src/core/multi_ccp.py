"""
Utilities for multi-family CCP handling.

- Build a single CCP per family from a set of gene trees (mixing trees → one CCP)
- Merge multiple families' CCP helpers into a single concatenated representation
  (batching families → one larger Pi over all clades)

We rebuild the sorted-by-parent segmentation globally so that segments with
len >= 2 (i.e., parents with >= 2 splits) come first, which is ideal for
seg_logsumexp kernels.
"""
from typing import List, Dict, Tuple
import torch
from .ccp import extract_ccp_from_trees, build_ccp_helpers, build_clade_species_mapping, get_root_clade_id
from .tree_helpers import build_species_helpers


def _rebuild_segmentation(
    split_parents: torch.Tensor,
    split_lefts: torch.Tensor,
    split_rights: torch.Tensor,
    log_split_probs: torch.Tensor,
    C: int,
) -> Dict:
    device = split_parents.device
    dtype = log_split_probs.dtype
    # Counts per parent
    split_counts = torch.bincount(split_parents, minlength=C)
    # Parent order: non-increasing counts
    parents_sorted = torch.argsort(split_counts, descending=True, stable=True)
    # Rank of each parent in the sorted list
    parent_rank = torch.empty(C, dtype=torch.long, device=device)
    parent_rank[parents_sorted] = torch.arange(C, dtype=torch.long, device=device)
    # Split order that groups by parent according to parents_sorted (stable within parent)
    split_order = torch.argsort(parent_rank.index_select(0, split_parents), stable=True)
    # Apply ordering
    split_parents_sorted = split_parents.index_select(0, split_order)
    split_lefts_sorted = split_lefts.index_select(0, split_order)
    split_rights_sorted = split_rights.index_select(0, split_order)
    log_split_probs_sorted = log_split_probs.index_select(0, split_order)
    # Build CSR-style pointers
    seg_counts = split_counts.index_select(0, parents_sorted)
    ptr = torch.empty(C + 1, dtype=torch.long, device=device)
    ptr[0] = 0
    torch.cumsum(seg_counts, dim=0, out=ptr[1:])
    # Partition segments by length
    num_segs_ge2 = int((seg_counts >= 2).sum().item())
    num_segs_eq1 = int((seg_counts == 1).sum().item())
    num_segs_eq0 = int((seg_counts == 0).sum().item())
    stop_reduce_ptr_idx = num_segs_ge2
    end_rows_ge2 = int(ptr[stop_reduce_ptr_idx].item())
    ptr_ge2 = ptr[: stop_reduce_ptr_idx + 1].clone()
    # Build leftright selector
    split_leftrights_sorted = torch.cat((split_lefts_sorted, split_rights_sorted), dim=0)
    return {
        'split_counts': split_counts,
        'split_order': split_order,
        'split_parents_sorted': split_parents_sorted,
        'split_lefts_sorted': split_lefts_sorted,
        'split_rights_sorted': split_rights_sorted,
        'split_leftrights_sorted': split_leftrights_sorted,
        'log_split_probs_sorted': log_split_probs_sorted,
        'parents_sorted': parents_sorted,
        'seg_parent_ids': parents_sorted,
        'seg_counts': seg_counts,
        'ptr': ptr,
        'ptr_ge2': ptr_ge2,
        'num_segs_ge2': num_segs_ge2,
        'num_segs_eq1': num_segs_eq1,
        'num_segs_eq0': num_segs_eq0,
        'stop_reduce_ptr_idx': stop_reduce_ptr_idx,
        'end_rows_ge2': end_rows_ge2,
        'C': C,
        'N_splits': int(split_parents.shape[0]),
    }


def merge_ccp_helpers(helpers_list: List[Dict], device: torch.device, dtype: torch.dtype) -> Dict:
    """Merge multiple per-family ccp_helpers into a single concatenated helper.

    The inputs must contain at least:
      - 'C', 'N_splits'
      - 'split_parents_sorted', 'split_lefts_sorted', 'split_rights_sorted'
      - 'log_split_probs_sorted'

    Returns a helper dict in the same shape as build_ccp_helpers output,
    but with C = sum C_i and N_splits = sum N_splits_i, and with a global
    sorted-by-parent segmentation.
    """
    # Offset and concatenate
    total_C = sum(int(h['C']) for h in helpers_list)
    parts_parents = []
    parts_lefts = []
    parts_rights = []
    parts_logs = []
    offset = 0
    for h in helpers_list:
        parents = h['split_parents_sorted'].to(device=device)
        lefts = h['split_lefts_sorted'].to(device=device)
        rights = h['split_rights_sorted'].to(device=device)
        logs = h['log_split_probs_sorted'].to(device=device, dtype=dtype)
        if offset:
            parents = parents + offset
            lefts = lefts + offset
            rights = rights + offset
        parts_parents.append(parents)
        parts_lefts.append(lefts)
        parts_rights.append(rights)
        parts_logs.append(logs)
        offset += int(h['C'])
    split_parents = torch.cat(parts_parents, dim=0)
    split_lefts = torch.cat(parts_lefts, dim=0)
    split_rights = torch.cat(parts_rights, dim=0)
    log_split_probs = torch.cat(parts_logs, dim=0)
    # Rebuild segmentation globally
    merged = _rebuild_segmentation(split_parents, split_lefts, split_rights, log_split_probs, total_C)
    return merged


def concat_clade_species_maps(maps: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate clade-species mapping [C_i,S] along clade dimension."""
    return torch.cat(maps, dim=0)


def offset_root_indices(root_indices: List[int], helper_list: List[Dict]) -> List[int]:
    out = []
    offset = 0
    for ridx, h in zip(root_indices, helper_list):
        out.append(int(ridx) + offset)
        offset += int(h['C'])
    return out


def build_aggregated_families_ccp(
    species_tree_path: str,
    families: "Dict[str, List[str]] | List[List[str]]",
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[Dict], List[str], Dict]:
    """Build one CCP per family by mixing many gene trees per family.

    families can be either:
      - dict: {family_name: [tree1, tree2, ...]}
      - list of lists: [[tree1a, tree2a, ...], [tree1b, tree2b, ...], ...]

    Returns:
      - per_family: list of dicts with keys 'ccp_helpers', 'clade_species_map', 'root_clade_id', 'C', 'N_splits'
      - names: list of family names (or auto-generated indices as strings)
      - species_helpers: species helper dict (shared across families)
    """
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    per_family: List[Dict] = []
    names: List[str] = []
    if isinstance(families, dict):
        items = list(families.items())
    else:
        # auto-name families as f0, f1, ...
        items = [(f"f{i}", fam) for i, fam in enumerate(families)]
    for name, tree_paths in items:
        ccp = extract_ccp_from_trees(tree_paths)
        ccp_helpers = build_ccp_helpers(ccp, device, dtype)
        clade_species = build_clade_species_mapping(ccp, species_helpers, device, dtype)
        root_cid = get_root_clade_id(ccp)
        per_family.append({
            'ccp_helpers': ccp_helpers,
            'clade_species_map': clade_species,
            'root_clade_id': int(root_cid),
            'C': int(ccp_helpers['C']),
            'N_splits': int(ccp_helpers['N_splits']),
        })
        names.append(name)
    return per_family, names, species_helpers
