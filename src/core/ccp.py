"""
Functions for building Conditional Clade Probabilities from gene trees.

This module provides functions to extract CCPs from single gene trees or
distributions of gene trees, following the ALE (Amalgamated Likelihood Estimation)
framework.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from ete3 import Tree
import torch
import re

from .clade import Clade, CladeSplit, CCPContainer


def build_ccp_from_single_tree(gene_tree_path: str, debug: bool = False) -> CCPContainer:
    """
    Build CCP container from a single gene tree with uniform rooting probabilities.
    For unrooted trees, each of the 2n-3 possible rootings should have equal probability 1/(2n-3).
    
    Args:
        gene_tree_path: Path to the gene tree file
        debug: Whether to print debug information
    """
    tree = Tree(gene_tree_path, format=1)
    
    ccp_container = CCPContainer()
    all_leaves = {leaf.name for leaf in tree.get_leaves()}
    n_leaves = len(all_leaves)
    
    # Count actual internal edges (potential rooting positions)
    # In an unrooted binary tree, we have 2n-3 edges total
    n_rootings = 2 * n_leaves - 3  # Expected number of rootings
    
    # Add the full tree clade (root clade containing all leaves)
    root_clade = Clade(all_leaves)
    ccp_container.add_clade(root_clade)
    
    # Each edge in the unrooted tree corresponds to one possible rooting
    # We need to avoid double-counting edges by ensuring each split is added only once
    splits_added = set()
    
    for node in tree.traverse():
        if node.is_root():
            continue  # Skip the arbitrary root of the rooted representation
            
        # Each node (except root) defines one edge and thus one possible rooting
        # Split the tree at this edge: clade below the node vs clade above the node
        below_leaves = {leaf.name for leaf in node.get_leaves()}
        above_leaves = all_leaves - below_leaves
        
        # Create canonical representation to avoid duplicates
        # Always put the smaller clade first
        if len(below_leaves) <= len(above_leaves):
            clade1_leaves, clade2_leaves = below_leaves, above_leaves
        else:
            clade1_leaves, clade2_leaves = above_leaves, below_leaves
        
        # Create a canonical split representation
        canonical_split = (frozenset(clade1_leaves), frozenset(clade2_leaves))
        
        # Only add if we haven't seen this split before
        if canonical_split not in splits_added:
            below_clade = Clade(below_leaves)
            above_clade = Clade(above_leaves)
            ccp_container.add_clade(below_clade)
            ccp_container.add_clade(above_clade)
            
            # Add the split representing this rooting with equal frequency
            ccp_container.add_split(root_clade, below_clade, above_clade, frequency=1.0)
            splits_added.add(canonical_split)
    
    # Verify we have the correct number of rootings
    root_splits = len(ccp_container.splits[root_clade])
    if debug:
        print(f"Debug: Found {root_splits} root splits, expected {n_rootings} for {n_leaves} leaves")
    
    # Note: The actual number of rootings might differ from 2n-3 due to tree structure
    # Accept the number we found and normalize probabilities accordingly
    if root_splits != n_rootings:
        print(f"Warning: Expected {n_rootings} rootings but found {root_splits}. Proceeding with found splits.")
    
    # Add all other clades and their internal splits for completeness
    # These are needed for the recursive computation but don't affect root probabilities
    for node in tree.traverse():
        if node.is_root() or node.is_leaf():
            continue
            
        # For internal nodes, add splits between their children
        children = node.get_children()
        if len(children) == 2:
            left, right = children
            left_leaves = {leaf.name for leaf in left.get_leaves()}
            right_leaves = {leaf.name for leaf in right.get_leaves()}
            above_leaves = all_leaves - left_leaves - right_leaves
            
            left_clade = Clade(left_leaves)
            right_clade = Clade(right_leaves)
            above_clade = Clade(above_leaves)
            
            # Add internal splits (these don't affect root clade probabilities)
            parent_clade = left_clade + right_clade
            ccp_container.add_clade(parent_clade)
            ccp_container.add_split(parent_clade, left_clade, right_clade, frequency=1.0)
            
            # Add splits involving the above clade
            if len(above_leaves) > 0:
                parent_left_above = left_clade + above_clade
                parent_right_above = right_clade + above_clade
                ccp_container.add_clade(parent_left_above)
                ccp_container.add_clade(parent_right_above)
                ccp_container.add_split(parent_left_above, left_clade, above_clade, frequency=1.0)
                ccp_container.add_split(parent_right_above, right_clade, above_clade, frequency=1.0)
    
    # Verify theoretical bounds for CCP construction
    n_leaves = len(all_leaves)
    expected_clades = 4 * n_leaves - 5  # 4n-5: each of 2n-3 branches defines 2 clades, plus full tree clade
    expected_splits = 5 * n_leaves - 9  # 5n-9: (2n-3) root splits + 3*(n-2) internal splits
    
    actual_clades = len(ccp_container.clades)
    actual_splits = sum(len(splits) for splits in ccp_container.splits.values())
    
    if debug:
        print(f"CCP construction verification for {n_leaves} leaves:")
        print(f"  Clades: {actual_clades} (expected {expected_clades})")
        print(f"  Splits: {actual_splits} (expected {expected_splits})")
    
    assert actual_clades == expected_clades, \
        f"Clade count mismatch: got {actual_clades}, expected {expected_clades} for {n_leaves} leaves"
    assert actual_splits == expected_splits, \
        f"Split count mismatch: got {actual_splits}, expected {expected_splits} for {n_leaves} leaves"
    
    # Normalize split probabilities
    ccp_container.compute_probabilities()
    
    return ccp_container


def extract_ccp_from_trees(tree_paths: List[str]) -> CCPContainer:
    """
    Extract CCPs from a distribution of gene trees.
    
    This function processes multiple gene trees and tracks the frequency
    of each clade split across the distribution.
    
    Args:
        tree_paths: List of paths to gene tree files
        
    Returns:
        CCPContainer with aggregated clades and split frequencies
    """
    ccp = CCPContainer()
    
    # Track split occurrences across all trees
    split_counter = Counter()
    
    for tree_path in tree_paths:
        tree = Tree(tree_path, format=1)
        
        # Process each internal node's split
        for node in tree.traverse("postorder"):
            if not node.is_leaf() and len(node.children) == 2:
                # Get clades for parent and children
                parent_leaves = {leaf.name for leaf in node.get_leaves()}
                left_leaves = {leaf.name for leaf in node.children[0].get_leaves()}
                right_leaves = {leaf.name for leaf in node.children[1].get_leaves()}
                
                # Create clade objects
                parent_clade = Clade(parent_leaves)
                left_clade = Clade(left_leaves)
                right_clade = Clade(right_leaves)
                
                # Register clades
                ccp.add_clade(parent_clade)
                ccp.add_clade(left_clade)
                ccp.add_clade(right_clade)
                
                # Count this split occurrence
                # Use frozensets to make the split hashable
                split_key = (parent_clade, left_clade, right_clade)
                split_counter[split_key] += 1
    
    # Add all splits with their frequencies
    for (parent, left, right), frequency in split_counter.items():
        ccp.add_split(parent, left, right, frequency=float(frequency))
    
    # Normalize to get probabilities
    ccp.compute_probabilities()
    
    return ccp


def get_root_clade_id(ccp: CCPContainer, debug: bool = False) -> int:
    """
    Find the ID of the root clade (containing all leaves).
    
    Args:
        ccp: CCP container
        debug: Whether to print debug information
        
    Returns:
        Integer ID of the root clade
        
    Raises:
        ValueError: If no unique root clade is found
    """
    # Get all root clades
    root_clades = []
    all_leaves = set()
    
    # Collect all leaf names
    for clade_id, clade in ccp.id_to_clade.items():
        if clade.is_leaf():
            all_leaves.update(clade.leaves)
    
    # Find clades containing all leaves
    for clade_id, clade in ccp.id_to_clade.items():
        if clade.leaves == all_leaves:
            root_clades.append(clade_id)
    
    if len(root_clades) == 0:
        raise ValueError("No root clade found (no clade contains all leaves)")
    if len(root_clades) > 1:
        # If multiple root clades exist, return the first one (they should be equivalent)
        # This can happen with the node/branch version duality
        if debug:
            print(f"Warning: Multiple root clades found: {root_clades}. Using the first one.")
    
    root_clade_id = root_clades[0]
    root_clade = ccp.id_to_clade[root_clade_id]
    
    if debug:
        print(f"Root clade ID: {root_clade_id}, Root clade: {root_clade}")
        print(f"Root clade contains {len(root_clade.leaves)} leaves: {sorted(root_clade.leaves)}")
    
    return root_clade_id


def build_ccp_helpers(ccp: CCPContainer, device: torch.device, dtype: torch.dtype) -> Dict:
    """
    Build helper tensors for GPU-parallel CCP computation.
    
    This function creates tensor representations of the CCP structure that
    enable efficient parallel computation on GPUs.
    
    Args:
        ccp: CCP container with clades and splits
        device: PyTorch device (CPU or CUDA)
        dtype: PyTorch data type for tensors
        
    Returns:
        Dictionary containing (key groups):
            - Base (original order):
              - split_parents: [N_splits] parent clade IDs (original enumeration order)
              - split_lefts:   [N_splits] left child clade IDs
              - split_rights:  [N_splits] right child clade IDs
              - log_split_probs: [N_splits] log probabilities
              - split_counts:  [C] number of splits per parent (original parent index order)
            - Sorted-by-parent (largest segments first):
              - split_order:   [N_splits] permutation to reorder splits so splits of same parent are contiguous
              - split_parents_sorted, split_lefts_sorted, split_rights_sorted, log_split_probs_sorted: [N_splits]
              - parents_sorted: [C] permutation of parent ids (segment index -> parent id), sorted by decreasing split count
              - seg_parent_ids: alias to parents_sorted
              - seg_counts:     [C] split counts in the parents_sorted order
              - ptr:            [C+1] CSR-style pointers over the sorted splits
            - Sizes/metadata:
              - C: Total number of clades
              - N_splits: Total number of splits
              - ccp: Reference to the original CCP container
    """
    C = len(ccp.clades)
    
    # Collect all splits in a consistent order
    all_splits = []
    for parent_clade in ccp.splits:
        for split in ccp.splits[parent_clade]:
            all_splits.append(split)
    
    N_splits = len(all_splits)
    
    if N_splits == 0:
        # Handle edge case of no splits (single leaf tree)
        return {
            'split_parents': torch.zeros(0, dtype=torch.long, device=device),
            'split_lefts': torch.zeros(0, dtype=torch.long, device=device),
            'split_rights': torch.zeros(0, dtype=torch.long, device=device),
            'log_split_probs': torch.zeros(0, dtype=dtype, device=device),
            'split_counts': torch.zeros(C, dtype=torch.long, device=device),
            'split_order': torch.zeros(0, dtype=torch.long, device=device),
            'ptr': torch.zeros(C + 1, dtype=torch.long, device=device),
            'parents_sorted': torch.arange(C, dtype=torch.long, device=device),
            'seg_parent_ids': torch.arange(C, dtype=torch.long, device=device),
            'seg_counts': torch.zeros(C, dtype=torch.long, device=device),
            # Segment reduction helpers
            'num_segs_ge2': 0,
            'num_segs_eq1': 0,
            'num_segs_eq0': C,
            'stop_reduce_ptr_idx': 0,     # index in ptr where ge2 segments stop
            'end_rows_ge2': 0,            # number of rows covered by ge2 segments
            'ptr_ge2': torch.zeros(1, dtype=torch.long, device=device),  # [1] = {0}
            'C': C,
            'N_splits': 0,
            'ccp': ccp
        }
    
    # Create tensors for split information
    split_parents = torch.zeros(N_splits, dtype=torch.long, device=device)
    split_lefts = torch.zeros(N_splits, dtype=torch.long, device=device)
    split_rights = torch.zeros(N_splits, dtype=torch.long, device=device)
    split_probs = torch.zeros(N_splits, dtype=dtype, device=device)
    
    for i, split in enumerate(all_splits):
        split_parents[i] = ccp.clade_to_id[split.parent]
        split_lefts[i] = ccp.clade_to_id[split.left]
        split_rights[i] = ccp.clade_to_id[split.right]
        split_probs[i] = split.probability
    
    # Base (original) log probs
    log_split_probs = torch.log(split_probs)

    # Counts per parent (original parent id order)
    split_counts = torch.bincount(split_parents, minlength=C)

    # Segment parent order: largest segments first; stable for ties by parent id
    parents_sorted = torch.argsort(split_counts, descending=True, stable=True)
    # Rank of each parent in the sorted list
    parent_rank = torch.empty(C, dtype=torch.long, device=device)
    parent_rank[parents_sorted] = torch.arange(C, dtype=torch.long, device=device)

    # Split order that groups by parent according to parents_sorted (stable within parent)
    split_order = torch.argsort(parent_rank.index_select(0, split_parents), stable=True)

    # Apply ordering to split-aligned tensors
    split_parents_sorted = split_parents.index_select(0, split_order)
    split_lefts_sorted = split_lefts.index_select(0, split_order)
    split_rights_sorted = split_rights.index_select(0, split_order)
    split_leftrights_sorted = torch.cat((split_lefts_sorted, split_rights_sorted), dim=0)
    log_split_probs_sorted = log_split_probs.index_select(0, split_order)

    # Build CSR-style pointers over the sorted splits
    seg_counts = split_counts.index_select(0, parents_sorted)
    ptr = torch.empty(C + 1, dtype=torch.long, device=device)
    ptr[0] = 0
    torch.cumsum(seg_counts, dim=0, out=ptr[1:])

    # Partition segments by length: >=2 first, then 1, then 0 (already ensured by descending sort)
    # Compute boundaries to run segmented LSE only on segments with length >= 2.
    num_segs_ge2 = int((seg_counts >= 2).sum().item())
    num_segs_eq1 = int((seg_counts == 1).sum().item())
    num_segs_eq0 = int((seg_counts == 0).sum().item())
    stop_reduce_ptr_idx = num_segs_ge2  # index in ptr for the end of ge2 segments
    end_rows_ge2 = int(ptr[stop_reduce_ptr_idx].item())  # total rows covered by ge2 segments
    ptr_ge2 = ptr[: stop_reduce_ptr_idx + 1].clone()  # independent small view for kernel calls

    return {
        # Base/original order
        'split_parents': split_parents,
        'split_lefts': split_lefts,
        'split_rights': split_rights,
        'log_split_probs': log_split_probs,
        'split_counts': split_counts,
        # Sorted-by-parent order
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
        # Segment reduction helpers
        'num_segs_ge2': num_segs_ge2,
        'num_segs_eq1': num_segs_eq1,
        'num_segs_eq0': num_segs_eq0,
        'stop_reduce_ptr_idx': stop_reduce_ptr_idx,
        'end_rows_ge2': end_rows_ge2,
        'ptr_ge2': ptr_ge2,
        # Sizes/metadata
        'C': C,
        'N_splits': N_splits,
        'ccp': ccp,
    }


def build_clade_species_mapping(ccp: CCPContainer, species_helpers: Dict, 
                               device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build mapping matrix between clades and species.
    
    Creates a binary matrix indicating which leaf clades map to which species.
    
    Args:
        ccp: CCP container with clades
        species_helpers: Dictionary with species tree information
        device: PyTorch device
        dtype: PyTorch data type
        
    Returns:
        Binary tensor of shape [C, S] where C is number of clades and S is number of species
    """
    C = len(ccp.clades)
    S = species_helpers["S"]
    
    clade_species_map = torch.zeros((C, S), dtype=dtype, device=device)
    
    for clade_id, clade in ccp.id_to_clade.items():
        if clade.is_leaf():
            leaf_name = clade.get_leaf_name()
            # Extract species name (handle both "5_6" -> "5" and "n223" -> "223" patterns)
            if '_' in leaf_name:
                spec_name = leaf_name.split('_')[0]  # Pattern: "5_6" -> "5"
            else:
                # Pattern: "n223" -> "223"
                match = re.match(r'n(\d+)', leaf_name)
                if match:
                    spec_name = match.group(1)
                else:
                    spec_name = leaf_name  # Fallback: use as-is
            
            if spec_name in species_helpers["species_by_name"]:
                species_node = species_helpers["species_by_name"][spec_name]
                species_idx = species_helpers["idx_sp"][species_node]
                clade_species_map[clade_id, species_idx] = 1.0
            else:
                raise KeyError(f"Species {spec_name} not found for gene leaf {leaf_name}")
    
    return clade_species_map
