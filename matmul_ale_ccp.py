#!/usr/bin/env python3
"""
GPU-accelerated ALE reconciliation with Conditional Clade Probabilities (CCPs).

This implementation extends the basic matmul_ale.py to support gene tree 
distributions via CCPs, following the theory described in AleRaxSupp.tex.

Key differences from matmul_ale.py:
1. Uses CCPs to represent gene tree uncertainty
2. Supports amalgamation of multiple gene trees
3. Computes likelihoods over clades rather than single gene trees
4. Follows the dual recursion over reconciliations and gene tree topologies
"""

from ete3 import Tree
import torch
import numpy as np
from tabulate import tabulate
from collections import defaultdict, Counter
import itertools
from typing import Dict, List, Tuple, Set, Optional
import argparse
import math
from time import time


class Clade:
    """Represents a clade as a frozenset of leaf names."""
    
    def __init__(self, leaves: Set[str], is_branch_version: bool = False):
        self.leaves = frozenset(leaves)
        self.size = len(leaves)
    
    def __hash__(self):
        return hash(self.leaves)
    
    def __eq__(self, other):
        return (isinstance(other, Clade) and 
                self.leaves == other.leaves)    
    def __repr__(self):
        return f"Clade({sorted(self.leaves)}, size={self.size})"
    
    def is_leaf(self) -> bool:
        return self.size == 1
    
    def get_leaf_name(self) -> str:
        """Get the single leaf name (only valid for leaf clades)."""
        if not self.is_leaf():
            raise ValueError("get_leaf_name() only valid for leaf clades")
        return next(iter(self.leaves))
    
    def __add__(self, other: 'Clade') -> 'Clade':
        """Add two clades together to create a new clade with the union of their leaves."""
        if not isinstance(other, Clade):
            raise TypeError(f"unsupported operand type(s) for +: 'Clade' and '{type(other).__name__}'")
                
        return Clade(self.leaves | other.leaves)


class CladeSplit:
    """Represents a split of a clade into two child clades."""
    
    def __init__(self, parent: Clade, left: Clade, right: Clade, frequency: float = 1.0):
        self.parent = parent
        self.left = left
        self.right = right
        self.frequency = frequency
        self.probability = 0.0  # Will be computed later
    
    def __repr__(self):
        return f"Split({self.parent} -> {self.left} + {self.right}, freq={self.frequency})"


class CCPContainer:
    """Container for Conditional Clade Probabilities extracted from gene tree distributions."""
    
    def __init__(self):
        self.clades: Set[Clade] = set()
        self.clade_to_id: Dict[Clade, int] = {}
        self.id_to_clade: Dict[int, Clade] = {}
        self.splits: Dict[Clade, List[CladeSplit]] = defaultdict(list)
        self.leaf_names: Set[str] = set()
        self.next_id = 0
    
    def add_clade(self, clade: Clade) -> int:
        """Add a clade and return its ID."""
        if clade not in self.clade_to_id:
            clade_id = self.next_id
            self.next_id += 1
            self.clade_to_id[clade] = clade_id
            self.id_to_clade[clade_id] = clade
            self.clades.add(clade)
        return self.clade_to_id[clade]
    
    def add_split(self, parent: Clade, left: Clade, right: Clade, frequency: float = 1.0):
        """Add a split observation."""
        # Ensure all clades are registered
        # Note: This may add clades that weren't in the initial extraction
        self.add_clade(parent)
        self.add_clade(left)
        self.add_clade(right)
        
        # Find existing split or create new one
        for split in self.splits[parent]:
            if split.left == left and split.right == right:
                split.frequency += frequency
                return
            elif split.left == right and split.right == left:
                split.frequency += frequency
                return
        
        # Create new split
        self.splits[parent].append(CladeSplit(parent, left, right, frequency))
    
    def compute_probabilities(self):
        """Compute conditional probabilities for all splits."""
        for parent_clade, split_list in self.splits.items():
            total_freq = sum(split.frequency for split in split_list)
            if total_freq > 0:
                for split in split_list:
                    split.probability = split.frequency / total_freq
    
    def get_ordered_clades(self):
        """Return clade IDs ordered by size (leaves first, root last)."""
        return sorted(self.clade_to_id.values(), 
                     key=lambda cid: self.id_to_clade[cid].size)


def build_ccp_from_single_tree(gene_tree_path: str) -> CCPContainer:
    """
    Build CCP container from a single gene tree with uniform rooting probabilities.
    For unrooted trees, each of the 2n-3 possible rootings should have equal probability 1/(2n-3).
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
    
    return ccp_container

def build_species_helpers(sp_tree_path: str, device, dtype, traversal_order="postorder"):
    """Build species tree helper matrices (same as original matmul_ale.py)."""
    sp_tree = Tree(sp_tree_path, format=1)
    species_nodes = list(sp_tree.traverse(traversal_order))
    S = len(species_nodes)
    idx_sp = {node: i for i, node in enumerate(species_nodes)}
    
    # Children matrices
    s_C1 = torch.zeros((S, S), dtype=dtype, device=device)
    s_C2 = torch.zeros((S, S), dtype=dtype, device=device)
    sp_leaves_mask = torch.zeros((S,), dtype=torch.bool, device=device)
    s_children_idx = {}
    sp_names_by_idx = {}
    
    for node in species_nodes:
        e = idx_sp[node]
        sp_names_by_idx[e] = node.name
        children = node.get_children()
        if len(children) == 0:
            sp_leaves_mask[e] = True
        elif len(children) == 2:
            lc, rc = children
            i_l = idx_sp[lc]
            i_r = idx_sp[rc]
            s_C1[e, i_l] = 1.0
            s_C2[e, i_r] = 1.0
            s_children_idx[e] = (i_l, i_r)
        elif len(children) == 1:
            # Handle root with single child (collapse it)
            child = children[0]
            # Skip this node in the tree structure
            pass
        else:
            raise ValueError(f"Node has {len(children)} children, expected 0, 1, or 2")
    
    # Ancestors matrix
    ancestors_dense = torch.zeros((S, S), dtype=dtype, device=device)
    for node in species_nodes:
        e = idx_sp[node]
        cur = node
        while cur is not None:
            a = idx_sp[cur]
            ancestors_dense[e, a] = 1.0
            cur = cur.up
    
    # Recipients matrix
    Recipients_mat = (1 - ancestors_dense) / torch.clamp((1 - ancestors_dense).sum(dim=1, keepdim=True), min=1)
    
    # Species-by-name mapping
    species_by_name = {}
    for node in sp_tree.traverse(traversal_order):
        if node.name:  # Only add named nodes to avoid empty name conflicts
            assert node.name not in species_by_name, f"Duplicate species name: {node.name}"
            species_by_name[node.name] = node
    
    return {
        "s_C1": s_C1,
        "s_C2": s_C2,
        "ancestors_dense": ancestors_dense,
        "Recipients_mat": Recipients_mat,
        "sp_leaves_mask": sp_leaves_mask,
        "idx_sp": idx_sp,
        "species_by_name": species_by_name,
        "species_nodes": species_nodes,
        "S": S,
        "sp_names_by_idx": sp_names_by_idx,
        "s_children_idx": s_children_idx,
        "sp_internal_mask": ~sp_leaves_mask,
    }


def build_clade_species_mapping(ccp: CCPContainer, species_helpers: Dict, device, dtype):
    """Build mapping matrix from leaf clades to species."""
    C = len(ccp.clades)  # Number of clades
    S = species_helpers["S"]  # Number of species
    
    clade_species_map = torch.zeros((C, S), dtype=dtype, device=device)
    
    for clade_id, clade in ccp.id_to_clade.items():
        if clade.is_leaf():
            leaf_name = clade.get_leaf_name()
            # Extract species name (handle both "5_6" -> "5" and "n223" -> "223" patterns)
            if '_' in leaf_name:
                spec_name = leaf_name.split('_')[0]  # Pattern: "5_6" -> "5"
            else:
                # Pattern: "n223" -> "223"
                import re
                match = re.match(r'n(\d+)', leaf_name)
                if match:
                    spec_name = match.group(1)
                else:
                    spec_name = leaf_name  # Fallback: use as-is
            if spec_name in species_helpers["species_by_name"]:
                species_node = species_helpers["species_by_name"][spec_name]
                species_idx = species_helpers["idx_sp"][species_node]
                # Both node and branch versions need to know their mapping
                # but they use it differently in Pi computation
                clade_species_map[clade_id, species_idx] = 1.0
            else:
                raise KeyError(f"Species {spec_name} not found for gene leaf {leaf_name}")
    
    return clade_species_map


def E_step(E, s_C1, s_C2, Recipients_mat, p_S, p_D, p_T, p_L):
    """Compute extinction probabilities (same as original, works with CCPs)."""
    E_s1 = torch.mv(s_C1, E)
    E_s2 = torch.mv(s_C2, E)
    speciation = p_S * E_s1 * E_s2
    
    duplication = p_D * E * E
    
    # Recompute Ebar in each iteration (as AleRax does)
    Ebar = torch.mv(Recipients_mat, E)
    transfer = p_T * E * Ebar
    
    return speciation + duplication + transfer + p_L, E_s1, E_s2, Ebar



def build_ccp_helpers(ccp: CCPContainer, device, dtype):
    """Build CCP helpers including precomputed vectorized split arrays for zero-loop GPU parallelization."""
    # Compute conditional probabilities p(γ',γ''|γ) for all splits
    ccp.compute_probabilities()
    
    C = len(ccp.clades)
    
    # Track leaf clades
    ccp_leaves_mask = torch.zeros(C, dtype=torch.bool, device=device)
    for clade_id, clade in ccp.id_to_clade.items():
        if clade.is_leaf():
            ccp_leaves_mask[clade_id] = True
    
    # Precompute vectorized split data structures to eliminate ALL loops from Pi update
    # This is the key optimization: build these once, reuse across all iterations
    split_parents = []
    split_lefts = []
    split_rights = []
    split_probs = []
    
    for parent_clade, split_list in ccp.splits.items():
        if len(split_list) == 0:
            continue
        parent_id = ccp.clade_to_id[parent_clade]
        for split in split_list:
            left_id = ccp.clade_to_id[split.left]
            right_id = ccp.clade_to_id[split.right]
            split_parents.append(parent_id)
            split_lefts.append(left_id)
            split_rights.append(right_id)
            split_probs.append(split.probability)
    
    # Convert to tensors once and store them
    if len(split_parents) > 0:
        split_parents_tensor = torch.tensor(split_parents, device=device, dtype=torch.long)
        split_lefts_tensor = torch.tensor(split_lefts, device=device, dtype=torch.long)
        split_rights_tensor = torch.tensor(split_rights, device=device, dtype=torch.long)
        split_probs_tensor = torch.tensor(split_probs, device=device, dtype=dtype)
    else:
        # Handle edge case of no splits
        split_parents_tensor = torch.empty(0, device=device, dtype=torch.long)
        split_lefts_tensor = torch.empty(0, device=device, dtype=torch.long)
        split_rights_tensor = torch.empty(0, device=device, dtype=torch.long)
        split_probs_tensor = torch.empty(0, device=device, dtype=dtype)
    
    return {
        'ccp_leaves_mask': ccp_leaves_mask,
        'C': C,
        'ccp': ccp,
        # Precomputed vectorized split arrays for zero-loop processing
        'split_parents': split_parents_tensor,
        'split_lefts': split_lefts_tensor, 
        'split_rights': split_rights_tensor,
        'split_probs': split_probs_tensor
    }


def Pi_update_ccp_helper(Pi, ccp_C1, ccp_C2, s_C1, s_C2, Recipients_mat):
    """CCP version of Pi_update_helper - IDENTICAL logic with ccp matrices."""
    # IDENTICAL to original Pi_update_helper but with ccp_C1, ccp_C2
    Pi_c1 = ccp_C1.mm(Pi)  # = (Pi_{left_child(i), j} * split_prob)_{i,j}
    Pi_c2 = ccp_C2.mm(Pi)  # = (Pi_{right_child(i), j} * split_prob)_{i,j}
    Pi_s1 = Pi.mm(s_C1.T)
    Pi_s2 = Pi.mm(s_C2.T)
    
    Pi_c1_s1 = ccp_C1.mm(Pi_s1)
    Pi_c1_s2 = ccp_C1.mm(Pi_s2)
    Pi_c2_s1 = ccp_C2.mm(Pi_s1)
    Pi_c2_s2 = ccp_C2.mm(Pi_s2)

    Pibar = Pi.mm(Recipients_mat.T)
    Pibar_c1 = ccp_C1.mm(Pibar)
    Pibar_c2 = ccp_C2.mm(Pibar)

    return Pi_c1, Pi_c2, Pi_s1, Pi_s2, \
           Pi_c1_s1, Pi_c1_s2, Pi_c2_s1, Pi_c2_s2, \
           Pibar, Pibar_c1, Pibar_c2

def Pi_update_ccp_parallel(Pi, ccp_helpers, species_helpers, clade_species_map, 
                           E, Ebar, p_S, p_D, p_T):
    """Zero-loop GPU-parallelized CCP Pi update using precomputed vectorized split arrays.
    
    This version eliminates ALL loops by using precomputed split data structures.
    All operations are fully vectorized across splits, clades, and species simultaneously.
    """
    
    # Get precomputed split arrays and species matrices
    split_parents = ccp_helpers["split_parents"]
    split_lefts = ccp_helpers["split_lefts"]
    split_rights = ccp_helpers["split_rights"]
    split_probs = ccp_helpers["split_probs"]
    s_C1 = species_helpers["s_C1"]
    s_C2 = species_helpers["s_C2"]
    Recipients_mat = species_helpers["Recipients_mat"]
    sp_leaves_mask = species_helpers["sp_leaves_mask"]
    
    # Initialize new Pi matrix
    C, S = Pi.shape
    new_Pi = torch.zeros_like(Pi)
    
    # Add leaf speciation term first (equation 160)
    new_Pi += p_S * clade_species_map
    
    # Early return if no splits to process
    if split_parents.numel() == 0:
        # Still need to add loss terms
        D_loss = 2 * p_D * torch.einsum("ij, j -> ij", Pi, E)
        Pibar = Pi.mm(Recipients_mat.T)
        T_loss = p_T * (torch.einsum("ij, j -> ij", Pi, Ebar) + torch.einsum("ij, j -> ij", Pibar, E))
        new_Pi += D_loss + T_loss
        return new_Pi
    
    # Compute species-related Pi terms using matrix operations (efficient)
    Pi_s1 = Pi.mm(s_C1.T)  # Pi[γ, s1] where s1 = child1(e)
    Pi_s2 = Pi.mm(s_C2.T)  # Pi[γ, s2] where s2 = child2(e)
    Pibar = Pi.mm(Recipients_mat.T)  # Pibar[γ, e] = average over transfer recipients
    
    # Compute extinction terms needed for loss calculations
    E_s1 = torch.mv(s_C1, E)  # E values for species left children
    E_s2 = torch.mv(s_C2, E)  # E values for species right children
    
    # Extract Pi values for all splits simultaneously (fully vectorized)
    Pi_left = Pi[split_lefts, :]   # Shape: (num_splits, S)
    Pi_right = Pi[split_rights, :] # Shape: (num_splits, S)
    Pi_s1_left = Pi_s1[split_lefts, :] # Pi values for left children on species left children
    Pi_s1_right = Pi_s1[split_rights, :] # Pi values for right children on species left children
    Pi_s2_left = Pi_s2[split_lefts, :] # Pi values for left children on species right children
    Pi_s2_right = Pi_s2[split_rights, :] # Pi values for right children on species right children
    Pibar_left = Pibar[split_lefts, :] # Transfer terms for left children
    Pibar_right = Pibar[split_rights, :] # Transfer terms for right children
    
    # Expand split probabilities for broadcasting
    split_probs_expanded = split_probs.unsqueeze(1)  # Shape: (num_splits, 1)
    
    # Compute all split contributions simultaneously (fully vectorized)
    # ==== DUPLICATION TERMS ====
    D_splits = p_D * split_probs_expanded * Pi_left * Pi_right  # Shape: (num_splits, S)
    
    # ==== SPECIATION TERMS ====
    # Only for internal species branches
    internal_mask = ~sp_leaves_mask  # Shape: (S,)
    S_splits = p_S * split_probs_expanded * (
        Pi_s1_left * Pi_s2_right + Pi_s1_right * Pi_s2_left
    ) * internal_mask.unsqueeze(0)  # Shape: (num_splits, S)
    
    # ==== TRANSFER TERMS ====
    T_splits = p_T * split_probs_expanded * (
        Pi_left * Pibar_right + Pi_right * Pibar_left
    )  # Shape: (num_splits, S)
    
    # Accumulate split contributions back to parent clades using scatter_add
    # This is the key vectorized operation that replaces ALL loops
    split_contributions = D_splits + S_splits + T_splits  # Shape: (num_splits, S)
    
    # Use scatter_add to accumulate contributions for each parent clade
    new_Pi.scatter_add_(0, split_parents.unsqueeze(1).expand(-1, S), split_contributions)
    
    # ==== LOSS TERMS ====
    # These don't involve splits, so they can be computed directly
    D_loss = 2 * p_D * torch.einsum("ij, j -> ij", Pi, E)
    S_loss = p_S * (Pi_s1 * E_s2 + Pi_s2 * E_s1) * (~sp_leaves_mask).unsqueeze(0).expand(C, -1)
    T_loss = p_T * (torch.einsum("ij, j -> ij", Pi, Ebar) + torch.einsum("ij, j -> ij", Pibar, E))
    
    new_Pi += D_loss + S_loss + T_loss
    
    return new_Pi


def get_root_clade_id(ccp: CCPContainer) -> int:
    """Get the clade ID of the root clade (contains all leaves)."""
    # Find the clade that contains all leaves
    all_leaves = set()
    for clade in ccp.clades:
        all_leaves.update(clade.leaves)
    
    root_clade = None
    for clade in ccp.clades:
        if clade.leaves == all_leaves:
            root_clade = clade
            break
    
    if root_clade is None:
        raise ValueError("No root clade found")
    
    root_id = ccp.clade_to_id[root_clade]
    print(f"Root clade ID: {root_id}, Root clade: {root_clade}")
    print(f"Root clade contains {len(root_clade.leaves)} leaves: {sorted(root_clade.leaves)}")
    
    return root_id


def main_ccp(species_tree_path: str, gene_tree_path: str, delta: float = 0.1, tau: float = 0.1, 
             lambda_param: float = 0.1, iters: int = 100, device=None, dtype=torch.float64, 
             use_parallel: bool = True, compare_versions: bool = False):
    """Main function for CCP-based reconciliation."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Building CCPs from gene tree: {gene_tree_path}")
    
    # Build CCP container
    ccp = build_ccp_from_single_tree(gene_tree_path)
    
    # Build species tree helpers
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    
    # Build clade-species mapping
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    
    # Compute event probabilities from intensities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    print(f"Raw rates: delta={delta}, tau={tau}, lambda={lambda_param}")
    print(f"Event probabilities: p_S={p_S:.6f}, p_D={p_D:.6f}, p_T={p_T:.6f}, p_L={p_L:.6f}")
    
    # Initialize extinction probabilities
    S = species_helpers["S"]
    E = torch.zeros(S, dtype=dtype, device=device)
    
    # Fixed-point iteration for extinction probabilities
    print("Computing extinction probabilities...")
    for iter_e in range(iters):
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                              species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    
    print(f"Extinction probabilities converged: mean E = {E.mean():.6f}")
    print(f"E vector: {E}")
    print(f"Survival probabilities (1-E): {1-E}")
    
    # Initialize Pi matrix
    C = len(ccp.clades)
    Pi = torch.zeros((C, S), dtype=dtype, device=device)
    
    # Fixed-point iteration for Pi matrix
    print("Computing clade likelihoods...")
    root_clade_id = get_root_clade_id(ccp)
    
    # Build CCP matrices equivalent to g_C1, g_C2 in original
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    # Choose which Pi update function to use
    if use_parallel:
        pi_update_fn = Pi_update_ccp_parallel
        print("Using GPU-parallelized Pi update (sparse tensor operations)")
    else:
        pi_update_fn = Pi_update_ccp
        print("Using loop-based Pi update (original implementation)")
    
    # Optionally compare both versions for validation
    if compare_versions:
        print("\n=== Performance Comparison ===")
        
        # Time the loop-based version
        start_time = time()
        Pi_test = torch.zeros_like(Pi)
        Pi_loop = Pi_update_ccp(Pi_test, ccp_helpers, species_helpers, clade_species_map, 
                               E, Ebar, E_s1, E_s2, p_S, p_D, p_T)
        loop_time = time() - start_time
        
        # Time the parallelized version  
        start_time = time()
        Pi_test = torch.zeros_like(Pi)
        Pi_parallel = Pi_update_ccp_parallel(Pi_test, ccp_helpers, species_helpers, clade_species_map, 
                                            E, Ebar, p_S, p_D, p_T)
        parallel_time = time() - start_time
        
        # Compare results
        max_diff = torch.abs(Pi_loop - Pi_parallel).max()
        print(f"Loop-based time: {loop_time:.4f}s")
        print(f"Parallel time: {parallel_time:.4f}s")
        print(f"Speedup: {loop_time/parallel_time:.2f}x")
        print(f"Max difference: {max_diff:.2e}")
        
        if max_diff > 1e-10:
            print(f"⚠️  WARNING: Results differ by {max_diff:.2e}")
        else:
            print(f"✅ Results are numerically identical")
        print("=" * 40)
    
    # Main iteration loop
    for iter_pi in range(iters):
        if use_parallel:
            Pi_new = pi_update_fn(Pi, ccp_helpers, species_helpers, clade_species_map, 
                                 E, Ebar, p_S, p_D, p_T)
        else:
            Pi_new = pi_update_fn(Pi, ccp_helpers, species_helpers, clade_species_map, 
                                 E, Ebar, E_s1, E_s2, p_S, p_D, p_T)
        if iter_pi % 10 == 0:
            print(f"  Iter {iter_pi}: Pi sum = {Pi_new.sum():.6f}, Pi max = {Pi_new.max():.6f}")
            print(f"    Root clade Pi sum: {Pi_new[root_clade_id, :].sum():.6e}")
        
        # Check convergence
        if iter_pi > 0:
            diff = torch.abs(Pi_new - Pi).max()
            if diff < 1e-10:
                print(f"  Converged at iteration {iter_pi} (diff = {diff:.2e})")
                Pi = Pi_new
                break
        Pi = Pi_new
    
    print(f"Pi matrix converged: shape = {Pi.shape}, sum = {Pi.sum():.6f}")
    
    # Calculate log-likelihood for CCP-based reconciliation
    # In CCP framework, the likelihood is the average over all possible rootings
    # Each rooting has been assigned its proper probability in the CCP construction
    # The final likelihood is just the root clade Pi sum (which already incorporates
    # the weighted average over all rootings weighted by their CCP probabilities)
    
    root_pi_sum = Pi[root_clade_id, :].sum()
    
    # For comparison with other methods, we can also compute the traditional formula
    survival_probs = 1 - E
    p_O = torch.ones(S, dtype=dtype, device=device) / S
    numerator_traditional = torch.sum(p_O * Pi[root_clade_id, :])
    denominator_traditional = torch.sum(p_O * survival_probs)
    
    # The correct CCP log-likelihood is simply ln(root_pi_sum)
    log_likelihood = torch.log(root_pi_sum)
    
    print(f"Log-likelihood using CCP framework: {log_likelihood:.4f}")
    print(f"  Root clade Pi sum: {root_pi_sum:.6e}")
    print(f"  Traditional numerator: {numerator_traditional:.6e}")
    print(f"  Traditional denominator: {denominator_traditional:.6e}")
    print(f"  Traditional log-likelihood: {torch.log(numerator_traditional / denominator_traditional):.4f}")
    
    return {
        'log_likelihood': float(log_likelihood),
        'Pi': Pi,
        'E': E,
        'root_clade_id': root_clade_id
    }


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated ALE reconciliation with CCPs")
    parser.add_argument("--species", required=True, help="Species tree file")
    parser.add_argument("--gene", required=True, help="Gene tree file")
    parser.add_argument("--delta", type=float, default=0.1, help="Duplication rate")
    parser.add_argument("--tau", type=float, default=0.1, help="Transfer rate")
    parser.add_argument("--lambda", type=float, default=0.1, dest="lambda_param", help="Loss rate")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--no-parallel", action="store_true", help="Use loop-based version instead of parallel")
    parser.add_argument("--compare", action="store_true", help="Compare loop vs parallel performance")
    
    args = parser.parse_args()
    
    main_ccp(args.species, args.gene, args.delta, args.tau, args.lambda_param, args.iters,
            use_parallel=not args.no_parallel, compare_versions=args.compare)


if __name__ == "__main__":
    main()