"""
Helper functions for processing species trees.

This module provides functions to build tensor representations of species trees
for efficient GPU computation in phylogenetic reconciliation.
"""

from typing import Dict, List, Tuple, Optional
from ete3 import Tree
import torch
import numpy as np


def build_species_helpers(sp_tree_path: str, device, dtype, traversal_order="postorder"):
    """
    Build helper tensors and lookups for a rooted, strictly binary species tree.
    This function linearizes the tree according to `traversal_order`, assigns each
    node an index e in [0, S), and constructs a set of adjacency/utility tensors.
    Args:
        sp_tree_path (str):
            Newick string or file path accepted by `ete3.Tree(sp_tree_path, format=1)`.
        device (torch.device):
            Target device for returned torch tensors.
        dtype (torch.dtype):
            Floating dtype for dense numeric tensors (e.g., torch.float32, torch.float64).
        traversal_order (str, default="postorder"):
            Traversal used to enumerate nodes and define indices. Common values are
            "postorder", "preorder", "levelorder", etc. All index-based structures
            refer to this order.
    Returns:
        dict: A dictionary with the following entries (S = number of nodes; I = number of internal nodes):
            - s_C1 (torch.Tensor, shape [S, S], dtype=dtype, device=device):
                Left-child indicator matrix. For an internal node with index e and its left child
                index i_l, s_C1[e, i_l] = 1 and the rest of row e are 0. Leaf rows are all zeros.
                At most one non-zero per row.
            - s_C2 (torch.Tensor, shape [S, S], dtype=dtype, device=device):
                Right-child indicator matrix. For an internal node with index e and its right child
                index i_r, s_C2[e, i_r] = 1 and the rest of row e are 0. Leaf rows are all zeros.
                At most one non-zero per row.
            - s_P_indexes (torch.LongTensor, shape [I], device=device):
                For each internal node encountered in `species_nodes` order, stores its index e.
                Length equals the number of internal nodes (I).
            - s_C1_indexes (torch.LongTensor, shape [I], device=device):
                For each internal node encountered in `species_nodes` order, stores the index of its
                left child. Length equals the number of internal nodes (I).
            - s_C2_indexes (torch.LongTensor, shape [I], device=device):
                For each internal node encountered in `species_nodes` order, stores the index of its
                right child. Length equals the number of internal nodes (I).
            - ancestors_dense (torch.Tensor, shape [S, S], dtype=dtype, device=device):
                Ancestor indicator matrix. Row e marks all ancestors (including itself) of node e:
                ancestors_dense[e, a] = 1 iff node a is an ancestor of e (or a == e); 0 otherwise.
            - Recipients_mat (torch.Tensor, shape [S, S], dtype=dtype, device=device):
                Row-stochastic matrix over non-ancestors. Row e assigns a uniform probability mass
                to all nodes that are not ancestors of e (and not e itself). For S > 1, each row
                sums to 1; for S == 1, the row sums to 0.
            - sp_leaves_mask (torch.BoolTensor, shape [S], device=device):
                Boolean mask with True at indices corresponding to leaves, False otherwise.
            - idx_sp (dict[ete3.TreeNode, int]):
                Mapping from ETE3 TreeNode objects to their indices e in `species_nodes`.
            - species_by_name (dict[str, ete3.TreeNode]):
                Mapping from non-empty, unique node names to their corresponding TreeNode objects.
                Unnamed nodes are omitted.
            - species_nodes (list[ete3.TreeNode]):
                List of TreeNode objects in the chosen `traversal_order`. All index-based tensors
                reference this ordering.
            - S (int):
                Total number of nodes in the species tree (len(species_nodes)).
            - sp_names_by_idx (dict[int, str]):
                Mapping from node index e to its .name string (may be empty for unnamed nodes).
            - s_children_idx (dict[int, tuple[int, int]]):
                For each internal node index e, the pair (i_l, i_r) giving the left and right child
                indices, respectively. Leaves are not present as keys.
            - sp_internal_mask (torch.BoolTensor, shape [S], device=device):
                Boolean mask with True at internal nodes (non-leaves), the complement of sp_leaves_mask.
    Notes:
        - The tree is assumed strictly binary: every node has either 0 or exactly 2 children.
          Any other arity raises a ValueError.
        - "Left" and "Right" follow the order returned by `TreeNode.get_children()` from ETE3.
        - All floating-point tensors use the provided `dtype` and `device`. Index tensors use
          torch.long; masks use torch.bool.
    """

    """Build species tree helper matrices (same as original matmul_ale.py)."""
    sp_tree = Tree(sp_tree_path, format=1)
    species_nodes = list(sp_tree.traverse(traversal_order))
    S = len(species_nodes)
    idx_sp = {node: i for i, node in enumerate(species_nodes)}
    
    # Children matrices
    s_C1 = torch.zeros((S, S), dtype=dtype, device=device)
    s_C2 = torch.zeros((S, S), dtype=dtype, device=device)
    s_P_indexes = []
    s_C1_indexes = []
    s_C2_indexes = []
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
            s_P_indexes.append(e)
            s_C1_indexes.append(i_l)
            s_C2_indexes.append(i_r)
            s_children_idx[e] = (i_l, i_r)
        else:
            raise ValueError(f"Node {node.name} has {len(children)} children, expected 0, or 2")
    s_P_indexes = torch.tensor(s_P_indexes, dtype=torch.long, device=device)
    s_P_indexes = torch.cat((s_P_indexes, s_P_indexes + S), dim=0)  # For both children gather
    s_C1_indexes = torch.tensor(s_C1_indexes, dtype=torch.long, device=device)
    s_C2_indexes = torch.tensor(s_C2_indexes, dtype=torch.long, device=device)
    s_C12_indexes = torch.cat((s_C1_indexes, s_C2_indexes), dim=0)  # For both children gather
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
        "s_P_indexes": s_P_indexes,
        "s_C1_indexes": s_C1_indexes,
        "s_C2_indexes": s_C2_indexes,
        "s_C12_indexes": s_C12_indexes,
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