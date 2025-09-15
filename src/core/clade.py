"""
Core clade data structures for phylogenetic reconciliation.

This module defines the fundamental data structures for representing clades
and clade splits in the CCP (Conditional Clade Probability) framework.
"""

from typing import Set, Optional, Dict, List
from collections import defaultdict


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