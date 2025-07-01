#!/usr/bin/env python3
"""Test the 3-clade extraction method for unrooted trees."""

from ete3 import Tree

# Load tree (arbitrary rooting for traversal)
tree = Tree("test_trees_3/g.nwk", format=1)
all_leaves = set([l.name for l in tree.get_leaves()])

print(f"Tree has {len(all_leaves)} leaves")

# Extract clades following the 3-clade method
clades = set()

# For each internal node, extract 3 clades
internal_nodes = [n for n in tree.traverse() if not n.is_leaf()]
print(f"Internal nodes: {len(internal_nodes)}")

for node in internal_nodes:
    children = node.get_children()
    if len(children) == 2:
        # Get leaves for two children
        left_leaves = set([l.name for l in children[0].get_leaves()])
        right_leaves = set([l.name for l in children[1].get_leaves()])
        
        # The third clade is the complement
        node_leaves = set([l.name for l in node.get_leaves()])
        complement_leaves = all_leaves - node_leaves
        
        # Add all three clades
        clades.add(frozenset(left_leaves))
        clades.add(frozenset(right_leaves))
        if len(complement_leaves) > 0:  # Don't add empty clade
            clades.add(frozenset(complement_leaves))

# Add the full tree clade
clades.add(frozenset(all_leaves))

print(f"Total clades: {len(clades)}")
print(f"Expected: 3*(n-1)+1 = 3*9+1 = 28 clades")

# Count by size
size_counts = {}
for clade in clades:
    size = len(clade)
    size_counts[size] = size_counts.get(size, 0) + 1

print(f"Clade sizes: {sorted(size_counts.items())}")

# With leaf duplication
leaf_clades = [c for c in clades if len(c) == 1]
print(f"Leaf clades: {len(leaf_clades)}")
total_with_duplication = len(clades) + len(leaf_clades)
print(f"With leaf duplication: {total_with_duplication}")