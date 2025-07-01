#!/usr/bin/env python3
"""Test simple clade extraction without all rootings."""

from ete3 import Tree

# Load tree
tree = Tree("test_trees_3/g.nwk", format=1)
all_leaves = set([l.name for l in tree.get_leaves()])

print(f"Tree has {len(all_leaves)} leaves")

# Extract clades from tree as-is
clades = set()

# Add leaf clades
for leaf in tree.get_leaves():
    clades.add(frozenset([leaf.name]))

# Add internal node clades  
for node in tree.traverse():
    if not node.is_leaf():
        node_leaves = frozenset([l.name for l in node.get_leaves()])
        clades.add(node_leaves)

# Add full tree clade
clades.add(frozenset(all_leaves))

print(f"Total clades: {len(clades)}")

# Count by size
size_counts = {}
for clade in clades:
    size = len(clade)
    size_counts[size] = size_counts.get(size, 0) + 1

print(f"Clade sizes: {sorted(size_counts.items())}")

# With leaf duplication
total_with_duplication = len(clades) + 10  # 10 leaf clades
print(f"With leaf duplication: {total_with_duplication}")