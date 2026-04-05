#!/usr/bin/env python3
"""Check event types in the BD tree."""

import rustree
from collections import Counter

tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.8, seed=42)
events = tree.get_bd_events()

print(f"Requested: n=20 extant species")
print(f"Tree has: {tree.num_leaves()} total leaves\n")

# Count event types
event_counts = Counter(events['event_type'])
print("Event type counts:")
for event_type, count in sorted(event_counts.items()):
    print(f"  {event_type}: {count}")

# Check leaf names
leaf_names = tree.leaf_names()
print(f"\nTotal leaf names: {len(leaf_names)}")
print(f"First 10 leaf names: {leaf_names[:10]}")

# The extant species should be numbered 0 to n-1
print(f"\nAccording to documentation:")
print(f"  - Nodes 0 to {20-1} should be extant species")
print(f"  - Nodes {20} onwards are other nodes (internal or extinct)")

# Count leaves vs extinctions
n_leaves = sum(1 for et in events['event_type'] if et == 'Leaf')
n_extinctions = sum(1 for et in events['event_type'] if et == 'Extinction')
n_speciations = sum(1 for et in events['event_type'] if et == 'Speciation')

print(f"\nEvent breakdown:")
print(f"  Leaf (extant): {n_leaves}")
print(f"  Extinction: {n_extinctions}")
print(f"  Speciation: {n_speciations}")
print(f"  Total: {n_leaves + n_extinctions + n_speciations}")
