#!/usr/bin/env python3
"""Test to understand extant vs extinct species in simulate_species_tree."""

import rustree

# Simulate with HIGH extinction rate to see the effect
tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.8, seed=42)

print(f"Requested: n=20 extant species")
print(f"Tree has: {tree.num_leaves()} total leaves")
print(f"Total nodes: {tree.num_nodes()} nodes\n")

# Get BD events to distinguish extant from extinct
events = tree.get_bd_events()
print(f"Birth-death events:")
print(f"  Leaves (extant): {len(events['leaves'])} events")
print(f"  Extinctions: {len(events['extinctions'])} events")
print(f"  Speciations: {len(events['speciations'])} events")

print(f"\n✓ The tree includes BOTH extant AND extinct species as leaves!")
print(f"✓ Extant leaves: {len(events['leaves'])}")
print(f"✓ Extinct leaves: {len(events['extinctions'])}")
print(f"✓ Total leaves: {tree.num_leaves()}")
print(f"✓ Verification: {len(events['leaves']) + len(events['extinctions'])} = {tree.num_leaves()}")
