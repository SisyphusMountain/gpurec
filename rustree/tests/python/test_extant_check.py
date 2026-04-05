#!/usr/bin/env python3
"""Check the structure of bd_events."""

import rustree

tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.8, seed=42)
events = tree.get_bd_events()

print("Keys in events dict:")
print(events.keys())
print("\nFull events dict:")
for key, value in events.items():
    print(f"{key}: {len(value) if isinstance(value, list) else value}")
