#!/usr/bin/env python3
"""
Example demonstrating birth-death events functionality in rustree.

This script shows how to:
1. Simulate a birth-death tree
2. Get events as a dictionary
3. Save events to CSV
4. Work with parsed Newick trees
"""

import sys
sys.path.insert(0, "../target/release")

import rustree
import tempfile
import os


def example_simulated_tree():
    """Example with a simulated birth-death tree."""
    print("=" * 60)
    print("Example 1: Simulated Birth-Death Tree")
    print("=" * 60)

    # Simulate a tree with 10 species
    tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.3, seed=42)
    print(f"Simulated tree with {tree.num_leaves()} leaves")
    print(f"Tree height: {tree.tree_height():.4f}")
    print()

    # Get events as dictionary
    events = tree.get_bd_events()
    print("Events dictionary keys:", list(events.keys()))
    print(f"Total events: {len(events['time'])}")
    print()

    # Count event types
    event_types = {}
    for event_type in events['event_type']:
        event_types[event_type] = event_types.get(event_type, 0) + 1

    print("Event type counts:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    print()

    # Show first few events
    print("First 5 events:")
    print(f"{'Time':<10} {'Node':<10} {'Event':<12} {'Child1':<10} {'Child2':<10}")
    print("-" * 60)
    for i in range(min(5, len(events['time']))):
        print(f"{events['time'][i]:<10.4f} "
              f"{events['node_name'][i]:<10} "
              f"{events['event_type'][i]:<12} "
              f"{events['child1_name'][i]:<10} "
              f"{events['child2_name'][i]:<10}")
    print()

    # Save to CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name

    tree.save_bd_events_csv(csv_path)
    print(f"Events saved to: {csv_path}")

    # Read and display file size
    file_size = os.path.getsize(csv_path)
    print(f"CSV file size: {file_size} bytes")

    # Show first few lines of CSV
    print("\nFirst 5 lines of CSV:")
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.rstrip()}")

    # Cleanup
    os.remove(csv_path)
    print()


def example_parsed_tree():
    """Example with a parsed Newick tree."""
    print("=" * 60)
    print("Example 2: Parsed Newick Tree")
    print("=" * 60)

    # Parse a Newick tree
    newick = "((A:1.0,B:1.0):0.5,(C:0.8,D:0.8):0.7):0.0;"
    tree = rustree.parse_species_tree(newick)
    print(f"Parsed tree: {newick}")
    print(f"Number of leaves: {tree.num_leaves()}")
    print()

    # Get events
    events = tree.get_bd_events()
    print(f"Total events: {len(events['time'])}")

    # Count event types
    event_types = {}
    for event_type in events['event_type']:
        event_types[event_type] = event_types.get(event_type, 0) + 1

    print("Event type counts:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    print()

    # Show all events (small tree)
    print("All events:")
    print(f"{'Time':<10} {'Node':<10} {'Event':<12} {'Child1':<10} {'Child2':<10}")
    print("-" * 60)
    for i in range(len(events['time'])):
        print(f"{events['time'][i]:<10.4f} "
              f"{events['node_name'][i]:<10} "
              f"{events['event_type'][i]:<12} "
              f"{events['child1_name'][i]:<10} "
              f"{events['child2_name'][i]:<10}")
    print()

    # Note: Parsed trees have no extinction events
    print("Note: Parsed trees only have Speciation and Leaf events")
    print("      (no Extinction events, as the tree only shows surviving lineages)")
    print()


def example_pure_birth():
    """Example with pure birth process (no extinction)."""
    print("=" * 60)
    print("Example 3: Pure Birth Process (mu = 0)")
    print("=" * 60)

    # Simulate with zero extinction
    tree = rustree.simulate_species_tree(n=8, lambda_=1.0, mu=0.0, seed=123)
    print(f"Simulated pure birth tree with {tree.num_leaves()} leaves")
    print()

    # Get events
    events = tree.get_bd_events()

    # Count event types
    event_types = {}
    for event_type in events['event_type']:
        event_types[event_type] = event_types.get(event_type, 0) + 1

    print("Event type counts:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    print()

    # Verify no extinctions
    extinction_count = events['event_type'].count('Extinction')
    print(f"Extinction events: {extinction_count}")
    print("Pure birth process (mu=0) produces no extinction events")
    print()


def example_with_extinction():
    """Example with high extinction rate."""
    print("=" * 60)
    print("Example 4: High Extinction Rate")
    print("=" * 60)

    # Simulate with high extinction
    tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.6, seed=456)
    print(f"Simulated tree with {tree.num_leaves()} leaves")
    print(f"Parameters: lambda=1.0, mu=0.6")
    print()

    # Get events
    events = tree.get_bd_events()

    # Count event types
    event_types = {}
    for event_type in events['event_type']:
        event_types[event_type] = event_types.get(event_type, 0) + 1

    print("Event type counts:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    print()

    # Show extinction events
    extinction_count = events['event_type'].count('Extinction')
    if extinction_count > 0:
        print(f"Found {extinction_count} extinction events")
        print("\nExtinction events:")
        print(f"{'Time':<10} {'Node':<10}")
        print("-" * 25)
        for i in range(len(events['time'])):
            if events['event_type'][i] == 'Extinction':
                print(f"{events['time'][i]:<10.4f} {events['node_name'][i]:<10}")
    else:
        print("No extinction events in this particular simulation")
        print("(This can happen with small trees even with mu > 0)")
    print()


if __name__ == "__main__":
    example_simulated_tree()
    example_parsed_tree()
    example_pure_birth()
    example_with_extinction()

    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
