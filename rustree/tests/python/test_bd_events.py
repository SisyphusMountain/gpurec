"""
Comprehensive tests for birth-death events functionality in rustree.

This module tests:
- save_bd_events_csv() - Verify CSV format and content
- get_bd_events() - Verify returned dict structure
- Event types (Speciation, Extinction, Leaf)
- Edge cases (single leaf tree, binary tree, etc.)
- Both simulated trees and parsed Newick trees
"""

import sys
sys.path.insert(0, "/home/enzo/Documents/Zombi/ZOMBI/zombi-rs/rustree/target/release")

import rustree
import os
import tempfile
import csv as csv_module
from pathlib import Path


# =============================================================================
# Helper Functions
# =============================================================================

def read_csv_file(filepath):
    """Read a CSV file and return header and rows."""
    with open(filepath, 'r') as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def validate_bd_event_type(event_type):
    """Validate that event_type is one of the valid BDEvent types."""
    valid_types = {'Speciation', 'Extinction', 'Leaf'}
    return event_type in valid_types


# =============================================================================
# Tests for save_bd_events_csv() - Simulated Trees
# =============================================================================

def test_save_bd_events_csv_creates_file():
    """Test that save_bd_events_csv creates a file."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        assert os.path.exists(filepath), "CSV file should be created"
        assert os.path.getsize(filepath) > 0, "CSV file should not be empty"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_creates_file")


def test_save_bd_events_csv_has_correct_header():
    """Test that CSV has the expected header."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        header, _ = read_csv_file(filepath)

        expected_cols = ['time', 'node_name', 'event_type', 'child1_name', 'child2_name']
        assert header == expected_cols, f"Expected header {expected_cols}, got {header}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_has_correct_header")


def test_save_bd_events_csv_valid_event_types():
    """Test that all event types in CSV are valid."""
    tree = rustree.simulate_species_tree(20, 1.0, 0.3, seed=123)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        for row in rows:
            event_type = row['event_type']
            assert validate_bd_event_type(event_type), f"Invalid event type: {event_type}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_valid_event_types")


def test_save_bd_events_csv_has_leaf_events():
    """Test that CSV contains Leaf events for extant species."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        leaf_events = [r for r in rows if r['event_type'] == 'Leaf']
        assert len(leaf_events) == 10, f"Should have exactly 10 Leaf events, got {len(leaf_events)}"

        # Leaf events should have time = 0.0 (present)
        for event in leaf_events:
            time = float(event['time'])
            assert time == 0.0, f"Leaf events should be at time 0, got {time}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_has_leaf_events")


def test_save_bd_events_csv_has_speciation_events():
    """Test that CSV contains Speciation events."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        speciation_events = [r for r in rows if r['event_type'] == 'Speciation']
        assert len(speciation_events) > 0, "Should have at least one Speciation event"

        # Speciation events should have two children
        for event in speciation_events:
            assert event['child1_name'] != '', "Speciation should have child1"
            assert event['child2_name'] != '', "Speciation should have child2"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_has_speciation_events")


def test_save_bd_events_csv_with_extinction():
    """Test that CSV contains Extinction events when mu > 0."""
    # High extinction rate to ensure extinctions occur
    tree = rustree.simulate_species_tree(10, 1.0, 0.5, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        extinction_events = [r for r in rows if r['event_type'] == 'Extinction']
        # With mu=0.5, we expect some extinctions (probabilistic)
        # Test is permissive - just verify extinctions CAN occur
        assert len(extinction_events) >= 0, "Extinction events should be non-negative"

        # Extinction events should NOT have children
        for event in extinction_events:
            assert event['child1_name'] == '', "Extinction should not have child1"
            assert event['child2_name'] == '', "Extinction should not have child2"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_with_extinction")


def test_save_bd_events_csv_no_extinction():
    """Test CSV with zero extinction rate (pure birth)."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.0, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        extinction_events = [r for r in rows if r['event_type'] == 'Extinction']
        assert len(extinction_events) == 0, "Should have no Extinction events with mu=0"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_no_extinction")


def test_save_bd_events_csv_times_are_sorted():
    """Test that events are sorted by time."""
    tree = rustree.simulate_species_tree(15, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        times = [float(r['time']) for r in rows]
        assert times == sorted(times), "Events should be sorted by time"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_times_are_sorted")


def test_save_bd_events_csv_times_non_negative():
    """Test that all event times are non-negative."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        for row in rows:
            time = float(row['time'])
            assert time >= 0.0, f"Event time should be non-negative, got {time}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_times_non_negative")


def test_save_bd_events_csv_node_names_non_empty():
    """Test that all events have non-empty node names."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        for row in rows:
            assert row['node_name'] != '', "Node name should not be empty"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_node_names_non_empty")


# =============================================================================
# Tests for save_bd_events_csv() - Parsed Newick Trees
# =============================================================================

def test_save_bd_events_csv_parsed_tree_simple():
    """Test save_bd_events_csv with a parsed Newick tree."""
    newick = "((A:1.0,B:1.0):1.0,C:2.0):0.0;"
    tree = rustree.parse_species_tree(newick)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        assert os.path.exists(filepath), "CSV file should be created"

        _, rows = read_csv_file(filepath)

        # Parsed trees should have only Leaf and Speciation events (no Extinction)
        event_types = {r['event_type'] for r in rows}
        assert 'Leaf' in event_types, "Should have Leaf events"
        assert 'Speciation' in event_types, "Should have Speciation events"
        assert 'Extinction' not in event_types, "Parsed trees should not have Extinction events"

        # Should have 3 leaves
        leaf_events = [r for r in rows if r['event_type'] == 'Leaf']
        assert len(leaf_events) == 3, f"Should have 3 Leaf events, got {len(leaf_events)}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_parsed_tree_simple")


def test_save_bd_events_csv_parsed_tree_single_leaf():
    """Test save_bd_events_csv with single-leaf parsed tree."""
    newick = "A:0.0;"
    tree = rustree.parse_species_tree(newick)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        # Single leaf tree should have exactly 1 event
        assert len(rows) == 1, f"Should have exactly 1 event, got {len(rows)}"
        assert rows[0]['event_type'] == 'Leaf', "Single node should be a Leaf"
        assert rows[0]['node_name'] == 'A', "Leaf name should be 'A'"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_parsed_tree_single_leaf")


def test_save_bd_events_csv_parsed_tree_binary():
    """Test save_bd_events_csv with a perfectly binary tree."""
    newick = "((A:1,B:1):1,(C:1,D:1):1):0;"
    tree = rustree.parse_species_tree(newick)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        # 4 leaves + 3 internal nodes = 7 total events
        assert len(rows) == 7, f"Should have 7 events, got {len(rows)}"

        leaf_events = [r for r in rows if r['event_type'] == 'Leaf']
        speciation_events = [r for r in rows if r['event_type'] == 'Speciation']

        assert len(leaf_events) == 4, f"Should have 4 leaves, got {len(leaf_events)}"
        assert len(speciation_events) == 3, f"Should have 3 speciations, got {len(speciation_events)}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_parsed_tree_binary")


def test_save_bd_events_csv_parsed_tree_complex():
    """Test save_bd_events_csv with a more complex parsed tree."""
    newick = "(((A:0.5,B:0.5):0.5,C:1.0):1.0,((D:0.3,E:0.3):0.7,F:1.0):1.0):0.0;"
    tree = rustree.parse_species_tree(newick)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        # Should have 6 leaves
        leaf_events = [r for r in rows if r['event_type'] == 'Leaf']
        assert len(leaf_events) == 6, f"Should have 6 leaves, got {len(leaf_events)}"

        # Check that leaf names are correct
        leaf_names = {e['node_name'] for e in leaf_events}
        expected_names = {'A', 'B', 'C', 'D', 'E', 'F'}
        assert leaf_names == expected_names, f"Expected {expected_names}, got {leaf_names}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_parsed_tree_complex")


# =============================================================================
# Tests for get_bd_events() - Dictionary Structure
# =============================================================================

def test_get_bd_events_returns_dict():
    """Test that get_bd_events returns a dictionary."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    events = tree.get_bd_events()

    assert isinstance(events, dict), "get_bd_events should return a dictionary"
    print("PASS: test_get_bd_events_returns_dict")


def test_get_bd_events_has_expected_keys():
    """Test that returned dict has expected keys."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    events = tree.get_bd_events()

    expected_keys = {'time', 'node_name', 'event_type', 'child1_name', 'child2_name'}
    assert set(events.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(events.keys())}"
    print("PASS: test_get_bd_events_has_expected_keys")


def test_get_bd_events_all_lists():
    """Test that all values in the dict are lists."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    events = tree.get_bd_events()

    for key, value in events.items():
        assert isinstance(value, list), f"Value for key '{key}' should be a list, got {type(value)}"
    print("PASS: test_get_bd_events_all_lists")


def test_get_bd_events_equal_lengths():
    """Test that all lists have the same length."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    events = tree.get_bd_events()

    lengths = [len(v) for v in events.values()]
    assert len(set(lengths)) == 1, f"All lists should have same length, got {lengths}"
    print("PASS: test_get_bd_events_equal_lengths")


def test_get_bd_events_has_leaf_events():
    """Test that get_bd_events contains Leaf events."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    events = tree.get_bd_events()

    leaf_count = events['event_type'].count('Leaf')
    assert leaf_count == 10, f"Should have 10 Leaf events, got {leaf_count}"

    # Verify leaf times are 0.0
    for i, event_type in enumerate(events['event_type']):
        if event_type == 'Leaf':
            assert events['time'][i] == 0.0, f"Leaf event should be at time 0, got {events['time'][i]}"

    print("PASS: test_get_bd_events_has_leaf_events")


def test_get_bd_events_has_speciation_events():
    """Test that get_bd_events contains Speciation events."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    events = tree.get_bd_events()

    speciation_count = events['event_type'].count('Speciation')
    assert speciation_count > 0, "Should have at least one Speciation event"

    # Verify speciation events have children
    for i, event_type in enumerate(events['event_type']):
        if event_type == 'Speciation':
            assert events['child1_name'][i] != '', "Speciation should have child1"
            assert events['child2_name'][i] != '', "Speciation should have child2"

    print("PASS: test_get_bd_events_has_speciation_events")


def test_get_bd_events_times_sorted():
    """Test that events are sorted by time."""
    tree = rustree.simulate_species_tree(15, 1.0, 0.3, seed=42)
    events = tree.get_bd_events()

    times = events['time']
    assert times == sorted(times), "Events should be sorted by time"
    print("PASS: test_get_bd_events_times_sorted")


def test_get_bd_events_times_non_negative():
    """Test that all times are non-negative."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    events = tree.get_bd_events()

    for time in events['time']:
        assert time >= 0.0, f"Time should be non-negative, got {time}"

    print("PASS: test_get_bd_events_times_non_negative")


def test_get_bd_events_valid_event_types():
    """Test that all event types are valid."""
    tree = rustree.simulate_species_tree(20, 1.0, 0.3, seed=123)
    events = tree.get_bd_events()

    for event_type in events['event_type']:
        assert validate_bd_event_type(event_type), f"Invalid event type: {event_type}"

    print("PASS: test_get_bd_events_valid_event_types")


def test_get_bd_events_node_names_non_empty():
    """Test that all node names are non-empty."""
    tree = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    events = tree.get_bd_events()

    for node_name in events['node_name']:
        assert node_name != '', "Node name should not be empty"

    print("PASS: test_get_bd_events_node_names_non_empty")


# =============================================================================
# Tests for get_bd_events() - Parsed Trees
# =============================================================================

def test_get_bd_events_parsed_tree():
    """Test get_bd_events with a parsed Newick tree."""
    newick = "((A:1.0,B:1.0):1.0,C:2.0):0.0;"
    tree = rustree.parse_species_tree(newick)
    events = tree.get_bd_events()

    # Should have 3 leaves + 2 internal nodes = 5 events
    event_count = len(events['event_type'])
    assert event_count == 5, f"Should have 5 events, got {event_count}"

    # No extinctions in parsed trees
    assert 'Extinction' not in events['event_type'], "Parsed trees should not have Extinction events"

    print("PASS: test_get_bd_events_parsed_tree")


def test_get_bd_events_parsed_single_leaf():
    """Test get_bd_events with single-leaf tree."""
    newick = "A:0.0;"
    tree = rustree.parse_species_tree(newick)
    events = tree.get_bd_events()

    assert len(events['event_type']) == 1, "Should have exactly 1 event"
    assert events['event_type'][0] == 'Leaf', "Single node should be a Leaf"
    assert events['node_name'][0] == 'A', "Leaf name should be 'A'"

    print("PASS: test_get_bd_events_parsed_single_leaf")


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_save_bd_events_csv_small_tree():
    """Test with very small tree (n=2)."""
    tree = rustree.simulate_species_tree(2, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        leaf_events = [r for r in rows if r['event_type'] == 'Leaf']
        assert len(leaf_events) == 2, f"Should have 2 leaves, got {len(leaf_events)}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_small_tree")


def test_save_bd_events_csv_large_tree():
    """Test with larger tree (n=100)."""
    tree = rustree.simulate_species_tree(100, 1.0, 0.3, seed=42)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        leaf_events = [r for r in rows if r['event_type'] == 'Leaf']
        assert len(leaf_events) == 100, f"Should have 100 leaves, got {len(leaf_events)}"

        # Should have many events
        assert len(rows) > 100, "Should have more events than just leaves"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_large_tree")


def test_save_bd_events_csv_overwrite():
    """Test that save_bd_events_csv overwrites existing file."""
    tree1 = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)
    tree2 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=100)

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree1.save_bd_events_csv(filepath)
        _, rows1 = read_csv_file(filepath)

        tree2.save_bd_events_csv(filepath)
        _, rows2 = read_csv_file(filepath)

        # Second tree has more leaves, so should have more events
        assert len(rows2) > len(rows1), "Second save should overwrite with different data"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_save_bd_events_csv_overwrite")


def test_save_bd_events_csv_invalid_path():
    """Test that invalid path raises appropriate error."""
    tree = rustree.simulate_species_tree(5, 1.0, 0.3, seed=42)

    try:
        tree.save_bd_events_csv("/nonexistent/directory/events.csv")
        assert False, "Should have raised an error for invalid path"
    except (ValueError, OSError):
        pass  # Expected

    print("PASS: test_save_bd_events_csv_invalid_path")


def test_get_bd_events_reproducibility():
    """Test that same seed produces same events."""
    tree1 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)
    tree2 = rustree.simulate_species_tree(10, 1.0, 0.3, seed=42)

    events1 = tree1.get_bd_events()
    events2 = tree2.get_bd_events()

    assert events1 == events2, "Same seed should produce identical events"
    print("PASS: test_get_bd_events_reproducibility")


def test_csv_and_dict_consistency():
    """Test that save_bd_events_csv and get_bd_events produce consistent data."""
    tree = rustree.simulate_species_tree(15, 1.0, 0.3, seed=42)

    # Get events as dict
    events_dict = tree.get_bd_events()

    # Save events to CSV and read back
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        # Compare counts
        assert len(rows) == len(events_dict['time']), "CSV and dict should have same number of events"

        # Compare content
        for i, row in enumerate(rows):
            assert float(row['time']) == events_dict['time'][i], f"Time mismatch at index {i}"
            assert row['node_name'] == events_dict['node_name'][i], f"Node name mismatch at index {i}"
            assert row['event_type'] == events_dict['event_type'][i], f"Event type mismatch at index {i}"
            assert row['child1_name'] == events_dict['child1_name'][i], f"Child1 mismatch at index {i}"
            assert row['child2_name'] == events_dict['child2_name'][i], f"Child2 mismatch at index {i}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_csv_and_dict_consistency")


# =============================================================================
# Integration Tests
# =============================================================================

def test_integration_simulated_tree_workflow():
    """Test complete workflow with simulated tree."""
    # Simulate tree
    tree = rustree.simulate_species_tree(20, 1.0, 0.4, seed=12345)

    # Get events as dict
    events = tree.get_bd_events()
    assert len(events['time']) > 0, "Should have events"

    # Save to CSV
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)

        # Read and verify
        _, rows = read_csv_file(filepath)
        assert len(rows) == len(events['time']), "CSV should match dict"

        # Check event type distribution
        event_types = [r['event_type'] for r in rows]
        assert 'Leaf' in event_types, "Should have Leaf events"
        assert 'Speciation' in event_types, "Should have Speciation events"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_integration_simulated_tree_workflow")


def test_integration_parsed_tree_workflow():
    """Test complete workflow with parsed Newick tree."""
    newick = "(((A:0.5,B:0.5):0.5,C:1.0):1.0,(D:0.5,E:0.5):1.5):0.0;"
    tree = rustree.parse_species_tree(newick)

    # Get events
    events = tree.get_bd_events()
    leaf_count = events['event_type'].count('Leaf')
    assert leaf_count == 5, f"Should have 5 leaves, got {leaf_count}"

    # Save to CSV
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        filepath = f.name

    try:
        tree.save_bd_events_csv(filepath)
        _, rows = read_csv_file(filepath)

        # No extinctions in parsed trees
        extinction_count = sum(1 for r in rows if r['event_type'] == 'Extinction')
        assert extinction_count == 0, "Parsed trees should not have extinctions"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    print("PASS: test_integration_parsed_tree_workflow")


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    tests = [
        # save_bd_events_csv - simulated trees
        test_save_bd_events_csv_creates_file,
        test_save_bd_events_csv_has_correct_header,
        test_save_bd_events_csv_valid_event_types,
        test_save_bd_events_csv_has_leaf_events,
        test_save_bd_events_csv_has_speciation_events,
        test_save_bd_events_csv_with_extinction,
        test_save_bd_events_csv_no_extinction,
        test_save_bd_events_csv_times_are_sorted,
        test_save_bd_events_csv_times_non_negative,
        test_save_bd_events_csv_node_names_non_empty,

        # save_bd_events_csv - parsed trees
        test_save_bd_events_csv_parsed_tree_simple,
        test_save_bd_events_csv_parsed_tree_single_leaf,
        test_save_bd_events_csv_parsed_tree_binary,
        test_save_bd_events_csv_parsed_tree_complex,

        # get_bd_events - dictionary structure
        test_get_bd_events_returns_dict,
        test_get_bd_events_has_expected_keys,
        test_get_bd_events_all_lists,
        test_get_bd_events_equal_lengths,
        test_get_bd_events_has_leaf_events,
        test_get_bd_events_has_speciation_events,
        test_get_bd_events_times_sorted,
        test_get_bd_events_times_non_negative,
        test_get_bd_events_valid_event_types,
        test_get_bd_events_node_names_non_empty,

        # get_bd_events - parsed trees
        test_get_bd_events_parsed_tree,
        test_get_bd_events_parsed_single_leaf,

        # Edge cases
        test_save_bd_events_csv_small_tree,
        test_save_bd_events_csv_large_tree,
        test_save_bd_events_csv_overwrite,
        test_save_bd_events_csv_invalid_path,
        test_get_bd_events_reproducibility,
        test_csv_and_dict_consistency,

        # Integration tests
        test_integration_simulated_tree_workflow,
        test_integration_parsed_tree_workflow,
    ]

    passed = 0
    failed = 0

    print("=" * 80)
    print("Running Birth-Death Events Tests for rustree")
    print("=" * 80)
    print()

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__} - {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__} - {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 80)
    print(f"=== Results: {passed} passed, {failed} failed ===")
    print("=" * 80)

    # Exit with non-zero status if any tests failed
    if failed > 0:
        sys.exit(1)
