# Birth-Death Events Tests

This directory contains comprehensive tests for the birth-death events functionality in rustree.

## Test File

- `test_bd_events.py` - Tests for `save_bd_events_csv()` and `get_bd_events()` methods

## Test Coverage

### save_bd_events_csv() Tests

**Simulated Trees:**
- File creation and content validation
- CSV header validation
- Event type validation (Speciation, Extinction, Leaf)
- Leaf events at present time (t=0)
- Speciation events with children
- Extinction events (with high mu)
- Zero extinction (pure birth process)
- Time ordering (events sorted by time)
- Non-negative times
- Non-empty node names

**Parsed Newick Trees:**
- Simple trees
- Single-leaf trees
- Binary trees
- Complex trees
- No extinction events in parsed trees

### get_bd_events() Tests

**Dictionary Structure:**
- Returns dictionary with correct keys
- All values are lists
- All lists have equal length
- Contains expected event types

**Event Validation:**
- Leaf events present
- Speciation events present
- Times sorted
- Times non-negative
- Valid event types
- Non-empty node names

**Parsed Trees:**
- Works with parsed Newick
- No extinctions in parsed trees

### Edge Cases

- Small trees (n=2)
- Large trees (n=100)
- File overwriting
- Invalid paths
- Reproducibility with seeds
- Consistency between CSV and dict outputs

### Integration Tests

- Complete workflows with simulated trees
- Complete workflows with parsed trees
- Consistency between `save_bd_events_csv()` and `get_bd_events()`

## Running the Tests

### Prerequisites

1. Build the rustree Python module:
```bash
cd rustree
maturin develop --release
```

2. Ensure Python dependencies are installed:
```bash
pip install pytest pandas  # optional, for enhanced testing
```

### Running Tests

Run all birth-death events tests:
```bash
python tests/python/test_bd_events.py
```

Run with pytest (if available):
```bash
pytest tests/python/test_bd_events.py -v
```

Run specific test:
```bash
python tests/python/test_bd_events.py  # Will run all tests in the file
```

## Test Output Format

Each test prints either:
- `PASS: test_name` - Test passed
- `FAIL: test_name - error message` - Test failed
- `ERROR: test_name - error message` - Test encountered an error

Final summary shows:
```
=== Results: X passed, Y failed ===
```

## Expected Behavior

### save_bd_events_csv()

Creates a CSV file with columns:
- `time`: Event time (backward from present at 0)
- `node_name`: Name of the node where event occurred
- `event_type`: One of 'Speciation', 'Extinction', or 'Leaf'
- `child1_name`: Name of first child (for speciation events, empty otherwise)
- `child2_name`: Name of second child (for speciation events, empty otherwise)

### get_bd_events()

Returns a dictionary with keys:
- `'time'`: List of event times (floats)
- `'node_name'`: List of node names (strings)
- `'event_type'`: List of event types (strings)
- `'child1_name'`: List of first child names (strings)
- `'child2_name'`: List of second child names (strings)

## Event Types

### Speciation
- Internal node where lineage splits into two
- Has two children
- Occurs at time > 0 (in the past)

### Extinction
- Lineage that went extinct before present
- No children (leaf in the full tree including extinct lineages)
- Only present in simulated trees with mu > 0

### Leaf
- Extant species at present time
- No children
- Always at time = 0.0
- Number of Leaf events equals number of extant species

## Notes

- Parsed Newick trees only generate Speciation and Leaf events (no Extinction)
- Simulated trees may have Extinction events when mu > 0
- Events are always sorted by time (oldest to most recent)
- All times are non-negative (backward time from present)
