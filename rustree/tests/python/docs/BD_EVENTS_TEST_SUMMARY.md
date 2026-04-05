# Birth-Death Events Testing Summary

## Created Files

### 1. Test File
**File:** `tests/python/test_bd_events.py`

Comprehensive test suite covering birth-death events functionality with **34 test cases**:

#### Test Categories

**save_bd_events_csv() - Simulated Trees (10 tests)**
- File creation and content validation
- CSV header correctness
- Valid event types
- Leaf events at t=0
- Speciation events with children
- Extinction events (when mu > 0)
- Zero extinction (pure birth)
- Time ordering
- Non-negative times
- Non-empty node names

**save_bd_events_csv() - Parsed Newick Trees (4 tests)**
- Simple trees
- Single-leaf trees
- Binary trees
- Complex trees

**get_bd_events() - Dictionary Structure (10 tests)**
- Returns dictionary
- Expected keys present
- All values are lists
- Equal list lengths
- Contains leaf events
- Contains speciation events
- Times sorted
- Times non-negative
- Valid event types
- Non-empty node names

**get_bd_events() - Parsed Trees (2 tests)**
- Parsed tree handling
- Single-leaf tree

**Edge Cases (6 tests)**
- Small trees (n=2)
- Large trees (n=100)
- File overwriting
- Invalid paths
- Reproducibility
- CSV/dict consistency

**Integration Tests (2 tests)**
- Simulated tree workflow
- Parsed tree workflow

### 2. Documentation
**File:** `tests/python/README_BD_EVENTS.md`

Complete documentation including:
- Test coverage overview
- Running instructions
- Expected behavior
- Event type descriptions
- CSV format specification
- Dictionary structure specification

### 3. Example Code
**File:** `examples/bd_events_example.py`

Four example scenarios:
1. Simulated birth-death tree
2. Parsed Newick tree
3. Pure birth process (mu=0)
4. High extinction rate

### 4. Python Bindings
**File:** `src/python/mod.rs` (modified)

Added two methods to `PySpeciesTree` class:

#### save_bd_events_csv(filepath: str)
Saves birth-death events to CSV file with columns:
- time
- node_name
- event_type
- child1_name
- child2_name

#### get_bd_events() -> dict
Returns dictionary with event data:
- 'time': List[float]
- 'node_name': List[str]
- 'event_type': List[str]
- 'child1_name': List[str]
- 'child2_name': List[str]

## Test Implementation Details

### Event Types Tested
1. **Speciation** - Lineage splits into two children
2. **Extinction** - Lineage goes extinct (only in simulated trees with mu > 0)
3. **Leaf** - Extant species at present (time = 0)

### Edge Cases Covered
- Single-leaf trees
- Binary trees
- Small trees (n=2)
- Large trees (n=100)
- Zero extinction (pure birth)
- High extinction rates
- Parsed vs simulated trees
- File I/O errors
- Data consistency

### Test Patterns Followed
Based on existing rustree test patterns:
- Clear test names describing what is tested
- Comprehensive assertions with descriptive messages
- Proper resource cleanup (temporary files)
- Error case testing
- Reproducibility testing with seeds
- Integration tests for complete workflows

## Usage Examples

### Basic Usage
```python
import rustree

# Simulate a tree
tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.3, seed=42)

# Get events as dictionary
events = tree.get_bd_events()
print(f"Total events: {len(events['time'])}")

# Save to CSV
tree.save_bd_events_csv("events.csv")
```

### With Pandas
```python
import rustree
import pandas as pd

tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.3, seed=42)
events = tree.get_bd_events()

# Convert to DataFrame
df = pd.DataFrame(events)
print(df.head())

# Analyze event types
print(df['event_type'].value_counts())
```

### Parsed Trees
```python
import rustree

# Parse Newick
newick = "((A:1.0,B:1.0):0.5,C:1.5):0.0;"
tree = rustree.parse_species_tree(newick)

# Get events (generated from tree structure)
events = tree.get_bd_events()

# Note: Parsed trees only have Speciation and Leaf events
# (no Extinction events)
```

## Running Tests

```bash
# Run test file directly
python rustree/tests/python/test_bd_events.py

# Or with pytest
pytest rustree/tests/python/test_bd_events.py -v

# Run examples
python rustree/examples/bd_events_example.py
```

## Implementation Notes

### Design Decisions

1. **Event Generation**: For both simulated and parsed trees, events are generated from the tree structure using `generate_events_from_tree()`. This ensures consistency and works with both tree sources.

2. **Dictionary vs DataFrame**: `get_bd_events()` returns a dictionary rather than a DataFrame to avoid requiring pandas as a dependency. Users can easily convert to DataFrame if needed.

3. **Time Convention**: Time is backward from present (present = 0), matching the birth-death simulation convention.

4. **CSV Format**: Uses standard CSV format with header row for easy import into other tools.

5. **Error Handling**: Comprehensive error checking with descriptive PyValueError messages.

### Future Enhancements

Potential future additions:
- Filter events by type
- Event statistics (mean event times, event rate estimation)
- Event visualization
- Support for additional event metadata
- Tree reconstruction from events

## Validation

All tests follow pytest conventions and can be run individually or as a suite. The test file includes:
- 34 independent test cases
- Comprehensive edge case coverage
- Integration tests
- Clear pass/fail output
- Automatic cleanup of temporary files
