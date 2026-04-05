# ALERax Integration for Rustree

## Overview

Rustree now includes Python bindings to call ALERax for gene tree reconciliation. This allows you to reconcile existing gene trees with species trees and analyze the results using rustree's rich API.

## What Was Implemented

### Core Rust Module (`src/alerax.rs`)
- **Command execution**: Calls ALERax via subprocess with proper error handling
- **Input preparation**: Writes species tree, gene trees, and families.txt configuration
- **Output parsing**: Parses ALERax rates, likelihoods, and RecPhyloXML samples
- **Validation**: Checks gene tree coverage and leaf name mapping before running
- **All 100 reconciliation samples** are returned for uncertainty analysis

### Python Bindings (`src/python.rs`)
- **`reconcile_with_alerax()`**: Main function for reconciliation
- **`PyAleRaxResult`**: Class containing reconciled trees and rates
- **Flexible input formats**: Supports Newick strings, file paths, lists, and dictionaries

## Installation

### Prerequisites
1. Install ALERax:
   ```bash
   conda install -c bioconda alerax
   ```
   Or download from: https://github.com/BenoitMorel/AleRax

2. Build the Python module:
   ```bash
   cd /Users/enzomarsot/Documents/git/rustree
   maturin develop --release
   ```

## Usage

### Basic Example - Single Gene Tree

```python
import rustree

# Load species tree
species_tree = rustree.parse_species_tree("testdata/test_data_3/output_zombi/T/ExtantTree.nwk")

# Reconcile a single gene tree
results = rustree.reconcile_with_alerax(
    species_tree=species_tree,
    gene_trees="testdata/test_data_3/output_zombi/G/Gene_trees/1_prunedtree.nwk",
    seed=42  # For reproducibility
)

# Access results
result = results["1_prunedtree"]  # Family name derived from filename
print(f"Samples: {len(result.gene_trees)}")  # 100 reconciliation samples
print(f"Duplication rate: {result.duplication_rate:.4f}")
print(f"Loss rate: {result.loss_rate:.4f}")
print(f"Transfer rate: {result.transfer_rate:.4f}")
print(f"Log-likelihood: {result.likelihood:.2f}")
```

### Access Reconciled Trees

```python
# Get the best reconciliation (first sample has highest likelihood)
best_tree = result.gene_trees[0]

# Count DTL events
events = best_tree.count_events()
print(f"Speciations: {events[0]}")
print(f"Duplications: {events[1]}")
print(f"Transfers: {events[2]}")
print(f"Losses: {events[3]}")
print(f"Leaves: {events[4]}")

# Visualize with thirdkind
best_tree.to_svg("reconciled.svg", open_browser=True)

# Export to CSV for analysis
df = best_tree.to_csv()
print(df.head())

# Save as RecPhyloXML
best_tree.save_xml("reconciled.xml")
```

### Analyze Multiple Samples

```python
# Analyze event variation across all 100 samples
import numpy as np

all_events = []
for gene_tree in result.gene_trees:
    events = gene_tree.count_events()
    all_events.append({
        'duplications': events[1],
        'transfers': events[2],
        'losses': events[3]
    })

import pandas as pd
events_df = pd.DataFrame(all_events)
print("\nEvent statistics across 100 samples:")
print(events_df.describe())
```

### Access Summary Statistics

```python
# Get summary statistics (aggregated across all samples)
stats = result.statistics

# Mean event counts
mean_events = stats.mean_event_counts
print(f"Mean speciations: {mean_events.speciations:.2f}")
print(f"Mean duplications: {mean_events.duplications:.2f}")
print(f"Mean transfers: {mean_events.transfers:.2f}")
print(f"Mean losses: {mean_events.losses:.2f}")

# Mean transfers between species pairs
print("\nMean transfers between species:")
for source in stats.mean_transfers:
    for dest, count in stats.mean_transfers[source].items():
        print(f"  {source} → {dest}: {count:.2f}")

# Events per species node
print("\nEvents per species:")
for species, events in stats.events_per_species.items():
    print(f"\n{species}:")
    print(f"  Speciations: {events.speciations:.2f}")
    print(f"  Duplications: {events.duplications:.2f}")
    print(f"  Transfers: {events.transfers:.2f}")
    print(f"  Losses: {events.losses:.2f}")
```

### Batch Reconciliation - Multiple Gene Families

```python
# Reconcile multiple gene families at once
gene_trees = {
    "family_1": "testdata/test_data_3/output_zombi/G/Gene_trees/1_prunedtree.nwk",
    "family_2": "testdata/test_data_3/output_zombi/G/Gene_trees/2_prunedtree.nwk"
}

results = rustree.reconcile_with_alerax(
    species_tree=species_tree,
    gene_trees=gene_trees,
    seed=42
)

# Access results by family name
for family_name, result in results.items():
    print(f"\n{family_name}:")
    print(f"  D={result.duplication_rate:.4f}")
    print(f"  L={result.loss_rate:.4f}")
    print(f"  T={result.transfer_rate:.4f}")
    print(f"  Samples: {len(result.gene_trees)}")
```

### Advanced Options

```python
results = rustree.reconcile_with_alerax(
    species_tree=species_tree,
    gene_trees=gene_trees,
    output_dir="./alerax_output",      # Specify output directory
    num_samples=100,                    # Number of reconciliation samples
    model="PER-FAMILY",                 # or "GLOBAL" for shared rates
    seed=42,                            # Random seed
    keep_output=True,                   # Keep ALERax files for debugging
    alerax_path="alerax"                # Path to alerax executable
)

# Check ALERax output files
import os
print("ALERax files:", os.listdir("./alerax_output/alerax_output"))
```

## API Reference

### `reconcile_with_alerax()`

Reconcile gene trees with species tree using ALERax.

**Parameters:**
- `species_tree` (PySpeciesTree | str): Species tree as object, Newick string, or file path
- `gene_trees` (str | List[str] | Dict[str, str]): Gene tree(s) as:
  - Single Newick string or file path
  - List of Newick strings/paths
  - Dict mapping family names to Newick strings/paths
- `output_dir` (str | None): Output directory (default: temporary directory)
- `num_samples` (int): Number of reconciliation samples (default: 100)
- `model` (str): "PER-FAMILY" or "GLOBAL" (default: "PER-FAMILY")
- `seed` (int | None): Random seed for reproducibility
- `keep_output` (bool): Whether to preserve ALERax output (default: False)
- `alerax_path` (str): Path to alerax executable (default: "alerax")

**Returns:**
- `Dict[str, PyAleRaxResult]`: Dictionary mapping family names to results

### `PyAleRaxResult`

Result of ALERax reconciliation for one gene family.

**Attributes:**
- `gene_trees` (List[PyGeneTree]): All reconciliation samples (typically 100)
- `duplication_rate` (float): Estimated duplication rate (D)
- `loss_rate` (float): Estimated loss rate (L)
- `transfer_rate` (float): Estimated transfer rate (T)
- `likelihood` (float): Log-likelihood of the reconciliation
- `statistics` (ReconciliationStatistics): Summary statistics across all samples

### `ReconciliationStatistics`

Summary statistics aggregated across all reconciliation samples.

**Attributes:**
- `mean_event_counts` (EventCounts): Mean event counts across all samples
- `mean_transfers` (Dict[str, Dict[str, float]]): Mean transfer counts between species pairs
  - Format: `{source_species: {dest_species: mean_count}}`
- `events_per_species` (Dict[str, EventCounts]): Mean events per species node
  - Format: `{species_name: EventCounts}`

### `EventCounts`

Event counts from reconciliation analysis.

**Attributes:**
- `speciations` (float): Number of speciation events
- `duplications` (float): Number of duplication events
- `transfers` (float): Number of transfer events
- `losses` (float): Number of loss events
- `speciation_losses` (float): Number of speciation loss events
- `duplication_losses` (float): Number of duplication loss events
- `transfer_losses` (float): Number of transfer loss events
- `leaves` (float): Number of leaf nodes

## Input Requirements

ALERax requires:
1. **Gene tree leaves must have at least 4 species** (ALERax minimum)
2. **Gene leaf names must map to species names** using format: `species_geneid`
   - Example: `n5_20` → species `n5`, gene id `20`
3. **Species names must exist in the species tree**

## Error Handling

The implementation provides helpful error messages:

```python
# ALERax not installed
>>> results = rustree.reconcile_with_alerax(...)
ValueError: ALERax not found at 'alerax'. Please install ALERax:
  conda install -c bioconda alerax
  or download from: https://github.com/BenoitMorel/AleRax

# Insufficient species coverage
>>> results = rustree.reconcile_with_alerax(species_tree, small_gene_tree)
ValueError: Input validation failed: Gene tree 'family_0' covers only 2 species
(minimum 4 required for reconciliation)

# Mismatched leaf names
>>> results = rustree.reconcile_with_alerax(species_tree, wrong_gene_tree)
ValueError: Input validation failed: Gene tree 'family_0' has leaves not in
species tree: ['unknown_species_1', 'unknown_species_2']
```

## Testing with Provided Data

The `testdata/test_data_3/` directory contains example data:

```python
import rustree

# Load data
species_tree = rustree.parse_species_tree("testdata/test_data_3/output_zombi/T/ExtantTree.nwk")
gene_tree_1 = "testdata/test_data_3/output_zombi/G/Gene_trees/1_prunedtree.nwk"
gene_tree_2 = "testdata/test_data_3/output_zombi/G/Gene_trees/2_prunedtree.nwk"

# Reconcile both families
results = rustree.reconcile_with_alerax(
    species_tree,
    {"family_1": gene_tree_1, "family_2": gene_tree_2},
    seed=42
)

# Analyze results
for name, result in results.items():
    print(f"\n{name}:")
    best = result.gene_trees[0]
    events = best.count_events()
    print(f"  Events: S={events[0]}, D={events[1]}, T={events[2]}, L={events[3]}")
    print(f"  Rates: D={result.duplication_rate:.4f}, T={result.transfer_rate:.4f}")
```

## Integration with Existing Functionality

All reconciled trees are `PyGeneTree` objects with full API access:

```python
best_tree = result.gene_trees[0]

# Sampling operations
extant_only = best_tree.sample_extant()
species_subset = best_tree.sample_species_leaves(["n5", "n11", "n12"])
by_names = best_tree.sample_by_names(["n5_20", "n11_52"])

# Export formats
best_tree.to_newick()                    # Newick string
best_tree.save_newick("tree.nwk")        # Save to file
best_tree.to_xml()                       # RecPhyloXML string
best_tree.save_xml("tree.xml")           # Save XML
best_tree.to_svg("tree.svg")             # Generate SVG (requires thirdkind)
df = best_tree.to_csv()                  # pandas DataFrame

# Analysis
distances = best_tree.pairwise_distances("metric", leaves_only=True)
num_nodes = best_tree.num_nodes()
num_extant = best_tree.num_extant()
gene_names = best_tree.extant_gene_names()

# Jupyter notebook display
best_tree.display()  # Shows SVG inline
```

## Implementation Files

The ALERax integration consists of:
- **`src/alerax.rs`**: Core Rust module (~370 lines)
- **`src/python.rs`**: Python bindings additions (~220 lines)
- **`src/lib.rs`**: Module registration
- **`Cargo.toml`**: Added `tempfile = "3.8"` dependency

All changes follow existing rustree patterns and reuse the RecPhyloXML parsing infrastructure.

## Next Steps

To build and test:
```bash
# Build Python module
maturin develop --release

# Run Python tests
python -c "
import rustree
species_tree = rustree.parse_species_tree('testdata/test_data_3/output_zombi/T/ExtantTree.nwk')
results = rustree.reconcile_with_alerax(species_tree, 'testdata/test_data_3/output_zombi/G/Gene_trees/1_prunedtree.nwk', seed=42)
print('Success! Reconciled with', len(results), 'families')
for name, result in results.items():
    print(f'{name}: {len(result.gene_trees)} samples, D={result.duplication_rate:.4f}')
"
```
