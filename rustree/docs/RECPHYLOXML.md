# RecPhyloXML Format Support in rustree

Comprehensive guide to parsing and exporting RecPhyloXML reconciliation files for gene-species tree relationships.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Format Support](#format-support)
5. [Usage Examples](#usage-examples)
   - [Rust API](#rust-api)
   - [Python API](#python-api)
   - [R API](#r-api)
6. [Data Structures](#data-structures)
7. [File Formats](#file-formats)
8. [Error Handling](#error-handling)
9. [Validation](#validation)
10. [Troubleshooting](#troubleshooting)
11. [See Also](#see-also)

---

## Overview

RecPhyloXML is the standard XML format for representing reconciled gene-species tree relationships. rustree provides full support for:

- **Parsing** RecPhyloXML files from ALERax and other tools
- **Exporting** simulated gene trees to RecPhyloXML
- **Round-trip** conversion: parse → modify → export

**Supported features:**
- ✅ Complete RecPhyloXML with both `<spTree>` and `<recGeneTree>` sections
- ✅ Gene-tree-only XML (no `<spTree>`) with separate Newick species tree
- ✅ All event types: Speciation, Duplication, Transfer, Loss, Leaf
- ✅ Branch lengths (optional, defaults to 0.0)
- ✅ Transfer events with `<branchingOut>` and `<transferBack>` tags
- ✅ Node names and species mappings
- ✅ Alternative event tags (`<leaf>`, `<P>`, `<T>`)

---

## Prerequisites

### Required
- **rustree** library installed (Python or R bindings, or Rust crate)

### Optional
- **ALERax** for reconciliation inference
- **thirdkind** for visualization (if exporting SVG)

### File Format Knowledge
- Basic understanding of XML structure
- Familiarity with phylogenetic tree formats (Newick, XML)

---

## Quick Start

### Python

```python
import rustree

# Parse RecPhyloXML file (contains both species and gene trees)
gene_tree = rustree.parse_recphyloxml("/tmp/alerax_output.xml")

# Access tree information
print(f"Gene tree nodes: {gene_tree.num_nodes()}")
print(f"Extant genes: {gene_tree.num_extant()}")

# Count events
s, d, t, l, leaves = gene_tree.count_events()
print(f"Duplications: {d}, Transfers: {t}, Losses: {l}")

# Export back to XML
gene_tree.save_xml("/tmp/output.xml")

# Export to CSV for analysis
gene_tree.to_csv("/tmp/events.csv")
```

### R

```r
# Note: R support for parsing RecPhyloXML is via manual file reading
# Export is fully supported

dyn.load("target/release/librustree.so")
source("R/rustree.R")

# Simulate and export
sp_tree <- simulate_species_tree(20L, 1.0, 0.5, seed = 42L)
gt <- simulate_dtl(sp_tree, 0.5, 0.2, 0.3, seed = 123L)

# Export to RecPhyloXML
save_xml(gt, "/tmp/gene_tree.xml")

# Export to CSV
save_csv(gt, "/tmp/gene_tree.csv")
```

### Rust

```rust
use rustree::RecTreeOwned;

// Parse from file (contains both species and gene trees)
let rec_tree = RecTreeOwned::from_xml_file("alerax_output.xml")?;

// Access the trees
println!("Species nodes: {}", rec_tree.species_tree.nodes.len());
println!("Gene nodes: {}", rec_tree.gene_tree.nodes.len());

// Query event information
let event = rec_tree.event_for(gene_node_idx);
let species_idx = rec_tree.species_node_for(gene_node_idx);

// Export back to XML
let xml = rec_tree.to_xml();
std::fs::write("output.xml", xml)?;
```

---

## Format Support

### Format 1: Complete RecPhyloXML

Contains both species tree and gene tree in a single XML file.

**Structure:**
```xml
<?xml version="1.0"?>
<recPhylo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <spTree>
    <phylogeny>
      <clade>
        <name>S1</name>
        <clade>...</clade>
        <clade>...</clade>
      </clade>
    </phylogeny>
  </spTree>
  <recGeneTree>
    <phylogeny>
      <clade>
        <name>g0</name>
        <eventsRec>
          <speciation speciesLocation="S1"/>
        </eventsRec>
        <clade>...</clade>
        <clade>...</clade>
      </clade>
    </phylogeny>
  </recGeneTree>
</recPhylo>
```

**Rust API:**
```rust
// Parse from string
let rec_tree = RecTreeOwned::from_xml(xml_content)?;

// Parse from file
let rec_tree = RecTreeOwned::from_xml_file("file.xml")?;
```

**Python API:**
```python
gene_tree = rustree.parse_recphyloxml("file.xml")
```

### Format 2: Separate Files

Gene tree XML without `<spTree>` section, species tree in separate Newick file.

**Gene tree XML:**
```xml
<?xml version="1.0"?>
<recPhylo>
  <recGeneTree>
    <phylogeny>
      <clade>
        <name>g0</name>
        <eventsRec>
          <speciation speciesLocation="S1"/>
        </eventsRec>
        ...
      </clade>
    </phylogeny>
  </recGeneTree>
</recPhylo>
```

**Species tree (Newick):**
```
((S1:1.0,S2:1.0):1.0,S3:2.0):0;
```

**Rust API:**
```rust
// Parse from separate files
let rec_tree = RecTreeOwned::from_separate_files(
    "species_tree.nwk",
    "gene_tree.xml"
)?;

// Or with pre-loaded species tree
let species_tree = parse_newick_to_flattree(&newick_str)?;
let rec_tree = RecTreeOwned::from_gene_tree_xml_file(
    "gene_tree.xml",
    species_tree
)?;
```

---

## Usage Examples

### Rust API

#### Parse Complete RecPhyloXML

```rust
use rustree::{RecTreeOwned, Event};

fn main() -> Result<(), String> {
    // Parse ALERax RecPhyloXML file
    let rec_tree = RecTreeOwned::from_xml_file("alerax_g.xml")?;

    // Count event types
    let duplication_count = rec_tree.event_mapping.iter()
        .filter(|e| **e == Event::Duplication)
        .count();

    println!("Duplications: {}", duplication_count);

    // Find all extant genes
    for (idx, node) in rec_tree.gene_tree.nodes.iter().enumerate() {
        if rec_tree.event_mapping[idx] == Event::Leaf {
            let species_idx = rec_tree.node_mapping[idx];
            let species_name = &rec_tree.species_tree.nodes[species_idx].name;
            println!("Gene {} is in species {}", node.name, species_name);
        }
    }

    // Export to XML
    let xml = rec_tree.to_xml();
    std::fs::write("output.xml", xml)?;

    Ok(())
}
```

#### Parse Separate Files

```rust
use rustree::RecTreeOwned;

fn main() -> Result<(), String> {
    // Parse species tree and gene tree from separate files
    let rec_tree = RecTreeOwned::from_separate_files(
        "species.nwk",
        "gene_reconciled.xml"
    )?;

    println!("Species tree: {} nodes", rec_tree.species_tree.nodes.len());
    println!("Gene tree: {} nodes", rec_tree.gene_tree.nodes.len());

    Ok(())
}
```

### Python API

#### Parse and Analyze

```python
import rustree
import pandas as pd

# Parse RecPhyloXML file
gt = rustree.parse_recphyloxml("/tmp/alerax_family_01.xml")

# Basic statistics
print(f"Total nodes: {gt.num_nodes()}")
print(f"Extant genes: {gt.num_extant()}")

# Event counts
spec, dup, trans, loss, leaves = gt.count_events()
print(f"\nEvent counts:")
print(f"  Speciations: {spec}")
print(f"  Duplications: {dup}")
print(f"  Transfers: {trans}")
print(f"  Losses: {loss}")

# Export to DataFrame for detailed analysis
df = gt.to_csv()
print(f"\nDataFrame columns: {df.columns.tolist()}")

# Analyze events by species
event_by_species = df.groupby('species_node')['event'].value_counts()
print("\nEvents by species:")
print(event_by_species)

# Save results
gt.save_xml("/tmp/output.xml")
gt.to_csv("/tmp/events.csv")
```

#### Round-Trip Conversion

```python
import rustree

# Simulate tree
sp_tree = rustree.simulate_species_tree(20, 1.0, 0.5, seed=42)
gt = sp_tree.simulate_dtl(0.5, 0.2, 0.3, seed=123)

# Export to RecPhyloXML
gt.save_xml("/tmp/simulated.xml")

# Parse it back
gt_loaded = rustree.parse_recphyloxml("/tmp/simulated.xml")

# Verify round-trip
s1, d1, t1, l1, _ = gt.count_events()
s2, d2, t2, l2, _ = gt_loaded.count_events()

assert (s1, d1, t1, l1) == (s2, d2, t2, l2), "Round-trip failed!"
print("Round-trip successful!")
```

### R API

```r
dyn.load("target/release/librustree.so")
source("R/rustree.R")

# Simulate trees
sp_tree <- simulate_species_tree(15L, 1.0, 0.4, seed = 42L)
gt <- simulate_dtl(sp_tree, 0.5, 0.2, 0.3, seed = 123L)

# Export to RecPhyloXML
save_xml(gt, "/tmp/gene_tree.xml")

# Export to CSV for analysis
save_csv(gt, "/tmp/gene_tree.csv")

# Read and analyze CSV
df <- read.csv("/tmp/gene_tree.csv")

# Count events
print("Event counts:")
print(table(df$event))

# Events by species
print("\nEvents by species:")
event_by_species <- table(df$species_node, df$event)
print(event_by_species)
```

---

## Data Structures

### RecTreeOwned

The main Rust structure for reconciled trees:

```rust
pub struct RecTreeOwned {
    pub species_tree: FlatTree,           // Species tree
    pub gene_tree: FlatTree,              // Gene tree
    pub node_mapping: Vec<usize>,         // Maps gene node idx → species node idx
    pub event_mapping: Vec<Event>,        // Maps gene node idx → Event type
}
```

### Event Types

```rust
pub enum Event {
    Speciation,   // Gene tree follows species tree split
    Duplication,  // Gene duplicated within a species
    Transfer,     // Horizontal gene transfer (donor side)
    Loss,         // Gene loss event
    Leaf,         // Extant gene (leaf node)
}
```

### FlatTree

Arena-allocated tree structure:

```rust
pub struct FlatTree {
    pub nodes: Vec<FlatNode>,    // All nodes in the tree
    pub root: usize,              // Root node index
}

pub struct FlatNode {
    pub name: String,
    pub parent: Option<usize>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub branch_length: f64,
    pub depth: f64,
    // ... other fields
}
```

---

## File Formats

### RecPhyloXML Event Tags

The parser supports multiple event tag formats:

**Standard events:**
```xml
<eventsRec>
  <speciation speciesLocation="S1"/>
</eventsRec>

<eventsRec>
  <duplication speciesLocation="S2"/>
</eventsRec>

<eventsRec>
  <branchingOut speciesLocation="S3"/>  <!-- Transfer donor -->
  <transferBack speciesLocation="S4"/>  <!-- Transfer recipient -->
</eventsRec>

<eventsRec>
  <loss speciesLocation="S5"/>
</eventsRec>

<eventsRec>
  <leaf speciesLocation="S6"/>
</eventsRec>
```

**Alternative tags:**
```xml
<eventsRec>
  <P speciesLocation="S1"/>  <!-- Present-day gene (treated as leaf) -->
</eventsRec>

<eventsRec>
  <T speciesLocation="S2"/>  <!-- Generic internal (treated as speciation) -->
</eventsRec>
```

### Branch Lengths

Branch lengths are optional. If missing, they default to 0.0:

```xml
<clade>
  <name>g0</name>
  <branchLength>1.5</branchLength>  <!-- Optional -->
  <eventsRec>...</eventsRec>
  <clade>...</clade>
</clade>
```

### Node Names

Node names can include internal identifiers:

```xml
<clade>
  <name>g0</name>  <!-- Gene node name -->
  <eventsRec>
    <speciation speciesLocation="S1"/>  <!-- Species node name -->
  </eventsRec>
</clade>
```

**Note on ambiguous names:** If multiple species nodes have the same name, the parser uses the first occurrence in tree traversal.

---

## Error Handling

### Rust

```rust
use rustree::RecTreeOwned;

match RecTreeOwned::from_xml_file("file.xml") {
    Ok(rec_tree) => {
        println!("Loaded successfully: {} genes", rec_tree.gene_tree.nodes.len());
    }
    Err(e) => {
        eprintln!("Failed to parse: {}", e);
        // Handle error
    }
}
```

### Python

```python
import rustree

try:
    gt = rustree.parse_recphyloxml("/tmp/file.xml")
    print(f"Loaded successfully: {gt.num_nodes()} nodes")
except ValueError as e:
    print(f"Parse error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Missing <spTree> or <recGeneTree>` | Incomplete XML structure | Verify XML format or use separate file parsing |
| `Species referenced in gene tree not found` | Species name mismatch | Check `speciesLocation` attributes match species tree names |
| `Invalid XML format` | Malformed XML | Validate XML with `xmllint` or similar tool |
| `I/O error: permission denied` | File permissions | Check file read permissions |
| `I/O error: file not found` | Wrong path | Verify file path is correct |

---

## Validation

### Validate XML Structure

Before parsing with rustree, validate XML syntax:

```bash
# Check XML well-formedness
xmllint --noout /tmp/file.xml

# Validate against schema (if available)
xmllint --schema recphylo.xsd /tmp/file.xml
```

### Verify Round-Trip

Test that export/import preserves data:

```python
import rustree

# Simulate tree
sp_tree = rustree.simulate_species_tree(10, 1.0, 0.5, seed=42)
gt_original = sp_tree.simulate_dtl(0.5, 0.2, 0.3, seed=123)

# Export
gt_original.save_xml("/tmp/test.xml")

# Re-import
gt_loaded = rustree.parse_recphyloxml("/tmp/test.xml")

# Compare event counts
orig_counts = gt_original.count_events()
loaded_counts = gt_loaded.count_events()

if orig_counts == loaded_counts:
    print("✓ Round-trip successful")
else:
    print("✗ Round-trip failed")
    print(f"  Original: {orig_counts}")
    print(f"  Loaded: {loaded_counts}")
```

### Check Species Mapping

Verify all gene nodes are mapped to valid species:

```python
import rustree
import pandas as pd

gt = rustree.parse_recphyloxml("/tmp/file.xml")
df = gt.to_csv()

# Check for unmapped nodes (should be rare)
unmapped = df[df['species_node'].isna()]
if len(unmapped) > 0:
    print(f"Warning: {len(unmapped)} unmapped nodes")
    print(unmapped[['name', 'event']])
else:
    print("✓ All nodes properly mapped")
```

---

## Troubleshooting

### Issue: Parse fails with "species not found"

**Symptoms:**
```
Error: Species referenced in gene tree not found in species tree: S5
```

**Causes:**
1. Species name mismatch between trees
2. Extra whitespace in names
3. Case sensitivity

**Solutions:**
```python
# Check species tree names
sp_names = set(sp_tree.leaf_names())
print(f"Species in tree: {sp_names}")

# Check what gene tree references
df = gt.to_csv()
gene_species = set(df['species_node'].dropna())
print(f"Species referenced by genes: {gene_species}")

# Find mismatches
missing = gene_species - sp_names
if missing:
    print(f"Missing species: {missing}")
```

### Issue: Missing branch lengths

**Symptoms:** All branch lengths are 0.0

**Cause:** XML file doesn't include `<branchLength>` tags

**Solution:** This is expected behavior. Branch lengths default to 0.0 if not provided.

```python
# Check if branch lengths are present
df = gt.to_csv()
has_lengths = (df['length'] != 0.0).any()
if not has_lengths:
    print("Note: No branch lengths in XML (all default to 0.0)")
```

### Issue: Large XML files are slow to parse

**Symptoms:** Parsing takes >10 seconds

**Cause:** Very large gene families (>10,000 nodes)

**Solutions:**
1. Use streaming parser (already implemented in rustree)
2. Split large families into separate files
3. Consider memory-mapped file parsing (future feature)

### Issue: Transfer events not recognized

**Symptoms:** Transfer count is 0 when transfers are expected

**Cause:** Non-standard transfer tags

**Check:**
```bash
# Look for transfer tags in XML
grep -i transfer /tmp/file.xml
grep -i branchingOut /tmp/file.xml
```

**Solution:** rustree recognizes `<branchingOut>` + `<transferBack>` tags. Ensure XML uses this format.

---

## See Also

- **[Python Tutorial](PYTHON_TUTORIAL.md)** - Python API documentation
- **[R Tutorial](R_TUTORIAL.md)** - R bindings documentation
- **[RecPhyloXML Specification](http://www.recg.org/)** - Official format specification
- **[ALERax Documentation](https://github.com/BenoitMorel/AleRax)** - Reconciliation software
- **rustree GitHub repository** - Source code and examples

---

**Document Version:** 2.0
**Last Updated:** 2026-02-14
**rustree Version:** 0.1.0
**Project Root:** repository root
