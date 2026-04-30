# rustree Python Tutorial

Comprehensive guide to using the rustree Python bindings for phylogenetic tree simulation, reconciliation, and analysis.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
   - [Species Trees](#species-trees)
   - [Gene Trees and Reconciliation](#gene-trees-and-reconciliation)
   - [DTL Events](#dtl-events)
5. [Simulation](#simulation)
   - [Birth-Death Species Trees](#birth-death-species-trees)
   - [Parsing Newick Trees](#parsing-newick-trees)
   - [DTL Gene Tree Simulation](#dtl-gene-tree-simulation)
   - [Per-Gene-Copy vs Per-Species Models](#per-gene-copy-vs-per-species-models)
   - [Assortative Transfers](#assortative-transfers)
6. [Reconciliation](#reconciliation)
   - [Parsing RecPhyloXML](#parsing-recphyloxml)
   - [Event Counting](#event-counting)
   - [Species Mapping](#species-mapping)
7. [Analysis](#analysis)
   - [Birth-Death Events](#birth-death-events)
   - [Pairwise Distances](#pairwise-distances)
   - [LTT Plots](#ltt-plots)
   - [Subtree Extraction](#subtree-extraction)
   - [Induced Transfers](#induced-transfers)
   - [Ghost Branch Lengths](#ghost-branch-lengths)
8. [Export and Visualization](#export-and-visualization)
   - [Newick Format](#newick-format)
   - [RecPhyloXML Format](#recphyloxml-format)
   - [CSV Export](#csv-export)
   - [SVG Visualization](#svg-visualization)
9. [Advanced Usage](#advanced-usage)
   - [Batch Processing](#batch-processing)
   - [Ensuring Extant Genes](#ensuring-extant-genes)
   - [Error Handling](#error-handling)
10. [Complete Workflows](#complete-workflows)
11. [API Reference](#api-reference)
12. [Troubleshooting](#troubleshooting)
13. [See Also](#see-also)

---

## Prerequisites

### Required
- **Python** 3.8 or higher
- **Rust** 1.70 or higher (for building from source)
- **maturin** for building Python wheels:
  ```bash
  pip install maturin
  ```

### Optional
- **matplotlib** for LTT plots:
  ```bash
  pip install matplotlib
  ```
- **pandas** for CSV analysis (automatically installed with rustree)
- **thirdkind** for SVG visualization:
  ```bash
  cargo install thirdkind
  ```

---

## Installation

### From Source

```bash
# Clone the repository
cd /path/to/rustree

# Build and install in development mode (editable)
maturin develop --release

# Or build a wheel for distribution
maturin build --release
pip install target/wheels/rustree-*.whl
```

### Verify Installation

```python
import rustree
print(rustree.__version__)  # Should print version number
```

---

## Quick Start

```python
import rustree

# Simulate a species tree with 50 extant species
sp_tree = rustree.simulate_species_tree(n=50, lambda_=1.0, mu=0.3, seed=42)
print(f"Species: {sp_tree.num_leaves()}")

# Simulate 10 gene families with DTL events
gene_trees = sp_tree.simulate_dtl_batch(
    n=10,
    lambda_d=0.5,  # Duplication rate
    lambda_t=0.2,  # Transfer rate
    lambda_l=0.3,  # Loss rate
    seed=123
)

# Analyze the first gene family
gt = gene_trees[0]
print(f"Extant genes: {gt.num_extant()}")
spec, dup, trans, loss, leaves = gt.count_events()
print(f"Events: D={dup}, T={trans}, L={loss}")

# Export results
sp_tree.save_newick("/tmp/species.nwk")
gt.save_xml("/tmp/gene_family_001.xml")
```

---

## Core Concepts

### Species Trees

Species trees represent the evolutionary history of species through speciation and extinction events. In rustree, species trees are simulated using a **birth-death process** or parsed from Newick format.

**Key operations:**
- `num_leaves()` - Count extant species
- `num_nodes()` - Count all nodes (including extinct lineages)
- `tree_height()` - Time from present to root (MRCA)
- `leaf_names()` - List of species names

### Gene Trees and Reconciliation

Gene trees evolve along species trees but can experience additional events:
- **Duplication (D)**: Gene copies within the same species
- **Transfer (T)**: Horizontal gene transfer between contemporary species
- **Loss (L)**: Gene lineage goes extinct
- **Speciation (S)**: Gene follows the species tree split

Each gene tree node is **reconciled** (mapped) to a species tree node, recording which species hosted that gene.

### DTL Events

DTL events occur stochastically along species tree branches. Two simulation models are available:

1. **Per-gene-copy model** (standard): Event rate ∝ number of gene copies
2. **Per-species model** (Zombi-style): Event rate ∝ number of alive species

See [DTL Gene Tree Simulation](#dtl-gene-tree-simulation) for details.

---

## Simulation

### Birth-Death Species Trees

Simulate species trees with specified speciation (λ) and extinction (μ) rates:

```python
import rustree

# Simulate tree with 100 extant species
sp_tree = rustree.simulate_species_tree(
    n=100,           # Number of extant species
    lambda_=1.0,     # Speciation (birth) rate
    mu=0.5,          # Extinction (death) rate (must be < lambda)
    seed=42          # Random seed for reproducibility (optional)
)

# Inspect the tree
print(f"Extant species: {sp_tree.num_leaves()}")
print(f"Total nodes: {sp_tree.num_nodes()}")
print(f"Tree height: {sp_tree.tree_height():.2f}")
print(f"Species names: {sp_tree.leaf_names()[:5]}...")  # First 5 names
```

**Note on extinction rate:** The condition μ < λ ensures net diversification. If μ ≥ λ, the lineage would eventually go extinct.

### Parsing Newick Trees

Parse existing Newick-format trees:

```python
# Parse from string
newick_str = "((A:1.0,B:1.0):1.0,(C:0.5,D:0.5):1.5):0;"
sp_tree = rustree.parse_species_tree(newick_str)

# Or read from file
with open("/tmp/species_tree.nwk", "r") as f:
    sp_tree = rustree.parse_species_tree(f.read())

print(f"Parsed {sp_tree.num_leaves()} species")
```

### DTL Gene Tree Simulation

#### Single Gene Tree

```python
# Simulate one gene family
gene_tree = sp_tree.simulate_dtl(
    lambda_d=0.5,    # Duplication rate
    lambda_t=0.2,    # Transfer rate
    lambda_l=0.3,    # Loss rate
    seed=123         # Optional seed
)

# Inspect results
print(f"Extant genes: {gene_tree.num_extant()}")
s, d, t, l, leaves = gene_tree.count_events()
print(f"Events: S={s}, D={d}, T={t}, L={l}, Leaves={leaves}")
```

#### Batch Simulation (Recommended)

For multiple gene families, use `simulate_dtl_batch()` which is much more efficient:

```python
# Simulate 100 gene families
gene_trees = sp_tree.simulate_dtl_batch(
    n=100,           # Number of gene families
    lambda_d=0.5,
    lambda_t=0.2,
    lambda_l=0.3,
    seed=123
)

# Analyze all families
extant_counts = [gt.num_extant() for gt in gene_trees]
print(f"Mean extant genes: {sum(extant_counts)/len(extant_counts):.1f}")
print(f"Range: {min(extant_counts)} - {max(extant_counts)}")
```

**Performance note:** Batch simulation pre-computes shared data structures, providing significant speedup (~2-3x) compared to looping `simulate_dtl()`.

### Per-Gene-Copy vs Per-Species Models

#### Per-Gene-Copy Model (Standard)

The standard DTL model uses event rates proportional to the number of **gene copies** alive at any time. More gene copies → higher total event rate.

```python
# Standard model: event rate scales with gene copy count
gene_tree = sp_tree.simulate_dtl(
    lambda_d=0.5,  # Rate per gene copy per unit time
    lambda_t=0.2,
    lambda_l=0.3,
    seed=123
)
```

**Characteristics:**
- Duplications increase the total event rate (more copies → more events)
- Can produce large gene families through runaway duplication
- Total event rate at time t: `(λ_d + λ_t + λ_l) × n_gene_copies(t)`

#### Per-Species Model (Zombi-Style)

The per-species model uses event rates proportional to the number of **alive species**, regardless of gene copy count. When an event fires, a random species is chosen; if it has no gene copies, the event fails silently.

```python
# Per-species model: event rate scales with species count
gene_tree = sp_tree.simulate_dtl_per_species(
    lambda_d=0.5,  # Rate per species per unit time
    lambda_t=0.2,
    lambda_l=0.3,
    seed=123
)

# Batch version (more efficient)
gene_trees = sp_tree.simulate_dtl_per_species_batch(
    n=100,
    lambda_d=0.5,
    lambda_t=0.2,
    lambda_l=0.3,
    seed=123
)
```

**Characteristics:**
- Duplications do NOT increase the total event rate
- Produces smaller gene families on average
- Total event rate at time t: `(λ_d + λ_t + λ_l) × n_alive_species(t)`

#### Comparing Models

```python
import rustree

sp_tree = rustree.simulate_species_tree(50, 1.0, 0.3, seed=42)

# Per-copy model
gt_copy = sp_tree.simulate_dtl(0.5, 0.2, 0.3, seed=1)

# Per-species model
gt_species = sp_tree.simulate_dtl_per_species(0.5, 0.2, 0.3, seed=1)

print("Per-copy model:")
s, d, t, l, _ = gt_copy.count_events()
print(f"  D={d}, T={t}, L={l}, extant={gt_copy.num_extant()}")

print("Per-species model:")
s, d, t, l, _ = gt_species.count_events()
print(f"  D={d}, T={t}, L={l}, extant={gt_species.num_extant()}")

# Per-species model typically shows fewer events and smaller families
```

### Assortative Transfers

By default, transfer recipients are chosen uniformly at random from contemporary species. **Assortative transfers** model distance-dependent transfer where closer species are more likely to exchange genes.

```python
# Distance-dependent transfers
# P(recipient) ∝ exp(-alpha × distance)
gene_tree = sp_tree.simulate_dtl(
    lambda_d=0.5,
    lambda_t=0.5,
    lambda_l=0.3,
    transfer_alpha=1.0,  # Distance decay parameter
    seed=123
)

# Batch with assortative transfers
gene_trees = sp_tree.simulate_dtl_batch(
    n=100,
    lambda_d=0.5,
    lambda_t=0.5,
    lambda_l=0.3,
    transfer_alpha=2.0,  # Stronger local preference
    seed=123
)
```

**Distance calculation:**
```
distance(A, B, t) = 2 × (t - depth_of_LCA(A, B))
```

**Parameter guidelines:**
- `transfer_alpha=None` or `0`: Uniform random (no distance preference)
- `transfer_alpha=1.0`: Moderate local preference
- `transfer_alpha=2.0`: Strong local preference
- `transfer_alpha=5.0`: Very strong local preference (rare long-distance transfers)

**Works with both models:**
```python
# Assortative transfers with per-species model
gene_tree = sp_tree.simulate_dtl_per_species(
    lambda_d=0.5,
    lambda_t=0.5,
    lambda_l=0.3,
    transfer_alpha=1.5,
    seed=123
)
```

---

## Reconciliation

### Parsing RecPhyloXML

Import reconciled gene trees from ALERax or other tools:

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
```

See [RECPHYLOXML.md](RECPHYLOXML.md) for detailed documentation.

### Event Counting

```python
# Get event counts
spec, dup, trans, loss, leaves = gene_tree.count_events()

print(f"Speciation events: {spec}")
print(f"Duplication events: {dup}")
print(f"Transfer events: {trans}")
print(f"Loss events: {loss}")
print(f"Total leaves: {leaves}")
```

### Species Mapping

Gene tree nodes are mapped to species tree nodes. Access this mapping via CSV export:

```python
# Export to DataFrame
df = gene_tree.to_csv()

# View reconciliation mapping
print(df[['name', 'species_node', 'event']].head(10))
```

---

## Analysis

### Birth-Death Events

Species trees simulated with `simulate_species_tree()` include birth-death event data:

```python
# Simulate species tree
sp_tree = rustree.simulate_species_tree(50, 1.0, 0.5, seed=42)

# Access birth-death events as dictionary
events = sp_tree.get_bd_events()
print(f"Event count: {len(events['times'])}")

# Save to CSV
sp_tree.save_bd_events_csv("/tmp/bd_events.csv")

# Analyze with pandas
import pandas as pd
df = pd.read_csv("/tmp/bd_events.csv")
print(df['event_type'].value_counts())
```

**CSV columns:**
- `time`: When the event occurred (backwards from present at 0)
- `node_name`: Node where event occurred
- `event_type`: "Speciation", "Extinction", or "Leaf"
- `child1_name`, `child2_name`: Child nodes (for speciation events)

### Pairwise Distances

Calculate distances between nodes for phylogenetic analysis:

```python
# Metric distance (sum of branch lengths)
metric_dists = sp_tree.pairwise_distances(
    distance_type="metric",
    leaves_only=True
)

# Returns list of PairwiseDistance objects
for dist in metric_dists[:5]:
    print(f"{dist.node1} <-> {dist.node2}: {dist.distance:.4f}")

# Topological distance (edge count)
topo_dists = sp_tree.pairwise_distances(
    distance_type="topological",
    leaves_only=True
)

# Save to CSV
sp_tree.save_pairwise_distances_csv(
    "/tmp/distances.csv",
    distance_type="metric",
    leaves_only=True
)
```

**Distance types:**
- `"metric"`, `"patristic"`, or `"branch"`: Sum of branch lengths (float)
- `"topological"` or `"topo"`: Number of edges (int)

**Scope:**
- `leaves_only=True`: Only leaf-to-leaf distances (extant species/genes)
- `leaves_only=False`: All pairwise distances including internal nodes

### LTT Plots

Lineages-through-time plots show how lineage count changes over evolutionary history:

```python
import rustree
import matplotlib.pyplot as plt

# Simulate tree
sp_tree = rustree.simulate_species_tree(100, 1.0, 0.5, seed=42)

# Option 1: Get data and plot manually
ltt = sp_tree.get_ltt_data()
plt.plot(ltt['times'], ltt['lineages'])
plt.xlabel('Time (past to present)')
plt.ylabel('Number of lineages')
plt.savefig('/tmp/ltt_manual.png')

# Option 2: Use built-in plotting
sp_tree.plot_ltt(filepath='/tmp/ltt_builtin.png')

# Option 3: Customize
sp_tree.plot_ltt(
    filepath='/tmp/ltt_custom.png',
    title='Birth-Death Tree (λ=1.0, μ=0.5)',
    xlabel='Time before present',
    ylabel='Lineage count'
)
```

See [LTT_PLOTS.md](LTT_PLOTS.md) for advanced examples.

### Subtree Extraction

Extract induced subtrees containing only specified taxa:

```python
# Get all species
all_species = sp_tree.leaf_names()

# Select subset
selected_species = all_species[:10]

# Extract induced subtree
subtree = sp_tree.extract_induced_subtree_by_names(selected_species)
print(f"Original: {sp_tree.num_leaves()} leaves")
print(f"Subtree: {subtree.num_leaves()} leaves")

# Also works with gene trees
gene_tree = sp_tree.simulate_dtl(0.5, 0.2, 0.3, seed=123)
gene_names = gene_tree.extant_gene_names()
sampled_genes = gene_names[:20]
sampled_gt = gene_tree.sample_by_names(sampled_genes)

# Or sample by species (filters gene tree to genes on selected species)
species_subset = sp_tree.leaf_names()[:5]
sampled_gt2 = gene_tree.sample_species_leaves(species_subset)
print(f"Genes after species sampling: {sampled_gt2.num_extant()}")
```

### Extracting Extant Species

When simulating species trees with extinction (`mu > 0`), the tree includes both extant and extinct lineages as leaves. To get only the extant species:

```python
import rustree

# Simulate tree with 20 extant species and extinction
tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.6, seed=42)
print(f"Original tree: {tree.num_leaves()} leaves")  # May be > 20

# Extract only extant species
extant_only = tree.sample_extant()
print(f"Extant only: {extant_only.num_leaves()} leaves")  # Exactly 20

# Verify using BD events
events = tree.get_bd_events()
n_extant = sum(1 for et in events['event_type'] if et == 'Leaf')
n_extinct = sum(1 for et in events['event_type'] if et == 'Extinction')
print(f"Extant: {n_extant}, Extinct: {n_extinct}")

# Use extant-only tree for downstream analysis
extant_only.save_newick("extant_species.nwk")
gene_trees = extant_only.simulate_dtl_batch(
    n=100, lambda_d=0.5, lambda_t=0.2, lambda_l=0.3
)
```

**Note:** `sample_extant()` only works on trees from `simulate_species_tree()`. Trees parsed from Newick files lack the `bd_event` annotations needed to distinguish extant from extinct species.

### Induced Transfers

When a gene tree is simulated on a complete species tree but analyzed with respect to a sampled (pruned) species tree, transfers can be projected onto the nearest branches in the sampled tree:

```python
import rustree

# Simulate a large species tree and gene tree with transfers
sp_tree = rustree.simulate_species_tree(n=100, lambda_=1.0, mu=0.5, seed=42)
gene_tree = sp_tree.simulate_dtl(
    lambda_d=0.5,
    lambda_t=0.2,  # Transfer rate
    lambda_l=0.3,
    seed=123
)

# Count transfers in complete tree
spec, dup, trans, loss, leaves = gene_tree.count_events()
print(f"Original: {trans} transfers")

# Sample a subset of species (e.g., 20 species)
all_leaf_names = sp_tree.leaf_names()
sampled_leaf_names = all_leaf_names[:20]
sampled_sp_tree = sp_tree.extract_induced_subtree_by_names(sampled_leaf_names)

# Compute induced transfers (project onto sampled tree)
induced = gene_tree.compute_induced_transfers(sampled_sp_tree, sampled_leaf_names)
print(f"Induced transfers: {len(induced)}")

# Analyze induced transfers
for transfer in induced[:5]:  # First 5
    print(f"Time {transfer.time:.3f}: "
          f"from {transfer.from_species_complete} → {transfer.to_species_complete} (complete), "
          f"from {transfer.from_species_sampled} → {transfer.to_species_sampled} (sampled)")

# Count successful projections
successful = sum(1 for t in induced
                 if t.from_species_sampled is not None
                 and t.to_species_sampled is not None)
print(f"{successful}/{len(induced)} transfers projected successfully")
```

**Induced Transfer Attributes:**
- `time`: Time of transfer event
- `gene_id`: Gene tree node index
- `from_species_complete`: Donor species index in complete tree
- `to_species_complete`: Recipient species index in complete tree
- `from_species_sampled`: Donor index in sampled tree (None if unprojectable)
- `to_species_sampled`: Recipient index in sampled tree (None if unprojectable)

### Ghost Branch Lengths

Ghost lengths represent the hidden evolutionary time from non-sampled branches:

```python
# Compute ghost lengths for sampled tree
ghost_lengths = gene_tree.compute_ghost_lengths(sampled_sp_tree, sampled_leaf_names)

# ghost_lengths[i] = total length of complete-tree branches projecting onto sampled node i
print(f"Total ghost length: {sum(ghost_lengths):.2f}")
print(f"Max ghost length: {max(ghost_lengths):.2f}")

# Identify nodes with significant ghost lineages
significant = [(i, gl) for i, gl in enumerate(ghost_lengths) if gl > 2.0]
for node_idx, ghost_len in significant:
    print(f"Node {node_idx}: {ghost_len:.2f} time units of ghost lineages")
```

**Use cases:**
- Correcting transfer rate estimates when analyzing sampled trees
- Understanding the impact of incomplete taxon sampling
- Detecting regions of the tree with high unsampled diversity

**Note:** Induced transfers and ghost lengths are only available for gene trees generated by simulation (not parsed from files).

---

## Export and Visualization

### Newick Format

```python
# Get as string
newick_str = sp_tree.to_newick()
print(newick_str[:100])  # First 100 characters

# Save to file
sp_tree.save_newick("/tmp/species.nwk")
gene_tree.save_newick("/tmp/gene_family.nwk")
```

### RecPhyloXML Format

RecPhyloXML is the standard format for reconciled gene trees:

```python
# Get as string
xml_str = gene_tree.to_xml()

# Save to file
gene_tree.save_xml("/tmp/gene_family.xml")

# Can be visualized with thirdkind or imported into other tools
```

### CSV Export

Export detailed node information including reconciliation mapping:

```python
# Export to DataFrame
df = gene_tree.to_csv()
print(df.head())

# Save to file
gene_tree.to_csv("/tmp/gene_family.csv")

# CSV columns:
# - name: node name
# - parent: parent node index (or NaN for root)
# - left_child, right_child: child indices (or NaN for leaves)
# - length: branch length
# - depth: depth from root
# - species_node: mapped species node name
# - event: event type (Speciation, Duplication, Transfer, Loss, Leaf)
```

### SVG Visualization

Generate publication-quality visualizations (requires `thirdkind`):

```python
# Generate and return as string
svg_content = gene_tree.to_svg()

# Save to file
gene_tree.to_svg("/tmp/gene_family.svg")

# Save and open in browser
gene_tree.to_svg("/tmp/gene_family.svg", open_browser=True)

# Display in Jupyter notebook
gene_tree.display()  # Shows SVG inline
```

**Installation requirement:**
```bash
cargo install thirdkind
```

---

## Advanced Usage

### Batch Processing

Process large numbers of gene families efficiently:

```python
import rustree

# Large-scale simulation
sp_tree = rustree.simulate_species_tree(100, 1.0, 0.4, seed=42)

# Simulate 1000 gene families (batching is efficient)
gene_trees = sp_tree.simulate_dtl_batch(
    n=1000,
    lambda_d=0.5,
    lambda_t=0.2,
    lambda_l=0.3,
    seed=123
)

# Analyze in batch
extant_counts = [gt.num_extant() for gt in gene_trees]
event_counts = [gt.count_events() for gt in gene_trees]

# Save results
import pandas as pd

summary = pd.DataFrame({
    'family_id': range(len(gene_trees)),
    'extant_genes': extant_counts,
    'duplications': [e[1] for e in event_counts],
    'transfers': [e[2] for e in event_counts],
    'losses': [e[3] for e in event_counts]
})

summary.to_csv('/tmp/gene_family_summary.csv', index=False)
```

### Ensuring Extant Genes

When loss rates are high, some simulated gene families may have zero extant genes (all went extinct). Use `require_extant=True` to ensure at least one gene survives:

```python
# Standard simulation (may return extinct families)
gene_tree = sp_tree.simulate_dtl(
    lambda_d=0.3,
    lambda_t=0.1,
    lambda_l=0.8,  # High loss rate
    seed=123
)
print(f"Extant genes: {gene_tree.num_extant()}")  # Might be 0

# Require at least one extant gene (retries until success)
gene_tree = sp_tree.simulate_dtl(
    lambda_d=0.3,
    lambda_t=0.1,
    lambda_l=0.8,
    require_extant=True,  # Ensures >= 1 extant gene
    seed=123
)
print(f"Extant genes: {gene_tree.num_extant()}")  # Guaranteed >= 1

# Also works with batch simulation
gene_trees = sp_tree.simulate_dtl_batch(
    n=100,
    lambda_d=0.3,
    lambda_t=0.1,
    lambda_l=0.8,
    require_extant=True,  # All trees have >= 1 extant gene
    seed=123
)
```

**Definition of "extant":** A gene is considered extant if:
1. It is a leaf node (not a loss event)
2. It is mapped to an extant species (species tree leaf, not an extinct lineage)

### Error Handling

```python
import rustree

# Graceful error handling
try:
    # Attempt to parse invalid Newick
    tree = rustree.parse_species_tree("((A:1,B:1")  # Missing closing parens
except ValueError as e:
    print(f"Parse error: {e}")

# Validate parameters
try:
    # Invalid: mu >= lambda (would cause certain extinction)
    tree = rustree.simulate_species_tree(50, lambda_=1.0, mu=1.5, seed=42)
except ValueError as e:
    print(f"Parameter error: {e}")

# Check for empty results
gene_tree = sp_tree.simulate_dtl(0.5, 0.2, 0.8, seed=123)
if gene_tree.num_extant() == 0:
    print("Warning: Gene family went extinct")
    # Handle accordingly or use require_extant=True
```

---

## Complete Workflows

### Workflow 1: Basic Simulation and Analysis

```python
import rustree
import pandas as pd

# 1. Simulate species tree
print("Simulating species tree...")
sp_tree = rustree.simulate_species_tree(50, 1.0, 0.3, seed=42)
print(f"Species tree: {sp_tree.num_leaves()} leaves, height={sp_tree.tree_height():.2f}")

# 2. Simulate gene families
print("Simulating 100 gene families...")
gene_trees = sp_tree.simulate_dtl_batch(
    100, 0.5, 0.2, 0.3,
    transfer_alpha=1.0,  # Distance-dependent transfers
    seed=123
)

# 3. Analyze results
extant_counts = [gt.num_extant() for gt in gene_trees]
print(f"Extant genes per family: min={min(extant_counts)}, max={max(extant_counts)}, mean={sum(extant_counts)/len(extant_counts):.1f}")

# 4. Save results
sp_tree.save_newick("/tmp/species.nwk")
for i, gt in enumerate(gene_trees[:10]):  # Save first 10
    gt.save_xml(f"/tmp/gene_family_{i:03d}.xml")

print("Done! Results saved to /tmp/")
```

### Workflow 2: Birth-Death Event Analysis

```python
import rustree
import pandas as pd
import matplotlib.pyplot as plt

# 1. Simulate species tree with events
sp_tree = rustree.simulate_species_tree(30, lambda_=1.2, mu=0.4, seed=42)

# 2. Export and analyze birth-death events
sp_tree.save_bd_events_csv("/tmp/bd_events.csv")
df = pd.read_csv("/tmp/bd_events.csv")

print("Event summary:")
print(df['event_type'].value_counts())

# 3. Plot event distribution
speciations = df[df['event_type'] == 'Speciation']['time']
extinctions = df[df['event_type'] == 'Extinction']['time']

plt.figure(figsize=(10, 6))
plt.hist(speciations, bins=20, alpha=0.7, label='Speciation')
plt.hist(extinctions, bins=20, alpha=0.7, label='Extinction')
plt.xlabel('Time (backwards from present)')
plt.ylabel('Count')
plt.legend()
plt.savefig('/tmp/event_times.png')
```

### Workflow 3: Distance-Based Phylogenetic Analysis

```python
import rustree
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 1. Simulate species tree
sp_tree = rustree.simulate_species_tree(20, 1.0, 0.3, seed=42)

# 2. Compute pairwise distances
sp_tree.save_pairwise_distances_csv(
    "/tmp/distances.csv",
    distance_type="metric",
    leaves_only=True
)

# 3. Load and convert to distance matrix
df = pd.read_csv("/tmp/distances.csv")
leaf_names = sp_tree.leaf_names()
n = len(leaf_names)

dist_matrix = np.zeros((n, n))
name_to_idx = {name: i for i, name in enumerate(leaf_names)}

for _, row in df.iterrows():
    i = name_to_idx[row['node1']]
    j = name_to_idx[row['node2']]
    dist_matrix[i, j] = row['distance']

# 4. Hierarchical clustering
condensed_dist = dist_matrix[np.triu_indices(n, k=1)]
linkage_matrix = linkage(condensed_dist, method='average')

plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=leaf_names, leaf_rotation=90)
plt.title("Hierarchical Clustering from Pairwise Distances")
plt.tight_layout()
plt.savefig('/tmp/clustering.png')
```

### Workflow 4: Model Comparison

```python
import rustree
import matplotlib.pyplot as plt

# Fixed species tree
sp_tree = rustree.simulate_species_tree(50, 1.0, 0.4, seed=42)

# Compare transfer models
scenarios = [
    {"transfer_alpha": None, "label": "Uniform transfers"},
    {"transfer_alpha": 1.0, "label": "Moderate local"},
    {"transfer_alpha": 3.0, "label": "Strong local"},
]

results = {}
for scenario in scenarios:
    gene_trees = sp_tree.simulate_dtl_batch(
        n=100,
        lambda_d=0.5,
        lambda_t=0.3,
        lambda_l=0.3,
        transfer_alpha=scenario['transfer_alpha'],
        seed=123
    )

    extant = [gt.num_extant() for gt in gene_trees]
    results[scenario['label']] = extant

# Plot comparison
plt.figure(figsize=(10, 6))
plt.boxplot(results.values(), labels=results.keys())
plt.ylabel('Extant genes per family')
plt.title('Effect of Transfer Model on Gene Family Size')
plt.savefig('/tmp/model_comparison.png')
```

---

## API Reference

### Module Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `simulate_species_tree(n, lambda_, mu, seed=None)` | Simulate birth-death tree | `PySpeciesTree` |
| `parse_species_tree(newick_str)` | Parse Newick string | `PySpeciesTree` |
| `parse_recphyloxml(filepath)` | Parse RecPhyloXML file | `PyGeneTree` |

### PySpeciesTree Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `to_newick()` | Convert to Newick string | `str` |
| `save_newick(filepath)` | Save to Newick file | `None` |
| `num_nodes()` | Count all nodes | `int` |
| `num_leaves()` | Count leaves | `int` |
| `tree_height()` | Get total height | `float` |
| `leaf_names()` | Get list of leaf names | `list[str]` |
| `root_index()` | Get root node index | `int` |
| `extract_induced_subtree_by_names(names)` | Extract subtree | `PySpeciesTree` |
| `simulate_dtl(lambda_d, lambda_t, lambda_l, transfer_alpha=None, require_extant=False, seed=None)` | Simulate gene tree (per-copy) | `PyGeneTree` |
| `simulate_dtl_batch(n, lambda_d, lambda_t, lambda_l, transfer_alpha=None, require_extant=False, seed=None)` | Simulate batch (per-copy) | `list[PyGeneTree]` |
| `simulate_dtl_per_species(...)` | Simulate gene tree (per-species) | `PyGeneTree` |
| `simulate_dtl_per_species_batch(...)` | Simulate batch (per-species) | `list[PyGeneTree]` |
| `get_bd_events()` | Get birth-death events | `dict` |
| `save_bd_events_csv(filepath)` | Save BD events to CSV | `None` |
| `get_ltt_data()` | Get LTT data | `dict` |
| `plot_ltt(filepath=None, title=None, xlabel=None, ylabel=None)` | Plot LTT | `None` |
| `pairwise_distances(distance_type, leaves_only=True)` | Compute distances | `list[PairwiseDistance]` |
| `save_pairwise_distances_csv(filepath, distance_type, leaves_only=True)` | Save distances | `None` |

### PyGeneTree Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `to_newick()` | Convert to Newick string | `str` |
| `save_newick(filepath)` | Save to Newick file | `None` |
| `num_nodes()` | Count all nodes | `int` |
| `num_extant()` | Count extant genes | `int` |
| `count_events()` | Return (S, D, T, L, Leaves) | `tuple[int, int, int, int, int]` |
| `extant_gene_names()` | Get extant gene names | `list[str]` |
| `sample_extant()` | Extract extant-only subtree | `PyGeneTree` |
| `sample_by_names(names)` | Extract subtree by names | `PyGeneTree` |
| `sample_species_leaves(species_names)` | Filter by species | `PyGeneTree` |
| `to_xml()` | Convert to RecPhyloXML | `str` |
| `save_xml(filepath)` | Save to RecPhyloXML | `None` |
| `to_csv(filepath=None)` | Export to DataFrame | `pandas.DataFrame` |
| `to_svg(filepath=None, open_browser=False)` | Generate SVG | `str` |
| `display()` | Display in Jupyter | `None` |
| `pairwise_distances(distance_type, leaves_only=True)` | Compute distances | `list[PairwiseDistance]` |
| `save_pairwise_distances_csv(filepath, distance_type, leaves_only=True)` | Save distances | `None` |

### Parameters

#### DTL Rates
- `lambda_d`: Duplication rate per unit time
- `lambda_t`: Transfer rate per unit time
- `lambda_l`: Loss rate per unit time

For per-copy model: rates are per gene copy
For per-species model: rates are per alive species

#### Transfer Model
- `transfer_alpha`: Distance decay parameter
  - `None` or `0`: Uniform random
  - `> 0`: Distance-dependent (higher = more local)

#### Distance Types
- `"metric"`, `"patristic"`, `"branch"`: Sum of branch lengths
- `"topological"`, `"topo"`: Number of edges

#### Other
- `require_extant`: If `True`, retry until ≥ 1 extant gene
- `leaves_only`: If `True`, only compute leaf-to-leaf distances
- `seed`: Random seed for reproducibility

---

## Troubleshooting

### Common Issues

**1. Import error: `ModuleNotFoundError: No module named 'rustree'`**

Solution: Install with maturin:
```bash
cd /path/to/rustree
maturin develop --release
```

**2. SVG generation fails: `thirdkind not found`**

Solution: Install thirdkind:
```bash
cargo install thirdkind
```

**3. Simulation hangs or is very slow**

Possible causes:
- Very large trees (>1000 species) with high DTL rates
- High loss rate with `require_extant=True` (many retries)

Solutions:
- Reduce tree size or DTL rates
- Use batch simulation (more efficient)
- Remove `require_extant=True` and filter results manually

**4. Many gene families have zero extant genes**

Cause: Loss rate too high relative to duplication/transfer rates

Solutions:
- Reduce `lambda_l`
- Increase `lambda_d` or `lambda_t`
- Use `require_extant=True`

**5. LTT plotting fails: `ModuleNotFoundError: No module named 'matplotlib'`**

Solution: Install matplotlib:
```bash
pip install matplotlib
```

### Getting Help

- Check [R_TUTORIAL.md](R_TUTORIAL.md) for R-specific examples
- See [RECPHYLOXML.md](RECPHYLOXML.md) for reconciliation file format details
- See [LTT_PLOTS.md](LTT_PLOTS.md) for advanced LTT plotting
- See [PERFORMANCE_PROFILING.md](PERFORMANCE_PROFILING.md) for performance optimization

---

## See Also

- **[R Tutorial](R_TUTORIAL.md)** - R bindings documentation with comparison notes
- **[RecPhyloXML Format](RECPHYLOXML.md)** - Reconciliation file format specification
- **[LTT Plots](LTT_PLOTS.md)** - Advanced lineages-through-time visualization
- **[Performance Profiling](PERFORMANCE_PROFILING.md)** - Optimization and benchmarking results
- **rustree GitHub repository** - Source code, issues, and contributions

---

**Document Version:** 2.0
**Last Updated:** 2026-02-14
**rustree Version:** 0.1.0
**Project Root:** repository root
