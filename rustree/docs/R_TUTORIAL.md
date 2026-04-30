# rustree R Package Tutorial

Comprehensive guide to building and using the rustree R bindings for phylogenetic tree simulation, reconciliation, and analysis.

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
6. [Analysis](#analysis)
   - [Birth-Death Events](#birth-death-events)
   - [Pairwise Distances](#pairwise-distances)
   - [Subtree Extraction](#subtree-extraction)
   - [Species Leaf Sampling](#species-leaf-sampling)
   - [Parsing RecPhyloXML](#parsing-recphyloxml)
   - [Streaming Simulation](#streaming-simulation-memory-efficient)
7. [Export and Visualization](#export-and-visualization)
   - [Newick Format](#newick-format)
   - [RecPhyloXML Format](#recphyloxml-format)
   - [CSV Export](#csv-export)
   - [SVG Visualization](#svg-visualization)
8. [Complete Workflows](#complete-workflows)
9. [API Reference](#api-reference)
10. [Data Structures](#data-structures)
11. [Comparison to Python Bindings](#comparison-to-python-bindings)
12. [Troubleshooting](#troubleshooting)
13. [See Also](#see-also)

---

## Prerequisites

### Required
- **R** 4.0 or higher
- **Rust** 1.70 or higher (for building from source)
- **Cargo** (comes with Rust)

### Optional
- **thirdkind** for SVG visualization:
  ```bash
  cargo install thirdkind
  ```
- **ggplot2** for enhanced plotting (recommended):
  ```r
  install.packages("ggplot2")
  ```

---

## Installation

### Build from Source

**Important:** Build with R feature only to avoid symbol conflicts:

```bash
cd /path/to/rustree

# Build with R feature only (exclude Python)
cargo build --release --no-default-features --features r
```

The compiled library will be at:
- Linux: `target/release/librustree.so`
- macOS: `target/release/librustree.dylib`
- Windows: `target/release/rustree.dll`

### Load in R

```r
# Load the library (adjust path as needed)
dyn.load("target/release/librustree.so")  # Linux
# dyn.load("target/release/librustree.dylib")  # macOS

# Source the wrapper functions
source("R/rustree.R")
```

### Verify Installation

```r
# Simulate a small tree to test
sp_tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
cat("Success! Tree has", tree_num_leaves(sp_tree), "leaves\n")
```

---

## Quick Start

```r
# Load library
dyn.load("target/release/librustree.so")
source("R/rustree.R")

# Simulate a species tree with 50 extant species
sp_tree <- simulate_species_tree(50L, 1.0, 0.3, seed = 42L)
cat("Species:", tree_num_leaves(sp_tree), "\n")

# Simulate 10 gene families with DTL events
gene_trees <- simulate_dtl_batch(
  sp_tree, 10L,
  lambda_d = 0.5,  # Duplication rate
  lambda_t = 0.2,  # Transfer rate
  lambda_l = 0.3,  # Loss rate
  seed = 123L
)

# Analyze the first gene family
gt <- gene_trees[[1]]
cat("Extant genes:", gene_tree_num_extant(gt), "\n")
cat("Events:", table(gt$event), "\n")

# Export results
save_newick(sp_tree, "/tmp/species.nwk")
save_xml(gt, "/tmp/gene_family_001.xml")
```

---

## Core Concepts

### Species Trees

Species trees represent the evolutionary history of species through speciation and extinction events. In rustree, species trees are simulated using a **birth-death process** or parsed from Newick format.

**Key operations:**
- `tree_num_leaves()` - Count extant species
- `tree_num_nodes()` - Count all nodes (including extinct lineages)
- `tree_height()` - Time from present to root (MRCA)
- `tree_leaf_names()` - List of species names

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

```r
# Simulate tree with 100 extant species
sp_tree <- simulate_species_tree(
  n = 100L,        # Number of extant species (use L for integer)
  lambda = 1.0,    # Speciation (birth) rate
  mu = 0.5,        # Extinction (death) rate (must be < lambda)
  seed = 42L       # Random seed for reproducibility (optional)
)

# Inspect the tree
cat("Extant species:", tree_num_leaves(sp_tree), "\n")
cat("Total nodes:", length(sp_tree$name), "\n")
cat("Tree height:", max(sp_tree$depth), "\n")
cat("Species names:", paste(tree_leaf_names(sp_tree)[1:5], collapse = ", "), "...\n")
```

**Note on extinction rate:** The condition μ < λ ensures net diversification. If μ ≥ λ, the lineage would eventually go extinct.

**R-specific note:** All integer arguments **must** use the `L` suffix (e.g., `50L` not `50`).

### Parsing Newick Trees

Parse existing Newick-format trees:

```r
# Parse from string
newick_str <- "((A:1.0,B:1.0):1.0,(C:0.5,D:0.5):1.5):0;"
sp_tree <- parse_newick(newick_str)

# Or read from file
newick_str <- readLines("/tmp/species_tree.nwk")
sp_tree <- parse_newick(newick_str)

cat("Parsed", tree_num_leaves(sp_tree), "species\n")
```

### DTL Gene Tree Simulation

#### Single Gene Tree

```r
# Simulate one gene family
gene_tree <- simulate_dtl(
  species_tree = sp_tree,
  lambda_d = 0.5,    # Duplication rate
  lambda_t = 0.2,    # Transfer rate
  lambda_l = 0.3,    # Loss rate
  seed = 123L        # Optional seed
)

# Inspect results
cat("Extant genes:", gene_tree_num_extant(gene_tree), "\n")
cat("Events:\n")
print(table(gene_tree$event))
```

#### Batch Simulation (Recommended)

For multiple gene families, use `simulate_dtl_batch()` which is much more efficient:

```r
# Simulate 100 gene families
gene_trees <- simulate_dtl_batch(
  species_tree = sp_tree,
  n = 100L,          # Number of gene families
  lambda_d = 0.5,
  lambda_t = 0.2,
  lambda_l = 0.3,
  seed = 123L
)

# Analyze all families
extant_counts <- sapply(gene_trees, gene_tree_num_extant)
cat("Mean extant genes:", mean(extant_counts), "\n")
cat("Range:", min(extant_counts), "-", max(extant_counts), "\n")
```

**Performance note:** Batch simulation pre-computes shared data structures, providing significant speedup (~2-3x) compared to looping `simulate_dtl()`.

### Per-Gene-Copy vs Per-Species Models

#### Per-Gene-Copy Model (Standard)

The standard DTL model uses event rates proportional to the number of **gene copies** alive at any time. More gene copies → higher total event rate.

```r
# Standard model: event rate scales with gene copy count
gene_tree <- simulate_dtl(
  species_tree = sp_tree,
  lambda_d = 0.5,  # Rate per gene copy per unit time
  lambda_t = 0.2,
  lambda_l = 0.3,
  seed = 123L
)
```

**Characteristics:**
- Duplications increase the total event rate (more copies → more events)
- Can produce large gene families through runaway duplication
- Total event rate at time t: `(λ_d + λ_t + λ_l) × n_gene_copies(t)`

#### Per-Species Model (Zombi-Style)

The per-species model uses event rates proportional to the number of **alive species**, regardless of gene copy count. When an event fires, a random species is chosen; if it has no gene copies, the event fails silently.

```r
# Per-species model: event rate scales with species count
gene_tree <- simulate_dtl_per_species(
  species_tree = sp_tree,
  lambda_d = 0.5,  # Rate per species per unit time
  lambda_t = 0.2,
  lambda_l = 0.3,
  seed = 123L
)

# Batch version (more efficient)
gene_trees <- simulate_dtl_per_species_batch(
  species_tree = sp_tree,
  n = 100L,
  lambda_d = 0.5,
  lambda_t = 0.2,
  lambda_l = 0.3,
  seed = 123L
)
```

**Characteristics:**
- Duplications do NOT increase the total event rate
- Produces smaller gene families on average
- Total event rate at time t: `(λ_d + λ_t + λ_l) × n_alive_species(t)`

#### Comparing Models

```r
sp_tree <- simulate_species_tree(50L, 1.0, 0.3, seed = 42L)

# Per-copy model
gt_copy <- simulate_dtl(sp_tree, 0.5, 0.2, 0.3, seed = 1L)

# Per-species model
gt_species <- simulate_dtl_per_species(sp_tree, 0.5, 0.2, 0.3, seed = 1L)

cat("Per-copy model:\n")
cat("  Events:", paste(names(table(gt_copy$event)), table(gt_copy$event), sep = "=", collapse = ", "), "\n")
cat("  Extant:", gene_tree_num_extant(gt_copy), "\n\n")

cat("Per-species model:\n")
cat("  Events:", paste(names(table(gt_species$event)), table(gt_species$event), sep = "=", collapse = ", "), "\n")
cat("  Extant:", gene_tree_num_extant(gt_species), "\n")

# Per-species model typically shows fewer events and smaller families
```

### Assortative Transfers

By default, transfer recipients are chosen uniformly at random from contemporary species. **Assortative transfers** model distance-dependent transfer where closer species are more likely to exchange genes.

```r
# Distance-dependent transfers
# P(recipient) ∝ exp(-alpha × distance)
gene_tree <- simulate_dtl(
  species_tree = sp_tree,
  lambda_d = 0.5,
  lambda_t = 0.5,
  lambda_l = 0.3,
  transfer_alpha = 1.0,  # Distance decay parameter
  seed = 123L
)

# Batch with assortative transfers
gene_trees <- simulate_dtl_batch(
  species_tree = sp_tree,
  n = 100L,
  lambda_d = 0.5,
  lambda_t = 0.5,
  lambda_l = 0.3,
  transfer_alpha = 2.0,  # Stronger local preference
  seed = 123L
)
```

**Distance calculation:**
```
distance(A, B, t) = 2 × (t - depth_of_LCA(A, B))
```

**Parameter guidelines:**
- `transfer_alpha = NULL` or `0`: Uniform random (no distance preference)
- `transfer_alpha = 1.0`: Moderate local preference
- `transfer_alpha = 2.0`: Strong local preference
- `transfer_alpha = 5.0`: Very strong local preference (rare long-distance transfers)

**Works with both models:**
```r
# Assortative transfers with per-species model
gene_tree <- simulate_dtl_per_species(
  species_tree = sp_tree,
  lambda_d = 0.5,
  lambda_t = 0.5,
  lambda_l = 0.3,
  transfer_alpha = 1.5,
  seed = 123L
)
```

---

## Analysis

### Birth-Death Events

Species trees simulated with `simulate_species_tree()` include birth-death event data:

```r
# Simulate species tree
sp_tree <- simulate_species_tree(50L, 1.0, 0.5, seed = 42L)

# Access birth-death events
events <- sp_tree$events
cat("Event count:", nrow(as.data.frame(events)), "\n")

# Save to CSV
save_bd_events_csv(sp_tree, "/tmp/bd_events.csv")

# Analyze with R
df <- read.csv("/tmp/bd_events.csv")
print(table(df$event_type))
```

**CSV columns:**
- `time`: When the event occurred (backwards from present at 0)
- `node_id`: Node index where event occurred
- `event_type`: "Speciation", "Extinction", or "Leaf"
- `child1`, `child2`: Child node indices (for speciation events)

### Pairwise Distances

Calculate distances between nodes for phylogenetic analysis:

```r
# Metric distance (sum of branch lengths)
metric_dists <- pairwise_distances(
  sp_tree,
  distance_type = "metric",
  leaves_only = TRUE
)

# View results
print(head(metric_dists))

# Topological distance (edge count)
topo_dists <- pairwise_distances(
  sp_tree,
  distance_type = "topological",
  leaves_only = TRUE
)

# Save to CSV
save_pairwise_distances_csv(
  sp_tree,
  "/tmp/distances.csv",
  distance_type = "metric",
  leaves_only = TRUE
)
```

**Distance types:**
- `"metric"`, `"patristic"`, or `"branch"`: Sum of branch lengths (numeric)
- `"topological"` or `"topo"`: Number of edges (integer)

**Scope:**
- `leaves_only = TRUE`: Only leaf-to-leaf distances (extant species/genes)
- `leaves_only = FALSE`: All pairwise distances including internal nodes

### Subtree Extraction

Extract induced subtrees containing only specified taxa:

```r
# Get all species
all_species <- tree_leaf_names(sp_tree)

# Select subset
selected_species <- all_species[1:10]

# Extract induced subtree
subtree <- extract_induced_subtree_by_names(sp_tree, selected_species)
cat("Original:", tree_num_leaves(sp_tree), "leaves\n")
cat("Subtree:", tree_num_leaves(subtree), "leaves\n")

# Also works with gene trees
gene_tree <- simulate_dtl(sp_tree, 0.5, 0.2, 0.3, seed = 123L)
gene_names <- tree_leaf_names(gene_tree)[1:20]
sampled_gt <- extract_induced_subtree_by_names(gene_tree, gene_names)
```

### Species Leaf Sampling

`sample_leaves()` samples a subset of species from the species tree and automatically filters the gene tree to keep only genes mapping to the sampled species. Reconciliation mappings are preserved using an LCA-based approach.

```r
# Simulate
sp_tree <- simulate_species_tree(50L, 1.0, 0.3, seed = 42L)
gene_tree <- simulate_dtl(sp_tree, 0.5, 0.2, 0.3, seed = 123L)

# Keep only a subset of species
selected <- tree_leaf_names(sp_tree)[1:10]
sampled_gt <- sample_leaves(gene_tree, selected)

cat("Original extant:", gene_tree_num_extant(gene_tree), "\n")
cat("After sampling:", gene_tree_num_extant(sampled_gt), "\n")
```

### Parsing RecPhyloXML

Import reconciled gene trees from RecPhyloXML files (e.g., output from ALERax):

```r
# Parse a RecPhyloXML file
gene_tree <- parse_recphyloxml("/path/to/reconciled_tree.xml")

# Inspect the result
cat("Extant genes:", gene_tree_num_extant(gene_tree), "\n")
cat("Events:", paste(names(table(gene_tree$event)), table(gene_tree$event), sep = "=", collapse = ", "), "\n")

# Round-trip: export a gene tree and re-import it
save_xml(gene_tree, "/tmp/exported.xml")
reimported <- parse_recphyloxml("/tmp/exported.xml")
```

### Streaming Simulation (Memory-Efficient)

For large-scale simulations where holding all gene trees in memory is impractical, streaming functions write each tree directly to disk as it is generated. Only one tree is in memory at a time.

**Key difference from batch functions:** Streaming functions take a **Newick string** as input (not a tree list), and write files to an output directory instead of returning R objects.

#### Streaming to RecPhyloXML

```r
# Get the species tree as a Newick string
sp_tree <- simulate_species_tree(50L, 1.0, 0.3, seed = 42L)
newick <- tree_to_newick(sp_tree)

# Stream 1000 gene trees to XML files (per-gene-copy model)
simulate_dtl_stream_xml(
  newick, 1000L,
  lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
  seed = 123L,
  output_dir = "/tmp/gene_trees_xml"
)
# Creates: /tmp/gene_trees_xml/gene_0000.xml, gene_0001.xml, ...

# Per-species (Zombi) model
simulate_dtl_per_species_stream_xml(
  newick, 1000L,
  lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
  seed = 123L,
  output_dir = "/tmp/gene_trees_per_species_xml"
)
```

#### Streaming to Newick

```r
# Stream to Newick files (no reconciliation info preserved)
simulate_dtl_stream_newick(
  newick, 1000L,
  lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
  seed = 123L,
  output_dir = "/tmp/gene_trees_nwk"
)
# Creates: /tmp/gene_trees_nwk/gene_0000.nwk, gene_0001.nwk, ...

# Per-species model to Newick
simulate_dtl_per_species_stream_newick(
  newick, 1000L,
  lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
  seed = 123L,
  output_dir = "/tmp/gene_trees_per_species_nwk"
)
```

#### Optional Parameters

All streaming functions also support:
- `transfer_alpha`: Distance decay for assortative transfers (`NULL` = uniform)
- `replacement_transfer`: Probability of replacement transfer (`NULL` = no replacement)
- `require_extant`: If `TRUE`, retry until each tree has at least 1 extant gene

```r
simulate_dtl_stream_xml(
  newick, 500L,
  lambda_d = 0.5, lambda_t = 0.3, lambda_l = 0.3,
  transfer_alpha = 1.5,
  replacement_transfer = 0.1,
  require_extant = TRUE,
  seed = 42L,
  output_dir = "/tmp/assortative_xml"
)
```

---

## Export and Visualization

### Newick Format

```r
# Get as string
newick_str <- tree_to_newick(sp_tree)
cat(substr(newick_str, 1, 100), "...\n")  # First 100 characters

# Save to file
save_newick(sp_tree, "/tmp/species.nwk")
save_newick(gene_tree, "/tmp/gene_family.nwk")
```

### RecPhyloXML Format

RecPhyloXML is the standard format for reconciled gene trees:

```r
# Get as string
xml_str <- gene_tree_to_xml(gene_tree)

# Save to file
save_xml(gene_tree, "/tmp/gene_family.xml")

# Can be visualized with thirdkind or imported into other tools
```

### CSV Export

Export detailed node information including reconciliation mapping:

```r
# Save to file (returns invisibly)
save_csv(gene_tree, "/tmp/gene_family.csv")

# Read and analyze
df <- read.csv("/tmp/gene_family.csv")
print(head(df))

# CSV columns:
# - node_id: node index
# - name: node name
# - parent: parent node index (or NA for root)
# - left_child, right_child: child indices (or NA for leaves)
# - length: branch length
# - depth: depth from root
# - species_node: mapped species node name
# - event: event type (Speciation, Duplication, Transfer, Loss, Leaf)
```

### SVG Visualization

Generate publication-quality visualizations (requires `thirdkind`):

```r
# Generate and return as string
svg_content <- gene_tree_to_svg(gene_tree)

# Save to file
gene_tree_to_svg(gene_tree, "/tmp/gene_family.svg")

# Save and open in browser
gene_tree_to_svg(gene_tree, "/tmp/gene_family.svg", open_browser = TRUE)
```

**Installation requirement:**
```bash
cargo install thirdkind
```

---

## Complete Workflows

### Workflow 1: Basic Simulation and Analysis

```r
# Load library
dyn.load("target/release/librustree.so")
source("R/rustree.R")

# 1. Simulate species tree
cat("Simulating species tree...\n")
sp_tree <- simulate_species_tree(50L, 1.0, 0.3, seed = 42L)
cat("Species tree:", tree_num_leaves(sp_tree), "leaves\n")

# 2. Simulate gene families
cat("Simulating 100 gene families...\n")
gene_trees <- simulate_dtl_batch(
  sp_tree, 100L, 0.5, 0.2, 0.3,
  transfer_alpha = 1.0,  # Distance-dependent transfers
  seed = 123L
)

# 3. Analyze results
extant_counts <- sapply(gene_trees, gene_tree_num_extant)
cat("Extant genes per family:\n")
cat("  Min:", min(extant_counts), "\n")
cat("  Max:", max(extant_counts), "\n")
cat("  Mean:", mean(extant_counts), "\n")

# 4. Save results
save_newick(sp_tree, "/tmp/species.nwk")
for (i in 1:min(10, length(gene_trees))) {
  save_xml(gene_trees[[i]], sprintf("/tmp/gene_family_%03d.xml", i))
}

cat("Done! Results saved to /tmp/\n")
```

### Workflow 2: Birth-Death Event Analysis

```r
library(ggplot2)  # Optional, for nicer plots

# 1. Simulate species tree with events
sp_tree <- simulate_species_tree(30L, lambda = 1.2, mu = 0.4, seed = 42L)

# 2. Export and analyze birth-death events
save_bd_events_csv(sp_tree, "/tmp/bd_events.csv")
df <- read.csv("/tmp/bd_events.csv")

cat("Event summary:\n")
print(table(df$event_type))

# 3. Plot event distribution
speciations <- df$time[df$event_type == "Speciation"]
extinctions <- df$time[df$event_type == "Extinction"]

par(mfrow = c(1, 2))
hist(speciations, breaks = 20, main = "Speciation Events",
     xlab = "Time (backwards from present)", col = "lightblue")
hist(extinctions, breaks = 20, main = "Extinction Events",
     xlab = "Time (backwards from present)", col = "salmon")
par(mfrow = c(1, 1))
```

### Workflow 3: Distance-Based Analysis

```r
# 1. Simulate species tree
sp_tree <- simulate_species_tree(20L, 1.0, 0.3, seed = 42L)

# 2. Compute pairwise distances
save_pairwise_distances_csv(
  sp_tree, "/tmp/distances.csv",
  distance_type = "metric",
  leaves_only = TRUE
)

# 3. Load and convert to distance matrix
df <- read.csv("/tmp/distances.csv")
leaf_names <- tree_leaf_names(sp_tree)
n <- length(leaf_names)

dist_matrix <- matrix(0, n, n, dimnames = list(leaf_names, leaf_names))
for (i in 1:nrow(df)) {
  dist_matrix[df$node1[i], df$node2[i]] <- df$distance[i]
}

# 4. Hierarchical clustering
hc <- hclust(as.dist(dist_matrix))
plot(hc, main = "Hierarchical Clustering from Pairwise Distances")
```

### Workflow 4: Model Comparison

```r
# Fixed species tree
sp_tree <- simulate_species_tree(50L, 1.0, 0.4, seed = 42L)

# Compare transfer models
scenarios <- list(
  list(transfer_alpha = NULL, label = "Uniform transfers"),
  list(transfer_alpha = 1.0, label = "Moderate local"),
  list(transfer_alpha = 3.0, label = "Strong local")
)

results <- list()
for (scenario in scenarios) {
  gene_trees <- simulate_dtl_batch(
    sp_tree, 100L,
    lambda_d = 0.5,
    lambda_t = 0.3,
    lambda_l = 0.3,
    transfer_alpha = scenario$transfer_alpha,
    seed = 123L
  )

  extant <- sapply(gene_trees, gene_tree_num_extant)
  results[[scenario$label]] <- extant
}

# Plot comparison
boxplot(results,
        ylab = "Extant genes per family",
        main = "Effect of Transfer Model on Gene Family Size",
        col = c("lightblue", "lightgreen", "lightyellow"))
```

---

## API Reference

### Species Tree Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `simulate_species_tree(n, lambda, mu, seed)` | Simulate birth-death tree | `list` |
| `parse_newick(newick_str)` | Parse Newick string | `list` |
| `tree_to_newick(tree)` | Convert to Newick | `character` |
| `tree_num_leaves(tree)` | Count leaves | `integer` |
| `tree_leaf_names(tree)` | Get leaf names | `character vector` |

### Gene Tree Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `simulate_dtl(sp_tree, lambda_d, lambda_t, lambda_l, transfer_alpha, require_extant, seed)` | Simulate gene tree (per-copy) | `list` |
| `simulate_dtl_batch(sp_tree, n, ...)` | Simulate batch (per-copy) | `list of lists` |
| `simulate_dtl_per_species(sp_tree, ...)` | Simulate gene tree (per-species) | `list` |
| `simulate_dtl_per_species_batch(sp_tree, n, ...)` | Simulate batch (per-species) | `list of lists` |
| `gene_tree_num_extant(gt)` | Count extant genes | `integer` |
| `gene_tree_to_newick(gt)` | Convert to Newick | `character` |
| `gene_tree_to_xml(gt)` | Convert to RecPhyloXML | `character` |

### Analysis Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `extract_induced_subtree_by_names(tree, leaf_names)` | Extract subtree | `list` |
| `sample_leaves(gene_tree, species_leaf_names)` | Sample species and filter gene tree | `list` |
| `pairwise_distances(tree, distance_type, leaves_only)` | Compute distances | `data.frame` |
| `save_pairwise_distances_csv(tree, filepath, distance_type, leaves_only)` | Save distances | `NULL` |

### Import/Export Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `parse_recphyloxml(filepath)` | Parse RecPhyloXML file | `list` |
| `save_newick(tree, filepath)` | Save to Newick | `NULL` |
| `save_xml(gt, filepath)` | Save to RecPhyloXML | `NULL` |
| `save_csv(gt, filepath)` | Save to CSV | `NULL` |
| `save_bd_events_csv(sp_tree, filepath)` | Save BD events | `NULL` |
| `gene_tree_to_svg(gt, filepath, open_browser)` | Generate SVG | `character` or `NULL` |

### Streaming Functions (Memory-Efficient)

| Function | Description | Returns |
|----------|-------------|---------|
| `simulate_dtl_stream_xml(newick, n, ...)` | Stream gene trees to XML (per-copy) | `character` (dir path) |
| `simulate_dtl_stream_newick(newick, n, ...)` | Stream gene trees to Newick (per-copy) | `character` (dir path) |
| `simulate_dtl_per_species_stream_xml(newick, n, ...)` | Stream gene trees to XML (per-species) | `character` (dir path) |
| `simulate_dtl_per_species_stream_newick(newick, n, ...)` | Stream gene trees to Newick (per-species) | `character` (dir path) |

### Parameters

#### DTL Rates
- `lambda_d`: Duplication rate per unit time (numeric)
- `lambda_t`: Transfer rate per unit time (numeric)
- `lambda_l`: Loss rate per unit time (numeric)

For per-copy model: rates are per gene copy
For per-species model: rates are per alive species

#### Transfer Model
- `transfer_alpha`: Distance decay parameter (numeric or `NULL`)
  - `NULL` or `0`: Uniform random
  - `> 0`: Distance-dependent (higher = more local)

#### Distance Types
- `"metric"`, `"patristic"`, `"branch"`: Sum of branch lengths
- `"topological"`, `"topo"`: Number of edges

#### Streaming Parameters
- `newick`: Species tree as a Newick string (character) — note: streaming functions take a string, not a tree list
- `output_dir`: Directory to write output files (created if it doesn't exist)
- `replacement_transfer`: Probability of replacement transfer (numeric or `NULL`)

#### Other
- `require_extant`: If `TRUE`, retry until ≥ 1 extant gene (logical)
- `leaves_only`: If `TRUE`, only compute leaf-to-leaf distances (logical)
- `seed`: Random seed for reproducibility (integer with `L` suffix or `NULL`)
- `n`: Number of entities (integer with `L` suffix)

**R-specific note:** All integer parameters must use `L` suffix (e.g., `50L`, `123L`).

---

## Data Structures

### Species Tree Structure

```r
sp_tree <- simulate_species_tree(10L, 1.0, 0.5, seed = 42L)

# Tree structure (flat representation)
sp_tree$name           # Character vector: node names
sp_tree$parent         # Integer vector: parent indices (NA for root)
sp_tree$left_child     # Integer vector: left child indices (NA for leaves)
sp_tree$right_child    # Integer vector: right child indices (NA for leaves)
sp_tree$length         # Numeric vector: branch lengths
sp_tree$depth          # Numeric vector: node depths from root
sp_tree$root           # Integer: index of root node

# Birth-death events
sp_tree$events$time         # Numeric vector: event times (backward from 0)
sp_tree$events$node_id      # Integer vector: node where event occurred
sp_tree$events$event_type   # Character vector: "Speciation", "Extinction", "Leaf"
sp_tree$events$child1       # Integer vector: first child index
sp_tree$events$child2       # Integer vector: second child index
```

### Gene Tree Structure

```r
gt <- simulate_dtl(sp_tree, 0.5, 0.2, 0.3, seed = 123L)

# Tree structure (same as species tree)
gt$name, gt$parent, gt$left_child, gt$right_child
gt$length, gt$depth, gt$root

# Reconciliation information
gt$species_node    # Character vector: mapped species node name
gt$event           # Character vector: event type

# The original species tree is embedded
gt$species_tree    # Nested list with species tree structure
```

### Working with Indices

R uses 1-based indexing, but the wrapper handles conversion automatically:

```r
# Access node by index (R uses 1-based)
node_idx <- 5
node_name <- sp_tree$name[node_idx]
parent_idx <- sp_tree$parent[node_idx]

# Navigate the tree
if (!is.na(parent_idx)) {
  parent_name <- sp_tree$name[parent_idx]
  cat("Node", node_name, "has parent", parent_name, "\n")
}

# Get children
left_idx <- sp_tree$left_child[node_idx]
right_idx <- sp_tree$right_child[node_idx]

if (!is.na(left_idx)) {
  left_name <- sp_tree$name[left_idx]
  right_name <- sp_tree$name[right_idx]
  cat("Node", node_name, "has children", left_name, "and", right_name, "\n")
}
```

---

## Comparison to Python Bindings

The R bindings provide similar functionality to the Python bindings but with R-specific conventions:

| Feature | Python | R |
|---------|--------|---|
| **Integer literals** | `50` | `50L` (must use `L` suffix) |
| **Missing values** | `None` | `NULL` |
| **List indexing** | `gene_trees[0]` (0-based) | `gene_trees[[1]]` (1-based) |
| **Method calls** | `sp_tree.num_leaves()` | `tree_num_leaves(sp_tree)` |
| **DataFrame** | Returns pandas DataFrame | Returns data.frame |
| **Visualization** | `display()` for Jupyter | Standard R plotting |

### Example Comparison

**Python:**
```python
import rustree
sp_tree = rustree.simulate_species_tree(50, 1.0, 0.3, seed=42)
gene_trees = sp_tree.simulate_dtl_batch(100, 0.5, 0.2, 0.3, seed=123)
gt = gene_trees[0]
print(gt.num_extant())
```

**R:**
```r
dyn.load("target/release/librustree.so")
source("R/rustree.R")
sp_tree <- simulate_species_tree(50L, 1.0, 0.3, seed = 42L)
gene_trees <- simulate_dtl_batch(sp_tree, 100L, 0.5, 0.2, 0.3, seed = 123L)
gt <- gene_trees[[1]]
cat(gene_tree_num_extant(gt), "\n")
```

---

## Troubleshooting

### Common Issues

**1. Library not loaded error**

```r
Error in dyn.load("target/release/librustree.so"): ...
```

Solution: Check path and ensure built with `--features r`:
```bash
cargo build --release --no-default-features --features r
```

**2. Integer/numeric type errors**

```r
Error: argument must be integer
```

Solution: Use `L` suffix for integers:
```r
# Wrong:
simulate_species_tree(50, 1.0, 0.3)

# Correct:
simulate_species_tree(50L, 1.0, 0.3)
```

**3. thirdkind not found**

```r
Error: thirdkind binary not found
```

Solution: Install thirdkind:
```bash
cargo install thirdkind
```

**4. Simulation hangs**

Possible causes:
- Very large trees (>1000 species) with high DTL rates
- High loss rate with `require_extant = TRUE` (many retries)

Solutions:
- Reduce tree size or DTL rates
- Use batch simulation (more efficient)
- Remove `require_extant = TRUE`

**5. Symbol conflicts when loading both Python and R**

Error: Undefined symbols or crashes

Solution: Build separately with only the needed feature:
```bash
# For R only:
cargo build --release --no-default-features --features r

# For Python only:
cargo build --release --no-default-features --features python
```

---

## See Also

- **[Python Tutorial](PYTHON_TUTORIAL.md)** - Python bindings with comparison notes
- **[RecPhyloXML Format](RECPHYLOXML.md)** - Reconciliation file format specification
- **[LTT Plots](LTT_PLOTS.md)** - Lineages-through-time visualization
- **[Performance Profiling](PERFORMANCE_PROFILING.md)** - Optimization and benchmarking
- **rustree GitHub repository** - Source code, issues, and contributions

---

**Document Version:** 2.0
**Last Updated:** 2026-02-14
**rustree Version:** 0.1.0
**Project Root:** repository root
