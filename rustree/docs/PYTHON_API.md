# Rustree Python API Reference

Rustree is a high-performance Rust library for phylogenetic tree simulation and analysis, with Python bindings via PyO3. It provides:

- **Birth-Death (BD) species tree simulation** with speciation and extinction
- **DTL gene tree simulation** with Duplication, Transfer, and Loss events
- **RecPhyloXML parsing and export** for reconciled trees
- **Tree sampling and subtree extraction**
- **Pairwise distance computation** (topological and metric)
- **ALERax integration** for phylogenetic reconciliation

## Installation

```bash
# From the rustree directory, with a Python virtualenv active:
pip install maturin
maturin develop --features python
```

## Quick Start

```python
import rustree

# 1. Simulate a species tree (10 extant species)
species_tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.3, seed=42)

# 2. Simulate a gene tree along the species tree
gene_tree = species_tree.simulate_dtl(lambda_d=0.5, lambda_t=0.3, lambda_l=0.2, seed=42)

# 3. Inspect the results
print(f"Species: {species_tree.num_leaves()} leaves, {species_tree.num_nodes()} nodes")
print(f"Gene tree: {gene_tree.num_extant()} extant genes, {gene_tree.num_nodes()} nodes")
print(f"Events: {gene_tree.count_events()}")

# 4. Export
species_tree.save_newick("species.nwk")
gene_tree.save_newick("gene.nwk")
gene_tree.save_xml("reconciled.recphyloxml")
df = gene_tree.to_csv("gene_tree.csv")
```

---

## Module-Level Functions

### `simulate_species_tree(n, lambda_, mu, seed=None)`

Simulate a species tree under a birth-death process conditioned on `n` extant species.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Number of extant species (must be > 0) |
| `lambda_` | `float` | Speciation (birth) rate (must be > 0) |
| `mu` | `float` | Extinction (death) rate (must be >= 0 and < `lambda_`) |
| `seed` | `int`, optional | Random seed for reproducibility |

**Returns:** `PySpeciesTree`

```python
tree = rustree.simulate_species_tree(n=20, lambda_=1.0, mu=0.5, seed=42)
```

### `parse_species_tree(newick_str)`

Parse a Newick string or file into a species tree.

| Parameter | Type | Description |
|-----------|------|-------------|
| `newick_str` | `str` | Newick-formatted string, or path to a `.nwk` file |

**Returns:** `PySpeciesTree`

```python
# From string
tree = rustree.parse_species_tree("((A:1,B:1):0.5,C:1.5):0;")

# From file
tree = rustree.parse_species_tree("species.nwk")
```

### `parse_recphyloxml(filepath)`

Parse a RecPhyloXML file into a gene tree with reconciliation information.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str` | Path to a RecPhyloXML file |

**Returns:** `PyGeneTree`

```python
gene_tree = rustree.parse_recphyloxml("reconciliation.xml")
print(gene_tree.count_events())
```

### `reconcile_with_alerax(...)`

Run ALERax to reconcile gene trees with a species tree. See [ALERax Integration](#alerax-integration) for details.

---

## `PySpeciesTree`

Represents a species tree, either simulated via birth-death or parsed from Newick.

### Tree Information

#### `num_nodes() -> int`
Total number of nodes (internal + leaves).

#### `num_leaves() -> int`
Number of extant species (leaf nodes).

#### `tree_height() -> float`
Total tree height (depth of the deepest leaf from the root).

#### `root_index() -> int`
Index of the root node in the internal node array.

#### `leaf_names() -> list[str]`
Names of all leaf (extant species) nodes.

```python
tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.3, seed=42)
print(f"Leaves: {tree.leaf_names()}")
print(f"Height: {tree.tree_height():.4f}")
```

### Newick Export

#### `to_newick() -> str`
Convert to Newick format string.

#### `save_newick(filepath: str)`
Save the tree to a Newick file.

```python
newick = tree.to_newick()
tree.save_newick("species_tree.nwk")
```

### Subtree Extraction

#### `extract_induced_subtree_by_names(names: list[str]) -> PySpeciesTree`

Extract the induced subtree containing only the specified leaf names and their MRCAs. Branch lengths are preserved.

| Parameter | Type | Description |
|-----------|------|-------------|
| `names` | `list[str]` | Leaf names to keep |

**Returns:** A new `PySpeciesTree` containing only the selected species.

```python
tree = rustree.simulate_species_tree(n=10, lambda_=1.0, mu=0.3, seed=42)
leaves = tree.leaf_names()
subtree = tree.extract_induced_subtree_by_names(leaves[:5])
print(f"Subtree: {subtree.num_leaves()} leaves")
```

### Birth-Death Events

#### `save_bd_events_csv(filepath: str)`
Save birth-death events to a CSV file with columns: `time`, `node_name`, `event_type`, `child1_name`, `child2_name`.

#### `get_bd_events() -> dict`
Get birth-death events as a dictionary (suitable for `pd.DataFrame()`).

Keys: `time`, `node_name`, `event_type`, `child1_name`, `child2_name`.

```python
import pandas as pd
events = tree.get_bd_events()
df = pd.DataFrame(events)
print(df.head())
```

### Lineages Through Time (LTT)

#### `get_ltt_data() -> dict`
Get LTT data as a dictionary with keys `times` and `lineages`.

#### `plot_ltt(filepath=None, title=None, xlabel=None, ylabel=None)`
Plot LTT curve using matplotlib. If `filepath` is given, saves to file; otherwise displays interactively.

```python
tree.plot_ltt(filepath="ltt.png", title="My LTT Plot")
```

### Pairwise Distances

#### `pairwise_distances(distance_type: str, leaves_only=True) -> pd.DataFrame`

Compute pairwise distances between nodes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `distance_type` | `str` | `"topological"` (edge count) or `"metric"` (sum of branch lengths) |
| `leaves_only` | `bool` | If `True`, only compute between leaves (default: `True`) |

**Returns:** pandas DataFrame with columns `node1`, `node2`, `distance`.

#### `save_pairwise_distances_csv(filepath, distance_type, leaves_only=True)`
Save pairwise distances directly to CSV (same columns as above).

```python
df = tree.pairwise_distances("metric", leaves_only=True)
tree.save_pairwise_distances_csv("distances.csv", "topological")
```

### Visualization (requires `thirdkind`)

#### `to_svg(filepath=None, open_browser=False, internal_names=True) -> str`
Generate SVG visualization using thirdkind. Returns SVG content as string.

#### `display(internal_names=True)`
Display the tree in a Jupyter notebook (requires IPython).

```python
# Save to file
tree.to_svg("species_tree.svg")

# In Jupyter
tree.display()
```

### DTL Gene Tree Simulation

#### `simulate_dtl(lambda_d, lambda_t, lambda_l, ...) -> PyGeneTree`

Simulate a gene tree along this species tree using the **per-gene-copy** DTL model. Event rates scale with the number of gene copies.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lambda_d` | `float` | Duplication rate per gene copy per unit time |
| `lambda_t` | `float` | Transfer rate per gene copy per unit time |
| `lambda_l` | `float` | Loss rate per gene copy per unit time |
| `transfer_alpha` | `float`, optional | Distance decay parameter for assortative transfers. `None` = uniform random recipient (default). Higher values = prefer closer species. |
| `replacement_transfer` | `float`, optional | Probability that a transfer replaces the recipient gene (0.0-1.0). `None` or `0.0` = additive transfers only. `1.0` = all transfers are replacements. |
| `require_extant` | `bool` | If `True`, retry simulation until at least one gene survives to present (default: `False`) |
| `seed` | `int`, optional | Random seed |

**Returns:** `PyGeneTree`

```python
gene = tree.simulate_dtl(
    lambda_d=0.5, lambda_t=0.3, lambda_l=0.2,
    transfer_alpha=1.0,        # assortative transfers
    replacement_transfer=0.5,  # 50% replacement transfers
    require_extant=True,
    seed=42
)
```

#### `simulate_dtl_batch(n, lambda_d, lambda_t, lambda_l, ...) -> list[PyGeneTree]`

Simulate `n` gene trees efficiently. Pre-computes shared data (depths, contemporaneity) once. Same parameters as `simulate_dtl` plus `n`.

```python
gene_trees = tree.simulate_dtl_batch(n=100, lambda_d=0.5, lambda_t=0.3, lambda_l=0.2, seed=42)
```

#### `simulate_dtl_per_species(lambda_d, lambda_t, lambda_l, ...) -> PyGeneTree`

Simulate a gene tree using the **per-species** DTL model (Zombi-style). Event rates scale with the number of *species hosting genes*, not the number of gene copies. This means duplications do not increase the event rate.

Same parameters as `simulate_dtl`.

```python
# Per-species model typically produces fewer events than per-gene-copy
gene_per_species = tree.simulate_dtl_per_species(0.5, 0.3, 0.2, seed=42)
```

#### `simulate_dtl_per_species_batch(n, lambda_d, lambda_t, lambda_l, ...) -> list[PyGeneTree]`

Batch version of `simulate_dtl_per_species`. Same parameters plus `n`.

---

## `PyGeneTree`

Represents a reconciled gene tree with its mapping to a species tree and event annotations.

### Tree Information

#### `num_nodes() -> int`
Total number of nodes (internal + leaves, including loss nodes).

#### `num_extant() -> int`
Number of extant genes (leaf nodes with `Leaf` events, excluding losses).

#### `count_events() -> tuple[int, int, int, int, int]`
Count events by type. Returns `(speciations, duplications, transfers, losses, leaves)`.

#### `extant_gene_names() -> list[str]`
Names of all extant genes (format: `"speciesname_N"`).

```python
gene = tree.simulate_dtl(0.5, 0.3, 0.2, seed=42)
s, d, t, l, leaves = gene.count_events()
print(f"S={s}, D={d}, T={t}, L={l}, Leaves={leaves}")
print(f"Extant genes: {gene.extant_gene_names()}")
```

### Export

#### `to_newick() -> str`
Convert the gene tree to Newick format.

#### `save_newick(filepath: str)`
Save gene tree to a Newick file.

#### `to_xml() -> str`
Export the reconciled tree (species + gene + reconciliation) to RecPhyloXML format.

#### `save_xml(filepath: str)`
Save RecPhyloXML to a file.

#### `to_csv(filepath=None) -> pd.DataFrame`

Export gene tree node data as a pandas DataFrame. Optionally saves to CSV file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | `str`, optional | Path to save CSV file |

**Returns:** pandas DataFrame with 13 columns:

| Column | Description |
|--------|-------------|
| `node_id` | Node index |
| `name` | Node name |
| `parent` | Parent node index (empty for root) |
| `left_child` | Left child node index (empty for leaves) |
| `left_child_name` | Name of the left child node |
| `right_child` | Right child node index (empty for leaves) |
| `right_child_name` | Name of the right child node |
| `length` | Branch length |
| `depth` | Node depth from root |
| `species_node` | Name of the mapped species tree node |
| `species_node_left` | Species node of the left child |
| `species_node_right` | Species node of the right child |
| `event` | Event type: `Speciation`, `Duplication`, `Transfer`, `Loss`, or `Leaf` |

```python
df = gene.to_csv("gene_tree_data.csv")
print(df[['name', 'species_node', 'event']].head(10))
```

### Pairwise Distances

#### `pairwise_distances(distance_type, leaves_only=True) -> pd.DataFrame`
Same as `PySpeciesTree.pairwise_distances()` but for the gene tree.

#### `save_pairwise_distances_csv(filepath, distance_type, leaves_only=True)`
Same as `PySpeciesTree.save_pairwise_distances_csv()` but for the gene tree.

### Sampling

#### `sample_extant() -> PyGeneTree`

Extract the induced subtree of extant genes only (removes loss nodes and dead lineages). Internal nodes after sampling have `None` species mapping.

```python
full_tree = tree.simulate_dtl(0.5, 0.3, 0.2, seed=42)
extant_tree = full_tree.sample_extant()
print(f"Full: {full_tree.num_nodes()} nodes -> Extant: {extant_tree.num_nodes()} nodes")
```

#### `sample_by_names(names: list[str]) -> PyGeneTree`

Extract the induced subtree containing only genes with the given names.

| Parameter | Type | Description |
|-----------|------|-------------|
| `names` | `list[str]` | Gene names to keep |

```python
all_genes = full_tree.extant_gene_names()
subset = full_tree.sample_by_names(all_genes[:3])
```

#### `sample_species_leaves(species_leaf_names: list[str]) -> PyGeneTree`

Sample a subset of species and filter the gene tree accordingly. Both the species tree and gene tree are reduced. Reconciliation mappings are preserved via LCA-based remapping.

| Parameter | Type | Description |
|-----------|------|-------------|
| `species_leaf_names` | `list[str]` | Species leaf names to keep |

```python
species_names = tree.leaf_names()[:5]
sampled = gene.sample_species_leaves(species_names)
print(f"Sampled gene tree: {sampled.num_nodes()} nodes")
```

### Visualization (requires `thirdkind`)

#### `to_svg(filepath=None, open_browser=False) -> str`
Generate SVG visualization of the reconciled tree (gene tree embedded in species tree).

#### `display()`
Display in Jupyter notebook.

```python
gene.to_svg("reconciled.svg")
gene.display()  # in Jupyter
```

---

## ALERax Integration

### `reconcile_with_alerax(...) -> dict[str, PyAleRaxResult]`

Reconcile gene trees with a species tree using [ALERax](https://github.com/BenoitMorel/AleRax). Requires ALERax to be installed and available in PATH.

| Parameter | Type | Description |
|-----------|------|-------------|
| `species_tree` | `PySpeciesTree` or `str` | Species tree (object, Newick string, or file path) |
| `gene_trees` | `str`, `list[str]`, or `dict[str, str]` | Gene tree(s) as Newick strings or file paths. Dict maps family names to trees. |
| `output_dir` | `str`, optional | Output directory (default: temp dir) |
| `num_samples` | `int` | Number of reconciliation samples (default: 100) |
| `model` | `str` | Model parametrization: `"PER-FAMILY"` or `"GLOBAL"` (default: `"PER-FAMILY"`) |
| `seed` | `int`, optional | Random seed |
| `keep_output` | `bool` | Preserve ALERax output files (default: `False`) |
| `alerax_path` | `str` | Path to ALERax executable (default: `"alerax"`) |

**Returns:** `dict[str, PyAleRaxResult]` mapping family names to results.

```python
species = rustree.parse_species_tree("species.nwk")
results = rustree.reconcile_with_alerax(
    species_tree=species,
    gene_trees="gene_family.nwk",
    seed=42
)

result = results["gene_family"]
print(f"D={result.duplication_rate:.4f}, T={result.transfer_rate:.4f}, L={result.loss_rate:.4f}")
print(f"Log-likelihood: {result.likelihood:.2f}")
print(f"Samples: {len(result.gene_trees)}")
```

### `PyAleRaxResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `gene_trees` | `list[PyGeneTree]` | Reconciliation samples (typically 100) |
| `duplication_rate` | `float` | Estimated duplication rate |
| `loss_rate` | `float` | Estimated loss rate |
| `transfer_rate` | `float` | Estimated transfer rate |
| `likelihood` | `float` | Log-likelihood of the reconciliation |
| `statistics` | `PyReconciliationStatistics` | Summary statistics |

### `PyReconciliationStatistics`

| Attribute | Type | Description |
|-----------|------|-------------|
| `mean_event_counts` | `PyEventCounts` | Mean counts across all samples |
| `mean_transfers` | `dict[str, dict[str, float]]` | Mean transfers between species pairs: `{source: {dest: count}}` |
| `events_per_species` | `dict[str, PyEventCounts]` | Mean events per species node |

#### `transfers_df() -> pd.DataFrame`
Mean transfers as DataFrame with columns: `source`, `destination`, `mean_count`.

#### `events_df() -> pd.DataFrame`
Per-species events as DataFrame with columns: `species`, `speciations`, `speciation_losses`, `duplications`, `duplication_losses`, `transfers`, `transfer_losses`, `losses`, `leaves`.

### `PyEventCounts`

| Attribute | Type | Description |
|-----------|------|-------------|
| `speciations` | `float` | Mean speciation count |
| `speciation_losses` | `float` | Mean speciation-loss count |
| `duplications` | `float` | Mean duplication count |
| `duplication_losses` | `float` | Mean duplication-loss count |
| `transfers` | `float` | Mean transfer count |
| `transfer_losses` | `float` | Mean transfer-loss count |
| `losses` | `float` | Mean loss count |
| `leaves` | `float` | Mean leaf count |

---

## DTL Model Details

### Per-Gene-Copy Model (`simulate_dtl`)

Events occur at a rate proportional to the **number of gene copies**. If there are `k` copies of a gene across species, the total event rate is `k * (lambda_d + lambda_t + lambda_l)`. Duplications create new copies, increasing the rate.

### Per-Species Model (`simulate_dtl_per_species`)

Events occur at a rate proportional to the **number of species hosting at least one gene copy**. Multiple copies in the same species do not increase the rate. When an event occurs, a random species is chosen, then a random gene copy within that species is affected. This is the model used by Zombi.

### Assortative Transfers (`transfer_alpha`)

When `transfer_alpha` is set, transfer recipients are chosen with probability weighted by phylogenetic distance:

```
P(recipient) ~ exp(-alpha * distance(donor, recipient))
```

- `alpha = 0`: Equivalent to uniform (no preference)
- `alpha > 0`: Prefer closer species
- Higher alpha = stronger preference for close relatives

### Replacement Transfers (`replacement_transfer`)

When `replacement_transfer` is set (0.0 to 1.0), each transfer has a probability of being a **replacement transfer**: the transferred gene replaces an existing copy in the recipient species (causing a loss in the recipient), rather than being added alongside existing copies.

- `None` or `0.0`: All transfers are additive
- `1.0`: All transfers are replacements
- `0.5`: 50% chance of replacement per transfer

---

## CSV Formats

### Gene Tree CSV (`to_csv`)

13 columns: `node_id`, `name`, `parent`, `left_child`, `left_child_name`, `right_child`, `right_child_name`, `length`, `depth`, `species_node`, `species_node_left`, `species_node_right`, `event`

### BD Events CSV (`save_bd_events_csv`)

5 columns: `time`, `node_name`, `event_type`, `child1_name`, `child2_name`

### Pairwise Distances CSV (`save_pairwise_distances_csv`)

3 columns: `node1`, `node2`, `distance`

---

## External Dependencies

| Tool | Purpose | Installation |
|------|---------|-------------|
| `thirdkind` | SVG tree visualization | `cargo install thirdkind` |
| `ALERax` | Phylogenetic reconciliation | See [ALERax GitHub](https://github.com/BenoitMorel/AleRax) |
| `pandas` | DataFrame creation | `pip install pandas` |
| `matplotlib` | LTT plotting | `pip install matplotlib` |
| `IPython` | Jupyter notebook display | `pip install ipython` |
