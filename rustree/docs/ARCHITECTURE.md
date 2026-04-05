# Rustree Architecture

## Overview

Rustree is a phylogenetic tree simulation and reconciliation library written in Rust, with Python (PyO3) and R (extendr) bindings.

## Module Map

```
src/
├── lib.rs                    # Public API facade
│
├── node/                     # Core tree data structures
│   ├── mod.rs                # FlatTree, FlatNode, Node
│   ├── iter.rs               # Tree traversal iterators
│   ├── traits.rs             # Shared traits (TreeNode)
│   ├── conversion.rs         # FlatNode <-> Node, remapping
│   ├── rectree.rs            # RecTree (reconciled gene tree)
│   └── gene_forest.rs        # GeneForest (multi-gene-tree container)
│
├── newick/                   # Newick format parsing (pest grammar)
│   ├── parser.rs             # parse_newick()
│   └── newick.pest           # PEG grammar
│
├── simulation/               # Tree generation
│   ├── bd/                   # Birth-Death process
│   │   ├── simulation.rs     # simulate_bd_tree_bwd()
│   │   ├── events.rs         # Event extraction
│   │   └── types.rs          # BDEvent, TreeEvent
│   └── dtl/                  # DTL (Duplication-Transfer-Loss)
│       ├── gillespie.rs      # Shared Gillespie loop
│       ├── per_gene.rs       # Per-gene-copy model
│       ├── per_species.rs    # Per-species model (Zombi-style)
│       ├── stream.rs         # DtlSimIter (lazy iterator)
│       ├── state.rs          # SimulationState
│       ├── event.rs          # DTLEvent
│       └── utils.rs          # Event counting, transfer selection
│
├── io/                       # I/O and serialization
│   ├── csv.rs                # CSV event export
│   ├── recphyloxml.rs        # RecPhyloXML parsing
│   ├── rectree_xml.rs        # RecTree XML serialization
│   └── rectree_csv.rs        # RecTree CSV I/O
│
├── sampling.rs               # Induced subtree extraction, LCA
├── comparison.rs             # Reconciliation comparison metrics
├── metric_functions.rs       # Pairwise distances, LCA tables
├── robinson_foulds.rs        # RF distance
├── surgery.rs                # SPR topology operations
├── induced_transfers.rs      # Transfer projection onto sampled trees
│
├── bindings_common/          # Shared validation for Python/R
│
├── python/                   # PyO3 bindings
│   ├── mod.rs                # Module registration
│   ├── species_tree.rs       # PySpeciesTree
│   ├── gene_tree.rs          # PyGeneTree
│   ├── training.rs           # ML tensor construction
│   ├── reconciliation.rs     # Comparison bindings
│   ├── alerax.rs             # ALERax bindings
│   ├── forest.rs             # GeneForest bindings
│   ├── sim_iter.rs           # Streaming DTL iterator
│   └── types.rs              # Type wrappers
│
├── r/                        # extendr R bindings
│   ├── mod.rs                # Module + extendr_module! macro + tests
│   ├── conversions.rs        # R <-> Rust data conversion
│   ├── species.rs            # Species/gene tree operations
│   ├── dtl.rs                # DTL simulation
│   ├── io.rs                 # File I/O
│   ├── viz.rs                # SVG visualization
│   ├── sampling.rs           # Sampling operations
│   ├── analysis.rs           # Metrics, parsing, induced transfers
│   ├── streaming.rs          # Streaming DTL to files
│   └── helpers.rs            # Internal helpers (RNG, parsing)
│
└── external/                 # External tool integration
    └── alerax.rs             # ALERax reconciliation
```

## Feature Flags

| Feature   | Enables           | Crate Type |
|-----------|-------------------|------------|
| (default) | Core library only | `rlib`     |
| `python`  | PyO3 + NumPy      | `cdylib`   |
| `r`       | extendr           | `cdylib`   |

## Data Flow

```
Newick string ──parse_newick()──> Node (recursive)
                                    │
                                    ├──to_flat_tree()──> FlatTree (indexed)
                                    │
simulate_bd_tree_bwd() ────────────>├──assign_depths()
                                    │
simulate_dtl() ────────────────────>├──> RecTree (gene + species + mapping)
                                    │
                           RecPhyloXML ──parse_recphyloxml()──> RecTree
```

## Error Handling

Currently uses `Result<T, String>` throughout. A unified `RustreeError` type is planned (see CODE_REVIEW_ACTION_PLAN.md).

## Key Design Decisions

- **FlatTree** is the primary representation: contiguous memory, cache-friendly traversal
- **Node** is used for parsing/construction, then converted to FlatTree
- **DTL simulation** uses a shared Gillespie loop with mode dispatch (per-gene vs per-species)
- **R bindings** use `include!()` to keep all `#[extendr]` items in one module scope (required by `extendr_module!` macro)
- **Python bindings** are proper submodules since PyO3 doesn't have this constraint
