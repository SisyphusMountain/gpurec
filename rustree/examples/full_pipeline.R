# ============================================================================
# rustree R pipeline: species tree -> gene trees -> pruning -> reconciliation
# ============================================================================

# --- Setup ---
dyn.load("target/release/librustree.so")  # Linux
# dyn.load("target/release/librustree.dylib")  # macOS
source("R/rustree.R")

# ============================================================================
# Step 1: Simulate a species tree (conditional birth-death process)
#         Conditioned on n extant species; backward-in-time algorithm
# ============================================================================
sp_tree <- simulate_species_tree(
  n      = 50L,    # number of extant species (integer!)
  lambda = 1.0,    # speciation (birth) rate
  mu     = 0.3,    # extinction (death) rate  (must be < lambda)
  seed   = 42L     # reproducibility (NULL for random)
)

cat("Species tree:", tree_num_leaves(sp_tree), "extant leaves\n")
cat("Total nodes (incl. extinct):", length(sp_tree$name), "\n")

# Export the complete species tree (with extinct lineages)
save_newick(sp_tree, "/tmp/species_complete.nwk")

# ============================================================================
# Step 2: Simulate gene trees under a DTL process (batched)
#         Two models available: per-gene-copy (default) or per-species (Zombi)
# ============================================================================

# --- Option A: Per-gene-copy model (rate proportional to number of gene copies) ---
gene_trees <- simulate_dtl_batch(
  species_tree = sp_tree,
  n            = 100L,   # number of gene families to simulate
  lambda_d     = 0.5,    # duplication rate
  lambda_t     = 0.2,    # transfer rate
  lambda_l     = 0.3,    # loss rate
  transfer_alpha = NULL, # NULL = uniform transfers; numeric = assortative
  require_extant = FALSE,# if TRUE, retry until at least 1 extant gene
  seed         = 123L
)

# --- Option B: Per-species model (rate proportional to number of alive species) ---
gene_trees_ps <- simulate_dtl_per_species_batch(
  species_tree = sp_tree,
  n            = 100L,
  lambda_d     = 0.5,
  lambda_t     = 0.2,
  lambda_l     = 0.3,
  seed         = 123L
)

# Quick summary of gene family sizes
extant_counts <- sapply(gene_trees, gene_tree_num_extant)
cat("Extant genes per family -- mean:", mean(extant_counts),
    "range:", min(extant_counts), "-", max(extant_counts), "\n")

# ============================================================================
# Step 3: Prune to extant leaves (or a subset of extant species)
# ============================================================================

# --- Option A: Keep only extant genes in a single gene tree ---
gt <- gene_trees[[1]]
gt_extant <- sample_extant(gt)
cat("Before pruning:", length(gt$name), "nodes\n")
cat("After pruning:", length(gt_extant$name), "nodes\n")

# --- Option B: Keep genes mapping to a subset of species ---
#     This jointly prunes the species tree and remaps the gene tree
all_species <- tree_leaf_names(sp_tree)
selected_species <- all_species[1:20]  # keep 20 of 50 species

gt_sampled <- sample_leaves(gt, selected_species)
cat("Genes in sampled species:", gene_tree_num_extant(gt_sampled), "\n")

# You can also prune the species tree alone
sp_subtree <- extract_induced_subtree_by_names(sp_tree, selected_species)
cat("Pruned species tree:", tree_num_leaves(sp_subtree), "leaves\n")

# ============================================================================
# Step 4: Compute induced transfers
#         Projects each transfer's donor/recipient from the complete species
#         tree onto the sampled subtree. This tells you which sampled species
#         branches the transfers map to after pruning.
# ============================================================================

# DTL events are attached to each gene tree as an attribute
dtl_events <- get_dtl_events(gt)
cat("\nDTL events from simulation:\n")
print(table(dtl_events$event_type))

# Compute induced transfers for the sampled species subset
induced <- induced_transfers(sp_tree, selected_species, dtl_events)
cat("\nInduced transfers (", nrow(induced), "transfers ):\n")
print(head(induced))

# The reconciliation event labels on the gene tree nodes:
cat("\nReconciliation events (full gene tree):\n")
print(table(gt$event))

# ============================================================================
# Export results
# ============================================================================

# RecPhyloXML -- standard format for reconciled gene trees
save_xml(gt_sampled, "/tmp/gene_family_sampled.xml")

# CSV -- tabular node info with species mapping and events
save_csv(gt_sampled, "/tmp/gene_family_sampled.csv")

# Newick -- topology + branch lengths only
save_newick(gt_sampled, "/tmp/gene_family_sampled.nwk")

# SVG visualization (requires: cargo install thirdkind)
# Note: only works on full (unpruned) gene trees
gene_tree_to_svg(gt, "/tmp/gene_family.svg")

# ============================================================================
# For large-scale runs: streaming to disk (one tree in memory at a time)
# ============================================================================
newick <- tree_to_newick(sp_tree)
simulate_dtl_stream_xml(
  newick, 1000L,
  lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
  seed = 42L,
  output_dir = "/tmp/gene_trees_xml"
)
# Creates /tmp/gene_trees_xml/gene_0000.xml, gene_0001.xml, ...
