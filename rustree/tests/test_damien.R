# Load the library (adjust path as needed)
dyn.load("target/release/librustree.so")

# Source the wrapper functions
source("R/rustree.R")

# Simulate a species tree with 50 extant species
sp_tree <- simulate_species_tree(50L, 1.0, 0.3, 42L)

# Check the tree
cat("Number of leaves:", tree_num_leaves(sp_tree), "\n")
cat("Newick:", tree_to_newick(sp_tree), "\n")

# NEW: Access birth-death events
cat("BD Events:", nrow(as.data.frame(sp_tree$events)), "\n")
table(sp_tree$events$event_type)

# Export birth-death events
save_bd_events_csv(sp_tree, "bd_events.csv")

# Simulate gene trees with DTL events
gene_trees <- simulate_dtl_batch(sp_tree, 10L, 0.5, 0.2, 0.3, seed = 123L)

# Work with the first gene tree
gt <- gene_trees[[1]]
cat("Number of extant genes:", gene_tree_num_extant(gt), "\n")

# Export gene tree events
save_csv(gt, "gene_tree.csv")

# Inspect the tree
tree_num_leaves(sp_tree)      # 100
tree_leaf_names(sp_tree)      # Vector of leaf names
tree_to_newick(sp_tree)       # Newick string


# Ensure all gene trees have at least one extant gene
# (useful when losses are high and some trees might go completely extinct)
gene_tree <- simulate_dtl(
  species_tree = sp_tree,
  lambda_d = 0.5,
  lambda_t = 0.2,
  lambda_l = 0.8,    # High loss rate
  require_extant = TRUE,  # Retry until we get an extant gene
  seed = 123L
)

# Works with batch too - only keeps trees with extant genes
gene_trees <- simulate_dtl_batch(
  species_tree = sp_tree,
  n = 100L,
  lambda_d = 0.5,
  lambda_t = 0.2,
  lambda_l = 1,
  require_extant = TRUE,  # All returned trees have >= 1 extant gene
  seed = 123L
)

# Access individual trees
gt1 <- gene_trees[[1]]
gt2 <- gene_trees[[2]]
