dyn.load("target/release/librustree.so")
source("R/rustree.R")
sp_tree <- simulate_species_tree(10L, 1.0, 0.2, 41L)
gene_trees_tmax <- simulate_dtl(
  species_tree = sp_tree,
  lambda_d = 0.01,
  lambda_t = 0.1,
  lambda_l = 0.01,
  seed = 123L
)