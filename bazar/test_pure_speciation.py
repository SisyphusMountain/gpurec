#!/usr/bin/env python3
"""
Test pure speciation case to understand the expected behavior.
"""

from ete3 import Tree
import torch
import math

# Load trees
sp_tree = Tree("test_trees_1/sp.nwk", format=1)
g_tree = Tree("test_trees_1/g.nwk", format=1)

print("Species tree:")
print(sp_tree)
print("\nGene tree:")
print(g_tree)

# In pure speciation with matching topologies, what should the likelihood be?
# Each gene leaf maps to exactly one species leaf
# Each internal gene node maps to exactly one species node

# Count nodes
sp_leaves = sp_tree.get_leaves()
g_leaves = g_tree.get_leaves()
print(f"\nSpecies leaves: {len(sp_leaves)}")
print(f"Gene leaves: {len(g_leaves)}")

# Check if topologies match
sp_leaf_names = sorted([n.name for n in sp_leaves])
g_leaf_names = sorted([n.name.split('_')[0] for n in g_leaves])
print(f"\nSpecies leaf names: {sp_leaf_names}")
print(f"Gene leaf names (species part): {g_leaf_names}")
print(f"Topologies match: {sp_leaf_names == g_leaf_names}")

# In pure speciation, the probability of the gene tree given the species tree
# should be 1.0 if they have the same topology (up to leaf labeling)

# For an unrooted gene tree with n leaves, there are 2n-3 possible rootings
n_leaves = len(g_leaves)
n_rootings = 2 * n_leaves - 3
print(f"\nNumber of possible rootings: {n_rootings}")

# If we average over all rootings, and only one matches perfectly,
# the likelihood would be 1/n_rootings
likelihood_one_matching = 1.0 / n_rootings
log_lik_one_matching = math.log(likelihood_one_matching)
print(f"\nIf only one rooting matches: ")
print(f"  Likelihood = {likelihood_one_matching:.6f}")
print(f"  Log-likelihood = {log_lik_one_matching:.6f}")

# But wait, in the CCP framework, we're considering all possible rootings
# and their reconciliations. The total likelihood is the sum over all rootings.

# With uniform origination probabilities and no extinction,
# the likelihood formula simplifies significantly.

# Check number of species branches
n_species_branches = 2 * len(sp_leaves) - 1  # For a rooted binary tree
print(f"\nNumber of species branches: {n_species_branches}")

# The AleRax likelihood includes normalization by S (number of species branches)
# Let's compute what we'd expect
print(f"\nWith S normalization factor:")
likelihood_with_S = likelihood_one_matching * n_species_branches
log_lik_with_S = math.log(likelihood_with_S)
print(f"  Likelihood = {likelihood_with_S:.6f}")
print(f"  Log-likelihood = {log_lik_with_S:.6f}")

# But -2.56495 is quite negative, suggesting a likelihood of about 0.077
target_loglik = -2.56495
target_lik = math.exp(target_loglik)
print(f"\nAleRax target:")
print(f"  Log-likelihood = {target_loglik}")
print(f"  Likelihood = {target_lik:.6f}")

# This is roughly 1/13, which is 1/n_rootings!
print(f"\n1/13 = {1/13:.6f}")
print(f"AleRax likelihood / (1/13) = {target_lik / (1/13):.6f}")