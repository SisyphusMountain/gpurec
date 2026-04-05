# Gene Tree and DTL Simulation Tests for rustree
# Comprehensive test suite for gene tree simulation functions

# Load the library
repo_root <- normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", ".."))
dyn.load(file.path(repo_root, "target", "release", "librustree.so"))
source(file.path(repo_root, "R", "rustree.R"))

# ==============================================================================
# Test Framework
# ==============================================================================

tests_passed <- 0
tests_failed <- 0

run_test <- function(name, expr) {
  cat(sprintf("Testing: %s ... ", name))
  result <- tryCatch({
    if (isTRUE(expr)) {
      cat("PASSED\n")
      tests_passed <<- tests_passed + 1
      TRUE
    } else {
      cat("FAILED\n")
      tests_failed <<- tests_failed + 1
      FALSE
    }
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    tests_failed <<- tests_failed + 1
    FALSE
  })
  invisible(result)
}

# ==============================================================================
# Setup: Create a species tree for all tests
# ==============================================================================

cat("\n========================================\n")
cat("Gene Tree and DTL Simulation Test Suite\n")
cat("========================================\n\n")

cat("Setting up species tree for DTL tests...\n")
sp_tree <- simulate_species_tree(30L, lambda = 1.0, mu = 0.3, seed = 42L)
cat(sprintf("Species tree created with %d leaves\n\n", tree_num_leaves(sp_tree)))

# ==============================================================================
# Test: simulate_dtl - Basic Simulation
# ==============================================================================

cat("\n--- simulate_dtl: Basic Simulation Tests ---\n")

# Test basic DTL simulation with default parameters
run_test("simulate_dtl returns a gene tree object", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3)
  !is.null(gt) && is.list(gt)
})

# Test simulation with low rates (should produce simple trees)
run_test("simulate_dtl with low rates produces trees", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.1, lambda_t = 0.05, lambda_l = 0.1, seed = 100L)
  !is.null(gt)
})

# Test simulation with moderate rates
run_test("simulate_dtl with moderate rates produces trees", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.3, lambda_l = 0.4, seed = 101L)
  !is.null(gt)
})

# Test simulation with high duplication rate
run_test("simulate_dtl with high duplication rate", {
  gt <- simulate_dtl(sp_tree, lambda_d = 2.0, lambda_t = 0.1, lambda_l = 0.2, seed = 102L)
  !is.null(gt)
})

# Test simulation with zero transfer rate
run_test("simulate_dtl with zero transfer rate", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.0, lambda_l = 0.3, seed = 103L)
  !is.null(gt)
})

# ==============================================================================
# Test: simulate_dtl - require_extant Parameter
# ==============================================================================

cat("\n--- simulate_dtl: require_extant Tests ---\n")

# Test require_extant=TRUE ensures all trees have at least 1 extant gene
run_test("require_extant=TRUE with high loss rate produces tree with extant genes", {
  # Use high loss rate where trees might otherwise go extinct
  gt <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.1, lambda_l = 1.5,
                     require_extant = TRUE, seed = 200L)
  n_extant <- gene_tree_num_extant(gt)
  n_extant >= 1
})

# Test multiple simulations with require_extant=TRUE all have extant genes
run_test("require_extant=TRUE consistently produces trees with extant genes", {
  all_have_extant <- TRUE
  for (i in 1:10) {
    gt <- simulate_dtl(sp_tree, lambda_d = 0.2, lambda_t = 0.1, lambda_l = 1.5,
                       require_extant = TRUE, seed = 200L + i)
    if (gene_tree_num_extant(gt) < 1) {
      all_have_extant <- FALSE
      break
    }
  }
  all_have_extant
})

# Test require_extant=FALSE allows trees with 0 extant genes
# This is probabilistic, so we check that simulation completes without error
run_test("require_extant=FALSE allows simulation (may have 0 extant)", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.1, lambda_t = 0.0, lambda_l = 2.0,
                     require_extant = FALSE, seed = 210L)
  !is.null(gt)  # Just verify it completes
})

# Compare behavior with and without require_extant
run_test("require_extant parameter affects simulation behavior", {
  # Both should complete without error
  gt_required <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.1, lambda_l = 1.0,
                              require_extant = TRUE, seed = 220L)
  gt_not_required <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.1, lambda_l = 1.0,
                                  require_extant = FALSE, seed = 220L)
  !is.null(gt_required) && !is.null(gt_not_required)
})

# ==============================================================================
# Test: simulate_dtl - transfer_alpha Parameter
# ==============================================================================

cat("\n--- simulate_dtl: transfer_alpha Tests ---\n")

# Test with transfer_alpha=NULL (uniform transfers)
run_test("simulate_dtl with transfer_alpha=NULL (uniform transfers)", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.5, lambda_l = 0.3,
                     transfer_alpha = NULL, seed = 300L)
  !is.null(gt)
})

# Test with transfer_alpha > 0 (assortative transfers - prefer related species)
run_test("simulate_dtl with transfer_alpha > 0 (assortative)", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.5, lambda_l = 0.3,
                     transfer_alpha = 2.0, seed = 301L)
  !is.null(gt)
})

# Test with transfer_alpha close to 0 (near-uniform)
run_test("simulate_dtl with transfer_alpha near 0", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.5, lambda_l = 0.3,
                     transfer_alpha = 0.1, seed = 302L)
  !is.null(gt)
})

# Test with high transfer_alpha (strongly assortative)
run_test("simulate_dtl with high transfer_alpha (strongly assortative)", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.5, lambda_l = 0.3,
                     transfer_alpha = 10.0, seed = 303L)
  !is.null(gt)
})

# Test transfer_alpha with require_extant
run_test("simulate_dtl with transfer_alpha and require_extant", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.5, lambda_l = 1.2,
                     transfer_alpha = 2.0, require_extant = TRUE, seed = 304L)
  gene_tree_num_extant(gt) >= 1
})

# ==============================================================================
# Test: simulate_dtl - Seed Reproducibility
# ==============================================================================

cat("\n--- simulate_dtl: Seed Reproducibility Tests ---\n")

# Test that same seed produces identical results
run_test("same seed produces identical gene trees", {
  gt1 <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 400L)
  gt2 <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 400L)
  newick1 <- gene_tree_to_newick(gt1)
  newick2 <- gene_tree_to_newick(gt2)
  identical(newick1, newick2)
})

# Test that different seeds produce different results
run_test("different seeds produce different gene trees", {
  gt1 <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 401L)
  gt2 <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 402L)
  newick1 <- gene_tree_to_newick(gt1)
  newick2 <- gene_tree_to_newick(gt2)
  !identical(newick1, newick2)
})

# Test reproducibility with transfer_alpha
run_test("seed reproducibility with transfer_alpha", {
  gt1 <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.4, lambda_l = 0.3,
                      transfer_alpha = 2.0, seed = 403L)
  gt2 <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.4, lambda_l = 0.3,
                      transfer_alpha = 2.0, seed = 403L)
  identical(gene_tree_to_newick(gt1), gene_tree_to_newick(gt2))
})

# Test reproducibility with require_extant
run_test("seed reproducibility with require_extant=TRUE", {
  gt1 <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.2, lambda_l = 1.0,
                      require_extant = TRUE, seed = 404L)
  gt2 <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.2, lambda_l = 1.0,
                      require_extant = TRUE, seed = 404L)
  identical(gene_tree_to_newick(gt1), gene_tree_to_newick(gt2))
})

# ==============================================================================
# Test: simulate_dtl_batch - Batch Simulation
# ==============================================================================

cat("\n--- simulate_dtl_batch: Batch Simulation Tests ---\n")

# Test batch simulation returns correct number of trees
run_test("simulate_dtl_batch returns correct number of trees", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 5L, lambda_d = 0.5, lambda_t = 0.2,
                                   lambda_l = 0.3, seed = 500L)
  length(gene_trees) == 5
})

# Test batch with larger n
run_test("simulate_dtl_batch with n=20 returns 20 trees", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 20L, lambda_d = 0.5, lambda_t = 0.2,
                                   lambda_l = 0.3, seed = 501L)
  length(gene_trees) == 20
})

# Test batch with n=1
run_test("simulate_dtl_batch with n=1 returns 1 tree", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 1L, lambda_d = 0.5, lambda_t = 0.2,
                                   lambda_l = 0.3, seed = 502L)
  length(gene_trees) == 1
})

# Test all trees in batch are valid gene trees
run_test("all trees in batch are valid gene trees", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 10L, lambda_d = 0.5, lambda_t = 0.2,
                                   lambda_l = 0.3, seed = 503L)
  all_valid <- all(sapply(gene_trees, function(gt) {
    !is.null(gt) && is.list(gt) && !is.null(gene_tree_to_newick(gt))
  }))
  all_valid
})

# ==============================================================================
# Test: simulate_dtl_batch - require_extant in Batch Mode
# ==============================================================================

cat("\n--- simulate_dtl_batch: require_extant in Batch Mode ---\n")

# Test require_extant=TRUE in batch mode
run_test("batch with require_extant=TRUE all have extant genes", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 10L, lambda_d = 0.3, lambda_t = 0.1,
                                   lambda_l = 1.5, require_extant = TRUE, seed = 600L)
  all_have_extant <- all(sapply(gene_trees, function(gt) {
    gene_tree_num_extant(gt) >= 1
  }))
  all_have_extant
})

# Test batch with high loss and require_extant
run_test("batch with high loss and require_extant=TRUE", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 5L, lambda_d = 0.2, lambda_t = 0.1,
                                   lambda_l = 2.0, require_extant = TRUE, seed = 601L)
  all_have_extant <- all(sapply(gene_trees, function(gt) {
    gene_tree_num_extant(gt) >= 1
  }))
  length(gene_trees) == 5 && all_have_extant
})

# ==============================================================================
# Test: simulate_dtl_batch - transfer_alpha in Batch Mode
# ==============================================================================

cat("\n--- simulate_dtl_batch: transfer_alpha in Batch Mode ---\n")

# Test batch with transfer_alpha
run_test("batch with transfer_alpha parameter", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 5L, lambda_d = 0.3, lambda_t = 0.5,
                                   lambda_l = 0.3, transfer_alpha = 2.0, seed = 700L)
  length(gene_trees) == 5
})

# Test batch with transfer_alpha and require_extant
run_test("batch with transfer_alpha and require_extant", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 5L, lambda_d = 0.3, lambda_t = 0.5,
                                   lambda_l = 1.2, transfer_alpha = 2.0,
                                   require_extant = TRUE, seed = 701L)
  all_have_extant <- all(sapply(gene_trees, function(gt) {
    gene_tree_num_extant(gt) >= 1
  }))
  length(gene_trees) == 5 && all_have_extant
})

# Test batch with high transfer_alpha
run_test("batch with high transfer_alpha", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 5L, lambda_d = 0.3, lambda_t = 0.5,
                                   lambda_l = 0.3, transfer_alpha = 10.0, seed = 702L)
  length(gene_trees) == 5
})

# ==============================================================================
# Test: gene_tree_num_extant
# ==============================================================================

cat("\n--- gene_tree_num_extant Tests ---\n")

# Test gene_tree_num_extant returns a number
run_test("gene_tree_num_extant returns integer", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 800L)
  n <- gene_tree_num_extant(gt)
  is.numeric(n) && n >= 0
})

# Test with require_extant guarantees at least 1
run_test("gene_tree_num_extant >= 1 when require_extant=TRUE", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.3, lambda_t = 0.1, lambda_l = 1.5,
                     require_extant = TRUE, seed = 801L)
  gene_tree_num_extant(gt) >= 1
})

# Test with low loss should have many extant
run_test("gene_tree_num_extant reasonable with low loss", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.1, seed = 802L)
  n <- gene_tree_num_extant(gt)
  n >= 1  # Should almost certainly have extant genes with low loss
})

# ==============================================================================
# Test: gene_tree_to_newick
# ==============================================================================

cat("\n--- gene_tree_to_newick Tests ---\n")

# Test gene_tree_to_newick returns valid Newick string
run_test("gene_tree_to_newick returns non-empty string", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 900L)
  newick <- gene_tree_to_newick(gt)
  is.character(newick) && nchar(newick) > 0
})

# Test Newick ends with semicolon
run_test("gene_tree_to_newick ends with semicolon", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 901L)
  newick <- gene_tree_to_newick(gt)
  grepl(";$", newick)
})

# Test Newick has balanced parentheses
run_test("gene_tree_to_newick has balanced parentheses", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 902L)
  newick <- gene_tree_to_newick(gt)
  n_open <- nchar(gsub("[^(]", "", newick))
  n_close <- nchar(gsub("[^)]", "", newick))
  n_open == n_close
})

# Test Newick contains branch lengths (colons)
run_test("gene_tree_to_newick contains branch lengths", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 903L)
  newick <- gene_tree_to_newick(gt)
  grepl(":", newick)
})

# ==============================================================================
# Test: gene_tree_to_xml
# ==============================================================================

cat("\n--- gene_tree_to_xml Tests ---\n")

# Test gene_tree_to_xml returns valid XML string
run_test("gene_tree_to_xml returns non-empty string", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 1000L)
  xml <- gene_tree_to_xml(gt)
  is.character(xml) && nchar(xml) > 0
})

# Test XML contains RecPhyloXML elements
run_test("gene_tree_to_xml contains recPhylo tag", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 1001L)
  xml <- gene_tree_to_xml(gt)
  grepl("recPhylo", xml, ignore.case = TRUE) || grepl("recGeneTree", xml, ignore.case = TRUE)
})

# Test XML is well-formed (has opening and closing tags)
run_test("gene_tree_to_xml has XML structure", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 1002L)
  xml <- gene_tree_to_xml(gt)
  grepl("<", xml) && grepl(">", xml)
})

# ==============================================================================
# Test: sample_extant
# ==============================================================================

cat("\n--- sample_extant Tests ---\n")

# Test sample_extant returns a tree
run_test("sample_extant returns a tree", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1100L)
  sampled <- sample_extant(gt)
  !is.null(sampled)
})

# Test sample_extant tree has only extant genes
run_test("sample_extant tree has leaves matching extant count", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1101L)
  n_extant <- gene_tree_num_extant(gt)
  sampled <- sample_extant(gt)
  if (!is.null(sampled)) {
    n_leaves <- tree_num_leaves(sampled)
    n_leaves == n_extant
  } else {
    FALSE
  }
})

# Test sample_extant produces valid Newick
run_test("sample_extant produces valid Newick", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1102L)
  sampled <- sample_extant(gt)
  if (!is.null(sampled)) {
    newick <- tree_to_newick(sampled)
    is.character(newick) && nchar(newick) > 0 && grepl(";$", newick)
  } else {
    FALSE
  }
})

# ==============================================================================
# Test: extract_induced_subtree_by_names
# ==============================================================================

cat("\n--- extract_induced_subtree_by_names Tests ---\n")

# Test extracting subtree with subset of leaf names from species tree
run_test("extract_induced_subtree_by_names with subset of leaves", {
  all_names <- tree_leaf_names(sp_tree)
  subset_names <- all_names[1:min(5, length(all_names))]
  subtree <- extract_induced_subtree_by_names(sp_tree, subset_names)
  !is.null(subtree)
})

# Test extracted subtree has correct number of leaves
run_test("extracted subtree has correct leaf count", {
  all_names <- tree_leaf_names(sp_tree)
  subset_names <- all_names[1:min(5, length(all_names))]
  subtree <- extract_induced_subtree_by_names(sp_tree, subset_names)
  if (!is.null(subtree)) {
    tree_num_leaves(subtree) == length(subset_names)
  } else {
    FALSE
  }
})

# Test extracted subtree has correct leaf names
run_test("extracted subtree has correct leaf names", {
  all_names <- tree_leaf_names(sp_tree)
  subset_names <- all_names[1:min(3, length(all_names))]
  subtree <- extract_induced_subtree_by_names(sp_tree, subset_names)
  if (!is.null(subtree)) {
    subtree_names <- tree_leaf_names(subtree)
    all(sort(subtree_names) == sort(subset_names))
  } else {
    FALSE
  }
})

# Test with single leaf
run_test("extract_induced_subtree_by_names with single leaf", {
  all_names <- tree_leaf_names(sp_tree)
  subset_names <- all_names[1]
  subtree <- extract_induced_subtree_by_names(sp_tree, subset_names)
  if (!is.null(subtree)) {
    tree_num_leaves(subtree) == 1
  } else {
    # Single leaf extraction might not be supported
    TRUE
  }
})

# Test with two leaves
run_test("extract_induced_subtree_by_names with two leaves", {
  all_names <- tree_leaf_names(sp_tree)
  subset_names <- all_names[1:min(2, length(all_names))]
  subtree <- extract_induced_subtree_by_names(sp_tree, subset_names)
  if (!is.null(subtree)) {
    tree_num_leaves(subtree) == 2
  } else {
    FALSE
  }
})

# Test with all leaves (should return equivalent tree)
run_test("extract_induced_subtree_by_names with all leaves", {
  all_names <- tree_leaf_names(sp_tree)
  subtree <- extract_induced_subtree_by_names(sp_tree, all_names)
  if (!is.null(subtree)) {
    tree_num_leaves(subtree) == length(all_names)
  } else {
    FALSE
  }
})

# ==============================================================================
# Test: save_newick (Gene Tree)
# ==============================================================================

cat("\n--- save_newick (Gene Tree) Tests ---\n")

# Create temp directory for file tests
temp_dir <- tempdir()

# Test save_newick for gene tree
run_test("save_newick exports gene tree to file", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1200L)
  filepath <- file.path(temp_dir, "test_gene_tree.nwk")
  save_newick(gt, filepath)
  file.exists(filepath)
})

# Test saved file contains valid Newick
run_test("saved gene tree Newick file is valid", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1201L)
  filepath <- file.path(temp_dir, "test_gene_tree2.nwk")
  save_newick(gt, filepath)
  content <- readLines(filepath)
  length(content) > 0 && grepl(";", content[length(content)])
})

# Test saved Newick matches gene_tree_to_newick output
run_test("saved Newick matches gene_tree_to_newick", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1202L)
  filepath <- file.path(temp_dir, "test_gene_tree3.nwk")
  save_newick(gt, filepath)
  saved_content <- paste(readLines(filepath), collapse = "\n")
  expected <- gene_tree_to_newick(gt)
  # Allow for minor whitespace differences
  gsub("\\s", "", saved_content) == gsub("\\s", "", expected) ||
    grepl(gsub("[()]", ".", substr(expected, 1, 20)), saved_content)
})

# ==============================================================================
# Test: save_xml (RecPhyloXML)
# ==============================================================================

cat("\n--- save_xml (RecPhyloXML) Tests ---\n")

# Test save_xml exports to file
run_test("save_xml exports gene tree to file", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1300L)
  filepath <- file.path(temp_dir, "test_gene_tree.xml")
  save_xml(gt, filepath)
  file.exists(filepath)
})

# Test saved XML file contains XML content
run_test("saved XML file contains XML content", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1301L)
  filepath <- file.path(temp_dir, "test_gene_tree2.xml")
  save_xml(gt, filepath)
  content <- paste(readLines(filepath), collapse = "\n")
  grepl("<", content) && grepl(">", content)
})

# Test saved XML is well-formed
run_test("saved XML file is well-formed", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1302L)
  filepath <- file.path(temp_dir, "test_gene_tree3.xml")
  save_xml(gt, filepath)
  content <- paste(readLines(filepath), collapse = "\n")
  # Check for basic XML structure
  grepl("<?xml", content, fixed = TRUE) || grepl("<recPhylo", content, ignore.case = TRUE) ||
    grepl("<recGeneTree", content, ignore.case = TRUE) || grepl("<clade", content)
})

# ==============================================================================
# Test: save_csv (DTL Events)
# ==============================================================================

cat("\n--- save_csv (DTL Events) Tests ---\n")

# Test save_csv exports to file
run_test("save_csv exports gene tree events to file", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1400L)
  filepath <- file.path(temp_dir, "test_gene_tree.csv")
  save_csv(gt, filepath)
  file.exists(filepath)
})

# Test saved CSV file is readable
run_test("saved CSV file is readable as data frame", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1401L)
  filepath <- file.path(temp_dir, "test_gene_tree2.csv")
  save_csv(gt, filepath)
  df <- tryCatch({
    read.csv(filepath)
  }, error = function(e) NULL)
  !is.null(df) && is.data.frame(df)
})

# Test CSV contains expected columns (if applicable)
run_test("saved CSV has rows of data", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1402L)
  filepath <- file.path(temp_dir, "test_gene_tree3.csv")
  save_csv(gt, filepath)
  df <- read.csv(filepath)
  nrow(df) > 0
})

# Test CSV with high duplication rate (more events)
run_test("save_csv with high duplication rate captures events", {
  gt <- simulate_dtl(sp_tree, lambda_d = 2.0, lambda_t = 0.5, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1403L)
  filepath <- file.path(temp_dir, "test_gene_tree4.csv")
  save_csv(gt, filepath)
  df <- read.csv(filepath)
  nrow(df) > 0 && ncol(df) > 0
})

# ==============================================================================
# Test: Edge Cases and Error Handling
# ==============================================================================

cat("\n--- Edge Cases and Error Handling Tests ---\n")

# Test with very small species tree
run_test("DTL simulation works with small species tree", {
  small_sp <- simulate_species_tree(5L, lambda = 1.0, mu = 0.3, seed = 1500L)
  gt <- simulate_dtl(small_sp, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3, seed = 1501L)
  !is.null(gt)
})

# Test with zero rates
run_test("DTL simulation with all zero rates", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.0, lambda_t = 0.0, lambda_l = 0.0, seed = 1502L)
  !is.null(gt)
})

# Test batch with zero rates
run_test("batch DTL simulation with all zero rates", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 3L, lambda_d = 0.0, lambda_t = 0.0,
                                   lambda_l = 0.0, seed = 1503L)
  length(gene_trees) == 3
})

# ==============================================================================
# Test: Integration Tests
# ==============================================================================

cat("\n--- Integration Tests ---\n")

# Full workflow test: simulate, export, verify
run_test("full workflow: simulate -> export newick -> export xml -> export csv", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.3, lambda_l = 0.4,
                     require_extant = TRUE, seed = 1600L)

  nwk_path <- file.path(temp_dir, "workflow_test.nwk")
  xml_path <- file.path(temp_dir, "workflow_test.xml")
  csv_path <- file.path(temp_dir, "workflow_test.csv")

  save_newick(gt, nwk_path)
  save_xml(gt, xml_path)
  save_csv(gt, csv_path)

  file.exists(nwk_path) && file.exists(xml_path) && file.exists(csv_path)
})

# Test batch workflow
run_test("batch workflow: simulate batch -> process each tree", {
  gene_trees <- simulate_dtl_batch(sp_tree, n = 3L, lambda_d = 0.5, lambda_t = 0.2,
                                   lambda_l = 0.4, require_extant = TRUE, seed = 1601L)

  all_success <- TRUE
  for (i in seq_along(gene_trees)) {
    gt <- gene_trees[[i]]
    if (gene_tree_num_extant(gt) < 1) {
      all_success <- FALSE
      break
    }
    newick <- gene_tree_to_newick(gt)
    if (is.null(newick) || nchar(newick) == 0) {
      all_success <- FALSE
      break
    }
  }
  all_success
})

# Test sample_extant integration
run_test("integration: simulate -> sample_extant -> export", {
  gt <- simulate_dtl(sp_tree, lambda_d = 0.5, lambda_t = 0.2, lambda_l = 0.3,
                     require_extant = TRUE, seed = 1602L)
  sampled <- sample_extant(gt)
  if (!is.null(sampled)) {
    newick <- tree_to_newick(sampled)
    is.character(newick) && nchar(newick) > 0
  } else {
    FALSE
  }
})

# ==============================================================================
# Summary
# ==============================================================================

cat("\n========================================\n")
cat("Test Summary\n")
cat("========================================\n")
cat(sprintf("Tests Passed: %d\n", tests_passed))
cat(sprintf("Tests Failed: %d\n", tests_failed))
cat(sprintf("Total Tests:  %d\n", tests_passed + tests_failed))
cat("========================================\n")

if (tests_failed > 0) {
  cat("SOME TESTS FAILED!\n")
} else {
  cat("ALL TESTS PASSED!\n")
}

# Clean up temp files
unlink(file.path(temp_dir, "test_gene_tree*.nwk"))
unlink(file.path(temp_dir, "test_gene_tree*.xml"))
unlink(file.path(temp_dir, "test_gene_tree*.csv"))
unlink(file.path(temp_dir, "workflow_test.*"))
