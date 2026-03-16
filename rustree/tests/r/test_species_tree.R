# Comprehensive R tests for species tree functions in rustree
# Run with: Rscript test_species_tree.R

# Load the library
dyn.load("/home/enzo/Documents/Zombi/ZOMBI/zombi-rs/rustree/target/release/librustree.so")
source("/home/enzo/Documents/Zombi/ZOMBI/zombi-rs/rustree/R/rustree.R")

# Test framework
test_passed <- 0
test_failed <- 0

run_test <- function(name, expr) {
  result <- tryCatch({
    eval(expr)
    cat("PASS:", name, "\n")
    test_passed <<- test_passed + 1
    TRUE
  }, error = function(e) {
    cat("FAIL:", name, "-", conditionMessage(e), "\n")
    test_failed <<- test_failed + 1
    FALSE
  })
}

assert_equal <- function(actual, expected, msg = "") {
  if (!identical(actual, expected)) {
    stop(paste("Expected", expected, "but got", actual, msg))
  }
}

assert_true <- function(cond, msg = "") {
  if (!cond) {
    stop(paste("Assertion failed:", msg))
  }
}

assert_length <- function(vec, expected_len, msg = "") {
  if (length(vec) != expected_len) {
    stop(paste("Expected length", expected_len, "but got", length(vec), msg))
  }
}

# Create temp directory for file tests
temp_dir <- tempdir()

cat("=== Species Tree Function Tests ===\n\n")

# ============================================================================
# Tests for simulate_species_tree
# ============================================================================
cat("--- simulate_species_tree tests ---\n")

run_test("simulate_species_tree: basic n=5", {
  tree <- simulate_species_tree(5L, 1.0, 0.3)
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 5L)
})

run_test("simulate_species_tree: n=20", {
  tree <- simulate_species_tree(20L, 1.0, 0.3)
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 20L)
})

run_test("simulate_species_tree: n=100", {
  tree <- simulate_species_tree(100L, 1.0, 0.3)
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 100L)
})

run_test("simulate_species_tree: reproducibility with same seed", {
  tree1 <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  tree2 <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  newick1 <- tree_to_newick(tree1)
  newick2 <- tree_to_newick(tree2)
  assert_equal(newick1, newick2)
})

run_test("simulate_species_tree: different seeds produce different trees", {
  tree1 <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  tree2 <- simulate_species_tree(10L, 1.0, 0.3, seed = 123L)
  newick1 <- tree_to_newick(tree1)
  newick2 <- tree_to_newick(tree2)
  assert_true(newick1 != newick2, "Trees with different seeds should differ")
})

run_test("simulate_species_tree: tree has expected structure", {
  tree <- simulate_species_tree(5L, 1.0, 0.3, seed = 42L)
  assert_true(!is.null(tree$name), "Tree should have name field")
  assert_true(!is.null(tree$parent), "Tree should have parent field")
  assert_true(!is.null(tree$left_child), "Tree should have left_child field")
  assert_true(!is.null(tree$right_child), "Tree should have right_child field")
  assert_true(!is.null(tree$length), "Tree should have length field")
  assert_true(!is.null(tree$depth), "Tree should have depth field")
  assert_true(!is.null(tree$root), "Tree should have root field")
  assert_true(!is.null(tree$events), "Tree should have events field")
})

run_test("simulate_species_tree: different lambda/mu rates", {
  tree1 <- simulate_species_tree(10L, 2.0, 0.5, seed = 42L)
  tree2 <- simulate_species_tree(10L, 1.0, 0.1, seed = 42L)
  assert_equal(tree_num_leaves(tree1), 10L)
  assert_equal(tree_num_leaves(tree2), 10L)
})

run_test("simulate_species_tree: error on invalid n", {
  error_raised <- FALSE
  tryCatch({
    simulate_species_tree(0L, 1.0, 0.3)
  }, error = function(e) {
    error_raised <<- TRUE
  })
  assert_true(error_raised, "Should raise error for n=0")
})

run_test("simulate_species_tree: error on lambda <= mu", {
  error_raised <- FALSE
  tryCatch({
    simulate_species_tree(5L, 0.3, 0.5)  # lambda < mu
  }, error = function(e) {
    error_raised <<- TRUE
  })
  assert_true(error_raised, "Should raise error when lambda <= mu")
})

# ============================================================================
# Tests for parse_newick
# ============================================================================
cat("\n--- parse_newick tests ---\n")

run_test("parse_newick: simple two-leaf tree", {
  tree <- parse_newick("(A:1,B:1):0;")
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 2L)
})

run_test("parse_newick: three-leaf tree", {
  tree <- parse_newick("((A:1,B:1):1,C:2):0;")
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 3L)
})

run_test("parse_newick: complex tree with different branch lengths", {
  tree <- parse_newick("((A:0.5,B:0.7):0.3,(C:1.0,D:0.8):0.2):0;")
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 4L)
})

run_test("parse_newick: tree without internal node names", {
  tree <- parse_newick("((A:1,B:1):1,(C:1,D:1):1):0;")
  leaf_names <- tree_leaf_names(tree)
  assert_length(leaf_names, 4)
  assert_true("A" %in% leaf_names, "A should be in leaf names")
  assert_true("B" %in% leaf_names, "B should be in leaf names")
  assert_true("C" %in% leaf_names, "C should be in leaf names")
  assert_true("D" %in% leaf_names, "D should be in leaf names")
})

run_test("parse_newick: tree with numeric names", {
  tree <- parse_newick("((1:1,2:1):1,3:2):0;")
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 3L)
})

run_test("parse_newick: deeply nested tree", {
  tree <- parse_newick("((((A:1,B:1):1,C:2):1,D:3):1,E:4):0;")
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 5L)
})

run_test("parse_newick: balanced binary tree", {
  tree <- parse_newick("(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1):0;")
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 8L)
})

run_test("parse_newick: tree with zero branch lengths", {
  tree <- parse_newick("((A:0,B:0):0,C:0):0;")
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 3L)
})

run_test("parse_newick: tree with very long branch lengths", {
  tree <- parse_newick("((A:100.5,B:200.3):50.1,C:150.7):0;")
  num_leaves <- tree_num_leaves(tree)
  assert_equal(num_leaves, 3L)
})

run_test("parse_newick: tree with underscores in names", {
  tree <- parse_newick("((Species_A:1,Species_B:1):1,Species_C:2):0;")
  leaf_names <- tree_leaf_names(tree)
  assert_true("Species_A" %in% leaf_names, "Species_A should be in leaf names")
})

# ============================================================================
# Tests for tree_to_newick (round-trip tests)
# ============================================================================
cat("\n--- tree_to_newick tests ---\n")

run_test("tree_to_newick: round-trip simple tree", {
  original <- "(A:1,B:1):0;"
  tree <- parse_newick(original)
  newick <- tree_to_newick(tree)
  tree2 <- parse_newick(newick)
  assert_equal(tree_num_leaves(tree), tree_num_leaves(tree2))
  leaf_names1 <- sort(tree_leaf_names(tree))
  leaf_names2 <- sort(tree_leaf_names(tree2))
  assert_true(all(leaf_names1 == leaf_names2), "Leaf names should match after round-trip")
})

run_test("tree_to_newick: round-trip complex tree", {
  original <- "((A:0.5,B:0.7):0.3,(C:1.0,D:0.8):0.2):0;"
  tree <- parse_newick(original)
  newick <- tree_to_newick(tree)
  tree2 <- parse_newick(newick)
  assert_equal(tree_num_leaves(tree), tree_num_leaves(tree2))
})

run_test("tree_to_newick: round-trip simulated tree", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  newick <- tree_to_newick(tree)
  tree2 <- parse_newick(newick)
  assert_equal(tree_num_leaves(tree), tree_num_leaves(tree2))
})

run_test("tree_to_newick: round-trip preserves topology", {
  tree <- simulate_species_tree(5L, 1.0, 0.3, seed = 42L)
  newick1 <- tree_to_newick(tree)
  tree2 <- parse_newick(newick1)
  newick2 <- tree_to_newick(tree2)
  tree3 <- parse_newick(newick2)
  assert_equal(tree_num_leaves(tree), tree_num_leaves(tree3))
  leaf_names1 <- sort(tree_leaf_names(tree))
  leaf_names3 <- sort(tree_leaf_names(tree3))
  assert_true(all(leaf_names1 == leaf_names3), "Leaf names should match after double round-trip")
})

run_test("tree_to_newick: output ends with semicolon", {
  tree <- simulate_species_tree(5L, 1.0, 0.3, seed = 42L)
  newick <- tree_to_newick(tree)
  assert_true(grepl(";$", newick), "Newick string should end with semicolon")
})

# ============================================================================
# Tests for tree_num_leaves
# ============================================================================
cat("\n--- tree_num_leaves tests ---\n")

run_test("tree_num_leaves: simulated tree n=5", {
  tree <- simulate_species_tree(5L, 1.0, 0.3, seed = 42L)
  assert_equal(tree_num_leaves(tree), 5L)
})

run_test("tree_num_leaves: simulated tree n=50", {
  tree <- simulate_species_tree(50L, 1.0, 0.3, seed = 42L)
  assert_equal(tree_num_leaves(tree), 50L)
})

run_test("tree_num_leaves: parsed tree 2 leaves", {
  tree <- parse_newick("(A:1,B:1):0;")
  assert_equal(tree_num_leaves(tree), 2L)
})

run_test("tree_num_leaves: parsed tree 4 leaves", {
  tree <- parse_newick("((A:1,B:1):1,(C:1,D:1):1):0;")
  assert_equal(tree_num_leaves(tree), 4L)
})

run_test("tree_num_leaves: parsed tree 8 leaves", {
  tree <- parse_newick("(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1):0;")
  assert_equal(tree_num_leaves(tree), 8L)
})

# ============================================================================
# Tests for tree_leaf_names
# ============================================================================
cat("\n--- tree_leaf_names tests ---\n")

run_test("tree_leaf_names: simple parsed tree", {
  tree <- parse_newick("(A:1,B:1):0;")
  leaf_names <- tree_leaf_names(tree)
  assert_length(leaf_names, 2)
  assert_true("A" %in% leaf_names, "A should be in leaf names")
  assert_true("B" %in% leaf_names, "B should be in leaf names")
})

run_test("tree_leaf_names: parsed tree with 4 leaves", {
  tree <- parse_newick("((Alpha:1,Beta:1):1,(Gamma:1,Delta:1):1):0;")
  leaf_names <- tree_leaf_names(tree)
  assert_length(leaf_names, 4)
  assert_true("Alpha" %in% leaf_names, "Alpha should be in leaf names")
  assert_true("Beta" %in% leaf_names, "Beta should be in leaf names")
  assert_true("Gamma" %in% leaf_names, "Gamma should be in leaf names")
  assert_true("Delta" %in% leaf_names, "Delta should be in leaf names")
})

run_test("tree_leaf_names: simulated tree has numeric names", {
  tree <- simulate_species_tree(5L, 1.0, 0.3, seed = 42L)
  leaf_names <- tree_leaf_names(tree)
  assert_length(leaf_names, 5)
  # Simulated trees have numeric names starting from 0
  for (name in leaf_names) {
    assert_true(!is.na(as.numeric(name)), paste("Leaf name should be numeric:", name))
  }
})

run_test("tree_leaf_names: count matches tree_num_leaves", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  leaf_names <- tree_leaf_names(tree)
  num_leaves <- tree_num_leaves(tree)
  assert_equal(length(leaf_names), num_leaves)
})

run_test("tree_leaf_names: unique names", {
  tree <- simulate_species_tree(20L, 1.0, 0.3, seed = 42L)
  leaf_names <- tree_leaf_names(tree)
  assert_equal(length(unique(leaf_names)), length(leaf_names), "Leaf names should be unique")
})

# ============================================================================
# Tests for save_newick
# ============================================================================
cat("\n--- save_newick tests ---\n")

run_test("save_newick: save simulated tree", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_tree1.nwk")
  save_newick(tree, filepath)
  assert_true(file.exists(filepath), "File should exist")
  content <- readLines(filepath)
  assert_true(length(content) > 0, "File should have content")
  assert_true(grepl(";$", content[1]), "File content should end with semicolon")
})

run_test("save_newick: save parsed tree", {
  tree <- parse_newick("((A:1,B:1):1,(C:1,D:1):1):0;")
  filepath <- file.path(temp_dir, "test_tree2.nwk")
  save_newick(tree, filepath)
  assert_true(file.exists(filepath), "File should exist")
})

run_test("save_newick: read back and verify", {
  original_tree <- simulate_species_tree(5L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_tree3.nwk")
  save_newick(original_tree, filepath)

  # Read back and parse
  content <- readLines(filepath)
  read_tree <- parse_newick(content[1])

  # Verify leaf count matches
  assert_equal(tree_num_leaves(original_tree), tree_num_leaves(read_tree))
})

run_test("save_newick: round-trip integrity", {
  original_tree <- simulate_species_tree(8L, 1.0, 0.3, seed = 42L)
  original_newick <- tree_to_newick(original_tree)

  filepath <- file.path(temp_dir, "test_tree4.nwk")
  save_newick(original_tree, filepath)

  content <- readLines(filepath)
  assert_equal(content[1], original_newick, "Saved content should match tree_to_newick output")
})

# ============================================================================
# Tests for save_bd_events_csv
# ============================================================================
cat("\n--- save_bd_events_csv tests ---\n")

run_test("save_bd_events_csv: creates file", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_events1.csv")
  save_bd_events_csv(tree, filepath)
  assert_true(file.exists(filepath), "File should exist")
})

run_test("save_bd_events_csv: has correct header", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_events2.csv")
  save_bd_events_csv(tree, filepath)

  lines <- readLines(filepath)
  header <- lines[1]
  expected_columns <- c("time", "node_name", "event_type", "child1_name", "child2_name")
  for (col in expected_columns) {
    assert_true(grepl(col, header), paste("Header should contain", col))
  }
})

run_test("save_bd_events_csv: CSV has multiple rows", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_events3.csv")
  save_bd_events_csv(tree, filepath)

  lines <- readLines(filepath)
  assert_true(length(lines) > 1, "CSV should have header plus data rows")
})

run_test("save_bd_events_csv: event types are valid", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_events4.csv")
  save_bd_events_csv(tree, filepath)

  df <- read.csv(filepath)
  valid_events <- c("Speciation", "Extinction", "Leaf")
  for (et in unique(df$event_type)) {
    assert_true(et %in% valid_events, paste("Event type should be valid:", et))
  }
})

run_test("save_bd_events_csv: leaf events match leaf count", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_events5.csv")
  save_bd_events_csv(tree, filepath)

  df <- read.csv(filepath)
  leaf_events <- sum(df$event_type == "Leaf")
  assert_equal(leaf_events, tree_num_leaves(tree))
})

run_test("save_bd_events_csv: times are numeric", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_events6.csv")
  save_bd_events_csv(tree, filepath)

  df <- read.csv(filepath)
  assert_true(is.numeric(df$time), "Time column should be numeric")
})

run_test("save_bd_events_csv: times are non-negative", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_events7.csv")
  save_bd_events_csv(tree, filepath)

  df <- read.csv(filepath)
  assert_true(all(df$time >= 0), "All times should be non-negative")
})

run_test("save_bd_events_csv: speciation events have children", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  filepath <- file.path(temp_dir, "test_events8.csv")
  save_bd_events_csv(tree, filepath)

  df <- read.csv(filepath)
  spec_events <- df[df$event_type == "Speciation", ]
  if (nrow(spec_events) > 0) {
    for (i in 1:nrow(spec_events)) {
      child1 <- spec_events$child1_name[i]
      child2 <- spec_events$child2_name[i]
      assert_true(!is.na(child1) && child1 != "", "Speciation should have child1")
      assert_true(!is.na(child2) && child2 != "", "Speciation should have child2")
    }
  }
})

run_test("save_bd_events_csv: works with parsed tree", {
  tree <- parse_newick("((A:1,B:1):1,(C:1,D:1):1):0;")
  filepath <- file.path(temp_dir, "test_events9.csv")
  save_bd_events_csv(tree, filepath)
  assert_true(file.exists(filepath), "File should exist for parsed tree")

  df <- read.csv(filepath)
  # Parsed trees should have Speciation and Leaf events (no Extinction)
  valid_events <- c("Speciation", "Leaf")
  for (et in unique(df$event_type)) {
    assert_true(et %in% valid_events, paste("Parsed tree event type should be valid:", et))
  }
})

run_test("save_bd_events_csv: node names appear in events", {
  tree <- parse_newick("((A:1,B:1):1,(C:1,D:1):1):0;")
  filepath <- file.path(temp_dir, "test_events10.csv")
  save_bd_events_csv(tree, filepath)

  df <- read.csv(filepath)
  leaf_names <- tree_leaf_names(tree)
  event_node_names <- unique(df$node_name)

  # All leaf names should appear as node_name in Leaf events
  leaf_events <- df[df$event_type == "Leaf", ]
  for (leaf in leaf_names) {
    assert_true(leaf %in% leaf_events$node_name, paste("Leaf", leaf, "should have Leaf event"))
  }
})

# ============================================================================
# Additional edge case tests
# ============================================================================
cat("\n--- Edge case tests ---\n")

run_test("edge case: single leaf tree simulation", {
  tree <- simulate_species_tree(1L, 1.0, 0.3, seed = 42L)
  assert_equal(tree_num_leaves(tree), 1L)
})

run_test("edge case: large tree n=200", {
  tree <- simulate_species_tree(200L, 1.0, 0.3, seed = 42L)
  assert_equal(tree_num_leaves(tree), 200L)
})

run_test("edge case: low extinction rate (mu near 0)", {
  tree <- simulate_species_tree(10L, 1.0, 0.001, seed = 42L)
  assert_equal(tree_num_leaves(tree), 10L)
})

run_test("edge case: high speciation rate", {
  tree <- simulate_species_tree(10L, 10.0, 1.0, seed = 42L)
  assert_equal(tree_num_leaves(tree), 10L)
})

run_test("edge case: tree structure consistency", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  # For a binary tree with n leaves, there should be n-1 internal nodes
  # Total nodes = 2n - 1 (for a fully binary tree without extinctions)
  # But with extinctions, we may have more nodes
  total_nodes <- length(tree$name)
  num_leaves <- tree_num_leaves(tree)
  assert_true(total_nodes >= 2 * num_leaves - 1,
              "Total nodes should be at least 2n-1 for n leaves")
})

run_test("edge case: branch lengths are non-negative", {
  tree <- simulate_species_tree(20L, 1.0, 0.3, seed = 42L)
  assert_true(all(tree$length >= 0), "All branch lengths should be non-negative")
})

run_test("edge case: depths assigned correctly", {
  tree <- simulate_species_tree(10L, 1.0, 0.3, seed = 42L)
  # All depths should be defined (non-NA)
  non_na_depths <- !is.na(tree$depth)
  assert_true(all(non_na_depths), "All nodes should have depths assigned")
})

# ============================================================================
# Summary
# ============================================================================
cat("\n=== Test Summary ===\n")
cat("Passed:", test_passed, "\n")
cat("Failed:", test_failed, "\n")
cat("Total:", test_passed + test_failed, "\n")

if (test_failed > 0) {
  cat("\nSome tests FAILED!\n")
  quit(status = 1)
} else {
  cat("\nAll tests PASSED!\n")
  quit(status = 0)
}
