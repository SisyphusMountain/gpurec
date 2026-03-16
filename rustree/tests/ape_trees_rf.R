# generate_rf_trees.R

# Load or install the ape package.
if (!require("ape", quietly = TRUE)) {
  install.packages("ape", repos = "http://cran.us.r-project.org")
  library(ape)
}

# Set directory to save trees and results.
dir_path <- "/home/enzo/Documents/git/rustree/tests/RF_trees"

# Create the directory if it doesn't exist.
if (!dir.exists(dir_path)) {
  dir.create(dir_path, recursive = TRUE)
}

# Parameters: adjust as needed.
num_trees <- 200   # Number of trees to generate.
num_leaves <- 10 # Number of leaves per tree.

# Generate random trees.
trees <- lapply(1:num_trees, function(i) rtree(num_leaves))
# Convert the list into a multiPhylo object which is required by dist.topo.
class(trees) <- "multiPhylo"

# Save each tree's Newick representation to a separate file.
for(i in seq_along(trees)) {
  newick_str <- write.tree(trees[[i]])
  file_name <- sprintf("%s/tree_%02d.newick", dir_path, i)
  writeLines(newick_str, con = file_name)
  cat(sprintf("Saved tree %d to %s\n", i, file_name))
}

# Compute pairwise RF distances using ape's dist.topo function.
rf_matrix <- dist.topo(trees, method = "PH85")
cat("Computed RF distances:\n")
print(rf_matrix)

# Save the full RF distance matrix to a text file.
rf_matrix_file <- sprintf("%s/rf_distance_matrix.txt", dir_path)
capture.output(print(rf_matrix), file = rf_matrix_file)
cat(sprintf("RF distance matrix saved to %s\n", rf_matrix_file))

# Optionally, save each individual pair's RF distance to separate text files.
# Convert the distance object to a full matrix.
rf_mat_as_matrix <- as.matrix(rf_matrix)
for(i in 1:(num_trees - 1)) {
  for(j in (i + 1):num_trees) {
    # Extract RF distance for trees i and j.
    rf_distance <- rf_mat_as_matrix[i, j]
    
    # Generate a file name for this pair.
    file_name <- sprintf("%s/rf_tree_%02d_%02d.txt", dir_path, i, j)
    
    # Write the result to the file.
    result_text <- sprintf("%d", rf_distance)
    writeLines(result_text, con = file_name)
    
    cat(sprintf("Saved RF distance for trees %d and %d to %s\n", i, j, file_name))
  }
}
