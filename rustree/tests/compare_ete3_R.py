#!/usr/bin/env python
import os
import glob
from ete3 import Tree
import rpy2.robjects as ro

def load_tree(filepath, format=1):
    """
    Loads a Newick tree from the given filepath using ete3.
    Returns an ete3 Tree object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tree file not found: {filepath}")
    with open(filepath, "r") as f:
        newick_str = f.read().strip()
    # Optionally, print the tree content for debugging:
    # print(f"Contents of {filepath}:\n{newick_str}\n")

    tree = Tree(newick_str, format=format)

    return tree

def compute_rf_ete3(t1, t2, unrooted=True):
    """
    Computes the unrooted RF distance between two ete3 trees.
    Returns the RF distance (first element of the returned tuple).
    """
    # ete3's `robinson_foulds` returns a tuple: (rf, max_rf, common_leaves, parts_t1, parts_t2)
    rf_data = t1.robinson_foulds(t2, unrooted_trees=unrooted)
    return rf_data[0]

def compute_rf_R(t1, t2):
    """
    Computes the RF distance using R's ape package via rpy2.
    The trees t1 and t2 are ete3 Tree objects.
    Returns the RF distance from R (using method "PH85").
    """
    # Ensure the ape package is loaded
    ro.r('library(ape)')
    # Get Newick strings from the ete3 trees
    nw1 = t1.write(format=1)
    nw2 = t2.write(format=1)
    ro.globalenv['nw1'] = nw1
    ro.globalenv['nw2'] = nw2
    # R code: reads the trees from text, creates a multiPhylo object,
    # computes the topological distance (PH85) and returns the first element.
    r_code = """
    t1 <- read.tree(text=nw1)
    t2 <- read.tree(text=nw2)
    d <- dist.topo(c(t1,t2), method="PH85")
    d[1]
    """
    # Execute R code
    rf_r = ro.r(r_code)[0]
    return rf_r

def main():
    # Path where trees are located
    tree_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RF_trees")
    # Pattern to match tree files (e.g., "tree_*.newick")
    pattern = os.path.join(tree_dir, "tree_*.newick")
    tree_files = sorted(glob.glob(pattern))
    
    if not tree_files:
        print(f"No tree files found in {tree_dir} matching pattern 'tree_*.newick'.")
        return
    
    # Load each tree using ete3 and store them in a dictionary mapping filename to Tree
    trees = {}
    for filepath in tree_files:
        try:
            trees[filepath] = load_tree(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    # Sort the file names for consistency
    sorted_files = sorted(trees.keys())
    
    # Print header for comparison table
    header = f"+----------+----------+-----------------+-----------------+"
    print(header)
    print(f"| {'Tree file 1':>8} | {'Tree file 2':>8} | {'RF (ete3)':>15} | {'RF (R ape)':>15} |")
    print(header)
    
    # Compare each unique pair of trees
    n = 5
    for i in range(n):
        for j in range(i+1, n):
            file1 = sorted_files[i]
            file2 = sorted_files[j]
            t1 = trees[file1]
            t2 = trees[file2]
            try:
                rf_ete3 = compute_rf_ete3(t1, t2, unrooted=True)
            except Exception as e:
                print(f"Error computing ete3 RF for {file1} and {file2}: {e}")
                continue
            try:
                rf_R = compute_rf_R(t1, t2)
            except Exception as e:
                print(f"Error computing R RF for {file1} and {file2}: {e}")
                continue
            print(f"| {os.path.basename(file1):>8} | {os.path.basename(file2):>8} | {rf_ete3:15} | {rf_R:15} |")
    print(header)

if __name__ == "__main__":
    main()
