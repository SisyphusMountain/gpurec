use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use pest::Parser;
use crate::node::{FlatTree, TraversalOrder};
use crate::newick::newick::{newick_to_tree, node_to_newick, NewickParser, Rule};

pub fn remove_node(
    flat_tree: &mut FlatTree,
    index: usize,
) {
    let parent_index = flat_tree[index]
        .parent
        .expect("Root node can't be removed");
    // The node must have a sister since it is not the root.
    let sister_index = if flat_tree[parent_index].left_child.unwrap() == index {
        flat_tree[parent_index].right_child.unwrap()
    } else {
        flat_tree[parent_index].left_child.unwrap()
    };
    // The grandparent index is optional since the parent could be the root.
    let grandparent_index_opt = flat_tree[parent_index].parent;

    // Remove the node and its parent from the tree using a Subtree Pruning and Regrafting (SPR) move.
    // An SPR move involves cutting a subtree and reattaching it at a different location in the tree.
    flat_tree[index].parent = None;
    // The sister node becomes the child of the grandparent after we remove the parent.
    // If the grandparent was the root, the sister becomes the new root.
    flat_tree[sister_index].parent = grandparent_index_opt;

    if let Some(grandparent_index) = grandparent_index_opt {
        // One of the children of the grandparent node was the parent node.²    
        // Find which one it was, and replace it with the sister node of the node that was removed.
        if flat_tree[grandparent_index].left_child == Some(parent_index) {
            flat_tree[grandparent_index].left_child = Some(sister_index);
        } else {
            flat_tree[grandparent_index].right_child = Some(sister_index);
        }
    } else {
        // If the parent was the root, the sister becomes the new root.
        flat_tree[sister_index].parent = None;
        flat_tree.root = sister_index;
    }
}
pub fn spr(
    flat_tree: &mut FlatTree,
    donor: usize,
    recipient: usize,
    time: f64,
) {
    let donor_parent = flat_tree[donor].parent.expect("The donor node should not be the root");
    let recipient_parent = flat_tree[recipient].parent.expect("The recipient node should not be the root");
    
    let recipient_sibling = if flat_tree[recipient_parent].left_child.unwrap() == recipient {
        flat_tree[recipient_parent].right_child.unwrap()
    } else {
        flat_tree[recipient_parent].left_child.unwrap()
    };

    let recipient_grandparent_opt = flat_tree[recipient_parent].parent;

    if donor_parent == recipient_parent {
        // In this case, only update the depth of the parent node of the donor (which is the same as the parent of the recipient).
        flat_tree[recipient_parent].depth = Some(time);
    } else {
        // Perform disconnects and reconnects to perform the SPR move. The nodes that must change are:
        // 1. The parent of the recipient node PR (changes father: it goes from PPR to PD, and changes child: it goes from SR to D).
        // 2. The sister of the recipient node SR (changes father: it goes from PD to PR).
        // 3. The parent of the donor node PD (changes child: it goes from D to PR).
        // 4. The donor node (changes parent: it goes from PD to PR).
        // 5. The grandparent of the recipient node PPR (changes child: it goes from PR to SR).

        // 1.
        // The root can change if the parent of the recipient is the root.
        // First, disconnect the receiver's parent, and give it the donor as a child. 
        // The receiver's parent has two children. When we disconnect PR, the sister of the receiver SR must be reconnected to the father of PR.
        // If there is 

        // If the recipient's parent is the root, the root of the tree changes. The recipient was one of the children of the root.
        // Now the root is the other child of the root.
        let recipient_has_grandparent =  !flat_tree[recipient_parent].parent.is_none();
        // The receiver's parent changes parent: the receiver's parent's parent is the donor's parent.
        flat_tree[recipient_parent].parent = Some(donor_parent);
        // The parent of the donor is now the child of the donor's parent.
        if flat_tree[recipient_parent].left_child.unwrap() == recipient {
            flat_tree[recipient_parent].left_child = Some(donor);
        } else {
            flat_tree[recipient_parent].right_child = Some(donor);
        }
        // Change the depth of the recipient's parent.
        flat_tree[recipient_parent].depth = Some(time);
        // Now the recipient's parent is correctly connected
        // It remains to modify: the receiver's parent's parent, the receiver's sister, the donor's parent, the donor
        // Fix the receiver's parent's parent: if the receiver was not the root, the receiver's parent's parent
        // now has a new child node, the sister of the receiver.
        // Also fix the sister of the receiver: it now has a new parent, the receiver's parent's parent.
        if recipient_has_grandparent {
            if flat_tree[recipient_grandparent_opt.unwrap()].left_child.unwrap() == recipient_parent {
                flat_tree[recipient_grandparent_opt.unwrap()].left_child = Some(recipient_sibling);
            } else {
                flat_tree[recipient_grandparent_opt.unwrap()].right_child = Some(recipient_sibling);
            }
            flat_tree[recipient_sibling].parent = recipient_grandparent_opt;
        } else {
            // If the recipient's parent was the root, the sister of the recipient becomes the new root.
            flat_tree[recipient_sibling].parent = None;
            flat_tree.root = recipient_sibling;
        }
        // Now the recipient's parent's parent and the recipient's sister are correctly connected.
        // We fix the donor's parent
        if flat_tree[donor_parent].left_child.unwrap() == donor {
            flat_tree[donor_parent].left_child = Some(recipient_parent);
        } else {
            flat_tree[donor_parent].right_child = Some(recipient_parent);
        }
        // The donor's parent is now correctly connected.
        // We fix the donor
        flat_tree[donor].parent = Some(recipient_parent);
        // The donor is now correctly connected.
        // The recipient's parent's parent, the recipient's sister, the donor's parent, and the donor are now correctly connected.
    }
}

/// Finds the deepest nodes (leaves) in the flat tree.
///
/// This function sorts the leaves by their depth in descending order and returns the top `nb_leaves` deepest nodes.
///
/// # Arguments
///
/// * `flat_tree` - A reference to the flat tree.
/// * `nb_leaves` - The number of deepest leaves to find.
///
/// # Returns
///
/// A vector containing the indexes of the deepest leaves.
pub fn find_deepest_nodes(flat_tree: &FlatTree, nb_leaves: usize) -> Vec<usize> {
    // Find the deepest leaves.
    let mut leaves_with_depths: Vec<(usize, f64)> = flat_tree
        .iter(TraversalOrder::PreOrder)
        .enumerate()
        .filter(|(_, node)| node.left_child.is_none() && node.right_child.is_none())
        .map(|(i, node)| (i, node.depth.unwrap()))
        .collect();

    // Sort by depth in descending order.
    leaves_with_depths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take the top `nb_leaves` deepest nodes.
    leaves_with_depths.iter().take(nb_leaves).map(|(i, _)| *i).collect()
}

/// Removes all unsampled leaves and their parents from the flat tree.
///
/// This function takes a flat tree and a vector of indexes of leaves to remove. It removes all the corresponding leaves as well as their parents.
/// The function modifies the tree in place and does not return anything.
///
/// # Arguments
///
/// * `flat_tree` - A mutable reference to the flat tree.
/// * `list_indexes` - A vector of indexes of leaves to remove from the tree.
fn remove_all_unsampled(flat_tree: &mut FlatTree, list_indexes: &Vec<usize>) {
    for index in list_indexes {
        remove_node(flat_tree, *index);
    }
}

/// Extracts all leaves from the flat tree.
///
/// Leaves are nodes without children.
///
/// # Arguments
///
/// * `flat_tree` - A reference to the flat tree.
///
/// # Returns
///
/// A vector containing the indexes of all leaf nodes in the tree.
fn find_all_leaves(flat_tree: &FlatTree) -> Vec<usize> {
    flat_tree
        .iter(TraversalOrder::PreOrder)
        .enumerate()
        .filter(|(_, node)| node.left_child.is_none() && node.right_child.is_none())
        .map(|(i, _)| i)
        .collect()
}

/// Extracts all leaves that are present in the sampled species tree.
///
/// # Arguments
///
/// * `flat_tree` - A reference to the original flat tree.
/// * `flat_sampled_tree` - A reference to the flat tree of the sampled species tree.
///
/// # Returns
///
/// A vector containing the indexes of all leaves in the original tree that are also in the sampled tree.
fn find_all_extant_leaves(flat_tree: &FlatTree, flat_sampled_tree: &FlatTree) -> Vec<usize> {
    let sampled_names: Vec<String> = flat_sampled_tree
        .iter(TraversalOrder::PreOrder)
        .filter(|node| node.left_child.is_none() && node.right_child.is_none())
        .map(|node| node.name.clone())
        .collect();

    flat_tree
        .iter(TraversalOrder::PreOrder)
        .enumerate()
        .filter(|(_, node)| {
            node.left_child.is_none()
                && node.right_child.is_none()
                && sampled_names.contains(&node.name)
        })
        .map(|(i, _)| i)
        .collect()
}

/// Randomly samples a specified number of leaves from the tree.
///
/// This function only samples leaves that are present in both the original tree and the sampled species tree.
///
/// # Arguments
///
/// * `flat_tree` - A reference to the original flat tree.
/// * `flat_sampled_tree` - A reference to the flat tree of the sampled species tree.
/// * `n_sampled_nodes` - The number of leaves to sample.
/// * `rng` - A mutable reference to a random number generator.
///
/// # Returns
///
/// A vector containing the indexes of the sampled leaves.
pub fn sample_random_leaves(
    flat_tree: &FlatTree,
    flat_sampled_tree: &FlatTree,
    n_sampled_nodes: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    let leaves = find_all_extant_leaves(flat_tree, flat_sampled_tree);
    let sampled_leaves: Vec<usize> = leaves
        .choose_multiple(rng, n_sampled_nodes)
        .cloned()
        .collect();
    sampled_leaves
}

/// Samples the leaves with the biggest lengths (diversified sampling).
pub fn sample_diversified_leaves(flat_tree: &FlatTree, n_leaves: usize) -> Vec<usize> {
    let mut leaves_with_length: Vec<(usize, f64)> = flat_tree
        .iter(TraversalOrder::PreOrder)
        .enumerate()
        .filter(|(_, node)| node.left_child.is_none() && node.right_child.is_none())
        .map(|(i, node)| (i, node.length))
        .collect();
    leaves_with_length.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    leaves_with_length.iter().take(n_leaves).map(|(i, _)| *i).collect()
}

/// Samples the leaves with the smallest lengths (clustered sampling).
pub fn sample_clustered_leaves(flat_tree: &FlatTree, n_leaves: usize) -> Vec<usize> {
    let mut leaves_with_length: Vec<(usize, f64)> = flat_tree
        .iter(TraversalOrder::PreOrder)
        .enumerate()
        .filter(|(_, node)| node.left_child.is_none() && node.right_child.is_none())
        .map(|(i, node)| (i, node.length))
        .collect();
    leaves_with_length.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    leaves_with_length.iter().take(n_leaves).map(|(i, _)| *i).collect()
}

/// Determines which leaves should be removed from the tree.
///
/// This function calculates the complement of the sampled leaves within the set of all leaves, effectively identifying the leaves to be removed.
///
/// # Arguments
///
/// * `leaves` - A vector containing the indexes of all leaves in the tree.
/// * `sampled_leaves` - A vector containing the indexes of the sampled leaves.
///
/// # Returns
///
/// A vector containing the indexes of the leaves that are not in the sampled list.
fn leaves_to_be_removed(leaves: &Vec<usize>, sampled_leaves: &Vec<usize>) -> Vec<usize> {
    leaves
        .iter()
        .filter(|leaf| !sampled_leaves.contains(leaf))
        .cloned()
        .collect()
}

/// Finds the index of the root of the tree starting from a given leaf.
///
/// The function traverses up the tree from the given leaf node by following parent pointers until it reaches the root.
///
/// # Arguments
///
/// * `flat_tree` - A reference to the flat tree.
/// * `true_leaf` - The index of a leaf node from which to start the traversal.
///
/// # Returns
///
/// The index of the root node in the flat tree.
fn find_root(flat_tree: &FlatTree, true_leaf: usize) -> usize {
    let mut current_node = true_leaf;
    let mut current_parent = flat_tree[current_node].parent;
    while let Some(parent) = current_parent {
        current_node = parent;
        current_parent = flat_tree[current_node].parent;
    }
    current_node
}

/// Finds the leaves in the gene tree that correspond to the sampled species.
///
/// # Arguments
///
/// * `flat_gene_tree` - A reference to the flat tree of the gene tree.
/// * `leaf_names` - A vector of leaf names to find in the gene tree.
///
/// # Returns
///
/// A vector containing the indexes of the leaves in the gene tree that correspond to the given names.
fn find_leaves_in_gene_tree(flat_gene_tree: &FlatTree, leaf_names: &Vec<String>) -> Vec<usize> {
    flat_gene_tree
        .iter(TraversalOrder::PreOrder)
        .enumerate()
        .filter(|(_, node)| {
            node.left_child.is_none()
                && node.right_child.is_none()
                && leaf_names.contains(&node.name)
        })
        .map(|(i, _)| i)
        .collect()
}

/// Samples a gene tree by removing unsampled leaves and writes the result to a file.
///
/// # Arguments
///
/// * `sampled_leaves_names` - A vector of names of the sampled leaves.
/// * `leaves_to_be_removed_names` - A vector of names of the leaves to be removed.
/// * `gene_tree_path` - The path to the gene tree file.
/// * `gene_index` - The index of the gene tree.
/// * `output_dir` - The output directory where the sampled gene tree will be saved.
///
/// # Returns
///
/// A `Result` containing the Newick string of the sampled gene tree, or an `io::Error` if an error occurs.
fn one_gene_sample_to_string(
    sampled_leaves_names: &Vec<String>,
    leaves_to_be_removed_names: &Vec<String>,
    gene_tree_path: &PathBuf,
    gene_index: u32,
    output_dir: &str,
) -> Result<String, io::Error> {
    // Open the gene tree and convert it to a flat tree.
    let gene_tree_str = fs::read_to_string(&gene_tree_path)
        .unwrap_or_else(|err| panic!("Error reading file '{}': {}", gene_tree_path.to_string_lossy(), err));

    let gene_tree_str = gene_tree_str.trim();

    let mut pairs = NewickParser::parse(Rule::newick, gene_tree_str)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    let mut node_tree = newick_to_tree(pairs.next().unwrap());

    let mut gene_tree = node_tree.pop().unwrap();
    gene_tree.zero_root_length();
    gene_tree.assign_depths(0.0);

    let mut flat_tree = gene_tree.to_flat_tree();

    // Find the indexes of the sampled leaves in the gene tree.
    let sampled_leaves = find_leaves_in_gene_tree(&flat_tree, sampled_leaves_names);
    let leaves_to_be_removed = find_leaves_in_gene_tree(&flat_tree, leaves_to_be_removed_names);

    // Remove unsampled leaves from the gene tree.
    remove_all_unsampled(&mut flat_tree, &leaves_to_be_removed);

    // Ensure the output directory exists
    fs::create_dir_all(output_dir)?;

    // Find the root of the new tree.
    let root_of_gene_tree = find_root(&flat_tree, sampled_leaves[0]);
    flat_tree.root = root_of_gene_tree;

    // Convert the flat tree back to a Node tree.
    let mut reconstructed_tree = flat_tree.to_node();

    // Update lengths based on depths.
    let root_depth = reconstructed_tree
        .depth
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Root depth not found"))?;
    reconstructed_tree.depths_to_lengths(root_depth);

    // Convert the gene tree to a Newick string.
    let reconstructed_newick = node_to_newick(&reconstructed_tree) + ";";

    // Save the gene tree as a Newick string in the output directory
    let gene_filename = format!("sampled_gene_{}.nwk", gene_index);
    let gene_filename = Path::new(output_dir).join(gene_filename);
    let mut gene_file = File::create(gene_filename)?;
    gene_file.write_all(reconstructed_newick.as_bytes())?;

    Ok(reconstructed_newick)
}

/// Samples the species tree and returns the Newick string along with sampled and removed leaf names.
///
/// The function performs the following steps:
/// 1. Reads and parses the species tree and the sampled species tree from files.
/// 2. Converts them to flat trees.
/// 3. Samples random leaves from the species tree that are present in the sampled species tree.
/// 4. Removes unsampled leaves from the species tree.
/// 5. Reconstructs the tree and updates node lengths based on depths.
/// 6. Converts the reconstructed tree to a Newick string and saves it to a file.
///
/// # Arguments
///
/// * `species_tree_path` - The path to the species tree file in Newick format.
/// * `sampled_species_tree_path` - The path to the sampled species tree file in Newick format.
/// * `n_sampled_nodes` - The number of leaves to sample.
/// * `output_dir` - The output directory where the sampled tree will be saved.
/// * `rng` - A mutable reference to a random number generator.
///
/// # Returns
///
/// A `Result` containing:
/// - The Newick string of the sampled species tree.
/// - A vector of names of sampled leaves.
/// - A vector of names of unsampled (removed) leaves.
///
/// If an error occurs, an `io::Error` is returned.
pub fn species_tree_sample_to_string(
    species_tree_path: &str,
    sampled_species_tree_path: &str,
    n_sampled_nodes: usize,
    output_dir: &str,
    rng: &mut StdRng,
) -> Result<(String, Vec<String>, Vec<String>), io::Error> {
    // Ensure the output directory exists
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        fs::create_dir_all(output_path)?;
    }

    // Read the species tree and the sampled species tree
    let species_tree_str = fs::read_to_string(species_tree_path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    let sampled_species_tree_str = fs::read_to_string(sampled_species_tree_path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    let species_tree_str = species_tree_str.trim();
    let sampled_species_tree_str = sampled_species_tree_str.trim();

    // Parse the species trees
    let mut pairs = NewickParser::parse(Rule::newick, species_tree_str)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    let mut node_tree = newick_to_tree(pairs.next().unwrap());
    let mut species_tree = node_tree.pop().unwrap();

    let mut pairs_sampled = NewickParser::parse(Rule::newick, sampled_species_tree_str)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    let mut node_tree_sampled = newick_to_tree(pairs_sampled.next().unwrap());
    let sampled_species_tree = node_tree_sampled.pop().unwrap();

    // Assign depths
    species_tree.zero_root_length();
    species_tree.assign_depths(0.0);
    let mut flat_tree = species_tree.to_flat_tree();

    let flat_sampled_tree = sampled_species_tree.to_flat_tree();

    // Sample random leaves
    let sampled_leaves = sample_random_leaves(&flat_tree, &flat_sampled_tree, n_sampled_nodes, rng);

    // Remove unsampled leaves
    let leaves = find_all_leaves(&flat_tree);
    let leaves_to_be_removed = leaves_to_be_removed(&leaves, &sampled_leaves);
    remove_all_unsampled(&mut flat_tree, &leaves_to_be_removed);

    // Update the root of the flat tree.
    let root_of_species_tree = find_root(&flat_tree, sampled_leaves[0]);
    flat_tree.root = root_of_species_tree;

    // Convert the flat tree back to a Node tree.
    let mut reconstructed_tree = flat_tree.to_node();

    // Update lengths based on depths.
    let root_depth = reconstructed_tree
        .depth
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Root depth not found"))?;
    reconstructed_tree.depths_to_lengths(root_depth);

    // Convert the species tree to a Newick string.
    let reconstructed_newick = node_to_newick(&reconstructed_tree) + ";";

    // Save the species tree as a Newick string in the output directory
    let species_filename = Path::new(output_dir).join("sampled_species_tree.nwk");
    let mut species_file = File::create(species_filename)?;
    species_file.write_all(reconstructed_newick.as_bytes())?;

    // Return the Newick string and the lists of sampled and removed leaf names.
    let sampled_leaves_names: Vec<String> = sampled_leaves
        .iter()
        .map(|i| flat_tree[*i].name.clone())
        .collect();
    let leaves_to_be_removed_names: Vec<String> = leaves_to_be_removed
        .iter()
        .map(|i| flat_tree[*i].name.clone())
        .collect();

    Ok((reconstructed_newick, sampled_leaves_names, leaves_to_be_removed_names))
}

/// Samples all gene trees in the specified range by removing unsampled leaves.
///
/// # Arguments
///
/// * `sampled_leaves_names` - A vector of names of the sampled leaves.
/// * `leaves_to_be_removed_names` - A vector of names of the leaves to be removed.
/// * `start_index` - The starting index of the gene trees to sample.
/// * `end_index` - The ending index of the gene trees to sample.
/// * `gene_trees_path` - The path to the directory containing the gene trees.
/// * `output_dir` - The output directory where the sampled gene trees will be saved.
pub fn sample_all_gene_trees(
    sampled_leaves_names: &Vec<String>,
    leaves_to_be_removed_names: &Vec<String>,
    start_index: usize,
    end_index: usize,
    gene_trees_path: &str,
    output_dir: &str,
) {
    let gene_trees_path = Path::new(gene_trees_path);
    for i in start_index..end_index {
        let gene_tree_filename = format!("gene_{}.nwk", i);
        let gene_tree_path = gene_trees_path.join("genes").join(gene_tree_filename);
        let _ = one_gene_sample_to_string(
            sampled_leaves_names,
            leaves_to_be_removed_names,
            &gene_tree_path,
            i as u32,
            &output_dir,
        );
    }
}

