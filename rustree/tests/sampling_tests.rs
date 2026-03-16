use std::fs;
use std::path::PathBuf;
use rustree::comparison::compare_nodes;
use rustree::newick::newick::parse_newick;
use rustree::sampling::{extract_induced_subtree, find_leaf_indices_by_names};

#[test]
fn test_extract_induced_subtree_keep_cd() {
    // Original tree: ((A:0.1,B:0.2)F:1.0,(C:0.3,D:0.4)E:0.5)G:0.0;
    // Keep only C and D -> should get (C:0.3,D:0.4)E:0.5; (same as test_tree_F.nwk)

    // Read the Newick string from the test file
    let newick_path = PathBuf::from("tests/test_tree.nwk");
    let newick_str = fs::read_to_string(newick_path).expect("Failed to read Newick file");

    // Parse the Newick string into a Node tree
    let mut node_tree = parse_newick(&newick_str).expect("Failed to parse Newick string");
    let mut root_node = node_tree.pop().unwrap();

    // Convert the Node tree to a FlatTree
    root_node.zero_root_length();
    root_node.assign_depths(0.0);
    let flat_tree = root_node.to_flat_tree();

    // Keep only leaves C and D
    let leaves_to_keep = find_leaf_indices_by_names(
        &flat_tree,
        &["C".to_string(), "D".to_string()]
    );

    // Extract induced subtree
    let (induced, _) = extract_induced_subtree(&flat_tree, &leaves_to_keep)
        .expect("Failed to extract induced subtree");

    // Reconstruct the tree
    let mut reconstructed_tree = induced.to_node();
    reconstructed_tree.assign_depths(0.0);
    let newick_result = reconstructed_tree.to_newick().expect("Failed to convert to Newick") + ";";
    println!("Resulting Newick string: {}", newick_result);

    // Read expected tree from test_tree_F.nwk
    let expected_path = PathBuf::from("tests/test_tree_F.nwk");
    let expected_newick = fs::read_to_string(expected_path)
        .expect("Failed to read expected Newick file");
    let mut expected_tree = parse_newick(&expected_newick)
        .expect("Failed to parse expected Newick string").pop().unwrap();
    expected_tree.assign_depths(0.0);

    // Display both trees
    let recon_newick = reconstructed_tree.to_newick().expect("Failed to convert to Newick") + ";";
    let expected_recon_newick = expected_tree.to_newick().expect("Failed to convert to Newick") + ";";
    println!("Reconstructed Newick string: {}", recon_newick);
    println!("Expected Newick string: {}", expected_recon_newick);

    // Compare the reconstructed tree and the expected tree
    match compare_nodes(&reconstructed_tree, &expected_tree, false, 0.0) {
        Ok(true) => println!("Reconstructed tree matches the expected tree."),
        Ok(false) => {
            println!("Reconstructed Node: {:?}", reconstructed_tree);
            println!("Expected Node: {:?}", expected_tree);
            panic!("Reconstructed tree does not match the expected tree.");
        }
        Err(e) => panic!("Error comparing trees: {}", e),
    }
}
