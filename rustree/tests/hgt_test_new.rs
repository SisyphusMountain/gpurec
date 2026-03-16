use std::fs;
use std::path::PathBuf;
use rustree::newick::newick::parse_newick;
use rustree::node::Node;
use rustree::comparison::compare_nodes_topology; // Import from comparison module
use rustree::surgery::spr_topology; // Import the surgery module function
use rustree::debug::diffed_flat_tree_table;


fn run_spr_test(moved_node_name: &str, receiver_name: &str, expected_filename: &str) {
    // Read the original tree file.
    let orig_path = PathBuf::from("./tests/hgt_trees/original_tree.nwk");
    let orig_str = fs::read_to_string(&orig_path)
        .expect(&format!("Failed to read original tree file: {:?}", orig_path));

    // Parse the original tree.
    let mut nodes = parse_newick(orig_str.trim())
        .expect("Failed to parse original tree");
    let mut orig_tree = nodes.pop().expect("No tree found in original file");

    // Convert to a flat tree.
    let mut flat_tree = orig_tree.to_flat_tree();
    println!("Original tree (Root: {}):", flat_tree.nodes[flat_tree.root].name);
    let old_table = diffed_flat_tree_table(&flat_tree, None, false); // Keep for diffing if useful

    // Locate donor and receiver nodes by name.
    let donor_index = flat_tree.find_node_index(moved_node_name) // Use the helper method
        .unwrap_or_else(|| panic!("Donor node '{}' not found", moved_node_name));
    let receiver_index = flat_tree.find_node_index(receiver_name) // Use the helper method
        .unwrap_or_else(|| panic!("Receiver node '{}' not found", receiver_name));

    println!("Performing SPR: Donor='{}'({}), Recipient='{}'({})",
        moved_node_name, donor_index, receiver_name, receiver_index);

    // --- Perform the SPR event using the surgery module function ---
    spr_topology(&mut flat_tree, donor_index, receiver_index)
        .expect(&format!("SPR operation failed for {} -> {}", moved_node_name, receiver_name));
    // --- SPR function should update flat_tree.root internally if needed ---

    // For testing topology, set all branch lengths to 1.0 AFTER SPR.
    // This ensures differences aren't due to length calculations we aren't testing.
    for node in flat_tree.nodes.iter_mut() {
        node.length = 1.0; // Or any constant value
    }

    // --- Remove manual root update - rely on flat_tree.spr having updated flat_tree.root ---
    // let new_root_index = flat_tree.nodes.iter().position(|node| node.parent.is_none())
    //     .expect("No root found after SPR");
    // flat_tree.root = new_root_index;
    // println!("Root after SPR (reported by FlatTree): {}", flat_tree.nodes[flat_tree.root].name);

    println!("Modified tree (Root: {}):", flat_tree.nodes[flat_tree.root].name); // Print root name from flat_tree.root
    let _new_table = diffed_flat_tree_table(&flat_tree, Some(&old_table[1..]), true);

    // Reconstruct the modified tree using the (potentially updated) root index from flat_tree.
    let computed_tree = flat_tree.to_node();

    // Read and parse the expected tree.
    let expected_path = PathBuf::from(format!("./tests/hgt_trees/{}", expected_filename));
    let expected_str = fs::read_to_string(&expected_path)
        .expect(&format!("Failed to read expected tree file: {:?}", expected_path));
    let mut expected_nodes = parse_newick(expected_str.trim())
        .expect("Failed to parse expected tree");
    let expected_tree = expected_nodes.pop().expect("No expected tree found");

    // Compare the computed tree with the expected tree topology.
    match compare_nodes_topology(&computed_tree, &expected_tree) {
        Ok(true) => println!("SPR test passed for {} -> {}", moved_node_name, receiver_name),
        Ok(false) => {
            // Generate Newick strings for easier visual comparison in panic message
            let computed_newick = computed_tree.to_newick().unwrap_or_else(|e| format!("<error: {}>", e));
            let expected_newick = expected_tree.to_newick().unwrap_or_else(|e| format!("<error: {}>", e));

            println!("Computed tree topology:\n{}", computed_newick);
            println!("Expected tree topology:\n{}", expected_newick);
            panic!("SPR tree topology for donor '{}' -> receiver '{}' does not match expected file '{}'.",
                   moved_node_name, receiver_name, expected_filename);
        }
        Err(e) => panic!("Error comparing tree topologies: {}", e),
    }
}



#[test]
fn test_spr_3_to_2() {
    // For donor "3" and receiver "2", expected topology is in 3_to_2.nwk.
    run_spr_test("2", "3", "3_to_2.nwk");
}

#[test]
fn test_spr_1_to_2() {
    // For donor "1" and receiver "2", expected topology is in 1_to_2.nwk.
    run_spr_test("2", "1", "1_to_2.nwk");
}

#[test]
fn test_spr_4_to_T10() {
    // For donor "4" and receiver "T10", expected topology is in 4_to_T10.nwk.
    run_spr_test("T10", "4", "4_to_T10.nwk");
}

#[test]
fn test_spr_4_to_5() {
    // For donor "4" and receiver "5", expected topology is in 4_to_5.nwk.
    run_spr_test("5", "4", "4_to_5.nwk");
}

#[test]
fn test_spr_0_to_T7() {
    // For donor "0" and receiver "T7", expected topology is in 0_to_T7.nwk.
    run_spr_test("T7", "0", "0_to_T7.nwk");
}