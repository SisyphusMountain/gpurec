// Example: Simulate DTL gene tree and export to XML and CSV
use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::{simulate_dtl, save_events_to_csv};
use rustree::newick::newick::parse_newick;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // Option 1: Use a predefined species tree from Newick
    let newick = "((A:1,B:1)AB:1,C:2)root:0;";
    let mut nodes = parse_newick(newick).unwrap();
    let species_tree_node = nodes.pop().expect("No tree found");
    let mut species_tree = species_tree_node.to_flat_tree();
    species_tree.assign_depths();

    // Simulate a gene tree with DTL events
    // Parameters: duplication rate=1.0, transfer rate=0.5, loss rate=0.5
    let (rec_tree, events) = simulate_dtl(&species_tree, species_tree.root, 1.0, 0.5, 0.5, None, None, false, &mut rng);

    // Export to XML with branch lengths
    let xml = rec_tree.to_xml();
    fs::write("output_rectree.xml", &xml).expect("Failed to write XML");
    println!("Reconciled tree exported to output_rectree.xml");

    // Save events to CSV
    save_events_to_csv(&events, &species_tree, &rec_tree.gene_tree, "output_dtl_events.csv").expect("Failed to write events CSV");
    println!("DTL events exported to output_dtl_events.csv ({} events)", events.len());

    // Option 2: Generate a random species tree using birth-death process
    let mut rng2 = StdRng::seed_from_u64(123);
    let (mut bd_tree, _bd_events) = simulate_bd_tree_bwd(5, 1.0, 0.3, &mut rng2);
    bd_tree.assign_depths();

    // Simulate gene tree on the random species tree
    let (rec_tree2, events2) = simulate_dtl(&bd_tree, bd_tree.root, 2.0, 1.0, 1.0, None, None, false, &mut rng2);

    // Export to XML
    let xml2 = rec_tree2.to_xml();
    fs::write("output_rectree_bd.xml", &xml2).expect("Failed to write XML");
    println!("Random species tree reconciliation exported to output_rectree_bd.xml");

    // Save events to CSV
    save_events_to_csv(&events2, &bd_tree, &rec_tree2.gene_tree, "output_dtl_bd_events.csv").expect("Failed to write events CSV");
    println!("DTL events (BD tree) exported to output_dtl_bd_events.csv ({} events)", events2.len());
}
