// Example: Parse a RecPhyloXML file (e.g., from ALERax)
use rustree::{RecTree, Event};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <recphyloxml_file>", args[0]);
        eprintln!("Example: {} alerax_output.xml", args[0]);
        std::process::exit(1);
    }

    let filepath = &args[1];

    println!("Parsing RecPhyloXML file: {}", filepath);

    match RecTree::from_xml_file(filepath) {
        Ok(rec_tree) => {
            println!("\n✓ Successfully parsed RecPhyloXML file!");

            // Display species tree info
            println!("\n=== Species Tree ===");
            println!("  Nodes: {}", rec_tree.species_tree.nodes.len());
            println!("  Root: {}", rec_tree.species_tree.nodes[rec_tree.species_tree.root].name);

            // Count leaves in species tree
            let species_leaves: Vec<_> = rec_tree.species_tree.nodes.iter()
                .filter(|n| n.left_child.is_none() && n.right_child.is_none())
                .collect();
            println!("  Leaves: {}", species_leaves.len());

            // Display gene tree info
            println!("\n=== Gene Tree ===");
            println!("  Nodes: {}", rec_tree.gene_tree.nodes.len());
            println!("  Root: {}", rec_tree.gene_tree.nodes[rec_tree.gene_tree.root].name);

            // Count events
            let speciation_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Speciation).count();
            let duplication_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Duplication).count();
            let transfer_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Transfer).count();
            let loss_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Loss).count();
            let leaf_count = rec_tree.event_mapping.iter().filter(|e| **e == Event::Leaf).count();

            println!("\n=== Event Counts ===");
            println!("  Speciation: {}", speciation_count);
            println!("  Duplication: {}", duplication_count);
            println!("  Transfer: {}", transfer_count);
            println!("  Loss: {}", loss_count);
            println!("  Leaf: {}", leaf_count);
            println!("  Total: {}", rec_tree.event_mapping.len());

            // Display some leaf genes
            println!("\n=== Sample Gene Tree Leaves ===");
            let leaf_nodes: Vec<_> = rec_tree.gene_tree.nodes.iter()
                .enumerate()
                .filter(|(idx, _)| rec_tree.event_mapping[*idx] == Event::Leaf)
                .take(10)
                .collect();

            for (idx, node) in leaf_nodes {
                let species_idx = rec_tree.node_mapping[idx].unwrap();
                let species_name = &rec_tree.species_tree.nodes[species_idx].name;
                println!("  {} → {}", node.name, species_name);
            }

            // Demonstrate XML export
            println!("\n=== XML Export ===");
            let xml_output = rec_tree.to_xml();
            println!("  Generated XML: {} bytes", xml_output.len());

            // Optionally save to file
            if args.len() >= 3 && args[2] == "--save" {
                let output_file = format!("{}.out.xml", filepath);
                std::fs::write(&output_file, &xml_output).expect("Failed to write output file");
                println!("  Saved to: {}", output_file);
            }

            println!("\n✓ Done!");
        }
        Err(e) => {
            eprintln!("\n✗ Failed to parse RecPhyloXML file: {}", e);
            std::process::exit(1);
        }
    }
}
