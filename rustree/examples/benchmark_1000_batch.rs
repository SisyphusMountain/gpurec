// Benchmark: 1000 species, 1000 gene trees using batch method
use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::simulate_dtl_batch;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

fn main() {
    println!("=== Large-Scale Batch DTL Benchmark ===");
    println!("Configuration:");
    println!("  Species tree: 1000 leaves");
    println!("  Gene trees: 1000");
    println!("  Parameters: d=0.2, t=0.2, l=0.1");
    println!("  Method: simulate_dtl_batch() with shared depths/contemporaneity\n");

    // Generate species tree
    println!("Phase 1: Generating species tree...");
    let mut rng = StdRng::seed_from_u64(42);
    let start = Instant::now();
    let (mut species_tree, _) = simulate_bd_tree_bwd(1000, 1.0, 0.0, &mut rng).unwrap();
    species_tree.assign_depths();
    let species_time = start.elapsed();
    println!("  ✓ Generated {} species in {:.3}s\n", species_tree.nodes.len(), species_time.as_secs_f64());

    // Simulate gene trees using batch method
    println!("Phase 2: Simulating 1000 DTL gene trees (batch method)...");
    let start = Instant::now();

    let (rec_trees, all_events) = simulate_dtl_batch(
        &species_tree,
        species_tree.root,
        0.2,
        0.2,
        0.1,
        None,
        None,
        1000,
        false,
        &mut rng,
    ).unwrap();

    let simulation_time = start.elapsed();

    let total_nodes: usize = rec_trees.iter().map(|t| t.gene_tree.nodes.len()).sum();
    let total_events: usize = all_events.iter().map(|e| e.len()).sum();
    let min_nodes = rec_trees.iter().map(|t| t.gene_tree.nodes.len()).min().unwrap();
    let max_nodes = rec_trees.iter().map(|t| t.gene_tree.nodes.len()).max().unwrap();

    println!("  ✓ Completed in {:.3}s\n", simulation_time.as_secs_f64());

    // Statistics
    println!("=== Results ===\n");
    println!("Timing:");
    println!("  Species tree generation: {:.3}s", species_time.as_secs_f64());
    println!("  Gene tree simulations:   {:.3}s", simulation_time.as_secs_f64());
    println!("  Total time:              {:.3}s\n", (species_time + simulation_time).as_secs_f64());

    println!("Performance:");
    println!("  Throughput: {:.1} trees/second", 1000.0 / simulation_time.as_secs_f64());
    println!("  Avg time per tree: {:.3} ms\n", simulation_time.as_secs_f64() * 1000.0 / 1000.0);

    println!("Gene tree statistics:");
    println!("  Total nodes: {}", total_nodes);
    println!("  Avg nodes per tree: {}", total_nodes / 1000);
    println!("  Min/Max nodes: {} / {}", min_nodes, max_nodes);
    println!("  Total events: {}", total_events);
    println!("  Avg events per tree: {}", total_events / 1000);

    println!("\n=== Improvement ===");
    println!("Previous (individual simulate_dtl): ~4.8s for 1000 trees");
    println!("Current (batch method): {:.3}s for 1000 trees", simulation_time.as_secs_f64());
    println!("Speedup: ~{:.1}x faster", 4.8 / simulation_time.as_secs_f64());
}
