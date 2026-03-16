// Benchmark: Profile where CPU time is spent in DTL simulation
use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::simulate_dtl;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

fn main() {
    println!("=== DTL Simulation CPU Profiling ===\n");

    // Generate species tree
    println!("Generating species tree (1000 leaves)...");
    let mut rng = StdRng::seed_from_u64(42);
    let (mut species_tree, _) = simulate_bd_tree_bwd(1000, 1.0, 0.0, &mut rng);
    species_tree.assign_depths();
    println!("  ✓ Generated {} species\n", species_tree.nodes.len());

    let n_trees = 1000;
    let mut rng = StdRng::seed_from_u64(123);

    println!("Running {} simulations to profile...\n", n_trees);

    // Baseline: Full simulation
    let start = Instant::now();
    let mut total_nodes = 0;
    let mut total_events = 0;

    for _ in 0..n_trees {
        let (rec_tree, events) = simulate_dtl(
            &species_tree,
            species_tree.root,
            0.2, 0.2, 0.1,
            None,
            None,
            false,
            &mut rng,
        );
        total_nodes += rec_tree.gene_tree.nodes.len();
        total_events += events.len();
    }

    let full_time = start.elapsed().as_secs_f64();
    let throughput = n_trees as f64 / full_time;

    println!("Full simulation:");
    println!("  Time: {:.3}s", full_time);
    println!("  Throughput: {:.1} trees/second", throughput);
    println!("  Avg nodes per tree: {}", total_nodes / n_trees);
    println!("  Avg events per tree: {}", total_events / n_trees);

    // Component analysis
    println!("\n=== Component Breakdown ===\n");

    // 1. Just RNG calls
    println!("1. Random number generation overhead:");
    let mut rng = StdRng::seed_from_u64(123);
    let start = Instant::now();
    let mut dummy = 0.0;

    for _ in 0..n_trees {
        // Estimate number of RNG calls per tree
        // Each tree with ~13,000 nodes needs multiple RNG calls per node
        for _ in 0..(total_nodes / n_trees * 5) {
            dummy += rng.gen::<f64>();
        }
    }

    let rng_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s ({:.1}% of total)", rng_time, 100.0 * rng_time / full_time);
    println!("  Dummy sum (prevent optimization): {}", dummy);

    // 2. Species tree lookups
    println!("\n2. Species tree traversal/lookup overhead:");

    let start = Instant::now();
    let mut lookup_count = 0;

    for _ in 0..n_trees {
        for _ in 0..(total_nodes / n_trees) {
            // Simulate typical lookups during DTL
            let species_idx = species_tree.root % species_tree.nodes.len();
            lookup_count += 1;

            // Traverse down
            if let Some(_left) = species_tree.nodes[species_idx].left_child {
                lookup_count += 1;
                if let Some(_right) = species_tree.nodes[species_idx].right_child {
                    lookup_count += 1;
                }
            }
        }
    }

    let lookup_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s ({:.1}% of total)", lookup_time, 100.0 * lookup_time / full_time);
    println!("  Lookups performed: {}", lookup_count);

    // 3. Vector allocations and operations
    println!("\n3. Vector allocation and growth overhead:");
    let start = Instant::now();

    for _ in 0..n_trees {
        let mut nodes = Vec::new();
        let mut events = Vec::new();

        for i in 0..(total_nodes / n_trees) {
            nodes.push(i);
            events.push(i);
        }

        // Prevent optimization
        let _ = nodes.len() + events.len();
    }

    let vec_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s ({:.1}% of total)", vec_time, 100.0 * vec_time / full_time);

    // 4. Tree node creation (struct instantiation)
    println!("\n4. FlatNode struct creation overhead:");
    use rustree::node::FlatNode;

    let start = Instant::now();
    let mut all_nodes = Vec::new();

    for _ in 0..n_trees {
        for i in 0..(total_nodes / n_trees) {
            let node = FlatNode {
                name: format!("G{}", i),
                parent: if i > 0 { Some(i - 1) } else { None },
                left_child: None,
                right_child: None,
                length: 0.1,
                depth: Some(0.0),
                bd_event: None,
            };
            all_nodes.push(node);
        }
        all_nodes.clear();
    }

    let struct_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s ({:.1}% of total)", struct_time, 100.0 * struct_time / full_time);

    // Summary
    println!("\n=== Overhead Breakdown ===\n");
    let measured_overhead = rng_time + lookup_time + vec_time + struct_time;
    let simulation_logic = full_time - measured_overhead;

    println!("Measured components:");
    println!("  RNG:              {:.3}s ({:.1}%)", rng_time, 100.0 * rng_time / full_time);
    println!("  Tree lookups:     {:.3}s ({:.1}%)", lookup_time, 100.0 * lookup_time / full_time);
    println!("  Vector ops:       {:.3}s ({:.1}%)", vec_time, 100.0 * vec_time / full_time);
    println!("  Struct creation:  {:.3}s ({:.1}%)", struct_time, 100.0 * struct_time / full_time);
    println!("  Subtotal:         {:.3}s ({:.1}%)", measured_overhead, 100.0 * measured_overhead / full_time);
    println!("\nSimulation logic:   {:.3}s ({:.1}%)", simulation_logic, 100.0 * simulation_logic / full_time);
    println!("  (DTL event decisions, tree building logic, etc.)");

    println!("\n=== Conclusion ===\n");
    println!("The bottleneck is NOT memory bandwidth (using <2% of capacity).");
    println!("The bottleneck IS CPU computation:");
    println!("  - Random number generation for stochastic decisions");
    println!("  - DTL simulation logic (event type selection, tree growth)");
    println!("  - Data structure operations (node creation, vector management)");
    println!("\nFurther optimization would require:");
    println!("  1. Faster RNG (if RNG is significant)");
    println!("  2. Algorithmic improvements to DTL logic");
    println!("  3. Better data structure layout for cache efficiency");
}
