// Benchmark: Multi-core with dynamic work distribution (work stealing)
use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::simulate_dtl;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;
use std::thread;
use std::sync::{Arc, Mutex};

fn main() {
    println!("=== Multi-Core with Dynamic Work Distribution ===\n");

    let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("Available CPUs: {}\n", num_cpus);

    // Generate species tree
    println!("Generating species tree (1000 leaves)...");
    let mut rng = StdRng::seed_from_u64(42);
    let (mut species_tree, _) = simulate_bd_tree_bwd(1000, 1.0, 0.0, &mut rng).unwrap();
    species_tree.assign_depths();
    println!("  ✓ Generated {} species\n", species_tree.nodes.len());

    let n_trees = 1000;

    for num_threads in [1, 2, 4, 8, 16, num_cpus].iter() {
        println!("=== Testing with {} thread{} ===", num_threads, if *num_threads > 1 { "s" } else { "" });

        let species_tree_arc = Arc::new(species_tree.clone());

        // Method 1: Static partitioning (baseline)
        println!("Method 1: Static partitioning");
        let trees_per_thread = n_trees / num_threads;

        let start = Instant::now();
        let mut handles = vec![];

        for thread_id in 0..*num_threads {
            let species_tree_clone = Arc::clone(&species_tree_arc);

            let handle = thread::spawn(move || {
                let mut rng_thread = StdRng::seed_from_u64(123 + thread_id as u64);
                let mut nodes = 0;
                let mut events = 0;

                for _ in 0..trees_per_thread {
                    let (rec_tree, tree_events) = simulate_dtl(
                        &species_tree_clone,
                        species_tree_clone.root,
                        0.2, 0.2, 0.1,
                        None,
                        None,
                        false,
                        &mut rng_thread,
                    ).unwrap();
                    nodes += rec_tree.gene_tree.nodes.len();
                    events += tree_events.len();
                }

                (nodes, events)
            });

            handles.push(handle);
        }

        for handle in handles {
            let (_n, _e) = handle.join().unwrap();
        }

        let static_time = start.elapsed().as_secs_f64();
        println!("  Time: {:.3}s", static_time);
        println!("  Throughput: {:.1} trees/second", n_trees as f64 / static_time);

        // Method 2: Dynamic work distribution
        println!("\nMethod 2: Dynamic work distribution");

        // Shared work queue (index of next tree to process)
        let next_tree = Arc::new(Mutex::new(0usize));

        let start = Instant::now();
        let mut handles = vec![];

        for thread_id in 0..*num_threads {
            let species_tree_clone = Arc::clone(&species_tree_arc);
            let next_tree_clone = Arc::clone(&next_tree);

            let handle = thread::spawn(move || {
                let mut rng_thread = StdRng::seed_from_u64(123 + thread_id as u64);
                let mut nodes = 0;
                let mut events = 0;
                let mut trees_done = 0;

                loop {
                    // Grab next tree index
                    let _tree_idx = {
                        let mut idx = next_tree_clone.lock().unwrap();
                        if *idx >= n_trees {
                            break;
                        }
                        let current = *idx;
                        *idx += 1;
                        current
                    };

                    // Process this tree
                    let (rec_tree, tree_events) = simulate_dtl(
                        &species_tree_clone,
                        species_tree_clone.root,
                        0.2, 0.2, 0.1,
                        None,
                        None,
                        false,
                        &mut rng_thread,
                    ).unwrap();
                    nodes += rec_tree.gene_tree.nodes.len();
                    events += tree_events.len();
                    trees_done += 1;
                }

                (nodes, events, trees_done)
            });

            handles.push(handle);
        }

        let mut per_thread_counts = Vec::new();

        for handle in handles {
            let (_n, _e, count) = handle.join().unwrap();
            per_thread_counts.push(count);
        }

        let dynamic_time = start.elapsed().as_secs_f64();
        println!("  Time: {:.3}s", dynamic_time);
        println!("  Throughput: {:.1} trees/second", n_trees as f64 / dynamic_time);

        // Show work distribution
        let min_count = *per_thread_counts.iter().min().unwrap();
        let max_count = *per_thread_counts.iter().max().unwrap();
        let avg_count = per_thread_counts.iter().sum::<usize>() as f64 / per_thread_counts.len() as f64;
        println!("  Work distribution: min={}, max={}, avg={:.1}", min_count, max_count, avg_count);
        println!("  Balance: {:.1}% (max/avg - 1)", 100.0 * (max_count as f64 / avg_count - 1.0));

        // Comparison
        let improvement = static_time / dynamic_time;
        println!("\nImprovement: {:.2}x faster ({:.1}% speedup)\n",
                 improvement, 100.0 * (improvement - 1.0));

        println!("{}\n", "=".repeat(70));
    }

    println!("=== Conclusion ===\n");
    println!("Dynamic work distribution eliminates load imbalance by letting threads");
    println!("grab work as needed, rather than pre-assigning fixed chunks.");
    println!("This is especially important when work units have variable complexity.");
}
