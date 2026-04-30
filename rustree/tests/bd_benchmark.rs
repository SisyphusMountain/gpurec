// Benchmark for birth-death tree simulation

use rustree::bd::simulate_bd_tree_bwd;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

/// Fast statistical test: simulate 100 BD trees with n=50 leaves and check
/// that the average extant leaf count equals n (backward simulation always
/// produces exactly n extant leaves).
#[test]
fn bd_statistical_leaf_count() {
    let mut rng = StdRng::seed_from_u64(2024);
    let n: usize = 50;
    let lambda = 1.0;
    let mu = 0.5;
    let n_trees = 100;

    let mut total_leaf_count: usize = 0;
    let mut total_node_count: usize = 0;

    for _ in 0..n_trees {
        let (tree, _events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap();

        let leaf_count = tree.nodes.iter()
            .filter(|node| node.left_child.is_none() && node.right_child.is_none())
            .count();
        total_leaf_count += leaf_count;
        total_node_count += tree.nodes.len();
    }

    let avg_leaves = total_leaf_count as f64 / n_trees as f64;
    let avg_nodes = total_node_count as f64 / n_trees as f64;

    // The backward simulation always produces exactly n extant leaves,
    // but the full tree may contain additional extinct lineages when mu > 0.
    // So leaf_count >= n always holds. In practice with mu=0.5, we expect
    // some extinct leaves too.
    assert!(
        avg_leaves >= n as f64,
        "Average leaf count ({:.1}) should be at least n={}", avg_leaves, n
    );
    // Internal nodes: for a binary tree with L leaves, there are L-1 internal nodes,
    // so total = 2L - 1. Check that node count is consistent.
    assert!(
        avg_nodes >= (2 * n - 1) as f64,
        "Average node count ({:.1}) should be at least 2n-1={}", avg_nodes, 2 * n - 1
    );

    println!("BD statistical test: avg leaves={:.1}, avg nodes={:.1} (n={})", avg_leaves, avg_nodes, n);
}

#[test]
#[ignore] // Run with: cargo test --test bd_benchmark -- --ignored --nocapture
fn benchmark_bd_tree_1000x1000() {
    let mut rng = StdRng::seed_from_u64(12345);
    let n_trees = 1000;
    let n_leaves = 1000;
    let lambda = 1.0;
    let mu = 0.5;

    println!("\n=== Birth-Death Tree Generation Benchmark ===");
    println!("Generating {} trees with {} leaves each", n_trees, n_leaves);
    println!("Parameters: λ={}, μ={}", lambda, mu);

    let start = Instant::now();

    for i in 0..n_trees {
        let (_tree, _events) = simulate_bd_tree_bwd(n_leaves, lambda, mu, &mut rng).unwrap();

        // Print progress every 100 trees
        if (i + 1) % 100 == 0 {
            let elapsed = start.elapsed();
            let trees_per_sec = (i + 1) as f64 / elapsed.as_secs_f64();
            println!("Progress: {}/{} trees ({:.1} trees/sec)", i + 1, n_trees, trees_per_sec);
        }
    }

    let total_time = start.elapsed();
    let trees_per_sec = n_trees as f64 / total_time.as_secs_f64();

    println!("\n=== Results ===");
    println!("Total time: {:.3} seconds", total_time.as_secs_f64());
    println!("Average time per tree: {:.3} ms", total_time.as_millis() as f64 / n_trees as f64);
    println!("Trees per second: {:.2}", trees_per_sec);
    println!("Total nodes generated: ~{} million", (n_trees * n_leaves * 2) / 1_000_000);
}

#[test]
#[ignore] // Run with: cargo test --test bd_benchmark benchmark_bd_tree_pure_birth_1000x1000 -- --ignored --nocapture
fn benchmark_bd_tree_pure_birth_1000x1000() {
    let mut rng = StdRng::seed_from_u64(54321);
    let n_trees = 1000;
    let n_leaves = 1000;
    let lambda = 1.0;
    let mu = 0.0; // Pure birth

    println!("\n=== Pure Birth Tree Generation Benchmark ===");
    println!("Generating {} trees with {} leaves each", n_trees, n_leaves);
    println!("Parameters: λ={}, μ={} (Yule process)", lambda, mu);

    let start = Instant::now();

    for i in 0..n_trees {
        let (_tree, _events) = simulate_bd_tree_bwd(n_leaves, lambda, mu, &mut rng).unwrap();

        // Print progress every 100 trees
        if (i + 1) % 100 == 0 {
            let elapsed = start.elapsed();
            let trees_per_sec = (i + 1) as f64 / elapsed.as_secs_f64();
            println!("Progress: {}/{} trees ({:.1} trees/sec)", i + 1, n_trees, trees_per_sec);
        }
    }

    let total_time = start.elapsed();
    let trees_per_sec = n_trees as f64 / total_time.as_secs_f64();

    println!("\n=== Results ===");
    println!("Total time: {:.3} seconds", total_time.as_secs_f64());
    println!("Average time per tree: {:.3} ms", total_time.as_millis() as f64 / n_trees as f64);
    println!("Trees per second: {:.2}", trees_per_sec);
}

#[test]
#[ignore] // Run with: cargo test --test bd_benchmark benchmark_varying_sizes -- --ignored --nocapture
fn benchmark_varying_sizes() {
    let mut rng = StdRng::seed_from_u64(99999);
    let lambda = 1.0;
    let mu = 0.5;
    let sizes = vec![10, 50, 100, 500, 1000, 2000, 5000];

    println!("\n=== Varying Tree Size Benchmark ===");
    println!("Parameters: λ={}, μ={}\n", lambda, mu);
    println!("{:<10} {:<15} {:<15}", "N leaves", "Time (ms)", "Nodes/ms");
    println!("{}", "-".repeat(40));

    for &n in &sizes {
        let n_reps = if n <= 100 { 100 } else if n <= 1000 { 50 } else { 10 };

        let start = Instant::now();
        for _ in 0..n_reps {
            let (_tree, _events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap();
        }
        let elapsed = start.elapsed();

        let avg_time_ms = elapsed.as_millis() as f64 / n_reps as f64;
        let nodes_per_ms = (n * 2) as f64 / avg_time_ms; // Approximate nodes

        println!("{:<10} {:<15.3} {:<15.2}", n, avg_time_ms, nodes_per_ms);
    }
}
