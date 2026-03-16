// Benchmark for birth-death tree simulation

use rustree::bd::simulate_bd_tree_bwd;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

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
        let (_tree, _events) = simulate_bd_tree_bwd(n_leaves, lambda, mu, &mut rng);

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
        let (_tree, _events) = simulate_bd_tree_bwd(n_leaves, lambda, mu, &mut rng);

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
            let (_tree, _events) = simulate_bd_tree_bwd(n, lambda, mu, &mut rng);
        }
        let elapsed = start.elapsed();

        let avg_time_ms = elapsed.as_millis() as f64 / n_reps as f64;
        let nodes_per_ms = (n * 2) as f64 / avg_time_ms; // Approximate nodes

        println!("{:<10} {:<15.3} {:<15.2}", n, avg_time_ms, nodes_per_ms);
    }
}
