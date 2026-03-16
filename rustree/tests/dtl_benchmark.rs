// Comprehensive benchmark for DTL (Duplication-Transfer-Loss) simulation

use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::{simulate_dtl, count_events, count_extant_genes};
use rustree::newick::newick::parse_newick;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

// Helper function to create a species tree from Newick
fn create_species_tree(newick: &str) -> rustree::node::FlatTree {
    let mut nodes = parse_newick(newick).unwrap();
    let root = nodes.pop().expect("No tree found");
    let mut tree = root.to_flat_tree();
    tree.assign_depths();
    tree
}

#[test]
#[ignore] // Run with: cargo test --test dtl_benchmark benchmark_dtl_varying_rates -- --ignored --nocapture
fn benchmark_dtl_varying_rates() {
    println!("\n=== DTL Simulation: Varying Event Rates ===");
    println!("Species tree: 10 species (generated via birth-death)");
    println!("Simulations per configuration: 100\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Generate a species tree once
    let (mut species_tree, _) = simulate_bd_tree_bwd(10, 1.0, 0.3, &mut rng);
    species_tree.assign_depths();

    // Test configurations: (lambda_d, lambda_t, lambda_l, label)
    let configs = vec![
        (0.0, 0.0, 0.0, "Pure speciation"),
        (1.0, 0.0, 0.0, "High duplication"),
        (0.0, 1.0, 0.0, "High transfer"),
        (0.0, 0.0, 1.0, "High loss"),
        (1.0, 1.0, 0.0, "Dup+Trans"),
        (1.0, 0.0, 1.0, "Dup+Loss"),
        (0.0, 1.0, 1.0, "Trans+Loss"),
        (1.0, 1.0, 1.0, "All events"),
        (0.5, 0.5, 0.5, "Balanced"),
        (2.0, 2.0, 0.5, "High D+T, low L"),
    ];

    println!("{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Configuration", "Time(ms)", "S", "D", "T", "L", "Leaves", "Extant");
    println!("{}", "-".repeat(100));

    for (lambda_d, lambda_t, lambda_l, label) in configs {
        let n_reps = 100;
        let mut total_s = 0;
        let mut total_d = 0;
        let mut total_t = 0;
        let mut total_l = 0;
        let mut total_leaves = 0;
        let mut total_extant = 0;

        let start = Instant::now();
        for _ in 0..n_reps {
            let (rec_tree, _events) = simulate_dtl(
                &species_tree,
                species_tree.root,
                lambda_d,
                lambda_t,
                lambda_l,
                None,
                None,
                false,
                &mut rng,
            ).unwrap();

            let (s, d, t, l, leaves) = count_events(&rec_tree);
            total_s += s;
            total_d += d;
            total_t += t;
            total_l += l;
            total_leaves += leaves;
            total_extant += count_extant_genes(&rec_tree);
        }
        let elapsed = start.elapsed();

        let avg_time_ms = elapsed.as_millis() as f64 / n_reps as f64;
        let avg_s = total_s / n_reps;
        let avg_d = total_d / n_reps;
        let avg_t = total_t / n_reps;
        let avg_l = total_l / n_reps;
        let avg_leaves = total_leaves / n_reps;
        let avg_extant = total_extant / n_reps;

        println!("{:<20} {:>10.3} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            label, avg_time_ms, avg_s, avg_d, avg_t, avg_l, avg_leaves, avg_extant);
    }
}

#[test]
#[ignore] // Run with: cargo test --test dtl_benchmark benchmark_dtl_varying_species_tree_size -- --ignored --nocapture
fn benchmark_dtl_varying_species_tree_size() {
    println!("\n=== DTL Simulation: Varying Species Tree Size ===");
    println!("Event rates: λ_d=1.0, λ_t=0.5, λ_l=0.5");
    println!("Simulations per tree size: 50\n");

    let mut rng = StdRng::seed_from_u64(12345);
    let lambda_d = 1.0;
    let lambda_t = 0.5;
    let lambda_l = 0.5;
    let sizes = vec![5, 10, 20, 50, 100, 200];

    println!("{:<15} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "N species", "Time(ms)", "Events", "S", "D", "T", "L", "Leaves");
    println!("{}", "-".repeat(95));

    for &n_species in &sizes {
        let n_reps = if n_species <= 50 { 50 } else { 20 };
        let mut total_time = 0.0;
        let mut total_events = 0;
        let mut total_s = 0;
        let mut total_d = 0;
        let mut total_t = 0;
        let mut total_l = 0;
        let mut total_leaves = 0;

        for _ in 0..n_reps {
            // Generate a new species tree for each replicate
            let (mut species_tree, _) = simulate_bd_tree_bwd(n_species, 1.0, 0.3, &mut rng);
            species_tree.assign_depths();

            let start = Instant::now();
            let (rec_tree, events) = simulate_dtl(
                &species_tree,
                species_tree.root,
                lambda_d,
                lambda_t,
                lambda_l,
                None,
                None,
                false,
                &mut rng,
            ).unwrap();
            let elapsed = start.elapsed();

            total_time += elapsed.as_secs_f64() * 1000.0;
            total_events += events.len();

            let (s, d, t, l, leaves) = count_events(&rec_tree);
            total_s += s;
            total_d += d;
            total_t += t;
            total_l += l;
            total_leaves += leaves;
        }

        let avg_time_ms = total_time / n_reps as f64;
        let avg_events = total_events / n_reps;
        let avg_s = total_s / n_reps;
        let avg_d = total_d / n_reps;
        let avg_t = total_t / n_reps;
        let avg_l = total_l / n_reps;
        let avg_leaves = total_leaves / n_reps;

        println!("{:<15} {:>12.3} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
            n_species, avg_time_ms, avg_events, avg_s, avg_d, avg_t, avg_l, avg_leaves);
    }
}

#[test]
#[ignore] // Run with: cargo test --test dtl_benchmark benchmark_dtl_scalability -- --ignored --nocapture
fn benchmark_dtl_scalability() {
    println!("\n=== DTL Simulation: Scalability Test ===");
    println!("Testing performance with increasing event rates");
    println!("Species tree: 20 species, 10 replicates per configuration\n");

    let mut rng = StdRng::seed_from_u64(99999);
    let (mut species_tree, _) = simulate_bd_tree_bwd(20, 1.0, 0.3, &mut rng);
    species_tree.assign_depths();

    // Test with increasing event rates
    let rate_multipliers = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

    println!("{:<15} {:>12} {:>12} {:>12} {:>12}",
        "Rate mult.", "Time(ms)", "Total events", "Events/ms", "Leaves");
    println!("{}", "-".repeat(65));

    for &mult in &rate_multipliers {
        let lambda_d = 1.0 * mult;
        let lambda_t = 0.5 * mult;
        let lambda_l = 0.3 * mult;
        let n_reps = 10;

        let mut total_time = 0.0;
        let mut total_events = 0;
        let mut total_leaves = 0;

        for _ in 0..n_reps {
            let start = Instant::now();
            let (rec_tree, events) = simulate_dtl(
                &species_tree,
                species_tree.root,
                lambda_d,
                lambda_t,
                lambda_l,
                None,
                None,
                false,
                &mut rng,
            ).unwrap();
            let elapsed = start.elapsed();

            total_time += elapsed.as_secs_f64() * 1000.0;
            total_events += events.len();
            total_leaves += count_events(&rec_tree).4;
        }

        let avg_time_ms = total_time / n_reps as f64;
        let avg_events = total_events / n_reps;
        let avg_leaves = total_leaves / n_reps;
        let events_per_ms = avg_events as f64 / avg_time_ms;

        println!("{:<15} {:>12.3} {:>12} {:>12.2} {:>12}",
            format!("{}x", mult), avg_time_ms, avg_events, events_per_ms, avg_leaves);
    }
}

#[test]
#[ignore] // Run with: cargo test --test dtl_benchmark benchmark_dtl_transfer_intensity -- --ignored --nocapture
fn benchmark_dtl_transfer_intensity() {
    println!("\n=== DTL Simulation: Transfer Event Analysis ===");
    println!("Varying transfer rates to study HGT patterns");
    println!("Species tree: 15 species, λ_d=1.0, λ_l=0.3, 50 reps\n");

    let mut rng = StdRng::seed_from_u64(54321);
    let (mut species_tree, _) = simulate_bd_tree_bwd(15, 1.0, 0.3, &mut rng);
    species_tree.assign_depths();

    let lambda_d = 1.0;
    let lambda_l = 0.3;
    let transfer_rates = vec![0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

    println!("{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "λ_transfer", "Time(ms)", "Transfers", "T/(S+D+T)", "Leaves", "T/Leaf");
    println!("{}", "-".repeat(75));

    for &lambda_t in &transfer_rates {
        let n_reps = 50;
        let mut total_time = 0.0;
        let mut total_transfers = 0;
        let mut total_events = 0;
        let mut total_leaves = 0;

        for _ in 0..n_reps {
            let start = Instant::now();
            let (rec_tree, _events) = simulate_dtl(
                &species_tree,
                species_tree.root,
                lambda_d,
                lambda_t,
                lambda_l,
                None,
                None,
                false,
                &mut rng,
            ).unwrap();
            let elapsed = start.elapsed();

            total_time += elapsed.as_secs_f64() * 1000.0;

            let (s, d, t, _l, leaves) = count_events(&rec_tree);
            total_transfers += t;
            total_events += s + d + t;
            total_leaves += leaves;
        }

        let avg_time_ms = total_time / n_reps as f64;
        let avg_transfers = total_transfers / n_reps;
        let avg_leaves = total_leaves / n_reps;
        let transfer_ratio = if total_events > 0 {
            total_transfers as f64 / total_events as f64
        } else {
            0.0
        };
        let transfers_per_leaf = if avg_leaves > 0 {
            avg_transfers as f64 / avg_leaves as f64
        } else {
            0.0
        };

        println!("{:<12.1} {:>12.3} {:>12} {:>12.3} {:>12} {:>12.3}",
            lambda_t, avg_time_ms, avg_transfers, transfer_ratio, avg_leaves, transfers_per_leaf);
    }
}

#[test]
#[ignore] // Run with: cargo test --test dtl_benchmark benchmark_dtl_loss_impact -- --ignored --nocapture
fn benchmark_dtl_loss_impact() {
    println!("\n=== DTL Simulation: Gene Loss Impact Analysis ===");
    println!("Varying loss rates to study gene family extinction");
    println!("Species tree: 10 species, λ_d=1.0, λ_t=0.5, 100 reps\n");

    let mut rng = StdRng::seed_from_u64(77777);
    let (mut species_tree, _) = simulate_bd_tree_bwd(10, 1.0, 0.3, &mut rng);
    species_tree.assign_depths();

    let lambda_d = 1.0;
    let lambda_t = 0.5;
    let loss_rates = vec![0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0];

    println!("{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "λ_loss", "Time(ms)", "Losses", "Extant", "Extinct(%)", "L/(D+T+L)");
    println!("{}", "-".repeat(75));

    for &lambda_l in &loss_rates {
        let n_reps = 100;
        let mut total_time = 0.0;
        let mut total_losses = 0;
        let mut total_extant = 0;
        let mut extinctions = 0;
        let mut total_dtl = 0;

        for _ in 0..n_reps {
            let start = Instant::now();
            let (rec_tree, _events) = simulate_dtl(
                &species_tree,
                species_tree.root,
                lambda_d,
                lambda_t,
                lambda_l,
                None,
                None,
                false,
                &mut rng,
            ).unwrap();
            let elapsed = start.elapsed();

            total_time += elapsed.as_secs_f64() * 1000.0;

            let (_s, d, t, l, _leaves) = count_events(&rec_tree);
            let extant = count_extant_genes(&rec_tree);

            total_losses += l;
            total_extant += extant;
            if extant == 0 {
                extinctions += 1;
            }
            total_dtl += d + t + l;
        }

        let avg_time_ms = total_time / n_reps as f64;
        let avg_losses = total_losses / n_reps;
        let avg_extant = total_extant / n_reps;
        let extinction_rate = (extinctions as f64 / n_reps as f64) * 100.0;
        let loss_ratio = if total_dtl > 0 {
            total_losses as f64 / total_dtl as f64
        } else {
            0.0
        };

        println!("{:<12.1} {:>12.3} {:>12} {:>12} {:>12.1} {:>12.3}",
            lambda_l, avg_time_ms, avg_losses, avg_extant, extinction_rate, loss_ratio);
    }
}

#[test]
#[ignore] // Run with: cargo test --test dtl_benchmark benchmark_dtl_large_scale -- --ignored --nocapture
fn benchmark_dtl_large_scale() {
    println!("\n=== DTL Simulation: Large-Scale Performance ===");
    println!("Testing with large species trees");
    println!("Event rates: λ_d=1.0, λ_t=0.5, λ_l=0.3\n");

    let mut rng = StdRng::seed_from_u64(11111);
    let lambda_d = 1.0;
    let lambda_t = 0.5;
    let lambda_l = 0.3;

    let sizes = vec![50, 100, 200, 500, 1000];

    println!("{:<15} {:>15} {:>15} {:>15} {:>15}",
        "N species", "Time(s)", "Total events", "Gene nodes", "Events/sec");
    println!("{}", "-".repeat(75));

    for &n_species in &sizes {
        let n_reps = if n_species <= 200 { 10 } else { 3 };
        let mut total_time = 0.0;
        let mut total_events = 0;
        let mut total_nodes = 0;

        for _ in 0..n_reps {
            // Generate species tree
            let (mut species_tree, _) = simulate_bd_tree_bwd(n_species, 1.0, 0.3, &mut rng);
            species_tree.assign_depths();

            let start = Instant::now();
            let (rec_tree, events) = simulate_dtl(
                &species_tree,
                species_tree.root,
                lambda_d,
                lambda_t,
                lambda_l,
                None,
                None,
                false,
                &mut rng,
            ).unwrap();
            let elapsed = start.elapsed();

            total_time += elapsed.as_secs_f64();
            total_events += events.len();
            total_nodes += rec_tree.gene_tree.nodes.len();
        }

        let avg_time_s = total_time / n_reps as f64;
        let avg_events = total_events / n_reps;
        let avg_nodes = total_nodes / n_reps;
        let events_per_sec = avg_events as f64 / avg_time_s;

        println!("{:<15} {:>15.3} {:>15} {:>15} {:>15.1}",
            n_species, avg_time_s, avg_events, avg_nodes, events_per_sec);
    }
}

#[test]
fn benchmark_dtl_quick_test() {
    println!("\n=== DTL Quick Benchmark (always runs) ===");
    println!("10 species tree, 20 replicates with balanced rates\n");

    let mut rng = StdRng::seed_from_u64(42);
    let (mut species_tree, _) = simulate_bd_tree_bwd(10, 1.0, 0.3, &mut rng);
    species_tree.assign_depths();

    let lambda_d = 1.0;
    let lambda_t = 0.5;
    let lambda_l = 0.5;
    let n_reps = 20;

    let mut total_time = 0.0;
    let mut total_events = 0;

    for _ in 0..n_reps {
        let start = Instant::now();
        let (_rec_tree, events) = simulate_dtl(
            &species_tree,
            species_tree.root,
            lambda_d,
            lambda_t,
            lambda_l,
            None,
            None,
            false,
            &mut rng,
        ).unwrap();
        let elapsed = start.elapsed();
        total_time += elapsed.as_secs_f64() * 1000.0;
        total_events += events.len();
    }

    let avg_time_ms = total_time / n_reps as f64;
    let avg_events = total_events / n_reps;

    println!("Average time per simulation: {:.3} ms", avg_time_ms);
    println!("Average events per simulation: {}", avg_events);
    println!("Simulations per second: {:.1}", 1000.0 / avg_time_ms);
}
