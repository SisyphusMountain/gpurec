use std::time::Instant;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::{simulate_dtl_batch, simulate_dtl_per_species_batch, count_events};

fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn main() {
    println!("DTL Simulation Benchmark");
    println!("========================");
    println!();

    // Parameters
    let species_sizes = [20, 50, 100, 200];
    let n_gene_trees = 1000;
    let lambda_d = 0.5;
    let lambda_t = 0.2;
    let lambda_l = 0.3;

    println!("DTL rates: λ_d={}, λ_t={}, λ_l={}", lambda_d, lambda_t, lambda_l);
    println!("Gene trees per species tree: {}", fmt_num(n_gene_trees));
    println!();

    // Header
    println!("{:>8} | {:>12} {:>12} | {:>12} {:>12} | {:>8}",
        "Species", "Per-Gene", "trees/sec", "Per-Species", "trees/sec", "Speedup");
    println!("{}", "-".repeat(80));

    for &n_species in &species_sizes {
        // Generate species tree
        let mut rng = StdRng::seed_from_u64(42);
        let (mut species_tree, _) = simulate_bd_tree_bwd(n_species, 1.0, 0.5, &mut rng);
        species_tree.assign_depths();

        // Benchmark per-gene-copy model (simulate_dtl_batch)
        let mut rng1 = StdRng::seed_from_u64(123);
        let start = Instant::now();
        let (trees1, _) = simulate_dtl_batch(
            &species_tree,
            species_tree.root,
            lambda_d,
            lambda_t,
            lambda_l,
            None,
            None,
            n_gene_trees,
            false,
            &mut rng1,
        );
        let per_gene_time = start.elapsed().as_secs_f64();
        let per_gene_rate = n_gene_trees as f64 / per_gene_time;

        // Benchmark per-species model (simulate_dtl_per_species_batch)
        let mut rng2 = StdRng::seed_from_u64(123);
        let start = Instant::now();
        let (trees2, _) = simulate_dtl_per_species_batch(
            &species_tree,
            species_tree.root,
            lambda_d,
            lambda_t,
            lambda_l,
            None,
            None,
            n_gene_trees,
            false,
            &mut rng2,
        );
        let per_species_time = start.elapsed().as_secs_f64();
        let per_species_rate = n_gene_trees as f64 / per_species_time;

        let speedup = per_gene_time / per_species_time;

        println!("{:>8} | {:>10.3}s {:>10.0}/s | {:>10.3}s {:>10.0}/s | {:>7.2}x",
            n_species,
            per_gene_time, per_gene_rate,
            per_species_time, per_species_rate,
            speedup);

        // Count events for comparison
        let mut pg_events = (0usize, 0usize, 0usize, 0usize); // S, D, T, L
        for tree in &trees1 {
            let (s, d, t, l, _) = count_events(tree);
            pg_events.0 += s;
            pg_events.1 += d;
            pg_events.2 += t;
            pg_events.3 += l;
        }

        let mut ps_events = (0usize, 0usize, 0usize, 0usize);
        for tree in &trees2 {
            let (s, d, t, l, _) = count_events(tree);
            ps_events.0 += s;
            ps_events.1 += d;
            ps_events.2 += t;
            ps_events.3 += l;
        }

        println!("         | D:{:>6} T:{:>6} L:{:>6} | D:{:>6} T:{:>6} L:{:>6} |",
            fmt_num(pg_events.1), fmt_num(pg_events.2), fmt_num(pg_events.3),
            fmt_num(ps_events.1), fmt_num(ps_events.2), fmt_num(ps_events.3));
    }

    println!();
    println!("Notes:");
    println!("  - Per-Gene: Rate scales with number of gene copies (more copies = more events)");
    println!("  - Per-Species: Rate is constant per species (Zombi-style, genome-level rate)");
    println!("  - Per-Species typically has fewer events because duplications don't increase rate");

    // High-throughput benchmark
    println!();
    println!("High-Throughput Benchmark (100 species, varying batch sizes)");
    println!("=============================================================");

    let mut rng = StdRng::seed_from_u64(42);
    let (mut species_tree, _) = simulate_bd_tree_bwd(100, 1.0, 0.5, &mut rng);
    species_tree.assign_depths();

    let batch_sizes = [100, 1_000, 10_000];

    println!("{:>10} | {:>12} {:>12} | {:>12} {:>12}",
        "Batch", "Per-Gene", "trees/sec", "Per-Species", "trees/sec");
    println!("{}", "-".repeat(65));

    for &batch_size in &batch_sizes {
        // Per-gene
        let mut rng1 = StdRng::seed_from_u64(123);
        let start = Instant::now();
        let _ = simulate_dtl_batch(
            &species_tree, species_tree.root,
            lambda_d, lambda_t, lambda_l, None,
            None, batch_size, false, &mut rng1,
        );
        let pg_time = start.elapsed().as_secs_f64();
        let pg_rate = batch_size as f64 / pg_time;

        // Per-species
        let mut rng2 = StdRng::seed_from_u64(123);
        let start = Instant::now();
        let _ = simulate_dtl_per_species_batch(
            &species_tree, species_tree.root,
            lambda_d, lambda_t, lambda_l, None,
            None, batch_size, false, &mut rng2,
        );
        let ps_time = start.elapsed().as_secs_f64();
        let ps_rate = batch_size as f64 / ps_time;

        println!("{:>10} | {:>10.3}s {:>10.0}/s | {:>10.3}s {:>10.0}/s",
            fmt_num(batch_size),
            pg_time, pg_rate,
            ps_time, ps_rate);
    }
}
