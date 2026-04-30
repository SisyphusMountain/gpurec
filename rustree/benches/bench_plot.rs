use std::fs;
use std::io::Write;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::{simulate_dtl, simulate_dtl_per_species};
use rustree::FlatTree;

const N_SAMPLES: usize = 15;
const TARGET_SECS: f64 = 0.05;
const OUTPUT_DIR: &str = "target/bench_plots";

fn make_species_tree(n: usize, seed: u64) -> FlatTree {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut tree, _) = simulate_bd_tree_bwd(n, 1.0, 0.3, &mut rng).unwrap();
    tree.assign_depths();
    tree
}

/// Measure throughput (items/sec) by running `work` N_SAMPLES times.
/// `work` produces `items_per_call` trees/sims per invocation.
fn measure_rate<F: FnMut()>(mut work: F, items_per_call: usize) -> (f64, f64, f64) {
    // warmup
    for _ in 0..3 {
        work();
    }

    let mut rates = Vec::with_capacity(N_SAMPLES);
    for _ in 0..N_SAMPLES {
        let start = Instant::now();
        work();
        let elapsed = start.elapsed().as_secs_f64();
        rates.push(items_per_call as f64 / elapsed);
    }

    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;
    let var: f64 = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let ci = 1.96 * var.sqrt() / n.sqrt();
    (mean, (mean - ci).max(0.0), mean + ci)
}

/// Calibrate batch size so each measurement call takes ~50ms.
fn calibrate<F: FnMut()>(mut single: F) -> usize {
    let start = Instant::now();
    single();
    let elapsed = start.elapsed().as_secs_f64();
    ((TARGET_SECS / elapsed) as usize).max(1)
}

struct Record {
    group: &'static str,
    variant: &'static str,
    param: f64,
    mean: f64,
    ci_low: f64,
    ci_high: f64,
}

fn main() {
    fs::create_dir_all(OUTPUT_DIR).unwrap();
    let mut results: Vec<Record> = Vec::new();
    let n_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    let lambda = 1.0;
    let mu = 0.5;
    let (d, t, l) = (0.5, 0.2, 0.3);

    // =================================================================
    // BD Scaling
    // =================================================================
    eprintln!("=== Birth-Death Scaling ===");
    for &n in &[50usize, 100, 200, 500, 1000, 2000, 5000, 10000] {
        let batch = calibrate(|| {
            let mut rng = StdRng::seed_from_u64(0);
            std::hint::black_box(simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap());
        });

        // Serial
        eprint!("  n={:<6} serial   (x{:<5})...", n, batch);
        let (mean, lo, hi) = measure_rate(
            || {
                let mut rng = StdRng::seed_from_u64(0);
                for _ in 0..batch {
                    std::hint::black_box(simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap());
                }
            },
            batch,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "bd", variant: "serial", param: n as f64, mean, ci_low: lo, ci_high: hi });

        // Parallel
        let par_batch = batch.max(n_cpus * 8);
        eprint!("  n={:<6} parallel (x{:<5})...", n, par_batch);
        let (mean, lo, hi) = measure_rate(
            || {
                (0..par_batch).into_par_iter().for_each(|i| {
                    let mut rng = StdRng::seed_from_u64(i as u64);
                    std::hint::black_box(simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap());
                });
            },
            par_batch,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "bd", variant: "parallel", param: n as f64, mean, ci_low: lo, ci_high: hi });
    }

    // =================================================================
    // DTL Per-Gene Scaling
    // =================================================================
    eprintln!("\n=== DTL Per-Gene Scaling ===");
    for &sp in &[10usize, 20, 50, 100, 200] {
        let species_tree = make_species_tree(sp, 99);
        let root = species_tree.root;

        let batch = calibrate(|| {
            let mut rng = StdRng::seed_from_u64(0);
            std::hint::black_box(simulate_dtl(&species_tree, root, d, t, l, None, None, false, &mut rng).unwrap());
        });

        eprint!("  sp={:<5} serial   (x{:<5})...", sp, batch);
        let (mean, lo, hi) = measure_rate(
            || {
                let mut rng = StdRng::seed_from_u64(0);
                for _ in 0..batch {
                    std::hint::black_box(simulate_dtl(&species_tree, root, d, t, l, None, None, false, &mut rng).unwrap());
                }
            },
            batch,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "dtl_pergene", variant: "serial", param: sp as f64, mean, ci_low: lo, ci_high: hi });

        let par_batch = batch.max(n_cpus * 8);
        eprint!("  sp={:<5} parallel (x{:<5})...", sp, par_batch);
        let (mean, lo, hi) = measure_rate(
            || {
                (0..par_batch).into_par_iter().for_each(|i| {
                    let mut rng = StdRng::seed_from_u64(i as u64);
                    std::hint::black_box(simulate_dtl(&species_tree, root, d, t, l, None, None, false, &mut rng).unwrap());
                });
            },
            par_batch,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "dtl_pergene", variant: "parallel", param: sp as f64, mean, ci_low: lo, ci_high: hi });
    }

    // =================================================================
    // DTL Per-Species Scaling
    // =================================================================
    eprintln!("\n=== DTL Per-Species Scaling ===");
    for &sp in &[10usize, 20, 50, 100, 200] {
        let species_tree = make_species_tree(sp, 99);
        let root = species_tree.root;

        let batch = calibrate(|| {
            let mut rng = StdRng::seed_from_u64(0);
            std::hint::black_box(simulate_dtl_per_species(&species_tree, root, d, t, l, None, None, false, &mut rng).unwrap());
        });

        eprint!("  sp={:<5} serial   (x{:<5})...", sp, batch);
        let (mean, lo, hi) = measure_rate(
            || {
                let mut rng = StdRng::seed_from_u64(0);
                for _ in 0..batch {
                    std::hint::black_box(simulate_dtl_per_species(&species_tree, root, d, t, l, None, None, false, &mut rng).unwrap());
                }
            },
            batch,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "dtl_perspecies", variant: "serial", param: sp as f64, mean, ci_low: lo, ci_high: hi });

        let par_batch = batch.max(n_cpus * 8);
        eprint!("  sp={:<5} parallel (x{:<5})...", sp, par_batch);
        let (mean, lo, hi) = measure_rate(
            || {
                (0..par_batch).into_par_iter().for_each(|i| {
                    let mut rng = StdRng::seed_from_u64(i as u64);
                    std::hint::black_box(simulate_dtl_per_species(&species_tree, root, d, t, l, None, None, false, &mut rng).unwrap());
                });
            },
            par_batch,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "dtl_perspecies", variant: "parallel", param: sp as f64, mean, ci_low: lo, ci_high: hi });
    }

    // =================================================================
    // DTL Batch Size Scaling (fixed species tree = 50, varying batch)
    // =================================================================
    eprintln!("\n=== DTL Batch Size Scaling (50 species) ===");
    let species_tree_batch = make_species_tree(50, 99);
    let root_batch = species_tree_batch.root;
    let (d2, t2, l2) = (0.2, 0.2, 0.1);

    for &bs in &[10usize, 50, 100, 500, 1000, 5000] {
        // Per-gene serial
        eprint!("  batch={:<6} pergene  serial  ...", bs);
        let (mean, lo, hi) = measure_rate(
            || {
                let mut rng = StdRng::seed_from_u64(0);
                for _ in 0..bs {
                    std::hint::black_box(simulate_dtl(&species_tree_batch, root_batch, d2, t2, l2, None, None, false, &mut rng).unwrap());
                }
            },
            bs,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "dtl_batch", variant: "pergene_serial", param: bs as f64, mean, ci_low: lo, ci_high: hi });

        // Per-gene parallel
        eprint!("  batch={:<6} pergene  parallel...", bs);
        let (mean, lo, hi) = measure_rate(
            || {
                (0..bs).into_par_iter().for_each(|i| {
                    let mut rng = StdRng::seed_from_u64(i as u64);
                    std::hint::black_box(simulate_dtl(&species_tree_batch, root_batch, d2, t2, l2, None, None, false, &mut rng).unwrap());
                });
            },
            bs,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "dtl_batch", variant: "pergene_parallel", param: bs as f64, mean, ci_low: lo, ci_high: hi });

        // Per-species serial
        eprint!("  batch={:<6} persp    serial  ...", bs);
        let (mean, lo, hi) = measure_rate(
            || {
                let mut rng = StdRng::seed_from_u64(0);
                for _ in 0..bs {
                    std::hint::black_box(simulate_dtl_per_species(&species_tree_batch, root_batch, d2, t2, l2, None, None, false, &mut rng).unwrap());
                }
            },
            bs,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "dtl_batch", variant: "perspecies_serial", param: bs as f64, mean, ci_low: lo, ci_high: hi });

        // Per-species parallel
        eprint!("  batch={:<6} persp    parallel...", bs);
        let (mean, lo, hi) = measure_rate(
            || {
                (0..bs).into_par_iter().for_each(|i| {
                    let mut rng = StdRng::seed_from_u64(i as u64);
                    std::hint::black_box(simulate_dtl_per_species(&species_tree_batch, root_batch, d2, t2, l2, None, None, false, &mut rng).unwrap());
                });
            },
            bs,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "dtl_batch", variant: "perspecies_parallel", param: bs as f64, mean, ci_low: lo, ci_high: hi });
    }

    // =================================================================
    // Thread Scaling
    // =================================================================
    eprintln!("\n=== Thread Scaling ===");
    let max_threads = n_cpus;
    let mut thread_counts: Vec<usize> = [1, 2, 4, 8]
        .iter()
        .copied()
        .filter(|&t| t <= max_threads)
        .collect();
    if !thread_counts.contains(&max_threads) {
        thread_counts.push(max_threads);
    }

    let batch_ts = 500;

    // BD
    let n_bd = 1000;
    for &threads in &thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        eprint!("  BD         threads={:<3}...", threads);
        let (mean, lo, hi) = measure_rate(
            || {
                pool.install(|| {
                    (0..batch_ts).into_par_iter().for_each(|i| {
                        let mut rng = StdRng::seed_from_u64(i as u64);
                        std::hint::black_box(simulate_bd_tree_bwd(n_bd, lambda, mu, &mut rng).unwrap());
                    });
                });
            },
            batch_ts,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "thread_scaling", variant: "bd", param: threads as f64, mean, ci_low: lo, ci_high: hi });
    }

    // DTL per-gene
    let species_tree_ts = make_species_tree(100, 99);
    let root_ts = species_tree_ts.root;
    for &threads in &thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        eprint!("  DTL pergene threads={:<3}...", threads);
        let (mean, lo, hi) = measure_rate(
            || {
                pool.install(|| {
                    (0..batch_ts).into_par_iter().for_each(|i| {
                        let mut rng = StdRng::seed_from_u64(i as u64);
                        std::hint::black_box(simulate_dtl(&species_tree_ts, root_ts, d, t, l, None, None, false, &mut rng).unwrap());
                    });
                });
            },
            batch_ts,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "thread_scaling", variant: "dtl_pergene", param: threads as f64, mean, ci_low: lo, ci_high: hi });
    }

    // DTL per-species
    for &threads in &thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        eprint!("  DTL persp   threads={:<3}...", threads);
        let (mean, lo, hi) = measure_rate(
            || {
                pool.install(|| {
                    (0..batch_ts).into_par_iter().for_each(|i| {
                        let mut rng = StdRng::seed_from_u64(i as u64);
                        std::hint::black_box(simulate_dtl_per_species(&species_tree_ts, root_ts, d, t, l, None, None, false, &mut rng).unwrap());
                    });
                });
            },
            batch_ts,
        );
        eprintln!(" {:>12.0} trees/s", mean);
        results.push(Record { group: "thread_scaling", variant: "dtl_perspecies", param: threads as f64, mean, ci_low: lo, ci_high: hi });
    }

    // =================================================================
    // Write CSV
    // =================================================================
    let csv_path = format!("{}/throughput.csv", OUTPUT_DIR);
    let mut f = fs::File::create(&csv_path).unwrap();
    writeln!(f, "group,variant,param,mean,ci_low,ci_high").unwrap();
    for r in &results {
        writeln!(
            f,
            "{},{},{},{:.2},{:.2},{:.2}",
            r.group, r.variant, r.param, r.mean, r.ci_low, r.ci_high
        )
        .unwrap();
    }

    eprintln!("\nResults written to {}", csv_path);
    eprintln!("Generate plots with: python3 benches/plot_benchmarks.py");
}
