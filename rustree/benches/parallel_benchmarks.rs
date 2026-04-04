use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::{simulate_dtl, simulate_dtl_per_species};
use rustree::FlatTree;

const BATCH: usize = 1000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_species_tree(n: usize, seed: u64) -> FlatTree {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut tree, _) = simulate_bd_tree_bwd(n, 1.0, 0.3, &mut rng);
    tree.assign_depths();
    tree
}

fn pilot_bd_events(n: usize, lambda: f64, mu: f64) -> u64 {
    let mut rng = StdRng::seed_from_u64(42);
    let total: usize = (0..10)
        .map(|_| simulate_bd_tree_bwd(n, lambda, mu, &mut rng).1.len())
        .sum();
    (total / 10) as u64
}

fn pilot_dtl_events(species_tree: &FlatTree, d: f64, t: f64, l: f64) -> u64 {
    let root = species_tree.root;
    let mut rng = StdRng::seed_from_u64(42);
    let total: usize = (0..10)
        .map(|_| simulate_dtl(species_tree, root, d, t, l, None, None, false, &mut rng).unwrap().1.len())
        .sum();
    (total / 10).max(1) as u64
}

fn pilot_dtl_per_species_events(species_tree: &FlatTree, d: f64, t: f64, l: f64) -> u64 {
    let root = species_tree.root;
    let mut rng = StdRng::seed_from_u64(42);
    let total: usize = (0..10)
        .map(|_| {
            simulate_dtl_per_species(species_tree, root, d, t, l, None, None, false, &mut rng)
                .unwrap()
                .1
                .len()
        })
        .sum();
    (total / 10).max(1) as u64
}

// ---------------------------------------------------------------------------
// BD parallel benchmarks
// ---------------------------------------------------------------------------

/// Run BATCH BD simulations in parallel (one RNG per task, seeded by index).
fn bd_parallel_scaling(c: &mut Criterion) {
    let lambda = 1.0;
    let mu = 0.5;
    for &n in &[100, 500, 1000, 5000, 10000] {
        let avg_events = pilot_bd_events(n, lambda, mu);
        let total_events = avg_events * BATCH as u64;
        let mut group = c.benchmark_group("bd_parallel_scaling");
        group.throughput(Throughput::Elements(total_events));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                (0..BATCH).into_par_iter().for_each(|i| {
                    let mut rng = StdRng::seed_from_u64(i as u64);
                    std::hint::black_box(simulate_bd_tree_bwd(n, lambda, mu, &mut rng));
                });
            });
        });
        group.finish();
    }
}

/// Compare single-threaded vs parallel for the same total work (BATCH sims).
fn bd_parallel_vs_serial(c: &mut Criterion) {
    let lambda = 1.0;
    let mu = 0.5;
    let n = 1000;
    let avg_events = pilot_bd_events(n, lambda, mu);
    let total_events = avg_events * BATCH as u64;

    let mut group = c.benchmark_group("bd_parallel_vs_serial");
    group.throughput(Throughput::Elements(total_events));

    group.bench_function("serial", |b| {
        b.iter(|| {
            for i in 0..BATCH {
                let mut rng = StdRng::seed_from_u64(i as u64);
                std::hint::black_box(simulate_bd_tree_bwd(n, lambda, mu, &mut rng));
            }
        });
    });

    group.bench_function("parallel", |b| {
        b.iter(|| {
            (0..BATCH).into_par_iter().for_each(|i| {
                let mut rng = StdRng::seed_from_u64(i as u64);
                std::hint::black_box(simulate_bd_tree_bwd(n, lambda, mu, &mut rng));
            });
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// DTL parallel benchmarks (per-gene)
// ---------------------------------------------------------------------------

fn dtl_parallel_pergene_scaling(c: &mut Criterion) {
    let (d, t, l) = (0.5, 0.2, 0.3);
    for &n in &[10, 50, 100, 200] {
        let species_tree = make_species_tree(n, 99);
        let root = species_tree.root;
        let avg_events = pilot_dtl_events(&species_tree, d, t, l);
        let total_events = avg_events * BATCH as u64;
        let mut group = c.benchmark_group("dtl_parallel_pergene_scaling");
        group.throughput(Throughput::Elements(total_events));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                (0..BATCH).into_par_iter().for_each(|i| {
                    let mut rng = StdRng::seed_from_u64(i as u64);
                    std::hint::black_box(simulate_dtl(
                        &species_tree, root, d, t, l, None, None, false, &mut rng,
                    ).unwrap());
                });
            });
        });
        group.finish();
    }
}

fn dtl_parallel_vs_serial(c: &mut Criterion) {
    let species_tree = make_species_tree(100, 99);
    let root = species_tree.root;
    let (d, t, l) = (0.5, 0.2, 0.3);
    let avg_events = pilot_dtl_events(&species_tree, d, t, l);
    let total_events = avg_events * BATCH as u64;

    let mut group = c.benchmark_group("dtl_parallel_vs_serial");
    group.throughput(Throughput::Elements(total_events));

    group.bench_function("serial", |b| {
        b.iter(|| {
            for i in 0..BATCH {
                let mut rng = StdRng::seed_from_u64(i as u64);
                std::hint::black_box(simulate_dtl(
                    &species_tree, root, d, t, l, None, None, false, &mut rng,
                ).unwrap());
            }
        });
    });

    group.bench_function("parallel", |b| {
        b.iter(|| {
            (0..BATCH).into_par_iter().for_each(|i| {
                let mut rng = StdRng::seed_from_u64(i as u64);
                std::hint::black_box(simulate_dtl(
                    &species_tree, root, d, t, l, None, None, false, &mut rng,
                ).unwrap());
            });
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// DTL parallel benchmarks (per-species)
// ---------------------------------------------------------------------------

fn dtl_parallel_perspecies_scaling(c: &mut Criterion) {
    let (d, t, l) = (0.5, 0.2, 0.3);
    for &n in &[10, 50, 100, 200] {
        let species_tree = make_species_tree(n, 99);
        let root = species_tree.root;
        let avg_events = pilot_dtl_per_species_events(&species_tree, d, t, l);
        let total_events = avg_events * BATCH as u64;
        let mut group = c.benchmark_group("dtl_parallel_perspecies_scaling");
        group.throughput(Throughput::Elements(total_events));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                (0..BATCH).into_par_iter().for_each(|i| {
                    let mut rng = StdRng::seed_from_u64(i as u64);
                    std::hint::black_box(simulate_dtl_per_species(
                        &species_tree, root, d, t, l, None, None, false, &mut rng,
                    ).unwrap());
                });
            });
        });
        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Thread scaling: vary thread count for fixed workload
// ---------------------------------------------------------------------------

fn bd_thread_scaling(c: &mut Criterion) {
    let lambda = 1.0;
    let mu = 0.5;
    let n = 1000;
    let avg_events = pilot_bd_events(n, lambda, mu);
    let total_events = avg_events * BATCH as u64;
    let max_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    let mut group = c.benchmark_group("bd_thread_scaling");
    group.throughput(Throughput::Elements(total_events));

    for threads in [1, 2, 4, 8, max_threads] {
        if threads > max_threads {
            continue;
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, _| {
                b.iter(|| {
                    pool.install(|| {
                        (0..BATCH).into_par_iter().for_each(|i| {
                            let mut rng = StdRng::seed_from_u64(i as u64);
                            std::hint::black_box(simulate_bd_tree_bwd(n, lambda, mu, &mut rng));
                        });
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bd_parallel_scaling,
    bd_parallel_vs_serial,
    dtl_parallel_pergene_scaling,
    dtl_parallel_vs_serial,
    dtl_parallel_perspecies_scaling,
    bd_thread_scaling,
);
criterion_main!(benches);
