use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustree::bd::simulate_bd_tree_bwd;

/// Run a pilot of 10 simulations to estimate average events per run.
fn pilot_bd_events(n: usize, lambda: f64, mu: f64) -> u64 {
    let mut rng = StdRng::seed_from_u64(42);
    let total: usize = (0..10)
        .map(|_| simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap().1.len())
        .sum();
    (total / 10) as u64
}

fn bd_scaling(c: &mut Criterion) {
    let lambda = 1.0;
    let mu = 0.5;
    for &n in &[100, 500, 1000, 5000, 10000] {
        let avg_events = pilot_bd_events(n, lambda, mu);
        let mut group = c.benchmark_group("bd_scaling");
        group.throughput(Throughput::Elements(avg_events));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = StdRng::seed_from_u64(123);
            b.iter(|| simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap());
        });
        group.finish();
    }
}

fn bd_pure_birth(c: &mut Criterion) {
    let lambda = 1.0;
    let mu = 0.0;
    for &n in &[100, 1000, 10000] {
        let avg_events = pilot_bd_events(n, lambda, mu);
        let mut group = c.benchmark_group("bd_pure_birth");
        group.throughput(Throughput::Elements(avg_events));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = StdRng::seed_from_u64(123);
            b.iter(|| simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap());
        });
        group.finish();
    }
}

fn bd_high_extinction(c: &mut Criterion) {
    let lambda = 1.0;
    let mu = 0.9;
    for &n in &[100, 500, 1000] {
        let avg_events = pilot_bd_events(n, lambda, mu);
        let mut group = c.benchmark_group("bd_high_extinction");
        group.throughput(Throughput::Elements(avg_events));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = StdRng::seed_from_u64(123);
            b.iter(|| simulate_bd_tree_bwd(n, lambda, mu, &mut rng).unwrap());
        });
        group.finish();
    }
}

criterion_group!(benches, bd_scaling, bd_pure_birth, bd_high_extinction);
criterion_main!(benches);
