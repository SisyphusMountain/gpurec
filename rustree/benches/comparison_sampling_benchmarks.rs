use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashSet;

use rustree::bd::simulate_bd_tree_bwd;
use rustree::comparison::{compare_nodes, compare_nodes_topology};
use rustree::sampling::{
    extract_induced_subtree, find_all_leaf_indices, extract_extant_subtree,
    build_leaf_pair_lca_map,
};

/// Build a species tree of size n with fixed seed.
fn make_tree(n: usize) -> rustree::node::FlatTree {
    let mut rng = StdRng::seed_from_u64(42);
    let (mut tree, _) = simulate_bd_tree_bwd(n, 1.0, 0.5, &mut rng).unwrap();
    tree.assign_depths();
    tree
}

// ============================================================================
// Comparison benchmarks
// ============================================================================

fn bench_compare_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare_nodes");
    for &n in &[50, 200, 500] {
        let tree = make_tree(n);
        let node = tree.to_node();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| compare_nodes(&node, &node, true, 1e-10).unwrap());
        });
    }
    group.finish();
}

fn bench_compare_nodes_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare_nodes_topology");
    for &n in &[50, 200, 500] {
        let tree = make_tree(n);
        let node = tree.to_node();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| compare_nodes_topology(&node, &node).unwrap());
        });
    }
    group.finish();
}

// ============================================================================
// Sampling benchmarks
// ============================================================================

fn bench_extract_induced_subtree(c: &mut Criterion) {
    let mut group = c.benchmark_group("extract_induced_subtree");
    for &n in &[50, 200, 500] {
        let tree = make_tree(n);
        let all_leaves = find_all_leaf_indices(&tree);

        // Sample half the leaves
        let half: HashSet<usize> = all_leaves.iter().take(all_leaves.len() / 2).copied().collect();

        group.bench_with_input(BenchmarkId::new("half_leaves", n), &n, |b, _| {
            b.iter(|| extract_induced_subtree(&tree, &half));
        });
    }
    group.finish();
}

fn bench_extract_extant_subtree(c: &mut Criterion) {
    let mut group = c.benchmark_group("extract_extant_subtree");
    for &n in &[50, 200, 500] {
        let tree = make_tree(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| extract_extant_subtree(&tree));
        });
    }
    group.finish();
}

fn bench_build_lca_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_leaf_pair_lca_map");
    for &n in &[20, 50, 100] {
        let tree = make_tree(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| build_leaf_pair_lca_map(&tree));
        });
    }
    group.finish();
}

criterion_group!(
    comparison_benches,
    bench_compare_nodes,
    bench_compare_nodes_topology,
);

criterion_group!(
    sampling_benches,
    bench_extract_induced_subtree,
    bench_extract_extant_subtree,
    bench_build_lca_map,
);

criterion_main!(comparison_benches, sampling_benches);
