// Simple profiling benchmark - runs batch simulation for cleaner profiling
use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::simulate_dtl_batch;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    // Generate species tree
    let mut rng = StdRng::seed_from_u64(42);
    let (mut species_tree, _) = simulate_bd_tree_bwd(1000, 1.0, 0.0, &mut rng);
    species_tree.assign_depths();

    // Run many iterations for better profiling signal
    let n_trees = 5000;
    let mut rng = StdRng::seed_from_u64(123);

    let (_rec_trees, _events) = simulate_dtl_batch(
        &species_tree,
        species_tree.root,
        0.2, 0.2, 0.1,
        None,
        None,
        n_trees,
        false,
        &mut rng,
    );

    println!("Completed {} trees", n_trees);
}
