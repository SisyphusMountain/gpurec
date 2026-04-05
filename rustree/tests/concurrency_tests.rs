// Thread-safety and concurrency tests (#81)
// Verify that parallel simulation produces correct results.

use rustree::bd::simulate_bd_tree_bwd;
use rustree::dtl::simulate_dtl;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::Arc;

#[test]
fn test_parallel_bd_simulation() {
    // Simulate multiple BD trees in parallel using rayon
    let results: Vec<_> = (0u64..20)
        .map(|seed| {
            std::thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(seed);
                simulate_bd_tree_bwd(10, 1.0, 0.3, &mut rng).unwrap()
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();

    // All should produce valid trees
    for (tree, events) in &results {
        assert!(!tree.nodes.is_empty());
        assert!(!events.is_empty());
        let leaves = tree.get_leaves().len();
        assert!(leaves >= 10, "Each tree should have at least 10 leaves (extant)");
    }

    // Different seeds should generally produce different trees
    let sizes: Vec<usize> = results.iter().map(|(t, _)| t.nodes.len()).collect();
    let unique_sizes: std::collections::HashSet<usize> = sizes.iter().copied().collect();
    // With 20 trials and extinction, highly likely to get multiple distinct sizes
    assert!(unique_sizes.len() > 1, "All 20 trees had identical sizes — suspicious");
}

#[test]
fn test_parallel_dtl_shared_species_tree() {
    // Multiple DTL simulations sharing the same species tree via Arc
    let mut rng = StdRng::seed_from_u64(42);
    // Small tree with low rates to keep simulations fast
    let (species_tree, _) = simulate_bd_tree_bwd(4, 1.0, 0.0, &mut rng).unwrap();
    let species_tree = Arc::new(species_tree);
    let origin = species_tree.root;

    let handles: Vec<_> = (0u64..5)
        .map(|seed| {
            let sp = Arc::clone(&species_tree);
            std::thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(seed + 100);
                // Very low rates to ensure fast completion
                simulate_dtl(&sp, origin, 0.05, 0.05, 0.05, None, None, false, &mut rng)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let mut success_count = 0;
    for result in &results {
        match result {
            Ok((rec_tree, _events)) => {
                assert!(!rec_tree.gene_tree.nodes.is_empty());
                success_count += 1;
            }
            Err(_) => {
                // Some simulations may fail — that's OK
            }
        }
    }
    assert!(success_count > 0, "At least one DTL simulation should succeed");
}

#[test]
fn test_parallel_determinism() {
    // Running the same seed in parallel should produce the same result as sequential
    let seed = 42u64;

    // Sequential
    let mut rng_seq = StdRng::seed_from_u64(seed);
    let (tree_seq, events_seq) = simulate_bd_tree_bwd(15, 1.0, 0.2, &mut rng_seq).unwrap();

    // Parallel (single thread, but through thread API)
    let handle = std::thread::spawn(move || {
        let mut rng = StdRng::seed_from_u64(seed);
        simulate_bd_tree_bwd(15, 1.0, 0.2, &mut rng).unwrap()
    });
    let (tree_par, events_par) = handle.join().unwrap();

    assert_eq!(tree_seq.nodes.len(), tree_par.nodes.len(),
        "Same seed should produce same tree regardless of thread");
    assert_eq!(events_seq.len(), events_par.len(),
        "Same seed should produce same events regardless of thread");
}
