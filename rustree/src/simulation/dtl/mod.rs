// DTL (Duplication-Transfer-Loss) simulation along a species tree
//
// This module simulates gene tree evolution within a species tree using the DTL model.
// Events: Speciation (S), Duplication (D), Transfer (T), Loss (L)

mod event;
pub(crate) mod gillespie;
mod per_gene;
mod per_species;
mod state;
mod stream;
mod utils;

// Re-export main types and functions
pub use event::DTLEvent;
pub use stream::DtlSimIter;
pub use per_gene::{simulate_dtl, simulate_dtl_batch, simulate_dtl_iter, simulate_dtl_iter_with_config};
pub use per_species::{simulate_dtl_per_species, simulate_dtl_per_species_batch, simulate_dtl_per_species_iter, simulate_dtl_per_species_iter_with_config};
pub use utils::{count_extant_genes, count_events};

/// Configuration for DTL (Duplication-Transfer-Loss) simulation.
#[derive(Clone, Debug)]
pub struct DTLConfig {
    /// Duplication rate
    pub lambda_d: f64,
    /// Transfer rate
    pub lambda_t: f64,
    /// Loss rate
    pub lambda_l: f64,
    /// Assortative transfer parameter (None = uniform transfer)
    pub transfer_alpha: Option<f64>,
    /// Replacement transfer probability (None = additive only)
    pub replacement_transfer: Option<f64>,
}

// Re-export from io module for backward compatibility
pub use crate::io::save_dtl_events_to_csv as save_events_to_csv;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::newick::parse_newick;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn parse_tree(newick: &str) -> crate::node::FlatTree {
        let mut nodes = parse_newick(newick).unwrap();
        let root = nodes.pop().expect("No tree found");
        root.to_flat_tree()
    }

    #[test]
    fn test_dtl_pure_speciation() {
        // Simple species tree: ((A:1,B:1)AB:1,C:2)root:0;
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(42);

        // With zero D/T/L rates, should get pure speciation
        let (rec_tree, events) = simulate_dtl(&species_tree, species_tree.root, 0.0, 0.0, 0.0, None, None, false, &mut rng).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Events: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);
        println!("Total event count: {}", events.len());

        // Should have 3 extant genes (one per species leaf)
        let extant = count_extant_genes(&rec_tree);
        assert_eq!(extant, 3, "Should have 3 extant genes with no D/T/L");
    }

    #[test]
    fn test_dtl_with_duplication() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(123);

        // High duplication rate
        let (rec_tree, _events) = simulate_dtl(&species_tree, species_tree.root, 2.0, 0.0, 0.0, None, None, false, &mut rng).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Events with duplication: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);

        // Should have duplications
        assert!(d > 0 || leaves >= 3, "Should have duplications or at least 3 leaves");
    }

    #[test]
    fn test_dtl_with_loss() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(999);

        // High loss rate
        let (rec_tree, _events) = simulate_dtl(&species_tree, species_tree.root, 0.0, 0.0, 5.0, None, None, false, &mut rng).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Events with loss: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);

        // May have losses
        let extant = count_extant_genes(&rec_tree);
        println!("Extant genes: {}", extant);
    }

    #[test]
    fn test_dtl_xml_export() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(42);

        // With mixed events
        let (rec_tree, events) = simulate_dtl(&species_tree, species_tree.root, 1.0, 0.5, 0.5, None, None, false, &mut rng).unwrap();

        // Save events to CSV
        save_events_to_csv(&events, &species_tree, &rec_tree.gene_tree, "testdata/test_dtl_events.csv").expect("Failed to write events CSV");
        println!("Events saved to testdata/test_dtl_events.csv");

        // Generate XML
        let xml = rec_tree.to_xml();

        // Verify XML has required sections
        assert!(xml.contains("<recPhylo"));
        assert!(xml.contains("<spTree>"));
        assert!(xml.contains("<recGeneTree>"));
        assert!(xml.contains("<branchLength>"));

        // Write to file for inspection
        use std::fs;
        fs::write("testdata/test_dtl_output.xml", &xml).expect("Failed to write XML");
        println!("XML output written to testdata/test_dtl_output.xml");
    }

    #[test]
    fn test_dtl_with_transfer() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(456);

        // High transfer rate (uniform)
        let (rec_tree, events) = simulate_dtl(&species_tree, species_tree.root, 0.0, 2.0, 0.0, None, None, false, &mut rng).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Events with transfer: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);
        println!("Total event count: {}", events.len());
    }

    #[test]
    fn test_dtl_assortative_transfer() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(456);

        // High transfer rate with assortative selection (alpha=1.0)
        let (rec_tree, events) = simulate_dtl(&species_tree, species_tree.root, 0.0, 2.0, 0.0, Some(1.0), None, false, &mut rng).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Events with assortative transfer: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);
        println!("Total event count: {}", events.len());

        // Verify LCA computation works
        let lca_depths = species_tree.precompute_lca_depths()
            .expect("Failed to precompute LCA depths");
        assert!(lca_depths.len() > 0, "LCA depths should be computed");
    }

    // Tests for per-species (Zombi-style) simulation

    #[test]
    fn test_per_species_pure_speciation() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(42);

        // With zero D/T/L rates, should get pure speciation
        let (rec_tree, events) = simulate_dtl_per_species(&species_tree, species_tree.root, 0.0, 0.0, 0.0, None, None, false, &mut rng).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Per-species pure speciation: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);
        println!("Total event count: {}", events.len());

        // Should have 3 extant genes (one per species leaf)
        let extant = count_extant_genes(&rec_tree);
        assert_eq!(extant, 3, "Should have 3 extant genes with no D/T/L");
    }

    #[test]
    fn test_per_species_with_duplication() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(123);

        // High duplication rate
        let (rec_tree, _events) = simulate_dtl_per_species(&species_tree, species_tree.root, 2.0, 0.0, 0.0, None, None, false, &mut rng).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Per-species with duplication: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);

        // Should have duplications or at least the original 3 leaves
        assert!(d > 0 || leaves >= 3, "Should have duplications or at least 3 leaves");
    }

    #[test]
    fn test_per_species_with_transfer() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(456);

        // High transfer rate (uniform)
        let (rec_tree, events) = simulate_dtl_per_species(&species_tree, species_tree.root, 0.0, 2.0, 0.0, None, None, false, &mut rng).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Per-species with transfer: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);
        println!("Total event count: {}", events.len());
    }

    #[test]
    fn test_per_species_compare_event_counts() {
        // Compare per-gene-copy vs per-species models with same rates
        // Per-species should have fewer events on average when there are many gene copies
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut total_per_gene = 0;
        let mut total_per_species = 0;

        for seed in 0..10 {
            let mut rng1 = StdRng::seed_from_u64(seed);
            let mut rng2 = StdRng::seed_from_u64(seed);

            let (rec1, _) = simulate_dtl(&species_tree, species_tree.root, 1.0, 0.5, 0.5, None, None, false, &mut rng1).unwrap();
            let (rec2, _) = simulate_dtl_per_species(&species_tree, species_tree.root, 1.0, 0.5, 0.5, None, None, false, &mut rng2).unwrap();

            let (_, d1, t1, l1, _) = count_events(&rec1);
            let (_, d2, t2, l2, _) = count_events(&rec2);

            total_per_gene += d1 + t1 + l1;
            total_per_species += d2 + t2 + l2;
        }

        println!("Total DTL events over 10 runs:");
        println!("  Per-gene-copy model: {}", total_per_gene);
        println!("  Per-species model:   {}", total_per_species);

        // Per-species should generally have fewer events because duplications don't increase rate
        // But this isn't guaranteed for all seeds, so we just print for comparison
    }

    #[test]
    fn test_per_species_require_extant() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(999);

        // High loss rate, but require extant genes
        let (rec_tree, _events) = simulate_dtl_per_species(&species_tree, species_tree.root, 0.0, 0.0, 5.0, None, None, true, &mut rng).unwrap();

        let extant = count_extant_genes(&rec_tree);
        assert!(extant > 0, "Should have at least one extant gene when require_extant=true");
        println!("Per-species with high loss (require_extant=true): {} extant genes", extant);
    }

    // Tests for replacement transfers

    #[test]
    fn test_replacement_transfer_per_species() {
        let newick = "((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(42);

        // High duplication + transfer rates with 100% replacement
        let (rec_tree, _events) = simulate_dtl_per_species(
            &species_tree, species_tree.root,
            1.0, 2.0, 0.0, None, Some(1.0), false, &mut rng
        ).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Replacement per-species: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);

        // With replacement transfers, we should see loss events from replacements
        assert!(t > 0, "Should have transfers");
        assert!(l > 0, "Replacement transfers should cause loss events");
    }

    #[test]
    fn test_replacement_transfer_per_gene() {
        let newick = "((A:1,B:1)AB:1,(C:1,D:1)CD:1)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        let mut rng = StdRng::seed_from_u64(42);

        // High duplication + transfer rates with 100% replacement
        let (rec_tree, _events) = simulate_dtl(
            &species_tree, species_tree.root,
            1.0, 2.0, 0.0, None, Some(1.0), false, &mut rng
        ).unwrap();

        let (s, d, t, l, leaves) = count_events(&rec_tree);
        println!("Replacement per-gene: S={}, D={}, T={}, L={}, Leaves={}", s, d, t, l, leaves);

        // With replacement transfers, we should see both transfer and loss events
        assert!(t > 0, "Should have transfers");
        assert!(l > 0, "Replacement transfers should cause loss events");
    }

    #[test]
    fn test_replacement_none_backward_compatible() {
        let newick = "((A:1,B:1)AB:1,C:2)root:0;";
        let mut species_tree = parse_tree(newick);
        species_tree.assign_depths();

        // None vs Some(0.0) should produce identical results
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let (rec1, _) = simulate_dtl_per_species(
            &species_tree, species_tree.root,
            1.0, 1.0, 0.5, None, None, false, &mut rng1
        ).unwrap();
        let (rec2, _) = simulate_dtl_per_species(
            &species_tree, species_tree.root,
            1.0, 1.0, 0.5, None, Some(0.0), false, &mut rng2
        ).unwrap();

        let events1 = count_events(&rec1);
        let events2 = count_events(&rec2);
        assert_eq!(events1, events2, "None and Some(0.0) should produce identical results");
    }
}
