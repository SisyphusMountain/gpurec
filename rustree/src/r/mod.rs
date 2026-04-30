#![allow(clippy::too_many_arguments)]
//! R bindings for rustree using extendr.
//!
//! Provides R access to:
//! - Newick tree parsing
//! - Birth-death species tree simulation
//! - DTL gene tree simulation
//! - Export to Newick, XML, CSV
//! - SVG visualization via thirdkind
//!
//! Implementation is split across subfiles for maintainability,
//! included here so all `#[extendr]`-generated items share one module scope.

mod conversions;
use conversions::*;

use extendr_api::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs;
use std::process::Command;

use crate::bd::{simulate_bd_tree_bwd, generate_events_from_tree};
use crate::dtl::{simulate_dtl, simulate_dtl_batch, simulate_dtl_per_species, simulate_dtl_per_species_batch};
use crate::dtl::{simulate_dtl_iter, simulate_dtl_per_species_iter};
use crate::induced_transfers::induced_transfers;
use crate::node::{FlatTree, Event, RecTree, GeneForest, remap_gene_tree_indices};
use crate::sampling::{extract_induced_subtree, extract_induced_subtree_by_names};

// ---------------------------------------------------------------------------
// Species tree & gene tree operations
// ---------------------------------------------------------------------------
include!("species.rs");

// ---------------------------------------------------------------------------
// DTL simulation
// ---------------------------------------------------------------------------
include!("dtl.rs");

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------
include!("io.rs");

// ---------------------------------------------------------------------------
// Visualization
// ---------------------------------------------------------------------------
include!("viz.rs");

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------
include!("sampling.rs");

// ---------------------------------------------------------------------------
// Analysis & metrics
// ---------------------------------------------------------------------------
include!("analysis.rs");

// ---------------------------------------------------------------------------
// Helpers (not exported to R)
// ---------------------------------------------------------------------------
include!("helpers.rs");

// ---------------------------------------------------------------------------
// Streaming DTL
// ---------------------------------------------------------------------------
include!("streaming.rs");

// Macro to generate exports.
// The `_r` suffix on each function name avoids collisions with the wrapper
// functions that extendr auto-generates (which use the unsuffixed names).
extendr_module! {
    mod rustree;
    fn simulate_species_tree_r;
    fn parse_newick_r;
    fn name_internal_nodes_r;
    fn tree_to_newick_r;
    fn tree_num_leaves_r;
    fn tree_leaf_names_r;
    fn gene_tree_num_extant_r;
    fn gene_tree_to_newick_r;
    fn gene_tree_to_xml_r;
    fn simulate_dtl_r;
    fn simulate_dtl_batch_r;
    fn simulate_dtl_per_species_r;
    fn simulate_dtl_per_species_batch_r;
    fn save_newick_r;
    fn save_xml_r;
    fn save_csv_r;
    fn save_bd_events_csv_r;
    fn gene_tree_to_svg_r;
    fn sample_extant_r;
    fn sample_leaves_r;
    fn extract_induced_subtree_by_names_r;
    fn pairwise_distances_r;
    fn save_pairwise_distances_csv_r;
    fn parse_recphyloxml_r;
    fn induced_transfers_r;
    fn simulate_dtl_stream_xml_r;
    fn simulate_dtl_stream_newick_r;
    fn simulate_dtl_per_species_stream_xml_r;
    fn simulate_dtl_per_species_stream_newick_r;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "r"))]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // parse_newick_to_flattree
    // -----------------------------------------------------------------------

    #[test]
    fn parse_newick_valid_simple_tree() {
        let tree = parse_newick_to_flattree("(A:1,B:1);").unwrap();
        assert_eq!(tree.nodes.len(), 3);
        let leaf_names: Vec<&str> = tree.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .map(|n| n.name.as_str())
            .collect();
        assert!(leaf_names.contains(&"A"));
        assert!(leaf_names.contains(&"B"));
    }

    #[test]
    fn parse_newick_valid_three_taxa() {
        let tree = parse_newick_to_flattree("((A:1,B:1):0.5,C:1.5);").unwrap();
        let leaf_count = tree.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .count();
        assert_eq!(leaf_count, 3);
    }

    #[test]
    fn parse_newick_empty_string() {
        let result = parse_newick_to_flattree("");
        assert!(result.is_err(), "Empty string should fail to parse");
    }

    #[test]
    fn parse_newick_malformed_missing_semicolon() {
        let result = parse_newick_to_flattree("(A:1,B:1)");
        assert!(result.is_err(), "Newick without semicolon should fail");
    }

    #[test]
    fn parse_newick_malformed_unbalanced_parens() {
        let result = parse_newick_to_flattree("((A:1,B:1);");
        assert!(result.is_err(), "Unbalanced parentheses should fail");
    }

    #[test]
    fn parse_newick_single_leaf() {
        let tree = parse_newick_to_flattree("A:1.0;");
        if let Ok(t) = tree {
            assert!(!t.nodes.is_empty());
        }
    }

    #[test]
    fn parse_newick_whitespace_only() {
        let result = parse_newick_to_flattree("   ");
        assert!(result.is_err(), "Whitespace-only string should fail");
    }

    #[test]
    fn parse_newick_depths_assigned() {
        let tree = parse_newick_to_flattree("(A:1,B:1):0;").unwrap();
        for node in &tree.nodes {
            assert!(node.depth.is_some(), "All nodes should have depth assigned, but '{}' does not", node.name);
        }
    }

    // -----------------------------------------------------------------------
    // make_rng
    // -----------------------------------------------------------------------

    #[test]
    fn make_rng_with_null() {
        test! {
            let seed = Robj::from(());
            let result = make_rng(&seed);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn make_rng_with_integer_seed() {
        test! {
            let seed = Robj::from(42 as i32);
            let result = make_rng(&seed);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn make_rng_deterministic_with_same_seed() {
        test! {
            use rand::Rng;
            let seed1 = Robj::from(123 as i32);
            let seed2 = Robj::from(123 as i32);
            let mut rng1 = make_rng(&seed1).unwrap();
            let mut rng2 = make_rng(&seed2).unwrap();
            let val1: f64 = rng1.gen();
            let val2: f64 = rng2.gen();
            assert_eq!(val1, val2, "Same seed should produce same random values");
        }
    }

    #[test]
    fn make_rng_different_seeds_differ() {
        test! {
            use rand::Rng;
            let seed1 = Robj::from(1 as i32);
            let seed2 = Robj::from(2 as i32);
            let mut rng1 = make_rng(&seed1).unwrap();
            let mut rng2 = make_rng(&seed2).unwrap();
            let val1: f64 = rng1.gen();
            let val2: f64 = rng2.gen();
            assert_ne!(val1, val2, "Different seeds should produce different values");
        }
    }

    #[test]
    fn make_rng_with_string_fails() {
        test! {
            let seed = Robj::from("not a number");
            let result = make_rng(&seed);
            assert!(result.is_err());
        }
    }

    #[test]
    fn make_rng_with_float_fails() {
        test! {
            let seed = Robj::from(3.14);
            let result = make_rng(&seed);
            let _ = result;
        }
    }

    // -----------------------------------------------------------------------
    // extract_alpha
    // -----------------------------------------------------------------------

    #[test]
    fn extract_alpha_null() {
        test! {
            let alpha = Robj::from(());
            let result = extract_alpha(&alpha).unwrap();
            assert_eq!(result, None);
        }
    }

    #[test]
    fn extract_alpha_valid_number() {
        test! {
            let alpha = Robj::from(2.5);
            let result = extract_alpha(&alpha).unwrap();
            assert_eq!(result, Some(2.5));
        }
    }

    #[test]
    fn extract_alpha_zero() {
        test! {
            let alpha = Robj::from(0.0);
            let result = extract_alpha(&alpha).unwrap();
            assert_eq!(result, Some(0.0));
        }
    }

    #[test]
    fn extract_alpha_negative() {
        test! {
            let alpha = Robj::from(-1.0);
            let result = extract_alpha(&alpha).unwrap();
            assert_eq!(result, Some(-1.0));
        }
    }

    #[test]
    fn extract_alpha_string_fails() {
        test! {
            let alpha = Robj::from("not a number");
            let result = extract_alpha(&alpha);
            assert!(result.is_err(), "String should fail for transfer_alpha");
        }
    }

    // -----------------------------------------------------------------------
    // extract_replacement
    // -----------------------------------------------------------------------

    #[test]
    fn extract_replacement_null() {
        test! {
            let repl = Robj::from(());
            let result = extract_replacement(&repl).unwrap();
            assert_eq!(result, None);
        }
    }

    #[test]
    fn extract_replacement_valid_number() {
        test! {
            let repl = Robj::from(0.5);
            let result = extract_replacement(&repl).unwrap();
            assert_eq!(result, Some(0.5));
        }
    }

    #[test]
    fn extract_replacement_string_fails() {
        test! {
            let repl = Robj::from("bad");
            let result = extract_replacement(&repl);
            assert!(result.is_err(), "String should fail for replacement_transfer");
        }
    }

    // -----------------------------------------------------------------------
    // Validation patterns
    // -----------------------------------------------------------------------

    #[test]
    fn species_tree_negative_n_rejected() {
        test! {
            let result = simulate_species_tree_r(-1, 1.0, 0.5, Robj::from(()));
            assert!(result.is_err());
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(err_msg.contains("positive"), "Error should mention 'positive', got: {}", err_msg);
        }
    }

    #[test]
    fn species_tree_zero_n_rejected() {
        test! {
            let result = simulate_species_tree_r(0, 1.0, 0.5, Robj::from(()));
            assert!(result.is_err());
        }
    }

    #[test]
    fn species_tree_negative_lambda_rejected() {
        test! {
            let result = simulate_species_tree_r(5, -1.0, 0.5, Robj::from(()));
            assert!(result.is_err());
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(err_msg.contains("positive"), "Error should mention 'positive', got: {}", err_msg);
        }
    }

    #[test]
    fn species_tree_negative_mu_rejected() {
        test! {
            let result = simulate_species_tree_r(5, 1.0, -0.5, Robj::from(()));
            assert!(result.is_err());
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(err_msg.contains("non-negative"), "Error should mention 'non-negative', got: {}", err_msg);
        }
    }

    #[test]
    fn species_tree_mu_equals_lambda_rejected() {
        test! {
            let result = simulate_species_tree_r(5, 1.0, 1.0, Robj::from(()));
            assert!(result.is_err());
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(err_msg.contains("greater than"), "Error should say lambda > mu, got: {}", err_msg);
        }
    }

    #[test]
    fn species_tree_mu_greater_than_lambda_rejected() {
        test! {
            let result = simulate_species_tree_r(5, 1.0, 2.0, Robj::from(()));
            assert!(result.is_err());
        }
    }

    #[test]
    fn species_tree_valid_params_succeed() {
        test! {
            let result = simulate_species_tree_r(3, 2.0, 0.5, Robj::from(42 as i32));
            assert!(result.is_ok(), "Valid parameters should succeed: {:?}", result.unwrap_err());
        }
    }

    #[test]
    fn species_tree_zero_mu_allowed() {
        test! {
            let result = simulate_species_tree_r(3, 1.0, 0.0, Robj::from(42 as i32));
            assert!(result.is_ok(), "mu=0 (pure birth) should be allowed: {:?}", result.unwrap_err());
        }
    }

    #[test]
    fn species_tree_seed_determinism() {
        test! {
            let r1 = simulate_species_tree_r(5, 2.0, 0.5, Robj::from(99 as i32)).unwrap();
            let r2 = simulate_species_tree_r(5, 2.0, 0.5, Robj::from(99 as i32)).unwrap();
            let names1: Vec<String> = r1.dollar("name").unwrap().as_str_vector().unwrap()
                .iter().map(|s| s.to_string()).collect();
            let names2: Vec<String> = r2.dollar("name").unwrap().as_str_vector().unwrap()
                .iter().map(|s| s.to_string()).collect();
            assert_eq!(names1, names2, "Same seed should produce identical trees");
        }
    }

    #[test]
    fn species_tree_string_seed_rejected() {
        test! {
            let result = simulate_species_tree_r(5, 2.0, 0.5, Robj::from("abc"));
            assert!(result.is_err());
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(err_msg.contains("integer"), "Error should mention 'integer', got: {}", err_msg);
        }
    }

    // -----------------------------------------------------------------------
    // parse_newick_r: end-to-end
    // -----------------------------------------------------------------------

    #[test]
    fn parse_newick_r_valid() {
        test! {
            let result = parse_newick_r("(A:1,B:1);");
            assert!(result.is_ok());
            let list = result.unwrap();
            let names: Vec<String> = list.dollar("name").unwrap().as_str_vector().unwrap()
                .iter().map(|s| s.to_string()).collect();
            assert!(names.contains(&"A".to_string()));
            assert!(names.contains(&"B".to_string()));
        }
    }

    #[test]
    fn parse_newick_r_empty_string() {
        test! {
            let result = parse_newick_r("");
            assert!(result.is_err());
        }
    }

    #[test]
    fn parse_newick_r_malformed() {
        test! {
            let result = parse_newick_r("((A,B)");
            assert!(result.is_err());
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(err_msg.contains("parse") || err_msg.contains("Newick"),
                "Error should mention parsing failure, got: {}", err_msg);
        }
    }

    // -----------------------------------------------------------------------
    // DTL rate validation
    // -----------------------------------------------------------------------

    #[test]
    fn dtl_stream_negative_rate_rejected() {
        test! {
            let result = simulate_dtl_stream_xml_r(
                "(A:1,B:1);", 1, -0.1, 0.1, 0.1,
                Robj::from(()), Robj::from(()), false,
                Robj::from(42 as i32), "/tmp/rustree_test_should_not_exist",
            );
            assert!(result.is_err());
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(err_msg.contains("non-negative"), "Error should mention 'non-negative', got: {}", err_msg);
        }
    }

    #[test]
    fn dtl_stream_zero_n_rejected() {
        test! {
            let result = simulate_dtl_stream_xml_r(
                "(A:1,B:1);", 0, 0.1, 0.1, 0.1,
                Robj::from(()), Robj::from(()), false,
                Robj::from(42 as i32), "/tmp/rustree_test_should_not_exist",
            );
            assert!(result.is_err());
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(err_msg.contains("positive"), "Error should mention 'positive', got: {}", err_msg);
        }
    }

    #[test]
    fn dtl_stream_negative_n_rejected() {
        test! {
            let result = simulate_dtl_stream_xml_r(
                "(A:1,B:1);", -5, 0.1, 0.1, 0.1,
                Robj::from(()), Robj::from(()), false,
                Robj::from(42 as i32), "/tmp/rustree_test_should_not_exist",
            );
            assert!(result.is_err());
        }
    }

    // -----------------------------------------------------------------------
    // name_internal_nodes_r
    // -----------------------------------------------------------------------

    #[test]
    fn name_internal_nodes_rejects_existing_internal_prefix() {
        test! {
            let tree = parse_newick_r("(internal0:1,B:1);");
            assert!(tree.is_ok());
            let result = name_internal_nodes_r(tree.unwrap());
            assert!(result.is_err());
            let err_msg = format!("{:?}", result.unwrap_err());
            assert!(err_msg.contains("internal"),
                "Error should mention 'internal' name conflict, got: {}", err_msg);
        }
    }
}
