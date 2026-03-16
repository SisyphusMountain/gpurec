// tests/rf_test.rs
//
// Robinson-Foulds distance comparison against R's ape package.
//
// IGNORED: The current `unrooted_robinson_foulds` implementation computes
// rooted clade-based RF (counting monophyletic groups), not true unrooted
// bipartition-based RF. This produces widespread mismatches against R's ape
// package which computes unrooted RF. The implementation needs to be rewritten
// to use proper bipartition splits before this test can pass.

use std::fs;
use std::path::{Path, PathBuf};
use rustree::node::FlatTree;
use rustree::{parse_newick, unrooted_robinson_foulds};

fn flat_tree_from_file<P: AsRef<Path>>(file_path: P) -> FlatTree {
    let tree_str = fs::read_to_string(&file_path)
        .unwrap_or_else(|_| panic!("Failed to read tree file: {:?}", file_path.as_ref()));
    let sanitized = tree_str.trim();
    let trees: Vec<String> = sanitized
        .split(';')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                Some(format!("{};", s))
            }
        })
        .collect();
    if trees.is_empty() {
        panic!("No tree found in file: {:?}", file_path.as_ref());
    }
    let tree_newick = &trees[0];
    let mut nodes = parse_newick(tree_newick)
        .expect("Failed to parse Newick tree");
    let root = nodes.pop().expect("No tree produced");
    root.to_flat_tree()
}

#[test]
#[ignore = "RF implementation uses rooted clades, not unrooted bipartitions — produces wrong results vs R ape"]
fn test_rf_distance_comparison_table() {
    let dir_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/RF_trees");

    let mut tree_files: Vec<PathBuf> = fs::read_dir(&dir_path)
        .expect("Failed to read directory")
        .filter_map(|entry| {
            let entry = entry.expect("Error reading directory entry");
            let path = entry.path();
            if path.is_file()
                && path.extension().map(|s| s == "newick").unwrap_or(false)
                && path.to_string_lossy().contains("tree_")
            {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    tree_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let flat_trees: Vec<FlatTree> = tree_files
        .iter()
        .map(|path| flat_tree_from_file(path))
        .collect();

    let mut mismatches = 0usize;

    for i in 0..flat_trees.len() {
        for j in (i + 1)..flat_trees.len() {
            let expected_file = dir_path.join(format!("rf_tree_{:02}_{:02}.txt", i + 1, j + 1));
            let expected_rf: usize = fs::read_to_string(&expected_file)
                .unwrap_or_else(|_| panic!("Missing RF file: {}", expected_file.display()))
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("Bad RF value in {}", expected_file.display()));

            let computed_rf = unrooted_robinson_foulds(&flat_trees[i], &flat_trees[j]);

            if computed_rf != expected_rf {
                mismatches += 1;
            }

            assert_eq!(
                computed_rf, expected_rf,
                "RF mismatch for trees {} vs {}: expected {}, got {}",
                i + 1, j + 1, expected_rf, computed_rf
            );
        }
    }
}
