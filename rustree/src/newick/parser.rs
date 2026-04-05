// rustree/src/newick/newick.rs
// This module contains functions to parse Newick formatted strings into Tree structures
// and to convert Tree structures back into Newick formatted strings.
use crate::node::{Node, FlatTree};
use crate::error::RustreeError;

#[derive(Parser)]
#[grammar = "newick/newick.pest"]
pub struct NewickParser; // The parser struct, needs to be public to be used outside
                         // The following two functions are used to convert a Newick string into an arborescent Tree structure.

/// Parse a Newick formatted string into a vector of Node trees.
///
/// # Arguments
/// * `newick_str` - A string slice containing the Newick formatted tree
///
/// # Returns
/// A Result containing a vector of Node trees on success, or an error message on failure.
pub fn parse_newick(newick_str: &str) -> Result<Vec<Node>, RustreeError> {
    use pest::Parser;
    let mut pairs = NewickParser::parse(Rule::newick, newick_str)
        .map_err(|e| RustreeError::Parse(e.to_string()))?;

    newick_to_tree(pairs.next().ok_or_else(|| RustreeError::Parse("No tree found in input".to_string()))?)
}

fn newick_to_tree(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Node>, RustreeError> {
    let mut vec_trees: Vec<Node> = Vec::new();
    for inner in pair.into_inner() {
        let tree = handle_pair(inner)?;
        if let Some(node) = tree {
            vec_trees.push(node);
        }
    }
    Ok(vec_trees)
}

pub fn handle_pair(pair: pest::iterators::Pair<Rule>) -> Result<Option<Node>, RustreeError> {
    match pair.as_rule() {
        Rule::newick => {
            for inner in pair.into_inner() {
                if inner.as_rule() == Rule::subtree {
                    return handle_pair(inner);
                }
            }
            Ok(None)
        }
        Rule::subtree => {
            let inner = pair.into_inner().next()
                .ok_or_else(|| RustreeError::Parse("Malformed Newick: subtree rule has no inner elements".to_string()))?;
            handle_pair(inner)
        }
        Rule::leaf => {
            let mut name = String::new();
            let mut length = 0.0;

            for inner_pair in pair.into_inner() {
                match inner_pair.as_rule() {
                    Rule::LABEL | Rule::NAME => {
                        name = extract_label(inner_pair);
                    }
                    Rule::LENGTH => {
                        let val = inner_pair.as_str();
                        length = val.parse::<f64>().map_err(|e| {
                            RustreeError::Parse(format!("Failed to parse branch length '{}': {}", val, e))
                        })?;
                    }
                    _ => {} // Ignore other rules
                }
            }
            Ok(Some(Node {
                name,
                left_child: None,
                right_child: None,
                depth: None,
                length,
            }))
        }
        Rule::internal => {
            let mut name = String::new();
            let mut length = 0.0;
            let mut subtrees: Vec<Node> = Vec::new();

            for inner_pair in pair.into_inner() {
                match inner_pair.as_rule() {
                    Rule::subtree => {
                        let subtree = handle_pair(inner_pair)?
                            .ok_or_else(|| RustreeError::Parse("Malformed Newick: subtree produced no node".to_string()))?;
                        subtrees.push(subtree);
                    }
                    Rule::LABEL | Rule::NAME => {
                        name = extract_label(inner_pair);
                    }
                    Rule::LENGTH => {
                        let val = inner_pair.as_str();
                        length = val.parse::<f64>().map_err(|e| {
                            RustreeError::Parse(format!("Failed to parse branch length '{}': {}", val, e))
                        })?;
                    }
                    _ => {}
                }
            }

            if subtrees.len() > 2 {
                let node_label = if name.is_empty() { "<unnamed>" } else { &name };
                return Err(RustreeError::Parse(format!(
                    "Non-binary node detected: node '{}' has {} children. Only binary trees are supported.",
                    node_label,
                    subtrees.len()
                )));
            }

            let first_subtree = subtrees.first().cloned();
            let second_subtree = subtrees.get(1).cloned();

            Ok(Some(Node {
                name,
                left_child: first_subtree.map(Box::new),
                right_child: second_subtree.map(Box::new),
                depth: None,
                length,
            }))
        }
        Rule::NAME | Rule::LABEL | Rule::QUOTED_NAME | Rule::LENGTH | Rule::WHITESPACE => Ok(None),
    }
}

/// Extract the label text from a LABEL, QUOTED_NAME, or NAME pair.
/// For quoted names, strips the surrounding single quotes.
fn extract_label(pair: pest::iterators::Pair<Rule>) -> String {
    match pair.as_rule() {
        Rule::LABEL => {
            // LABEL wraps either QUOTED_NAME or NAME
            if let Some(inner) = pair.into_inner().next() {
                extract_label(inner)
            } else {
                String::new()
            }
        }
        Rule::QUOTED_NAME => {
            let s = pair.as_str();
            // Strip surrounding single quotes
            s[1..s.len() - 1].to_string()
        }
        Rule::NAME => pair.as_str().to_string(),
        _ => pair.as_str().to_string(),
    }
}
impl Node {
    pub fn to_newick(&self) -> Result<String, RustreeError> {
        node_to_newick_recursive(self, 0)
    }
}

impl FlatTree {
    pub fn to_newick(&self) -> Result<String, RustreeError> {
        let node_tree = self.to_node();
        node_to_newick_recursive(&node_tree, 0)
    }
}

fn node_to_newick_recursive(node: &Node, index: usize) -> Result<String, RustreeError> {
    /* Takes a node and returns the corresponding subtree in Newick format.
        --------------------------------
        INPUT:
            - node: the node to convert to Newick format.
            - index: a running counter used to identify nodes in error messages.
        OUTPUT:
            - Ok(newick) with the Newick representation of the subtree rooted at node.
            - Err(msg) if a single-child (unary) node is encountered.
        Warning: rounds the lengths to 6 decimal places.
    */
    match (&node.left_child, &node.right_child) {
        (Some(left_child), Some(right_child)) => {
            // Internal node with both children (binary).
            let left_str = node_to_newick_recursive(left_child, index + 1)?;
            let right_str = node_to_newick_recursive(right_child, index + 2)?;
            Ok(format!(
                "({},{}){}:{:.6}",
                left_str,
                right_str,
                node.name,
                node.length
            ))
        }
        (None, None) => {
            // Leaf node.
            Ok(format!("{}:{:.6}", node.name, node.length))
        }
        _ => {
            // Single-child (unary) node: exactly one of left/right is Some.
            Err(RustreeError::Tree(format!(
                "Single-child node detected: node '{}' (index {}) has only one child. \
                 Only binary trees (with 0 or 2 children per node) are supported.",
                node.name, index
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Original tests ---

    #[test]
    fn test_parse_with_colons() {
        let result = parse_newick("(A:1.0,B:2.0):0.0;");
        assert!(result.is_ok());
        let nodes = result.unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].left_child.as_ref().unwrap().name, "A");
    }

    #[test]
    fn test_parse_without_colons() {
        let result = parse_newick("(A,B);");
        assert!(result.is_ok());
        let nodes = result.unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].left_child.as_ref().unwrap().name, "A");
        assert_eq!(nodes[0].right_child.as_ref().unwrap().name, "B");
    }

    #[test]
    fn test_parse_nested_without_colons() {
        let result = parse_newick("((A,B),C);");
        assert!(result.is_ok());
        let nodes = result.unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn test_parse_with_whitespace() {
        let result = parse_newick("( A:1.0 , B:2.0 ):0.0 ;");
        assert!(result.is_ok());
        let nodes = result.unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].left_child.as_ref().unwrap().name, "A");
    }

    #[test]
    fn test_parse_mixed_colons() {
        let result = parse_newick("(A:1.0,B);");
        assert!(result.is_ok());
        let nodes = result.unwrap();
        let left = nodes[0].left_child.as_ref().unwrap();
        let right = nodes[0].right_child.as_ref().unwrap();
        assert_eq!(left.name, "A");
        assert_eq!(left.length, 1.0);
        assert_eq!(right.name, "B");
        assert_eq!(right.length, 0.0);
    }

    #[test]
    fn test_parse_newlines_in_newick() {
        let result = parse_newick("(\n  A:1.0,\n  B:2.0\n):0.0;");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_unnamed_leaves() {
        let result = parse_newick("(,);");
        assert!(result.is_ok());
        let nodes = result.unwrap();
        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].left_child.is_some());
        assert!(nodes[0].right_child.is_some());
        assert_eq!(nodes[0].left_child.as_ref().unwrap().name, "");
        assert_eq!(nodes[0].right_child.as_ref().unwrap().name, "");
    }

    #[test]
    fn test_parse_unnamed_with_lengths() {
        let result = parse_newick("(:1.0,:2.0);");
        assert!(result.is_ok());
        let nodes = result.unwrap();
        let left = nodes[0].left_child.as_ref().unwrap();
        let right = nodes[0].right_child.as_ref().unwrap();
        assert_eq!(left.name, "");
        assert_eq!(left.length, 1.0);
        assert_eq!(right.name, "");
        assert_eq!(right.length, 2.0);
    }

    #[test]
    fn test_parse_quoted_labels() {
        let result = parse_newick("('Species 1':1.0,'Species 2':2.0):0.0;");
        assert!(result.is_ok(), "Failed to parse quoted labels: {:?}", result.err());
        let nodes = result.unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].left_child.as_ref().unwrap().name, "Species 1");
        assert_eq!(nodes[0].right_child.as_ref().unwrap().name, "Species 2");
    }

    #[test]
    fn test_parse_mixed_quoted_unquoted() {
        let result = parse_newick("('Homo sapiens':1.0,Mouse:2.0);");
        assert!(result.is_ok());
        let nodes = result.unwrap();
        assert_eq!(nodes[0].left_child.as_ref().unwrap().name, "Homo sapiens");
        assert_eq!(nodes[0].right_child.as_ref().unwrap().name, "Mouse");
    }

    #[test]
    fn test_parse_quoted_internal_label() {
        let result = parse_newick("(A:1.0,B:2.0)'ancestor':0.5;");
        assert!(result.is_ok());
        let nodes = result.unwrap();
        assert_eq!(nodes[0].name, "ancestor");
    }

    #[test]
    fn test_roundtrip_preserves_topology() {
        let newick = "(A:1.000000,B:2.000000):0.000000;";
        let nodes = parse_newick(newick).unwrap();
        let output = nodes[0].to_newick().unwrap();
        assert_eq!(format!("{};", output), newick);
    }

    // --- Additional edge-case tests ---

    #[test]
    fn parse_newick_preserves_names() {
        let nodes = parse_newick("(Alpha:1.0,Beta:2.0);").unwrap();
        let root = &nodes[0];
        let left_name = root.left_child.as_ref().unwrap().name.as_str();
        let right_name = root.right_child.as_ref().unwrap().name.as_str();
        let names = vec![left_name, right_name];
        assert!(names.contains(&"Alpha"), "Should contain 'Alpha'");
        assert!(names.contains(&"Beta"), "Should contain 'Beta'");
    }

    #[test]
    fn parse_newick_preserves_branch_lengths() {
        let nodes = parse_newick("(A:1.5,B:2.5);").unwrap();
        let root = &nodes[0];
        let left_len = root.left_child.as_ref().unwrap().length;
        let right_len = root.right_child.as_ref().unwrap().length;
        let lengths = vec![left_len, right_len];
        assert!(lengths.contains(&1.5));
        assert!(lengths.contains(&2.5));
    }

    #[test]
    fn parse_newick_empty_string_fails() {
        let result = parse_newick("");
        assert!(result.is_err(), "Empty string should fail");
    }

    #[test]
    fn parse_newick_whitespace_only_fails() {
        let result = parse_newick("   ");
        assert!(result.is_err(), "Whitespace-only should fail");
    }

    #[test]
    fn parse_newick_unbalanced_parens_fails() {
        let result = parse_newick("((A:1,B:1);");
        assert!(result.is_err(), "Unbalanced parentheses should fail");
    }

    #[test]
    fn parse_newick_missing_semicolon_fails() {
        let result = parse_newick("(A:1,B:1)");
        assert!(result.is_err(), "Missing semicolon should fail");
    }

    #[test]
    fn parse_newick_garbage_input_fails() {
        let result = parse_newick("this is not newick");
        assert!(result.is_err(), "Garbage input should fail");
    }

    #[test]
    fn parse_newick_nested_tree() {
        let nodes = parse_newick("(((A:1,B:1):1,(C:1,D:1):1):1,E:3);").unwrap();
        assert_eq!(nodes.len(), 1);
        let tree = nodes[0].to_flat_tree();
        let leaf_count = tree.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .count();
        assert_eq!(leaf_count, 5);
    }

    #[test]
    fn parse_newick_zero_branch_length() {
        let nodes = parse_newick("(A:0,B:0);");
        assert!(nodes.is_ok(), "Zero branch lengths should be valid");
    }
}
