// rustree/src/main.rs
// just testing code here


use rustree::node::{Node, FlatNode, FlatTree, TraversalOrder, HasName};

fn print_traversal_names<T, I>(iter: I)
where
    I: Iterator<Item = T>,
    T: HasName,
{
    for node in iter {
        println!("{}", node.name());
    }
}

fn main() {
    // Example: Build a simple Node tree manually.
    let node_tree = Node {
        name: "root".to_string(),
        left_child: Some(Box::new(Node {
            name: "left".to_string(),
            left_child: None,
            right_child: None,
            depth: None,
            length: 0.0,
        })),
        right_child: Some(Box::new(Node {
            name: "right".to_string(),
            left_child: None,
            right_child: None,
            depth: None,
            length: 0.0,
        })),
        depth: None,
        length: 0.0,
    };

    // Preorder traversal of Node
    println!("Preorder traversal of Node:");
    print_traversal_names(node_tree.iter(TraversalOrder::PreOrder));

    // Inorder traversal of Node
    println!("\nInorder traversal of Node:");
    print_traversal_names(node_tree.iter(TraversalOrder::InOrder));

    // Postorder traversal of Node
    println!("\nPostorder traversal of Node:");
    print_traversal_names(node_tree.iter(TraversalOrder::PostOrder));

    // Initialize a FlatTree
    let flat_tree = FlatTree {
        nodes: vec![
            FlatNode {
                name: "root".to_string(),
                left_child: Some(1),
                right_child: Some(2),
                parent: None,
                depth: None,
                length: 0.0,
            },
            FlatNode {
                name: "left".to_string(),
                left_child: None,
                right_child: None,
                parent: Some(0),
                depth: None,
                length: 0.0,
            },
            FlatNode {
                name: "right".to_string(),
                left_child: None,
                right_child: None,
                parent: Some(0),
                depth: None,
                length: 0.0,
            },
        ],
        root: 0,
    };

    // Preorder traversal of FlatTree
    println!("\nPreorder traversal of FlatTree:");
    print_traversal_names(flat_tree.iter(TraversalOrder::PreOrder));

    // Inorder traversal of FlatTree
    println!("\nInorder traversal of FlatTree:");
    print_traversal_names(flat_tree.iter(TraversalOrder::InOrder));

    // Postorder traversal of FlatTree
    println!("\nPostorder traversal of FlatTree:");
    print_traversal_names(flat_tree.iter(TraversalOrder::PostOrder));
}
