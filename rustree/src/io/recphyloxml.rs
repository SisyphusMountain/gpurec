//! RecPhyloXML parsing functionality for reading reconciled gene trees.
//!
//! This module provides functions to parse RecPhyloXML files (e.g., from ALERax)
//! into the rustree data structures.

use crate::node::{FlatNode, FlatTree, rectree::Event};
use quick_xml::events::Event as XmlEvent;
use quick_xml::Reader;
use std::collections::HashMap;
use std::fs;

/// Errors that can occur during RecPhyloXML parsing.
#[derive(Debug)]
pub enum ParseError {
    XmlError(quick_xml::Error),
    IoError(std::io::Error),
    MissingSection(String),
    InvalidFormat(String),
    MissingSpecies(String),
    InvalidEvent(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ParseError::XmlError(e) => write!(f, "XML parsing error: {}", e),
            ParseError::IoError(e) => write!(f, "IO error: {}", e),
            ParseError::MissingSection(s) => write!(f, "Missing XML section: {}", s),
            ParseError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
            ParseError::MissingSpecies(s) => write!(f, "Species not found: {}", s),
            ParseError::InvalidEvent(s) => write!(f, "Invalid event: {}", s),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<quick_xml::Error> for ParseError {
    fn from(err: quick_xml::Error) -> Self {
        ParseError::XmlError(err)
    }
}

impl From<std::io::Error> for ParseError {
    fn from(err: std::io::Error) -> Self {
        ParseError::IoError(err)
    }
}

/// Intermediate structure for parsing species tree nodes.
#[derive(Debug, Clone)]
struct SpeciesNode {
    name: String,
    branch_length: f64,
    children: Vec<SpeciesNode>,
}

impl SpeciesNode {
    fn new(name: String) -> Self {
        SpeciesNode {
            name,
            branch_length: 0.0,
            children: Vec::new(),
        }
    }
}

/// Intermediate structure for parsing gene tree nodes.
#[derive(Debug, Clone)]
struct GeneNode {
    name: String,
    branch_length: f64,
    species_location: String,
    event_type: Event,
    children: Vec<GeneNode>,
}

impl GeneNode {
    fn new(name: String) -> Self {
        GeneNode {
            name,
            branch_length: 0.0,
            species_location: String::new(),
            event_type: Event::Leaf,
            children: Vec::new(),
        }
    }
}

/// Parse a RecPhyloXML string and return the components for creating a RecTree.
///
/// Returns a tuple of (species_tree, gene_tree, node_mapping, event_mapping).
pub fn parse_recphyloxml(
    xml_content: &str,
) -> Result<(FlatTree, FlatTree, Vec<Option<usize>>, Vec<Event>), ParseError> {
    // Parse the species tree
    let species_root = parse_species_tree(xml_content)?;
    let (species_tree, species_name_map) = species_node_to_flat_tree(&species_root)?;

    // Parse the gene tree
    let gene_root = parse_gene_tree(xml_content)?;
    let (gene_tree, node_mapping, event_mapping) =
        gene_node_to_flat_tree(&gene_root, &species_name_map)?;

    Ok((species_tree, gene_tree, node_mapping, event_mapping))
}

/// Parse a RecPhyloXML file and return the components for creating a RecTree.
pub fn parse_recphyloxml_file(
    filepath: &str,
) -> Result<(FlatTree, FlatTree, Vec<Option<usize>>, Vec<Event>), ParseError> {
    let xml_content = fs::read_to_string(filepath)?;
    parse_recphyloxml(&xml_content)
}

/// Parse the species tree section from RecPhyloXML.
fn parse_species_tree(xml_content: &str) -> Result<SpeciesNode, ParseError> {
    let mut reader = Reader::from_str(xml_content);
    reader.trim_text(true);

    let mut buf = Vec::new();
    let mut in_sp_tree = false;
    let mut in_phylogeny = false;
    let mut depth = 0;
    let mut clade_stack: Vec<SpeciesNode> = Vec::new();
    let mut current_text = String::new();
    let mut in_name = false;
    let mut in_branch_length = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(XmlEvent::Start(ref e)) => {
                let name = e.name();
                let tag_name = String::from_utf8_lossy(name.as_ref());

                match tag_name.as_ref() {
                    "spTree" => in_sp_tree = true,
                    "phylogeny" if in_sp_tree => in_phylogeny = true,
                    "clade" if in_phylogeny => {
                        depth += 1;
                        clade_stack.push(SpeciesNode::new(String::new()));
                    }
                    "name" if in_phylogeny && depth > 0 => {
                        in_name = true;
                        current_text.clear();
                    }
                    "branchLength" if in_phylogeny && depth > 0 => {
                        in_branch_length = true;
                        current_text.clear();
                    }
                    _ => {}
                }
            }
            Ok(XmlEvent::Text(e)) => {
                if in_name || in_branch_length {
                    current_text.push_str(&e.unescape()?.into_owned());
                }
            }
            Ok(XmlEvent::End(ref e)) => {
                let name = e.name();
                let tag_name = String::from_utf8_lossy(name.as_ref());

                match tag_name.as_ref() {
                    "name" if in_name => {
                        if let Some(node) = clade_stack.last_mut() {
                            node.name = current_text.trim().to_string();
                        }
                        in_name = false;
                    }
                    "branchLength" if in_branch_length => {
                        if let Some(node) = clade_stack.last_mut() {
                            let trimmed = current_text.trim();
                            node.branch_length = match trimmed.parse() {
                                Ok(v) => v,
                                Err(_) => {
                                    eprintln!("Warning: failed to parse branch length '{}' in species tree, defaulting to 0.0", trimmed);
                                    0.0
                                }
                            };
                        }
                        in_branch_length = false;
                    }
                    "clade" if in_phylogeny && depth > 0 => {
                        depth -= 1;
                        let completed_node = clade_stack.pop().ok_or_else(|| {
                            ParseError::InvalidFormat("Malformed XML: closing </clade> without matching opening tag in species tree".to_string())
                        })?;

                        if depth == 0 {
                            // This is the root node
                            return Ok(completed_node);
                        } else {
                            // Add to parent
                            if let Some(parent) = clade_stack.last_mut() {
                                parent.children.push(completed_node);
                            }
                        }
                    }
                    "spTree" => in_sp_tree = false,
                    _ => {}
                }
            }
            Ok(XmlEvent::Eof) => break,
            Err(e) => return Err(ParseError::XmlError(e)),
            _ => {}
        }
        buf.clear();
    }

    Err(ParseError::MissingSection("spTree".to_string()))
}

/// Parse the gene tree section from RecPhyloXML.
fn parse_gene_tree(xml_content: &str) -> Result<GeneNode, ParseError> {
    let mut reader = Reader::from_str(xml_content);
    reader.trim_text(true);

    let mut buf = Vec::new();
    let mut in_rec_gene_tree = false;
    let mut in_phylogeny = false;
    let mut depth = 0;
    let mut clade_stack: Vec<GeneNode> = Vec::new();
    let mut current_text = String::new();
    let mut in_name = false;
    let mut in_branch_length = false;
    let mut in_events_rec = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(XmlEvent::Start(ref e)) | Ok(XmlEvent::Empty(ref e)) => {
                let name = e.name();
                let tag_name = String::from_utf8_lossy(name.as_ref());

                match tag_name.as_ref() {
                    "recGeneTree" => in_rec_gene_tree = true,
                    "phylogeny" if in_rec_gene_tree => in_phylogeny = true,
                    "clade" if in_phylogeny => {
                        depth += 1;
                        clade_stack.push(GeneNode::new(String::new()));
                    }
                    "name" if in_phylogeny && depth > 0 => {
                        in_name = true;
                        current_text.clear();
                    }
                    "branchLength" if in_phylogeny && depth > 0 => {
                        in_branch_length = true;
                        current_text.clear();
                    }
                    "eventsRec" if in_phylogeny && depth > 0 => {
                        in_events_rec = true;
                    }
                    // Event tags
                    "speciation" if in_events_rec => {
                        if let Some(node) = clade_stack.last_mut() {
                            node.event_type = Event::Speciation;
                            if let Some(species_loc) = get_attribute(e, b"speciesLocation") {
                                node.species_location = species_loc;
                            }
                        }
                    }
                    "duplication" if in_events_rec => {
                        if let Some(node) = clade_stack.last_mut() {
                            node.event_type = Event::Duplication;
                            if let Some(species_loc) = get_attribute(e, b"speciesLocation") {
                                node.species_location = species_loc;
                            }
                        }
                    }
                    "branchingOut" if in_events_rec => {
                        if let Some(node) = clade_stack.last_mut() {
                            node.event_type = Event::Transfer;
                            if let Some(species_loc) = get_attribute(e, b"speciesLocation") {
                                node.species_location = species_loc;
                            }
                        }
                    }
                    "transferBack" if in_events_rec => {
                        if let Some(node) = clade_stack.last_mut() {
                            // Transfer recipient - update species location to destination
                            if let Some(dest_species) = get_attribute(e, b"destinationSpecies") {
                                node.species_location = dest_species;
                            }
                        }
                    }
                    "loss" if in_events_rec => {
                        if let Some(node) = clade_stack.last_mut() {
                            node.event_type = Event::Loss;
                            if let Some(species_loc) = get_attribute(e, b"speciesLocation") {
                                node.species_location = species_loc;
                            }
                        }
                    }
                    "leaf" if in_events_rec => {
                        if let Some(node) = clade_stack.last_mut() {
                            node.event_type = Event::Leaf;
                            if let Some(species_loc) = get_attribute(e, b"speciesLocation") {
                                node.species_location = species_loc;
                            }
                        }
                    }
                    "P" if in_events_rec => {
                        // Alternative format for present-day genes (same as leaf)
                        if let Some(node) = clade_stack.last_mut() {
                            node.event_type = Event::Leaf;
                            if let Some(species_loc) = get_attribute(e, b"speciesLocation") {
                                node.species_location = species_loc;
                            }
                        }
                    }
                    "T" if in_events_rec => {
                        // Zombi format: T denotes horizontal gene transfer
                        if let Some(node) = clade_stack.last_mut() {
                            node.event_type = Event::Transfer;
                            if let Some(species_loc) = get_attribute(e, b"speciesLocation") {
                                node.species_location = species_loc;
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(XmlEvent::Text(e)) => {
                if in_name || in_branch_length {
                    current_text.push_str(&e.unescape()?.into_owned());
                }
            }
            Ok(XmlEvent::End(ref e)) => {
                let name = e.name();
                let tag_name = String::from_utf8_lossy(name.as_ref());

                match tag_name.as_ref() {
                    "name" if in_name => {
                        if let Some(node) = clade_stack.last_mut() {
                            node.name = current_text.trim().to_string();
                        }
                        in_name = false;
                    }
                    "branchLength" if in_branch_length => {
                        if let Some(node) = clade_stack.last_mut() {
                            let trimmed = current_text.trim();
                            node.branch_length = match trimmed.parse() {
                                Ok(v) => v,
                                Err(_) => {
                                    eprintln!("Warning: failed to parse branch length '{}' in gene tree, defaulting to 0.0", trimmed);
                                    0.0
                                }
                            };
                        }
                        in_branch_length = false;
                    }
                    "eventsRec" if in_events_rec => {
                        in_events_rec = false;
                    }
                    "clade" if in_phylogeny && depth > 0 => {
                        depth -= 1;
                        let completed_node = clade_stack.pop().ok_or_else(|| {
                            ParseError::InvalidFormat("Malformed XML: closing </clade> without matching opening tag in gene tree".to_string())
                        })?;

                        if depth == 0 {
                            // This is the root node
                            return Ok(completed_node);
                        } else {
                            // Add to parent
                            if let Some(parent) = clade_stack.last_mut() {
                                parent.children.push(completed_node);
                            }
                        }
                    }
                    "recGeneTree" => in_rec_gene_tree = false,
                    _ => {}
                }
            }
            Ok(XmlEvent::Eof) => break,
            Err(e) => return Err(ParseError::XmlError(e)),
            _ => {}
        }
        buf.clear();
    }

    Err(ParseError::MissingSection("recGeneTree".to_string()))
}

/// Helper function to get an attribute value from an XML element.
fn get_attribute(element: &quick_xml::events::BytesStart, attr_name: &[u8]) -> Option<String> {
    for attr in element.attributes() {
        if let Ok(attr) = attr {
            if attr.key.as_ref() == attr_name {
                return String::from_utf8(attr.value.to_vec()).ok();
            }
        }
    }
    None
}

/// Convert a SpeciesNode tree to a FlatTree with a name-to-index map.
fn species_node_to_flat_tree(
    root: &SpeciesNode,
) -> Result<(FlatTree, HashMap<String, usize>), ParseError> {
    let mut nodes = Vec::new();
    let mut name_map = HashMap::new();

    fn traverse(
        node: &SpeciesNode,
        parent: Option<usize>,
        nodes: &mut Vec<FlatNode>,
        name_map: &mut HashMap<String, usize>,
    ) -> Result<usize, ParseError> {
        // Reject non-binary nodes
        if node.children.len() > 2 {
            return Err(ParseError::InvalidFormat(format!(
                "Non-binary node detected: species node '{}' has {} children. Only binary trees are supported.",
                node.name,
                node.children.len()
            )));
        }

        let index = nodes.len();

        // Add entry to name map (first occurrence wins for ambiguous names)
        if !name_map.contains_key(&node.name) {
            name_map.insert(node.name.clone(), index);
        }

        // Create the FlatNode with placeholder children
        nodes.push(FlatNode {
            name: node.name.clone(),
            left_child: None,
            right_child: None,
            parent,
            depth: None,
            length: node.branch_length,
            bd_event: None,
        });

        // Process children
        if node.children.len() > 0 {
            let left_idx = traverse(&node.children[0], Some(index), nodes, name_map)?;
            nodes[index].left_child = Some(left_idx);

            if node.children.len() > 1 {
                let right_idx = traverse(&node.children[1], Some(index), nodes, name_map)?;
                nodes[index].right_child = Some(right_idx);
            }
        }

        Ok(index)
    }

    let root_idx = traverse(root, None, &mut nodes, &mut name_map)?;

    Ok((
        FlatTree {
            nodes,
            root: root_idx,
        },
        name_map,
    ))
}

/// Convert a GeneNode tree to a FlatTree with mapping vectors.
fn gene_node_to_flat_tree(
    root: &GeneNode,
    species_name_map: &HashMap<String, usize>,
) -> Result<(FlatTree, Vec<Option<usize>>, Vec<Event>), ParseError> {
    let mut nodes = Vec::new();
    let mut node_mapping = Vec::new();
    let mut event_mapping = Vec::new();

    fn traverse(
        node: &GeneNode,
        parent: Option<usize>,
        nodes: &mut Vec<FlatNode>,
        node_mapping: &mut Vec<Option<usize>>,
        event_mapping: &mut Vec<Event>,
        species_name_map: &HashMap<String, usize>,
    ) -> Result<usize, ParseError> {
        let index = nodes.len();

        // Lookup species index
        let species_idx = species_name_map
            .get(&node.species_location)
            .copied()
            .ok_or_else(|| {
                ParseError::MissingSpecies(format!(
                    "Species '{}' not found in species tree (gene node: '{}', event: {:?})",
                    node.species_location,
                    node.name,
                    node.event_type
                ))
            })?;

        // Create the FlatNode with placeholder children
        nodes.push(FlatNode {
            name: node.name.clone(),
            left_child: None,
            right_child: None,
            parent,
            depth: None,
            length: node.branch_length,
            bd_event: None,
        });

        // Add mappings
        node_mapping.push(Some(species_idx));
        event_mapping.push(node.event_type.clone());

        // Reject non-binary nodes
        if node.children.len() > 2 {
            return Err(ParseError::InvalidFormat(format!(
                "Non-binary node detected: gene node '{}' has {} children. Only binary trees are supported.",
                node.name,
                node.children.len()
            )));
        }

        // Process children
        if node.children.len() > 0 {
            let left_idx = traverse(
                &node.children[0],
                Some(index),
                nodes,
                node_mapping,
                event_mapping,
                species_name_map,
            )?;
            nodes[index].left_child = Some(left_idx);

            if node.children.len() > 1 {
                let right_idx = traverse(
                    &node.children[1],
                    Some(index),
                    nodes,
                    node_mapping,
                    event_mapping,
                    species_name_map,
                )?;
                nodes[index].right_child = Some(right_idx);
            }
        }

        Ok(index)
    }

    let root_idx = traverse(
        root,
        None,
        &mut nodes,
        &mut node_mapping,
        &mut event_mapping,
        species_name_map,
    )?;

    Ok((
        FlatTree {
            nodes,
            root: root_idx,
        },
        node_mapping,
        event_mapping,
    ))
}

/// Build a HashMap mapping species names to their indices in the FlatTree.
///
/// This is used when parsing gene-tree-only XML where the species tree
/// is provided separately (e.g., from a Newick file).
pub fn build_species_name_map(species_tree: &FlatTree) -> HashMap<String, usize> {
    species_tree
        .nodes
        .iter()
        .enumerate()
        .map(|(idx, node)| (node.name.clone(), idx))
        .collect()
}

/// Parse gene-tree-only RecPhyloXML (no `<spTree>` section).
///
/// This function is used when the species tree is provided separately
/// (e.g., from a Newick file) and the XML only contains the reconciled
/// gene tree.
///
/// # Arguments
/// * `xml_content` - XML string containing only `<recGeneTree>` section
/// * `species_tree` - Pre-parsed species tree as FlatTree
///
/// # Returns
/// Tuple of (gene_tree, node_mapping, event_mapping)
pub fn parse_gene_tree_only(
    xml_content: &str,
    species_tree: &FlatTree,
) -> Result<(FlatTree, Vec<Option<usize>>, Vec<Event>), ParseError> {
    // Build name map from provided species tree
    let species_name_map = build_species_name_map(species_tree);

    // Parse only gene tree section
    let gene_root = parse_gene_tree(xml_content)?;
    let (gene_tree, node_mapping, event_mapping) =
        gene_node_to_flat_tree(&gene_root, &species_name_map)?;

    Ok((gene_tree, node_mapping, event_mapping))
}

/// Parse gene-tree-only RecPhyloXML file (no `<spTree>` section).
///
/// File-based version of `parse_gene_tree_only`.
pub fn parse_gene_tree_only_file(
    xml_filepath: &str,
    species_tree: &FlatTree,
) -> Result<(FlatTree, Vec<Option<usize>>, Vec<Event>), ParseError> {
    let xml_content = fs::read_to_string(xml_filepath)?;
    parse_gene_tree_only(&xml_content, species_tree)
}

/// Extract per-node annotations from a gene-tree-only XML.
///
/// Walks the parsed XML gene tree and returns a map from node name to
/// (species_location, Event) for every node (including loss nodes).
/// Callers can then look up only the names they care about.
pub fn parse_xml_gene_annotations(
    xml_content: &str,
) -> Result<HashMap<String, (String, Event)>, ParseError> {
    let gene_root = parse_gene_tree(xml_content)?;
    let mut annotations = HashMap::new();
    collect_annotations(&gene_root, &mut annotations);
    Ok(annotations)
}

/// File-based version of `parse_xml_gene_annotations`.
pub fn parse_xml_gene_annotations_file(
    xml_filepath: &str,
) -> Result<HashMap<String, (String, Event)>, ParseError> {
    let xml_content = fs::read_to_string(xml_filepath)?;
    parse_xml_gene_annotations(&xml_content)
}

fn collect_annotations(node: &GeneNode, map: &mut HashMap<String, (String, Event)>) {
    if !node.name.is_empty() {
        map.insert(
            node.name.clone(),
            (node.species_location.clone(), node.event_type.clone()),
        );
    }
    for child in &node.children {
        collect_annotations(child, map);
    }
}
