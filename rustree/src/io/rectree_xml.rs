//! XML serialization and parsing for reconciled trees (RecTree).

use crate::node::{FlatTree, RecTree};
use crate::node::rectree::Event;

/// Generate RecPhyloXML with multiple gene trees sharing one species tree.
///
/// The species tree is taken from the first RecTree. All gene trees are written
/// as separate `<recGeneTree>` sections, which thirdkind renders overlaid.
pub fn multi_to_xml(rec_trees: &[&RecTree]) -> String {
    if rec_trees.is_empty() {
        return String::new();
    }

    let first = rec_trees[0];
    let estimated_size = rec_trees.iter()
        .map(|rt| rt.gene_tree.nodes.len() * 200)
        .sum::<usize>() + 1000;
    let mut xml = String::with_capacity(estimated_size);

    // Header
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<recPhylo\n");
    xml.push_str("\txmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
    xml.push_str("\txsi:schemaLocation=\"http://www.recg.org ./recGeneTreeXML.xsd\"\n");
    xml.push_str("\txmlns=\"http://www.recg.org\">\n");

    // Species tree (shared, from first RecTree)
    xml.push_str("<spTree>\n<phylogeny>\n");
    first.write_species_clade(&mut xml, first.species_tree.root, 0);
    xml.push_str("</phylogeny>\n</spTree>\n");

    // Gene tree sections
    for rt in rec_trees {
        xml.push_str("<recGeneTree>\n<phylogeny rooted=\"true\">\n");
        rt.write_gene_clade(&mut xml, rt.gene_tree.root, 0);
        xml.push_str("</phylogeny>\n</recGeneTree>\n");
    }

    xml.push_str("</recPhylo>\n");
    xml
}

// ============================================================================
// RecTree XML serialization
// ============================================================================

impl RecTree {
    /// Exports the reconciled tree to RecPhyloXML format with branch lengths.
    pub fn to_xml(&self) -> String {
        let estimated_size = self.gene_tree.nodes.len() * 200 + 1000;
        let mut xml = String::with_capacity(estimated_size);

        // Header
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<recPhylo\n");
        xml.push_str("\txmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
        xml.push_str("\txsi:schemaLocation=\"http://www.recg.org ./recGeneTreeXML.xsd\"\n");
        xml.push_str("\txmlns=\"http://www.recg.org\">\n");

        // Species tree section
        xml.push_str("<spTree>\n<phylogeny>\n");
        self.write_species_clade(&mut xml, self.species_tree.root, 0);
        xml.push_str("</phylogeny>\n</spTree>\n");

        // Gene tree section
        xml.push_str("<recGeneTree>\n<phylogeny rooted=\"true\">\n");
        self.write_gene_clade(&mut xml, self.gene_tree.root, 0);
        xml.push_str("</phylogeny>\n</recGeneTree>\n");

        xml.push_str("</recPhylo>\n");
        xml
    }

    /// Helper function to write a species tree clade to XML.
    pub(crate) fn write_species_clade(&self, xml: &mut String, node_idx: usize, indent: usize) {
        let node = &self.species_tree.nodes[node_idx];
        let indent_str = Self::get_indent(indent);

        xml.push_str(&indent_str);
        xml.push_str("<clade>\n");
        xml.push_str(&indent_str);
        xml.push_str("\t<name>");
        if node.name.is_empty() {
            xml.push_str("root");
        } else {
            xml.push_str(&node.name);
        }
        xml.push_str("</name>\n");
        xml.push_str(&indent_str);
        xml.push_str("\t<branchLength>");
        xml.push_str(&node.length.to_string());
        xml.push_str("</branchLength>\n");

        if let Some(left_idx) = node.left_child {
            self.write_species_clade(xml, left_idx, indent + 1);
        }
        if let Some(right_idx) = node.right_child {
            self.write_species_clade(xml, right_idx, indent + 1);
        }

        xml.push_str(&indent_str);
        xml.push_str("</clade>\n");
    }

    /// Helper function to write a gene tree clade to XML with reconciliation events.
    pub(crate) fn write_gene_clade(&self, xml: &mut String, node_idx: usize, indent: usize) {
        let node = &self.gene_tree.nodes[node_idx];
        let species_idx_opt = self.node_mapping[node_idx];
        // Fallback to root species for unmapped nodes (e.g., after pruning).
        // Thirdkind requires speciesLocation on every event.
        let root_name = &self.species_tree.nodes[self.species_tree.root].name;
        let fallback_name = if root_name.is_empty() { "root" } else { root_name.as_str() };
        let species_name: Option<&str> = Some(
            species_idx_opt
                .map(|idx| {
                    let name = self.species_tree.nodes[idx].name.as_str();
                    if name.is_empty() { fallback_name } else { name }
                })
                .unwrap_or(fallback_name)
        );
        let event = &self.event_mapping[node_idx];
        let indent_str = Self::get_indent(indent);

        xml.push_str(&indent_str);
        xml.push_str("<clade>\n");

        // Node name
        xml.push_str(&indent_str);
        xml.push_str("\t<name>");
        match event {
            Event::Loss => xml.push_str("loss"),
            _ => xml.push_str(&node.name),
        }
        xml.push_str("</name>\n");

        xml.push_str(&indent_str);
        xml.push_str("\t<branchLength>");
        xml.push_str(&node.length.to_string());
        xml.push_str("</branchLength>\n");

        // Events section
        xml.push_str(&indent_str);
        xml.push_str("\t<eventsRec>\n");

        // Check if this is a transfer recipient
        let is_transfer_recipient = if let Some(parent_idx) = node.parent {
            if self.event_mapping[parent_idx] == Event::Transfer {
                let parent_species_idx = self.node_mapping[parent_idx];
                // Both must be Some and different for a transfer
                matches!((species_idx_opt, parent_species_idx), (Some(a), Some(b)) if a != b)
            } else {
                false
            }
        } else {
            false
        };

        if is_transfer_recipient {
            if let Some(name) = species_name {
                xml.push_str(&indent_str);
                xml.push_str("\t\t<transferBack destinationSpecies=\"");
                xml.push_str(name);
                xml.push_str("\"/>\n");
            }
        }

        // Write the appropriate event tags
        match event {
            Event::Speciation => {
                xml.push_str(&indent_str);
                if let Some(name) = species_name {
                    xml.push_str("\t\t<speciation speciesLocation=\"");
                    xml.push_str(name);
                    xml.push_str("\"/>\n");
                } else {
                    xml.push_str("\t\t<speciation/>\n");
                }
            }
            Event::Duplication => {
                xml.push_str(&indent_str);
                if let Some(name) = species_name {
                    xml.push_str("\t\t<duplication speciesLocation=\"");
                    xml.push_str(name);
                    xml.push_str("\"/>\n");
                } else {
                    xml.push_str("\t\t<duplication/>\n");
                }
            }
            Event::Transfer => {
                if node.left_child.is_some() && node.right_child.is_some() {
                    xml.push_str(&indent_str);
                    if let Some(name) = species_name {
                        xml.push_str("\t\t<branchingOut speciesLocation=\"");
                        xml.push_str(name);
                        xml.push_str("\"/>\n");
                    } else {
                        xml.push_str("\t\t<branchingOut/>\n");
                    }
                }
            }
            Event::Loss => {
                xml.push_str(&indent_str);
                if let Some(name) = species_name {
                    xml.push_str("\t\t<loss speciesLocation=\"");
                    xml.push_str(name);
                    xml.push_str("\"/>\n");
                } else {
                    xml.push_str("\t\t<loss/>\n");
                }
            }
            Event::Leaf => {
                xml.push_str(&indent_str);
                if let Some(name) = species_name {
                    xml.push_str("\t\t<leaf speciesLocation=\"");
                    xml.push_str(name);
                    xml.push_str("\"/>\n");
                } else {
                    xml.push_str("\t\t<leaf/>\n");
                }
            }
        }

        xml.push_str(&indent_str);
        xml.push_str("\t</eventsRec>\n");

        if let Some(left_idx) = node.left_child {
            self.write_gene_clade(xml, left_idx, indent + 1);
        }
        if let Some(right_idx) = node.right_child {
            self.write_gene_clade(xml, right_idx, indent + 1);
        }

        xml.push_str(&indent_str);
        xml.push_str("</clade>\n");
    }

    /// Get indent string for XML formatting.
    fn get_indent(indent: usize) -> String {
        const INDENTS: [&str; 10] = [
            "", "\t", "\t\t", "\t\t\t", "\t\t\t\t", "\t\t\t\t\t",
            "\t\t\t\t\t\t", "\t\t\t\t\t\t\t", "\t\t\t\t\t\t\t\t", "\t\t\t\t\t\t\t\t\t"
        ];
        if indent < INDENTS.len() {
            INDENTS[indent].to_string()
        } else {
            "\t".repeat(indent)
        }
    }

    // ========================================================================
    // Parsing constructors
    // ========================================================================

    /// Parse a RecPhyloXML string and create a RecTree.
    pub fn from_xml(xml_content: &str) -> Result<Self, String> {
        use super::recphyloxml::parse_recphyloxml;

        let (species_tree, gene_tree, node_mapping, event_mapping) =
            parse_recphyloxml(xml_content).map_err(|e| e.to_string())?;

        Ok(RecTree::new_owned(species_tree, gene_tree, node_mapping, event_mapping))
    }

    /// Parse a RecPhyloXML file and create a RecTree.
    pub fn from_xml_file(filepath: &str) -> Result<Self, String> {
        use super::recphyloxml::parse_recphyloxml_file;

        let (species_tree, gene_tree, node_mapping, event_mapping) =
            parse_recphyloxml_file(filepath).map_err(|e| e.to_string())?;

        Ok(RecTree::new_owned(species_tree, gene_tree, node_mapping, event_mapping))
    }

    /// Parse gene-tree-only RecPhyloXML with a separate species tree.
    ///
    /// This is useful when the species tree is provided separately (e.g., from a Newick file)
    /// and the XML only contains the reconciled gene tree (no `<spTree>` section).
    pub fn from_gene_tree_xml(xml_content: &str, species_tree: FlatTree) -> Result<Self, String> {
        use super::recphyloxml::parse_gene_tree_only;

        let (gene_tree, node_mapping, event_mapping) =
            parse_gene_tree_only(xml_content, &species_tree).map_err(|e| e.to_string())?;

        Ok(RecTree::new_owned(species_tree, gene_tree, node_mapping, event_mapping))
    }

    /// Parse gene-tree-only RecPhyloXML file with a separate species tree.
    pub fn from_gene_tree_xml_file(
        xml_filepath: &str,
        species_tree: FlatTree,
    ) -> Result<Self, String> {
        use super::recphyloxml::parse_gene_tree_only_file;

        let (gene_tree, node_mapping, event_mapping) =
            parse_gene_tree_only_file(xml_filepath, &species_tree).map_err(|e| e.to_string())?;

        Ok(RecTree::new_owned(species_tree, gene_tree, node_mapping, event_mapping))
    }

    /// Parse reconciled tree from separate files: Newick species tree + gene tree XML.
    ///
    /// # Example
    /// ```no_run
    /// use rustree::RecTree;
    ///
    /// let rec_tree = RecTree::from_separate_files(
    ///     "species_tree.nwk",
    ///     "gene_tree_rec.xml"
    /// )?;
    /// # Ok::<(), String>(())
    /// ```
    pub fn from_separate_files(
        species_newick_path: &str,
        gene_xml_path: &str,
    ) -> Result<Self, String> {
        use crate::newick::newick::parse_newick;
        use std::fs;

        let species_newick = fs::read_to_string(species_newick_path)
            .map_err(|e| format!("Failed to read species tree file: {}", e))?;

        let mut species_nodes = parse_newick(&species_newick)
            .map_err(|e| format!("Failed to parse species tree Newick: {}", e))?;

        let species_root = species_nodes
            .pop()
            .ok_or_else(|| "No tree found in species Newick file".to_string())?;

        let mut species_tree = species_root.to_flat_tree();
        species_tree.assign_depths();

        Self::from_gene_tree_xml_file(gene_xml_path, species_tree)
    }
}
