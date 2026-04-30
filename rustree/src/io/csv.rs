//! CSV file writing utilities for tree events.

use std::fs::File;
use std::io::{self, Write, BufWriter};
use crate::bd::TreeEvent;
use crate::dtl::DTLEvent;
use crate::node::FlatTree;

/// Save birth-death events to a CSV file.
///
/// # Arguments
/// * `events` - Vector of TreeEvent to save
/// * `tree` - The species tree to resolve node names from indices
/// * `filename` - Path to the output CSV file
///
/// # Returns
/// Result indicating success or error
pub fn save_bd_events_to_csv(events: &[TreeEvent], tree: &FlatTree, filename: &str) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", TreeEvent::csv_header())?;
    for event in events {
        writeln!(writer, "{}", event.to_csv_row(tree).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?)?;
    }
    writer.flush()?;
    Ok(())
}

/// Save DTL events to a CSV file.
///
/// # Arguments
/// * `events` - Vector of DTLEvent to save
/// * `species_tree` - The species tree to resolve species names from indices
/// * `gene_tree` - The gene tree to resolve gene node names from indices
/// * `filename` - Path to the output CSV file
///
/// # Returns
/// Result indicating success or IO error
pub fn save_dtl_events_to_csv(events: &[DTLEvent], species_tree: &FlatTree, gene_tree: &FlatTree, filename: &str) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", DTLEvent::csv_header())?;
    for event in events {
        writeln!(writer, "{}", event.to_csv_row(species_tree, gene_tree))?;
    }
    writer.flush()?;
    Ok(())
}
