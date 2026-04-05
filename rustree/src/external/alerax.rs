/// ALERax integration module for gene tree reconciliation
///
/// This module provides functionality to call the external ALERax tool
/// for reconciling gene trees with species trees and parsing the results.
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use crate::node::{FlatTree, RecTree, map_by_topology};
use crate::node::gene_forest::GeneForest;

/// Configuration for ALERax reconciliation
pub struct AleRaxConfig {
    pub species_tree_path: PathBuf,
    pub families: Vec<GeneFamily>,
    pub families_file_path: PathBuf,
    pub output_dir: PathBuf,
    pub num_samples: usize,
    pub model_parametrization: ModelType,
    pub gene_tree_rooting: Option<String>,
    pub seed: Option<u64>,
    pub alerax_path: String,
}

/// Model parametrization type for ALERax
#[derive(Clone, Copy)]
pub enum ModelType {
    PerFamily,
    Global,
}

impl ModelType {
    fn as_str(&self) -> &str {
        match self {
            ModelType::PerFamily => "PER-FAMILY",
            ModelType::Global => "GLOBAL",
        }
    }
}

/// Gene family definition
#[derive(Clone)]
pub struct GeneFamily {
    pub name: String,
    pub gene_tree_path: PathBuf,
    pub gene_tree_newick: String,
}

/// Summary statistics for reconciliation events
#[derive(Clone, Debug)]
pub struct ReconciliationStatistics {
    /// Mean event counts across all samples
    pub mean_event_counts: EventCounts,
    /// Mean transfers between species pairs (source -> destination -> count)
    pub mean_transfers: HashMap<String, HashMap<String, f64>>,
    /// Mean events per species node (species -> event type -> count)
    pub events_per_species: HashMap<String, EventCounts>,
}

/// Event counts for reconciliation
#[derive(Clone, Debug, Default)]
pub struct EventCounts {
    pub speciations: f64,
    pub speciation_losses: f64,
    pub duplications: f64,
    pub duplication_losses: f64,
    pub transfers: f64,
    pub transfer_losses: f64,
    pub losses: f64,
    pub leaves: f64,
}

/// Result of ALERax reconciliation for one family
pub struct AleRaxFamilyResult {
    pub family_name: String,
    pub reconciled_trees: Vec<RecTree>,
    pub duplication_rate: f64,
    pub loss_rate: f64,
    pub transfer_rate: f64,
    pub likelihood: f64,
    pub statistics: ReconciliationStatistics,
}

/// A row from ALERax's meanSpeciesEventCounts.txt or totalSpeciesEventCounts.txt.
///
/// Represents the mean event counts for one species node across all
/// reconciliation samples.
#[derive(Clone, Debug)]
pub struct SpeciesEventRow {
    pub species_label: String,
    pub speciations: f64,
    pub duplications: f64,
    pub losses: f64,
    pub transfers: f64,
    pub presence: f64,
    pub origination: f64,
    pub copies: f64,
    pub singletons: f64,
    pub transfers_to: f64,
}

/// A transfer row from ALERax's totalTransfers.txt or meanTransfers.txt.
#[derive(Clone, Debug)]
pub struct TransferRow {
    pub source: String,
    pub destination: String,
    pub count: f64,
}

/// Full result of reconciling a GeneForest with ALERax.
///
/// Contains per-family results plus aggregate statistics across all families.
pub struct AleRaxForestResult {
    /// Per-family results (family_name -> result)
    pub family_results: HashMap<String, AleRaxFamilyResult>,
    /// Per-family mean species event counts (family_name -> Vec<SpeciesEventRow>)
    pub mean_species_event_counts: HashMap<String, Vec<SpeciesEventRow>>,
    /// Aggregate species event counts across all families
    pub total_species_event_counts: Vec<SpeciesEventRow>,
    /// Aggregate transfers across all families
    pub total_transfers: Vec<TransferRow>,
    /// The output directory (if keep_output=true)
    pub output_dir: Option<PathBuf>,
}

/// Check if ALERax is installed and accessible
fn check_alerax_installed(alerax_path: &str) -> Result<(), String> {
    // ALERax doesn't support --version, so just try to run it with --help
    // If it fails to execute at all, we know it's not installed
    Command::new(alerax_path)
        .arg("--help")
        .output()
        .map_err(|_| format!(
            "ALERax not found at '{}'. Please install ALERax:
  conda install -c bioconda alerax
  or download from: https://github.com/BenoitMorel/AleRax",
            alerax_path
        ))?;
    Ok(())
}

/// Extract species name from gene name (format: species_geneid)
fn extract_species_from_gene_name(gene_name: &str) -> String {
    gene_name.split('_')
        .next()
        .unwrap_or(gene_name)
        .to_string()
}

/// Validate that gene tree leaves map to species tree leaves
pub fn validate_inputs(
    species_tree: &FlatTree,
    gene_trees: &[(String, String)],  // (family_name, newick_string)
) -> Result<(), String> {
    // Check species tree is valid
    if species_tree.nodes.is_empty() {
        return Err("Species tree is empty".to_string());
    }

    let species_leaf_names: HashSet<String> = species_tree.nodes.iter()
        .filter(|n| n.left_child.is_none() && n.right_child.is_none())
        .map(|n| n.name.clone())
        .collect();

    if species_leaf_names.is_empty() {
        return Err("Species tree has no leaves".to_string());
    }

    // Check each gene tree
    for (family_name, newick) in gene_trees {
        // Parse gene tree to check leaves
        let mut nodes = crate::newick::parse_newick(newick)
            .map_err(|e| format!("Failed to parse gene tree '{}': {}", family_name, e))?;

        let root = nodes.pop()
            .ok_or_else(|| format!("Gene tree for family '{}' is empty", family_name))?;

        let gene_tree = root.to_flat_tree();

        // Extract gene leaf names and map to species
        let gene_leaf_species: HashSet<String> = gene_tree.nodes.iter()
            .filter(|n| n.left_child.is_none() && n.right_child.is_none())
            .map(|n| extract_species_from_gene_name(&n.name))
            .collect();

        // Check if gene leaves are subset of species leaves
        let unmapped: Vec<&String> = gene_leaf_species.iter()
            .filter(|name| !species_leaf_names.contains(*name))
            .collect();

        if !unmapped.is_empty() {
            return Err(format!(
                "Gene tree '{}' has leaves not in species tree: {:?}",
                family_name, unmapped
            ));
        }

        // Check minimum coverage (ALERax requires at least 4 species)
        if gene_leaf_species.len() < 4 {
            return Err(format!(
                "Gene tree '{}' covers only {} species (minimum 4 required for reconciliation)",
                family_name, gene_leaf_species.len()
            ));
        }
    }

    Ok(())
}

/// Create families.txt file in ALERax format
fn create_families_file(families: &[GeneFamily]) -> String {
    let mut content = String::from("[FAMILIES]\n");
    for family in families {
        content.push_str(&format!("- {}\n", family.name));
        content.push_str(&format!("gene_tree = {}\n", family.gene_tree_path.display()));
    }
    content
}

/// Prepare ALERax input files
fn prepare_alerax_input(config: &AleRaxConfig, species_tree_newick: &str) -> Result<(), String> {
    // Write species tree to file (ensure it ends with semicolon)
    let species_newick = if species_tree_newick.trim().ends_with(';') {
        species_tree_newick.to_string()
    } else {
        format!("{};", species_tree_newick.trim())
    };

    fs::write(&config.species_tree_path, species_newick)
        .map_err(|e| format!("Failed to write species tree file: {}", e))?;

    // Write each gene tree to file (ensure they end with semicolon)
    for family in &config.families {
        let gene_newick = if family.gene_tree_newick.trim().ends_with(';') {
            family.gene_tree_newick.clone()
        } else {
            format!("{};", family.gene_tree_newick.trim())
        };

        fs::write(&family.gene_tree_path, gene_newick)
            .map_err(|e| format!("Failed to write gene tree for family '{}': {}", family.name, e))?;
    }

    // Create families.txt file
    let families_txt = create_families_file(&config.families);
    fs::write(&config.families_file_path, families_txt)
        .map_err(|e| format!("Failed to write families file: {}", e))?;

    Ok(())
}

/// Build ALERax command
fn build_alerax_command(config: &AleRaxConfig) -> Command {
    let mut cmd = Command::new(&config.alerax_path);

    cmd.arg("-s").arg(&config.species_tree_path);
    cmd.arg("-f").arg(&config.families_file_path);
    cmd.arg("-p").arg(&config.output_dir);
    cmd.arg("--model-parametrization").arg(config.model_parametrization.as_str());
    cmd.arg("--gene-tree-samples").arg(config.num_samples.to_string());

    if let Some(ref rooting) = config.gene_tree_rooting {
        cmd.arg("--gene-tree-rooting").arg(rooting);
    }

    if let Some(seed) = config.seed {
        cmd.arg("--seed").arg(seed.to_string());
    }

    cmd
}

/// Parse ALERax rates file (D L T)
fn parse_rates_file(path: &Path) -> Result<(f64, f64, f64), String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read rates file: {}", e))?;

    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 2 {
        return Err("Invalid rates file format (expected at least 2 lines)".to_string());
    }

    // Line 1: "D L T"
    // Line 2: "0.098 1e-10 0.066"
    let values: Vec<&str> = lines[1].split_whitespace().collect();
    if values.len() != 3 {
        return Err(format!("Expected 3 rate values, found {}", values.len()));
    }

    let d = values[0].parse()
        .map_err(|_| format!("Failed to parse duplication rate: {}", values[0]))?;
    let l = values[1].parse()
        .map_err(|_| format!("Failed to parse loss rate: {}", values[1]))?;
    let t = values[2].parse()
        .map_err(|_| format!("Failed to parse transfer rate: {}", values[2]))?;

    Ok((d, l, t))
}

/// Parse ALERax per-family likelihoods file
fn parse_likelihoods_file(path: &Path) -> Result<HashMap<String, f64>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read likelihoods file: {}", e))?;

    let mut likelihoods = HashMap::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            let family_name = parts[0].to_string();
            let likelihood = parts[1].parse()
                .map_err(|_| format!("Failed to parse likelihood for family {}", family_name))?;
            likelihoods.insert(family_name, likelihood);
        }
    }

    Ok(likelihoods)
}

/// Parse ALERax eventCounts file for a single sample
fn parse_event_counts_file(path: &Path) -> Result<EventCounts, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read eventCounts file: {}", e))?;

    let mut counts = EventCounts::default();

    for line in content.lines() {
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() == 2 {
            let event_type = parts[0].trim();
            let count: f64 = parts[1].trim().parse()
                .map_err(|_| format!("Failed to parse count for {}", event_type))?;

            match event_type {
                "S" => counts.speciations = count,
                "SL" => counts.speciation_losses = count,
                "D" => counts.duplications = count,
                "DL" => counts.duplication_losses = count,
                "T" => counts.transfers = count,
                "TL" => counts.transfer_losses = count,
                "L" => counts.losses = count,
                "Leaf" => counts.leaves = count,
                _ => {} // Ignore unknown event types
            }
        }
    }

    Ok(counts)
}

/// Parse ALERax transfers file for a single sample
fn parse_transfers_file(path: &Path) -> Result<HashMap<String, HashMap<String, f64>>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read transfers file: {}", e))?;

    let mut transfers: HashMap<String, HashMap<String, f64>> = HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 3 {
            let source = parts[0].to_string();
            let destination = parts[1].to_string();
            let count: f64 = parts[2].parse()
                .map_err(|_| format!("Failed to parse transfer count in line: {}", line))?;

            transfers.entry(source)
                .or_default()
                .insert(destination, count);
        }
    }

    Ok(transfers)
}

/// Compute events per species from a reconciled tree
fn compute_events_per_species(rec_tree: &RecTree) -> HashMap<String, EventCounts> {
    use crate::node::Event;

    let mut species_events: HashMap<String, EventCounts> = HashMap::new();

    // Traverse all nodes in the reconciled tree
    for i in 0..rec_tree.gene_tree.nodes.len() {
        // Get the species node this gene node maps to
        let species_idx = match rec_tree.node_mapping[i] {
            Some(idx) => idx,
            None => continue, // skip unmapped nodes
        };
        let species_node = &rec_tree.species_tree.nodes[species_idx];
        let species_name = species_node.name.clone();

        // Get or create event counts for this species
        let counts = species_events.entry(species_name).or_default();

        // Increment count based on event type
        match rec_tree.event_mapping[i] {
            Event::Speciation => counts.speciations += 1.0,
            Event::Duplication => counts.duplications += 1.0,
            Event::Transfer => counts.transfers += 1.0,
            Event::Loss => counts.losses += 1.0,
            Event::Leaf => counts.leaves += 1.0,
        }
    }

    species_events
}

/// Aggregate statistics across multiple samples
fn aggregate_statistics(
    output_dir: &Path,
    family_name: &str,
    num_samples: usize,
    reconciled_trees: &[RecTree],
) -> Result<ReconciliationStatistics, String> {
    let reconciliation_dir = output_dir.join("reconciliations").join("all");

    // Aggregate event counts across samples
    let mut total_counts = EventCounts::default();
    for sample_idx in 0..num_samples {
        let event_counts_path = reconciliation_dir.join(format!("{}_eventCounts_{}.txt", family_name, sample_idx));
        if !event_counts_path.exists() {
            continue;
        }

        let counts = parse_event_counts_file(&event_counts_path)?;
        total_counts.speciations += counts.speciations;
        total_counts.speciation_losses += counts.speciation_losses;
        total_counts.duplications += counts.duplications;
        total_counts.duplication_losses += counts.duplication_losses;
        total_counts.transfers += counts.transfers;
        total_counts.transfer_losses += counts.transfer_losses;
        total_counts.losses += counts.losses;
        total_counts.leaves += counts.leaves;
    }

    // Compute mean event counts
    let n_samples = reconciled_trees.len() as f64;
    let mean_event_counts = EventCounts {
        speciations: total_counts.speciations / n_samples,
        speciation_losses: total_counts.speciation_losses / n_samples,
        duplications: total_counts.duplications / n_samples,
        duplication_losses: total_counts.duplication_losses / n_samples,
        transfers: total_counts.transfers / n_samples,
        transfer_losses: total_counts.transfer_losses / n_samples,
        losses: total_counts.losses / n_samples,
        leaves: total_counts.leaves / n_samples,
    };

    // Aggregate transfers across samples
    let mut transfer_totals: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for sample_idx in 0..num_samples {
        let transfers_path = reconciliation_dir.join(format!("{}_transfers_{}.txt", family_name, sample_idx));
        if !transfers_path.exists() {
            continue;
        }

        let transfers = parse_transfers_file(&transfers_path)?;
        for (source, destinations) in transfers {
            for (dest, count) in destinations {
                *transfer_totals.entry(source.clone())
                    .or_default()
                    .entry(dest.clone())
                    .or_insert(0.0) += count;
            }
        }
    }

    // Compute mean transfers
    let mut mean_transfers: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (source, destinations) in transfer_totals {
        for (dest, total) in destinations {
            mean_transfers.entry(source.clone())
                .or_default()
                .insert(dest, total / n_samples);
        }
    }

    // Compute events per species across all samples
    let mut species_event_totals: HashMap<String, EventCounts> = HashMap::new();
    for rec_tree in reconciled_trees {
        let species_events = compute_events_per_species(rec_tree);
        for (species, counts) in species_events {
            let total = species_event_totals.entry(species).or_default();
            total.speciations += counts.speciations;
            total.speciation_losses += counts.speciation_losses;
            total.duplications += counts.duplications;
            total.duplication_losses += counts.duplication_losses;
            total.transfers += counts.transfers;
            total.transfer_losses += counts.transfer_losses;
            total.losses += counts.losses;
            total.leaves += counts.leaves;
        }
    }

    // Compute mean events per species
    let mut events_per_species: HashMap<String, EventCounts> = HashMap::new();
    for (species, totals) in species_event_totals {
        events_per_species.insert(species, EventCounts {
            speciations: totals.speciations / n_samples,
            speciation_losses: totals.speciation_losses / n_samples,
            duplications: totals.duplications / n_samples,
            duplication_losses: totals.duplication_losses / n_samples,
            transfers: totals.transfers / n_samples,
            transfer_losses: totals.transfer_losses / n_samples,
            losses: totals.losses / n_samples,
            leaves: totals.leaves / n_samples,
        });
    }

    Ok(ReconciliationStatistics {
        mean_event_counts,
        mean_transfers,
        events_per_species,
    })
}

/// Parse ALERax reconciliation samples
fn parse_alerax_results(
    output_dir: &Path,
    families: &[GeneFamily],
    num_samples: usize,
) -> Result<HashMap<String, AleRaxFamilyResult>, String> {
    let mut results = HashMap::new();

    // Parse likelihoods file once
    let likelihoods_path = output_dir.join("per_fam_likelihoods.txt");
    let likelihoods = parse_likelihoods_file(&likelihoods_path)?;

    for family in families {
        // Parse rates
        let rates_path = output_dir
            .join("model_parameters")
            .join(format!("{}_rates.txt", family.name));
        let (d, l, t) = parse_rates_file(&rates_path)?;

        // Parse reconciliation samples
        let reconciliation_dir = output_dir.join("reconciliations").join("all");
        let mut reconciled_trees = Vec::new();

        for sample_idx in 0..num_samples {
            let xml_path = reconciliation_dir
                .join(format!("{}_sample_{}.xml", family.name, sample_idx));

            // Skip if sample doesn't exist (ALERax might generate fewer samples)
            if !xml_path.exists() {
                continue;
            }

            // Use existing RecTree::from_xml_file to parse
            let rec_tree = RecTree::from_xml_file(xml_path.to_str().unwrap())
                .map_err(|e| format!("Failed to parse RecPhyloXML for family '{}' sample {}: {}",
                    family.name, sample_idx, e))?;
            reconciled_trees.push(rec_tree);
        }

        if reconciled_trees.is_empty() {
            return Err(format!("No reconciliation samples found for family '{}'", family.name));
        }

        let likelihood = likelihoods.get(&family.name)
            .copied()
            .ok_or_else(|| format!("No likelihood found for family '{}'", family.name))?;

        // Compute summary statistics
        let statistics = aggregate_statistics(output_dir, &family.name, num_samples, &reconciled_trees)?;

        results.insert(family.name.clone(), AleRaxFamilyResult {
            family_name: family.name.clone(),
            reconciled_trees,
            duplication_rate: d,
            loss_rate: l,
            transfer_rate: t,
            likelihood,
            statistics,
        });
    }

    Ok(results)
}

/// Run ALERax and return reconciliation results
pub fn run_alerax(
    config: AleRaxConfig,
    species_tree_newick: &str,
) -> Result<HashMap<String, AleRaxFamilyResult>, String> {
    // Check if ALERax is installed
    check_alerax_installed(&config.alerax_path)?;

    // Prepare input files
    prepare_alerax_input(&config, species_tree_newick)?;

    // Build and execute command
    let mut cmd = build_alerax_command(&config);

    // Show command for debugging
    let cmd_str = format!("{:?}", cmd);
    log::info!("Running ALERax command: {}", cmd_str);

    let output = cmd.output()
        .map_err(|e| format!("Failed to execute ALERax: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        return Err(format!(
            "ALERax failed with exit code {:?}:\nStderr:\n{}\nStdout:\n{}",
            output.status.code(), stderr, stdout
        ));
    }

    // Check if output directory was created
    if !config.output_dir.exists() {
        return Err(format!(
            "ALERax completed but output directory does not exist: {}\nStdout:\n{}\nStderr:\n{}",
            config.output_dir.display(), stdout, stderr
        ));
    }

    // Parse results
    parse_alerax_results(&config.output_dir, &config.families, config.num_samples)
        .map_err(|e| format!(
            "{}.\n\nALERax output directory: {}\nStdout:\n{}\nStderr:\n{}",
            e, config.output_dir.display(), stdout, stderr
        ))
}

/// Parse ALERax meanSpeciesEventCounts.txt or totalSpeciesEventCounts.txt.
///
/// Format: CSV with header
/// `species_label, speciations, duplications, losses, transfers, presence, origination, copies, singletons, transfers_to`
fn parse_species_event_counts_file(path: &Path) -> Result<Vec<SpeciesEventRow>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read species event counts file '{}': {}", path.display(), e))?;

    let mut rows = Vec::new();
    let mut lines = content.lines();

    // Skip header line
    let _header = lines.next()
        .ok_or_else(|| format!("Empty species event counts file: {}", path.display()))?;

    for line in lines {
        let line = line.trim();
        if line.is_empty() { continue; }

        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 10 {
            return Err(format!(
                "Expected 10 columns in species event counts, got {}: {}",
                parts.len(), line
            ));
        }

        rows.push(SpeciesEventRow {
            species_label: parts[0].to_string(),
            speciations: parts[1].parse().map_err(|_| format!("Bad speciations value: '{}'", parts[1]))?,
            duplications: parts[2].parse().map_err(|_| format!("Bad duplications value: '{}'", parts[2]))?,
            losses: parts[3].parse().map_err(|_| format!("Bad losses value: '{}'", parts[3]))?,
            transfers: parts[4].parse().map_err(|_| format!("Bad transfers value: '{}'", parts[4]))?,
            presence: parts[5].parse().map_err(|_| format!("Bad presence value: '{}'", parts[5]))?,
            origination: parts[6].parse().map_err(|_| format!("Bad origination value: '{}'", parts[6]))?,
            copies: parts[7].parse().map_err(|_| format!("Bad copies value: '{}'", parts[7]))?,
            singletons: parts[8].parse().map_err(|_| format!("Bad singletons value: '{}'", parts[8]))?,
            transfers_to: parts[9].parse().map_err(|_| format!("Bad transfers_to value: '{}'", parts[9]))?,
        });
    }

    Ok(rows)
}

/// Parse ALERax totalTransfers.txt or meanTransfers.txt.
///
/// Format: space-separated `source destination count`
fn parse_transfer_rows_file(path: &Path) -> Result<Vec<TransferRow>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read transfers file '{}': {}", path.display(), e))?;

    let mut rows = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 { continue; }

        rows.push(TransferRow {
            source: parts[0].to_string(),
            destination: parts[1].to_string(),
            count: parts[2].parse()
                .map_err(|_| format!("Bad transfer count '{}' in file {}", parts[2], path.display()))?,
        });
    }

    Ok(rows)
}

/// Run ALERax with streaming stderr output.
///
/// Streams ALERax stderr to the terminal in real-time so the user can see progress.
/// Captures both stdout and stderr for error reporting.
fn run_alerax_streaming(config: &AleRaxConfig, species_tree_newick: &str) -> Result<(), String> {
    check_alerax_installed(&config.alerax_path)?;
    prepare_alerax_input(config, species_tree_newick)?;

    let mut cmd = build_alerax_command(config);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let cmd_str = format!("{:?}", cmd);
    log::info!("Running ALERax: {}", cmd_str);

    let mut child = cmd.spawn()
        .map_err(|e| format!("Failed to spawn ALERax: {}", e))?;

    // Read stderr in a thread — print each line in real-time AND capture
    let stderr = child.stderr.take()
        .ok_or_else(|| "Failed to capture ALERax stderr".to_string())?;
    let stderr_handle = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        let mut captured = String::new();
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    log::debug!("[ALERax] {}", line);
                    captured.push_str(&line);
                    captured.push('\n');
                }
                Err(_) => break,
            }
        }
        captured
    });

    // Read stdout in a thread — capture silently
    let stdout = child.stdout.take()
        .ok_or_else(|| "Failed to capture ALERax stdout".to_string())?;
    let stdout_handle = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        let mut captured = String::new();
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    captured.push_str(&line);
                    captured.push('\n');
                }
                Err(_) => break,
            }
        }
        captured
    });

    let status = child.wait()
        .map_err(|e| format!("Failed to wait for ALERax: {}", e))?;

    let stderr_output = stderr_handle.join()
        .map_err(|_| "Failed to join stderr reader thread".to_string())?;
    let stdout_output = stdout_handle.join()
        .map_err(|_| "Failed to join stdout reader thread".to_string())?;

    if !status.success() {
        return Err(format!(
            "ALERax failed with exit code {:?}:\nStderr:\n{}\nStdout:\n{}",
            status.code(), stderr_output, stdout_output
        ));
    }

    if !config.output_dir.exists() {
        return Err(format!(
            "ALERax completed but output directory does not exist: {}\nStdout:\n{}\nStderr:\n{}",
            config.output_dir.display(), stdout_output, stderr_output
        ));
    }

    Ok(())
}

/// Rename species labels in SpeciesEventRow vectors using a name mapping.
fn rename_species_event_rows(
    rows: &mut [SpeciesEventRow],
    mapping: &HashMap<String, String>,
) {
    for row in rows {
        if let Some(orig) = mapping.get(&row.species_label) {
            row.species_label = orig.clone();
        }
    }
}

/// Rename source/destination in TransferRow vectors using a name mapping.
fn rename_transfer_rows(
    rows: &mut [TransferRow],
    mapping: &HashMap<String, String>,
) {
    for row in rows {
        if let Some(orig) = mapping.get(&row.source) {
            row.source = orig.clone();
        }
        if let Some(orig) = mapping.get(&row.destination) {
            row.destination = orig.clone();
        }
    }
}

/// Reconcile a GeneForest with ALERax.
///
/// This function:
/// 1. Creates a temp directory (or uses provided `output_dir`)
/// 2. Writes species tree and gene trees as Newick files
/// 3. Creates families.txt for ALERax
/// 4. Runs ALERax with streaming console output
/// 5. Parses all results (RecPhyloXML, rates, likelihoods, summary stats)
/// 6. Auto-renames species tree nodes and gene tree leaf names back to original names
/// 7. Returns a comprehensive result object
#[allow(clippy::too_many_arguments)]
pub fn reconcile_forest(
    forest: &GeneForest,
    output_dir: Option<PathBuf>,
    num_samples: usize,
    model: ModelType,
    gene_tree_rooting: Option<String>,
    seed: Option<u64>,
    alerax_path: &str,
    keep_output: bool,
) -> Result<AleRaxForestResult, String> {
    if forest.is_empty() {
        return Err("GeneForest has no gene trees to reconcile".to_string());
    }

    let original_species_tree = forest.species_tree();
    let species_newick = original_species_tree.to_newick()
        .map_err(|e| format!("Failed to serialize species tree: {}", e))?;

    // Create output directory
    let (output_path, _temp_dir) = match output_dir {
        Some(dir) => {
            fs::create_dir_all(&dir)
                .map_err(|e| format!("Failed to create output directory: {}", e))?;
            (dir, None)
        }
        None => {
            let temp_dir = tempfile::TempDir::new()
                .map_err(|e| format!("Failed to create temp directory: {}", e))?;
            let path = temp_dir.path().to_path_buf();
            (path, Some(temp_dir))
        }
    };

    let input_dir = output_path.join("input");
    fs::create_dir_all(&input_dir)
        .map_err(|e| format!("Failed to create input directory: {}", e))?;

    let species_tree_path = input_dir.join("species_tree.newick");
    let families_file_path = input_dir.join("families.txt");

    // Build gene families
    let mut families = Vec::new();
    for (idx, rec_tree) in forest.iter().enumerate() {
        let family_name = format!("family_{}", idx);
        let gene_newick = rec_tree.gene_tree.to_newick()
            .map_err(|e| format!("Failed to serialize gene tree {}: {}", idx, e))?;
        let gene_tree_path = input_dir.join(format!("{}.newick", family_name));

        families.push(GeneFamily {
            name: family_name,
            gene_tree_path,
            gene_tree_newick: gene_newick,
        });
    }

    let alerax_output_dir = output_path.join("alerax_output");

    let config = AleRaxConfig {
        species_tree_path,
        families: families.clone(),
        families_file_path,
        output_dir: alerax_output_dir.clone(),
        num_samples,
        model_parametrization: model,
        gene_tree_rooting,
        seed,
        alerax_path: alerax_path.to_string(),
    };

    // Run ALERax with streaming output
    run_alerax_streaming(&config, &species_newick)?;

    // Parse per-family results (existing logic)
    let mut family_results = parse_alerax_results(&alerax_output_dir, &families, num_samples)?;

    // Parse summary statistics files
    let summaries_dir = alerax_output_dir.join("reconciliations").join("summaries");
    let mut mean_species_event_counts: HashMap<String, Vec<SpeciesEventRow>> = HashMap::new();

    for family in &families {
        let path = summaries_dir.join(format!("{}_meanSpeciesEventCounts.txt", family.name));
        if path.exists() {
            let rows = parse_species_event_counts_file(&path)?;
            mean_species_event_counts.insert(family.name.clone(), rows);
        }
    }

    let total_event_counts_path = alerax_output_dir
        .join("reconciliations")
        .join("totalSpeciesEventCounts.txt");
    let mut total_species_event_counts = if total_event_counts_path.exists() {
        parse_species_event_counts_file(&total_event_counts_path)?
    } else {
        Vec::new()
    };

    let total_transfers_path = alerax_output_dir
        .join("reconciliations")
        .join("totalTransfers.txt");
    let mut total_transfers = if total_transfers_path.exists() {
        parse_transfer_rows_file(&total_transfers_path)?
    } else {
        Vec::new()
    };

    // Auto-rename: map ALERax species names back to original names
    // Get ALERax species tree from the first parsed RecTree
    let first_family_name = &families[0].name;
    let first_family_result = family_results.get(first_family_name)
        .ok_or_else(|| "No results for first family".to_string())?;
    let alerax_species_tree = &first_family_result.reconciled_trees[0].species_tree;

    // Build topology mapping: alerax_idx -> original_idx
    let topo_mapping = map_by_topology(original_species_tree, alerax_species_tree)
        .map_err(|e| format!("Failed to build topology mapping for auto-rename: {}", e))?;

    // Build name mapping: alerax_name -> original_name
    let mut alerax_to_original: HashMap<String, String> = HashMap::new();
    for (alerax_idx, original_idx) in &topo_mapping {
        let alerax_name = &alerax_species_tree.nodes[*alerax_idx].name;
        let original_name = &original_species_tree.nodes[*original_idx].name;
        alerax_to_original.insert(alerax_name.clone(), original_name.clone());
    }

    // Build a single renamed species tree and share it
    let renamed_species = {
        let mut tree = (**alerax_species_tree).clone();
        for node in &mut tree.nodes {
            if let Some(orig) = alerax_to_original.get(&node.name) {
                node.name = orig.clone();
            }
        }
        Arc::new(tree)
    };

    // Rename in all family results
    for result in family_results.values_mut() {
        for rec_tree in &mut result.reconciled_trees {
            // Replace species tree Arc
            rec_tree.species_tree = Arc::clone(&renamed_species);
            // Rename gene tree leaf names
            for node in &mut rec_tree.gene_tree.nodes {
                if node.left_child.is_none() && node.right_child.is_none() {
                    if let Some(pos) = node.name.rfind('_') {
                        let alerax_species = &node.name[..pos];
                        let suffix = &node.name[pos..];
                        if let Some(original_species) = alerax_to_original.get(alerax_species) {
                            node.name = format!("{}{}", original_species, suffix);
                        }
                    }
                }
            }
        }

        // Rename species keys in ReconciliationStatistics
        let old_eps = std::mem::take(&mut result.statistics.events_per_species);
        for (species_name, counts) in old_eps {
            let renamed = alerax_to_original.get(&species_name)
                .cloned()
                .unwrap_or(species_name);
            result.statistics.events_per_species.insert(renamed, counts);
        }

        let old_mt = std::mem::take(&mut result.statistics.mean_transfers);
        for (source, dests) in old_mt {
            let renamed_source = alerax_to_original.get(&source)
                .cloned()
                .unwrap_or(source);
            let mut renamed_dests = HashMap::new();
            for (dest, count) in dests {
                let renamed_dest = alerax_to_original.get(&dest)
                    .cloned()
                    .unwrap_or(dest);
                renamed_dests.insert(renamed_dest, count);
            }
            result.statistics.mean_transfers.insert(renamed_source, renamed_dests);
        }
    }

    // Rename in summary statistics
    for rows in mean_species_event_counts.values_mut() {
        rename_species_event_rows(rows, &alerax_to_original);
    }
    rename_species_event_rows(&mut total_species_event_counts, &alerax_to_original);
    rename_transfer_rows(&mut total_transfers, &alerax_to_original);

    // Handle output directory retention
    let kept_output_dir = if keep_output {
        if let Some(td) = _temp_dir {
            let path = td.path().to_path_buf();
            let _ = td.keep(); // prevents cleanup
            Some(path)
        } else {
            Some(output_path)
        }
    } else {
        None
    };

    Ok(AleRaxForestResult {
        family_results,
        mean_species_event_counts,
        total_species_event_counts,
        total_transfers,
        output_dir: kept_output_dir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_species_from_gene_name() {
        assert_eq!(extract_species_from_gene_name("n5_20"), "n5");
        assert_eq!(extract_species_from_gene_name("species_A_gene123"), "species");
        assert_eq!(extract_species_from_gene_name("single"), "single");
    }

    #[test]
    fn test_create_families_file() {
        let families = vec![
            GeneFamily {
                name: "fam1".to_string(),
                gene_tree_path: PathBuf::from("/tmp/tree1.nwk"),
                gene_tree_newick: "".to_string(),
            },
            GeneFamily {
                name: "fam2".to_string(),
                gene_tree_path: PathBuf::from("/tmp/tree2.nwk"),
                gene_tree_newick: "".to_string(),
            },
        ];

        let content = create_families_file(&families);
        assert!(content.contains("[FAMILIES]"));
        assert!(content.contains("- fam1"));
        assert!(content.contains("gene_tree = /tmp/tree1.nwk"));
        assert!(content.contains("- fam2"));
        assert!(content.contains("gene_tree = /tmp/tree2.nwk"));
    }

    #[test]
    fn test_model_type_as_str() {
        assert_eq!(ModelType::PerFamily.as_str(), "PER-FAMILY");
        assert_eq!(ModelType::Global.as_str(), "GLOBAL");
    }

    #[test]
    fn test_parse_species_event_counts_file() {
        let path = Path::new("testdata/test_data_3/output_alerax/reconciliations/summaries/family_1_meanSpeciesEventCounts.txt");
        if !path.exists() {
            return; // Skip if test data not available
        }
        let rows = parse_species_event_counts_file(path).unwrap();
        assert!(!rows.is_empty(), "Should have parsed species event count rows");

        // Check that Root row exists
        let root_row = rows.iter().find(|r| r.species_label == "Root");
        assert!(root_row.is_some(), "Should have a Root row");
        let root = root_row.unwrap();
        assert!(root.speciations >= 0.0);
        assert!(root.copies >= 0.0);
    }

    #[test]
    fn test_parse_total_species_event_counts() {
        let path = Path::new("testdata/test_data_3/output_alerax/reconciliations/totalSpeciesEventCounts.txt");
        if !path.exists() {
            return;
        }
        let rows = parse_species_event_counts_file(path).unwrap();
        assert!(!rows.is_empty());

        // Total should have same species labels as per-family
        let labels: Vec<&str> = rows.iter().map(|r| r.species_label.as_str()).collect();
        assert!(labels.contains(&"Root"));
    }

    #[test]
    fn test_parse_transfer_rows_file() {
        let path = Path::new("testdata/test_data_3/output_alerax/reconciliations/totalTransfers.txt");
        if !path.exists() {
            return;
        }
        let rows = parse_transfer_rows_file(path).unwrap();
        assert!(!rows.is_empty(), "Should have parsed transfer rows");

        // Check structure
        for row in &rows {
            assert!(!row.source.is_empty());
            assert!(!row.destination.is_empty());
            assert!(row.count >= 0.0);
        }
    }

    #[test]
    fn test_rename_species_event_rows() {
        let mut rows = vec![
            SpeciesEventRow {
                species_label: "n1".to_string(),
                speciations: 1.0, duplications: 0.0, losses: 0.0, transfers: 0.0,
                presence: 1.0, origination: 0.0, copies: 1.0, singletons: 1.0, transfers_to: 0.0,
            },
            SpeciesEventRow {
                species_label: "n2".to_string(),
                speciations: 2.0, duplications: 0.0, losses: 0.0, transfers: 0.0,
                presence: 1.0, origination: 0.0, copies: 2.0, singletons: 0.0, transfers_to: 0.0,
            },
        ];

        let mut mapping = HashMap::new();
        mapping.insert("n1".to_string(), "SpeciesA".to_string());
        mapping.insert("n2".to_string(), "SpeciesB".to_string());

        rename_species_event_rows(&mut rows, &mapping);
        assert_eq!(rows[0].species_label, "SpeciesA");
        assert_eq!(rows[1].species_label, "SpeciesB");
    }

    #[test]
    fn test_rename_transfer_rows() {
        let mut rows = vec![
            TransferRow { source: "n1".to_string(), destination: "n2".to_string(), count: 1.0 },
        ];

        let mut mapping = HashMap::new();
        mapping.insert("n1".to_string(), "SpeciesA".to_string());
        mapping.insert("n2".to_string(), "SpeciesB".to_string());

        rename_transfer_rows(&mut rows, &mapping);
        assert_eq!(rows[0].source, "SpeciesA");
        assert_eq!(rows[0].destination, "SpeciesB");
    }
}
