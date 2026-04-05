// Streaming DTL R binding functions

/// Stream DTL gene trees (per-gene model) to RecPhyloXML files.
///
/// Simulates `n` gene trees one at a time using the per-gene-copy DTL model
/// and writes each tree as a RecPhyloXML file in `output_dir`. Files are
/// named `gene_0000.xml`, `gene_0001.xml`, etc.
///
/// This is memory-efficient for large batches because only one tree is held
/// in memory at a time.
///
/// @param newick Species tree in Newick format
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate
/// @param lambda_t Transfer rate
/// @param lambda_l Loss rate
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced
/// @param seed Optional random seed for reproducibility
/// @param output_dir Directory to write XML files to (created if it doesn't exist)
/// @return The output directory path
/// @export
#[extendr]
fn simulate_dtl_stream_xml_r(
    newick: &str,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
    output_dir: &str,
) -> Result<String> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    crate::bindings_common::validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;

    let species_tree = parse_newick_to_flattree(newick)?;
    let mut rng = make_rng(&seed)?;
    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    simulate_dtl_iter(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        alpha, repl,
        n as usize, require_extant, &mut rng,
    )
    .map_err(Error::Other)?
    .save_xml(output_dir)
    .map_err(Error::Other)?;

    Ok(output_dir.to_string())
}

/// Stream DTL gene trees (per-gene model) to Newick files.
///
/// Simulates `n` gene trees one at a time using the per-gene-copy DTL model
/// and writes each gene tree in Newick format to `output_dir`. Files are
/// named `gene_0000.nwk`, `gene_0001.nwk`, etc.
///
/// Note: Newick format does not preserve reconciliation information (species
/// mapping, events). Use `simulate_dtl_stream_xml_r` for full reconciled trees.
///
/// @param newick Species tree in Newick format
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate
/// @param lambda_t Transfer rate
/// @param lambda_l Loss rate
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced
/// @param seed Optional random seed for reproducibility
/// @param output_dir Directory to write Newick files to (created if it doesn't exist)
/// @return The output directory path
/// @export
#[extendr]
fn simulate_dtl_stream_newick_r(
    newick: &str,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
    output_dir: &str,
) -> Result<String> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    crate::bindings_common::validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;

    let species_tree = parse_newick_to_flattree(newick)?;
    let mut rng = make_rng(&seed)?;
    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    simulate_dtl_iter(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        alpha, repl,
        n as usize, require_extant, &mut rng,
    )
    .map_err(Error::Other)?
    .save_newick(output_dir)
    .map_err(Error::Other)?;

    Ok(output_dir.to_string())
}

/// Stream DTL gene trees (per-species/Zombi model) to RecPhyloXML files.
///
/// Simulates `n` gene trees one at a time using the Zombi-style per-species
/// DTL model and writes each tree as a RecPhyloXML file in `output_dir`.
/// Files are named `gene_0000.xml`, `gene_0001.xml`, etc.
///
/// In the per-species model, the event rate is proportional to the number of
/// SPECIES with gene copies, NOT the number of gene copies themselves.
///
/// @param newick Species tree in Newick format
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate per species per unit time
/// @param lambda_t Transfer rate per species per unit time
/// @param lambda_l Loss rate per species per unit time
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced
/// @param seed Optional random seed for reproducibility
/// @param output_dir Directory to write XML files to (created if it doesn't exist)
/// @return The output directory path
/// @export
#[extendr]
fn simulate_dtl_per_species_stream_xml_r(
    newick: &str,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
    output_dir: &str,
) -> Result<String> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    crate::bindings_common::validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;

    let species_tree = parse_newick_to_flattree(newick)?;
    let mut rng = make_rng(&seed)?;
    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    simulate_dtl_per_species_iter(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        alpha, repl,
        n as usize, require_extant, &mut rng,
    )
    .map_err(Error::Other)?
    .save_xml(output_dir)
    .map_err(Error::Other)?;

    Ok(output_dir.to_string())
}

/// Stream DTL gene trees (per-species/Zombi model) to Newick files.
///
/// Simulates `n` gene trees one at a time using the Zombi-style per-species
/// DTL model and writes each gene tree in Newick format to `output_dir`.
/// Files are named `gene_0000.nwk`, `gene_0001.nwk`, etc.
///
/// Note: Newick format does not preserve reconciliation information (species
/// mapping, events). Use `simulate_dtl_per_species_stream_xml_r` for full
/// reconciled trees.
///
/// @param newick Species tree in Newick format
/// @param n Number of gene trees to simulate
/// @param lambda_d Duplication rate per species per unit time
/// @param lambda_t Transfer rate per species per unit time
/// @param lambda_l Loss rate per species per unit time
/// @param transfer_alpha Optional distance decay for assortative transfers (NULL = uniform)
/// @param replacement_transfer Optional probability of replacement transfer (NULL = no replacement)
/// @param require_extant If TRUE, retry until a tree with extant genes is produced
/// @param seed Optional random seed for reproducibility
/// @param output_dir Directory to write Newick files to (created if it doesn't exist)
/// @return The output directory path
/// @export
#[extendr]
fn simulate_dtl_per_species_stream_newick_r(
    newick: &str,
    n: i32,
    lambda_d: f64,
    lambda_t: f64,
    lambda_l: f64,
    transfer_alpha: Robj,
    replacement_transfer: Robj,
    require_extant: bool,
    seed: Robj,
    output_dir: &str,
) -> Result<String> {
    if n <= 0 {
        return Err("Number of trees must be positive".into());
    }
    crate::bindings_common::validate_dtl_rates(lambda_d, lambda_t, lambda_l)?;

    let species_tree = parse_newick_to_flattree(newick)?;
    let mut rng = make_rng(&seed)?;
    let alpha = extract_alpha(&transfer_alpha)?;
    let repl = extract_replacement(&replacement_transfer)?;

    let origin_species = species_tree.root;
    simulate_dtl_per_species_iter(
        &species_tree, origin_species,
        lambda_d, lambda_t, lambda_l,
        alpha, repl,
        n as usize, require_extant, &mut rng,
    )
    .map_err(Error::Other)?
    .save_newick(output_dir)
    .map_err(Error::Other)?;

    Ok(output_dir.to_string())
}
