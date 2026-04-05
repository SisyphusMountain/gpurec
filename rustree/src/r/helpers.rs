// Internal helper functions (not exported to R)

/// Helper: parse a Newick string into a FlatTree with depths assigned.
fn parse_newick_to_flattree(newick: &str) -> Result<FlatTree> {
    use crate::newick::parse_newick;

    let mut nodes = parse_newick(newick)
        .map_err(|e| format!("Failed to parse Newick: {}", e))?;
    let mut root = nodes.pop()
        .ok_or("No tree found in Newick string")?;
    root.zero_root_length();
    root.assign_depths(0.0);
    let mut tree = root.to_flat_tree();
    tree.assign_depths();
    Ok(tree)
}

/// Helper: create an StdRng from an optional R seed value.
fn make_rng(seed: &Robj) -> Result<StdRng> {
    if seed.is_null() || seed.is_na() {
        Ok(StdRng::from_entropy())
    } else {
        match seed.as_integer() {
            Some(s) => Ok(StdRng::seed_from_u64(s as u64)),
            None => Err("seed must be an integer".into()),
        }
    }
}

/// Helper: extract optional transfer_alpha from an Robj.
fn extract_alpha(transfer_alpha: &Robj) -> Result<Option<f64>> {
    if transfer_alpha.is_null() || transfer_alpha.is_na() {
        Ok(None)
    } else {
        Ok(Some(transfer_alpha.as_real().ok_or("transfer_alpha must be a number")?))
    }
}

/// Helper: extract optional replacement_transfer from an Robj.
fn extract_replacement(replacement_transfer: &Robj) -> Result<Option<f64>> {
    if replacement_transfer.is_null() || replacement_transfer.is_na() {
        Ok(None)
    } else {
        Ok(Some(replacement_transfer.as_real().ok_or("replacement_transfer must be a number")?))
    }
}
