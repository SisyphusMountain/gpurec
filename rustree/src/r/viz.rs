// SVG visualization R binding functions

/// Generate SVG visualization using thirdkind.
///
/// @param gene_tree_list A gene tree list from simulate_dtl_r
/// @param filepath Optional path to save the SVG (NULL to return string only)
/// @param open_browser Whether to open in browser (default FALSE)
/// @return SVG content as string
/// @export
#[extendr]
fn gene_tree_to_svg_r(gene_tree_list: List, filepath: Robj, open_browser: bool) -> Result<String> {
    let xml = gene_tree_to_xml_r(gene_tree_list)?;

    let temp_dir = std::env::temp_dir();
    let xml_path = temp_dir.join("rustree_temp.recphyloxml");
    let svg_path = temp_dir.join("rustree_temp.svg");

    fs::write(&xml_path, &xml)
        .map_err(|e| format!("Failed to write temp XML: {}", e))?;

    let mut cmd = Command::new("thirdkind");
    cmd.arg("-f").arg(&xml_path)
       .arg("-o").arg(&svg_path);

    if open_browser {
        cmd.arg("-b");
    }

    let output = cmd.output()
        .map_err(|e| format!("Failed to run thirdkind. Is it installed? (`cargo install thirdkind`)\nError: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("thirdkind failed: {}", stderr).into());
    }

    let svg = fs::read_to_string(&svg_path)
        .map_err(|e| format!("Failed to read SVG output: {}", e))?;

    if !filepath.is_null() {
        let path = filepath.as_str().ok_or("filepath must be a string")?;
        fs::write(path, &svg)
            .map_err(|e| format!("Failed to write SVG file: {}", e))?;
    }

    let _ = fs::remove_file(&xml_path);
    let _ = fs::remove_file(&svg_path);

    Ok(svg)
}
