use crate::node::FlatTree;
use regex::Regex;
use lazy_static::lazy_static;

/// Strips ANSI escape sequences from a string.
fn strip_ansi_codes(s: &str) -> String {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    }
    RE.replace_all(s, "").to_string()
}

/// Compare two string values. If they differ, returns the new value in red with the old value in parentheses.
fn format_diff(old_val: &str, new_val: &str) -> String {
    if old_val == new_val {
        new_val.to_string()
    } else {
        format!("\x1b[31m{}\x1b[0m ({})", new_val, old_val)
    }
}

/// Build the "new data" (without color codes) as a 2D vector of strings.
/// Each row corresponds to a node in the flat tree.
fn build_new_data(flat_tree: &FlatTree) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for (i, node) in flat_tree.nodes.iter().enumerate() {
        rows.push(vec![
            i.to_string(),
            node.name.clone(),
            node.left_child.map_or("None".to_string(), |v| v.to_string()),
            node.right_child.map_or("None".to_string(), |v| v.to_string()),
            node.parent.map_or("None".to_string(), |v| v.to_string()),
            node.depth.map_or("None".to_string(), |v| format!("{:.6}", v)),
            format!("{:.6}", node.length),
        ]);
    }
    rows
}

/// Merge new data with optional old data, colorizing cells that have changed.
fn build_diffed_rows(new_data: &[Vec<String>], old_data: Option<&[Vec<String>]>) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for (i, new_row) in new_data.iter().enumerate() {
        let row = if let Some(old) = old_data {
            if i < old.len() {
                new_row.iter()
                    .zip(&old[i])
                    .map(|(new_val, old_val)| format_diff(old_val, new_val))
                    .collect()
            } else {
                new_row.clone()
            }
        } else {
            new_row.clone()
        };
        rows.push(row);
    }
    rows
}

/// Compute maximum visible widths for each column.
fn compute_column_widths(rows: &[Vec<String>]) -> Vec<usize> {
    if rows.is_empty() {
        return vec![];
    }
    let col_count = rows[0].len();
    let mut widths = vec![0; col_count];
    for row in rows {
        for (col_idx, cell) in row.iter().enumerate() {
            let visible_len = strip_ansi_codes(cell).len();
            if visible_len > widths[col_idx] {
                widths[col_idx] = visible_len;
            }
        }
    }
    widths
}

/// Pads `text` so that its visible length is `width`, preserving ANSI codes.
fn pad_to_width(text: &str, width: usize) -> String {
    let visible_len = strip_ansi_codes(text).len();
    if visible_len >= width {
        text.to_string()
    } else {
        format!("{}{}", text, " ".repeat(width - visible_len))
    }
}

/// Prints the rows (2D vector of strings) aligned according to `column_widths`.
fn print_diffed_table(rows: &[Vec<String>], column_widths: &[usize]) {
    for row in rows {
        let mut line_parts = Vec::new();
        for (col_idx, cell) in row.iter().enumerate() {
            let padded = pad_to_width(cell, column_widths[col_idx]);
            line_parts.push(padded);
        }
        println!("| {} |", line_parts.join(" | "));
    }
}

/// Builds and returns a diffed table (as a 2D vector of strings) from the flat tree.
/// If `old_data` is provided, cells that changed will be shown in red with the old value in parentheses.
/// The header row is added as the first row.
/// If `print_now` is true, the table is printed.
pub fn diffed_flat_tree_table(
    flat_tree: &FlatTree,
    old_data: Option<&[Vec<String>]>,
    print_now: bool,
) -> Vec<Vec<String>> {
    // Build new data from flat_tree.
    let new_data = build_new_data(flat_tree);
    // Build diffed rows by comparing with old_data.
    let mut diffed = build_diffed_rows(&new_data, old_data);
    // Prepend the header row.
    let header = vec![
        "Index".to_string(),
        "Name".to_string(),
        "Left Child".to_string(),
        "Right Child".to_string(),
        "Parent".to_string(),
        "Depth".to_string(),
        "Length".to_string(),
    ];
    diffed.insert(0, header);
    // Compute column widths based on visible length.
    let widths = compute_column_widths(&diffed);
    if print_now {
        print_diffed_table(&diffed, &widths);
    }
    diffed
}
