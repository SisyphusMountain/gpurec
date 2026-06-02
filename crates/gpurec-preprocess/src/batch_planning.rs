use crate::PreprocessError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
pub struct BatchPlanRequest {
    pub clade_counts: Vec<i64>,
    pub family_chunk_size: i64,
    #[serde(default)]
    pub clade_budget: Option<i64>,
    #[serde(default = "default_batch_packing")]
    pub batch_packing: String,
    #[serde(default)]
    pub indices: Option<Vec<i64>>,
    #[serde(default)]
    pub total: Option<i64>,
    #[serde(default)]
    pub split_counts: Option<Vec<i64>>,
    #[serde(default)]
    pub leaf_counts: Option<Vec<i64>>,
    #[serde(default)]
    pub small_family_max_leaves: Option<i64>,
    #[serde(default)]
    pub nonleaf_counts: Option<Vec<i64>>,
    #[serde(default)]
    pub schedule_depths: Option<Vec<i64>>,
    #[serde(default)]
    pub max_wave_size: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FamilyBatchPlanOutput {
    pub indices: Vec<i64>,
    pub clades: i64,
    pub splits: i64,
}

fn default_batch_packing() -> String {
    "sequential".to_string()
}

pub fn plan_family_batches_request(
    request: &BatchPlanRequest,
) -> Result<Vec<FamilyBatchPlanOutput>, PreprocessError> {
    plan_family_batches_impl(
        &request.clade_counts,
        request.family_chunk_size,
        request.clade_budget,
        &request.batch_packing,
        request.indices.as_deref(),
        request.total,
        request.split_counts.as_deref(),
        request.leaf_counts.as_deref(),
        request.nonleaf_counts.as_deref(),
        request.schedule_depths.as_deref(),
        request.max_wave_size,
        request.small_family_max_leaves,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn plan_family_batches(
    clade_counts: &[i64],
    family_chunk_size: i64,
    clade_budget: Option<i64>,
    batch_packing: &str,
    indices: Option<&[i64]>,
    total: Option<i64>,
    split_counts: Option<&[i64]>,
    leaf_counts: Option<&[i64]>,
    nonleaf_counts: Option<&[i64]>,
    schedule_depths: Option<&[i64]>,
    max_wave_size: Option<i64>,
) -> Result<Vec<FamilyBatchPlanOutput>, PreprocessError> {
    plan_family_batches_impl(
        clade_counts,
        family_chunk_size,
        clade_budget,
        batch_packing,
        indices,
        total,
        split_counts,
        leaf_counts,
        nonleaf_counts,
        schedule_depths,
        max_wave_size,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn plan_family_batches_impl(
    clade_counts: &[i64],
    family_chunk_size: i64,
    clade_budget: Option<i64>,
    batch_packing: &str,
    indices: Option<&[i64]>,
    total: Option<i64>,
    split_counts: Option<&[i64]>,
    leaf_counts: Option<&[i64]>,
    nonleaf_counts: Option<&[i64]>,
    schedule_depths: Option<&[i64]>,
    max_wave_size: Option<i64>,
    small_family_max_leaves: Option<i64>,
) -> Result<Vec<FamilyBatchPlanOutput>, PreprocessError> {
    let selected = selected_indices(indices, total, clade_counts.len())?;
    validate_selected_indices(&selected, clade_counts.len())?;
    if let Some(counts) = split_counts {
        require_indexed_stats("split_counts", "split_counts", Some(counts), &selected)?;
    }
    if family_chunk_size < 0 {
        return invalid("family_chunk_size must be non-negative");
    }
    let packing = normalize_batch_packing(batch_packing)?;
    if let Some(budget) = clade_budget {
        if budget <= 0 {
            return invalid("clade_budget must be positive when provided");
        }
    }
    if let Some(max_leaves) = small_family_max_leaves {
        if max_leaves < 0 {
            return invalid("small_family_max_leaves must be non-negative when provided");
        }
    }

    if packing == "clade_first_fit" {
        clade_budget.ok_or_else(|| {
            PreprocessError::InvalidInput(
                "batch_packing='clade_first_fit' requires clade_budget".to_string(),
            )
        })?;
    }
    if packing == "depth_first_fit" {
        clade_budget.ok_or_else(|| {
            PreprocessError::InvalidInput(
                "batch_packing='depth_first_fit' requires clade_budget".to_string(),
            )
        })?;
        require_indexed_stats(
            "batch_packing='depth_first_fit'",
            "leaf_counts",
            leaf_counts,
            &selected,
        )?;
        require_indexed_stats(
            "batch_packing='depth_first_fit'",
            "nonleaf_counts",
            nonleaf_counts,
            &selected,
        )?;
        require_indexed_stats(
            "batch_packing='depth_first_fit'",
            "schedule_depths",
            schedule_depths,
            &selected,
        )?;
    }

    let resolved_max_wave_size = if packing == "depth_first_fit" {
        let wave_cap = match max_wave_size {
            Some(value) => value,
            None => selected.iter().map(|idx| clade_counts[*idx]).sum(),
        };
        if wave_cap <= 0 {
            return invalid("max_wave_size must be positive");
        }
        Some(wave_cap)
    } else {
        max_wave_size
    };

    let small_family_leaf_counts = match small_family_max_leaves {
        Some(_) => Some(require_indexed_stats(
            "small_family_max_leaves",
            "leaf_counts",
            leaf_counts,
            &selected,
        )?),
        None => None,
    };
    let selected_groups =
        selected_groups_by_leaf_count(&selected, small_family_leaf_counts, small_family_max_leaves);

    let mut plans = Vec::new();
    for group in selected_groups {
        plans.extend(plan_family_batches_for_selected(
            &group,
            clade_counts,
            split_counts,
            family_chunk_size,
            clade_budget,
            packing,
            leaf_counts,
            nonleaf_counts,
            schedule_depths,
            resolved_max_wave_size,
        )?);
    }
    Ok(plans)
}

#[allow(clippy::too_many_arguments)]
fn plan_family_batches_for_selected(
    selected: &[usize],
    clade_counts: &[i64],
    split_counts: Option<&[i64]>,
    family_chunk_size: i64,
    clade_budget: Option<i64>,
    packing: &str,
    leaf_counts: Option<&[i64]>,
    nonleaf_counts: Option<&[i64]>,
    schedule_depths: Option<&[i64]>,
    max_wave_size: Option<i64>,
) -> Result<Vec<FamilyBatchPlanOutput>, PreprocessError> {
    match packing {
        "clade_first_fit" => {
            let budget = clade_budget.ok_or_else(|| {
                PreprocessError::InvalidInput(
                    "batch_packing='clade_first_fit' requires clade_budget".to_string(),
                )
            })?;
            plan_clade_first_fit(
                selected,
                clade_counts,
                split_counts,
                family_chunk_size,
                budget,
            )
        }
        "depth_first_fit" => {
            let budget = clade_budget.ok_or_else(|| {
                PreprocessError::InvalidInput(
                    "batch_packing='depth_first_fit' requires clade_budget".to_string(),
                )
            })?;
            let leaves = require_indexed_stats(
                "batch_packing='depth_first_fit'",
                "leaf_counts",
                leaf_counts,
                selected,
            )?;
            let nonleaves = require_indexed_stats(
                "batch_packing='depth_first_fit'",
                "nonleaf_counts",
                nonleaf_counts,
                selected,
            )?;
            let depths = require_indexed_stats(
                "batch_packing='depth_first_fit'",
                "schedule_depths",
                schedule_depths,
                selected,
            )?;
            let wave_cap = max_wave_size.ok_or_else(|| {
                PreprocessError::InvalidInput(
                    "batch_packing='depth_first_fit' requires max_wave_size".to_string(),
                )
            })?;
            plan_depth_first_fit(
                selected,
                clade_counts,
                split_counts,
                leaves,
                nonleaves,
                depths,
                family_chunk_size,
                budget,
                wave_cap,
            )
        }
        "sequential" => plan_sequential(
            selected,
            clade_counts,
            split_counts,
            family_chunk_size,
            clade_budget,
        ),
        _ => invalid(format!("unknown batch_packing {packing:?}")),
    }
}

fn selected_groups_by_leaf_count(
    selected: &[usize],
    leaf_counts: Option<&[i64]>,
    small_family_max_leaves: Option<i64>,
) -> Vec<Vec<usize>> {
    let Some(max_leaves) = small_family_max_leaves else {
        return vec![selected.to_vec()];
    };
    if max_leaves == 0 {
        return vec![selected.to_vec()];
    }
    let Some(leaves) = leaf_counts else {
        return vec![selected.to_vec()];
    };
    let mut boundaries = vec![max_leaves];
    let mut next = max_leaves;
    while next < 256 {
        next *= 2;
        boundaries.push(next);
    }
    boundaries.push(i64::MAX);
    let mut groups = vec![Vec::new(); boundaries.len()];
    for idx in selected {
        let leaf_count = leaves[*idx];
        let bucket = boundaries
            .iter()
            .position(|boundary| leaf_count <= *boundary)
            .unwrap_or(boundaries.len() - 1);
        groups[bucket].push(*idx);
    }
    groups
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect()
}

fn invalid<T>(message: impl Into<String>) -> Result<T, PreprocessError> {
    Err(PreprocessError::InvalidInput(message.into()))
}

fn normalize_batch_packing(value: &str) -> Result<&'static str, PreprocessError> {
    let text = value.trim().to_ascii_lowercase().replace('-', "_");
    match text.as_str() {
        "" | "sequential" | "contiguous" | "input_order" => Ok("sequential"),
        "clade_first_fit" | "first_fit_decreasing" | "ffd" | "clade_ffd" => Ok("clade_first_fit"),
        "depth_first_fit"
        | "depth_ffd"
        | "critical_path_first_fit"
        | "critical_first_fit"
        | "wave_first_fit" => Ok("depth_first_fit"),
        _ => invalid(format!(
            "batch_packing must be 'sequential', 'clade_first_fit', or \
             'depth_first_fit', got {value:?}"
        )),
    }
}

fn selected_indices(
    indices: Option<&[i64]>,
    total: Option<i64>,
    clade_count_len: usize,
) -> Result<Vec<usize>, PreprocessError> {
    if let Some(indices) = indices {
        return indices
            .iter()
            .map(|idx| {
                if *idx < 0 {
                    invalid(format!("family index {idx} is outside valid range"))
                } else {
                    Ok(*idx as usize)
                }
            })
            .collect();
    }
    let total = total.unwrap_or(clade_count_len as i64);
    if total < 0 {
        return invalid("total must be non-negative");
    }
    Ok((0..total as usize).collect())
}

fn validate_selected_indices(
    selected: &[usize],
    clade_count_len: usize,
) -> Result<(), PreprocessError> {
    let mut seen = std::collections::HashSet::with_capacity(selected.len());
    for (position, idx) in selected.iter().enumerate() {
        if *idx >= clade_count_len {
            return invalid(format!(
                "family index {idx} at selected position {position} is outside valid range [0, {clade_count_len})"
            ));
        }
        if !seen.insert(*idx) {
            return invalid(format!(
                "duplicate family index {idx} at selected position {position}"
            ));
        }
    }
    Ok(())
}

fn require_indexed_stats<'a>(
    context: &str,
    name: &str,
    values: Option<&'a [i64]>,
    selected: &[usize],
) -> Result<&'a [i64], PreprocessError> {
    let values = values
        .ok_or_else(|| PreprocessError::InvalidInput(format!("{context} requires {name}")))?;
    if let Some(required) = selected.iter().max().map(|idx| idx + 1) {
        if values.len() < required {
            return invalid(format!("{name} must cover selected family indices"));
        }
    }
    Ok(values)
}

fn plan_from_chunks(
    chunks: Vec<Vec<usize>>,
    clade_counts: &[i64],
    split_counts: Option<&[i64]>,
) -> Vec<FamilyBatchPlanOutput> {
    chunks
        .into_iter()
        .map(|chunk| {
            let clades = chunk.iter().map(|idx| clade_counts[*idx]).sum();
            let splits = split_counts
                .map(|counts| chunk.iter().map(|idx| counts[*idx]).sum())
                .unwrap_or(0);
            FamilyBatchPlanOutput {
                indices: chunk.into_iter().map(|idx| idx as i64).collect(),
                clades,
                splits,
            }
        })
        .collect()
}

fn plan_clade_first_fit(
    selected: &[usize],
    clade_counts: &[i64],
    split_counts: Option<&[i64]>,
    family_limit: i64,
    budget: i64,
) -> Result<Vec<FamilyBatchPlanOutput>, PreprocessError> {
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut chunk_clades: Vec<i64> = Vec::new();
    let mut order = selected.to_vec();
    order.sort_by(|left, right| clade_counts[*right].cmp(&clade_counts[*left]));

    for idx in order {
        let n_clades = clade_counts[idx];
        let mut best_j = None;
        let mut best_remaining = None;
        for (j, current_clades) in chunk_clades.iter().enumerate() {
            if family_limit > 0 && chunks[j].len() >= family_limit as usize {
                continue;
            }
            let remaining = budget - current_clades - n_clades;
            if remaining < 0 {
                continue;
            }
            if best_remaining.map(|best| remaining < best).unwrap_or(true) {
                best_j = Some(j);
                best_remaining = Some(remaining);
            }
        }
        if let Some(j) = best_j {
            chunks[j].push(idx);
            chunk_clades[j] += n_clades;
        } else {
            chunks.push(vec![idx]);
            chunk_clades.push(n_clades);
        }
    }
    Ok(plan_from_chunks(chunks, clade_counts, split_counts))
}

#[allow(clippy::too_many_arguments)]
fn plan_depth_first_fit(
    selected: &[usize],
    clade_counts: &[i64],
    split_counts: Option<&[i64]>,
    leaf_counts: &[i64],
    nonleaf_counts: &[i64],
    schedule_depths: &[i64],
    family_limit: i64,
    budget: i64,
    wave_cap: i64,
) -> Result<Vec<FamilyBatchPlanOutput>, PreprocessError> {
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut chunk_clades: Vec<i64> = Vec::new();
    let mut chunk_leaves: Vec<i64> = Vec::new();
    let mut chunk_nonleaves: Vec<i64> = Vec::new();
    let mut chunk_depths: Vec<i64> = Vec::new();
    let mut order = selected.to_vec();
    order.sort_by(|left, right| {
        (schedule_depths[*right], clade_counts[*right])
            .cmp(&(schedule_depths[*left], clade_counts[*left]))
    });

    for idx in order {
        let n_clades = clade_counts[idx];
        let n_leaves = leaf_counts[idx];
        let n_nonleaves = nonleaf_counts[idx];
        let depth = schedule_depths[idx];
        let mut best_j = None;
        let mut best_key: Option<(i64, i64, i64)> = None;
        for (j, current_clades) in chunk_clades.iter().enumerate() {
            if family_limit > 0 && chunks[j].len() >= family_limit as usize {
                continue;
            }
            let new_clades = current_clades + n_clades;
            if new_clades > budget {
                continue;
            }
            let before = lower_bound(
                chunk_leaves[j],
                chunk_nonleaves[j],
                chunk_depths[j],
                wave_cap,
            );
            let after = lower_bound(
                chunk_leaves[j] + n_leaves,
                chunk_nonleaves[j] + n_nonleaves,
                chunk_depths[j].max(depth),
                wave_cap,
            );
            let remaining = budget - new_clades;
            let key = (after - before, after, remaining);
            if best_key.map(|best| key < best).unwrap_or(true) {
                best_j = Some(j);
                best_key = Some(key);
            }
        }
        if let Some(j) = best_j {
            chunks[j].push(idx);
            chunk_clades[j] += n_clades;
            chunk_leaves[j] += n_leaves;
            chunk_nonleaves[j] += n_nonleaves;
            chunk_depths[j] = chunk_depths[j].max(depth);
        } else {
            chunks.push(vec![idx]);
            chunk_clades.push(n_clades);
            chunk_leaves.push(n_leaves);
            chunk_nonleaves.push(n_nonleaves);
            chunk_depths.push(depth);
        }
    }
    Ok(plan_from_chunks(chunks, clade_counts, split_counts))
}

fn lower_bound(leaves_count: i64, nonleaves_count: i64, depth: i64, wave_cap: i64) -> i64 {
    let leaf_waves = ceil_div_nonnegative(leaves_count, wave_cap);
    let work_waves = ceil_div_nonnegative(nonleaves_count, wave_cap);
    leaf_waves + depth.max(work_waves)
}

fn ceil_div_nonnegative(value: i64, divisor: i64) -> i64 {
    if value <= 0 {
        0
    } else {
        (value + divisor - 1) / divisor
    }
}

fn plan_sequential(
    selected: &[usize],
    clade_counts: &[i64],
    split_counts: Option<&[i64]>,
    family_limit: i64,
    budget: Option<i64>,
) -> Result<Vec<FamilyBatchPlanOutput>, PreprocessError> {
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_clades = 0i64;

    for idx in selected {
        let n_clades = clade_counts[*idx];
        let family_cap_hit = family_limit > 0 && current.len() >= family_limit as usize;
        let clade_cap_hit = budget
            .map(|budget| !current.is_empty() && current_clades + n_clades > budget)
            .unwrap_or(false);
        if family_cap_hit || clade_cap_hit {
            chunks.push(std::mem::take(&mut current));
            current_clades = 0;
        }
        current.push(*idx);
        current_clades += n_clades;
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    Ok(plan_from_chunks(chunks, clade_counts, split_counts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_indices(plans: &[FamilyBatchPlanOutput]) -> Vec<Vec<i64>> {
        plans.iter().map(|plan| plan.indices.clone()).collect()
    }

    #[test]
    fn clade_first_fit_matches_python_fixture() {
        let plans = plan_family_batches(
            &[8, 7, 6, 5, 4],
            0,
            Some(12),
            "clade_first_fit",
            None,
            Some(5),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(plan_indices(&plans), vec![vec![0, 4], vec![1, 3], vec![2]]);
    }

    #[test]
    fn depth_first_fit_matches_python_fixture() {
        let plans = plan_family_batches(
            &[6, 6, 6, 6, 6],
            0,
            Some(12),
            "depth_first_fit",
            None,
            Some(5),
            Some(&[10, 20, 30, 40, 50]),
            Some(&[1, 1, 1, 1, 1]),
            Some(&[5, 5, 5, 5, 5]),
            Some(&[10, 9, 2, 1, 1]),
            Some(8),
        )
        .unwrap();
        assert_eq!(plan_indices(&plans), vec![vec![0, 1], vec![2, 3], vec![4]]);
        assert_eq!(
            plans.iter().map(|plan| plan.splits).collect::<Vec<_>>(),
            vec![30, 70, 50]
        );
    }

    #[test]
    fn request_prioritizes_small_families_under_leaf_threshold() {
        let request = BatchPlanRequest {
            clade_counts: vec![5, 3, 4, 4, 6, 2],
            family_chunk_size: 2,
            clade_budget: Some(6),
            batch_packing: "clade_first_fit".to_string(),
            indices: None,
            total: Some(6),
            split_counts: None,
            leaf_counts: Some(vec![9, 4, 7, 2, 6, 4]),
            small_family_max_leaves: Some(4),
            nonleaf_counts: None,
            schedule_depths: None,
            max_wave_size: None,
        };

        let plans = plan_family_batches_request(&request).unwrap();

        assert_eq!(
            plan_indices(&plans),
            vec![vec![3, 5], vec![1], vec![4], vec![2], vec![0]]
        );
        assert!(plans.iter().all(|plan| plan.clades <= 6));
        assert!(plans.iter().all(|plan| plan.indices.len() <= 2));
    }

    #[test]
    fn request_buckets_families_by_leaf_count_bands() {
        let request = BatchPlanRequest {
            clade_counts: vec![2, 2, 2, 2, 2, 2],
            family_chunk_size: 0,
            clade_budget: Some(12),
            batch_packing: "clade_first_fit".to_string(),
            indices: None,
            total: Some(6),
            split_counts: None,
            leaf_counts: Some(vec![4, 5, 8, 9, 16, 33]),
            small_family_max_leaves: Some(4),
            nonleaf_counts: None,
            schedule_depths: None,
            max_wave_size: None,
        };

        let plans = plan_family_batches_request(&request).unwrap();

        assert_eq!(
            plan_indices(&plans),
            vec![vec![0], vec![1, 2], vec![3, 4], vec![5]]
        );
    }
}
