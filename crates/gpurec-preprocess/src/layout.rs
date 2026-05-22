use crate::PreprocessError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
pub struct WaveLayoutRequest {
    pub waves: Vec<Vec<i64>>,
    pub phases: Vec<i64>,
    pub c: usize,
    pub n_splits: usize,
    pub split_leftrights_sorted: Vec<i64>,
    pub split_parents_sorted: Vec<i64>,
    pub leaf_row_index: Vec<i64>,
    pub leaf_col_index: Vec<i64>,
    pub root_clade_ids: Vec<i64>,
    #[serde(default)]
    pub family_clade_counts: Option<Vec<i64>>,
    #[serde(default)]
    pub family_clade_offsets: Option<Vec<i64>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WaveMetaPlan {
    pub start: i64,
    pub end: i64,
    #[serde(rename = "W")]
    pub w: i64,
    pub has_splits: bool,
    pub phase: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_indices: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sl: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sr: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reduce_idx: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_eq1: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eq1_reduce_idx: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ge2_ptr: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ge2_parent_ids: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ge2_max_fanout: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WaveLayoutPlan {
    pub perm: Vec<i64>,
    pub c: i64,
    pub leaf_row_index: Vec<i64>,
    pub leaf_species_index: Vec<i64>,
    pub root_clade_ids: Vec<i64>,
    pub root_clade_ids_cpu: Vec<i64>,
    pub wave_metas: Vec<WaveMetaPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_idx: Option<Vec<i64>>,
}

pub fn build_wave_layout_plan_request(
    request: &WaveLayoutRequest,
) -> Result<WaveLayoutPlan, PreprocessError> {
    build_wave_layout_plan(
        &request.waves,
        &request.phases,
        request.c,
        request.n_splits,
        &request.split_leftrights_sorted,
        &request.split_parents_sorted,
        &request.leaf_row_index,
        &request.leaf_col_index,
        &request.root_clade_ids,
        request.family_clade_counts.as_deref(),
        request.family_clade_offsets.as_deref(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_wave_layout_plan(
    waves: &[Vec<i64>],
    phases: &[i64],
    c: usize,
    n_splits: usize,
    split_leftrights_sorted: &[i64],
    split_parents_sorted: &[i64],
    leaf_row_index: &[i64],
    leaf_col_index: &[i64],
    root_clade_ids: &[i64],
    family_clade_counts: Option<&[i64]>,
    family_clade_offsets: Option<&[i64]>,
) -> Result<WaveLayoutPlan, PreprocessError> {
    let family_clade_ranges =
        validate_family_clade_metadata(family_clade_counts, family_clade_offsets, c)?;
    if phases.len() != waves.len() {
        return invalid(format!(
            "waves and phases must have matching lengths, got {} and {}",
            waves.len(),
            phases.len()
        ));
    }
    if split_leftrights_sorted.len() != 2 * n_splits {
        return invalid(format!(
            "split_leftrights_sorted has length {} but expected {}",
            split_leftrights_sorted.len(),
            2 * n_splits
        ));
    }
    if split_parents_sorted.len() != n_splits {
        return invalid(format!(
            "split_parents_sorted has length {} but expected {n_splits}",
            split_parents_sorted.len()
        ));
    }
    if leaf_col_index.len() != leaf_row_index.len() {
        return invalid(format!(
            "leaf_col_index has length {} but expected {}",
            leaf_col_index.len(),
            leaf_row_index.len()
        ));
    }
    validate_clade_id_values("split_leftrights_sorted", split_leftrights_sorted, c)?;
    validate_clade_id_values("split_parents_sorted", split_parents_sorted, c)?;
    validate_clade_id_values("leaf_row_index", leaf_row_index, c)?;
    validate_clade_id_values("root_clade_ids", root_clade_ids, c)?;

    let mut all_clades = Vec::with_capacity(c);
    let mut wave_starts = Vec::with_capacity(waves.len() + 1);
    wave_starts.push(0usize);
    for wave in waves {
        for clade in wave {
            all_clades.push(*clade);
        }
        wave_starts.push(all_clades.len());
    }
    validate_wave_clade_coverage(&all_clades, c)?;

    let inv_perm = all_clades;
    let mut perm = vec![0i64; c];
    for (new_idx, original) in inv_perm.iter().enumerate() {
        perm[*original as usize] = new_idx as i64;
    }

    let lefts_orig = &split_leftrights_sorted[..n_splits];
    let rights_orig = &split_leftrights_sorted[n_splits..];
    let lefts_new = lefts_orig
        .iter()
        .map(|clade| perm[*clade as usize])
        .collect::<Vec<_>>();
    let rights_new = rights_orig
        .iter()
        .map(|clade| perm[*clade as usize])
        .collect::<Vec<_>>();
    let sp_new = split_parents_sorted
        .iter()
        .map(|parent| perm[*parent as usize])
        .collect::<Vec<_>>();

    let leaf_row_new = leaf_row_index
        .iter()
        .map(|row| perm[*row as usize])
        .collect::<Vec<_>>();
    let root_ids_new = root_clade_ids
        .iter()
        .map(|root| perm[*root as usize])
        .collect::<Vec<_>>();
    let mut leaf_species_index = vec![-1i64; c];
    for (row, species) in leaf_row_new.iter().zip(leaf_col_index.iter()) {
        leaf_species_index[*row as usize] = *species;
    }

    let wave_ends = &wave_starts[1..];
    let mut split_order = (0..n_splits)
        .map(|idx| (split_wave_index(wave_ends, sp_new[idx] as usize), idx))
        .collect::<Vec<_>>();
    split_order.sort_by_key(|(wave_idx, split_idx)| (*wave_idx, *split_idx));

    let mut wave_metas = Vec::with_capacity(waves.len());
    let mut order_head = 0usize;
    for wi in 0..waves.len() {
        let ws = wave_starts[wi];
        let we = wave_starts[wi + 1];
        let w = we - ws;
        let split_start = order_head;
        while order_head < split_order.len() && split_order[order_head].0 == wi {
            order_head += 1;
        }
        let split_indices = split_order[split_start..order_head]
            .iter()
            .map(|(_, split_idx)| *split_idx)
            .collect::<Vec<_>>();

        let mut meta = WaveMetaPlan {
            start: ws as i64,
            end: we as i64,
            w: w as i64,
            has_splits: !split_indices.is_empty(),
            phase: phases[wi],
            split_indices: None,
            sl: None,
            sr: None,
            reduce_idx: None,
            n_eq1: None,
            eq1_reduce_idx: None,
            ge2_ptr: None,
            ge2_parent_ids: None,
            ge2_max_fanout: None,
        };

        if !split_indices.is_empty() {
            populate_split_meta(
                &mut meta,
                &split_indices,
                ws,
                w,
                &sp_new,
                &lefts_new,
                &rights_new,
            );
        }
        wave_metas.push(meta);
    }

    let family_idx = family_clade_ranges.map(|ranges| {
        let mut family_idx_orig = vec![-1i64; c];
        for (family, (offset, count)) in ranges.iter().enumerate() {
            for clade in *offset..(*offset + *count) {
                family_idx_orig[clade] = family as i64;
            }
        }
        inv_perm
            .iter()
            .map(|clade| family_idx_orig[*clade as usize])
            .collect::<Vec<_>>()
    });

    Ok(WaveLayoutPlan {
        perm,
        c: c as i64,
        leaf_row_index: leaf_row_new,
        leaf_species_index,
        root_clade_ids: root_ids_new.clone(),
        root_clade_ids_cpu: root_ids_new,
        wave_metas,
        family_idx,
    })
}

fn invalid<T>(message: impl Into<String>) -> Result<T, PreprocessError> {
    Err(PreprocessError::InvalidInput(message.into()))
}

fn validate_wave_clade_coverage(all_clades: &[i64], c: usize) -> Result<(), PreprocessError> {
    if all_clades.len() != c {
        return invalid(format!(
            "Wave layout covers {} clades but C={c}",
            all_clades.len()
        ));
    }
    let mut seen = vec![false; c];
    for (position, clade) in all_clades.iter().enumerate() {
        let clade_id = clade_id("Wave layout", *clade, position, c)?;
        if seen[clade_id] {
            return invalid(format!("Wave layout contains duplicate clade {clade_id}"));
        }
        seen[clade_id] = true;
    }
    Ok(())
}

fn clade_id(name: &str, value: i64, position: usize, c: usize) -> Result<usize, PreprocessError> {
    if value < 0 || value as usize >= c {
        return invalid(format!(
            "{name} contains clade {value} at position {position}, outside valid range [0, {c})"
        ));
    }
    Ok(value as usize)
}

fn validate_clade_id_values(name: &str, values: &[i64], c: usize) -> Result<(), PreprocessError> {
    for (position, value) in values.iter().enumerate() {
        clade_id(name, *value, position, c)?;
    }
    Ok(())
}

fn validate_family_clade_metadata(
    family_clade_counts: Option<&[i64]>,
    family_clade_offsets: Option<&[i64]>,
    c: usize,
) -> Result<Option<Vec<(usize, usize)>>, PreprocessError> {
    match (family_clade_counts, family_clade_offsets) {
        (None, None) => Ok(None),
        (None, Some(_)) | (Some(_), None) => {
            invalid("family_clade_counts and family_clade_offsets must be provided together")
        }
        (Some(counts), Some(offsets)) => {
            if counts.len() != offsets.len() {
                return invalid(
                    "family_clade_counts and family_clade_offsets must have matching lengths",
                );
            }
            let mut covered = vec![false; c];
            let mut ranges = Vec::with_capacity(counts.len());
            for family_index in 0..counts.len() {
                let offset = offsets[family_index];
                let count = counts[family_index];
                if offset < 0 {
                    return invalid(format!(
                        "family_clade_offsets[{family_index}] must be non-negative"
                    ));
                }
                if count < 0 {
                    return invalid(format!(
                        "family_clade_counts[{family_index}] must be non-negative"
                    ));
                }
                let offset = offset as usize;
                let count = count as usize;
                let end = offset + count;
                if end > c {
                    return invalid(format!(
                        "family {family_index} clade range [{offset}, {end}) is outside C={c}"
                    ));
                }
                for clade in offset..end {
                    if covered[clade] {
                        return invalid(format!("family clade metadata overlaps clade {clade}"));
                    }
                    covered[clade] = true;
                }
                ranges.push((offset, count));
            }
            for (clade, is_covered) in covered.iter().enumerate() {
                if !is_covered {
                    return invalid(format!(
                        "family clade metadata does not cover clade {clade}"
                    ));
                }
            }
            Ok(Some(ranges))
        }
    }
}

fn split_wave_index(wave_ends: &[usize], parent_new: usize) -> usize {
    wave_ends.partition_point(|end| *end <= parent_new)
}

fn populate_split_meta(
    meta: &mut WaveMetaPlan,
    split_indices: &[usize],
    wave_start: usize,
    wave_width: usize,
    sp_new: &[i64],
    lefts_new: &[i64],
    rights_new: &[i64],
) {
    let mut clade_split_counts = vec![0usize; wave_width];
    let mut reduce_idx = split_indices
        .iter()
        .map(|split_idx| (sp_new[*split_idx] as usize) - wave_start)
        .collect::<Vec<_>>();
    for reduce in &reduce_idx {
        clade_split_counts[*reduce] += 1;
    }

    let mut ordered_positions = (0..split_indices.len()).collect::<Vec<_>>();
    ordered_positions.sort_by_key(|position| {
        let reduce = reduce_idx[*position];
        let is_ge2 = usize::from(clade_split_counts[reduce] > 1);
        (is_ge2 * (wave_width + 1) + reduce, *position)
    });

    let ordered_split_indices = ordered_positions
        .iter()
        .map(|position| split_indices[*position] as i64)
        .collect::<Vec<_>>();
    reduce_idx = ordered_positions
        .iter()
        .map(|position| reduce_idx[*position])
        .collect();

    let n_eq1 = reduce_idx
        .iter()
        .filter(|reduce| clade_split_counts[**reduce] == 1)
        .count();
    let ge2_parent_count = clade_split_counts
        .iter()
        .filter(|count| **count >= 2)
        .count();

    meta.split_indices = Some(ordered_split_indices.clone());
    meta.sl = Some(
        ordered_split_indices
            .iter()
            .map(|split_idx| lefts_new[*split_idx as usize])
            .collect(),
    );
    meta.sr = Some(
        ordered_split_indices
            .iter()
            .map(|split_idx| rights_new[*split_idx as usize])
            .collect(),
    );
    meta.reduce_idx = Some(reduce_idx.iter().map(|reduce| *reduce as i64).collect());
    meta.n_eq1 = Some(n_eq1 as i64);
    if n_eq1 > 0 {
        meta.eq1_reduce_idx = Some(
            reduce_idx[..n_eq1]
                .iter()
                .map(|reduce| *reduce as i64)
                .collect(),
        );
    }
    if ge2_parent_count > 0 {
        let ge2_reduce = &reduce_idx[n_eq1..];
        let mut ge2_parent_ids = Vec::new();
        let mut ge2_counts = Vec::new();
        for reduce in ge2_reduce {
            if ge2_parent_ids.last() == Some(reduce) {
                let last = ge2_counts.len() - 1;
                ge2_counts[last] += 1i64;
            } else {
                ge2_parent_ids.push(*reduce);
                ge2_counts.push(1i64);
            }
        }
        let mut ge2_ptr = Vec::with_capacity(ge2_counts.len() + 1);
        ge2_ptr.push(0i64);
        for count in &ge2_counts {
            let next = ge2_ptr.last().copied().unwrap_or(0) + *count;
            ge2_ptr.push(next);
        }
        meta.ge2_ptr = Some(ge2_ptr);
        meta.ge2_parent_ids = Some(
            ge2_parent_ids
                .into_iter()
                .map(|reduce| reduce as i64)
                .collect(),
        );
        meta.ge2_max_fanout = ge2_counts.into_iter().max();
    }
}
