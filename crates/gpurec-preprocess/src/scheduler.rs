use crate::PreprocessError;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Debug, Deserialize)]
pub struct ScheduleRequest {
    pub items: Vec<ScheduleItem>,
    pub family_clade_offsets: Vec<i64>,
    pub max_wave_size: Option<usize>,
    #[serde(default)]
    pub max_root_wave_size: Option<usize>,
    #[serde(default)]
    pub max_dts_partial_rows: Option<usize>,
    #[serde(default = "default_dts_partial_tile_splits")]
    pub dts_partial_tile_splits: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ScheduleItem {
    pub ccp: ScheduleCcp,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ScheduleCcp {
    #[serde(rename = "C", alias = "c")]
    pub c: usize,
    #[serde(rename = "N_splits", alias = "n_splits")]
    pub n_splits: usize,
    #[serde(default)]
    pub split_counts: Option<Vec<i64>>,
    pub split_parents_sorted: Vec<i64>,
    pub split_leftrights_sorted: Vec<i64>,
    pub root_clade_id: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ScheduleOutput {
    pub waves: Vec<Vec<i64>>,
    pub phases: Vec<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FamilyScheduleSummary {
    pub clade_count: i64,
    pub leaf_count: i64,
    pub nonleaf_count: i64,
    pub max_level: i64,
}

#[derive(Clone, Debug)]
struct FamilySchedule {
    c: usize,
    split_counts: Vec<usize>,
    children: Vec<Vec<usize>>,
    parents_of: Vec<Vec<usize>>,
    remaining: Vec<usize>,
    bfs_level: Vec<usize>,
    priority: Vec<usize>,
    nonleaf_count: usize,
    max_level: usize,
    root_id: usize,
}

type LocalClade = (usize, usize);

fn default_dts_partial_tile_splits() -> usize {
    64
}

pub fn schedule_global_phased_waves_request(
    request: &ScheduleRequest,
) -> Result<ScheduleOutput, PreprocessError> {
    schedule_global_phased_waves(
        &request.items,
        &request.family_clade_offsets,
        request.max_wave_size,
        request.max_root_wave_size,
        request.max_dts_partial_rows,
        request.dts_partial_tile_splits,
    )
}

pub fn family_schedule_summary(
    ccp: &ScheduleCcp,
) -> Result<FamilyScheduleSummary, PreprocessError> {
    let data = family_schedule_data(ccp)?;
    Ok(FamilyScheduleSummary {
        clade_count: data.c as i64,
        leaf_count: (data.c - data.nonleaf_count) as i64,
        nonleaf_count: data.nonleaf_count as i64,
        max_level: data.max_level as i64,
    })
}

pub fn schedule_global_phased_waves(
    items: &[ScheduleItem],
    family_clade_offsets: &[i64],
    max_wave_size: Option<usize>,
    max_root_wave_size: Option<usize>,
    max_dts_partial_rows: Option<usize>,
    dts_partial_tile_splits: usize,
) -> Result<ScheduleOutput, PreprocessError> {
    if items.len() != family_clade_offsets.len() {
        return invalid("items and family_clade_offsets must have matching lengths");
    }
    let total_clades = items.iter().map(|item| item.ccp.c).sum::<usize>();
    if total_clades == 0 {
        return Ok(ScheduleOutput {
            waves: Vec::new(),
            phases: Vec::new(),
        });
    }
    let wave_cap = max_wave_size.unwrap_or(total_clades);
    if wave_cap == 0 {
        return invalid("max_wave_size must be positive");
    }
    if max_root_wave_size == Some(0) {
        return invalid("max_root_wave_size must be positive");
    }
    if max_dts_partial_rows == Some(0) {
        return invalid("max_dts_partial_rows must be positive when provided");
    }
    if dts_partial_tile_splits == 0 {
        return invalid("dts_partial_tile_splits must be positive");
    }

    let families = items
        .iter()
        .map(|item| family_schedule_data(&item.ccp))
        .collect::<Result<Vec<_>, _>>()?;

    let mut waves = leaf_phase_waves(&families, family_clade_offsets, wave_cap)?;
    let mut phases = vec![1; waves.len()];

    let (_policy, batches) = select_nonleaf_schedule_candidate(
        &families,
        wave_cap,
        max_root_wave_size,
        max_dts_partial_rows,
        dts_partial_tile_splits,
    )?;
    let (nonleaf_waves, nonleaf_phases) = materialize_nonleaf_waves(
        &batches,
        &families,
        family_clade_offsets,
        max_root_wave_size,
    )?;
    waves.extend(nonleaf_waves);
    phases.extend(nonleaf_phases);

    validate_scheduled_clades(&waves, total_clades)?;
    Ok(ScheduleOutput { waves, phases })
}

fn invalid<T>(message: impl Into<String>) -> Result<T, PreprocessError> {
    Err(PreprocessError::InvalidInput(message.into()))
}

fn family_schedule_data(ccp: &ScheduleCcp) -> Result<FamilySchedule, PreprocessError> {
    let c = ccp.c;
    let n = ccp.n_splits;
    let root_id = as_clade_id("root_clade_id", ccp.root_clade_id, c)?;
    if ccp.split_parents_sorted.len() != n {
        return invalid(format!(
            "split_parents_sorted has length {} but N_splits={n}",
            ccp.split_parents_sorted.len()
        ));
    }
    if ccp.split_leftrights_sorted.len() != 2 * n {
        return invalid(format!(
            "split_leftrights_sorted has length {} but 2*N_splits={}",
            ccp.split_leftrights_sorted.len(),
            2 * n
        ));
    }

    let lefts = &ccp.split_leftrights_sorted[..n];
    let rights = &ccp.split_leftrights_sorted[n..];
    let mut derived_counts = vec![0usize; c];
    let mut children = vec![Vec::new(); c];
    let mut parents_of = vec![Vec::new(); c];
    let mut remaining = vec![0usize; c];
    let mut child_sets: Vec<HashSet<usize>> = (0..c).map(|_| HashSet::new()).collect();

    for row in 0..n {
        let parent = as_parent_id(ccp.split_parents_sorted[row], row, c)?;
        derived_counts[parent] += 1;
        for (name, position, child_value) in [
            ("split_leftrights_sorted", row, lefts[row]),
            ("split_leftrights_sorted", n + row, rights[row]),
        ] {
            let child = as_child_id(name, child_value, position, c)?;
            if child_sets[parent].insert(child) {
                children[parent].push(child);
                parents_of[child].push(parent);
                remaining[parent] += 1;
            }
        }
    }

    let split_counts = if let Some(counts) = &ccp.split_counts {
        if counts.len() != c {
            return invalid(format!(
                "split_counts has length {} but C={c}",
                counts.len()
            ));
        }
        let mut parsed = Vec::with_capacity(c);
        for value in counts {
            if *value < 0 {
                return invalid("split_counts must be non-negative");
            }
            parsed.push(*value as usize);
        }
        if parsed != derived_counts {
            return invalid("split_counts does not match split_parents_sorted");
        }
        parsed
    } else {
        derived_counts
    };

    let (bfs_level, max_level) = bottom_up_levels(c, &parents_of, &remaining);
    let priority = root_distance_priority(c, &children, &bfs_level, max_level);
    let nonleaf_count = split_counts.iter().filter(|count| **count != 0).count();
    Ok(FamilySchedule {
        c,
        split_counts,
        children,
        parents_of,
        remaining,
        bfs_level,
        priority,
        nonleaf_count,
        max_level,
        root_id,
    })
}

fn bottom_up_levels(
    c: usize,
    parents_of: &[Vec<usize>],
    remaining: &[usize],
) -> (Vec<usize>, usize) {
    let mut bfs_level = vec![0usize; c];
    let mut remaining_bfs = remaining.to_vec();
    let mut queue = (0..c)
        .filter(|idx| remaining_bfs[*idx] == 0)
        .collect::<Vec<_>>();
    let mut head = 0usize;
    let mut max_level = 0usize;
    while head < queue.len() {
        let clade = queue[head];
        head += 1;
        for parent in &parents_of[clade] {
            if bfs_level[*parent] <= bfs_level[clade] {
                bfs_level[*parent] = bfs_level[clade] + 1;
                max_level = max_level.max(bfs_level[*parent]);
            }
            remaining_bfs[*parent] -= 1;
            if remaining_bfs[*parent] == 0 {
                queue.push(*parent);
            }
        }
    }
    (bfs_level, max_level)
}

fn root_distance_priority(
    c: usize,
    children: &[Vec<usize>],
    bfs_level: &[usize],
    max_level: usize,
) -> Vec<usize> {
    let mut levels = vec![Vec::new(); max_level + 1];
    for clade in 0..c {
        levels[bfs_level[clade]].push(clade);
    }
    let mut priority = vec![0usize; c];
    for level in (0..=max_level).rev() {
        for clade in &levels[level] {
            for child in &children[*clade] {
                priority[*child] = priority[*child].max(priority[*clade] + 1);
            }
        }
    }
    priority
}

fn as_clade_id(name: &str, value: i64, c: usize) -> Result<usize, PreprocessError> {
    if value < 0 || value as usize >= c {
        return invalid(format!("{name} {value} outside valid range [0, {c})"));
    }
    Ok(value as usize)
}

fn as_parent_id(value: i64, row: usize, c: usize) -> Result<usize, PreprocessError> {
    if value < 0 || value as usize >= c {
        return invalid(format!(
            "split_parents_sorted contains parent {value} at row {row}, outside valid range [0, {c})"
        ));
    }
    Ok(value as usize)
}

fn as_child_id(
    name: &str,
    value: i64,
    position: usize,
    c: usize,
) -> Result<usize, PreprocessError> {
    if value < 0 || value as usize >= c {
        return invalid(format!(
            "{name} contains child {value} at row {position}, outside valid range [0, {c})"
        ));
    }
    Ok(value as usize)
}

fn leaf_phase_waves(
    families: &[FamilySchedule],
    family_clade_offsets: &[i64],
    wave_cap: usize,
) -> Result<Vec<Vec<i64>>, PreprocessError> {
    let mut leaves = Vec::new();
    for (fi, family) in families.iter().enumerate() {
        for (clade, count) in family.split_counts.iter().enumerate() {
            if *count == 0 {
                leaves.push((fi, clade));
            }
        }
    }
    leaves.sort();
    leaves
        .chunks(wave_cap)
        .map(|chunk| materialize_local_clades(chunk, family_clade_offsets))
        .collect()
}

fn materialize_local_clades(
    batch: &[LocalClade],
    family_clade_offsets: &[i64],
) -> Result<Vec<i64>, PreprocessError> {
    batch
        .iter()
        .map(|(fi, clade)| {
            family_clade_offsets
                .get(*fi)
                .map(|offset| *offset + *clade as i64)
                .ok_or_else(|| {
                    PreprocessError::InvalidInput(format!("missing clade offset for family {fi}"))
                })
        })
        .collect()
}

fn split_root_wave(
    batch: &[LocalClade],
    families: &[FamilySchedule],
    root_cap: Option<usize>,
) -> Vec<Vec<LocalClade>> {
    if root_cap.is_none() || batch.len() <= root_cap.unwrap_or(usize::MAX) {
        return vec![batch.to_vec()];
    }
    if batch
        .iter()
        .all(|(fi, clade)| *clade == families[*fi].root_id)
    {
        return batch
            .chunks(root_cap.expect("checked is_some"))
            .map(|chunk| chunk.to_vec())
            .collect();
    }
    vec![batch.to_vec()]
}

fn materialize_nonleaf_waves(
    batches: &[Vec<LocalClade>],
    families: &[FamilySchedule],
    family_clade_offsets: &[i64],
    root_cap: Option<usize>,
) -> Result<(Vec<Vec<i64>>, Vec<i64>), PreprocessError> {
    let mut waves = Vec::new();
    let mut phases = Vec::new();
    for batch in batches {
        for chunk in split_root_wave(batch, families, root_cap) {
            if chunk.is_empty() {
                continue;
            }
            phases.push(
                if chunk
                    .iter()
                    .all(|(fi, clade)| *clade == families[*fi].root_id)
                {
                    3
                } else {
                    2
                },
            );
            waves.push(materialize_local_clades(&chunk, family_clade_offsets)?);
        }
    }
    Ok((waves, phases))
}

fn clade_dts_partial_tiles(family: &FamilySchedule, clade: usize, tile_splits: usize) -> usize {
    let split_count = family.split_counts[clade];
    if split_count < 2 {
        0
    } else {
        split_count.div_ceil(tile_splits)
    }
}

fn dts_guard_allows_append(
    batch_nonempty: bool,
    batch_ge2_groups: usize,
    batch_max_tiles: usize,
    candidate_tiles: usize,
    max_dts_partial_rows: Option<usize>,
) -> bool {
    let Some(max_rows) = max_dts_partial_rows else {
        return true;
    };
    if !batch_nonempty {
        return true;
    }
    let new_ge2_groups = batch_ge2_groups + usize::from(candidate_tiles > 0);
    let new_max_tiles = batch_max_tiles.max(candidate_tiles);
    new_ge2_groups * new_max_tiles <= max_rows
}

fn schedule_forward_nonleaf_waves(
    families: &[FamilySchedule],
    wave_cap: usize,
    root_cap: Option<usize>,
    max_dts_partial_rows: Option<usize>,
    dts_partial_tile_splits: usize,
) -> Vec<Vec<LocalClade>> {
    let mut scheduled = families
        .iter()
        .map(|family| vec![false; family.c])
        .collect::<Vec<_>>();
    let mut queued = families
        .iter()
        .map(|family| vec![false; family.c])
        .collect::<Vec<_>>();
    let mut remaining = families
        .iter()
        .map(|family| family.remaining.clone())
        .collect::<Vec<_>>();

    for (fi, family) in families.iter().enumerate() {
        for (clade, count) in family.split_counts.iter().enumerate() {
            if *count == 0 {
                scheduled[fi][clade] = true;
                for parent in &family.parents_of[clade] {
                    remaining[fi][*parent] -= 1;
                }
            }
        }
    }

    let mut ready: BinaryHeap<Reverse<(i64, usize, usize)>> = BinaryHeap::new();
    push_all_ready_forward(families, &scheduled, &mut queued, &remaining, &mut ready);

    let mut batches = Vec::new();
    while !ready.is_empty() {
        let mut batch = Vec::new();
        let mut batch_ge2_groups = 0usize;
        let mut batch_max_tiles = 0usize;
        let mut deferred = Vec::new();
        while !ready.is_empty() && batch.len() < wave_cap {
            let entry = ready.pop().expect("ready not empty").0;
            let (_neg_priority, fi, clade) = entry;
            if scheduled[fi][clade] {
                continue;
            }
            queued[fi][clade] = false;
            if remaining[fi][clade] != 0 {
                continue;
            }
            let tiles = clade_dts_partial_tiles(&families[fi], clade, dts_partial_tile_splits);
            if !dts_guard_allows_append(
                !batch.is_empty(),
                batch_ge2_groups,
                batch_max_tiles,
                tiles,
                max_dts_partial_rows,
            ) {
                queued[fi][clade] = true;
                deferred.push(entry);
                continue;
            }
            batch.push((fi, clade));
            batch_ge2_groups += usize::from(tiles > 0);
            batch_max_tiles = batch_max_tiles.max(tiles);
        }
        for entry in deferred {
            ready.push(Reverse(entry));
        }
        if batch.is_empty() {
            continue;
        }
        for chunk in split_root_wave(&batch, families, root_cap) {
            batches.push(chunk.clone());
            for (fi, clade) in &chunk {
                scheduled[*fi][*clade] = true;
            }
            for (fi, clade) in &chunk {
                for parent in &families[*fi].parents_of[*clade] {
                    remaining[*fi][*parent] -= 1;
                    push_ready_forward(
                        families,
                        &scheduled,
                        &mut queued,
                        &remaining,
                        &mut ready,
                        *fi,
                        *parent,
                    );
                }
            }
        }
    }
    batches
}

fn push_all_ready_forward(
    families: &[FamilySchedule],
    scheduled: &[Vec<bool>],
    queued: &mut [Vec<bool>],
    remaining: &[Vec<usize>],
    ready: &mut BinaryHeap<Reverse<(i64, usize, usize)>>,
) {
    for (fi, family) in families.iter().enumerate() {
        for clade in 0..family.c {
            push_ready_forward(families, scheduled, queued, remaining, ready, fi, clade);
        }
    }
}

fn push_ready_forward(
    families: &[FamilySchedule],
    scheduled: &[Vec<bool>],
    queued: &mut [Vec<bool>],
    remaining: &[Vec<usize>],
    ready: &mut BinaryHeap<Reverse<(i64, usize, usize)>>,
    fi: usize,
    clade: usize,
) {
    if scheduled[fi][clade] || queued[fi][clade] || remaining[fi][clade] != 0 {
        return;
    }
    queued[fi][clade] = true;
    ready.push(Reverse((-(families[fi].priority[clade] as i64), fi, clade)));
}

fn schedule_reverse_compacted_nonleaf_waves(
    families: &[FamilySchedule],
    wave_cap: usize,
    root_cap: Option<usize>,
    max_dts_partial_rows: Option<usize>,
    dts_partial_tile_splits: usize,
) -> Vec<Vec<LocalClade>> {
    let mut scheduled = families
        .iter()
        .map(|family| vec![false; family.c])
        .collect::<Vec<_>>();
    let mut queued = families
        .iter()
        .map(|family| vec![false; family.c])
        .collect::<Vec<_>>();
    let mut successors_remaining = nonleaf_successor_counts(families);
    let mut ready: BinaryHeap<Reverse<(i64, i64, usize, usize)>> = BinaryHeap::new();
    push_all_ready_reverse(
        families,
        &scheduled,
        &mut queued,
        &successors_remaining,
        &mut ready,
    );

    let mut reverse_batches = Vec::new();
    while !ready.is_empty() {
        let mut batch = Vec::new();
        let mut batch_ge2_groups = 0usize;
        let mut batch_max_tiles = 0usize;
        let mut deferred = Vec::new();
        while !ready.is_empty() && batch.len() < wave_cap {
            let entry = ready.pop().expect("ready not empty").0;
            let (_priority, _neg_fanout, fi, clade) = entry;
            if scheduled[fi][clade] || successors_remaining[fi][clade] != 0 {
                continue;
            }
            queued[fi][clade] = false;
            let tiles = clade_dts_partial_tiles(&families[fi], clade, dts_partial_tile_splits);
            if !dts_guard_allows_append(
                !batch.is_empty(),
                batch_ge2_groups,
                batch_max_tiles,
                tiles,
                max_dts_partial_rows,
            ) {
                queued[fi][clade] = true;
                deferred.push(entry);
                continue;
            }
            batch.push((fi, clade));
            batch_ge2_groups += usize::from(tiles > 0);
            batch_max_tiles = batch_max_tiles.max(tiles);
        }
        for entry in deferred {
            ready.push(Reverse(entry));
        }
        if batch.is_empty() {
            continue;
        }
        for chunk in split_root_wave(&batch, families, root_cap) {
            reverse_batches.push(chunk.clone());
            for (fi, clade) in &chunk {
                scheduled[*fi][*clade] = true;
            }
            for (fi, clade) in &chunk {
                for child in &families[*fi].children[*clade] {
                    if families[*fi].split_counts[*child] == 0 {
                        continue;
                    }
                    successors_remaining[*fi][*child] -= 1;
                    push_ready_reverse(
                        families,
                        &scheduled,
                        &mut queued,
                        &successors_remaining,
                        &mut ready,
                        *fi,
                        *child,
                    );
                }
            }
        }
    }
    reverse_batches.reverse();
    reverse_batches
}

fn nonleaf_successor_counts(families: &[FamilySchedule]) -> Vec<Vec<usize>> {
    let mut successors_remaining = families
        .iter()
        .map(|family| vec![0usize; family.c])
        .collect::<Vec<_>>();
    for (fi, family) in families.iter().enumerate() {
        for clade in 0..family.c {
            if family.split_counts[clade] == 0 {
                continue;
            }
            successors_remaining[fi][clade] = family.parents_of[clade]
                .iter()
                .filter(|parent| family.split_counts[**parent] != 0)
                .count();
        }
    }
    successors_remaining
}

fn push_all_ready_reverse(
    families: &[FamilySchedule],
    scheduled: &[Vec<bool>],
    queued: &mut [Vec<bool>],
    successors_remaining: &[Vec<usize>],
    ready: &mut BinaryHeap<Reverse<(i64, i64, usize, usize)>>,
) {
    for (fi, family) in families.iter().enumerate() {
        for clade in 0..family.c {
            push_ready_reverse(
                families,
                scheduled,
                queued,
                successors_remaining,
                ready,
                fi,
                clade,
            );
        }
    }
}

fn push_ready_reverse(
    families: &[FamilySchedule],
    scheduled: &[Vec<bool>],
    queued: &mut [Vec<bool>],
    successors_remaining: &[Vec<usize>],
    ready: &mut BinaryHeap<Reverse<(i64, i64, usize, usize)>>,
    fi: usize,
    clade: usize,
) {
    if families[fi].split_counts[clade] == 0
        || scheduled[fi][clade]
        || queued[fi][clade]
        || successors_remaining[fi][clade] != 0
    {
        return;
    }
    queued[fi][clade] = true;
    ready.push(Reverse((
        families[fi].priority[clade] as i64,
        -(families[fi].children[clade].len() as i64),
        fi,
        clade,
    )));
}

fn nonleaf_earliest_wave(family: &FamilySchedule, clade: usize) -> usize {
    family.bfs_level[clade].saturating_sub(1)
}

fn nonleaf_wave_lower_bound(families: &[FamilySchedule], wave_cap: usize) -> usize {
    let total_nonleaves = families
        .iter()
        .map(|family| family.nonleaf_count)
        .sum::<usize>();
    let max_depth = families
        .iter()
        .map(|family| family.max_level)
        .max()
        .unwrap_or(0);
    let work_waves = total_nonleaves.div_ceil(wave_cap);
    max_depth.max(work_waves)
}

fn schedule_deadline_nonleaf_waves(
    families: &[FamilySchedule],
    wave_cap: usize,
    target_waves: usize,
    max_dts_partial_rows: Option<usize>,
    dts_partial_tile_splits: usize,
) -> Option<Vec<Vec<LocalClade>>> {
    let total_nonleaves = families
        .iter()
        .map(|family| family.nonleaf_count)
        .sum::<usize>();
    if total_nonleaves == 0 {
        return Some(Vec::new());
    }
    if target_waves == 0 {
        return None;
    }

    let mut scheduled = families
        .iter()
        .map(|family| vec![false; family.c])
        .collect::<Vec<_>>();
    let mut queued = families
        .iter()
        .map(|family| vec![false; family.c])
        .collect::<Vec<_>>();
    let mut successors_remaining = nonleaf_successor_counts(families);
    let mut ready: BinaryHeap<Reverse<(i64, i64, i64, usize, usize)>> = BinaryHeap::new();
    push_all_ready_deadline(
        families,
        &scheduled,
        &mut queued,
        &successors_remaining,
        &mut ready,
    );

    let mut reverse_batches = Vec::new();
    let mut scheduled_count = 0usize;
    for wave_idx in (0..target_waves).rev() {
        let mut batch = Vec::new();
        let mut batch_ge2_groups = 0usize;
        let mut batch_max_tiles = 0usize;
        let mut deferred = Vec::new();

        while !ready.is_empty() && batch.len() < wave_cap {
            let entry = ready.pop().expect("ready not empty").0;
            let (neg_earliest, _neg_priority, _neg_fanout, fi, clade) = entry;
            if scheduled[fi][clade] || successors_remaining[fi][clade] != 0 {
                continue;
            }
            queued[fi][clade] = false;
            let earliest = (-neg_earliest) as usize;
            if earliest > wave_idx {
                return None;
            }
            let tiles = clade_dts_partial_tiles(&families[fi], clade, dts_partial_tile_splits);
            if !dts_guard_allows_append(
                !batch.is_empty(),
                batch_ge2_groups,
                batch_max_tiles,
                tiles,
                max_dts_partial_rows,
            ) {
                queued[fi][clade] = true;
                deferred.push(entry);
                continue;
            }
            batch.push((fi, clade));
            batch_ge2_groups += usize::from(tiles > 0);
            batch_max_tiles = batch_max_tiles.max(tiles);
        }
        for entry in deferred {
            ready.push(Reverse(entry));
        }

        if live_ready_misses_deadline(
            &ready,
            &scheduled,
            &successors_remaining,
            wave_idx as i64 - 1,
        ) {
            return None;
        }
        if batch.is_empty() {
            if scheduled_count < total_nonleaves
                && !any_unscheduled_ready_nonleaf(families, &scheduled, &successors_remaining)
            {
                return None;
            }
            continue;
        }

        reverse_batches.push(batch.clone());
        for (fi, clade) in &batch {
            scheduled[*fi][*clade] = true;
            scheduled_count += 1;
        }
        for (fi, clade) in &batch {
            for child in &families[*fi].children[*clade] {
                if families[*fi].split_counts[*child] == 0 {
                    continue;
                }
                successors_remaining[*fi][*child] -= 1;
                push_ready_deadline(
                    families,
                    &scheduled,
                    &mut queued,
                    &successors_remaining,
                    &mut ready,
                    *fi,
                    *child,
                );
            }
        }
    }

    if scheduled_count != total_nonleaves {
        return None;
    }
    reverse_batches.reverse();
    Some(reverse_batches)
}

fn push_all_ready_deadline(
    families: &[FamilySchedule],
    scheduled: &[Vec<bool>],
    queued: &mut [Vec<bool>],
    successors_remaining: &[Vec<usize>],
    ready: &mut BinaryHeap<Reverse<(i64, i64, i64, usize, usize)>>,
) {
    for (fi, family) in families.iter().enumerate() {
        for clade in 0..family.c {
            push_ready_deadline(
                families,
                scheduled,
                queued,
                successors_remaining,
                ready,
                fi,
                clade,
            );
        }
    }
}

fn push_ready_deadline(
    families: &[FamilySchedule],
    scheduled: &[Vec<bool>],
    queued: &mut [Vec<bool>],
    successors_remaining: &[Vec<usize>],
    ready: &mut BinaryHeap<Reverse<(i64, i64, i64, usize, usize)>>,
    fi: usize,
    clade: usize,
) {
    if families[fi].split_counts[clade] == 0
        || scheduled[fi][clade]
        || queued[fi][clade]
        || successors_remaining[fi][clade] != 0
    {
        return;
    }
    queued[fi][clade] = true;
    ready.push(Reverse((
        -(nonleaf_earliest_wave(&families[fi], clade) as i64),
        -(families[fi].priority[clade] as i64),
        -(families[fi].children[clade].len() as i64),
        fi,
        clade,
    )));
}

fn live_ready_misses_deadline(
    ready: &BinaryHeap<Reverse<(i64, i64, i64, usize, usize)>>,
    scheduled: &[Vec<bool>],
    successors_remaining: &[Vec<usize>],
    wave_idx: i64,
) -> bool {
    for Reverse((neg_earliest, _neg_priority, _neg_fanout, fi, clade)) in ready {
        if scheduled[*fi][*clade] || successors_remaining[*fi][*clade] != 0 {
            continue;
        }
        if -neg_earliest > wave_idx {
            return true;
        }
    }
    false
}

fn any_unscheduled_ready_nonleaf(
    families: &[FamilySchedule],
    scheduled: &[Vec<bool>],
    successors_remaining: &[Vec<usize>],
) -> bool {
    for (fi, family) in families.iter().enumerate() {
        for clade in 0..family.c {
            if !scheduled[fi][clade]
                && successors_remaining[fi][clade] == 0
                && family.split_counts[clade] != 0
            {
                return true;
            }
        }
    }
    false
}

fn schedule_coffman_graham_nonleaf_waves(
    families: &[FamilySchedule],
    wave_cap: usize,
    max_dts_partial_rows: Option<usize>,
    dts_partial_tile_splits: usize,
) -> Result<Vec<Vec<LocalClade>>, PreprocessError> {
    let mut labels = families
        .iter()
        .map(|family| vec![-1isize; family.c])
        .collect::<Vec<_>>();
    let mut successors_remaining = nonleaf_successor_counts(families);
    let total_nonleaves = families
        .iter()
        .map(|family| family.nonleaf_count)
        .sum::<usize>();
    let mut ready: BinaryHeap<Reverse<(Vec<isize>, i64, usize, usize)>> = BinaryHeap::new();
    let mut queued = families
        .iter()
        .map(|family| vec![false; family.c])
        .collect::<Vec<_>>();
    push_all_ready_cg(
        families,
        &labels,
        &mut queued,
        &successors_remaining,
        &mut ready,
    );

    let mut labeled_count = 0usize;
    let mut label_order = Vec::new();
    while let Some(Reverse((_successor_labels, _priority, fi, clade))) = ready.pop() {
        if labels[fi][clade] >= 0 || successors_remaining[fi][clade] != 0 {
            continue;
        }
        queued[fi][clade] = false;
        labels[fi][clade] = labeled_count as isize;
        labeled_count += 1;
        label_order.push((fi, clade));
        for child in &families[fi].children[clade] {
            if families[fi].split_counts[*child] == 0 {
                continue;
            }
            successors_remaining[fi][*child] -= 1;
            push_ready_cg(
                families,
                &labels,
                &mut queued,
                &successors_remaining,
                &mut ready,
                fi,
                *child,
            );
        }
    }
    if labeled_count != total_nonleaves {
        return invalid(format!(
            "Coffman-Graham scheduler did not label all non-leaf clades: labeled={labeled_count}, total={total_nonleaves}"
        ));
    }

    let mut assigned_wave = families
        .iter()
        .map(|family| vec![-1isize; family.c])
        .collect::<Vec<_>>();
    let mut waves: Vec<Vec<LocalClade>> = Vec::new();
    let mut wave_ge2_groups: Vec<usize> = Vec::new();
    let mut wave_max_tiles: Vec<usize> = Vec::new();
    for (fi, clade) in label_order.into_iter().rev() {
        let mut earliest = 0usize;
        for child in &families[fi].children[clade] {
            if families[fi].split_counts[*child] == 0 {
                continue;
            }
            let child_wave = assigned_wave[fi][*child];
            if child_wave < 0 {
                return invalid("CG level assignment visited a parent before its child");
            }
            earliest = earliest.max(child_wave as usize + 1);
        }
        let tiles = clade_dts_partial_tiles(&families[fi], clade, dts_partial_tile_splits);
        let mut level = earliest;
        loop {
            ensure_wave(level, &mut waves, &mut wave_ge2_groups, &mut wave_max_tiles);
            if waves[level].len() < wave_cap
                && dts_guard_allows_append(
                    !waves[level].is_empty(),
                    wave_ge2_groups[level],
                    wave_max_tiles[level],
                    tiles,
                    max_dts_partial_rows,
                )
            {
                waves[level].push((fi, clade));
                assigned_wave[fi][clade] = level as isize;
                wave_ge2_groups[level] += usize::from(tiles > 0);
                wave_max_tiles[level] = wave_max_tiles[level].max(tiles);
                break;
            }
            level += 1;
        }
    }
    Ok(waves.into_iter().filter(|wave| !wave.is_empty()).collect())
}

fn push_all_ready_cg(
    families: &[FamilySchedule],
    labels: &[Vec<isize>],
    queued: &mut [Vec<bool>],
    successors_remaining: &[Vec<usize>],
    ready: &mut BinaryHeap<Reverse<(Vec<isize>, i64, usize, usize)>>,
) {
    for (fi, family) in families.iter().enumerate() {
        for clade in 0..family.c {
            push_ready_cg(
                families,
                labels,
                queued,
                successors_remaining,
                ready,
                fi,
                clade,
            );
        }
    }
}

fn push_ready_cg(
    families: &[FamilySchedule],
    labels: &[Vec<isize>],
    queued: &mut [Vec<bool>],
    successors_remaining: &[Vec<usize>],
    ready: &mut BinaryHeap<Reverse<(Vec<isize>, i64, usize, usize)>>,
    fi: usize,
    clade: usize,
) {
    if families[fi].split_counts[clade] == 0
        || labels[fi][clade] >= 0
        || queued[fi][clade]
        || successors_remaining[fi][clade] != 0
    {
        return;
    }
    queued[fi][clade] = true;
    let mut successor_labels = families[fi].parents_of[clade]
        .iter()
        .filter(|parent| families[fi].split_counts[**parent] != 0)
        .map(|parent| labels[fi][*parent])
        .collect::<Vec<_>>();
    successor_labels.sort_by(|a, b| b.cmp(a));
    ready.push(Reverse((
        successor_labels,
        families[fi].priority[clade] as i64,
        fi,
        clade,
    )));
}

fn ensure_wave(
    level: usize,
    waves: &mut Vec<Vec<LocalClade>>,
    wave_ge2_groups: &mut Vec<usize>,
    wave_max_tiles: &mut Vec<usize>,
) {
    while waves.len() <= level {
        waves.push(Vec::new());
        wave_ge2_groups.push(0);
        wave_max_tiles.push(0);
    }
}

fn materialized_nonleaf_wave_count(
    batches: &[Vec<LocalClade>],
    families: &[FamilySchedule],
    root_cap: Option<usize>,
) -> usize {
    batches
        .iter()
        .map(|batch| split_root_wave(batch, families, root_cap).len())
        .sum()
}

fn select_nonleaf_schedule_candidate(
    families: &[FamilySchedule],
    wave_cap: usize,
    root_cap: Option<usize>,
    max_dts_partial_rows: Option<usize>,
    dts_partial_tile_splits: usize,
) -> Result<(String, Vec<Vec<LocalClade>>), PreprocessError> {
    let forward_batches = schedule_forward_nonleaf_waves(
        families,
        wave_cap,
        root_cap,
        max_dts_partial_rows,
        dts_partial_tile_splits,
    );
    let mut best_name = "forward".to_string();
    let mut best_batches = forward_batches;
    let mut best_nonleaf_count = materialized_nonleaf_wave_count(&best_batches, families, root_cap);

    let lower_bound = nonleaf_wave_lower_bound(families, wave_cap);
    if best_nonleaf_count <= lower_bound {
        return Ok((best_name, best_batches));
    }

    for target_waves in lower_bound..best_nonleaf_count {
        let Some(candidate_batches) = schedule_deadline_nonleaf_waves(
            families,
            wave_cap,
            target_waves,
            max_dts_partial_rows,
            dts_partial_tile_splits,
        ) else {
            continue;
        };
        let candidate_count =
            materialized_nonleaf_wave_count(&candidate_batches, families, root_cap);
        if candidate_count < best_nonleaf_count {
            best_name = "deadline".to_string();
            best_batches = candidate_batches;
            best_nonleaf_count = candidate_count;
            if best_nonleaf_count <= lower_bound {
                return Ok((best_name, best_batches));
            }
        }
    }

    for (candidate_name, candidate_batches) in [
        (
            "reverse_compacted",
            schedule_reverse_compacted_nonleaf_waves(
                families,
                wave_cap,
                root_cap,
                max_dts_partial_rows,
                dts_partial_tile_splits,
            ),
        ),
        (
            "coffman_graham",
            schedule_coffman_graham_nonleaf_waves(
                families,
                wave_cap,
                max_dts_partial_rows,
                dts_partial_tile_splits,
            )?,
        ),
    ] {
        let candidate_count =
            materialized_nonleaf_wave_count(&candidate_batches, families, root_cap);
        if candidate_count < best_nonleaf_count {
            best_name = candidate_name.to_string();
            best_batches = candidate_batches;
            best_nonleaf_count = candidate_count;
        }
    }
    Ok((best_name, best_batches))
}

fn validate_scheduled_clades(
    waves: &[Vec<i64>],
    total_clades: usize,
) -> Result<(), PreprocessError> {
    let mut seen = HashSet::new();
    let mut rows = 0usize;
    for wave in waves {
        for clade in wave {
            rows += 1;
            seen.insert(*clade);
        }
    }
    if rows != total_clades || seen.len() != total_clades {
        return invalid(format!(
            "global wave scheduler did not cover all clades: scheduled={}, rows={rows}, total={total_clades}",
            seen.len()
        ));
    }
    Ok(())
}
