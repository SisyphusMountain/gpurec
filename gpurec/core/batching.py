import heapq
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch


_SCHEDULER_BACKEND_ENV = "GPUREC_SCHEDULER_BACKEND"


def collate_gene_families(
    batch: List[Dict[str, Any]],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """
    Collate multiple preprocessed gene-family CCP payloads into one batched layout.

    Each item in `batch` must be a dict with keys:
      - 'ccp': dict containing at least
          'C' (int), 'N_splits' (int),
          'split_leftrights_sorted' (Long[2*N_i]),
          'log_split_probs_sorted' (Float[N_i]),
          'split_parents_sorted' (Long[N_i]),
          'num_segs_eq1' (int),
          'end_rows_ge2' (int).
      - 'leaf_row_index': Long[K_i]
      - 'leaf_col_index': Long[K_i]
      - 'root_clade_id': int

    Returns a dict with:
      - 'ccp': merged CCP helpers consumed by build_wave_layout()
      - 'root_clade_ids': Long[F] (roots offset into concatenated clade axis)
      - 'family_meta': bookkeeping (clade offsets, per-family sizes)
    """
    if not batch:
        raise ValueError("batch must contain at least one family")

    # ---- running offsets and accumulators ----
    clade_offset = 0

    root_ids: List[int] = []
    leaf_row_parts, leaf_col_parts = [], []
    # (>=2) block
    # we will accumulate left and right halves separately to preserve [all_lefts ; all_rights] order
    ge2_left_parts: List[torch.Tensor] = []
    ge2_right_parts: List[torch.Tensor] = []
    ge2_logp_parts: List[torch.Tensor] = []

    # (=1) block
    eq1_left_parts: List[torch.Tensor] = []
    eq1_right_parts: List[torch.Tensor] = []
    eq1_logp_parts: List[torch.Tensor] = []

    # split_parents_sorted (one entry per split, needed for wave scheduling)
    ge2_split_parents_parts: List[torch.Tensor] = []
    eq1_split_parents_parts: List[torch.Tensor] = []

    # Totals & per-family meta
    total_C = 0
    total_N = 0

    family_meta: List[Dict[str, int]] = []

    for family_index, item in enumerate(batch):
        ccp = item["ccp"]
        root_i: int = int(item["root_clade_id"])

        # Family sizes
        C_i = int(ccp["C"])
        N_i = int(ccp["N_splits"])
        num_eq1_i = int(ccp["num_segs_eq1"])
        end_rows_ge2_i = int(ccp["end_rows_ge2"])
        _validate_split_block_lengths(
            family_index=family_index,
            n_splits=N_i,
            num_eq1=num_eq1_i,
            end_rows_ge2=end_rows_ge2_i,
        )

        # Pull split arrays
        leftrights_i = ccp["split_leftrights_sorted"].to(torch.long).cpu()  # [2*N_i]
        logp_i = ccp["log_split_probs_sorted"].to(dtype).cpu()              # [N_i]
        _require_numel(
            "split_leftrights_sorted",
            leftrights_i,
            2 * N_i,
            family_index=family_index,
        )
        _require_numel(
            "log_split_probs_sorted",
            logp_i,
            N_i,
            family_index=family_index,
        )

        # Split left/right halves so we can cut the ge2 vs eq1 ranges cleanly
        lefts_i = leftrights_i[:N_i]
        rights_i = leftrights_i[N_i:]

        # Offsets on clade indices for this family
        lefts_i = lefts_i + clade_offset
        rights_i = rights_i + clade_offset

        if "split_parents_sorted" not in ccp:
            raise RuntimeError("preprocessed CCP helpers must include split_parents_sorted")
        sp_sorted_i = ccp["split_parents_sorted"].to(torch.long).cpu()
        _require_numel(
            "split_parents_sorted",
            sp_sorted_i,
            N_i,
            family_index=family_index,
        )

        # (>=2) rows for this family
        if end_rows_ge2_i > 0:
            ge2_left = lefts_i[:end_rows_ge2_i]
            ge2_right = rights_i[:end_rows_ge2_i]
            ge2_left_parts.append(ge2_left)
            ge2_right_parts.append(ge2_right)
            ge2_logp_parts.append(logp_i[:end_rows_ge2_i])
            ge2_split_parents_parts.append(sp_sorted_i[:end_rows_ge2_i] + clade_offset)

        # (=1) rows for this family (exactly one split per clade)
        if num_eq1_i > 0:
            start = end_rows_ge2_i
            stop = end_rows_ge2_i + num_eq1_i
            eq1_left = lefts_i[start:stop]
            eq1_right = rights_i[start:stop]
            eq1_left_parts.append(eq1_left)
            eq1_right_parts.append(eq1_right)
            eq1_logp_parts.append(logp_i[start:stop])
            eq1_split_parents_parts.append(sp_sorted_i[start:stop] + clade_offset)

        # Species/clade leaf mappings
        lr = item["leaf_row_index"].to(torch.long).to(device) + clade_offset
        lc = item["leaf_col_index"].to(torch.long).to(device)
        leaf_row_parts.append(lr)
        leaf_col_parts.append(lc)
        root_ids.append(root_i + clade_offset)

        # Totals and offsets
        total_C += C_i
        total_N += N_i

        family_meta.append({
            "C": C_i,
            "clade_offset": clade_offset,
        })

        clade_offset += C_i  # advance clade offset for next family

    # ---- Build the batched arrays in the global order: [all GE2 rows] then [all EQ1 rows] ----
    # split_leftrights_sorted must be [all_lefts ; all_rights], each of length total_N
    if len(ge2_left_parts) > 0:
        ge2_left = torch.cat(ge2_left_parts, dim=0)
        ge2_right = torch.cat(ge2_right_parts, dim=0)
    else:
        ge2_left = torch.empty((0,), dtype=torch.long)
        ge2_right = torch.empty((0,), dtype=torch.long)
    if len(eq1_left_parts) > 0:
        eq1_left = torch.cat(eq1_left_parts, dim=0)
        eq1_right = torch.cat(eq1_right_parts, dim=0)
    else:
        eq1_left = torch.empty((0,), dtype=torch.long)
        eq1_right = torch.empty((0,), dtype=torch.long)
    left_all = torch.cat([ge2_left, eq1_left], dim=0)
    right_all = torch.cat([ge2_right, eq1_right], dim=0)
    split_leftrights_sorted_batch = torch.cat([left_all, right_all], dim=0)

    # log_split_probs_sorted (N_total)
    if len(ge2_logp_parts) > 0:
        ge2_logp = torch.cat(ge2_logp_parts, dim=0)
    else:
        ge2_logp = torch.empty((0,), dtype=dtype)
    if len(eq1_logp_parts) > 0:
        eq1_logp = torch.cat(eq1_logp_parts, dim=0)
    else:
        eq1_logp = torch.empty((0,), dtype=dtype)
    log_split_probs_sorted_batch = torch.cat([ge2_logp, eq1_logp], dim=0)

    # split_parents_sorted: [all ge2 split parents ; all eq1 split parents]
    if len(ge2_split_parents_parts) > 0:
        ge2_sp = torch.cat(ge2_split_parents_parts, dim=0)
    else:
        ge2_sp = torch.empty((0,), dtype=torch.long)
    if len(eq1_split_parents_parts) > 0:
        eq1_sp = torch.cat(eq1_split_parents_parts, dim=0)
    else:
        eq1_sp = torch.empty((0,), dtype=torch.long)
    split_parents_sorted_batch = torch.cat([ge2_sp, eq1_sp], dim=0)

    _require_numel("split_leftrights_sorted", split_leftrights_sorted_batch, 2 * total_N)
    _require_numel("log_split_probs_sorted", log_split_probs_sorted_batch, total_N)
    _require_numel("split_parents_sorted", split_parents_sorted_batch, total_N)

    leaf_row_index = torch.cat(leaf_row_parts, 0).to(device)
    leaf_col_index = torch.cat(leaf_col_parts, 0).to(device)
    out = {
        "ccp": {
            "C": total_C,
            "N_splits": total_N,
            "split_leftrights_sorted": split_leftrights_sorted_batch.to(device),
            "log_split_probs_sorted": log_split_probs_sorted_batch.to(device),
            "split_parents_sorted": split_parents_sorted_batch.to(device),
        },
        "leaf_row_index": leaf_row_index,
        "leaf_col_index": leaf_col_index,
        "root_clade_ids": torch.tensor(root_ids, dtype=torch.long, device=device),
        "family_meta": family_meta,
    }
    return out


def _cpu_long_list(value: Any) -> List[int]:
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


def _numel(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel())
    try:
        return len(value)
    except TypeError as exc:
        raise ValueError("CCP helper values must be tensors or sized sequences") from exc


def _require_numel(
    name: str,
    value: Any,
    expected: int,
    *,
    family_index: int | None = None,
) -> None:
    actual = _numel(value)
    if actual == expected:
        return
    prefix = "" if family_index is None else f"family {family_index} "
    raise ValueError(f"{prefix}{name} has length {actual} but expected {expected}")


def _validate_split_block_lengths(
    *,
    family_index: int,
    n_splits: int,
    num_eq1: int,
    end_rows_ge2: int,
) -> None:
    if n_splits < 0:
        raise ValueError(f"family {family_index} N_splits must be non-negative")
    if num_eq1 < 0:
        raise ValueError(f"family {family_index} num_segs_eq1 must be non-negative")
    if end_rows_ge2 < 0:
        raise ValueError(f"family {family_index} end_rows_ge2 must be non-negative")
    if end_rows_ge2 + num_eq1 != n_splits:
        raise ValueError(
            f"family {family_index} split block lengths cover "
            f"{end_rows_ge2 + num_eq1} rows but N_splits={n_splits}"
        )


def _validate_wave_clade_coverage(all_clades: Sequence[int], C: int) -> None:
    if len(all_clades) != C:
        raise ValueError(f"Wave layout covers {len(all_clades)} clades but C={C}")
    seen: set[int] = set()
    for position, clade in enumerate(all_clades):
        clade_id = int(clade)
        if clade_id < 0 or clade_id >= C:
            raise ValueError(
                f"Wave layout contains clade {clade_id} at position {position}, "
                f"outside valid range [0, {C})"
            )
        if clade_id in seen:
            raise ValueError(f"Wave layout contains duplicate clade {clade_id}")
        seen.add(clade_id)


def _family_metadata_index(name: str, family_index: int, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name}[{family_index}] must be an integer")
    try:
        return value.__index__()
    except AttributeError as exc:
        raise ValueError(f"{name}[{family_index}] must be an integer") from exc


def _validate_family_clade_metadata(
    family_clade_counts: Sequence[int] | None,
    family_clade_offsets: Sequence[int] | None,
    C: int,
) -> List[Tuple[int, int]] | None:
    if family_clade_counts is None and family_clade_offsets is None:
        return None
    if family_clade_counts is None or family_clade_offsets is None:
        raise ValueError(
            "family_clade_counts and family_clade_offsets must be provided together"
        )
    if len(family_clade_counts) != len(family_clade_offsets):
        raise ValueError(
            "family_clade_counts and family_clade_offsets must have matching lengths"
        )

    covered = [False] * C
    ranges: List[Tuple[int, int]] = []
    for family_index, (offset_value, count_value) in enumerate(
        zip(family_clade_offsets, family_clade_counts)
    ):
        offset = _family_metadata_index(
            "family_clade_offsets",
            family_index,
            offset_value,
        )
        count = _family_metadata_index(
            "family_clade_counts",
            family_index,
            count_value,
        )
        if offset < 0:
            raise ValueError(
                f"family_clade_offsets[{family_index}] must be non-negative"
            )
        if count < 0:
            raise ValueError(
                f"family_clade_counts[{family_index}] must be non-negative"
            )
        end = offset + count
        if end > C:
            raise ValueError(
                f"family {family_index} clade range [{offset}, {end}) "
                f"is outside C={C}"
            )
        for clade in range(offset, end):
            if covered[clade]:
                raise ValueError(
                    f"family clade metadata overlaps clade {clade}"
                )
            covered[clade] = True
        ranges.append((offset, count))

    for clade, is_covered in enumerate(covered):
        if not is_covered:
            raise ValueError(f"family clade metadata does not cover clade {clade}")
    return ranges


def _ccp_split_counts(ccp: Dict[str, Any], C: int, parents: Sequence[int]) -> List[int]:
    derived = [0] * C
    for row, parent in enumerate(parents):
        parent_id = int(parent)
        if parent_id < 0 or parent_id >= C:
            raise ValueError(
                f"split_parents_sorted contains parent {parent_id} at row {row}, "
                f"outside valid range [0, {C})"
            )
        derived[parent_id] += 1
    if "split_counts" in ccp:
        counts = _cpu_long_list(ccp["split_counts"])
        if len(counts) != C:
            raise ValueError(f"split_counts has length {len(counts)} but C={C}")
        if counts != derived:
            raise ValueError("split_counts does not match split_parents_sorted")
        return counts
    return derived


def _validate_split_child_id(name: str, value: int, *, row: int, C: int) -> int:
    child_id = int(value)
    if child_id < 0 or child_id >= C:
        raise ValueError(
            f"{name} contains child {child_id} at row {row}, "
            f"outside valid range [0, {C})"
        )
    return child_id


def _validate_clade_id_values(name: str, value: Any, *, C: int) -> None:
    ids = _cpu_long_list(value)
    for position, clade in enumerate(ids):
        clade_id = int(clade)
        if clade_id < 0 or clade_id >= C:
            raise ValueError(
                f"{name} contains clade {clade_id} at position {position}, "
                f"outside valid range [0, {C})"
            )


def _family_schedule_data(ccp: Dict[str, Any]) -> Dict[str, Any]:
    """Build the dependency data needed for cross-family wave scheduling."""
    C = int(ccp["C"])
    N = int(ccp["N_splits"])
    if "root_clade_id" not in ccp:
        raise ValueError("ccp helpers must include root_clade_id")
    root_id = int(ccp["root_clade_id"])
    if root_id < 0 or root_id >= C:
        raise ValueError(
            f"root_clade_id {root_id} outside valid range [0, {C})"
        )
    parents = _cpu_long_list(ccp["split_parents_sorted"])
    leftrights = _cpu_long_list(ccp["split_leftrights_sorted"])
    if len(parents) != N:
        raise ValueError(f"split_parents_sorted has length {len(parents)} but N_splits={N}")
    if len(leftrights) != 2 * N:
        raise ValueError(
            f"split_leftrights_sorted has length {len(leftrights)} but 2*N_splits={2 * N}"
        )
    lefts = leftrights[:N]
    rights = leftrights[N:]
    split_counts = _ccp_split_counts(ccp, C, parents)

    children: List[List[int]] = [[] for _ in range(C)]
    parents_of: List[List[int]] = [[] for _ in range(C)]
    remaining = [0] * C
    child_sets: List[set[int]] = [set() for _ in range(C)]
    for row, (p, l, r) in enumerate(zip(parents, lefts, rights)):
        p = int(p)
        for child in (
            _validate_split_child_id(
                "split_leftrights_sorted",
                int(l),
                row=row,
                C=C,
            ),
            _validate_split_child_id(
                "split_leftrights_sorted",
                int(r),
                row=N + row,
                C=C,
            ),
        ):
            if child not in child_sets[p]:
                child_sets[p].add(child)
                children[p].append(child)
                parents_of[child].append(p)
                remaining[p] += 1

    # Bottom-up levels identify when a clade can first become ready; lambda is
    # a root-distance priority used to keep long chains moving when a ready set
    # is larger than the wave cap.
    bfs_level = [0] * C
    remaining_bfs = list(remaining)
    queue = [c for c in range(C) if remaining_bfs[c] == 0]
    head = 0
    max_level = 0
    while head < len(queue):
        c = queue[head]
        head += 1
        for p in parents_of[c]:
            if bfs_level[p] <= bfs_level[c]:
                bfs_level[p] = bfs_level[c] + 1
                max_level = max(max_level, bfs_level[p])
            remaining_bfs[p] -= 1
            if remaining_bfs[p] == 0:
                queue.append(p)

    levels: List[List[int]] = [[] for _ in range(max_level + 1)]
    for c, level in enumerate(bfs_level):
        levels[level].append(c)
    priority = [0] * C
    for level in range(max_level, -1, -1):
        for c in levels[level]:
            for child in children[c]:
                priority[child] = max(priority[child], priority[c] + 1)

    return {
        "C": C,
        "split_counts": split_counts,
        "children": children,
        "parents_of": parents_of,
        "remaining": remaining,
        "bfs_level": bfs_level,
        "priority": priority,
        "leaf_count": sum(1 for count in split_counts if int(count) == 0),
        "nonleaf_count": sum(1 for count in split_counts if int(count) != 0),
        "max_level": max_level,
        "root_id": root_id,
    }


def family_schedule_summary(ccp: Dict[str, Any]) -> Dict[str, int]:
    """Return compact per-family scheduling stats for resident batch packing."""
    data = _family_schedule_data(ccp)
    return {
        "clade_count": int(data["C"]),
        "leaf_count": int(data["leaf_count"]),
        "nonleaf_count": int(data["nonleaf_count"]),
        "max_level": int(data["max_level"]),
    }


def _leaf_phase_waves(
    families: Sequence[Dict[str, Any]],
    family_clade_offsets: Sequence[int],
    *,
    wave_cap: int,
) -> List[List[int]]:
    all_leaves: List[Tuple[int, int]] = []
    for fi, fam in enumerate(families):
        for c, count in enumerate(fam["split_counts"]):
            if int(count) == 0:
                all_leaves.append((fi, c))

    all_leaves.sort(key=lambda item: (item[0], item[1]))
    return [
        [
            int(family_clade_offsets[fi]) + c
            for fi, c in all_leaves[start:start + wave_cap]
        ]
        for start in range(0, len(all_leaves), wave_cap)
    ]


def _split_root_wave(
    batch: Sequence[Tuple[int, int]],
    families: Sequence[Dict[str, Any]],
    *,
    root_cap: int | None,
) -> List[List[Tuple[int, int]]]:
    if root_cap is None or len(batch) <= root_cap:
        return [list(batch)]
    if all(c == families[fi]["root_id"] for fi, c in batch):
        return [
            list(batch[start:start + root_cap])
            for start in range(0, len(batch), root_cap)
        ]
    return [list(batch)]


def _materialize_nonleaf_waves(
    batches: Sequence[Sequence[Tuple[int, int]]],
    families: Sequence[Dict[str, Any]],
    family_clade_offsets: Sequence[int],
    *,
    root_cap: int | None,
) -> Tuple[List[List[int]], List[int]]:
    waves: List[List[int]] = []
    phases: List[int] = []
    for batch in batches:
        for chunk in _split_root_wave(batch, families, root_cap=root_cap):
            if not chunk:
                continue
            waves.append([
                int(family_clade_offsets[fi]) + c
                for fi, c in chunk
            ])
            phases.append(
                3 if all(c == families[fi]["root_id"] for fi, c in chunk) else 2
            )
    return waves, phases


def _clade_dts_partial_tiles(
    fam: Dict[str, Any],
    c: int,
    *,
    tile_splits: int,
) -> int:
    split_count = int(fam["split_counts"][c])
    if split_count < 2:
        return 0
    return (split_count + tile_splits - 1) // tile_splits


def _dts_guard_allows_append(
    *,
    batch_nonempty: bool,
    batch_ge2_groups: int,
    batch_max_tiles: int,
    candidate_tiles: int,
    max_dts_partial_rows: int | None,
) -> bool:
    if max_dts_partial_rows is None or not batch_nonempty:
        return True
    new_ge2_groups = batch_ge2_groups + (1 if candidate_tiles > 0 else 0)
    new_max_tiles = max(batch_max_tiles, candidate_tiles)
    return new_ge2_groups * new_max_tiles <= max_dts_partial_rows


def _schedule_forward_nonleaf_waves(
    families: Sequence[Dict[str, Any]],
    *,
    wave_cap: int,
    root_cap: int | None,
    max_dts_partial_rows: int | None,
    dts_partial_tile_splits: int,
) -> List[List[Tuple[int, int]]]:
    scheduled = [[False] * fam["C"] for fam in families]
    queued = [[False] * fam["C"] for fam in families]
    remaining = [list(fam["remaining"]) for fam in families]

    for fi, fam in enumerate(families):
        for c, count in enumerate(fam["split_counts"]):
            if int(count) == 0:
                scheduled[fi][c] = True
                for parent in fam["parents_of"][c]:
                    remaining[fi][parent] -= 1

    ready: List[Tuple[int, int, int]] = []

    def push_ready(fi: int, c: int) -> None:
        if scheduled[fi][c] or queued[fi][c] or remaining[fi][c] != 0:
            return
        queued[fi][c] = True
        priority = int(families[fi]["priority"][c])
        # heapq is a min-heap.  Use negative priority so long root-distance
        # chains are drained first, then stable family/clade tie breakers.
        heapq.heappush(ready, (-priority, fi, c))

    for fi, fam in enumerate(families):
        for c in range(fam["C"]):
            push_ready(fi, c)

    batches: List[List[Tuple[int, int]]] = []
    while ready:
        batch: List[Tuple[int, int]] = []
        batch_ge2_groups = 0
        batch_max_tiles = 0
        deferred: List[Tuple[int, int, int]] = []
        while ready and len(batch) < wave_cap:
            entry = heapq.heappop(ready)
            _neg_priority, fi, c = entry
            if scheduled[fi][c]:
                continue
            queued[fi][c] = False
            if remaining[fi][c] != 0:
                continue
            tiles = _clade_dts_partial_tiles(
                families[fi],
                c,
                tile_splits=dts_partial_tile_splits,
            )
            if not _dts_guard_allows_append(
                batch_nonempty=bool(batch),
                batch_ge2_groups=batch_ge2_groups,
                batch_max_tiles=batch_max_tiles,
                candidate_tiles=tiles,
                max_dts_partial_rows=max_dts_partial_rows,
            ):
                queued[fi][c] = True
                deferred.append(entry)
                continue
            batch.append((fi, c))
            batch_ge2_groups += 1 if tiles > 0 else 0
            batch_max_tiles = max(batch_max_tiles, tiles)
        for entry in deferred:
            heapq.heappush(ready, entry)
        if not batch:
            continue

        for chunk in _split_root_wave(batch, families, root_cap=root_cap):
            batches.append(chunk)
            for fi, c in chunk:
                scheduled[fi][c] = True
            for fi, c in chunk:
                for parent in families[fi]["parents_of"][c]:
                    remaining[fi][parent] -= 1
                    push_ready(fi, parent)

    return batches


def _schedule_reverse_compacted_nonleaf_waves(
    families: Sequence[Dict[str, Any]],
    *,
    wave_cap: int,
    root_cap: int | None,
    max_dts_partial_rows: int | None,
    dts_partial_tile_splits: int,
) -> List[List[Tuple[int, int]]]:
    """Build a latest-valid non-leaf schedule and return it in forward order."""
    scheduled = [[False] * fam["C"] for fam in families]
    queued = [[False] * fam["C"] for fam in families]
    successors_remaining: List[List[int]] = [
        [0] * fam["C"]
        for fam in families
    ]

    for fi, fam in enumerate(families):
        for c in range(fam["C"]):
            if int(fam["split_counts"][c]) == 0:
                continue
            successors_remaining[fi][c] = sum(
                1
                for parent in fam["parents_of"][c]
                if int(fam["split_counts"][parent]) != 0
            )

    ready: List[Tuple[int, int, int, int]] = []

    def push_ready(fi: int, c: int) -> None:
        if (
            int(families[fi]["split_counts"][c]) == 0
            or scheduled[fi][c]
            or queued[fi][c]
            or successors_remaining[fi][c] != 0
        ):
            return
        queued[fi][c] = True
        priority = int(families[fi]["priority"][c])
        fanout = len(families[fi]["children"][c])
        # Reverse scheduling starts from sinks.  Lower root-distance priority
        # is later in the forward pass; larger fanout helps unlock predecessors.
        heapq.heappush(ready, (priority, -fanout, fi, c))

    for fi, fam in enumerate(families):
        for c in range(fam["C"]):
            push_ready(fi, c)

    reverse_batches: List[List[Tuple[int, int]]] = []
    while ready:
        batch: List[Tuple[int, int]] = []
        batch_ge2_groups = 0
        batch_max_tiles = 0
        deferred: List[Tuple[int, int, int, int]] = []
        while ready and len(batch) < wave_cap:
            entry = heapq.heappop(ready)
            _priority, _neg_fanout, fi, c = entry
            if scheduled[fi][c] or successors_remaining[fi][c] != 0:
                continue
            queued[fi][c] = False
            tiles = _clade_dts_partial_tiles(
                families[fi],
                c,
                tile_splits=dts_partial_tile_splits,
            )
            if not _dts_guard_allows_append(
                batch_nonempty=bool(batch),
                batch_ge2_groups=batch_ge2_groups,
                batch_max_tiles=batch_max_tiles,
                candidate_tiles=tiles,
                max_dts_partial_rows=max_dts_partial_rows,
            ):
                queued[fi][c] = True
                deferred.append(entry)
                continue
            batch.append((fi, c))
            batch_ge2_groups += 1 if tiles > 0 else 0
            batch_max_tiles = max(batch_max_tiles, tiles)
        for entry in deferred:
            heapq.heappush(ready, entry)
        if not batch:
            continue

        for chunk in _split_root_wave(batch, families, root_cap=root_cap):
            reverse_batches.append(chunk)
            for fi, c in chunk:
                scheduled[fi][c] = True
            for fi, c in chunk:
                for child in families[fi]["children"][c]:
                    if int(families[fi]["split_counts"][child]) == 0:
                        continue
                    successors_remaining[fi][child] -= 1
                    push_ready(fi, child)

    return list(reversed(reverse_batches))


def _nonleaf_earliest_wave(fam: Dict[str, Any], c: int) -> int:
    # Leaf clades are processed in a separate phase, so the first non-leaf
    # wave is bottom-up level 1 shifted to index 0.
    return max(0, int(fam["bfs_level"][c]) - 1)


def _nonleaf_wave_lower_bound(
    families: Sequence[Dict[str, Any]],
    *,
    wave_cap: int,
) -> int:
    total_nonleaves = sum(int(fam["nonleaf_count"]) for fam in families)
    max_depth = max((int(fam["max_level"]) for fam in families), default=0)
    work_waves = (total_nonleaves + wave_cap - 1) // wave_cap
    return max(max_depth, work_waves)


def _schedule_deadline_nonleaf_waves(
    families: Sequence[Dict[str, Any]],
    *,
    wave_cap: int,
    target_waves: int,
    max_dts_partial_rows: int | None,
    dts_partial_tile_splits: int,
) -> List[List[Tuple[int, int]]] | None:
    """Try to fit the post-leaf DAG into a fixed number of non-leaf waves."""
    total_nonleaves = sum(int(fam["nonleaf_count"]) for fam in families)
    if total_nonleaves == 0:
        return []
    if target_waves <= 0:
        return None

    scheduled = [[False] * fam["C"] for fam in families]
    queued = [[False] * fam["C"] for fam in families]
    successors_remaining: List[List[int]] = [
        [0] * fam["C"]
        for fam in families
    ]

    for fi, fam in enumerate(families):
        for c in range(fam["C"]):
            if int(fam["split_counts"][c]) == 0:
                continue
            successors_remaining[fi][c] = sum(
                1
                for parent in fam["parents_of"][c]
                if int(fam["split_counts"][parent]) != 0
            )

    ready: List[Tuple[int, int, int, int, int]] = []

    def push_ready(fi: int, c: int) -> None:
        if (
            int(families[fi]["split_counts"][c]) == 0
            or scheduled[fi][c]
            or queued[fi][c]
            or successors_remaining[fi][c] != 0
        ):
            return
        queued[fi][c] = True
        earliest = _nonleaf_earliest_wave(families[fi], c)
        priority = int(families[fi]["priority"][c])
        fanout = len(families[fi]["children"][c])
        heapq.heappush(ready, (-earliest, -priority, -fanout, fi, c))

    def live_ready_misses_deadline(wave_idx: int) -> bool:
        for neg_earliest, _neg_priority, _neg_fanout, fi, c in ready:
            if scheduled[fi][c] or successors_remaining[fi][c] != 0:
                continue
            if -neg_earliest > wave_idx:
                return True
        return False

    for fi, fam in enumerate(families):
        for c in range(fam["C"]):
            push_ready(fi, c)

    reverse_batches: List[List[Tuple[int, int]]] = []
    scheduled_count = 0
    for wave_idx in range(target_waves - 1, -1, -1):
        batch: List[Tuple[int, int]] = []
        batch_ge2_groups = 0
        batch_max_tiles = 0
        deferred: List[Tuple[int, int, int, int, int]] = []

        while ready and len(batch) < wave_cap:
            entry = heapq.heappop(ready)
            neg_earliest, _neg_priority, _neg_fanout, fi, c = entry
            if scheduled[fi][c] or successors_remaining[fi][c] != 0:
                continue
            queued[fi][c] = False
            earliest = -neg_earliest
            if earliest > wave_idx:
                return None

            tiles = _clade_dts_partial_tiles(
                families[fi],
                c,
                tile_splits=dts_partial_tile_splits,
            )
            if not _dts_guard_allows_append(
                batch_nonempty=bool(batch),
                batch_ge2_groups=batch_ge2_groups,
                batch_max_tiles=batch_max_tiles,
                candidate_tiles=tiles,
                max_dts_partial_rows=max_dts_partial_rows,
            ):
                queued[fi][c] = True
                deferred.append(entry)
                continue

            batch.append((fi, c))
            batch_ge2_groups += 1 if tiles > 0 else 0
            batch_max_tiles = max(batch_max_tiles, tiles)

        for entry in deferred:
            heapq.heappush(ready, entry)

        if live_ready_misses_deadline(wave_idx - 1):
            return None
        if not batch:
            if scheduled_count < total_nonleaves and not any(
                (
                    not scheduled[fi][c]
                    and successors_remaining[fi][c] == 0
                    and int(families[fi]["split_counts"][c]) != 0
                )
                for fi, fam in enumerate(families)
                for c in range(fam["C"])
            ):
                return None
            continue

        reverse_batches.append(batch)
        for fi, c in batch:
            scheduled[fi][c] = True
            scheduled_count += 1
        for fi, c in batch:
            for child in families[fi]["children"][c]:
                if int(families[fi]["split_counts"][child]) == 0:
                    continue
                successors_remaining[fi][child] -= 1
                push_ready(fi, child)

    if scheduled_count != total_nonleaves:
        return None
    return list(reversed(reverse_batches))


def _schedule_coffman_graham_nonleaf_waves(
    families: Sequence[Dict[str, Any]],
    *,
    wave_cap: int,
    max_dts_partial_rows: int | None,
    dts_partial_tile_splits: int,
) -> List[List[Tuple[int, int]]]:
    """Layer the non-leaf DAG with a width cap using a CG-style ordering."""
    labels = [[-1] * fam["C"] for fam in families]
    successors_remaining: List[List[int]] = [
        [0] * fam["C"]
        for fam in families
    ]
    total_nonleaves = 0

    for fi, fam in enumerate(families):
        for c in range(fam["C"]):
            if int(fam["split_counts"][c]) == 0:
                continue
            total_nonleaves += 1
            successors_remaining[fi][c] = sum(
                1
                for parent in fam["parents_of"][c]
                if int(fam["split_counts"][parent]) != 0
            )

    ready: List[Tuple[Tuple[int, ...], int, int, int]] = []
    queued = [[False] * fam["C"] for fam in families]

    def push_ready(fi: int, c: int) -> None:
        if (
            int(families[fi]["split_counts"][c]) == 0
            or labels[fi][c] >= 0
            or queued[fi][c]
            or successors_remaining[fi][c] != 0
        ):
            return
        queued[fi][c] = True
        successor_labels = tuple(
            sorted(
                (
                    labels[fi][parent]
                    for parent in families[fi]["parents_of"][c]
                    if int(families[fi]["split_counts"][parent]) != 0
                ),
                reverse=True,
            )
        )
        heapq.heappush(
            ready,
            (successor_labels, int(families[fi]["priority"][c]), fi, c),
        )

    for fi, fam in enumerate(families):
        for c in range(fam["C"]):
            push_ready(fi, c)

    labeled_count = 0
    label_order: List[Tuple[int, int]] = []
    while ready:
        _successor_labels, _priority, fi, c = heapq.heappop(ready)
        if labels[fi][c] >= 0 or successors_remaining[fi][c] != 0:
            continue
        queued[fi][c] = False
        labels[fi][c] = labeled_count
        labeled_count += 1
        label_order.append((fi, c))

        for child in families[fi]["children"][c]:
            if int(families[fi]["split_counts"][child]) == 0:
                continue
            successors_remaining[fi][child] -= 1
            push_ready(fi, child)

    if labeled_count != total_nonleaves:
        raise RuntimeError(
            "Coffman-Graham scheduler did not label all non-leaf clades: "
            f"labeled={labeled_count}, total={total_nonleaves}"
        )

    assigned_wave = [[-1] * fam["C"] for fam in families]
    waves: List[List[Tuple[int, int]]] = []
    wave_ge2_groups: List[int] = []
    wave_max_tiles: List[int] = []

    def ensure_wave(level: int) -> None:
        while len(waves) <= level:
            waves.append([])
            wave_ge2_groups.append(0)
            wave_max_tiles.append(0)

    for fi, c in reversed(label_order):
        earliest = 0
        for child in families[fi]["children"][c]:
            if int(families[fi]["split_counts"][child]) == 0:
                continue
            child_wave = assigned_wave[fi][child]
            if child_wave < 0:
                raise RuntimeError("CG level assignment visited a parent before its child")
            earliest = max(earliest, child_wave + 1)

        tiles = _clade_dts_partial_tiles(
            families[fi],
            c,
            tile_splits=dts_partial_tile_splits,
        )
        level = earliest
        while True:
            ensure_wave(level)
            if (
                len(waves[level]) < wave_cap
                and _dts_guard_allows_append(
                    batch_nonempty=bool(waves[level]),
                    batch_ge2_groups=wave_ge2_groups[level],
                    batch_max_tiles=wave_max_tiles[level],
                    candidate_tiles=tiles,
                    max_dts_partial_rows=max_dts_partial_rows,
                )
            ):
                waves[level].append((fi, c))
                assigned_wave[fi][c] = level
                wave_ge2_groups[level] += 1 if tiles > 0 else 0
                wave_max_tiles[level] = max(wave_max_tiles[level], tiles)
                break
            level += 1

    return [wave for wave in waves if wave]


def _materialized_nonleaf_wave_count(
    batches: Sequence[Sequence[Tuple[int, int]]],
    families: Sequence[Dict[str, Any]],
    *,
    root_cap: int | None,
) -> int:
    count = 0
    for batch in batches:
        count += len(_split_root_wave(batch, families, root_cap=root_cap))
    return count


def _simple_leaf_first_wave_lower_bound(
    families: Sequence[Dict[str, Any]],
    *,
    wave_cap: int,
) -> int:
    total_leaves = sum(int(fam["leaf_count"]) for fam in families)
    leaf_waves = (total_leaves + wave_cap - 1) // wave_cap
    return leaf_waves + _nonleaf_wave_lower_bound(families, wave_cap=wave_cap)


def _select_nonleaf_schedule_candidate(
    families: Sequence[Dict[str, Any]],
    *,
    wave_cap: int,
    root_cap: int | None,
    max_dts_partial_rows: int | None,
    dts_partial_tile_splits: int,
) -> Tuple[str, List[List[Tuple[int, int]]]]:
    """Return the winning non-leaf scheduler policy and its local batches."""
    forward_batches = _schedule_forward_nonleaf_waves(
        families,
        wave_cap=wave_cap,
        root_cap=root_cap,
        max_dts_partial_rows=max_dts_partial_rows,
        dts_partial_tile_splits=dts_partial_tile_splits,
    )
    best_name = "forward"
    best_batches = forward_batches
    best_nonleaf_count = _materialized_nonleaf_wave_count(
        best_batches,
        families,
        root_cap=root_cap,
    )

    nonleaf_lower_bound = _nonleaf_wave_lower_bound(
        families,
        wave_cap=wave_cap,
    )
    if best_nonleaf_count <= nonleaf_lower_bound:
        return best_name, best_batches

    for target_waves in range(nonleaf_lower_bound, best_nonleaf_count):
        candidate_batches = _schedule_deadline_nonleaf_waves(
            families,
            wave_cap=wave_cap,
            target_waves=target_waves,
            max_dts_partial_rows=max_dts_partial_rows,
            dts_partial_tile_splits=dts_partial_tile_splits,
        )
        if candidate_batches is None:
            continue
        candidate_count = _materialized_nonleaf_wave_count(
            candidate_batches,
            families,
            root_cap=root_cap,
        )
        if candidate_count < best_nonleaf_count:
            best_name = "deadline"
            best_batches = candidate_batches
            best_nonleaf_count = candidate_count
            if best_nonleaf_count <= nonleaf_lower_bound:
                return best_name, best_batches

    for candidate_name, candidate_batches in (
        (
            "reverse_compacted",
            _schedule_reverse_compacted_nonleaf_waves(
                families,
                wave_cap=wave_cap,
                root_cap=root_cap,
                max_dts_partial_rows=max_dts_partial_rows,
                dts_partial_tile_splits=dts_partial_tile_splits,
            ),
        ),
        (
            "coffman_graham",
            _schedule_coffman_graham_nonleaf_waves(
                families,
                wave_cap=wave_cap,
                max_dts_partial_rows=max_dts_partial_rows,
                dts_partial_tile_splits=dts_partial_tile_splits,
            ),
        ),
    ):
        candidate_count = _materialized_nonleaf_wave_count(
            candidate_batches,
            families,
            root_cap=root_cap,
        )
        if candidate_count < best_nonleaf_count:
            best_name = candidate_name
            best_batches = candidate_batches
            best_nonleaf_count = candidate_count

    return best_name, best_batches


def schedule_global_phased_waves(
    items: Sequence[Dict[str, Any]],
    family_clade_offsets: Sequence[int],
    *,
    max_wave_size: int | None,
    max_root_wave_size: int | None = None,
    max_dts_partial_rows: int | None = None,
    dts_partial_tile_splits: int = 64,
) -> Tuple[List[List[int]], List[int]]:
    """Schedule one resident batch with globally packed ready waves.

    The retained kernels still need leaf clades handled before non-leaf clades,
    but after the leaf phase this schedules all ready clades from all families
    into waves capped by ``max_wave_size``.  This replaces the older lockstep
    ``family wave k -> resident wave k`` collation, which left many waves far
    below the GPU-friendly cap.
    """
    if len(items) != len(family_clade_offsets):
        raise ValueError("items and family_clade_offsets must have matching lengths")
    total_clades = sum(int(item["ccp"]["C"]) for item in items)
    if total_clades == 0:
        return [], []
    wave_cap = total_clades if max_wave_size is None else int(max_wave_size)
    if wave_cap <= 0:
        raise ValueError("max_wave_size must be positive")
    root_cap = (
        None
        if max_root_wave_size is None
        else int(max_root_wave_size)
    )
    if root_cap is not None and root_cap <= 0:
        raise ValueError("max_root_wave_size must be positive")
    if max_dts_partial_rows is not None:
        max_dts_partial_rows = int(max_dts_partial_rows)
        if max_dts_partial_rows <= 0:
            raise ValueError("max_dts_partial_rows must be positive when provided")
    dts_partial_tile_splits = int(dts_partial_tile_splits)
    if dts_partial_tile_splits <= 0:
        raise ValueError("dts_partial_tile_splits must be positive")

    families = [_family_schedule_data(item["ccp"]) for item in items]

    waves: List[List[int]] = []
    phases: List[int] = []

    leaf_waves = _leaf_phase_waves(
        families,
        family_clade_offsets,
        wave_cap=wave_cap,
    )
    waves.extend(leaf_waves)
    phases.extend([1] * len(leaf_waves))

    _best_name, best_batches = _select_nonleaf_schedule_candidate(
        families,
        wave_cap=wave_cap,
        root_cap=root_cap,
        max_dts_partial_rows=max_dts_partial_rows,
        dts_partial_tile_splits=dts_partial_tile_splits,
    )

    nonleaf_waves, nonleaf_phases = _materialize_nonleaf_waves(
        best_batches,
        families,
        family_clade_offsets,
        root_cap=root_cap,
    )
    waves.extend(nonleaf_waves)
    phases.extend(nonleaf_phases)

    scheduled_clades = [clade for wave in waves for clade in wave]
    scheduled_count = len(set(scheduled_clades))
    if len(scheduled_clades) != total_clades or scheduled_count != total_clades:
        raise RuntimeError(
            "global wave scheduler did not cover all clades: "
            f"scheduled={scheduled_count}, rows={len(scheduled_clades)}, "
            f"total={total_clades}"
        )
    return waves, phases


def _tensor_from_plan(
    values: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.tensor(list(values), dtype=dtype, device=device).contiguous()


def _build_wave_layout_rust(
    waves: List[List[int]],
    phases: List[int],
    ccp_helpers: Dict[str, Any],
    leaf_row_index: torch.Tensor,
    leaf_col_index: torch.Tensor,
    root_clade_ids: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    family_clade_counts: List[int] | None,
    family_clade_offsets: List[int] | None,
) -> Dict[str, Any]:
    C = int(ccp_helpers["C"])
    N_splits = int(ccp_helpers["N_splits"])
    if C > torch.iinfo(torch.int32).max:
        raise ValueError(f"wave split metadata requires int32 clade ids, got C={C}")
    if "split_parents_sorted" not in ccp_helpers:
        raise RuntimeError("preprocessed CCP helpers must include split_parents_sorted")
    _require_numel(
        "log_split_probs_sorted",
        ccp_helpers["log_split_probs_sorted"],
        N_splits,
    )

    from gpurec.core.schedule_rust import build_wave_layout_plan

    plan = build_wave_layout_plan(
        waves=waves,
        phases=phases,
        c=C,
        n_splits=N_splits,
        split_leftrights_sorted=ccp_helpers["split_leftrights_sorted"],
        split_parents_sorted=ccp_helpers["split_parents_sorted"],
        leaf_row_index=leaf_row_index,
        leaf_col_index=leaf_col_index,
        root_clade_ids=root_clade_ids,
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
    )

    log_split_probs = torch.as_tensor(
        ccp_helpers["log_split_probs_sorted"],
        dtype=dtype,
        device=device,
    )
    index_dtype = torch.int32
    wave_metas: List[Dict[str, Any]] = []
    for meta_plan in plan["wave_metas"]:
        meta: Dict[str, Any] = {
            "start": int(meta_plan["start"]),
            "end": int(meta_plan["end"]),
            "W": int(meta_plan["W"]),
            "has_splits": bool(meta_plan["has_splits"]),
            "phase": int(meta_plan["phase"]),
        }
        if meta["has_splits"]:
            split_indices = _tensor_from_plan(
                meta_plan["split_indices"],
                dtype=torch.long,
                device=device,
            )
            meta["sl"] = _tensor_from_plan(
                meta_plan["sl"],
                dtype=index_dtype,
                device=device,
            )
            meta["sr"] = _tensor_from_plan(
                meta_plan["sr"],
                dtype=index_dtype,
                device=device,
            )
            meta["log_split_probs"] = log_split_probs[
                split_indices
            ].unsqueeze(1).contiguous()
            reduce_idx = _tensor_from_plan(
                meta_plan["reduce_idx"],
                dtype=index_dtype,
                device=device,
            )
            meta["reduce_idx"] = reduce_idx
            meta["n_eq1"] = int(meta_plan["n_eq1"])
            if "eq1_reduce_idx" in meta_plan:
                meta["eq1_reduce_idx"] = _tensor_from_plan(
                    meta_plan["eq1_reduce_idx"],
                    dtype=index_dtype,
                    device=device,
                )
            if "ge2_ptr" in meta_plan:
                meta["ge2_ptr"] = _tensor_from_plan(
                    meta_plan["ge2_ptr"],
                    dtype=torch.long,
                    device=device,
                )
                meta["ge2_parent_ids"] = _tensor_from_plan(
                    meta_plan["ge2_parent_ids"],
                    dtype=index_dtype,
                    device=device,
                )
                meta["ge2_max_fanout"] = int(meta_plan["ge2_max_fanout"])
        wave_metas.append(meta)

    result = {
        "perm": _tensor_from_plan(plan["perm"], dtype=torch.long, device=device),
        "C": int(plan["c"]),
        "leaf_row_index": _tensor_from_plan(
            plan["leaf_row_index"],
            dtype=torch.long,
            device=device,
        ),
        "leaf_species_index": _tensor_from_plan(
            plan["leaf_species_index"],
            dtype=torch.long,
            device=device,
        ),
        "root_clade_ids": _tensor_from_plan(
            plan["root_clade_ids"],
            dtype=torch.long,
            device=device,
        ),
        "root_clade_ids_cpu": [int(value) for value in plan["root_clade_ids_cpu"]],
        "wave_metas": wave_metas,
    }
    if "family_idx" in plan:
        result["family_idx"] = _tensor_from_plan(
            plan["family_idx"],
            dtype=torch.long,
            device=device,
        )
    return result


def build_wave_layout(
    waves: List[List[int]],
    phases: List[int],
    ccp_helpers: Dict[str, Any],
    leaf_row_index: torch.Tensor,
    leaf_col_index: torch.Tensor,
    root_clade_ids: torch.Tensor,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    family_clade_counts: List[int] | None = None,
    family_clade_offsets: List[int] | None = None,
) -> Dict[str, Any]:
    """Build wave-ordered layout: permute clade indices so each wave is contiguous.

    After this, wave k's clades occupy Pi[wave_starts[k] : wave_starts[k+1]]
    as a contiguous block, eliminating gather/scatter in the self-loop.

    Args:
        waves: list of lists of global clade IDs per wave
        phases: phase label per wave (1=leaf, 2=internal, 3=root)
        ccp_helpers: merged CCP dict (from collate_gene_families)
        leaf_row_index: Long[K] clade indices for leaf mapping
        leaf_col_index: Long[K] species indices for leaf mapping
        root_clade_ids: Long[F] root clade indices
        device: target device
        dtype: float dtype
        family_clade_counts: per-family clade counts [G] (for family_idx)
        family_clade_offsets: per-family clade offsets [G] (for family_idx)

    Returns:
        Dict with:
          'perm': Long[C] — original-to-new mapping
          'C': total clade count
          'leaf_row_index': remapped leaf row indices
          'leaf_species_index': species id per leaf clade, -1 otherwise
          'root_clade_ids': remapped root clade IDs
          'wave_metas': list of per-wave metadata dicts
          'family_idx': Long[C] clade→family (only if family_clade_counts provided)
    """
    backend = os.environ.get(_SCHEDULER_BACKEND_ENV, "python").strip().lower()
    if backend == "rust":
        return _build_wave_layout_rust(
            waves,
            phases,
            ccp_helpers,
            leaf_row_index,
            leaf_col_index,
            root_clade_ids,
            device,
            dtype,
            family_clade_counts,
            family_clade_offsets,
        )
    if backend not in {"", "python", "py"}:
        raise ValueError(
            f"{_SCHEDULER_BACKEND_ENV} must be 'python' or 'rust', got {backend!r}"
        )

    C = int(ccp_helpers['C'])
    N_splits = int(ccp_helpers['N_splits'])
    family_clade_ranges = _validate_family_clade_metadata(
        family_clade_counts,
        family_clade_offsets,
        C,
    )
    if len(phases) != len(waves):
        raise ValueError(
            "waves and phases must have matching lengths, "
            f"got {len(waves)} and {len(phases)}"
        )
    if C > torch.iinfo(torch.int32).max:
        raise ValueError(f"wave split metadata requires int32 clade ids, got C={C}")
    _require_numel(
        "split_leftrights_sorted",
        ccp_helpers['split_leftrights_sorted'],
        2 * N_splits,
    )
    _require_numel(
        "log_split_probs_sorted",
        ccp_helpers['log_split_probs_sorted'],
        N_splits,
    )
    if 'split_parents_sorted' not in ccp_helpers:
        raise RuntimeError("preprocessed CCP helpers must include split_parents_sorted")
    _require_numel(
        "split_parents_sorted",
        ccp_helpers['split_parents_sorted'],
        N_splits,
    )
    _validate_clade_id_values(
        "split_leftrights_sorted",
        ccp_helpers['split_leftrights_sorted'],
        C=C,
    )
    _validate_clade_id_values(
        "split_parents_sorted",
        ccp_helpers['split_parents_sorted'],
        C=C,
    )
    _require_numel(
        "leaf_col_index",
        leaf_col_index,
        _numel(leaf_row_index),
    )
    _validate_clade_id_values("leaf_row_index", leaf_row_index, C=C)
    _validate_clade_id_values("root_clade_ids", root_clade_ids, C=C)

    # --- 2a. Build permutation ---
    all_clades: List[int] = []
    wave_starts_list: List[int] = [0]
    for wave_ids in waves:
        all_clades.extend(wave_ids)
        wave_starts_list.append(len(all_clades))

    _validate_wave_clade_coverage(all_clades, C)

    inv_perm = torch.tensor(all_clades, dtype=torch.long, device=device)
    perm = torch.empty(C, dtype=torch.long, device=device)
    perm[inv_perm] = torch.arange(C, dtype=torch.long, device=device)

    # --- 2b. Remap all clade-index tensors (fully vectorized) ---
    split_lr = ccp_helpers['split_leftrights_sorted'].to(device=device, dtype=torch.long)
    lefts_orig = split_lr[:N_splits]
    rights_orig = split_lr[N_splits:]
    lefts_new = perm[lefts_orig]
    rights_new = perm[rights_orig]

    split_parents = ccp_helpers['split_parents_sorted']
    sp_new = perm[split_parents.to(device=device, dtype=torch.long)]

    leaf_row_new = perm[leaf_row_index.to(device=device, dtype=torch.long)]
    root_ids_new = perm[root_clade_ids.to(device=device, dtype=torch.long)]
    root_ids_new_cpu = [int(x) for x in root_ids_new.detach().cpu().tolist()]
    leaf_col_new = leaf_col_index.to(device=device, dtype=torch.long)
    leaf_species_index = torch.full((C,), -1, dtype=torch.long, device=device)
    if leaf_row_new.numel() > 0:
        leaf_species_index[leaf_row_new] = leaf_col_new

    log_split_probs = ccp_helpers['log_split_probs_sorted']
    if torch.is_tensor(log_split_probs):
        log_split_probs = log_split_probs.to(device=device, dtype=dtype)

    # --- 2c. Vectorized per-wave metadata ---
    # For each split, find which wave its parent belongs to via searchsorted
    # sp_new[i] is the new-space parent clade of split i, which is in [0, C)
    # wave_starts is sorted → searchsorted gives the wave index
    wave_starts_cpu = torch.tensor(wave_starts_list, dtype=torch.long)
    sp_new_cpu = sp_new.cpu()

    # searchsorted: find wave index for each split's parent
    # wave_starts_list = [0, w0_end, w1_end, ...]. searchsorted(right) - 1 gives wave idx.
    split_wave_idx = torch.searchsorted(wave_starts_cpu[1:], sp_new_cpu, right=True)
    # split_wave_idx[i] = wave index of split i's parent

    # Sort splits by wave index for efficient slicing
    sort_order = split_wave_idx.argsort()
    split_wave_sorted = split_wave_idx[sort_order]

    # Find boundaries: where does each wave's splits start/end in the sorted order
    n_waves = len(waves)
    # Use searchsorted on the sorted wave indices
    wave_split_starts = torch.searchsorted(split_wave_sorted, torch.arange(n_waves, dtype=torch.long))
    wave_split_ends = torch.searchsorted(split_wave_sorted, torch.arange(n_waves, dtype=torch.long), right=True)

    # Move sort_order to device for indexing
    sort_order_dev = sort_order.to(device)

    wave_metas: List[Dict[str, Any]] = []
    for wi in range(n_waves):
        ws = wave_starts_list[wi]
        we = wave_starts_list[wi + 1]
        W = we - ws

        ss = int(wave_split_starts[wi].item())
        se = int(wave_split_ends[wi].item())
        n_ws = se - ss

        meta: Dict[str, Any] = {
            'start': ws,
            'end': we,
            'W': W,
            'has_splits': n_ws > 0,
            'phase': phases[wi],
        }

        if n_ws > 0:
            wst = sort_order_dev[ss:se]  # split indices for this wave
            reduce_idx = sp_new[wst] - ws  # [n_ws] wave-local clade index

            # Sort splits: single-split clades first, then multi-split clades.
            clade_split_counts = torch.zeros(W, dtype=torch.long, device=device)
            clade_split_counts.scatter_add_(0, reduce_idx,
                                            torch.ones(n_ws, dtype=torch.long, device=device))
            # Per-split: count for the parent clade of that split
            per_split_count = clade_split_counts[reduce_idx]  # [n_ws]
            # Composite sort key: eq1 first (is_ge2=0), ge2 after (is_ge2=1),
            # within ge2 sorted by parent clade (ascending) for CSR contiguity.
            sort_key = (per_split_count > 1).long() * (W + 1) + reduce_idx
            inner_order = sort_key.argsort(stable=True)
            wst = wst[inner_order]
            reduce_idx = reduce_idx[inner_order]

            n_eq1 = int((per_split_count == 1).sum().item())
            n_ge2_clades = int((clade_split_counts >= 2).sum().item())

            index_dtype = torch.int32
            sl_index = lefts_new[wst].to(index_dtype).contiguous()
            sr_index = rights_new[wst].to(index_dtype).contiguous()
            reduce_idx_index = reduce_idx.to(index_dtype).contiguous()

            meta['sl'] = sl_index
            meta['sr'] = sr_index
            meta['log_split_probs'] = log_split_probs[wst].unsqueeze(1).contiguous()
            meta['reduce_idx'] = reduce_idx_index
            meta['n_eq1'] = n_eq1

            if n_eq1 > 0:
                meta['eq1_reduce_idx'] = reduce_idx_index[:n_eq1]

            if n_ge2_clades > 0:
                # Build CSR pointers for the ge2 portion (splits n_eq1:).
                # Splits are sorted by parent clade, so same-parent splits are contiguous.
                ge2_reduce = reduce_idx[n_eq1:]  # [n_ge2_splits]
                # Unique parent clades in order of first appearance (= ascending,
                # since we sorted by clade index)
                ge2_parent_ids, ge2_counts = ge2_reduce.unique_consecutive(return_counts=True)
                ge2_ptr = torch.zeros(len(ge2_parent_ids) + 1, dtype=torch.long, device=device)
                torch.cumsum(ge2_counts, dim=0, out=ge2_ptr[1:])

                meta['ge2_ptr'] = ge2_ptr
                meta['ge2_parent_ids'] = ge2_parent_ids.to(index_dtype).contiguous()  # wave-local clade indices
                meta['ge2_max_fanout'] = int(ge2_counts.max().item())

        wave_metas.append(meta)

    result = {
        'perm': perm,
        'C': C,
        'leaf_row_index': leaf_row_new,
        'leaf_species_index': leaf_species_index,
        'root_clade_ids': root_ids_new,
        'root_clade_ids_cpu': root_ids_new_cpu,
        'wave_metas': wave_metas,
    }

    # Build clade→family mapping in wave-ordered space
    if family_clade_ranges is not None:
        family_idx_orig = torch.full((C,), -1, dtype=torch.long, device=device)
        for g, (offset, c_g) in enumerate(family_clade_ranges):
            family_idx_orig[offset:offset + c_g] = g
        # Permute to wave-ordered space: result[new_idx] = family of original clade inv_perm[new_idx]
        result['family_idx'] = family_idx_orig[inv_perm]

    return result
