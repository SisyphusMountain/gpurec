import json
from bisect import bisect_right

import torch

from gpurec.core.native import load_native_module

_NATIVE_MODULE = "gpurec_preprocess"
_NATIVE_LIBRARY = "libgpurec_preprocess.so"
_EXPECTED_SCHEMA_VERSION = 1


def _normalize_batch_packing(batch_packing: str, clade_budget: int | None) -> str:
    if clade_budget is None and batch_packing in {"depth_first_fit", "clade_first_fit"}:
        return "sequential"
    return batch_packing


def _load_native_module():
    return load_native_module(_NATIVE_MODULE, _NATIVE_LIBRARY)


def _validate_schema_version(raw: dict, *, payload_name: str) -> None:
    version = raw.get("schema_version")
    if version != _EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {payload_name} schema_version {version!r}; "
            f"expected {_EXPECTED_SCHEMA_VERSION}"
        )


def preprocess_dataset(
    species_path: str,
    families,
    *,
    family_chunk_size: int | None = 300,
    clade_budget: int | None = 315_000,
    batch_packing: str = "depth_first_fit",
    max_wave_size: int = 8192,
    family_group_assignments: list[int] | None = None,
):
    batch_packing = _normalize_batch_packing(batch_packing, clade_budget)
    raw = json.loads(
        _load_native_module().preprocess_dataset(
            str(species_path),
            [str(path) for path in families],
            family_chunk_size,
            clade_budget,
            batch_packing,
            max_wave_size,
            [int(label) for label in family_group_assignments]
            if family_group_assignments is not None
            else None,
        )
    )
    _validate_schema_version(raw, payload_name="preprocess_dataset")
    species = raw["species"]
    species["unnorm_row_max"] = torch.tensor(species["unnorm_row_max"], dtype=torch.float32)
    for key in (
        "sp_child1", "sp_child2", "sp_parent", "sp_subtree_start", "sp_subtree_end",
        "compact_level_parents", "compact_level_child1", "compact_level_child2",
    ):
        species[key] = torch.tensor(species[key], dtype=torch.int32)
    species["compact_level_ptr"] = torch.tensor(species["compact_level_ptr"], dtype=torch.int64)

    raw["batches"] = [[int(index) for index in batch] for batch in raw.get("batches", [])]
    return raw


def build_wave_layout_from_plan(payload, *, device: torch.device | str):
    plan = payload["plan"]
    logp = [float(value) for value in payload["log_split_probs_sorted"]]
    index_dtype = torch.int32
    wave_metas = []
    for raw_meta in plan["wave_metas"]:
        start = int(raw_meta["start"])
        width = int(raw_meta["W"])
        meta = {
            "start": start,
            "end": int(raw_meta.get("end", start + width)),
            "W": width,
            "has_splits": bool(raw_meta.get("has_splits", False)),
            "phase": int(raw_meta.get("phase", 2 if raw_meta.get("has_splits") else 1)),
        }
        if raw_meta.get("has_splits"):
            split_indices = [int(idx) for idx in raw_meta.get("split_indices", [])]
            meta["sl"] = torch.tensor(raw_meta["sl"], dtype=index_dtype, device=device).contiguous()
            meta["sr"] = torch.tensor(raw_meta["sr"], dtype=index_dtype, device=device).contiguous()
            meta["reduce_idx"] = torch.tensor(raw_meta["reduce_idx"], dtype=index_dtype, device=device).contiguous()
            meta["log_split_probs"] = torch.tensor(
                [logp[idx] for idx in split_indices],
                dtype=torch.float32,
                device=device,
            ).contiguous()
            meta["n_eq1"] = int(raw_meta.get("n_eq1", 0))
            if raw_meta.get("eq1_reduce_idx"):
                meta["eq1_reduce_idx"] = torch.tensor(
                    raw_meta["eq1_reduce_idx"],
                    dtype=index_dtype,
                    device=device,
                ).contiguous()
            if raw_meta.get("ge2_ptr"):
                meta["ge2_ptr"] = torch.tensor(raw_meta["ge2_ptr"], dtype=torch.long, device=device).contiguous()
                meta["ge2_parent_ids"] = torch.tensor(
                    raw_meta["ge2_parent_ids"],
                    dtype=index_dtype,
                    device=device,
                ).contiguous()
                meta["ge2_max_fanout"] = int(raw_meta["ge2_max_fanout"])
        wave_metas.append(meta)

    return {
        "perm": torch.tensor(plan["perm"], dtype=torch.long, device=device).contiguous(),
        "leaf_species_index": torch.tensor(
            plan["leaf_species_index"],
            dtype=index_dtype,
            device=device,
        ).contiguous(),
        "root_clade_ids": torch.tensor(
            plan["root_clade_ids"],
            dtype=torch.long,
            device=device,
        ).contiguous(),
        "wave_metas": wave_metas,
        "family_idx": torch.tensor(
            plan.get("family_idx", [0] * int(plan["c"])),
            dtype=torch.long,
            device=device,
        ).contiguous(),
    }


def plan_batch_wave_layouts(
    families,
    *,
    family_chunk_size: int | None = 300,
    clade_budget: int | None = 315_000,
    batch_packing: str = "depth_first_fit",
    max_wave_size: int = 8192,
    family_group_assignments: list[int] | None = None,
):
    batch_packing = _normalize_batch_packing(batch_packing, clade_budget)
    raw = json.loads(
        _load_native_module().plan_batch_layouts(
            json.dumps(families),
            family_chunk_size,
            clade_budget,
            batch_packing,
            max_wave_size,
            [int(label) for label in family_group_assignments]
            if family_group_assignments is not None
            else None,
        )
    )
    _validate_schema_version(raw, payload_name="plan_batch_layouts")
    raw["batches"] = [[int(index) for index in batch] for batch in raw["batches"]]
    return raw


def _split_rows_for_family(family):
    C = int(family["C"])
    leaf_rows = [int(row) for row in family["leaf_row_index"]]
    if "split_parents_sorted" in family:
        n_splits = int(family.get("N_splits", len(family["split_parents_sorted"])))
        leftrights = family["split_leftrights_sorted"]
        parents = [int(parent) for parent in family["split_parents_sorted"]]
        lefts = [int(value) for value in leftrights[:n_splits]]
        rights = [int(value) for value in leftrights[n_splits:]]
        logp = [float(value) for value in family["log_split_probs_sorted"]]
    else:
        leftrights = family["split_leftrights_sorted"]
        leaf_set = set(leaf_rows)
        parents = [clade for clade in range(C) if clade not in leaf_set]
        lefts = [int(value) for value in leftrights[: len(parents)]]
        rights = [int(value) for value in leftrights[len(parents) :]]
        logp = [0.0] * len(parents)
    return leaf_rows, parents, lefts, rights, logp


def build_wave_layout(families, *, device: torch.device | str, max_wave_size: int = 8192):
    offsets, levels_by_family, leaf_pairs = [], [], []
    split_parents_global, lefts_global, rights_global, logp_global = [], [], [], []
    root_ids_global = []

    clade_offset = 0
    for family in families:
        C = int(family["C"])
        leaf_rows, parents, lefts, rights, logp = _split_rows_for_family(family)
        leaf_set = set(leaf_rows)
        levels = [0 if clade in leaf_set else -1 for clade in range(C)]
        for _ in range(C):
            changed = False
            for parent, left, right in zip(parents, lefts, rights):
                child_level = max(levels[left], levels[right])
                if child_level >= 0 and levels[parent] < child_level + 1:
                    levels[parent] = child_level + 1
                    changed = True
            if not changed:
                break
        levels = [level if level >= 0 else 1 for level in levels]

        offsets.append(clade_offset)
        levels_by_family.append(levels)
        split_parents_global.extend(parent + clade_offset for parent in parents)
        lefts_global.extend(left + clade_offset for left in lefts)
        rights_global.extend(right + clade_offset for right in rights)
        logp_global.extend(logp)
        leaf_pairs.extend(
            (row + clade_offset, species)
            for row, species in zip(leaf_rows, family["leaf_col_index"])
        )
        root_ids_global.append(int(family.get("root_clade_id", C - 1)) + clade_offset)
        clade_offset += C

    waves = [
        [row for row, _ in leaf_pairs[start : start + max_wave_size]]
        for start in range(0, len(leaf_pairs), max_wave_size)
    ]
    max_level = max((max(levels) for levels in levels_by_family), default=0)
    for level in range(1, max_level + 1):
        clades = [
            offsets[fi] + clade
            for fi, levels in enumerate(levels_by_family)
            for clade, clade_level in enumerate(levels)
            if clade_level == level
        ]
        for start in range(0, len(clades), max_wave_size):
            waves.append(clades[start : start + max_wave_size])

    C = clade_offset
    all_clades = [clade for wave in waves for clade in wave]
    wave_starts = [0]
    for wave in waves:
        wave_starts.append(wave_starts[-1] + len(wave))

    perm = [0] * C
    for new_idx, original in enumerate(all_clades):
        perm[original] = new_idx

    lefts_new = [perm[clade] for clade in lefts_global]
    rights_new = [perm[clade] for clade in rights_global]
    parents_new = [perm[parent] for parent in split_parents_global]
    wave_ends = wave_starts[1:]
    split_order = sorted(
        (bisect_right(wave_ends, parents_new[idx]), idx)
        for idx in range(len(split_parents_global))
    )

    index_dtype = torch.int32
    lefts_t = torch.tensor(lefts_new, dtype=index_dtype, device=device)
    rights_t = torch.tensor(rights_new, dtype=index_dtype, device=device)
    parents_t = torch.tensor(parents_new, dtype=index_dtype, device=device)
    logp_t = torch.tensor(logp_global, dtype=torch.float32, device=device)
    wave_metas = []
    order_head = 0
    for wave_idx, wave in enumerate(waves):
        start = wave_starts[wave_idx]
        split_start = order_head
        while order_head < len(split_order) and split_order[order_head][0] == wave_idx:
            order_head += 1
        split_indices = [idx for _, idx in split_order[split_start:order_head]]
        meta = {"start": start, "W": len(wave)}
        if split_indices:
            split_indices.sort(key=lambda idx: (parents_new[idx] - start, idx))
            counts = {}
            for idx in split_indices:
                reduce = parents_new[idx] - start
                counts[reduce] = counts.get(reduce, 0) + 1
            split_indices.sort(key=lambda idx: ((counts[parents_new[idx] - start] > 1), parents_new[idx] - start, idx))
            n_eq1 = sum(1 for idx in split_indices if counts[parents_new[idx] - start] == 1)
            split_tensor = torch.tensor(split_indices, dtype=torch.long, device=device)
            meta["sl"] = lefts_t[split_tensor].contiguous()
            meta["sr"] = rights_t[split_tensor].contiguous()
            reduce_idx = (parents_t[split_tensor] - start).contiguous()
            meta["reduce_idx"] = reduce_idx
            meta["log_split_probs"] = logp_t[split_tensor].contiguous()
            meta["n_eq1"] = n_eq1
            if n_eq1:
                meta["eq1_reduce_idx"] = reduce_idx[:n_eq1].contiguous()
            ge2_reduce = reduce_idx[n_eq1:].detach().cpu().tolist()
            if ge2_reduce:
                ge2_parent_ids = []
                ge2_counts = []
                for reduce in ge2_reduce:
                    if ge2_parent_ids and ge2_parent_ids[-1] == reduce:
                        ge2_counts[-1] += 1
                    else:
                        ge2_parent_ids.append(reduce)
                        ge2_counts.append(1)
                ptr = [0]
                for count in ge2_counts:
                    ptr.append(ptr[-1] + count)
                meta["ge2_ptr"] = torch.tensor(ptr, dtype=torch.long, device=device).contiguous()
                meta["ge2_parent_ids"] = torch.tensor(ge2_parent_ids, dtype=index_dtype, device=device).contiguous()
                meta["ge2_max_fanout"] = max(ge2_counts)
        wave_metas.append(meta)

    leaf_species = [-1] * C
    for row, species in leaf_pairs:
        leaf_species[perm[row]] = species

    return {
        "perm": torch.tensor(perm, dtype=torch.long, device=device).contiguous(),
        "leaf_species_index": torch.tensor(leaf_species, dtype=torch.int32, device=device).contiguous(),
        "root_clade_ids": torch.tensor(
            [perm[root] for root in root_ids_global],
            dtype=torch.long,
            device=device,
        ).contiguous(),
        "wave_metas": wave_metas,
        "family_idx": torch.tensor(
            [bisect_right(offsets, clade) - 1 for clade in all_clades],
            dtype=torch.long,
            device=device,
        ).contiguous(),
    }


def plan_family_batches(
    families,
    *,
    family_chunk_size: int | None = 300,
    clade_budget: int | None = 315_000,
    batch_packing: str = "depth_first_fit",
):
    family_cap = int(family_chunk_size or 0)
    clade_cap = None if clade_budget is None else int(clade_budget)
    items = [
        (
            idx,
            int(family["C"]),
            int(family.get("schedule_depth", 0)),
        )
        for idx, family in enumerate(families)
    ]
    if batch_packing == "depth_first_fit":
        items.sort(key=lambda item: (-item[2], -item[1], item[0]))
    elif batch_packing == "clade_first_fit":
        items.sort(key=lambda item: (-item[1], -item[2], item[0]))

    if batch_packing == "sequential":
        batches, current, current_clades = [], [], 0
        for idx, clades, _depth in items:
            over_family = family_cap and len(current) >= family_cap
            over_clades = current and clade_cap is not None and current_clades + clades > clade_cap
            if over_family or over_clades:
                batches.append(current)
                current, current_clades = [], 0
            current.append(idx)
            current_clades += clades
        if current:
            batches.append(current)
        return batches

    batches: list[list] = []
    for idx, clades, _depth in items:
        for batch in batches:
            indices, used = batch
            family_room = not family_cap or len(indices) < family_cap
            clade_room = clade_cap is None or used + clades <= clade_cap or not indices
            if family_room and clade_room:
                indices.append(idx)
                batch[1] = used + clades
                break
        else:
            batches.append([[idx], clades])
    return [sorted(indices) for indices, _used in batches]
