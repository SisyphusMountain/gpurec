import json
import importlib.util
import sys
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import torch

_NATIVE_LIBRARY_PATH = Path(__file__).resolve().parents[3] / "crates/gpurec-preprocess/target/release/libgpurec_preprocess.so"


@lru_cache(maxsize=1)
def _load_native_module():
    spec = importlib.util.spec_from_file_location("gpurec_preprocess", str(_NATIVE_LIBRARY_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpurec_preprocess"] = module
    spec.loader.exec_module(module)
    return module


def preprocess_dataset(species_path: str, families):
    raw = json.loads(_load_native_module().preprocess_dataset(str(species_path), [str(path) for path in families]))
    species = raw["species"]
    species["unnorm_row_max"] = torch.tensor(species["unnorm_row_max"], dtype=torch.float32)
    for key in (
        "sp_child1", "sp_child2", "sp_parent", "sp_subtree_start", "sp_subtree_end",
        "compact_level_parents", "compact_level_child1", "compact_level_child2",
    ):
        species[key] = torch.tensor(species[key], dtype=torch.int32)
    species["compact_level_ptr"] = torch.tensor(species["compact_level_ptr"], dtype=torch.int64)

    return raw


def build_wave_layout(families, *, device: torch.device | str):
    offsets, levels_by_family, leaf_pairs = [], [], []
    split_parents_global, lefts_global, rights_global = [], [], []

    clade_offset = 0
    for family in families:
        C = int(family["C"])
        leftrights = family["split_leftrights_sorted"]
        leaf_rows = family["leaf_row_index"]
        leaf_set = set(leaf_rows)
        parents = [clade for clade in range(C) if clade not in leaf_set]
        lefts = leftrights[: len(parents)]
        rights = leftrights[len(parents) :]
        levels = [0] * C
        for parent, left, right in zip(parents, lefts, rights):
            levels[parent] = max(levels[parent], levels[left] + 1, levels[right] + 1)

        offsets.append(clade_offset)
        levels_by_family.append(levels)
        split_parents_global.extend(parent + clade_offset for parent in parents)
        lefts_global.extend(left + clade_offset for left in lefts)
        rights_global.extend(right + clade_offset for right in rights)
        leaf_pairs.extend(
            (row + clade_offset, species)
            for row, species in zip(leaf_rows, family["leaf_col_index"])
        )
        clade_offset += C

    waves = [
        [row for row, _ in leaf_pairs[start : start + 8192]]
        for start in range(0, len(leaf_pairs), 8192)
    ]
    max_level = max((max(levels) for levels in levels_by_family), default=0)
    for level in range(1, max_level + 1):
        clades = [
            offsets[fi] + clade
            for fi, levels in enumerate(levels_by_family)
            for clade, clade_level in enumerate(levels)
            if clade_level == level
        ]
        for start in range(0, len(clades), 8192):
            waves.append(clades[start : start + 8192])

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
            split_tensor = torch.tensor(split_indices, dtype=torch.long, device=device)
            meta["sl"] = lefts_t[split_tensor].contiguous()
            meta["sr"] = rights_t[split_tensor].contiguous()
            meta["reduce_idx"] = (parents_t[split_tensor] - start).contiguous()
        wave_metas.append(meta)

    leaf_species = [-1] * C
    for row, species in leaf_pairs:
        leaf_species[perm[row]] = species

    return {
        "perm": torch.tensor(perm, dtype=torch.long, device=device).contiguous(),
        "leaf_species_index": torch.tensor(leaf_species, dtype=torch.int32, device=device).contiguous(),
        "root_clade_ids": torch.tensor(
            [perm[(offsets[idx + 1] if idx + 1 < len(offsets) else C) - 1] for idx in range(len(offsets))],
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
