import json
from bisect import bisect_right
from collections.abc import Mapping

import numpy as np
import torch

from gpurec.config.precision import PrecisionOptions, resolve_torch_dtype, torch_dtype_name
from gpurec.core.native import load_native_module

_NATIVE_MODULE = "gpurec_preprocess"
_NATIVE_LIBRARY = "libgpurec_preprocess.so"
_EXPECTED_SCHEMA_VERSION = 1


def _resolve_accumulator_dtype(accumulator_dtype: torch.dtype | str | None) -> torch.dtype:
    """Resolve the precision-policy default for standalone scheduling helpers."""
    if accumulator_dtype is None:
        return PrecisionOptions().accumulator_torch_dtype
    if isinstance(accumulator_dtype, str):
        return resolve_torch_dtype(accumulator_dtype)
    # Validate tensor dtypes at this boundary instead of failing later in a
    # kernel launcher with a less useful error.
    torch_dtype_name(accumulator_dtype)
    return accumulator_dtype


def _resolve_model_dtype(model_dtype: torch.dtype | str | None) -> torch.dtype:
    """Resolve the dense-kernel dtype for standalone scheduling helpers."""
    if model_dtype is None:
        return PrecisionOptions().model_torch_dtype
    if isinstance(model_dtype, str):
        return resolve_torch_dtype(model_dtype)
    torch_dtype_name(model_dtype)
    return model_dtype


def _normalize_batch_packing(batch_packing: str, clade_budget: int | None) -> str:
    if clade_budget is None and batch_packing in {"depth_first_fit", "clade_first_fit"}:
        return "sequential"
    return batch_packing


# numpy dtype for each torch index dtype the wave layout builds. The payload arrives either as
# numpy int64 arrays (ParsedFamilies / preprocess_from_parsed) or as Python lists (the legacy JSON
# path); narrowing in numpy first keeps the host->device copy at the tensor's final width.
_NUMPY_FOR_TORCH_INDEX = {torch.int32: np.int32, torch.int64: np.int64}

# Per-batch clade budget the depth-first packer is tuned for, and the ceiling the device-derived
# budget in gpurec/core/memory_policy.clade_budget_for_device is never allowed to exceed. Single
# source for the signature defaults below and for GeneReconModel's.
DEFAULT_CLADE_BUDGET = 315_000


def _index_tensor(values, *, dtype: torch.dtype, device) -> torch.Tensor:
    """``torch.tensor(values, dtype=dtype, device=device)`` without a per-element Python loop.

    Accepts a numpy array or a plain Python sequence and produces exactly the tensor the
    element-by-element construction produced before: same dtype, shape, values and contiguity.
    """
    array = np.ascontiguousarray(np.asarray(values, dtype=_NUMPY_FOR_TORCH_INDEX[dtype]))
    return torch.from_numpy(array).to(device=device).contiguous()


def _float64_array(values) -> np.ndarray:
    """The payload's f64 values as a contiguous float64 numpy array (no copy when already one)."""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


# The two model dtypes a wave's split log-probabilities can be handed to a kernel as. Not a
# setting: gpurec/config/precision.py admits exactly these two model dtypes.
_SPLIT_PROB_DTYPES = (torch.float32, torch.float64)


class _LazySplitProbabilities(Mapping):
    """A wave's split log-probabilities, put on the DEVICE only for the dtypes actually asked for.

    The payload stores these values as f64. Both device variants used to be built up front, so that
    a later ``model.to(...)`` or an explicit runtime dtype could never be served a widened fp32
    copy. Both are still available and still built from the SAME f64 source -- no precision is lost
    -- but the device tensor for a dtype is only created when that dtype is first requested, and
    cached from then on. An fp32 model therefore never materializes the fp64 variant, which nothing
    reads: on the 5123-family Coleman set (60.42M splits) that copy is 483 MiB of device memory,
    a fifth of the 2.22 GiB of per-batch statics that stay resident for the whole fit.

    The f64 source stays on the HOST, where the payload already put it, so this trades device bytes
    for host bytes -- 483 MiB of host RAM at Coleman scale, against 96 GiB of job memory.
    """

    def __init__(self, host_values, device):
        self._host = host_values          # torch.float64 CPU tensor: the payload's own values
        self._device = device
        self._device_tensors = {}

    def __getitem__(self, dtype):
        if dtype not in _SPLIT_PROB_DTYPES:
            raise KeyError(dtype)
        tensor = self._device_tensors.get(dtype)
        if tensor is None:
            tensor = self._host.to(dtype=dtype, device=self._device).contiguous()
            self._device_tensors[dtype] = tensor
        return tensor

    def __iter__(self):
        return iter(_SPLIT_PROB_DTYPES)

    def __len__(self):
        return len(_SPLIT_PROB_DTYPES)


def _materialize_split_probabilities(values, *, device) -> Mapping:
    """The wave's split log-probabilities as a lazy {dtype: device tensor} view of the f64 payload."""
    return _LazySplitProbabilities(torch.from_numpy(_float64_array(values)), device)


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
    clade_budget: int | None = DEFAULT_CLADE_BUDGET,
    batch_packing: str = "depth_first_fit",
    max_wave_size: int = 8192,
    family_group_assignments: list[int] | None = None,
    accumulator_dtype: torch.dtype | str | None = None,
):
    accumulator_dtype = _resolve_accumulator_dtype(accumulator_dtype)
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
    raw["species"] = _species_tensors(raw["species"], accumulator_dtype=accumulator_dtype)
    raw["batches"] = [[int(index) for index in batch] for batch in raw.get("batches", [])]
    return raw


_SPECIES_INT32_KEYS = (
    "sp_child1", "sp_child2", "sp_parent", "sp_subtree_start", "sp_subtree_end",
    "compact_level_parents", "compact_level_child1", "compact_level_child2",
)


def _species_tensors(species: dict, *, accumulator_dtype: torch.dtype) -> dict:
    """Convert the species payload's arrays to the tensors the model consumes (in place)."""
    # The Rust preprocessor returns f64 row maxima. Preserve them at the configured
    # accumulation precision because this row maximum participates in the transfer softmax.
    species["unnorm_row_max"] = torch.tensor(
        species["unnorm_row_max"], dtype=accumulator_dtype
    )
    for key in _SPECIES_INT32_KEYS:
        species[key] = torch.tensor(species[key], dtype=torch.int32)
    species["compact_level_ptr"] = torch.tensor(species["compact_level_ptr"], dtype=torch.int64)
    species["sp_height"] = _species_heights(species)
    return species


def _species_heights(species: dict) -> torch.Tensor:
    """Each species node's height: 0 at a leaf, 1 + the taller child's height above.

    The same number the preprocessor used to bucket internal nodes into ``compact_level_*``, read
    back off those tables: bucket ``i`` holds exactly the nodes of height ``i + 1``, and every node
    absent from them is a leaf. Kernels that walk the tree level by level over a whole species row
    at once -- rather than over one level's node list -- need the height per SPECIES to know which
    lanes each level updates; see ``_solve_reconciliation_self_loop_jvp_exact_kernel``.
    """
    height = torch.zeros(int(species["S"]), dtype=torch.int32)
    level_ptr = species["compact_level_ptr"]
    parents = species["compact_level_parents"]
    for level in range(int(level_ptr.numel()) - 1):
        start = int(level_ptr[level])
        end = int(level_ptr[level + 1])
        height[parents[start:end].long()] = level + 1
    return height


def parse_families(species_path, family_paths):
    """Parse the species tree and every gene family ONCE, returning a resident Rust handle.

    The returned ``ParsedFamilies`` object keeps the parsed per-family payloads in Rust. Feed it
    to :func:`preprocess_from_parsed` (or ``GeneReconModel(parsed_families=..., family_indices=...)``)
    to rebuild a model over any subset of the same families without re-reading a single file and
    without a JSON round trip.
    """
    return _load_native_module().ParsedFamilies.parse(
        str(species_path), [str(path) for path in family_paths]
    )


def plan_batches_from_parsed(
    parsed,
    indices,
    *,
    family_chunk_size: int | None,
    clade_budget: int | None,
    batch_packing: str,
    max_wave_size: int,
    family_group_assignments: list[int] | None,
):
    """Re-plan batches + wave layouts over ``indices`` of an already-parsed dataset.

    Returns ``{"batches", "batch_wave_layouts"}`` exactly as ``plan_batch_wave_layouts`` does,
    except that every numeric array inside is a numpy array instead of a JSON-decoded list, and
    ``batches`` hold positions into ``indices`` (not into the whole parsed dataset).
    """
    batch_packing = _normalize_batch_packing(batch_packing, clade_budget)
    raw = parsed.plan(
        [int(index) for index in indices],
        family_chunk_size,
        clade_budget,
        batch_packing,
        max_wave_size,
        [int(label) for label in family_group_assignments]
        if family_group_assignments is not None
        else None,
    )
    _validate_schema_version(raw, payload_name="ParsedFamilies.plan")
    raw["batches"] = [[int(index) for index in batch] for batch in raw["batches"]]
    return raw


def preprocess_from_parsed(
    parsed,
    indices,
    *,
    family_chunk_size: int | None,
    clade_budget: int | None,
    batch_packing: str,
    max_wave_size: int,
    family_group_assignments: list[int] | None,
    accumulator_dtype: torch.dtype | str | None,
):
    """The ``preprocess_dataset`` payload for a subset of an already-parsed dataset.

    Same dict structure, same tensors, same ordering -- but no file is re-read and no JSON text
    is produced: the numeric arrays arrive as numpy arrays straight out of Rust.
    """
    accumulator_dtype = _resolve_accumulator_dtype(accumulator_dtype)
    indices = [int(index) for index in indices]
    plan = plan_batches_from_parsed(
        parsed,
        indices,
        family_chunk_size=family_chunk_size,
        clade_budget=clade_budget,
        batch_packing=batch_packing,
        max_wave_size=max_wave_size,
        family_group_assignments=family_group_assignments,
    )
    return {
        "schema_version": _EXPECTED_SCHEMA_VERSION,
        "species": _species_tensors(parsed.species(), accumulator_dtype=accumulator_dtype),
        "families": parsed.families(indices),
        "batches": plan["batches"],
        "batch_wave_layouts": plan["batch_wave_layouts"],
    }


def species_name_to_index(species_tree_path: str) -> dict[str, int]:
    """{species_name: post-order gp index}, matching preprocess_dataset's ordering.

    Wraps the Rust ``species_leaf_index_map`` pyfunction. Use to resolve a
    ``{species_name: fraction_missing}`` dict onto the model's species indices.
    """
    raw = json.loads(_load_native_module().species_leaf_index_map(str(species_tree_path)))
    return {str(name): int(index) for name, index in raw.items()}


def build_wave_layout_from_plan(
    payload,
    *,
    device: torch.device | str,
    model_dtype: torch.dtype | str | None = None,
    accumulator_dtype: torch.dtype | str | None = None,
):
    model_dtype = _resolve_model_dtype(model_dtype)
    _resolve_accumulator_dtype(accumulator_dtype)
    plan = payload["plan"]
    # ``log_split_probs_sorted`` is f64 in the payload -- a numpy array on the parsed path, a list
    # of Python floats on the legacy JSON path. One vectorized gather per wave replaces the former
    # per-split Python indexing.
    logp = _float64_array(payload["log_split_probs_sorted"])
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
            split_indices = np.asarray(raw_meta.get("split_indices", []), dtype=np.int64)
            split_probabilities = _materialize_split_probabilities(
                logp[split_indices], device=device
            )
            meta["sl"] = _index_tensor(raw_meta["sl"], dtype=index_dtype, device=device)
            meta["sr"] = _index_tensor(raw_meta["sr"], dtype=index_dtype, device=device)
            meta["reduce_idx"] = _index_tensor(
                raw_meta["reduce_idx"], dtype=index_dtype, device=device
            )
            # Rust computes and serializes these values as f64. Keep prebuilt
            # variants for both supported model dtypes so ``model.to(...)`` and
            # explicit runtime parameters never widen an fp32-quantized tensor.
            meta["_log_split_probs_by_dtype"] = split_probabilities
            meta["log_split_probs"] = split_probabilities[model_dtype]
            meta["n_eq1"] = int(raw_meta.get("n_eq1", 0))
            # ``len(...)`` rather than truthiness: a numpy array has no boolean value.
            eq1_reduce_idx = raw_meta.get("eq1_reduce_idx")
            if eq1_reduce_idx is not None and len(eq1_reduce_idx):
                meta["eq1_reduce_idx"] = _index_tensor(
                    eq1_reduce_idx, dtype=index_dtype, device=device
                )
            ge2_ptr = raw_meta.get("ge2_ptr")
            if ge2_ptr is not None and len(ge2_ptr):
                meta["ge2_ptr"] = _index_tensor(ge2_ptr, dtype=torch.long, device=device)
                meta["ge2_parent_ids"] = _index_tensor(
                    raw_meta["ge2_parent_ids"], dtype=index_dtype, device=device
                )
                meta["ge2_max_fanout"] = int(raw_meta["ge2_max_fanout"])
        wave_metas.append(meta)

    family_idx = plan.get("family_idx")
    if family_idx is None:
        family_idx = np.zeros(int(plan["c"]), dtype=np.int64)
    return {
        "perm": _index_tensor(plan["perm"], dtype=torch.long, device=device),
        "leaf_species_index": _index_tensor(
            plan["leaf_species_index"], dtype=index_dtype, device=device
        ),
        "root_clade_ids": _index_tensor(
            plan["root_clade_ids"], dtype=torch.long, device=device
        ),
        "wave_metas": wave_metas,
        "family_idx": _index_tensor(family_idx, dtype=torch.long, device=device),
    }


def plan_batch_wave_layouts(
    families,
    *,
    family_chunk_size: int | None = 300,
    clade_budget: int | None = DEFAULT_CLADE_BUDGET,
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
    """Leaf rows and (parent, left, right, log-prior) split rows as plain Python lists.

    Accepts either numpy-backed payload arrays (the parsed path) or the JSON path's Python lists.
    """
    C = int(family["C"])
    leaf_rows = np.asarray(family["leaf_row_index"], dtype=np.int64).tolist()
    leftrights = np.asarray(family["split_leftrights_sorted"], dtype=np.int64)
    if "split_parents_sorted" in family:
        parents_array = np.asarray(family["split_parents_sorted"], dtype=np.int64)
        n_splits = int(family.get("N_splits", len(parents_array)))
        parents = parents_array.tolist()
        lefts = leftrights[:n_splits].tolist()
        rights = leftrights[n_splits:].tolist()
        logp = _float64_array(family["log_split_probs_sorted"]).tolist()
    else:
        leaf_set = set(leaf_rows)
        parents = [clade for clade in range(C) if clade not in leaf_set]
        lefts = leftrights[: len(parents)].tolist()
        rights = leftrights[len(parents) :].tolist()
        logp = [0.0] * len(parents)
    return leaf_rows, parents, lefts, rights, logp


def build_wave_layout(
    families,
    *,
    device: torch.device | str,
    max_wave_size: int = 8192,
    model_dtype: torch.dtype | str | None = None,
    accumulator_dtype: torch.dtype | str | None = None,
):
    model_dtype = _resolve_model_dtype(model_dtype)
    _resolve_accumulator_dtype(accumulator_dtype)
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
    # ``logp_global`` contains Python doubles decoded from Rust f64 values. Keep them on the host
    # as f64 and let each wave's ``_LazySplitProbabilities`` put only the dtypes a kernel asks for
    # on the device (the plan path below does the same), instead of building both device variants
    # of the whole array and then slicing them per wave.
    logp_host = torch.from_numpy(_float64_array(logp_global))
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
            split_probabilities = _LazySplitProbabilities(
                logp_host[split_tensor.cpu()].contiguous(), device
            )
            meta["_log_split_probs_by_dtype"] = split_probabilities
            meta["log_split_probs"] = split_probabilities[model_dtype]
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
    clade_budget: int | None = DEFAULT_CLADE_BUDGET,
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
