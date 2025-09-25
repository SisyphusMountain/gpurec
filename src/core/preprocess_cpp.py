"""C++-accelerated preprocessing pipeline for CCP construction."""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Dict, Any, Tuple

import torch
from torch.utils.cpp_extension import load

from .clade import Clade, CCPContainer

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CPP_SRC = _REPO_ROOT / "src" / "core" / "cpp" / "preprocess.cpp"


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    return load(
        name="preprocess_cpp",
        sources=[str(_CPP_SRC)],
        extra_cflags=["-O3"],
        verbose=False,
    )


def preprocess_fast(
    species_tree_path: str,
    gene_tree_path: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Run the C++ preprocessing pipeline and convert results to torch tensors."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ext = _load_extension()
    raw = ext.preprocess(species_tree_path, gene_tree_path)

    species_raw = raw["species"]
    ccp_raw = raw["ccp"]
    clade_species_map = raw["clade_species_map"].to(dtype=dtype, device=device)

    # Species helpers tensors
    species_helpers = {
        "S": int(species_raw["S"]),
        "names": species_raw["names"],
        "s_P_indexes": species_raw["s_P_indexes"].to(device=device),
        "s_C12_indexes": species_raw["s_C12_indexes"].to(device=device),
        "Recipients_mat": species_raw["Recipients_mat"].to(dtype=dtype, device=device),
        "species_name_to_index": species_raw["species_name_to_index"],
    }



    # Populate helper tensors directly from C++ output
    ccp_helpers = {
        "split_counts": ccp_raw["split_counts"].to(device=device),
        "split_order": ccp_raw["split_order"].to(device=device),
        "split_parents_sorted": ccp_raw["split_parents_sorted"].to(device=device),
        "split_leftrights_sorted": ccp_raw["split_leftrights_sorted"].to(device=device),
        "log_split_probs_sorted": ccp_raw["log_split_probs_sorted"].to(dtype=dtype, device=device),
        "parents_sorted": ccp_raw["parents_sorted"].to(device=device),
        "seg_parent_ids": ccp_raw["seg_parent_ids"].to(device=device),
        "seg_counts": ccp_raw["seg_counts"].to(device=device),
        "ptr": ccp_raw["ptr"].to(device=device),
        "ptr_ge2": ccp_raw["ptr_ge2"].to(device=device),
        "num_segs_ge2": int(ccp_raw["num_segs_ge2"]),
        "num_segs_eq1": int(ccp_raw["num_segs_eq1"]),
        "num_segs_eq0": int(ccp_raw["num_segs_eq0"]),
        "stop_reduce_ptr_idx": int(ccp_raw["stop_reduce_ptr_idx"]),
        "end_rows_ge2": int(ccp_raw["end_rows_ge2"]),
        "C": int(ccp_raw["C"]),
        "N_splits": int(ccp_raw["N_splits"]),
        # Optional extras for summaries/UI
        "clade_leaf_labels": ccp_raw.get("clade_leaf_labels", None),
        "clade_is_leaf": ccp_raw.get("clade_is_leaf", None),
    }

    root_clade_id = int(ccp_raw["root_clade_id"])

    return {
        "species_helpers": species_helpers,
        "ccp_helpers": ccp_helpers,
        "clade_species_map": clade_species_map,
        "root_clade_id": root_clade_id,
    }


def compare_with_python(
    species_tree_path: str,
    gene_tree_path: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[bool, Dict[str, Any]]:
    return _compare_impl(
        species_tree_path,
        gene_tree_path,
        device=device,
        dtype=dtype,
        fast_precomputed=None,
    )


def compare_with_python_precomputed(
    species_tree_path: str,
    gene_tree_path: str,
    fast_precomputed: Dict[str, Any],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[bool, Dict[str, Any]]:
    return _compare_impl(
        species_tree_path,
        gene_tree_path,
        device=device,
        dtype=dtype,
        fast_precomputed=fast_precomputed,
    )


def _compare_impl(
    species_tree_path: str,
    gene_tree_path: str,
    *,
    device: torch.device | None,
    dtype: torch.dtype,
    fast_precomputed: Dict[str, Any] | None,
) -> Tuple[bool, Dict[str, Any]]:
    """Utility to compare C++ preprocessing output with the current Python pipeline."""
    from .ccp import build_ccp_from_single_tree, build_ccp_helpers
    from .tree_helpers import build_species_helpers
    from .ccp import build_clade_species_mapping, get_root_clade_id

    if fast_precomputed is None:
        fast = preprocess_fast(species_tree_path, gene_tree_path, device=device, dtype=dtype)
    else:
        fast = fast_precomputed

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    species_py = build_species_helpers(species_tree_path, device=device, dtype=dtype)
    ccp_py = build_ccp_from_single_tree(gene_tree_path)
    ccp_helpers_py = build_ccp_helpers(ccp_py, device=device, dtype=dtype)
    clade_species_py = build_clade_species_mapping(ccp_py, species_py, device=device, dtype=dtype)
    root_py = get_root_clade_id(ccp_py)
    fast_root = get_root_clade_id(fast["ccp"])

    def canonical_clade_map(container: CCPContainer) -> Dict[frozenset[str], int]:
        return {container.id_to_clade[cid].leaves: cid for cid in container.clade_to_id.values()}

    fast_clade_map = canonical_clade_map(fast["ccp"])
    py_clade_map = canonical_clade_map(ccp_py)

    def canonical_splits(container: CCPContainer) -> set[tuple[frozenset[str], frozenset[frozenset[str]]]]:
        splits_set = set()
        for clade in container.clades:
            cid = container.clade_to_id[clade]
            for split in container.splits[clade]:
                children = frozenset({split.left.leaves, split.right.leaves})
                splits_set.add((container.id_to_clade[cid].leaves, children))
        return splits_set

    splits_fast = canonical_splits(fast["ccp"])
    splits_py = canonical_splits(ccp_py)

    reorder_rows = []
    for leaves, py_idx in py_clade_map.items():
        if leaves not in fast_clade_map:
            reorder_rows = None
            break
        reorder_rows.append(fast_clade_map[leaves])

    details: Dict[str, Dict[str, Any]] = {}

    def _tensor_identical(name: str, a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, str]:
        if a.shape != b.shape:
            return False, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
        if a.dtype != b.dtype:
            b = b.to(dtype=a.dtype)
        if torch.equal(a, b):
            return True, "identical"
        msg_parts = []
        if a.is_floating_point():
            diff = a - b
            mask = torch.isfinite(a) & torch.isfinite(b)
            if mask.any():
                max_diff = torch.max(torch.abs(diff[mask])).item()
                msg_parts.append(f"max_abs_diff={max_diff:.3e}")
            unequal = (~torch.isclose(a, b, atol=0.0, rtol=0.0)).sum().item()
            msg_parts.append(f"unequal_elems={int(unequal)}")
        else:
            unequal = torch.ne(a, b).sum().item()
            msg_parts.append(f"unequal_elems={int(unequal)}")
        return False, ", ".join(msg_parts)

    species_keys = [
        "s_C1",
        "s_C2",
        "ancestors_dense",
        "Recipients_mat",
        "s_P_indexes",
        "s_C1_indexes",
        "s_C2_indexes",
        "s_C12_indexes",
        "sp_leaves_mask",
        "sp_internal_mask",
    ]
    for key in species_keys:
        match, msg = _tensor_identical(
            f"species.{key}",
            fast["species_helpers"][key],
            species_py[key],
        )
        details[f"species.{key}"] = {"match": match, "info": msg}

    helper_keys = [
        "seg_counts",
        "ptr",
        "ptr_ge2",
    ]
    for key in helper_keys:
        match, msg = _tensor_identical(
            f"ccp.{key}",
            fast["ccp_helpers"][key],
            ccp_helpers_py[key],
        )
        details[f"ccp.{key}"] = {"match": match, "info": msg}

    match_cs = False
    msg_cs = "no matching clade permutation"
    if reorder_rows is not None:
        permuted = fast["clade_species_map"][reorder_rows]
        match_cs, msg_cs = _tensor_identical(
            "clade_species_map",
            permuted,
            clade_species_py,
        )
    details["clade_species_map"] = {"match": match_cs, "info": msg_cs}

    if reorder_rows is not None:
        fast_id_to_py = {fast_id: py_idx for py_idx, fast_id in enumerate(reorder_rows)}
        parents_fast = fast["ccp_helpers"]["split_parents"].cpu().tolist()
        lefts_fast = fast["ccp_helpers"]["split_lefts"].cpu().tolist()
        rights_fast = fast["ccp_helpers"]["split_rights"].cpu().tolist()
        logs_fast = fast["ccp_helpers"]["log_split_probs"].cpu().tolist()
        tuples_fast = []
        missing_mapping = False
        for p, l, r, logp in zip(parents_fast, lefts_fast, rights_fast, logs_fast):
            if p not in fast_id_to_py or l not in fast_id_to_py or r not in fast_id_to_py:
                missing_mapping = True
                break
            mapped = (fast_id_to_py[p], fast_id_to_py[l], fast_id_to_py[r], logp)
            parent_id = mapped[0]
            left_id, right_id = sorted((mapped[1], mapped[2]))
            tuples_fast.append((parent_id, left_id, right_id, mapped[3]))
        parents_py = ccp_helpers_py["split_parents"].cpu().tolist()
        lefts_py = ccp_helpers_py["split_lefts"].cpu().tolist()
        rights_py = ccp_helpers_py["split_rights"].cpu().tolist()
        logs_py = ccp_helpers_py["log_split_probs"].cpu().tolist()
        tuples_py = []
        for p, l, r, logp in zip(parents_py, lefts_py, rights_py, logs_py):
            l_id, r_id = sorted((l, r))
            tuples_py.append((p, l_id, r_id, logp))
        if not missing_mapping:
            tuples_fast.sort()
            tuples_py.sort()
            sorted_match = tuples_fast == tuples_py
            info_sorted = "identical" if sorted_match else f"first diff fast={tuples_fast[:1]} py={tuples_py[:1]}"
        else:
            sorted_match = False
            info_sorted = "missing clade mapping"
    else:
        sorted_match = False
        info_sorted = "no permutation mapping"
    details["ccp.sorted_triples"] = {"match": sorted_match, "info": info_sorted}

    if not missing_mapping:
        base_parents_fast = [t[0] for t in tuples_fast]
        base_lefts_fast = [t[1] for t in tuples_fast]
        base_rights_fast = [t[2] for t in tuples_fast]
        base_logs_fast = [t[3] for t in tuples_fast]
        base_parents_py = [t[0] for t in tuples_py]
        base_lefts_py = [t[1] for t in tuples_py]
        base_rights_py = [t[2] for t in tuples_py]
        base_logs_py = [t[3] for t in tuples_py]
        details["ccp.split_parents(up_to_perm)"] = {"match": base_parents_fast == base_parents_py, "info": "identical"}
        details["ccp.split_lefts(up_to_perm)"] = {"match": base_lefts_fast == base_lefts_py, "info": "identical"}
        details["ccp.split_rights(up_to_perm)"] = {"match": base_rights_fast == base_rights_py, "info": "identical"}
        log_diff = max((abs(a - b) for a, b in zip(base_logs_fast, base_logs_py)), default=0.0)
        details["ccp.log_split_probs(up_to_perm)"] = {"match": base_logs_fast == base_logs_py, "info": f"max_abs_diff={log_diff:.3e}"}

        mapped_split_counts = [0] * len(reorder_rows)
        for parent in base_parents_fast:
            mapped_split_counts[parent] += 1
        counts_py_by_parent = ccp_helpers_py["split_counts"].cpu().tolist()
        counts_match_per_parent = mapped_split_counts == counts_py_by_parent
        counts_info_per_parent = "identical" if counts_match_per_parent else f"diff example fast={mapped_split_counts[:5]} py={counts_py_by_parent[:5]}"
        details["ccp.split_counts_per_parent"] = {"match": counts_match_per_parent, "info": counts_info_per_parent}

        counts_fast_sorted = sorted(fast["ccp_helpers"]["split_counts"].cpu().tolist())
        counts_py_sorted = sorted(counts_py_by_parent)
        counts_match = counts_fast_sorted == counts_py_sorted
        counts_info = "identical multiset" if counts_match else f"fast={counts_fast_sorted[:5]} py={counts_py_sorted[:5]}"
        details["ccp.split_counts_multiset"] = {"match": counts_match, "info": counts_info}

        parents_fast = fast["ccp_helpers"]["split_parents"].cpu().tolist()
        lefts_fast = fast["ccp_helpers"]["split_lefts"].cpu().tolist()
        rights_fast = fast["ccp_helpers"]["split_rights"].cpu().tolist()
        logs_fast_orig = fast["ccp_helpers"]["log_split_probs"].cpu().tolist()
        parents_fast_mapped = [fast_id_to_py[p] for p in parents_fast]
        lefts_fast_mapped = [fast_id_to_py[l] for l in lefts_fast]
        rights_fast_mapped = [fast_id_to_py[r] for r in rights_fast]
        parents_py_orig = parents_py
        lefts_py_orig = lefts_py
        rights_py_orig = rights_py
        logs_py_orig = logs_py

        order_fast = fast["ccp_helpers"]["split_order"].cpu().tolist()
        order_py = ccp_helpers_py["split_order"].cpu().tolist()
        ordered_fast = [(parents_fast_mapped[i], lefts_fast_mapped[i], rights_fast_mapped[i], logs_fast_orig[i]) for i in order_fast]
        ordered_py = [(parents_py_orig[i], lefts_py_orig[i], rights_py_orig[i], logs_py_orig[i]) for i in order_py]
        ordered_fast.sort()
        ordered_py.sort()
        details["ccp.sorted_triples_via_order"] = {"match": ordered_fast == ordered_py, "info": "identical" if ordered_fast == ordered_py else f"first diff fast={ordered_fast[:1]} py={ordered_py[:1]}"}

        parents_sorted_fast = [fast_id_to_py[p] for p in fast["ccp_helpers"]["parents_sorted"].cpu().tolist()]
        parents_sorted_py = ccp_helpers_py["parents_sorted"].cpu().tolist()
        details["ccp.parents_sorted(up_to_perm)"] = {
            "match": True,
            "info": "order differs" if parents_sorted_fast != parents_sorted_py else "identical",
        }

        seg_parent_ids_fast = [fast_id_to_py[p] for p in fast["ccp_helpers"]["seg_parent_ids"].cpu().tolist()]
        seg_parent_ids_py = ccp_helpers_py["seg_parent_ids"].cpu().tolist()
        details["ccp.seg_parent_ids(up_to_perm)"] = {
            "match": True,
            "info": "order differs" if seg_parent_ids_fast != seg_parent_ids_py else "identical",
        }

        split_parents_sorted_fast = [fast_id_to_py[p] for p in fast["ccp_helpers"]["split_parents_sorted"].cpu().tolist()]
        split_parents_sorted_py = ccp_helpers_py["split_parents_sorted"].cpu().tolist()
        details["ccp.split_parents_sorted(up_to_perm)"] = {
            "match": True,
            "info": "order differs" if split_parents_sorted_fast != split_parents_sorted_py else "identical",
        }

        split_lefts_sorted_fast = [fast_id_to_py[p] for p in fast["ccp_helpers"]["split_lefts_sorted"].cpu().tolist()]
        split_lefts_sorted_py = ccp_helpers_py["split_lefts_sorted"].cpu().tolist()
        details["ccp.split_lefts_sorted(up_to_perm)"] = {
            "match": True,
            "info": "order differs" if split_lefts_sorted_fast != split_lefts_sorted_py else "identical",
        }

        split_rights_sorted_fast = [fast_id_to_py[p] for p in fast["ccp_helpers"]["split_rights_sorted"].cpu().tolist()]
        split_rights_sorted_py = ccp_helpers_py["split_rights_sorted"].cpu().tolist()
        details["ccp.split_rights_sorted(up_to_perm)"] = {
            "match": True,
            "info": "order differs" if split_rights_sorted_fast != split_rights_sorted_py else "identical",
        }

        split_leftrights_sorted_fast = fast["ccp_helpers"]["split_leftrights_sorted"].cpu().tolist()
        mapped_leftrights = [fast_id_to_py[x] for x in split_leftrights_sorted_fast]
        split_leftrights_sorted_py = ccp_helpers_py["split_leftrights_sorted"].cpu().tolist()
        details["ccp.split_leftrights_sorted(up_to_perm)"] = {
            "match": True,
            "info": "order differs" if mapped_leftrights != split_leftrights_sorted_py else "identical",
        }

        log_sorted_fast = fast["ccp_helpers"]["log_split_probs_sorted"].cpu().tolist()
        log_sorted_py = ccp_helpers_py["log_split_probs_sorted"].cpu().tolist()
        log_sorted_diff = max((abs(a - b) for a, b in zip(log_sorted_fast, log_sorted_py)), default=0.0)
        details["ccp.log_split_probs_sorted(up_to_perm)"] = {
            "match": True,
            "info": f"max_abs_diff={log_sorted_diff:.3e}",
        }

    else:
        details["ccp.split_counts_multiset"] = {"match": False, "info": "missing clade mapping"}

    details["split_set"] = {
        "match": splits_fast == splits_py,
        "info": "identical" if splits_fast == splits_py else "split sets differ",
    }
    details["num_clades"] = {
        "match": len(fast_clade_map) == len(py_clade_map),
        "info": f"fast={len(fast_clade_map)}, python={len(py_clade_map)}",
    }
    details["root_clade_id"] = {
        "match": True,
        "info": f"fast={fast_root}, python={root_py}",
    }

    overall = all(entry["match"] for entry in details.values())
    return overall, details
