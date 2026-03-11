"""C++-accelerated preprocessing pipeline for CCP construction."""

from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Dict, Any, Tuple

import torch
from torch.utils.cpp_extension import load


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CPP_DIR = _REPO_ROOT / "src" / "core" / "cpp"
_CPP_SRC = _CPP_DIR / "preprocess.cpp"


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    sources = [
        str(_CPP_SRC),
        str(_CPP_DIR / "tree_utils.cpp"),
        str(_CPP_DIR / "clade_utils.cpp"),
    ]
    return load(
        name="preprocess_cpp",
        sources=sources,
        extra_cflags=["-O3", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )


def preprocess_fast(
    species_tree_path: str,
    gene_tree_paths: "str | list[str]",
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    """Run the C++ preprocessing pipeline and convert results to torch tensors."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accept a single path or a list
    if isinstance(gene_tree_paths, str):
        gene_tree_paths = [gene_tree_paths]

    ext = _load_extension()
    raw = ext.preprocess(species_tree_path, gene_tree_paths)

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