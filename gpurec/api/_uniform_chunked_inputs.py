"""Private input helpers for :mod:`gpurec.api.uniform_chunked`."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from gpurec.core.batch_planning import (
    normalize_batch_packing,
    normalize_clade_budget,
)
from gpurec.core.model import normalize_family_selection

from ._validation import (
    auto_nonnegative_int as _auto_nonnegative_int,
    auto_positive_int as _auto_positive_int,
    bool_value,
    nonnegative_float,
    nonnegative_int_sequence as _normalize_nonnegative_int_sequence,
    optional_positive_int,
    positive_even_int,
    positive_int,
    positive_int_sequence as _normalize_positive_int_sequence,
)


def _normalize_uniform_solver_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate public solver kwargs before CUDA setup or AleRax parsing."""
    normalized = dict(kwargs)
    removed_cache_kwargs = sorted(
        set(normalized).intersection(
            {"preprocess_cache_dir", "refresh_preprocess_cache"}
        )
    )
    if removed_cache_kwargs:
        names = ", ".join(removed_cache_kwargs)
        raise TypeError(f"preprocess caching has been removed; unsupported: {names}")
    if "set_optimized_env" in normalized:
        raise TypeError(
            "optimized environment toggles have been removed; unsupported: "
            "set_optimized_env"
        )
    if "dtype" in normalized:
        normalized["dtype"] = _validate_uniform_dtype(normalized["dtype"])
    if normalized.get("fixed_iters_E") is not None:
        normalized["fixed_iters_E"] = positive_int(
            "fixed_iters_E",
            normalized["fixed_iters_E"],
        )
    if "fixed_iters_Pi" in normalized:
        normalized["fixed_iters_Pi"] = positive_even_int(
            "fixed_iters_Pi",
            normalized["fixed_iters_Pi"],
        )
    if "max_iters_E" in normalized:
        normalized["max_iters_E"] = positive_int(
            "max_iters_E",
            normalized["max_iters_E"],
        )
    if "neumann_terms" in normalized:
        normalized["neumann_terms"] = positive_int(
            "neumann_terms",
            normalized["neumann_terms"],
        )
    if "tol_E" in normalized:
        normalized["tol_E"] = nonnegative_float("tol_E", normalized["tol_E"])
    if "pruning_threshold" in normalized:
        normalized["pruning_threshold"] = nonnegative_float(
            "pruning_threshold",
            normalized["pruning_threshold"],
        )
    for name in (
        "use_pruning",
        "warm_start_E",
        "profile",
    ):
        if name in normalized:
            normalized[name] = bool_value(name, normalized[name])
    if "family_chunk_size" in normalized:
        normalized["family_chunk_size"] = _auto_nonnegative_int(
            "family_chunk_size",
            normalized["family_chunk_size"],
        )
    if normalized.get("preprocess_cpu_cores") is not None:
        normalized["preprocess_cpu_cores"] = positive_int(
            "preprocess_cpu_cores",
            normalized["preprocess_cpu_cores"],
        )
    if "max_wave_size" in normalized:
        normalized["max_wave_size"] = _auto_positive_int(
            "max_wave_size",
            normalized["max_wave_size"],
        )
    if "max_root_wave_size" in normalized:
        normalized["max_root_wave_size"] = optional_positive_int(
            "max_root_wave_size",
            normalized["max_root_wave_size"],
        )
    if "clade_budget" in normalized:
        normalized["clade_budget"] = normalize_clade_budget(
            normalized["clade_budget"]
        )
    if "batch_packing" in normalized:
        normalized["batch_packing"] = normalize_batch_packing(
            normalized["batch_packing"]
        )
    if "family_chunk_candidates" in normalized:
        normalized["family_chunk_candidates"] = _normalize_nonnegative_int_sequence(
            "family_chunk_candidates",
            normalized["family_chunk_candidates"],
        )
    if "max_wave_candidates" in normalized:
        normalized["max_wave_candidates"] = _normalize_positive_int_sequence(
            "max_wave_candidates",
            normalized["max_wave_candidates"],
        )
    return normalized


def _validate_uniform_dtype(dtype: Any) -> torch.dtype:
    if dtype not in (torch.float32, torch.float64, torch.bfloat16):
        raise ValueError(f"dtype must be fp32, fp64, or bf16, got {dtype}")
    return dtype


def _selected_gene_paths(
    folder: Path,
    *,
    gene_glob: str,
    start: int,
    max_families: int | None,
) -> list[str]:
    start, max_families = normalize_family_selection(start, max_families)
    paths = sorted(folder.glob(gene_glob))
    if not paths and gene_glob == "g_*.nwk":
        single = folder / "g.nwk"
        if single.exists():
            paths = [single]
    if not paths:
        raise FileNotFoundError(f"no gene trees matching {gene_glob!r} in {folder}")
    stop = None if max_families is None else start + max_families
    selected = paths[start:stop]
    if not selected:
        raise ValueError(
            f"empty family selection: start={start}, max_families={max_families}, "
            f"available={len(paths)}"
        )
    return [str(p) for p in selected]
