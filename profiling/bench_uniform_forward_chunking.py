#!/usr/bin/env python3
"""Benchmark chunk policies for the uniform forward likelihood.

This harness targets Proposal 5 in
``docs/uniform-forward-optimization-proposals.md``.  It keeps the global
uniform E fixed point shared across chunks, builds one wave layout per resident
chunk, then times the full Pi/root-likelihood sweep over all chunks.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from gpurec.core.batching import (
    build_wave_layout,
    collate_gene_families,
    collate_wave,
    split_phase_waves,
)
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.model import GeneDataset
from gpurec.core.scheduling import compute_clade_waves


DEFAULT_FLAGS = {
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
    "GPUREC_FORWARD_DTS_OVERLAP_MODE": "off",
}


@dataclass
class ChunkSpec:
    indices: list[int]
    clades: int
    splits: int


@dataclass
class BuiltChunk:
    spec: ChunkSpec
    wave_layout: dict
    waves: int
    max_wave: int
    split_rows: int


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "0", "none", "null"):
        return None
    return int(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET", "tests/data/test_trees_1000"))
    parser.add_argument("--start", type=int, default=int(os.getenv("FAMILY_START", "0")))
    parser.add_argument("--fams", type=int, default=int(os.getenv("FAMS", "1000")))
    parser.add_argument("--reps", type=int, default=int(os.getenv("REPS", "3")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "1")))
    parser.add_argument("--fixed-iters", type=int, default=int(os.getenv("FIXED_ITERS_PI", "6")))
    parser.add_argument("--cache-dir", default=os.getenv("PREPROCESS_CACHE_DIR", "/tmp/gpurec_preprocess_cache"))
    parser.add_argument("--max-wave-size", default=os.getenv("MAX_WAVE_SIZE", "32768"))
    parser.add_argument("--max-root-wave-size", default=os.getenv("MAX_ROOT_WAVE_SIZE", ""))
    parser.add_argument("--family-chunk-size", type=int, default=int(os.getenv("FAMILY_CHUNK_SIZE", "150")))
    parser.add_argument("--clade-budget", default=os.getenv("CLADE_BUDGET", ""))
    parser.add_argument(
        "--auto-budget",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("AUTO_BUDGET", "0") != "0",
        help="Choose the clade budget from current free GPU memory.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=float(os.getenv("MEMORY_FRACTION", "0.65")),
        help="Fraction of free memory available to Pi/Pibar when --auto-budget is used.",
    )
    parser.add_argument(
        "--extra-gib",
        type=float,
        default=float(os.getenv("EXTRA_GIB", "1.5")),
        help="Reserved non-Pi/Pibar memory when --auto-budget is used.",
    )
    parser.add_argument(
        "--theta-rate",
        type=float,
        default=float(os.getenv("THETA_RATE", "0.05")),
        help="Shared D/L/T rate used for the global benchmark.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        default=os.getenv("STATS_ONLY", "0") != "0",
    )
    parser.add_argument(
        "--profile-cuda-api",
        action="store_true",
        default=os.getenv("PROFILE_CUDA_API", "0") != "0",
        help="Bracket timed repetitions with cudaProfilerStart/Stop for Nsys capture.",
    )
    parser.add_argument(
        "--compare-unchunked-max-fams",
        type=int,
        default=int(os.getenv("COMPARE_UNCHUNKED_MAX_FAMS", "50")),
        help="If fams is at most this value, compare against one unchunked layout.",
    )
    parser.add_argument(
        "--empty-cache-between-reps",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("EMPTY_CACHE_BETWEEN_REPS", "0") != "0",
    )
    args = parser.parse_args()
    args.max_wave_size = _parse_optional_int(args.max_wave_size)
    args.max_root_wave_size = _parse_optional_int(args.max_root_wave_size)
    args.clade_budget = _parse_optional_int(args.clade_budget)
    if args.family_chunk_size <= 0:
        args.family_chunk_size = 0
    return args


def _selected_genes(root: Path, start: int, fams: int) -> list[str]:
    genes = sorted(root.glob("g_*.nwk"))
    stop = start + fams
    if stop > len(genes):
        raise ValueError(f"requested families [{start}:{stop}], but only found {len(genes)} genes")
    return [str(p) for p in genes[start:stop]]


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _auto_clade_budget(S: int, dtype: torch.dtype, memory_fraction: float, extra_gib: float) -> int:
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    usable = max(0.0, free_bytes * memory_fraction - extra_gib * (1024 ** 3))
    bytes_per_clade = 2 * S * _dtype_nbytes(dtype)
    return max(1, int(usable // bytes_per_clade))


def _make_chunks(
    indices: Sequence[int],
    clade_counts: Sequence[int],
    split_counts: Sequence[int],
    *,
    family_chunk_size: int,
    clade_budget: int | None,
) -> list[ChunkSpec]:
    chunks: list[ChunkSpec] = []
    current: list[int] = []
    current_clades = 0
    current_splits = 0

    def flush() -> None:
        nonlocal current, current_clades, current_splits
        if current:
            chunks.append(ChunkSpec(list(current), current_clades, current_splits))
            current = []
            current_clades = 0
            current_splits = 0

    for idx in indices:
        n_clades = int(clade_counts[idx])
        n_splits = int(split_counts[idx])
        family_cap_hit = family_chunk_size > 0 and len(current) >= family_chunk_size
        clade_cap_hit = (
            clade_budget is not None
            and current
            and current_clades + n_clades > clade_budget
        )
        if family_cap_hit or clade_cap_hit:
            flush()
        current.append(int(idx))
        current_clades += n_clades
        current_splits += n_splits
    flush()
    return chunks


def _set_shared_theta(ds: GeneDataset, rate: float, *, device: torch.device, dtype: torch.dtype) -> None:
    theta = torch.log2(torch.tensor([rate, rate, rate], dtype=dtype, device=device))
    for fam in ds.families:
        fam["theta"] = theta.clone()


def _compute_shared_inputs(ds: GeneDataset, args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype):
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = ds._extract_batch_params(
        [0],
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
    )
    species_helpers, ancestors_T = ds._species_helpers_for_mode(
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
    )
    E_out = E_fixed_point(
        species_helpers=species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=transfer_mat,
        max_transfer_mat=max_transfer_vec,
        max_iters=2000,
        tolerance=1e-8,
        warm_start_E=None,
        dtype=dtype,
        device=device,
        pibar_mode="uniform",
        ancestors_T=ancestors_T,
    )
    return species_helpers, E_out, (log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec)


def _build_chunk(
    ds: GeneDataset,
    spec: ChunkSpec,
    *,
    species_helpers: dict,
    device: torch.device,
    dtype: torch.dtype,
    max_wave_size: int | None,
    max_root_wave_size: int | None,
) -> BuiltChunk:
    items = []
    fam_waves = []
    fam_phases = []
    for idx in spec.indices:
        fam = ds.families[idx]
        items.append({
            "ccp": fam["ccp_helpers"],
            "leaf_row_index": fam["leaf_row_index"],
            "leaf_col_index": fam["leaf_col_index"],
            "root_clade_id": int(fam["root_clade_id"]),
        })
        waves_i, phases_i = compute_clade_waves(fam["ccp_helpers"])
        fam_waves.append(waves_i)
        fam_phases.append(phases_i)

    batched = collate_gene_families(items, dtype=dtype, device=device)
    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(fam_waves, offsets)
    max_n_waves = max(len(p) for p in fam_phases)
    cross_phases: list[int] = []
    for k in range(max_n_waves):
        phase_k = 1
        for phases_i in fam_phases:
            if k < len(phases_i):
                phase_k = max(phase_k, phases_i[k])
        cross_phases.append(phase_k)
    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=None,
        max_wave_size=max_wave_size,
    )
    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=3,
        max_wave_size=max_root_wave_size,
    )
    family_clade_counts = [m["C"] for m in batched["family_meta"]]
    family_clade_offsets = [m["clade_offset"] for m in batched["family_meta"]]
    wave_layout = build_wave_layout(
        waves=cross_waves,
        phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device,
        dtype=dtype,
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
    )
    metas = wave_layout["wave_metas"]
    max_wave = max((int(m["W"]) for m in metas), default=0)
    split_rows = sum(int(m["sl"].numel()) for m in metas if m.get("has_splits", False))
    # Keep species_helpers in the signature so call sites make the dependency
    # explicit; the layout itself does not consume it.
    _ = species_helpers
    return BuiltChunk(
        spec=spec,
        wave_layout=wave_layout,
        waves=len(metas),
        max_wave=max_wave,
        split_rows=split_rows,
    )


def _time_cuda_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _run_chunks(
    built_chunks: Sequence[BuiltChunk],
    *,
    species_helpers: dict,
    E_out: dict,
    params: tuple,
    fixed_iters: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = params
    total = torch.zeros((), device=device, dtype=dtype)
    for built in built_chunks:
        pi_out = Pi_wave_forward(
            wave_layout=built.wave_layout,
            species_helpers=species_helpers,
            E=E_out["E"],
            Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"],
            E_s2=E_out["E_s2"],
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            device=device,
            dtype=dtype,
            local_iters=2000,
            local_tolerance=1e-6,
            fixed_iters=fixed_iters,
            pibar_mode="uniform",
            return_original=False,
            need_pibar=False,
        )
        total = total + compute_log_likelihood(
            pi_out["Pi_wave_ordered"],
            E_out["E"],
            built.wave_layout["root_clade_ids"],
        ).sum()
        del pi_out
    return total


def _print_chunks(chunks: Sequence[BuiltChunk]) -> None:
    print("chunk_table idx first last families clades splits waves max_wave split_rows")
    for ci, built in enumerate(chunks):
        indices = built.spec.indices
        print(
            "chunk",
            ci,
            indices[0],
            indices[-1],
            len(indices),
            built.spec.clades,
            built.spec.splits,
            built.waves,
            built.max_wave,
            built.split_rows,
        )


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)

    device = torch.device("cuda")
    dtype = torch.float32
    root = Path(args.dataset)
    genes = _selected_genes(root, args.start, args.fams)

    t0 = time.perf_counter()
    ds = GeneDataset(
        species_tree_path=str(root / "sp.nwk"),
        gene_tree_paths=genes,
        genewise=False,
        specieswise=False,
        pairwise=False,
        dtype=dtype,
        device=device,
        preprocess_cache_dir=args.cache_dir,
    )
    _set_shared_theta(ds, args.theta_rate, device=device, dtype=dtype)
    preprocess_s = time.perf_counter() - t0

    if args.auto_budget:
        args.clade_budget = _auto_clade_budget(
            int(ds.S),
            dtype,
            args.memory_fraction,
            args.extra_gib,
        )

    clade_counts = [int(f["C"]) for f in ds.families]
    split_counts = [int(f["N_splits"]) for f in ds.families]
    selected = list(range(len(ds.families)))
    specs = _make_chunks(
        selected,
        clade_counts,
        split_counts,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
    )

    species_helpers, E_out, params = _compute_shared_inputs(ds, args, device=device, dtype=dtype)
    torch.cuda.synchronize()

    t1 = time.perf_counter()
    built_chunks = [
        _build_chunk(
            ds,
            spec,
            species_helpers=species_helpers,
            device=device,
            dtype=dtype,
            max_wave_size=args.max_wave_size,
            max_root_wave_size=args.max_root_wave_size,
        )
        for spec in specs
    ]
    torch.cuda.synchronize()
    layout_s = time.perf_counter() - t1

    total_clades = sum(spec.clades for spec in specs)
    total_splits = sum(spec.splits for spec in specs)
    max_chunk_clades = max((spec.clades for spec in specs), default=0)
    min_chunk_clades = min((spec.clades for spec in specs), default=0)
    total_waves = sum(b.waves for b in built_chunks)
    print(
        "policy",
        "families", len(ds.families),
        "chunks", len(built_chunks),
        "family_chunk_size", args.family_chunk_size,
        "clade_budget", args.clade_budget if args.clade_budget is not None else "none",
        "max_wave_size", args.max_wave_size if args.max_wave_size is not None else "none",
        "max_root_wave_size", args.max_root_wave_size if args.max_root_wave_size is not None else "none",
        "S", int(ds.S),
        "total_clades", total_clades,
        "total_splits", total_splits,
        "min_chunk_clades", min_chunk_clades,
        "max_chunk_clades", max_chunk_clades,
        "total_waves", total_waves,
        "preprocess_s", f"{preprocess_s:.6f}",
        "layout_s", f"{layout_s:.6f}",
    )
    _print_chunks(built_chunks)
    if args.stats_only:
        return

    if len(ds.families) <= args.compare_unchunked_max_fams and len(built_chunks) > 1:
        unchunked = [
            _build_chunk(
                ds,
                ChunkSpec(selected, sum(clade_counts), sum(split_counts)),
                species_helpers=species_helpers,
                device=device,
                dtype=dtype,
                max_wave_size=args.max_wave_size,
                max_root_wave_size=args.max_root_wave_size,
            )
        ]
        nll_chunked = _run_chunks(
            built_chunks,
            species_helpers=species_helpers,
            E_out=E_out,
            params=params,
            fixed_iters=args.fixed_iters,
            device=device,
            dtype=dtype,
        )
        nll_unchunked = _run_chunks(
            unchunked,
            species_helpers=species_helpers,
            E_out=E_out,
            params=params,
            fixed_iters=args.fixed_iters,
            device=device,
            dtype=dtype,
        )
        torch.cuda.synchronize()
        print(
            "compare_unchunked",
            "chunked_nll", float(nll_chunked.detach().cpu()),
            "unchunked_nll", float(nll_unchunked.detach().cpu()),
            "abs_diff", float((nll_chunked - nll_unchunked).abs().detach().cpu()),
        )
        del unchunked, nll_chunked, nll_unchunked
        gc.collect()
        torch.cuda.empty_cache()

    for _ in range(args.warmups):
        out = _run_chunks(
            built_chunks,
            species_helpers=species_helpers,
            E_out=E_out,
            params=params,
            fixed_iters=args.fixed_iters,
            device=device,
            dtype=dtype,
        )
        del out
        torch.cuda.synchronize()
        if args.empty_cache_between_reps:
            gc.collect()
            torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    nll_value = math.nan
    if args.profile_cuda_api:
        torch.cuda.profiler.start()
    for _ in range(args.reps):
        ms, nll = _time_cuda_ms(
            lambda: _run_chunks(
                built_chunks,
                species_helpers=species_helpers,
                E_out=E_out,
                params=params,
                fixed_iters=args.fixed_iters,
                device=device,
                dtype=dtype,
            )
        )
        times.append(ms)
        nll_value = float(nll.detach().cpu())
        del nll
        if args.empty_cache_between_reps:
            gc.collect()
            torch.cuda.empty_cache()
    if args.profile_cuda_api:
        torch.cuda.profiler.stop()

    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(
        "timing",
        "reps", len(times),
        "median_ms", statistics.median(times),
        "mean_ms", statistics.mean(times),
        "min_ms", min(times),
        "max_ms", max(times),
        "nll", nll_value,
        "peak_gib", peak_gib,
    )


if __name__ == "__main__":
    main()
