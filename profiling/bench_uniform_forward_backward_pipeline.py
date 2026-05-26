#!/usr/bin/env python3
"""Chunked global/uniform forward+backward pipeline benchmark.

This harness targets the current chunked global/uniform training path without
moving any production interfaces.  It solves the shared global E fixed point
once per pass, runs full saved-state uniform forward over each resident family
chunk, immediately runs ``Pi_wave_backward`` for that chunk, then applies the E
adjoint/theta VJP once from the accumulated chunk adjoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gpurec._argparse_types import nonnegative_int_arg, positive_int_arg
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.forward import pi_training_state_request
from gpurec.core.likelihood import E_fixed_point, compute_nll_root_rows
from gpurec.core.memory_policy import choose_uniform_pipeline_policy
from gpurec.core.model import GeneDataset, normalize_family_inputs
from gpurec.core.preprocess_rust import RustPreprocessExtension
from gpurec.optimization.implicit_grad import _e_adjoint_and_theta_vjp
from gpurec.api.uniform_chunked import (
    _UniformBuiltChunk as BuiltChunk,
    _built_chunks_from_rust,
    _dtype_name_for_rust,
    _selected_gene_paths,
)


@dataclass
class StaticInputs:
    root: Path
    genes: list[str]
    dataset: GeneDataset
    species_helpers: dict[str, Any]
    ancestors_T: torch.Tensor | None
    unnorm_row_max: torch.Tensor
    theta: torch.Tensor
    built_chunks: list[BuiltChunk]
    rust_preprocessed: Any
    root_clade_id_lists: list[list[int]]
    dtype: torch.dtype
    device: torch.device
    preprocess_s: float
    layout_s: float


def _parse_optional_positive_int(name: str, value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "0", "none", "null"):
        return None
    return positive_int_arg(name)(text)


def _parse_auto_positive_int(name: str, value: str | int | None) -> int | str:
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in ("", "auto", "default"):
        return "auto"
    return positive_int_arg(name)(text)


def _parse_auto_optional_positive_int(
    name: str,
    value: str | int | None,
) -> int | str | None:
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in ("", "auto", "default"):
        return "auto"
    if text in ("0", "none", "null"):
        return None
    return positive_int_arg(name)(text)


def _parse_dtype(value: str) -> torch.dtype:
    text = value.strip().lower()
    if text in ("float32", "fp32", "single"):
        return torch.float32
    if text in ("float64", "fp64", "double"):
        return torch.float64
    raise argparse.ArgumentTypeError("dtype must be float32/fp32 or float64/fp64")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/data/test_trees_1000")
    parser.add_argument("--start", type=nonnegative_int_arg("start"), default=0)
    parser.add_argument("--fams", type=positive_int_arg("fams"), default=1000)
    parser.add_argument("--family-chunk-size", default="auto")
    parser.add_argument("--max-wave-size", default="auto")
    parser.add_argument("--fixed-iters", default="6")
    parser.add_argument("--reps", type=positive_int_arg("reps"), default=3)
    parser.add_argument("--warmups", type=nonnegative_int_arg("warmups"), default=1)
    parser.add_argument("--dtype", type=_parse_dtype, default=_parse_dtype("float32"))
    parser.add_argument("--profile-cuda-api", action="store_true", default=False)
    parser.add_argument("--theta-rate", type=float, default=0.05)
    parser.add_argument(
        "--max-iters-E",
        type=positive_int_arg("max-iters-E"),
        default=2000,
    )
    parser.add_argument("--tol-E", type=float, default=1e-8)
    parser.add_argument(
        "--neumann-terms",
        type=positive_int_arg("neumann-terms"),
        default=3,
    )
    parser.add_argument("--use-pruning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pruning-threshold", type=float, default=1e-6)
    parser.add_argument("--stats-only", action="store_true", default=False)
    parser.add_argument(
        "--progress-jsonl",
        action="store_true",
        default=False,
        help="Emit flushed JSONL progress records during setup and benchmark execution.",
    )
    parser.add_argument(
        "--preflight-only",
        "--setup-only",
        action="store_true",
        default=False,
        help="Build static inputs, print setup/status information, then exit before benchmark passes.",
    )
    parser.add_argument(
        "--preflight-window-size",
        type=nonnegative_int_arg("preflight-window-size"),
        default=0,
        help=(
            "Diagnostic setup-only mode: validate the requested family range in "
            "sequential windows of this size and discard each window before "
            "continuing. This does not run warmups/reps and is not a performance "
            "benchmark. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--strict-optimized-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report and fail early when required optimized global/uniform gates are inactive.",
    )
    parser.add_argument(
        "--compare-unchunked-max-fams",
        type=nonnegative_int_arg("compare-unchunked-max-fams"),
        default=8,
        help="For fam counts at or below this value, compare chunked and one-chunk loss/gradient.",
    )
    parser.add_argument(
        "--fail-on-correctness-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--empty-cache-between-reps",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    try:
        args.family_chunk_size = _parse_auto_positive_int(
            "family-chunk-size",
            args.family_chunk_size,
        )
        args.max_wave_size = _parse_auto_optional_positive_int(
            "max-wave-size",
            args.max_wave_size,
        )
        args.fixed_iters = _parse_optional_positive_int(
            "fixed-iters",
            args.fixed_iters,
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def _time_cuda_event(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _progress_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "progress_jsonl", False))


def _current_rss_mib() -> float | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 3)
    except OSError:
        return None
    return None


def _peak_rss_mib() -> float | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return None
    if sys.platform == "darwin":
        value /= 1024
    return round(float(value) / 1024.0, 3)


def _cuda_memory_fields() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "cuda_allocated_gib": None,
            "cuda_reserved_gib": None,
            "cuda_driver_free_gib": None,
            "cuda_driver_total_gib": None,
        }
    fields: dict[str, float | None] = {
        "cuda_allocated_gib": None,
        "cuda_reserved_gib": None,
        "cuda_driver_free_gib": None,
        "cuda_driver_total_gib": None,
    }
    try:
        fields["cuda_allocated_gib"] = round(
            torch.cuda.memory_allocated() / (1024 ** 3),
            6,
        )
        fields["cuda_reserved_gib"] = round(
            torch.cuda.memory_reserved() / (1024 ** 3),
            6,
        )
    except (RuntimeError, TypeError):
        return fields
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError:
        return fields
    fields["cuda_driver_free_gib"] = round(free_bytes / (1024 ** 3), 6)
    fields["cuda_driver_total_gib"] = round(total_bytes / (1024 ** 3), 6)
    return fields


def _progress_resource_fields() -> dict[str, float | None]:
    try:
        disk = shutil.disk_usage(REPO_ROOT)
        disk_free_gib = round(disk.free / (1024 ** 3), 6)
    except OSError:
        disk_free_gib = None
    return {
        "rss_mib": _current_rss_mib(),
        "rss_peak_mib": _peak_rss_mib(),
        "disk_free_gib": disk_free_gib,
        **_cuda_memory_fields(),
    }


def _emit_progress(args: argparse.Namespace, event: str, **fields: Any) -> None:
    if not _progress_enabled(args):
        return
    payload = {
        "record": "bench_uniform_forward_backward_pipeline",
        "event": event,
        "time_s": round(time.time(), 6),
        **_progress_resource_fields(),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _chunk_progress_row(idx: int, built: BuiltChunk) -> dict[str, int]:
    return {
        "idx": idx,
        "family_start": int(built.spec.indices[0]),
        "family_stop": int(built.spec.indices[-1]) + 1,
        "families": len(built.spec.indices),
        "clades": int(built.spec.clades),
        "splits": int(built.spec.splits),
        "waves": int(built.waves),
        "max_wave": int(built.max_wave),
        "split_rows": int(built.split_rows),
        "max_wave_split_rows": int(built.max_wave_split_rows),
    }


def _static_progress_summary(
    static: StaticInputs,
    args: argparse.Namespace,
) -> dict[str, Any]:
    total_clades = sum(int(b.spec.clades) for b in static.built_chunks)
    total_splits = sum(int(b.spec.splits) for b in static.built_chunks)
    total_waves = sum(int(b.waves) for b in static.built_chunks)
    return {
        "dataset": str(static.root),
        "family_start": int(args.start),
        "family_stop": int(args.start + args.fams),
        "families": len(static.genes),
        "chunks": len(static.built_chunks),
        "family_chunk_size": args.family_chunk_size,
        "max_wave_size": args.max_wave_size if args.max_wave_size is not None else None,
        "fixed_iters": args.fixed_iters,
        "dtype": str(static.dtype).replace("torch.", ""),
        "device": str(static.device),
        "S": int(static.dataset.S),
        "total_clades": total_clades,
        "total_splits": total_splits,
        "total_waves": total_waves,
        "max_wave": max((int(b.max_wave) for b in static.built_chunks), default=0),
        "max_wave_split_rows": max(
            (int(b.max_wave_split_rows) for b in static.built_chunks),
            default=0,
        ),
        "preprocess_s": round(static.preprocess_s, 6),
        "layout_s": round(static.layout_s, 6),
    }


def _nvtx_push(name: str, *, enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop(*, enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _nvtx_call(name: str, fn, *, enabled: bool):
    _nvtx_push(name, enabled=enabled)
    try:
        return fn()
    finally:
        _nvtx_pop(enabled=enabled)


def _root_clade_ids_to_list(root_ids: Any) -> list[int]:
    if torch.is_tensor(root_ids):
        return [int(x) for x in root_ids.detach().cpu().tolist()]
    return [int(x) for x in root_ids]


def _root_clade_id_lists_for_chunks(chunks: Sequence[BuiltChunk]) -> list[list[int]]:
    return [
        _root_clade_ids_to_list(built.wave_layout["root_clade_ids"])
        for built in chunks
    ]


def _has_fused_backward_module() -> bool:
    try:
        from gpurec.core.kernels import wave_backward as _wave_backward  # noqa: F401
    except Exception:
        return False
    return True


def _optimized_feature_status(
    args: argparse.Namespace,
    static: StaticInputs,
) -> dict[str, int | str]:
    has_fused = _has_fused_backward_module()
    cuda_dtype_ok = static.device.type == "cuda" and static.dtype in (torch.float32, torch.float64)
    s_gt_256 = int(static.dataset.S) > 256
    fixed_even = args.fixed_iters is not None and args.fixed_iters % 2 == 0

    full_saved_tensors_for_backward = 1
    root_row_output = 0
    forward_parent_reduced_dts = int(static.device.type == "cuda")
    backward_parent_reduced_dts = int(static.device.type == "cuda" and cuda_dtype_ok)
    pingpong = int(
        fixed_even
        and static.device.type == "cuda"
    )
    fused_uniform_backward = int(
        has_fused
        and cuda_dtype_ok
        and s_gt_256
    )
    kernelized_active_mask = int(
        has_fused
        and cuda_dtype_ok
    )
    kernelized_backward_dts = int(
        static.device.type == "cuda"
        and cuda_dtype_ok
    )
    fused_dts_backward_accum = int(
        kernelized_backward_dts
    )
    compact_tree_pibar_vjp = int(
        has_fused
        and cuda_dtype_ok
        and s_gt_256
    )
    proposal0_self_loop = int(
        cuda_dtype_ok
        and s_gt_256
    )
    optimized = int(
        root_row_output == 0
        and full_saved_tensors_for_backward == 1
        and forward_parent_reduced_dts
        and pingpong
        and fused_uniform_backward
        and kernelized_active_mask
        and kernelized_backward_dts
        and fused_dts_backward_accum
        and compact_tree_pibar_vjp
        and proposal0_self_loop
    )
    return {
        "verdict": "optimized" if optimized else "non_optimized",
        "generic_pytorch_path": int(not optimized),
        "root_row_output": root_row_output,
        "full_saved_tensors_for_backward": full_saved_tensors_for_backward,
        "forward_parent_reduced_dts": forward_parent_reduced_dts,
        "backward_parent_reduced_dts": backward_parent_reduced_dts,
        "pingpong": pingpong,
        "fused_uniform_backward": fused_uniform_backward,
        "kernelized_active_mask": kernelized_active_mask,
        "kernelized_backward_dts": kernelized_backward_dts,
        "fused_dts_backward_accum": fused_dts_backward_accum,
        "compact_tree_pibar_vjp": compact_tree_pibar_vjp,
        "proposal0_self_loop": proposal0_self_loop,
        "cuda_dtype_ok": int(cuda_dtype_ok),
        "species_gate_s_gt_256": int(s_gt_256),
        "has_fused_backward_module": int(has_fused),
        "cross_pibar_impl": "tree",
        "strict_optimized_kernels": int(args.strict_optimized_kernels),
    }


def _validate_optimized_feature_status(
    args: argparse.Namespace,
    status: dict[str, int | str],
) -> None:
    if not args.strict_optimized_kernels:
        return
    required = (
        "full_saved_tensors_for_backward",
        "forward_parent_reduced_dts",
        "pingpong",
        "fused_uniform_backward",
        "kernelized_active_mask",
        "kernelized_backward_dts",
        "fused_dts_backward_accum",
        "compact_tree_pibar_vjp",
        "proposal0_self_loop",
    )
    missing = [key for key in required if int(status[key]) != 1]
    if int(status["root_row_output"]) != 0:
        missing.append("root_row_output_disabled")
    if missing:
        details = " ".join(f"{key}={status[key]}" for key in sorted(status))
        raise RuntimeError(
            "strict optimized global/uniform pipeline requested, but required "
            f"optimized features are inactive: {', '.join(missing)}. "
            f"Active status: {details}"
        )


def _make_static_inputs(args: argparse.Namespace) -> StaticInputs:
    if _progress_enabled(args):
        _emit_progress(
            args,
            "static_inputs_start",
            dataset=str(args.dataset),
            fams=int(args.fams),
            start=int(args.start),
        )
    if not torch.cuda.is_available():
        if _progress_enabled(args):
            _emit_progress(
                args,
                "static_inputs_failed",
                stage="cuda_check",
                reason="cuda_unavailable",
            )
        raise RuntimeError("CUDA is required for the optimized uniform pipeline harness")

    device = torch.device("cuda")
    dtype = args.dtype
    root = Path(args.dataset)
    genes = _selected_gene_paths(
        root,
        gene_glob="g_*.nwk",
        start=args.start,
        max_families=args.fams,
    )
    if _progress_enabled(args):
        _emit_progress(
            args,
            "gene_selection_done",
            dataset=str(root),
            family_start=int(args.start),
            requested_families=int(args.fams),
            selected_families=len(genes),
        )

    t0 = time.perf_counter()
    family_tree_paths, family_names, leaf_species_maps = normalize_family_inputs(
        genes,
        [Path(gene).stem for gene in genes],
        None,
    )
    families_input = {
        name: paths
        for name, paths in zip(family_names, family_tree_paths)
    }
    rust_preprocessed = RustPreprocessExtension().preprocess_dataset(
        str(root / "sp.nwk"),
        families_input,
        leaf_species_maps={},
        include_species_matrices=False,
        num_threads=0,
    )
    dataset = GeneDataset._from_preprocessed_raw(
        raw=rust_preprocessed.to_torch(),
        species_tree_path=str(root / "sp.nwk"),
        gene_tree_paths=family_tree_paths,
        genewise=False,
        specieswise=False,
        dtype=dtype,
        device=device,
        family_names=family_names,
        leaf_species_maps=leaf_species_maps,
    )
    preprocess_s = time.perf_counter() - t0
    if _progress_enabled(args):
        _emit_progress(
            args,
            "dataset_loaded",
            families=len(dataset.families),
            S=int(dataset.S),
            total_clades=sum(int(f["C"]) for f in dataset.families),
            total_splits=sum(int(f["N_splits"]) for f in dataset.families),
            preprocess_s=round(preprocess_s, 6),
        )

    species_helpers, ancestors_T = dataset._species_helpers_for_mode(
        device=device,
        dtype=dtype,
    )
    unnorm_row_max = dataset.unnorm_row_max.to(device=device, dtype=dtype)
    theta = torch.log2(
        torch.tensor([args.theta_rate, args.theta_rate, args.theta_rate], device=device, dtype=dtype)
    )

    rust_counts = rust_preprocessed.family_basic_counts()
    clade_counts = [int(value) for value in rust_counts["clade_counts"]]
    split_counts = [int(value) for value in rust_counts["split_counts"]]
    memory_policy = None
    if args.family_chunk_size == "auto" or args.max_wave_size == "auto":
        family_candidates = (
            (25, 50, 10, 75, 100)
            if args.family_chunk_size == "auto"
            else (int(args.family_chunk_size),)
        )
        if args.max_wave_size == "auto":
            wave_candidates = (8192, 16384, 4096, 32768)
        elif args.max_wave_size is None:
            wave_candidates = (max(1, sum(clade_counts)),)
        else:
            wave_candidates = (int(args.max_wave_size),)
        memory_policy = choose_uniform_pipeline_policy(
            clade_counts,
            int(dataset.S),
            dtype,
            device=device,
            family_chunk_candidates=family_candidates,
            max_wave_candidates=wave_candidates,
        )
        if args.family_chunk_size == "auto":
            args.family_chunk_size = memory_policy.family_chunk_size
        if args.max_wave_size == "auto":
            args.max_wave_size = memory_policy.max_wave_size
    args.memory_policy = memory_policy

    t1 = time.perf_counter()
    built_chunks = _built_chunks_from_rust(
        rust_preprocessed.build_chunked_layouts(
            family_chunk_size=int(args.family_chunk_size),
            clade_budget=None,
            batch_packing="sequential",
            max_wave_size=args.max_wave_size,
            max_root_wave_size=None,
            dtype=_dtype_name_for_rust(dtype),
        ),
        device=device,
        dtype=dtype,
    )
    if _progress_enabled(args):
        _emit_progress(
            args,
            "chunk_policy_selected",
            chunks=len(built_chunks),
            family_chunk_size=args.family_chunk_size,
            max_wave_size=args.max_wave_size if args.max_wave_size is not None else None,
            memory_policy_reason=memory_policy.reason if memory_policy is not None else None,
            estimated_payload_gib=(
                round(memory_policy.estimated_payload_bytes / (1024 ** 3), 6)
                if memory_policy is not None
                else None
            ),
        )
    for chunk_idx, built in enumerate(built_chunks):
        if _progress_enabled(args):
            _emit_progress(args, "chunk_built", **_chunk_progress_row(chunk_idx, built))
    root_clade_id_lists = _root_clade_id_lists_for_chunks(built_chunks)
    torch.cuda.synchronize()
    layout_s = time.perf_counter() - t1

    static = StaticInputs(
        root=root,
        genes=genes,
        dataset=dataset,
        species_helpers=species_helpers,
        ancestors_T=ancestors_T,
        unnorm_row_max=unnorm_row_max,
        theta=theta,
        built_chunks=built_chunks,
        rust_preprocessed=rust_preprocessed,
        root_clade_id_lists=root_clade_id_lists,
        dtype=dtype,
        device=device,
        preprocess_s=preprocess_s,
        layout_s=layout_s,
    )
    if _progress_enabled(args):
        _emit_progress(args, "static_inputs_done", **_static_progress_summary(static, args))
    return static


def _compute_e_and_params(
    static: StaticInputs,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    log_pS, log_pD, log_pL, max_transfer_vec = extract_parameters_uniform(
        static.theta,
        static.unnorm_row_max,
        specieswise=False,
    )
    e_out = E_fixed_point(
        species_helpers=static.species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        max_transfer_mat=max_transfer_vec,
        max_iters=args.max_iters_E,
        tolerance=args.tol_E,
        warm_start_E=None,
        dtype=static.dtype,
        device=static.device,
        ancestors_T=static.ancestors_T,
        e_shape=(int(static.dataset.S),),
    )
    return e_out, (log_pS, log_pD, log_pL, max_transfer_vec)


def _accumulate_pi_backward(
    acc: dict[str, Any] | None,
    pi_bwd: dict[str, Any],
) -> dict[str, Any]:
    tensor_keys = (
        "grad_E",
        "grad_Ebar",
        "grad_E_s1",
        "grad_E_s2",
        "grad_log_pD",
        "grad_log_pS",
        "grad_max_transfer_mat",
    )
    scalar_keys = (
        "n_waves_total",
        "n_waves_skipped",
        "n_waves_processed",
        "n_clades_total",
        "n_clades_skipped",
        "n_clades_active",
    )
    if acc is None:
        acc = {
            key: pi_bwd[key].detach().clone()
            for key in tensor_keys
        }
        for key in scalar_keys:
            acc[key] = int(pi_bwd.get(key, 0))
        return acc

    for key in tensor_keys:
        acc[key].add_(pi_bwd[key])
    for key in scalar_keys:
        acc[key] = int(acc.get(key, 0)) + int(pi_bwd.get(key, 0))
    return acc


def _forward_chunk(
    built: BuiltChunk,
    static: StaticInputs,
    args: argparse.Namespace,
    e_out: dict[str, torch.Tensor],
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
    log_pS, log_pD, log_pL, max_transfer_vec = params
    pi_out = pi_training_state_request().run(
        wave_layout=built.wave_layout,
        species_helpers=static.species_helpers,
        E=e_out["E"],
        Ebar=e_out["E_bar"],
        E_s1=e_out["E_s1"],
        E_s2=e_out["E_s2"],
        log_pS=log_pS,
        log_pD=log_pD,
        max_transfer_mat=max_transfer_vec,
        device=static.device,
        dtype=static.dtype,
        fixed_iters=args.fixed_iters,
    )
    root_clade_ids = built.wave_layout["root_clade_ids"]
    loss = compute_nll_root_rows(
        pi_out["Pi_wave_ordered"][root_clade_ids, :],
        e_out["E"],
    ).sum()
    return pi_out, loss


def _backward_chunk(
    built: BuiltChunk,
    root_clade_ids_perm: Sequence[int],
    static: StaticInputs,
    args: argparse.Namespace,
    e_out: dict[str, torch.Tensor],
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pi_out: dict[str, torch.Tensor | None],
) -> dict[str, Any]:
    log_pS, log_pD, log_pL, max_transfer_vec = params
    return Pi_wave_backward(
        wave_layout=built.wave_layout,
        Pi_star_wave=pi_out["Pi_wave_ordered"],
        Pibar_star_wave=pi_out["Pibar_wave_ordered"],
        E=e_out["E"],
        Ebar=e_out["E_bar"],
        E_s1=e_out["E_s1"],
        E_s2=e_out["E_s2"],
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        max_transfer_mat=max_transfer_vec,
        species_helpers=static.species_helpers,
        root_clade_ids_perm=root_clade_ids_perm,
        device=static.device,
        dtype=static.dtype,
        neumann_terms=args.neumann_terms,
        use_pruning=args.use_pruning,
        pruning_threshold=args.pruning_threshold,
        uniform_pibar_row_max=pi_out.get("uniform_pibar_row_max"),
    )


def _finish_theta_gradient(
    pi_bwd_acc: dict[str, Any],
    static: StaticInputs,
    args: argparse.Namespace,
    e_out: dict[str, torch.Tensor],
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, Any]:
    log_pS, log_pD, log_pL, max_transfer_vec = params
    root_count = torch.zeros(
        (len(static.genes),),
        device=static.device,
        dtype=torch.long,
    )
    return _e_adjoint_and_theta_vjp(
        pi_bwd_acc,
        e_out["E"],
        e_out["E_bar"],
        e_out["E_s1"],
        e_out["E_s2"],
        log_pS,
        log_pD,
        log_pL,
        max_transfer_vec,
        static.species_helpers,
        root_count,
        static.theta,
        static.unnorm_row_max,
        False,
        static.device,
        static.dtype,
        genewise=False,
        ancestors_T=static.ancestors_T,
    )


def _run_pipeline_pass(
    chunks: Sequence[BuiltChunk],
    static: StaticInputs,
    args: argparse.Namespace,
    *,
    timed: bool,
    root_clade_id_lists: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    if root_clade_id_lists is None:
        if len(chunks) == len(static.built_chunks) and all(
            chunk is static_chunk
            for chunk, static_chunk in zip(chunks, static.built_chunks)
        ):
            root_clade_id_lists = static.root_clade_id_lists
        else:
            root_clade_id_lists = _root_clade_id_lists_for_chunks(chunks)
    if timed:
        torch.cuda.reset_peak_memory_stats()

    forward_ms = 0.0
    pi_forward_ms = 0.0
    e_ms = 0.0
    backward_ms = 0.0
    pi_backward_ms = 0.0
    e_adjoint_ms = 0.0
    saved_full_state = True
    pibar_row_max_saved = True
    root_rows_only = False
    chunk_rows: list[dict[str, Any]] = []

    if timed:
        _nvtx_push("global_uniform_forward_backward_pass", enabled=args.profile_cuda_api)
        e_ms, e_params = _time_cuda_event(
            lambda: _nvtx_call(
                "e_fixed_point_once",
                lambda: _compute_e_and_params(static, args),
                enabled=args.profile_cuda_api,
            )
        )
    else:
        e_params = _compute_e_and_params(static, args)
        torch.cuda.synchronize()
    e_out, params = e_params
    forward_ms += e_ms

    total_loss = torch.zeros((), device=static.device, dtype=static.dtype)
    pi_bwd_acc: dict[str, Any] | None = None

    for chunk_idx, built in enumerate(chunks):
        if timed:
            fwd_ms, fwd_result = _time_cuda_event(
                lambda built=built, chunk_idx=chunk_idx: _nvtx_call(
                    f"chunk_{chunk_idx}_forward",
                    lambda: _forward_chunk(built, static, args, e_out, params),
                    enabled=args.profile_cuda_api,
                )
            )
        else:
            fwd_result = _forward_chunk(built, static, args, e_out, params)
            torch.cuda.synchronize()
            fwd_ms = 0.0
        pi_out, chunk_loss = fwd_result
        pi_forward_ms += fwd_ms
        forward_ms += fwd_ms
        total_loss = total_loss + chunk_loss

        saved_full_state = saved_full_state and (
            torch.is_tensor(pi_out.get("Pi_wave_ordered"))
            and torch.is_tensor(pi_out.get("Pibar_wave_ordered"))
        )
        pibar_row_max_saved = pibar_row_max_saved and torch.is_tensor(
            pi_out.get("uniform_pibar_row_max")
        )
        root_rows_only = root_rows_only or torch.is_tensor(pi_out.get("Pi_root_rows"))

        if timed:
            bwd_ms, pi_bwd = _time_cuda_event(
                lambda built=built, pi_out=pi_out, chunk_idx=chunk_idx: _nvtx_call(
                    f"chunk_{chunk_idx}_pi_backward",
                    lambda: _backward_chunk(
                        built,
                        root_clade_id_lists[chunk_idx],
                        static,
                        args,
                        e_out,
                        params,
                        pi_out,
                    ),
                    enabled=args.profile_cuda_api,
                )
            )
        else:
            pi_bwd = _backward_chunk(
                built,
                root_clade_id_lists[chunk_idx],
                static,
                args,
                e_out,
                params,
                pi_out,
            )
            torch.cuda.synchronize()
            bwd_ms = 0.0
        pi_backward_ms += bwd_ms
        backward_ms += bwd_ms
        pi_bwd_acc = _accumulate_pi_backward(pi_bwd_acc, pi_bwd)

        chunk_rows.append(
            {
                "idx": chunk_idx,
                "family_start": int(built.spec.indices[0]),
                "family_stop": int(built.spec.indices[-1]) + 1,
                "families": len(built.spec.indices),
                "clades": built.spec.clades,
                "splits": built.spec.splits,
                "waves": built.waves,
                "max_wave": built.max_wave,
                "split_rows": built.split_rows,
                "forward_ms": fwd_ms,
                "pi_backward_ms": bwd_ms,
            }
        )

        del pi_out, pi_bwd, chunk_loss

    if pi_bwd_acc is None:
        raise RuntimeError("no chunks were run")

    if timed:
        e_adjoint_ms, grad_result = _time_cuda_event(
            lambda: _nvtx_call(
                "e_adjoint_once",
                lambda: _finish_theta_gradient(pi_bwd_acc, static, args, e_out, params),
                enabled=args.profile_cuda_api,
            )
        )
        _nvtx_pop(enabled=args.profile_cuda_api)
    else:
        grad_result = _finish_theta_gradient(pi_bwd_acc, static, args, e_out, params)
        torch.cuda.synchronize()
        e_adjoint_ms = 0.0
    grad_theta, _grad_stats = grad_result
    backward_ms += e_adjoint_ms

    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
    grad_detached = grad_theta.detach()
    grad_norm = float(torch.linalg.vector_norm(grad_detached).detach().cpu())
    grad_finite = bool(torch.isfinite(grad_detached).all().detach().cpu())
    loss_value = float(total_loss.detach().cpu())

    return {
        "forward_ms": forward_ms,
        "e_ms": e_ms,
        "pi_forward_ms": pi_forward_ms,
        "backward_ms": backward_ms,
        "pi_backward_ms": pi_backward_ms,
        "e_adjoint_ms": e_adjoint_ms,
        "total_ms": forward_ms + backward_ms,
        "peak_gib": peak_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "loss": loss_value,
        "grad": grad_detached.clone(),
        "grad_norm": grad_norm,
        "grad_finite": int(grad_finite),
        "saved_full_state": int(saved_full_state),
        "pibar_row_max_saved": int(pibar_row_max_saved),
        "root_rows_only": int(root_rows_only),
        "chunk_rows": chunk_rows,
    }


def _one_chunk_layout(static: StaticInputs, args: argparse.Namespace) -> list[BuiltChunk]:
    return _built_chunks_from_rust(
        static.rust_preprocessed.build_chunked_layouts(
            family_chunk_size=0,
            clade_budget=None,
            batch_packing="sequential",
            max_wave_size=args.max_wave_size,
            max_root_wave_size=None,
            dtype=_dtype_name_for_rust(static.dtype),
        ),
        device=static.device,
        dtype=static.dtype,
    )


def _maybe_compare_unchunked(
    static: StaticInputs,
    args: argparse.Namespace,
) -> None:
    if len(static.genes) > args.compare_unchunked_max_fams:
        print(
            "compare_unchunked",
            "skipped",
            "reason", "fams_above_threshold",
            "fams", len(static.genes),
            "threshold", args.compare_unchunked_max_fams,
        )
        return
    if len(static.built_chunks) <= 1:
        print(
            "compare_unchunked",
            "skipped",
            "reason", "single_chunk",
            "fams", len(static.genes),
        )
        return

    unchunked = _one_chunk_layout(static, args)
    chunked_result = _run_pipeline_pass(static.built_chunks, static, args, timed=False)
    unchunked_result = _run_pipeline_pass(unchunked, static, args, timed=False)

    grad_a = chunked_result["grad"]
    grad_b = unchunked_result["grad"]
    grad_abs_diff = float((grad_a - grad_b).abs().max().detach().cpu())
    grad_ref = float(grad_b.abs().max().detach().cpu())
    grad_rel_diff = grad_abs_diff / max(grad_ref, 1.0)
    loss_abs_diff = abs(float(chunked_result["loss"]) - float(unchunked_result["loss"]))
    if static.dtype == torch.float64:
        loss_atol = 1e-8
        loss_rtol = 1e-10
        grad_atol = 1e-7
        grad_rtol = 1e-7
    else:
        loss_atol = 5e-4
        loss_rtol = 1e-6
        grad_atol = 5e-4
        grad_rtol = 5e-4
    loss_allowed = loss_atol + loss_rtol * max(abs(float(unchunked_result["loss"])), 1.0)
    grad_allowed = grad_atol + grad_rtol * max(grad_ref, 1.0)
    verdict = (
        "pass"
        if loss_abs_diff <= loss_allowed and grad_abs_diff <= grad_allowed
        else "fail"
    )
    print(
        "compare_unchunked",
        "chunked_loss", f"{float(chunked_result['loss']):.10f}",
        "unchunked_loss", f"{float(unchunked_result['loss']):.10f}",
        "loss_abs_diff", f"{loss_abs_diff:.8e}",
        "grad_max_abs_diff", f"{grad_abs_diff:.8e}",
        "grad_rel_diff", f"{grad_rel_diff:.8e}",
        "loss_allowed", f"{loss_allowed:.8e}",
        "grad_allowed", f"{grad_allowed:.8e}",
        "loss_atol", f"{loss_atol:.1e}",
        "loss_rtol", f"{loss_rtol:.1e}",
        "grad_atol", f"{grad_atol:.1e}",
        "grad_rtol", f"{grad_rtol:.1e}",
        "chunked_grad_finite", chunked_result["grad_finite"],
        "unchunked_grad_finite", unchunked_result["grad_finite"],
        "verdict", verdict,
    )
    if verdict != "pass" and args.fail_on_correctness_mismatch:
        raise RuntimeError("chunked vs unchunked correctness comparison failed")

    del chunked_result, unchunked_result, unchunked
    gc.collect()
    torch.cuda.empty_cache()


def _print_policy(static: StaticInputs, args: argparse.Namespace) -> None:
    total_clades = sum(b.spec.clades for b in static.built_chunks)
    total_splits = sum(b.spec.splits for b in static.built_chunks)
    total_waves = sum(b.waves for b in static.built_chunks)
    max_wave = max((b.max_wave for b in static.built_chunks), default=0)
    max_wave_split_rows = max((b.max_wave_split_rows for b in static.built_chunks), default=0)
    max_phase3_split_rows = max(
        (
            int(meta["sl"].numel())
            for b in static.built_chunks
            for meta in b.wave_layout["wave_metas"]
            if meta.get("has_splits", False)
            and any(
                int(meta["start"]) <= int(root_id) < int(meta["end"])
                for root_id in b.wave_layout["root_clade_ids_cpu"]
            )
        ),
        default=0,
    )
    print(
        "pipeline_policy",
        "mode", "global",
        "dataset", static.root,
        "family_range", f"{args.start}:{args.start + args.fams}",
        "families", len(static.genes),
        "chunks", len(static.built_chunks),
        "family_chunk_size", args.family_chunk_size,
        "max_wave_size", args.max_wave_size if args.max_wave_size is not None else "none",
        "fixed_iters", args.fixed_iters if args.fixed_iters is not None else "none",
        "dtype", str(static.dtype).replace("torch.", ""),
        "S", int(static.dataset.S),
        "total_clades", total_clades,
        "total_splits", total_splits,
        "total_waves", total_waves,
        "max_wave", max_wave,
        "max_wave_split_rows", max_wave_split_rows,
        "max_phase3_split_rows", max_phase3_split_rows,
        "use_pruning", int(args.use_pruning),
        "pruning_threshold", args.pruning_threshold,
        "preprocess_s", f"{static.preprocess_s:.6f}",
        "layout_s", f"{static.layout_s:.6f}",
    )
    memory_policy = getattr(args, "memory_policy", None)
    if memory_policy is not None:
        print(
            "memory_policy",
            "family_chunk_size", memory_policy.family_chunk_size,
            "max_wave_size", memory_policy.max_wave_size,
            "estimated_payload_gib", f"{memory_policy.estimated_payload_bytes / (1024 ** 3):.3f}",
            "budget_gib", (
                f"{memory_policy.budget_bytes / (1024 ** 3):.3f}"
                if memory_policy.budget_bytes is not None
                else "unknown"
            ),
            "reason", memory_policy.reason,
        )
    print("chunk_table idx first last families clades splits waves max_wave split_rows max_wave_split_rows")
    for idx, built in enumerate(static.built_chunks):
        print(
            "chunk",
            idx,
            int(built.spec.indices[0]),
            int(built.spec.indices[-1]),
            len(built.spec.indices),
            built.spec.clades,
            built.spec.splits,
            built.waves,
            built.max_wave,
            built.split_rows,
            built.max_wave_split_rows,
        )


def _print_active_path_flags(
    static: StaticInputs,
    args: argparse.Namespace,
    status: dict[str, int | str],
) -> None:
    print(
        "active_path_flags",
        "mode", "global",
        "fixed_iters", args.fixed_iters if args.fixed_iters is not None else "none",
        "chunks", len(static.built_chunks),
        "root_row_output", status["root_row_output"],
        "full_saved_tensors_for_backward", status["full_saved_tensors_for_backward"],
        "forward_parent_reduced_dts", status["forward_parent_reduced_dts"],
        "backward_parent_reduced_dts", status["backward_parent_reduced_dts"],
        "pingpong", status["pingpong"],
        "fused_uniform_backward", status["fused_uniform_backward"],
        "kernelized_active_mask", status["kernelized_active_mask"],
        "kernelized_backward_dts", status["kernelized_backward_dts"],
        "fused_dts_backward_accum", status["fused_dts_backward_accum"],
        "compact_tree_pibar_vjp", status["compact_tree_pibar_vjp"],
        "proposal0_self_loop", status["proposal0_self_loop"],
        "strict_optimized_kernels", int(args.strict_optimized_kernels),
    )
    print(
        "optimized_path_verdict",
        "verdict", status["verdict"],
        "generic_pytorch_path", status["generic_pytorch_path"],
        "root_row_output", status["root_row_output"],
        "full_saved_tensors_for_backward", status["full_saved_tensors_for_backward"],
        "forward_parent_reduced_dts", status["forward_parent_reduced_dts"],
        "backward_parent_reduced_dts", status["backward_parent_reduced_dts"],
        "pingpong", status["pingpong"],
        "fused_uniform_backward", status["fused_uniform_backward"],
        "kernelized_active_mask", status["kernelized_active_mask"],
        "kernelized_backward_dts", status["kernelized_backward_dts"],
        "fused_dts_backward_accum", status["fused_dts_backward_accum"],
        "compact_tree_pibar_vjp", status["compact_tree_pibar_vjp"],
        "proposal0_self_loop", status["proposal0_self_loop"],
        "cuda_dtype_ok", status["cuda_dtype_ok"],
        "species_gate_s_gt_256", status["species_gate_s_gt_256"],
        "has_fused_backward_module", status["has_fused_backward_module"],
        "cross_pibar_impl", status["cross_pibar_impl"],
        "strict_optimized_kernels", status["strict_optimized_kernels"],
    )
    print(
        "strict_optimized_verdict",
        "pass" if status["verdict"] == "optimized" and args.strict_optimized_kernels else (
            "disabled" if not args.strict_optimized_kernels else "fail"
        ),
        "strict_optimized_kernels", int(args.strict_optimized_kernels),
    )


def _clone_args_for_preflight_window(
    args: argparse.Namespace,
    *,
    start: int,
    fams: int,
) -> argparse.Namespace:
    window_args = argparse.Namespace(**vars(args))
    window_args.start = start
    window_args.fams = fams
    window_args.memory_policy = None
    return window_args


def _run_static_preflight(
    args: argparse.Namespace,
) -> StaticInputs:
    static = _make_static_inputs(args)
    _print_policy(static, args)
    status = _optimized_feature_status(args, static)
    if _progress_enabled(args):
        _emit_progress(args, "optimized_status", **status)
    _print_active_path_flags(static, args, status)
    _validate_optimized_feature_status(args, status)
    if _progress_enabled(args):
        _emit_progress(args, "preflight_done", **_static_progress_summary(static, args))
    return static


def _run_windowed_preflight(args: argparse.Namespace) -> None:
    window_size = int(args.preflight_window_size)
    if window_size <= 0:
        raise ValueError("preflight_window_size must be positive")

    selected = _selected_gene_paths(
        Path(args.dataset),
        gene_glob="g_*.nwk",
        start=args.start,
        max_families=args.fams,
    )
    selected_total = len(selected)
    windows = (selected_total + window_size - 1) // window_size
    print(
        "windowed_preflight",
        "dataset", args.dataset,
        "family_range", f"{args.start}:{args.start + args.fams}",
        "selected_families", selected_total,
        "window_size", window_size,
        "windows", windows,
        "performance_evidence", 0,
    )
    if _progress_enabled(args):
        _emit_progress(
            args,
            "windowed_preflight_start",
            dataset=str(args.dataset),
            family_start=int(args.start),
            requested_families=int(args.fams),
            selected_families=selected_total,
            window_size=window_size,
            windows=windows,
            performance_evidence=0,
        )

    for window_idx, offset in enumerate(range(0, selected_total, window_size)):
        window_start = int(args.start + offset)
        window_fams = int(min(window_size, selected_total - offset))
        window_stop = window_start + window_fams
        window_args = _clone_args_for_preflight_window(
            args,
            start=window_start,
            fams=window_fams,
        )
        print(
            "preflight_window",
            "idx", window_idx,
            "family_range", f"{window_start}:{window_stop}",
            "families", window_fams,
        )
        if _progress_enabled(args):
            _emit_progress(
                args,
                "preflight_window_start",
                idx=window_idx,
                family_start=window_start,
                family_stop=window_stop,
                families=window_fams,
            )
        static: StaticInputs | None = None
        try:
            static = _run_static_preflight(window_args)
            if _progress_enabled(args):
                _emit_progress(
                    args,
                    "preflight_window_done",
                    idx=window_idx,
                    **_static_progress_summary(static, window_args),
                )
        finally:
            del static
            gc.collect()
            torch.cuda.empty_cache()

    if _progress_enabled(args):
        _emit_progress(
            args,
            "windowed_preflight_done",
            dataset=str(args.dataset),
            selected_families=selected_total,
            window_size=window_size,
            windows=windows,
            performance_evidence=0,
        )
    print(
        "windowed_preflight_done",
        "selected_families", selected_total,
        "windows", windows,
        "performance_evidence", 0,
    )


def _print_rep(rep: int, result: dict[str, Any]) -> None:
    print(
        "pipeline_rep",
        rep,
        "forward_ms", f"{result['forward_ms']:.3f}",
        "e_ms", f"{result['e_ms']:.3f}",
        "pi_forward_ms", f"{result['pi_forward_ms']:.3f}",
        "backward_ms", f"{result['backward_ms']:.3f}",
        "pi_backward_ms", f"{result['pi_backward_ms']:.3f}",
        "e_adjoint_ms", f"{result['e_adjoint_ms']:.3f}",
        "total_ms", f"{result['total_ms']:.3f}",
        "peak_gib", f"{result['peak_gib']:.3f}",
        "peak_reserved_gib", f"{result['peak_reserved_gib']:.3f}",
        "loss", f"{result['loss']:.10f}",
        "grad_norm", f"{result['grad_norm']:.8e}",
        "grad_finite", result["grad_finite"],
        "saved_full_state", result["saved_full_state"],
        "pibar_row_max_saved", result["pibar_row_max_saved"],
        "root_rows_only", result["root_rows_only"],
    )


def _print_summary(results: list[dict[str, Any]]) -> None:
    forward = [float(r["forward_ms"]) for r in results]
    backward = [float(r["backward_ms"]) for r in results]
    total = [float(r["total_ms"]) for r in results]
    peak = [float(r["peak_gib"]) for r in results]
    peak_reserved = [float(r["peak_reserved_gib"]) for r in results]
    grad_finite = int(all(int(r["grad_finite"]) for r in results))
    last = results[-1]
    print(
        "metrics",
        "forward_ms", f"{statistics.median(forward):.3f}",
        "backward_ms", f"{statistics.median(backward):.3f}",
        "total_ms", f"{statistics.median(total):.3f}",
        "peak_gib", f"{max(peak):.3f}",
        "peak_reserved_gib", f"{max(peak_reserved):.3f}",
        "loss", f"{float(last['loss']):.10f}",
        "grad_norm", f"{float(last['grad_norm']):.8e}",
        "grad_finite", grad_finite,
    )
    print(
        "pipeline_summary",
        "reps", len(results),
        "forward_median_ms", f"{statistics.median(forward):.3f}",
        "forward_mean_ms", f"{statistics.mean(forward):.3f}",
        "backward_median_ms", f"{statistics.median(backward):.3f}",
        "backward_mean_ms", f"{statistics.mean(backward):.3f}",
        "total_median_ms", f"{statistics.median(total):.3f}",
        "total_mean_ms", f"{statistics.mean(total):.3f}",
        "max_peak_gib", f"{max(peak):.3f}",
        "max_peak_reserved_gib", f"{max(peak_reserved):.3f}",
        "loss_last", f"{float(last['loss']):.10f}",
        "grad_norm_last", f"{float(last['grad_norm']):.8e}",
        "grad_finite", grad_finite,
    )


def main() -> None:
    args = _parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    if args.preflight_window_size > 0:
        _run_windowed_preflight(args)
        return

    static = _run_static_preflight(args)

    if args.preflight_only or args.stats_only:
        return

    if _progress_enabled(args):
        _emit_progress(
            args,
            "benchmark_start",
            warmups=int(args.warmups),
            reps=int(args.reps),
            compare_unchunked_max_fams=int(args.compare_unchunked_max_fams),
        )
    _maybe_compare_unchunked(static, args)

    for warmup_idx in range(args.warmups):
        if _progress_enabled(args):
            _emit_progress(args, "warmup_start", idx=warmup_idx)
        result = _run_pipeline_pass(static.built_chunks, static, args, timed=False)
        if _progress_enabled(args):
            _emit_progress(args, "warmup_done", idx=warmup_idx)
        del result
        if args.empty_cache_between_reps:
            gc.collect()
            torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []
    for rep in range(args.reps):
        if _progress_enabled(args):
            _emit_progress(args, "rep_start", idx=rep)
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            result = _run_pipeline_pass(static.built_chunks, static, args, timed=True)
        finally:
            if args.profile_cuda_api:
                torch.cuda.cudart().cudaProfilerStop()
        _print_rep(rep, result)
        if _progress_enabled(args):
            _emit_progress(
                args,
                "rep_done",
                idx=rep,
                total_ms=round(float(result["total_ms"]), 6),
                peak_gib=round(float(result["peak_gib"]), 6),
                grad_finite=int(result["grad_finite"]),
            )
        results.append(result)
        if args.empty_cache_between_reps:
            gc.collect()
            torch.cuda.empty_cache()

    _print_summary(results)
    if _progress_enabled(args):
        _emit_progress(args, "benchmark_done", reps=len(results))


if __name__ == "__main__":
    main()
