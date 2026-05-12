#!/usr/bin/env python3
"""Chunked global/uniform forward+backward pipeline benchmark.

This harness targets the training path from
``docs/forward-backward-full-pipeline-plan.md`` without moving any production
interfaces.  It solves the shared global E fixed point once per pass, runs full
saved-state uniform forward over each resident family chunk, immediately runs
``Pi_wave_backward`` for that chunk, then applies the E adjoint/theta VJP once
from the accumulated chunk adjoints.
"""

from __future__ import annotations

import argparse
import gc
import os
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

from gpurec.core.backward import Pi_wave_backward
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.memory_policy import choose_uniform_pipeline_policy
from gpurec.core.model import GeneDataset
from gpurec.optimization.implicit_grad import _e_adjoint_and_theta_vjp
from gpurec.api.uniform_chunked import (
    UNIFORM_OPTIMIZED_DEFAULT_FLAGS as DEFAULT_FLAGS,
    UniformBuiltChunk as BuiltChunk,
    UniformChunkSpec as ChunkSpec,
    _build_chunk,
    _make_chunks,
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
    root_clade_id_lists: list[list[int]]
    selected_indices: list[int]
    dtype: torch.dtype
    device: torch.device
    preprocess_s: float
    layout_s: float


class FallbackTracker:
    """Detect use of generic self-loop code paths during a benchmark pass."""

    def __init__(self) -> None:
        self.generic_self_loop_calls = 0
        self.generic_self_loop_names: set[str] = set()
        self._installed = False

    def install(self, *, block: bool) -> None:
        if self._installed:
            return
        from gpurec.core import backward as backward_core

        for name in ("_self_loop_vjp_precompute", "_gmres_self_loop_solve"):
            original = getattr(backward_core, name, None)
            if original is None:
                continue

            def wrapped(*args, _original=original, _name=name, **kwargs):
                self.generic_self_loop_calls += 1
                self.generic_self_loop_names.add(_name)
                if block:
                    raise RuntimeError(
                        "strict global/uniform pipeline reached the generic "
                        f"self-loop fallback: {_name}"
                    )
                return _original(*args, **kwargs)

            setattr(backward_core, name, wrapped)
        self._installed = True

    def reset(self) -> None:
        self.generic_self_loop_calls = 0
        self.generic_self_loop_names.clear()


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "0", "none", "null"):
        return None
    return int(text)


def _parse_auto_int(value: str | int | None) -> int | str:
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in ("", "auto", "default"):
        return "auto"
    return int(text)


def _parse_auto_optional_int(value: str | int | None) -> int | str | None:
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in ("", "auto", "default"):
        return "auto"
    if text in ("0", "none", "null"):
        return None
    return int(text)


def _parse_dtype(value: str) -> torch.dtype:
    text = value.strip().lower()
    if text in ("float32", "fp32", "single"):
        return torch.float32
    if text in ("float64", "fp64", "double"):
        return torch.float64
    raise argparse.ArgumentTypeError("dtype must be float32/fp32 or float64/fp64")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET", "tests/data/test_trees_1000"))
    parser.add_argument("--start", type=int, default=int(os.getenv("FAMILY_START", "0")))
    parser.add_argument("--fams", type=int, default=int(os.getenv("FAMS", "1000")))
    parser.add_argument("--family-chunk-size", default=os.getenv("FAMILY_CHUNK_SIZE", "auto"))
    parser.add_argument("--max-wave-size", default=os.getenv("MAX_WAVE_SIZE", "auto"))
    parser.add_argument("--fixed-iters", default=os.getenv("FIXED_ITERS_PI", "6"))
    parser.add_argument("--reps", type=int, default=int(os.getenv("REPS", "3")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "1")))
    parser.add_argument("--cache-dir", default=os.getenv("PREPROCESS_CACHE_DIR", "/tmp/gpurec_preprocess_cache"))
    parser.add_argument("--dtype", type=_parse_dtype, default=_parse_dtype(os.getenv("DTYPE", "float32")))
    parser.add_argument("--profile-cuda-api", action="store_true", default=os.getenv("PROFILE_CUDA_API", "0") != "0")
    parser.add_argument("--theta-rate", type=float, default=float(os.getenv("THETA_RATE", "0.05")))
    parser.add_argument("--max-iters-E", type=int, default=int(os.getenv("MAX_ITERS_E", "2000")))
    parser.add_argument("--tol-E", type=float, default=float(os.getenv("TOL_E", "1e-8")))
    parser.add_argument("--max-iters-Pi", type=int, default=int(os.getenv("MAX_ITERS_PI", "2000")))
    parser.add_argument("--tol-Pi", type=float, default=float(os.getenv("TOL_PI", "1e-6")))
    parser.add_argument("--neumann-terms", type=int, default=int(os.getenv("NEUMANN_TERMS", "3")))
    parser.add_argument("--use-pruning", action=argparse.BooleanOptionalAction, default=os.getenv("USE_PRUNING", "1") != "0")
    parser.add_argument("--pruning-threshold", type=float, default=float(os.getenv("PRUNING_THRESHOLD", "1e-6")))
    parser.add_argument("--stats-only", action="store_true", default=os.getenv("STATS_ONLY", "0") != "0")
    parser.add_argument(
        "--strict-optimized-kernels",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("STRICT_OPTIMIZED_KERNELS", "1") != "0",
        help="Report and fail early when required optimized global/uniform gates are inactive.",
    )
    parser.add_argument(
        "--compare-unchunked-max-fams",
        type=int,
        default=int(os.getenv("COMPARE_UNCHUNKED_MAX_FAMS", "8")),
        help="For fam counts at or below this value, compare chunked and one-chunk loss/gradient.",
    )
    parser.add_argument(
        "--fail-on-correctness-mismatch",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("FAIL_ON_CORRECTNESS_MISMATCH", "0") != "0",
    )
    parser.add_argument(
        "--empty-cache-between-reps",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("EMPTY_CACHE_BETWEEN_REPS", "0") != "0",
    )
    args = parser.parse_args()
    args.family_chunk_size = _parse_auto_int(args.family_chunk_size)
    args.max_wave_size = _parse_auto_optional_int(args.max_wave_size)
    args.fixed_iters = _parse_optional_int(args.fixed_iters)
    if isinstance(args.family_chunk_size, int) and args.family_chunk_size < 0:
        raise ValueError("--family-chunk-size must be non-negative")
    if args.reps <= 0:
        raise ValueError("--reps must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    return args


def _time_cuda_event(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


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
        _env_enabled("GPUREC_SELF_LOOP_2D_TRITON", "auto")
        and cuda_dtype_ok
        and s_gt_256
        and (
            getattr(args, "memory_policy", None) is None
            or bool(getattr(args.memory_policy, "proposal0", False))
        )
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
        "generic_pytorch_fallback": int(not optimized),
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
    if not torch.cuda.is_available():
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

    t0 = time.perf_counter()
    dataset = GeneDataset(
        species_tree_path=str(root / "sp.nwk"),
        gene_tree_paths=genes,
        genewise=False,
        specieswise=False,
        dtype=dtype,
        device=device,
        preprocess_cache_dir=args.cache_dir,
    )
    preprocess_s = time.perf_counter() - t0

    species_helpers, ancestors_T = dataset._species_helpers_for_mode(
        device=device,
        dtype=dtype,
    )
    unnorm_row_max = dataset.unnorm_row_max.to(device=device, dtype=dtype)
    theta = torch.log2(
        torch.tensor([args.theta_rate, args.theta_rate, args.theta_rate], device=device, dtype=dtype)
    )

    clade_counts = [int(f["C"]) for f in dataset.families]
    split_counts = [int(f["N_splits"]) for f in dataset.families]
    selected_indices = list(range(len(dataset.families)))
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
        os.environ["GPUREC_SELF_LOOP_2D_TRITON"] = "1" if memory_policy.proposal0 else "0"
    args.memory_policy = memory_policy
    specs = _make_chunks(
        selected_indices,
        clade_counts,
        split_counts,
        family_chunk_size=args.family_chunk_size,
        clade_budget=None,
    )

    t1 = time.perf_counter()
    built_chunks = [
        _build_chunk(
            dataset,
            spec,
            device=device,
            dtype=dtype,
            max_wave_size=args.max_wave_size,
            max_root_wave_size=None,
        )
        for spec in specs
    ]
    root_clade_id_lists = _root_clade_id_lists_for_chunks(built_chunks)
    torch.cuda.synchronize()
    layout_s = time.perf_counter() - t1

    return StaticInputs(
        root=root,
        genes=genes,
        dataset=dataset,
        species_helpers=species_helpers,
        ancestors_T=ancestors_T,
        unnorm_row_max=unnorm_row_max,
        theta=theta,
        built_chunks=built_chunks,
        root_clade_id_lists=root_clade_id_lists,
        selected_indices=selected_indices,
        dtype=dtype,
        device=device,
        preprocess_s=preprocess_s,
        layout_s=layout_s,
    )


def _compute_e_and_params(
    static: StaticInputs,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor]]:
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = extract_parameters_uniform(
        static.theta,
        static.unnorm_row_max,
        specieswise=False,
    )
    e_out = E_fixed_point(
        species_helpers=static.species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=transfer_mat,
        max_transfer_mat=max_transfer_vec,
        max_iters=args.max_iters_E,
        tolerance=args.tol_E,
        warm_start_E=None,
        dtype=static.dtype,
        device=static.device,
        ancestors_T=static.ancestors_T,
    )
    return e_out, (log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec)


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
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor],
) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = params
    pi_out = Pi_wave_forward(
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
        return_original=False,
        return_root_rows=False,
    )
    loss = compute_log_likelihood(
        pi_out["Pi_wave_ordered"],
        e_out["E"],
        built.wave_layout["root_clade_ids"],
    ).sum()
    return pi_out, loss


def _backward_chunk(
    built: BuiltChunk,
    root_clade_ids_perm: Sequence[int],
    static: StaticInputs,
    args: argparse.Namespace,
    e_out: dict[str, torch.Tensor],
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor],
    pi_out: dict[str, torch.Tensor | None],
) -> dict[str, Any]:
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = params
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
        ancestors_T=static.ancestors_T,
        uniform_pibar_row_max=pi_out.get("uniform_pibar_row_max"),
    )


def _finish_theta_gradient(
    pi_bwd_acc: dict[str, Any],
    static: StaticInputs,
    args: argparse.Namespace,
    e_out: dict[str, torch.Tensor],
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor],
) -> tuple[torch.Tensor, Any]:
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = params
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
    tracker: FallbackTracker,
    *,
    timed: bool,
    root_clade_id_lists: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    tracker.reset()
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
        "generic_self_loop_calls": tracker.generic_self_loop_calls,
        "generic_self_loop_names": ",".join(sorted(tracker.generic_self_loop_names)) or "none",
        "chunk_rows": chunk_rows,
    }


def _one_chunk_layout(static: StaticInputs, args: argparse.Namespace) -> list[BuiltChunk]:
    spec = ChunkSpec(
        indices=list(static.selected_indices),
        clades=sum(int(f["C"]) for f in static.dataset.families),
        splits=sum(int(f["N_splits"]) for f in static.dataset.families),
    )
    return [
        _build_chunk(
            static.dataset,
            spec,
            device=static.device,
            dtype=static.dtype,
            max_wave_size=args.max_wave_size,
            max_root_wave_size=None,
        )
    ]


def _maybe_compare_unchunked(
    static: StaticInputs,
    args: argparse.Namespace,
    tracker: FallbackTracker,
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
    chunked_result = _run_pipeline_pass(static.built_chunks, static, args, tracker, timed=False)
    unchunked_result = _run_pipeline_pass(unchunked, static, args, tracker, timed=False)

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
            if int(meta["phase"]) == 3 and meta.get("has_splits", False)
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
            "proposal0", int(memory_policy.proposal0),
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
        "proposal0_self_loop", os.environ.get("GPUREC_SELF_LOOP_2D_TRITON", "unset"),
        "proposal0_block_w", os.environ.get("GPUREC_SELF_LOOP_2D_BLOCK_W", "unset"),
        "strict_optimized_kernels", int(args.strict_optimized_kernels),
    )
    print(
        "optimized_path_verdict",
        "verdict", status["verdict"],
        "generic_pytorch_fallback", status["generic_pytorch_fallback"],
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
        "generic_self_loop_calls", result["generic_self_loop_calls"],
        "generic_self_loop_names", result["generic_self_loop_names"],
    )


def _print_summary(results: list[dict[str, Any]]) -> None:
    forward = [float(r["forward_ms"]) for r in results]
    backward = [float(r["backward_ms"]) for r in results]
    total = [float(r["total_ms"]) for r in results]
    peak = [float(r["peak_gib"]) for r in results]
    peak_reserved = [float(r["peak_reserved_gib"]) for r in results]
    generic_calls = sum(int(r["generic_self_loop_calls"]) for r in results)
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
        "generic_self_loop_calls", generic_calls,
    )


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)

    torch.cuda.empty_cache()
    gc.collect()

    static = _make_static_inputs(args)
    _print_policy(static, args)
    print("env_flags")
    for key in sorted(k for k in os.environ if k.startswith("GPUREC_")):
        print(key, os.environ[key])

    status = _optimized_feature_status(args, static)
    _print_active_path_flags(static, args, status)
    _validate_optimized_feature_status(args, status)

    if args.stats_only:
        return

    tracker = FallbackTracker()
    tracker.install(block=args.strict_optimized_kernels)

    _maybe_compare_unchunked(static, args, tracker)

    for _ in range(args.warmups):
        result = _run_pipeline_pass(static.built_chunks, static, args, tracker, timed=False)
        del result
        if args.empty_cache_between_reps:
            gc.collect()
            torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []
    for rep in range(args.reps):
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            result = _run_pipeline_pass(static.built_chunks, static, args, tracker, timed=True)
        finally:
            if args.profile_cuda_api:
                torch.cuda.cudart().cudaProfilerStop()
        _print_rep(rep, result)
        results.append(result)
        if args.empty_cache_between_reps:
            gc.collect()
            torch.cuda.empty_cache()

    _print_summary(results)


if __name__ == "__main__":
    main()
