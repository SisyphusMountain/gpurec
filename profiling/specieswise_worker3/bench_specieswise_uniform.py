#!/usr/bin/env python3
"""Profile specieswise/uniform optimization without touching core code."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from gpurec import GeneReconModel
from gpurec.api.autograd import _extract_parameters
from gpurec.core.batching import build_wave_layout, collate_gene_families, collate_wave
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.scheduling import compute_clade_waves
from gpurec.optimization.implicit_grad import implicit_grad_loglik_vjp_wave


DEFAULT_FLAGS = {
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
    "GPUREC_FORWARD_FAMILY_INDEXED_CONSTS": "1",
    "GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS": "1",
    "GPUREC_FORWARD_LEAF_INDEX": "1",
    "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
    "GPUREC_FUSED_DTS_BACKWARD_ACCUM": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL": "tree",
    "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
    "GPUREC_BACKWARD_LEAF_INDEX": "1",
    "GPUREC_FUSED_WAVE_PARAM_ACCUM": "1",
    "GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES": "1",
    "GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS": "1",
    "GPUREC_DTS_GRAD_MT_TWO_STAGE": "1",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/data/test_trees_1000")
    parser.add_argument("--fams", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--dtype", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--fixed-iters-pi", type=int, default=6)
    parser.add_argument("--max-wave-size", default="32768")
    parser.add_argument("--max-root-wave-size", default="")
    parser.add_argument("--cache-dir", default="/tmp/gpurec_specieswise_worker3_cache")
    parser.add_argument("--out-dir", default="profiling/specieswise_worker3/artifacts")
    parser.add_argument("--label", default="before")
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--profile-cuda-api", action="store_true")
    parser.add_argument(
        "--engine",
        choices=("autograd", "chunked"),
        default="autograd",
        help="autograd is the public resident model path; chunked mirrors optimizer batching.",
    )
    parser.add_argument("--family-batch-size", type=int, default=10)
    parser.add_argument(
        "--profile-phase",
        choices=("full_step", "forward", "backward", "chunk0_grad"),
        default="full_step",
    )
    args = parser.parse_args()
    mws = str(args.max_wave_size).strip().lower()
    args.max_wave_size = None if mws in ("", "0", "none", "null") else int(mws)
    mrws = str(args.max_root_wave_size).strip().lower()
    args.max_root_wave_size = None if mrws in ("", "0", "none", "null") else int(mrws)
    return args


def _selected_genes(root: Path, start: int, fams: int) -> list[str]:
    genes = sorted(root.glob("g_*.nwk"))
    stop = start + fams
    if stop > len(genes):
        raise ValueError(f"requested families [{start}:{stop}], found {len(genes)}")
    return [str(p) for p in genes[start:stop]]


def _mem() -> dict[str, float]:
    torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "free_gib": free_b / (1024**3),
        "total_gib": total_b / (1024**3),
    }


def _shape(model: GeneReconModel) -> dict[str, Any]:
    wl = model.static.wave_layout
    metas = wl["wave_metas"]
    split_rows = sum(
        int(m["sl"].numel()) if m.get("has_splits", False) else 0
        for m in metas
    )
    C = sum(int(m["W"]) for m in metas)
    max_w = max((int(m["W"]) for m in metas), default=0)
    max_splits = max(
        (int(m["sl"].numel()) for m in metas if m.get("has_splits", False)),
        default=0,
    )
    top_split_waves = []
    for k, meta in enumerate(metas):
        if not meta.get("has_splits", False):
            continue
        top_split_waves.append(
            {
                "k": k,
                "start": int(meta["start"]),
                "W": int(meta["W"]),
                "split_rows": int(meta["sl"].numel()),
                "n_eq1": int(meta.get("n_eq1", 0)),
                "n_ge2_groups": int(meta.get("n_ge2_clades", 0)),
                "max_ge2_fanout": int(meta.get("ge2_max_fanout", 0) or 0),
            }
        )
    top_split_waves.sort(key=lambda row: row["split_rows"], reverse=True)
    return {
        "S": int(model.n_species),
        "G": int(model.n_families),
        "C": int(C),
        "waves": len(metas),
        "maxW": int(max_w),
        "split_rows": int(split_rows),
        "leaves": int(wl["leaf_row_index"].numel()),
        "roots": int(model.static.root_clade_ids.numel()),
        "theta_numel": int(model.theta.numel()),
        "theta_shape": list(model.theta.shape),
        "max_splits": int(max_splits),
        "top_split_waves": top_split_waves[:12],
    }


def _layout_shape(wave_layout: dict[str, Any]) -> dict[str, int]:
    metas = wave_layout["wave_metas"]
    return {
        "C": int(sum(int(m["W"]) for m in metas)),
        "waves": int(len(metas)),
        "maxW": int(max((int(m["W"]) for m in metas), default=0)),
        "split_rows": int(
            sum(
                int(m["sl"].numel()) if m.get("has_splits", False) else 0
                for m in metas
            )
        ),
        "leaves": int(wave_layout["leaf_row_index"].numel()),
        "roots": int(wave_layout["root_clade_ids"].numel()),
    }


def _event_time_ms(fn) -> tuple[float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _build_chunk_layout(
    families: list[dict[str, Any]],
    start: int,
    stop: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    fam_items = []
    fam_waves = []
    fam_phases = []
    for fam in families[start:stop]:
        fam_items.append(
            {
                "ccp": fam["ccp_helpers"],
                "leaf_row_index": fam["leaf_row_index"],
                "leaf_col_index": fam["leaf_col_index"],
                "root_clade_id": int(fam["root_clade_id"]),
            }
        )
        waves, phases = compute_clade_waves(fam["ccp_helpers"])
        fam_waves.append(waves)
        fam_phases.append(phases)

    batched = collate_gene_families(fam_items, dtype=dtype, device=device)
    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(fam_waves, offsets)
    max_n_waves = max(len(p) for p in fam_phases)
    cross_phases = []
    for k in range(max_n_waves):
        phase_k = 1
        for fp in fam_phases:
            if k < len(fp):
                phase_k = max(phase_k, fp[k])
        cross_phases.append(phase_k)
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
    return wave_layout, batched["root_clade_ids"]


def _chunked_step(
    model: GeneReconModel,
    *,
    family_batch_size: int,
    lr: float,
    profile_cuda_api: bool,
    profile_phase: str,
) -> dict[str, Any]:
    static = model.static
    theta = model.theta.detach().clone()
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = _extract_parameters(
        theta, static
    )

    e_wall0 = time.perf_counter()
    e_ms, E_out = _event_time_ms(
        lambda: E_fixed_point(
            species_helpers=static.species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            max_iters=static.max_iters_E,
            tolerance=static.tol_E,
            warm_start_E=static.warm_E,
            dtype=static.dtype,
            device=static.device,
            pibar_mode=static.pibar_mode,
            ancestors_T=static.ancestors_T,
        )
    )
    e_wall_ms = (time.perf_counter() - e_wall0) * 1000.0

    ranges = [
        (i, min(i + family_batch_size, model.n_families))
        for i in range(0, model.n_families, family_batch_size)
    ]
    grad_theta = torch.zeros_like(theta)
    nll = 0.0
    chunk_rows = []
    pi_ms_total = 0.0
    grad_ms_total = 0.0
    layout_wall_ms_total = 0.0
    pi_bwd_s_total = 0.0
    cg_s_total = 0.0
    theta_vjp_s_total = 0.0
    max_chunk_shape: dict[str, int] | None = None

    for chunk_id, (start, stop) in enumerate(ranges):
        layout_wall0 = time.perf_counter()
        wl_b, roots_b = _build_chunk_layout(
            model._dataset.families,
            start,
            stop,
            dtype=static.dtype,
            device=static.device,
        )
        layout_wall_ms = (time.perf_counter() - layout_wall0) * 1000.0
        layout_wall_ms_total += layout_wall_ms
        chunk_shape = _layout_shape(wl_b)
        if max_chunk_shape is None or chunk_shape["maxW"] > max_chunk_shape["maxW"]:
            max_chunk_shape = chunk_shape

        def _run_pi():
            pi_out_b = Pi_wave_forward(
                wave_layout=wl_b,
                species_helpers=static.species_helpers,
                E=E_out["E"],
                Ebar=E_out["E_bar"],
                E_s1=E_out["E_s1"],
                E_s2=E_out["E_s2"],
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                transfer_mat=transfer_mat,
                max_transfer_mat=max_transfer_vec,
                device=static.device,
                dtype=static.dtype,
                local_iters=static.max_iters_Pi,
                local_tolerance=static.tol_Pi,
                fixed_iters=static.fixed_iters_Pi,
                pibar_mode=static.pibar_mode,
                return_original=False,
            )
            logL_b = compute_log_likelihood(
                pi_out_b["Pi_wave_ordered"],
                E_out["E"],
                wl_b["root_clade_ids"],
            )
            return pi_out_b, logL_b

        pi_wall0 = time.perf_counter()
        pi_ms, (pi_out_b, logL_b) = _event_time_ms(_run_pi)
        pi_wall_ms = (time.perf_counter() - pi_wall0) * 1000.0
        pi_ms_total += pi_ms
        nll += float(logL_b.sum().item())

        def _run_grad():
            return implicit_grad_loglik_vjp_wave(
                wl_b,
                static.species_helpers,
                Pi_star_wave=pi_out_b["Pi_wave_ordered"],
                Pibar_star_wave=pi_out_b["Pibar_wave_ordered"],
                E_star=E_out["E"],
                E_s1=E_out["E_s1"],
                E_s2=E_out["E_s2"],
                Ebar=E_out["E_bar"],
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                max_transfer_mat=max_transfer_vec,
                root_clade_ids_perm=wl_b["root_clade_ids"],
                theta=theta,
                unnorm_row_max=static.unnorm_row_max,
                specieswise=True,
                device=static.device,
                dtype=static.dtype,
                neumann_terms=static.neumann_terms,
                use_pruning=static.use_pruning,
                pruning_threshold=static.pruning_threshold,
                cg_tol=static.cg_tol,
                cg_maxiter=static.cg_maxiter,
                gmres_restart=static.gmres_restart,
                pibar_mode=static.pibar_mode,
                transfer_mat=transfer_mat,
                transfer_mat_unnormalized=static.transfer_mat_unnormalized,
                ancestors_T=static.ancestors_T,
                uniform_pibar_row_max=pi_out_b.get("uniform_pibar_row_max"),
            )

        should_profile_chunk = (
            profile_cuda_api
            and profile_phase in ("full_step", "backward", "chunk0_grad")
            and chunk_id == 0
        )
        if should_profile_chunk:
            torch.cuda.cudart().cudaProfilerStart()
        grad_wall0 = time.perf_counter()
        grad_ms, (grad_theta_b, statsG_b) = _event_time_ms(_run_grad)
        grad_wall_ms = (time.perf_counter() - grad_wall0) * 1000.0
        if should_profile_chunk:
            torch.cuda.cudart().cudaProfilerStop()
        grad_ms_total += grad_ms
        grad_theta = grad_theta + grad_theta_b
        pi_bwd_s_total += float(getattr(statsG_b, "pi_bwd_time", 0.0))
        cg_s_total += float(getattr(statsG_b, "cg_time", 0.0))
        theta_vjp_s_total += float(getattr(statsG_b, "theta_vjp_time", 0.0))
        chunk_rows.append(
            {
                "chunk": chunk_id,
                "range": [start, stop],
                "layout_wall_ms": layout_wall_ms,
                "pi_event_ms": pi_ms,
                "pi_wall_ms": pi_wall_ms,
                "grad_event_ms": grad_ms,
                "grad_wall_ms": grad_wall_ms,
                "shape": chunk_shape,
                "pi_bwd_s": float(getattr(statsG_b, "pi_bwd_time", 0.0)),
                "cg_s": float(getattr(statsG_b, "cg_time", 0.0)),
                "theta_vjp_s": float(getattr(statsG_b, "theta_vjp_time", 0.0)),
            }
        )
        del pi_out_b, logL_b, wl_b, roots_b

    grad_theta = grad_theta / float(max(len(ranges), 1))
    opt_ms, theta_new = _event_time_ms(
        lambda: torch.clamp(theta - lr * grad_theta, min=torch.log2(theta.new_tensor(1e-10)))
    )
    static.warm_E = E_out["E"].detach()
    return {
        "engine": "chunked",
        "family_batch_size": family_batch_size,
        "n_chunks": len(ranges),
        "E_event_ms": e_ms,
        "E_wall_ms": e_wall_ms,
        "Pi_event_ms": pi_ms_total,
        "grad_event_ms": grad_ms_total,
        "layout_wall_ms": layout_wall_ms_total,
        "optimizer_ms": opt_ms,
        "full_step_ms_sum": e_ms + pi_ms_total + grad_ms_total + opt_ms,
        "nll": nll,
        "E_iters": int(E_out["iterations"]),
        "grad_inf": float(grad_theta.abs().max().detach().cpu()),
        "theta_new_mean": float(theta_new.mean().detach().cpu()),
        "pi_bwd_s": pi_bwd_s_total,
        "cg_s": cg_s_total,
        "theta_vjp_s": theta_vjp_s_total,
        "max_chunk_shape": max_chunk_shape,
        "chunk_rows": chunk_rows,
    }


def _one_step(model: GeneReconModel, opt: torch.optim.Optimizer) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    forward_ms, loss = _event_time_ms(lambda: model())
    backward_ms, _ = _event_time_ms(lambda: loss.backward())

    def _opt_step() -> None:
        opt.step()
        model.clamp_theta_()

    opt_ms, _ = _event_time_ms(_opt_step)
    torch.cuda.synchronize()
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": opt_ms,
        "full_step_ms_sum": forward_ms + backward_ms + opt_ms,
        "loss": float(loss.detach().cpu()),
    }


def _profiled_step(
    model: GeneReconModel,
    opt: torch.optim.Optimizer,
    *,
    phase: str,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    result: dict[str, Any] = {}
    if phase == "forward":
        torch.cuda.cudart().cudaProfilerStart()
        forward_ms, loss = _event_time_ms(lambda: model())
        torch.cuda.cudart().cudaProfilerStop()
        backward_ms, _ = _event_time_ms(lambda: loss.backward())
        opt_ms, _ = _event_time_ms(lambda: (opt.step(), model.clamp_theta_()))
    elif phase == "backward":
        forward_ms, loss = _event_time_ms(lambda: model())
        torch.cuda.cudart().cudaProfilerStart()
        backward_ms, _ = _event_time_ms(lambda: loss.backward())
        torch.cuda.cudart().cudaProfilerStop()
        opt_ms, _ = _event_time_ms(lambda: (opt.step(), model.clamp_theta_()))
    else:
        torch.cuda.cudart().cudaProfilerStart()
        forward_ms, loss = _event_time_ms(lambda: model())
        backward_ms, _ = _event_time_ms(lambda: loss.backward())
        opt_ms, _ = _event_time_ms(lambda: (opt.step(), model.clamp_theta_()))
        torch.cuda.cudart().cudaProfilerStop()
    result.update(
        {
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": opt_ms,
            "full_step_ms_sum": forward_ms + backward_ms + opt_ms,
            "loss": float(loss.detach().cpu()),
        }
    )
    return result


def _summ(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _event_attr(evt, name: str) -> float:
    value = getattr(evt, name, 0.0)
    return float(value() if callable(value) else value)


def _profiler_buckets(prof) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    buckets: dict[str, dict[str, float]] = {}
    patterns = {
        "indexing": ("index", "gather", "scatter", "select", "slice", "take"),
        "materialization": ("empty", "zeros", "ones", "full", "clone", "copy", "to", "cat", "stack", "contiguous"),
        "reductions": ("sum", "max", "min", "any", "all", "logsumexp"),
        "elementwise": ("add", "sub", "mul", "div", "exp", "log", "where", "clamp", "abs", "neg"),
        "cuda_runtime": ("cudaLaunchKernel", "cudaMemcpy", "cudaMemset", "cudaDeviceSynchronize"),
    }

    for evt in prof.key_averages():
        key = evt.key
        calls = int(getattr(evt, "count", 0))
        cpu_us = _event_attr(evt, "cpu_time_total")
        cuda_us = _event_attr(evt, "cuda_time_total") or _event_attr(evt, "self_cuda_time_total")
        rows.append(
            {
                "key": key,
                "calls": calls,
                "cpu_total_us": cpu_us,
                "cuda_total_us": cuda_us,
                "self_cpu_total_us": _event_attr(evt, "self_cpu_time_total"),
                "self_cuda_total_us": _event_attr(evt, "self_cuda_time_total"),
            }
        )
        lower = key.lower()
        matched = False
        for bucket, terms in patterns.items():
            if any(term.lower() in lower for term in terms):
                dst = buckets.setdefault(
                    bucket,
                    {"calls": 0.0, "cpu_total_us": 0.0, "cuda_total_us": 0.0},
                )
                dst["calls"] += calls
                dst["cpu_total_us"] += cpu_us
                dst["cuda_total_us"] += cuda_us
                matched = True
        if not matched:
            dst = buckets.setdefault(
                "other",
                {"calls": 0.0, "cpu_total_us": 0.0, "cuda_total_us": 0.0},
            )
            dst["calls"] += calls
            dst["cpu_total_us"] += cpu_us
            dst["cuda_total_us"] += cuda_us

    rows.sort(key=lambda row: row["cuda_total_us"], reverse=True)
    bucket_rows = [
        {
            "bucket": bucket,
            "calls": int(vals["calls"]),
            "cpu_total_ms": vals["cpu_total_us"] / 1000.0,
            "cuda_total_ms": vals["cuda_total_us"] / 1000.0,
        }
        for bucket, vals in sorted(
            buckets.items(),
            key=lambda item: item[1]["cuda_total_us"],
            reverse=True,
        )
    ]
    return rows[:60], bucket_rows


def _run_torch_profiler(
    model: GeneReconModel,
    opt: torch.optim.Optimizer,
    out_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    trace_path = out_dir / f"{prefix}_torch_profiler_trace.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        _one_step(model, opt)
    prof.export_chrome_trace(str(trace_path))
    top_ops, buckets = _profiler_buckets(prof)
    return {
        "trace_path": str(trace_path),
        "top_ops": top_ops,
        "buckets": buckets,
    }


def _run_chunked_torch_profiler(
    model: GeneReconModel,
    out_dir: Path,
    prefix: str,
    *,
    family_batch_size: int,
    lr: float,
) -> dict[str, Any]:
    trace_path = out_dir / f"{prefix}_chunked_torch_profiler_trace.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        _chunked_step(
            model,
            family_batch_size=family_batch_size,
            lr=lr,
            profile_cuda_api=False,
            profile_phase="full_step",
        )
    prof.export_chrome_trace(str(trace_path))
    top_ops, buckets = _profiler_buckets(prof)
    return {
        "trace_path": str(trace_path),
        "top_ops": top_ops,
        "buckets": buckets,
    }


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)
    if args.engine == "chunked":
        # Current specieswise tensors do not use the scalar-only fused path.
        os.environ["GPUREC_FUSED_UNIFORM_BACKWARD"] = "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    root = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.label}_specieswise_uniform_{args.fams}"

    dtype = torch.float64 if args.dtype == "fp64" else torch.float32
    genes = _selected_genes(root, args.start, args.fams)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    t_build0 = time.perf_counter()
    model = GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=genes,
        mode="specieswise",
        pibar_mode="uniform",
        device="cuda",
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_Pi=args.fixed_iters_pi,
        neumann_terms=3,
        preprocess_cache_dir=args.cache_dir,
        max_wave_size=args.max_wave_size,
        max_root_wave_size=args.max_root_wave_size,
    )
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t_build0
    shape = _shape(model)
    build_mem = _mem()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.engine == "chunked":
        for _ in range(args.warmups):
            _chunked_step(
                model,
                family_batch_size=args.family_batch_size,
                lr=args.lr,
                profile_cuda_api=False,
                profile_phase=args.profile_phase,
            )
        torch.cuda.synchronize()

        rep_rows = []
        for rep in range(args.reps):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            row = _chunked_step(
                model,
                family_batch_size=args.family_batch_size,
                lr=args.lr,
                profile_cuda_api=args.profile_cuda_api and rep == 0,
                profile_phase=args.profile_phase,
            )
            row["wall_ms"] = (time.perf_counter() - t0) * 1000.0
            row["rep"] = rep
            row["memory"] = _mem()
            rep_rows.append(row)

        torch_profiler = None
        if args.torch_profiler:
            torch_profiler = _run_chunked_torch_profiler(
                model,
                out_dir,
                prefix,
                family_batch_size=args.family_batch_size,
                lr=args.lr,
            )

        summary = {
            "label": args.label,
            "engine": args.engine,
            "dataset": str(root),
            "family_range": [args.start, args.start + args.fams],
            "mode": "specieswise",
            "pibar_mode": "uniform",
            "dtype": args.dtype,
            "device": torch.cuda.get_device_name(0),
            "cuda_capability": torch.cuda.get_device_capability(0),
            "torch_version": torch.__version__,
            "fams": args.fams,
            "family_batch_size": args.family_batch_size,
            "build_s": build_s,
            "shape": shape,
            "build_memory": build_mem,
            "env_flags": {
                key: os.environ[key]
                for key in sorted(os.environ)
                if key.startswith("GPUREC_")
            },
            "timings": {
                "E_event_ms": _summ([row["E_event_ms"] for row in rep_rows]),
                "Pi_event_ms": _summ([row["Pi_event_ms"] for row in rep_rows]),
                "grad_event_ms": _summ([row["grad_event_ms"] for row in rep_rows]),
                "layout_wall_ms": _summ([row["layout_wall_ms"] for row in rep_rows]),
                "optimizer_ms": _summ([row["optimizer_ms"] for row in rep_rows]),
                "full_step_ms_sum": _summ([row["full_step_ms_sum"] for row in rep_rows]),
                "wall_ms": _summ([row["wall_ms"] for row in rep_rows]),
            },
            "losses": [row["nll"] for row in rep_rows],
            "rep_rows": rep_rows,
            "torch_profiler": torch_profiler,
        }
        json_path = out_dir / f"{prefix}_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print("summary_path", json_path)
        return

    for _ in range(args.warmups):
        _one_step(model, opt)
    torch.cuda.synchronize()

    rep_rows = []
    for rep in range(args.reps):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        if args.profile_cuda_api and rep == 0:
            row = _profiled_step(model, opt, phase=args.profile_phase)
        else:
            row = _one_step(model, opt)
        row["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        row["rep"] = rep
        row["memory"] = _mem()
        rep_rows.append(row)

    torch_profiler = None
    if args.torch_profiler:
        torch_profiler = _run_torch_profiler(model, opt, out_dir, prefix)

    summary = {
        "label": args.label,
        "dataset": str(root),
        "family_range": [args.start, args.start + args.fams],
        "mode": "specieswise",
        "pibar_mode": "uniform",
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0),
        "cuda_capability": torch.cuda.get_device_capability(0),
        "torch_version": torch.__version__,
        "fams": args.fams,
        "build_s": build_s,
        "shape": shape,
        "build_memory": build_mem,
        "env_flags": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("GPUREC_")
        },
        "timings": {
            "forward_ms": _summ([row["forward_ms"] for row in rep_rows]),
            "backward_ms": _summ([row["backward_ms"] for row in rep_rows]),
            "optimizer_ms": _summ([row["optimizer_ms"] for row in rep_rows]),
            "full_step_ms_sum": _summ([row["full_step_ms_sum"] for row in rep_rows]),
            "wall_ms": _summ([row["wall_ms"] for row in rep_rows]),
        },
        "losses": [row["loss"] for row in rep_rows],
        "rep_rows": rep_rows,
        "torch_profiler": torch_profiler,
    }
    json_path = out_dir / f"{prefix}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("summary_path", json_path)


if __name__ == "__main__":
    main()
