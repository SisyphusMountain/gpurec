#!/usr/bin/env python3
"""Uniform backward benchmark helper for Proposal 8 profiling.

This intentionally lives outside core code. It times one warmed backward pass
for the existing global/uniform workload and prints static wave/family shape
metrics useful for evaluating family/chunk-aware pruning.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

import torch

from gpurec.api.autograd import _extract_parameters
from gpurec import GeneReconModel
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood


DEFAULT_FLAGS = {
    "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
    "GPUREC_FUSED_DTS_BACKWARD_ACCUM": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL": "tree",
    "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_BACKWARD_LEAF_INDEX": "1",
    "GPUREC_FUSED_WAVE_PARAM_ACCUM": "1",
    "GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES": "1",
    "GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS": "1",
    "GPUREC_DTS_GRAD_MT_TWO_STAGE": "1",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET", "tests/data/test_trees_1000"))
    parser.add_argument("--fams", type=int, default=int(os.getenv("FAMS", "50")))
    parser.add_argument("--start", type=int, default=int(os.getenv("FAMILY_START", "0")))
    parser.add_argument("--reps", type=int, default=int(os.getenv("REPS", "9")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "5")))
    parser.add_argument("--max-wave-size", default=os.getenv("MAX_WAVE_SIZE", "32768"))
    parser.add_argument("--max-root-wave-size", type=int, default=None)
    parser.add_argument("--pruning-threshold", type=float, default=float(os.getenv("PRUNING_THRESHOLD", "1e-6")))
    parser.add_argument("--use-pruning", action=argparse.BooleanOptionalAction, default=os.getenv("USE_PRUNING", "1") != "0")
    parser.add_argument("--stats-only", action="store_true", default=os.getenv("STATS_ONLY", "0") != "0")
    parser.add_argument("--diag-only", action="store_true", default=os.getenv("DIAG_ONLY", "0") != "0")
    parser.add_argument("--diag-chunk-rows", type=int, default=int(os.getenv("DIAG_CHUNK_ROWS", "256")))
    parser.add_argument("--profile-cuda-api", action="store_true", default=os.getenv("PROFILE_CUDA_API", "0") != "0")
    parser.add_argument("--cache-dir", default=os.getenv("PREPROCESS_CACHE_DIR", "/tmp/gpurec_preprocess_cache"))
    args = parser.parse_args()
    mws = str(args.max_wave_size).strip().lower()
    args.max_wave_size = None if mws in ("", "0", "none", "null") else int(mws)
    return args


def _selected_genes(root: Path, start: int, fams: int) -> list[str]:
    genes = sorted(root.glob("g_*.nwk"))
    stop = start + fams
    if stop > len(genes):
        raise ValueError(f"requested families [{start}:{stop}], but only found {len(genes)} genes in {root}")
    return [str(p) for p in genes[start:stop]]


def _family_runs(values: torch.Tensor) -> int:
    if values.numel() == 0:
        return 0
    return int(1 + (values[1:] != values[:-1]).sum().item())


def _print_wave_shape(model: GeneReconModel) -> None:
    wl = model.static.wave_layout
    metas = wl["wave_metas"]
    family_idx = wl.get("family_idx")
    if family_idx is not None:
        family_idx = family_idx.detach().cpu()

    split_rows = sum(int(m["sl"].numel()) if m.get("has_splits", False) else 0 for m in metas)
    leaves = int(model.static.wave_layout["leaf_row_index"].numel())
    roots = int(model.static.root_clade_ids.numel())
    max_w = max((int(m["W"]) for m in metas), default=0)

    print(
        "shape",
        "S", model.n_species,
        "G", model.n_families,
        "C", int(model.static.Pi_shape[0]) if hasattr(model.static, "Pi_shape") else int(sum(m["W"] for m in metas)),
        "waves", len(metas),
        "maxW", max_w,
        "split_rows", split_rows,
        "leaves", leaves,
        "roots", roots,
    )

    rows = []
    for k, meta in enumerate(metas):
        ws = int(meta["start"])
        we = int(meta["end"])
        w = int(meta["W"])
        n_splits = int(meta["sl"].numel()) if meta.get("has_splits", False) else 0
        fanout = (n_splits / w) if w else 0.0
        fams = runs = split_fams = 0
        if family_idx is not None:
            fi = family_idx[ws:we]
            fams = int(torch.unique(fi).numel())
            runs = _family_runs(fi)
            if n_splits:
                parent_fi = family_idx[ws + meta["reduce_idx"].detach().cpu()]
                split_fams = int(torch.unique(parent_fi).numel())
        rows.append((k, ws, w, n_splits, fanout, fams, runs, split_fams))

    print("top_wave_rows k start W split_rows fanout families family_runs split_families")
    for row in sorted(rows, key=lambda r: r[2], reverse=True)[:12]:
        print("%d %d %d %d %.3f %d %d %d" % row)

    print("top_split_rows k start W split_rows fanout families family_runs split_families")
    for row in sorted(rows, key=lambda r: r[3], reverse=True)[:12]:
        print("%d %d %d %d %.3f %d %d %d" % row)


@torch.no_grad()
def _run_pi_backward_diag(model: GeneReconModel, args: argparse.Namespace) -> None:
    static = model.static
    theta = model.theta.detach()
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = _extract_parameters(
        theta, static
    )

    E_out = E_fixed_point(
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
    pi_out = Pi_wave_forward(
        wave_layout=static.wave_layout,
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
        family_idx=(
            static.wave_layout.get("family_idx") if static.genewise else None
        ),
    )
    nll = compute_log_likelihood(
        pi_out["Pi_wave_ordered"],
        E_out["E"],
        static.wave_layout["root_clade_ids"],
    ).sum()

    old_diag = os.environ.get("GPUREC_BACKWARD_FAMILY_CHUNK_DIAG")
    old_rows = os.environ.get("GPUREC_BACKWARD_FAMILY_CHUNK_ROWS")
    os.environ["GPUREC_BACKWARD_FAMILY_CHUNK_DIAG"] = "1"
    os.environ["GPUREC_BACKWARD_FAMILY_CHUNK_ROWS"] = str(args.diag_chunk_rows)
    try:
        pi_bwd = Pi_wave_backward(
            wave_layout=static.wave_layout,
            Pi_star_wave=pi_out["Pi_wave_ordered"],
            Pibar_star_wave=pi_out["Pibar_wave_ordered"],
            E=E_out["E"],
            Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"],
            E_s2=E_out["E_s2"],
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            max_transfer_mat=max_transfer_vec,
            species_helpers=static.species_helpers,
            root_clade_ids_perm=static.wave_layout["root_clade_ids"],
            device=static.device,
            dtype=static.dtype,
            neumann_terms=static.neumann_terms,
            use_pruning=static.use_pruning,
            pruning_threshold=static.pruning_threshold,
            pibar_mode=static.pibar_mode,
            transfer_mat=transfer_mat,
            ancestors_T=static.ancestors_T,
            uniform_pibar_row_max=pi_out.get("uniform_pibar_row_max"),
        )
    finally:
        if old_diag is None:
            os.environ.pop("GPUREC_BACKWARD_FAMILY_CHUNK_DIAG", None)
        else:
            os.environ["GPUREC_BACKWARD_FAMILY_CHUNK_DIAG"] = old_diag
        if old_rows is None:
            os.environ.pop("GPUREC_BACKWARD_FAMILY_CHUNK_ROWS", None)
        else:
            os.environ["GPUREC_BACKWARD_FAMILY_CHUNK_ROWS"] = old_rows

    torch.cuda.synchronize()
    print("diag_loss", f"{float(nll.detach().cpu()):.8f}")
    diag = pi_bwd.get("family_chunk_pruning_diag", {})
    print("family_chunk_pruning_diag")
    for key in sorted(diag):
        print(key, diag[key])


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)

    root = Path(args.dataset)
    genes = _selected_genes(root, args.start, args.fams)

    torch.cuda.empty_cache()
    gc.collect()

    t0 = time.perf_counter()
    model = GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=genes,
        mode="global",
        pibar_mode="uniform",
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_Pi=6,
        neumann_terms=3,
        use_pruning=args.use_pruning,
        pruning_threshold=args.pruning_threshold,
        preprocess_cache_dir=args.cache_dir,
        max_wave_size=args.max_wave_size,
        max_root_wave_size=args.max_root_wave_size,
    )
    torch.cuda.synchronize()

    print("build_s", f"{time.perf_counter() - t0:.3f}")
    print("dataset", root, "family_range", f"{args.start}:{args.start + args.fams}")
    print("use_pruning", args.use_pruning, "pruning_threshold", args.pruning_threshold)
    print("env_flags")
    for key in sorted(k for k in os.environ if k.startswith("GPUREC_")):
        print(key, os.environ[key])
    _print_wave_shape(model)

    if args.stats_only:
        return
    if args.diag_only:
        _run_pi_backward_diag(model, args)
        return

    for _ in range(args.warmups):
        model.zero_grad(set_to_none=True)
        loss = model()
        loss.backward()
        torch.cuda.synchronize()

    times = []
    peaks = []
    for _ in range(args.reps):
        model.zero_grad(set_to_none=True)
        loss = model()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStart()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        if args.profile_cuda_api:
            torch.cuda.cudart().cudaProfilerStop()
        times.append(start.elapsed_time(end))
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)

    print("loss", float(loss.detach().cpu()))
    print(
        "backward_ms",
        "mean", f"{statistics.mean(times):.3f}",
        "median", f"{statistics.median(times):.3f}",
        "min", f"{min(times):.3f}",
        "times", ",".join(f"{t:.3f}" for t in times),
    )
    print("peak_alloc_gb", f"{max(peaks):.3f}")


if __name__ == "__main__":
    main()
