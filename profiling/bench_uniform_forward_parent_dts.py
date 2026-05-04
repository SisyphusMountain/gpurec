#!/usr/bin/env python3
"""Benchmark uniform forward parent-reduced DTS variants.

This helper times the Pi-wave forward interval after parameter extraction and
E fixed point have already been computed. It is intended for proposal 0 in
docs/uniform-forward-optimization-proposals.md.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
from pathlib import Path

import torch

from gpurec import GeneReconModel
from gpurec.api.model import _build_static_state
from gpurec.api.autograd import _extract_parameters
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import (
    E_fixed_point,
    compute_log_likelihood,
    compute_log_likelihood_root_rows,
)
from gpurec.core.model import GeneDataset


DEFAULT_FLAGS = {
    "GPUREC_UNIFORM_PINGPONG": "1",
}


class _StaticBenchModel:
    def __init__(self, *, static, theta: torch.Tensor, n_species: int, n_families: int):
        self._static = static
        self.theta = theta
        self.n_species = n_species
        self.n_families = n_families

    @property
    def static(self):
        return self._static


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no", "row", "rows", "expanded")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.getenv("DATASET", "tests/data/test_trees_1000"))
    parser.add_argument("--fams", type=int, default=int(os.getenv("FAMS", "50")))
    parser.add_argument("--start", type=int, default=int(os.getenv("FAMILY_START", "0")))
    parser.add_argument("--reps", type=int, default=int(os.getenv("REPS", "9")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "5")))
    parser.add_argument("--fixed-iters", type=int, default=int(os.getenv("FIXED_ITERS_PI", "6")))
    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp64"),
        default=os.getenv("DTYPE", "fp32"),
        help="Floating-point dtype used for the model and timed Pi interval.",
    )
    parser.add_argument("--max-wave-size", default=os.getenv("MAX_WAVE_SIZE", "32768"))
    parser.add_argument("--max-root-wave-size", default=os.getenv("MAX_ROOT_WAVE_SIZE", ""))
    parser.add_argument("--cache-dir", default=os.getenv("PREPROCESS_CACHE_DIR", "/tmp/gpurec_preprocess_cache"))
    parser.add_argument(
        "--mode",
        choices=("global", "specieswise", "genewise", "genewise_specieswise"),
        default=os.getenv("MODE", "global"),
        help="Parameter-sharing mode used to construct the benchmark state.",
    )
    parser.add_argument("--variant", choices=("old", "new"), default=os.getenv("VARIANT", "new"))
    parser.add_argument(
        "--min-splits",
        type=int,
        default=int(os.getenv("MIN_SPLITS", os.getenv("GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS", "0"))),
    )
    parser.add_argument(
        "--impl",
        default=os.getenv("IMPL", os.getenv("GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL", "tiled")),
    )
    parser.add_argument(
        "--tile-splits",
        type=int,
        default=int(os.getenv("TILE_SPLITS", os.getenv("GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS", "64"))),
    )
    parser.add_argument(
        "--ge2-only",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("GE2_ONLY", os.getenv("GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY", "1")) != "0",
    )
    parser.add_argument("--compare", action="store_true", default=os.getenv("COMPARE", "0") != "0")
    parser.add_argument("--full-diff-max-fams", type=int, default=int(os.getenv("FULL_DIFF_MAX_FAMS", "10")))
    parser.add_argument("--profile-cuda-api", action="store_true", default=os.getenv("PROFILE_CUDA_API", "0") != "0")
    parser.add_argument("--stats-only", action="store_true", default=os.getenv("STATS_ONLY", "0") != "0")
    parser.add_argument(
        "--need-pibar",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("NEED_PIBAR", "1") != "0",
    )
    parser.add_argument(
        "--root-rows",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ROOT_ROWS", "0") != "0",
        help="Return only root Pi rows instead of the full wave-ordered Pi output.",
    )
    parser.add_argument(
        "--leaf-index",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("GPUREC_FORWARD_LEAF_INDEX", "1") != "0",
        help="Enable compact uniform leaf-index term in the wave-step kernel.",
    )
    parser.add_argument(
        "--family-indexed-consts",
        action=argparse.BooleanOptionalAction,
        default=_env_enabled("GPUREC_FORWARD_FAMILY_INDEXED_CONSTS", True),
        help="Load genewise uniform constants from [G,S] inside Triton instead of pre-expanding [W,S].",
    )
    parser.add_argument(
        "--compare-family-consts",
        action="store_true",
        default=os.getenv("COMPARE_FAMILY_CONSTS", "0") != "0",
        help="Compare row-expanded constants against family-indexed constants before timing.",
    )
    parser.add_argument(
        "--family-indexed-dts-params",
        action=argparse.BooleanOptionalAction,
        default=_env_enabled("GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS", True),
        help="Load genewise uniform DTS pD/pS from [G] or [G,S] inside kernels instead of pre-expanding per split.",
    )
    parser.add_argument(
        "--compare-family-dts-params",
        action="store_true",
        default=os.getenv("COMPARE_FAMILY_DTS_PARAMS", "0") != "0",
        help="Compare row-expanded DTS pD/pS against family-indexed DTS pD/pS before timing.",
    )
    parser.add_argument(
        "--topology-int32",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("GPUREC_FORWARD_TOPOLOGY_INT32", "1") != "0",
    )
    parser.add_argument(
        "--wave-block-s",
        type=int,
        default=int(os.getenv("GPUREC_FORWARD_WAVE_BLOCK_S", "0")),
        help="Override GPUREC_FORWARD_WAVE_BLOCK_S; 0 keeps the default heuristic.",
    )
    parser.add_argument(
        "--wave-num-warps",
        type=int,
        default=int(os.getenv("GPUREC_FORWARD_WAVE_NUM_WARPS", "0")),
        help="Override GPUREC_FORWARD_WAVE_NUM_WARPS; 0 keeps the default.",
    )
    parser.add_argument(
        "--overlap-mode",
        choices=("off", "next", "ready"),
        default=os.getenv("OVERLAP_MODE", os.getenv("GPUREC_FORWARD_DTS_OVERLAP_MODE", "off")),
    )
    parser.add_argument(
        "--overlap-max-pending",
        type=int,
        default=int(os.getenv("GPUREC_FORWARD_DTS_OVERLAP_MAX_PENDING", "2")),
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
        raise ValueError(f"requested families [{start}:{stop}], but only found {len(genes)} genes")
    return [str(p) for p in genes[start:stop]]


def _set_variant(args: argparse.Namespace, variant: str) -> None:
    os.environ["GPUREC_FORWARD_PARENT_REDUCED_DTS"] = "0" if variant == "old" else "1"
    os.environ["GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS"] = str(args.min_splits)
    os.environ["GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL"] = str(args.impl)
    os.environ["GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS"] = str(args.tile_splits)
    os.environ["GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY"] = "1" if args.ge2_only else "0"


def _set_family_indexed_consts(enabled: bool) -> None:
    os.environ["GPUREC_FORWARD_FAMILY_INDEXED_CONSTS"] = "1" if enabled else "0"


def _set_family_indexed_dts_params(enabled: bool) -> None:
    os.environ["GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS"] = "1" if enabled else "0"


def _time_cuda_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _tensor_max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    chunk_elems = 16 * 1024 * 1024
    if a.numel() <= chunk_elems:
        return float((a - b).abs().max().detach().cpu())
    a_flat = a.contiguous().view(-1)
    b_flat = b.contiguous().view(-1)
    scratch = torch.empty((chunk_elems,), dtype=a.dtype, device=a.device)
    max_diff = 0.0
    for start in range(0, a_flat.numel(), chunk_elems):
        stop = min(start + chunk_elems, a_flat.numel())
        span = stop - start
        tmp = scratch[:span]
        torch.sub(a_flat[start:stop], b_flat[start:stop], out=tmp)
        tmp.abs_()
        max_diff = max(max_diff, float(tmp.max().detach().cpu()))
    return max_diff


def _print_shape(model: GeneReconModel) -> None:
    metas = model.static.wave_layout["wave_metas"]
    split_rows = sum(int(m["sl"].numel()) if m.get("has_splits", False) else 0 for m in metas)
    ready_overlap_split_rows = sum(
        int(m["sl"].numel())
        for m in metas
        if m.get("has_splits", False) and int(m.get("dts_overlap_gap", 0)) > 0
    )
    ready_overlap_waves = sum(
        1 for m in metas
        if m.get("has_splits", False) and int(m.get("dts_overlap_gap", 0)) > 0
    )
    max_ready_gap = max((int(m.get("dts_overlap_gap", 0)) for m in metas), default=0)
    ge2_groups = sum(int(m.get("n_ge2_clades", 0)) for m in metas)
    eq1_rows = sum(int(m.get("n_eq1", 0)) for m in metas)
    C = sum(int(m["W"]) for m in metas)
    max_w = max((int(m["W"]) for m in metas), default=0)
    max_splits = max((int(m["sl"].numel()) for m in metas if m.get("has_splits", False)), default=0)
    max_fanout = max((int(m.get("ge2_max_fanout", 0) or 0) for m in metas), default=0)
    print(
        "shape",
        "S", model.n_species,
        "G", model.n_families,
        "C", C,
        "waves", len(metas),
        "maxW", max_w,
        "split_rows", split_rows,
        "ready_overlap_waves", ready_overlap_waves,
        "ready_overlap_split_rows", ready_overlap_split_rows,
        "max_ready_gap", max_ready_gap,
        "eq1_rows", eq1_rows,
        "ge2_groups", ge2_groups,
        "max_splits", max_splits,
        "max_ge2_fanout", max_fanout,
    )
    rows = []
    for k, meta in enumerate(metas):
        if not meta.get("has_splits", False):
            continue
        w = int(meta["W"])
        n_splits = int(meta["sl"].numel())
        n_eq1 = int(meta.get("n_eq1", 0))
        n_ge2 = int(meta.get("n_ge2_clades", 0))
        max_ge2 = int(meta.get("ge2_max_fanout", 0) or 0)
        ready_after = int(meta.get("dts_ready_after", -1))
        overlap_gap = int(meta.get("dts_overlap_gap", 0))
        rows.append((k, int(meta["start"]), w, n_splits, ready_after, overlap_gap, n_eq1, n_ge2, max_ge2))
    print("top_split_waves k start W split_rows dts_ready_after overlap_gap n_eq1 n_ge2_groups max_ge2_fanout")
    for row in sorted(rows, key=lambda r: r[3], reverse=True)[:12]:
        print("%d %d %d %d %d %d %d %d %d" % row)


def _prepare(args: argparse.Namespace) -> tuple[GeneReconModel, dict, tuple]:
    root = Path(args.dataset)
    genes = _selected_genes(root, args.start, args.fams)
    dtype = torch.float64 if args.dtype == "fp64" else torch.float32
    if args.mode == "genewise_specieswise":
        dataset = GeneDataset(
            str(root / "sp.nwk"),
            genes,
            genewise=True,
            specieswise=True,
            pairwise=False,
            dtype=dtype,
            device=torch.device("cuda"),
            preprocess_cache_dir=args.cache_dir,
            retain_dense_species_matrices=False,
        )
        static = _build_static_state(
            dataset,
            pibar_mode="uniform",
            max_iters_E=2000,
            tol_E=1e-8,
            max_iters_Pi=2000,
            tol_Pi=1e-6,
            fixed_iters_Pi=args.fixed_iters,
            neumann_terms=3,
            use_pruning=True,
            pruning_threshold=1e-6,
            cg_tol=1e-8,
            cg_maxiter=500,
            gmres_restart=40,
            max_wave_size=args.max_wave_size,
            max_root_wave_size=args.max_root_wave_size,
        )
        base = torch.log2(torch.tensor([0.05, 0.05, 0.05], dtype=dtype, device="cuda"))
        theta = base.view(1, 1, 3).expand(len(genes), int(dataset.S), -1).clone()
        model = _StaticBenchModel(
            static=static,
            theta=theta,
            n_species=int(dataset.S),
            n_families=len(genes),
        )
    else:
        model = GeneReconModel.from_trees(
            str(root / "sp.nwk"),
            genes,
            mode=args.mode,
            pibar_mode="uniform",
            device="cuda",
            dtype=dtype,
            theta_init_rates=(0.05, 0.05, 0.05),
            fixed_iters_Pi=args.fixed_iters,
            max_wave_size=args.max_wave_size,
            max_root_wave_size=args.max_root_wave_size,
            preprocess_cache_dir=args.cache_dir,
        )
    static = model.static
    theta = model.theta.detach()
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = _extract_parameters(theta, static)
    E_out = E_fixed_point(
        species_helpers=static.species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=transfer_mat,
        max_transfer_mat=max_transfer_vec,
        max_iters=static.max_iters_E,
        tolerance=static.tol_E,
        warm_start_E=None,
        dtype=static.dtype,
        device=static.device,
        pibar_mode=static.pibar_mode,
        ancestors_T=static.ancestors_T,
    )
    return model, E_out, (log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec)


def _run_pi(
    model: GeneReconModel,
    E_out: dict,
    params: tuple,
    *,
    need_pibar: bool,
    overlap_mode: str,
    root_rows: bool,
) -> tuple[torch.Tensor, dict]:
    static = model.static
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = params
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
        overlap_streams=overlap_mode != "off",
        pibar_mode=static.pibar_mode,
        return_original=False,
        need_pibar=need_pibar,
        return_root_rows=root_rows,
        family_idx=(static.wave_layout.get("family_idx") if static.genewise else None),
    )
    if root_rows:
        nll = compute_log_likelihood_root_rows(pi_out["Pi_root_rows"], E_out["E"]).sum()
    else:
        nll = compute_log_likelihood(
            pi_out["Pi_wave_ordered"],
            E_out["E"],
            static.wave_layout["root_clade_ids"],
        ).sum()
    return nll, pi_out


def _compare(args: argparse.Namespace, model: GeneReconModel, E_out: dict, params: tuple) -> None:
    _set_variant(args, "old")
    old_nll, old_out = _run_pi(
        model, E_out, params,
        need_pibar=args.need_pibar,
        overlap_mode=args.overlap_mode,
        root_rows=args.root_rows,
    )
    torch.cuda.synchronize()
    old_nll_value = float(old_nll.detach().cpu())
    keep_full_diff = args.fams <= args.full_diff_max_fams
    if not keep_full_diff:
        del old_out
        torch.cuda.empty_cache()
    _set_variant(args, "new")
    new_nll, new_out = _run_pi(
        model, E_out, params,
        need_pibar=args.need_pibar,
        overlap_mode=args.overlap_mode,
        root_rows=args.root_rows,
    )
    torch.cuda.synchronize()
    nll_diff = float((new_nll - old_nll).abs().detach().cpu())
    print("compare", "old_nll", old_nll_value, "new_nll", float(new_nll.detach().cpu()), "nll_abs_diff", nll_diff)
    if keep_full_diff and not args.root_rows:
        pi_diff = _tensor_max_abs_diff(old_out["Pi_wave_ordered"], new_out["Pi_wave_ordered"])
        if old_out["Pibar_wave_ordered"] is not None and new_out["Pibar_wave_ordered"] is not None:
            pibar_diff = _tensor_max_abs_diff(old_out["Pibar_wave_ordered"], new_out["Pibar_wave_ordered"])
        else:
            pibar_diff = "not_returned"
        print("compare_tensors", "Pi_max_abs", pi_diff, "Pibar_max_abs", pibar_diff)
        del old_out
    del new_out, old_nll, new_nll
    gc.collect()
    torch.cuda.empty_cache()


def _compare_family_consts(args: argparse.Namespace, model: GeneReconModel, E_out: dict, params: tuple) -> None:
    _set_variant(args, args.variant)
    _set_family_indexed_consts(False)
    row_nll, row_out = _run_pi(
        model, E_out, params,
        need_pibar=args.need_pibar,
        overlap_mode=args.overlap_mode,
        root_rows=args.root_rows,
    )
    torch.cuda.synchronize()
    row_nll_value = float(row_nll.detach().cpu())
    keep_full_diff = args.fams <= args.full_diff_max_fams
    if not keep_full_diff:
        del row_out
        torch.cuda.empty_cache()

    _set_family_indexed_consts(True)
    family_nll, family_out = _run_pi(
        model, E_out, params,
        need_pibar=args.need_pibar,
        overlap_mode=args.overlap_mode,
        root_rows=args.root_rows,
    )
    torch.cuda.synchronize()
    nll_diff = float((family_nll - row_nll).abs().detach().cpu())
    print(
        "compare_family_consts",
        "row_nll", row_nll_value,
        "family_nll", float(family_nll.detach().cpu()),
        "nll_abs_diff", nll_diff,
    )
    if keep_full_diff and not args.root_rows:
        pi_diff = _tensor_max_abs_diff(row_out["Pi_wave_ordered"], family_out["Pi_wave_ordered"])
        if row_out["Pibar_wave_ordered"] is not None and family_out["Pibar_wave_ordered"] is not None:
            pibar_diff = _tensor_max_abs_diff(row_out["Pibar_wave_ordered"], family_out["Pibar_wave_ordered"])
        else:
            pibar_diff = "not_returned"
        print("compare_family_const_tensors", "Pi_max_abs", pi_diff, "Pibar_max_abs", pibar_diff)
        del row_out
    del family_out, row_nll, family_nll
    _set_family_indexed_consts(args.family_indexed_consts)
    gc.collect()
    torch.cuda.empty_cache()


def _compare_family_dts_params(args: argparse.Namespace, model: GeneReconModel, E_out: dict, params: tuple) -> None:
    _set_variant(args, args.variant)
    _set_family_indexed_consts(args.family_indexed_consts)

    compare_need_pibar = True
    _set_family_indexed_dts_params(False)
    row_nll, row_out = _run_pi(
        model, E_out, params,
        need_pibar=compare_need_pibar,
        overlap_mode=args.overlap_mode,
        root_rows=args.root_rows,
    )
    torch.cuda.synchronize()
    row_nll_value = float(row_nll.detach().cpu())
    keep_full_diff = args.fams <= args.full_diff_max_fams
    if not keep_full_diff:
        del row_out
        torch.cuda.empty_cache()

    _set_family_indexed_dts_params(True)
    family_nll, family_out = _run_pi(
        model, E_out, params,
        need_pibar=compare_need_pibar,
        overlap_mode=args.overlap_mode,
        root_rows=args.root_rows,
    )
    torch.cuda.synchronize()
    nll_diff = float((family_nll - row_nll).abs().detach().cpu())
    print(
        "compare_family_dts_params",
        "row_expanded_nll", row_nll_value,
        "family_indexed_nll", float(family_nll.detach().cpu()),
        "nll_abs_diff", nll_diff,
        "need_pibar", int(compare_need_pibar),
    )
    if keep_full_diff:
        if args.root_rows:
            pi_diff = _tensor_max_abs_diff(row_out["Pi_root_rows"], family_out["Pi_root_rows"])
            pi_label = "Pi_root_rows_max_abs"
        else:
            pi_diff = _tensor_max_abs_diff(row_out["Pi_wave_ordered"], family_out["Pi_wave_ordered"])
            pi_label = "Pi_max_abs"
        if row_out["Pibar_wave_ordered"] is not None and family_out["Pibar_wave_ordered"] is not None:
            pibar_diff = _tensor_max_abs_diff(row_out["Pibar_wave_ordered"], family_out["Pibar_wave_ordered"])
        else:
            pibar_diff = "not_returned"
        print("compare_family_dts_param_tensors", pi_label, pi_diff, "Pibar_max_abs", pibar_diff)
        del row_out
    del family_out, row_nll, family_nll
    _set_family_indexed_dts_params(args.family_indexed_dts_params)
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = _parse_args()
    for key, value in DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)
    os.environ["GPUREC_FORWARD_LEAF_INDEX"] = "1" if args.leaf_index else "0"
    _set_family_indexed_consts(args.family_indexed_consts)
    _set_family_indexed_dts_params(args.family_indexed_dts_params)
    os.environ["GPUREC_FORWARD_TOPOLOGY_INT32"] = "1" if args.topology_int32 else "0"
    if args.wave_block_s > 0:
        os.environ["GPUREC_FORWARD_WAVE_BLOCK_S"] = str(args.wave_block_s)
    else:
        os.environ.pop("GPUREC_FORWARD_WAVE_BLOCK_S", None)
    if args.wave_num_warps > 0:
        os.environ["GPUREC_FORWARD_WAVE_NUM_WARPS"] = str(args.wave_num_warps)
    else:
        os.environ.pop("GPUREC_FORWARD_WAVE_NUM_WARPS", None)
    os.environ["GPUREC_FORWARD_DTS_OVERLAP_MODE"] = args.overlap_mode
    os.environ["GPUREC_FORWARD_DTS_OVERLAP_MAX_PENDING"] = str(args.overlap_max_pending)
    _set_variant(args, args.variant)
    model, E_out, params = _prepare(args)
    _print_shape(model)
    if args.stats_only:
        return
    if args.compare:
        _compare(args, model, E_out, params)
    if args.compare_family_consts:
        _compare_family_consts(args, model, E_out, params)
    if args.compare_family_dts_params:
        _compare_family_dts_params(args, model, E_out, params)

    for _ in range(args.warmups):
        _set_variant(args, args.variant)
        _set_family_indexed_consts(args.family_indexed_consts)
        _set_family_indexed_dts_params(args.family_indexed_dts_params)
        _, out = _run_pi(
            model, E_out, params,
            need_pibar=args.need_pibar,
            overlap_mode=args.overlap_mode,
            root_rows=args.root_rows,
        )
        del out
    torch.cuda.synchronize()

    times = []
    nll_value = None
    if args.profile_cuda_api:
        torch.cuda.profiler.start()
    for _ in range(args.reps):
        _set_variant(args, args.variant)
        _set_family_indexed_consts(args.family_indexed_consts)
        _set_family_indexed_dts_params(args.family_indexed_dts_params)
        ms, (nll, out) = _time_cuda_ms(
            lambda: _run_pi(
                model, E_out, params,
                need_pibar=args.need_pibar,
                overlap_mode=args.overlap_mode,
                root_rows=args.root_rows,
            )
        )
        times.append(ms)
        nll_value = float(nll.detach().cpu())
        live_after_last_gib = torch.cuda.memory_allocated() / (1024 ** 3)
        del out, nll
    if args.profile_cuda_api:
        torch.cuda.profiler.stop()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(
        "timing",
        "variant", args.variant,
        "mode", args.mode,
        "dtype", args.dtype,
        "min_splits", args.min_splits,
        "impl", args.impl,
        "tile_splits", args.tile_splits,
        "ge2_only", int(args.ge2_only),
        "need_pibar", int(args.need_pibar),
        "root_rows", int(args.root_rows),
        "leaf_index", int(args.leaf_index),
        "family_indexed_consts", int(args.family_indexed_consts),
        "compare_family_consts", int(args.compare_family_consts),
        "family_indexed_dts_params", int(args.family_indexed_dts_params),
        "compare_family_dts_params", int(args.compare_family_dts_params),
        "topology_int32", int(args.topology_int32),
        "wave_block_s", args.wave_block_s,
        "wave_num_warps", args.wave_num_warps,
        "overlap_mode", args.overlap_mode,
        "overlap_max_pending", args.overlap_max_pending,
        "reps", len(times),
        "median_ms", statistics.median(times),
        "mean_ms", statistics.mean(times),
        "min_ms", min(times),
        "max_ms", max(times),
        "nll", nll_value,
        "live_after_last_gib", live_after_last_gib,
        "peak_gib", peak,
    )


if __name__ == "__main__":
    main()
