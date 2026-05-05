#!/usr/bin/env python3
"""Benchmark and compare specieswise uniform forward/backward paths."""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from gpurec import GeneReconModel
from gpurec.api.autograd import _extract_parameters
from gpurec.core.batching import (
    build_wave_layout,
    collate_gene_families,
    collate_wave,
    split_phase_waves,
)
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import (
    E_fixed_point,
    compute_log_likelihood,
    compute_log_likelihood_root_rows,
)
from gpurec.core.model import GeneDataset
from gpurec.core.scheduling import compute_clade_waves


REFERENCE_FLAGS = {
    "GPUREC_FORWARD_LEAF_INDEX": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "0",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "0",
    "GPUREC_FUSED_UNIFORM_BACKWARD": "0",
    "GPUREC_KERNELIZED_BACKWARD_DTS": "0",
    "GPUREC_BACKWARD_PARENT_REDUCED_DTS": "0",
    "GPUREC_FUSED_CROSS_PIBAR_VJP": "0",
    "GPUREC_BACKWARD_LEAF_INDEX": "0",
    "GPUREC_DTS_PIBAR_UD_FUSION": "0",
}

OPTIMIZED_FLAGS = {
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_FORWARD_LEAF_INDEX": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
    "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
    "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
    "GPUREC_BACKWARD_PARENT_REDUCED_DTS": "tiled",
    "GPUREC_BACKWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL": "tree",
    "GPUREC_BACKWARD_LEAF_INDEX": "1",
    "GPUREC_FUSED_DTS_BACKWARD_ACCUM": "1",
    "GPUREC_DTS_PIBAR_UD_FUSION": "1",
    "GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES": "1",
    "GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS": "1",
    "GPUREC_DTS_GRAD_MT_TWO_STAGE": "1",
}


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
    parser.add_argument("--fams", type=int, default=int(os.getenv("FAMS", "50")))
    parser.add_argument("--start", type=int, default=int(os.getenv("FAMILY_START", "0")))
    parser.add_argument("--reps", type=int, default=int(os.getenv("REPS", "7")))
    parser.add_argument("--warmups", type=int, default=int(os.getenv("WARMUPS", "3")))
    parser.add_argument("--fixed-iters", type=int, default=int(os.getenv("FIXED_ITERS_PI", "6")))
    parser.add_argument("--max-iters-pi", type=int, default=int(os.getenv("MAX_ITERS_PI", "2000")))
    parser.add_argument("--tol-pi", type=float, default=float(os.getenv("TOL_PI", "1e-6")))
    parser.add_argument("--max-iters-e", type=int, default=int(os.getenv("MAX_ITERS_E", "2000")))
    parser.add_argument("--tol-e", type=float, default=float(os.getenv("TOL_E", "1e-8")))
    parser.add_argument("--neumann-terms", type=int, default=int(os.getenv("NEUMANN_TERMS", "3")))
    parser.add_argument("--cache-dir", default=os.getenv("PREPROCESS_CACHE_DIR", "/tmp/gpurec_preprocess_cache"))
    parser.add_argument("--max-wave-size", default=os.getenv("MAX_WAVE_SIZE", "32768"))
    parser.add_argument("--max-root-wave-size", default=os.getenv("MAX_ROOT_WAVE_SIZE", ""))
    parser.add_argument("--family-chunk-size", type=int, default=int(os.getenv("FAMILY_CHUNK_SIZE", "0")))
    parser.add_argument("--dtype", choices=("fp32", "fp64"), default=os.getenv("DTYPE", "fp32"))
    parser.add_argument("--variant", choices=("reference", "optimized"), default=os.getenv("VARIANT", "optimized"))
    parser.add_argument("--target", choices=("forward", "backward", "both"), default=os.getenv("TARGET", "both"))
    parser.add_argument(
        "--theta-profile",
        choices=("constant", "deterministic"),
        default=os.getenv("THETA_PROFILE", "deterministic"),
        help="Specieswise theta profile. constant is useful for global/specieswise semantic checks.",
    )
    parser.add_argument("--theta-rate", type=float, default=float(os.getenv("THETA_RATE", "0.05")))
    parser.add_argument("--compare-forward", action="store_true", default=os.getenv("COMPARE_FORWARD", "0") != "0")
    parser.add_argument("--compare-backward", action="store_true", default=os.getenv("COMPARE_BACKWARD", "0") != "0")
    parser.add_argument("--compare-semantics", action="store_true", default=os.getenv("COMPARE_SEMANTICS", "0") != "0")
    parser.add_argument("--full-diff-max-fams", type=int, default=int(os.getenv("FULL_DIFF_MAX_FAMS", "10")))
    parser.add_argument("--stats-only", action="store_true", default=os.getenv("STATS_ONLY", "0") != "0")
    parser.add_argument("--print-commands", action="store_true", default=os.getenv("PRINT_COMMANDS", "0") != "0")
    parser.add_argument(
        "--need-pibar",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("NEED_PIBAR", "1") != "0",
    )
    parser.add_argument(
        "--root-rows",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ROOT_ROWS", "0") != "0",
        help="Forward timing returns root rows only when enabled.",
    )
    parser.add_argument(
        "--use-pruning",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("USE_PRUNING", "1") != "0",
    )
    parser.add_argument(
        "--pruning-threshold",
        type=float,
        default=float(os.getenv("PRUNING_THRESHOLD", "1e-6")),
    )
    args = parser.parse_args()
    args.max_wave_size = _parse_optional_int(args.max_wave_size)
    args.max_root_wave_size = _parse_optional_int(args.max_root_wave_size)
    return args


def _print_commands(args: argparse.Namespace) -> None:
    script = "profiling/bench_specieswise_uniform.py"
    common = [
        "python",
        script,
        "--dataset",
        str(args.dataset),
        "--variant",
        "optimized",
        "--reps",
        str(args.reps),
        "--warmups",
        str(args.warmups),
    ]
    for fams in (50, 150, 1000):
        compare = ["--compare-forward"] if fams == 50 else []
        chunk = ["--family-chunk-size", "50"] if fams == 1000 else []
        print("cmd_forward_%d" % fams, " ".join(common + [
            "--fams",
            str(fams),
            "--target",
            "forward",
            "--root-rows",
            "--no-need-pibar",
        ] + chunk + compare))
    for fams in (50, 150, 1000):
        print("cmd_backward_%d" % fams, " ".join(common + [
            "--fams",
            str(fams),
            "--target",
            "backward",
        ]))
    print(
        "cmd_backward_compare_small",
        "python profiling/bench_specieswise_uniform.py --dataset tests/data/test_trees_100 "
        "--variant optimized --reps 2 --warmups 1 --fams 50 --target backward --compare-backward "
        "--max-iters-pi 500 --tol-pi 1e-3 --max-iters-e 500 --tol-e 1e-3 --fixed-iters 3 --neumann-terms 2",
    )


def _selected_genes(root: Path, start: int, fams: int) -> list[str]:
    genes = sorted(root.glob("g_*.nwk"))
    stop = start + fams
    if stop > len(genes):
        raise ValueError(f"requested families [{start}:{stop}], but only found {len(genes)} genes")
    return [str(p) for p in genes[start:stop]]


def _set_variant(variant: str) -> None:
    flags = REFERENCE_FLAGS if variant == "reference" else OPTIMIZED_FLAGS
    for key, value in flags.items():
        os.environ[key] = value


def _dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.float64 if args.dtype == "fp64" else torch.float32


def _base_theta(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.log2(torch.tensor([args.theta_rate] * 3, dtype=dtype, device=device))


def _specieswise_theta(
    n_species: int,
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    base = _base_theta(args, dtype, device)
    theta = base.unsqueeze(0).expand(n_species, 3).clone()
    if args.theta_profile == "deterministic":
        offsets = torch.linspace(-0.08, 0.08, n_species, dtype=dtype, device=device)
        theta[:, 0] += offsets
        theta[:, 1] -= 0.5 * offsets
        theta[:, 2] += 0.25 * torch.sin(offsets * math.pi)
    return theta


def _make_model(args: argparse.Namespace, genes: list[str], *, mode: str = "specieswise") -> GeneReconModel:
    root = Path(args.dataset)
    dtype = _dtype(args)
    model = GeneReconModel.from_trees(
        str(root / "sp.nwk"),
        genes,
        mode=mode,
        pibar_mode="uniform",
        device="cuda",
        dtype=dtype,
        theta_init_rates=(args.theta_rate, args.theta_rate, args.theta_rate),
        fixed_iters_Pi=args.fixed_iters,
        max_iters_Pi=args.max_iters_pi,
        tol_Pi=args.tol_pi,
        max_iters_E=args.max_iters_e,
        tol_E=args.tol_e,
        neumann_terms=args.neumann_terms,
        use_pruning=args.use_pruning,
        pruning_threshold=args.pruning_threshold,
        preprocess_cache_dir=args.cache_dir,
        max_wave_size=args.max_wave_size,
        max_root_wave_size=args.max_root_wave_size,
    )
    with torch.no_grad():
        if mode == "specieswise":
            model.theta.copy_(_specieswise_theta(model.n_species, args, dtype, torch.device("cuda")))
        elif mode == "global":
            model.theta.copy_(_base_theta(args, dtype, torch.device("cuda")))
    return model


class _ForwardBenchModel:
    def __init__(self, *, dataset: GeneDataset, theta: torch.Tensor, static: SimpleNamespace):
        self._dataset = dataset
        self.theta = theta
        self._static = static

    @property
    def static(self):
        return self._static

    @property
    def n_species(self) -> int:
        return int(self._dataset.S)

    @property
    def n_families(self) -> int:
        return len(self._dataset.families)


def _make_forward_state(args: argparse.Namespace, genes: list[str]) -> _ForwardBenchModel:
    root = Path(args.dataset)
    dtype = _dtype(args)
    device = torch.device("cuda")
    dataset = GeneDataset(
        str(root / "sp.nwk"),
        genes,
        genewise=False,
        specieswise=True,
        pairwise=False,
        dtype=dtype,
        device=device,
        preprocess_cache_dir=args.cache_dir,
        retain_dense_species_matrices=False,
    )
    species_helpers, ancestors_T = dataset._species_helpers_for_mode(
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
    )
    theta = _specieswise_theta(int(dataset.S), args, dtype, device)
    static = SimpleNamespace(
        device=device,
        dtype=dtype,
        wave_layout=None,
        species_helpers=species_helpers,
        root_clade_ids=None,
        unnorm_row_max=dataset.unnorm_row_max.to(device=device, dtype=dtype),
        transfer_mat_unnormalized=None,
        ancestors_T=ancestors_T,
        genewise=False,
        specieswise=True,
        pibar_mode="uniform",
        max_iters_E=args.max_iters_e,
        tol_E=args.tol_e,
        max_iters_Pi=args.max_iters_pi,
        tol_Pi=args.tol_pi,
        fixed_iters_Pi=args.fixed_iters,
    )
    return _ForwardBenchModel(dataset=dataset, theta=theta, static=static)


def _chunk_indices(n_families: int, chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0 or chunk_size >= n_families:
        return [list(range(n_families))]
    return [
        list(range(start, min(start + chunk_size, n_families)))
        for start in range(0, n_families, chunk_size)
    ]


def _build_chunk_wave_layout(model: GeneReconModel, indices: list[int], args: argparse.Namespace) -> dict:
    dataset = model._dataset
    device = dataset.device
    dtype = dataset.dtype
    items = [
        {
            "ccp": dataset.families[idx]["ccp_helpers"],
            "leaf_row_index": dataset.families[idx]["leaf_row_index"],
            "leaf_col_index": dataset.families[idx]["leaf_col_index"],
            "root_clade_id": int(dataset.families[idx]["root_clade_id"]),
        }
        for idx in indices
    ]
    batched = collate_gene_families(items, dtype=dtype, device=device)
    families_waves = []
    families_phases = []
    for idx in indices:
        waves_i, phases_i = compute_clade_waves(dataset.families[idx]["ccp_helpers"])
        families_waves.append(waves_i)
        families_phases.append(phases_i)

    cross_waves = collate_wave(
        families_waves,
        [m["clade_offset"] for m in batched["family_meta"]],
    )
    max_n_waves = max(len(p) for p in families_phases)
    cross_phases = [
        max(fp[k] for fp in families_phases if k < len(fp))
        for k in range(max_n_waves)
    ]
    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=None,
        max_wave_size=args.max_wave_size,
    )
    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=3,
        max_wave_size=args.max_root_wave_size,
    )
    return build_wave_layout(
        waves=cross_waves,
        phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device,
        dtype=dtype,
        family_clade_counts=[m["C"] for m in batched["family_meta"]],
        family_clade_offsets=[m["clade_offset"] for m in batched["family_meta"]],
    )


def _build_forward_chunks(model: GeneReconModel, args: argparse.Namespace) -> list[dict]:
    indices_by_chunk = _chunk_indices(model.n_families, args.family_chunk_size)
    if len(indices_by_chunk) == 1 and model.static.wave_layout is not None:
        return [model.static.wave_layout]
    return [_build_chunk_wave_layout(model, indices, args) for indices in indices_by_chunk]


def _prepare_forward(model: GeneReconModel):
    static = model.static
    params = _extract_parameters(model.theta.detach(), static)
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = params
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
    return E_out, params


def _run_pi(
    model: GeneReconModel,
    E_out: dict,
    params: tuple,
    *,
    need_pibar: bool,
    root_rows: bool,
    wave_layout: dict | None = None,
):
    static = model.static
    if wave_layout is None:
        wave_layout = static.wave_layout
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = params
    pi_out = Pi_wave_forward(
        wave_layout=wave_layout,
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
        need_pibar=need_pibar,
        return_root_rows=root_rows,
        family_idx=None,
    )
    if root_rows:
        nll = compute_log_likelihood_root_rows(pi_out["Pi_root_rows"], E_out["E"]).sum()
    else:
        nll = compute_log_likelihood(
            pi_out["Pi_wave_ordered"],
            E_out["E"],
            wave_layout["root_clade_ids"],
        ).sum()
    return nll, pi_out


def _run_forward_once(
    model: GeneReconModel,
    E_out: dict,
    params: tuple,
    *,
    need_pibar: bool,
    root_rows: bool,
    wave_layouts: list[dict] | None = None,
):
    if (
        not wave_layouts
        or (
            model.static.wave_layout is not None
            and len(wave_layouts) == 1
            and wave_layouts[0] is model.static.wave_layout
        )
    ):
        return _run_pi(model, E_out, params, need_pibar=need_pibar, root_rows=root_rows)

    total = torch.zeros((), dtype=model.static.dtype, device=model.static.device)
    last_out = None
    for wave_layout in wave_layouts:
        nll, out = _run_pi(
            model,
            E_out,
            params,
            need_pibar=need_pibar,
            root_rows=root_rows,
            wave_layout=wave_layout,
        )
        total = total + nll
        if last_out is not None:
            del last_out
        last_out = out
    return total, last_out


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
    return float((a - b).abs().max().detach().cpu())


def _print_shape(model: GeneReconModel) -> None:
    if model.static.wave_layout is None:
        print("shape", "S", model.n_species, "G", model.n_families, "C", "chunked", "waves", "chunked")
        return
    metas = model.static.wave_layout["wave_metas"]
    split_rows = sum(int(m["sl"].numel()) if m.get("has_splits", False) else 0 for m in metas)
    C = sum(int(m["W"]) for m in metas)
    max_w = max((int(m["W"]) for m in metas), default=0)
    max_splits = max((int(m["sl"].numel()) for m in metas if m.get("has_splits", False)), default=0)
    print(
        "shape",
        "S", model.n_species,
        "G", model.n_families,
        "C", C,
        "waves", len(metas),
        "maxW", max_w,
        "split_rows", split_rows,
        "max_splits", max_splits,
    )


def _compare_forward(
    args: argparse.Namespace,
    model: GeneReconModel,
    E_out: dict,
    params: tuple,
    wave_layouts: list[dict] | None,
) -> None:
    chunked = bool(wave_layouts and len(wave_layouts) > 1)
    keep_full = args.fams <= args.full_diff_max_fams and not chunked
    _set_variant("reference")
    ref_nll, ref_out = _run_forward_once(
        model,
        E_out,
        params,
        need_pibar=args.need_pibar,
        root_rows=args.root_rows,
        wave_layouts=wave_layouts,
    )
    torch.cuda.synchronize()
    ref_nll_value = float(ref_nll.detach().cpu())
    if not keep_full:
        del ref_out
        torch.cuda.empty_cache()

    _set_variant("optimized")
    opt_nll, opt_out = _run_forward_once(
        model,
        E_out,
        params,
        need_pibar=args.need_pibar,
        root_rows=args.root_rows,
        wave_layouts=wave_layouts,
    )
    torch.cuda.synchronize()
    print(
        "compare_forward",
        "reference_nll", ref_nll_value,
        "optimized_nll", float(opt_nll.detach().cpu()),
        "nll_abs_diff", float((opt_nll - ref_nll).abs().detach().cpu()),
    )
    if keep_full:
        if args.root_rows:
            pi_diff = _tensor_max_abs_diff(ref_out["Pi_root_rows"], opt_out["Pi_root_rows"])
            pi_label = "Pi_root_rows_max_abs"
        else:
            pi_diff = _tensor_max_abs_diff(ref_out["Pi_wave_ordered"], opt_out["Pi_wave_ordered"])
            pi_label = "Pi_max_abs"
        if ref_out["Pibar_wave_ordered"] is not None and opt_out["Pibar_wave_ordered"] is not None:
            pibar_diff = _tensor_max_abs_diff(ref_out["Pibar_wave_ordered"], opt_out["Pibar_wave_ordered"])
        else:
            pibar_diff = "not_returned"
        print("compare_forward_tensors", pi_label, pi_diff, "Pibar_max_abs", pibar_diff)
        del ref_out
    del opt_out, ref_nll, opt_nll
    gc.collect()
    torch.cuda.empty_cache()


def _loss_and_grad(model: GeneReconModel, variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    _set_variant(variant)
    model.static.warm_E = None
    model.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    torch.cuda.synchronize()
    if model.theta.grad is None:
        raise RuntimeError("theta.grad was not populated")
    return loss.detach().clone(), model.theta.grad.detach().clone()


def _compare_backward(args: argparse.Namespace, model: GeneReconModel) -> None:
    ref_loss, ref_grad = _loss_and_grad(model, "reference")
    opt_loss, opt_grad = _loss_and_grad(model, "optimized")
    grad_diff = _tensor_max_abs_diff(ref_grad, opt_grad)
    print(
        "compare_backward",
        "reference_loss", float(ref_loss.cpu()),
        "optimized_loss", float(opt_loss.cpu()),
        "loss_abs_diff", float((opt_loss - ref_loss).abs().cpu()),
        "grad_max_abs", grad_diff,
    )


def _compare_semantics(args: argparse.Namespace, genes: list[str]) -> None:
    sem_args = argparse.Namespace(**vars(args))
    sem_args.theta_profile = "constant"
    _set_variant(args.variant)
    global_model = _make_model(sem_args, genes, mode="global")
    species_model = _make_model(sem_args, genes, mode="specieswise")
    global_loss, global_grad = _loss_and_grad(global_model, args.variant)
    species_loss, species_grad = _loss_and_grad(species_model, args.variant)
    print(
        "compare_semantics",
        "variant", args.variant,
        "global_loss", float(global_loss.cpu()),
        "specieswise_loss", float(species_loss.cpu()),
        "loss_abs_diff", float((species_loss - global_loss).abs().cpu()),
        "summed_grad_max_abs",
        _tensor_max_abs_diff(global_grad, species_grad.sum(dim=0)),
    )


def _run_forward_timing(
    args: argparse.Namespace,
    model: GeneReconModel,
    E_out: dict,
    params: tuple,
    wave_layouts: list[dict] | None,
) -> None:
    _set_variant(args.variant)
    for _ in range(args.warmups):
        _, out = _run_forward_once(
            model,
            E_out,
            params,
            need_pibar=args.need_pibar,
            root_rows=args.root_rows,
            wave_layouts=wave_layouts,
        )
        del out
    torch.cuda.synchronize()

    times = []
    nll_value = None
    for _ in range(args.reps):
        _set_variant(args.variant)
        ms, (nll, out) = _time_cuda_ms(
            lambda: _run_forward_once(
                model,
                E_out,
                params,
                need_pibar=args.need_pibar,
                root_rows=args.root_rows,
                wave_layouts=wave_layouts,
            )
        )
        times.append(ms)
        nll_value = float(nll.detach().cpu())
        del out, nll
    print(
        "forward_ms",
        "variant", args.variant,
        "mode", "specieswise",
        "dtype", args.dtype,
        "fams", args.fams,
        "need_pibar", int(args.need_pibar),
        "root_rows", int(args.root_rows),
        "family_chunk_size", args.family_chunk_size,
        "chunks", 0 if not wave_layouts else len(wave_layouts),
        "reps", len(times),
        "median", f"{statistics.median(times):.3f}",
        "mean", f"{statistics.mean(times):.3f}",
        "min", f"{min(times):.3f}",
        "max", f"{max(times):.3f}",
        "nll", nll_value,
        "peak_gib", f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f}",
    )


def _run_backward_timing(args: argparse.Namespace, model: GeneReconModel) -> None:
    _set_variant(args.variant)
    for _ in range(args.warmups):
        model.zero_grad(set_to_none=True)
        loss = model()
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()

    times = []
    loss_value = None
    for _ in range(args.reps):
        _set_variant(args.variant)
        model.zero_grad(set_to_none=True)
        loss = model()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        loss_value = float(loss.detach().cpu())
    print(
        "backward_ms",
        "variant", args.variant,
        "mode", "specieswise",
        "dtype", args.dtype,
        "fams", args.fams,
        "reps", len(times),
        "median", f"{statistics.median(times):.3f}",
        "mean", f"{statistics.mean(times):.3f}",
        "min", f"{min(times):.3f}",
        "max", f"{max(times):.3f}",
        "loss", loss_value,
        "peak_gib", f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f}",
    )


def main() -> None:
    args = _parse_args()
    if args.print_commands:
        _print_commands(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    root = Path(args.dataset)
    genes = _selected_genes(root, args.start, args.fams)
    torch.cuda.empty_cache()
    gc.collect()

    _set_variant(args.variant)
    chunked_forward_only = (
        args.family_chunk_size > 0
        and args.target == "forward"
        and not args.compare_backward
        and not args.compare_semantics
    )
    if chunked_forward_only:
        model = _make_forward_state(args, genes)
    else:
        model = _make_model(args, genes, mode="specieswise")
    print("dataset", root, "family_range", f"{args.start}:{args.start + args.fams}")
    print("variant", args.variant, "target", args.target, "theta_profile", args.theta_profile)
    _print_shape(model)
    wave_layouts = None
    if args.target in ("forward", "both") or args.compare_forward:
        wave_layouts = _build_forward_chunks(model, args)
        if len(wave_layouts) > 1:
            print("forward_chunks", len(wave_layouts), "family_chunk_size", args.family_chunk_size)

    if args.compare_semantics:
        _compare_semantics(args, genes)

    E_out, params = _prepare_forward(model)
    if args.stats_only:
        return
    if args.compare_forward:
        _compare_forward(args, model, E_out, params, wave_layouts)
    if args.compare_backward:
        _compare_backward(args, model)

    if args.target in ("forward", "both"):
        _run_forward_timing(args, model, E_out, params, wave_layouts)
    if args.target in ("backward", "both"):
        _run_backward_timing(args, model)


if __name__ == "__main__":
    main()
