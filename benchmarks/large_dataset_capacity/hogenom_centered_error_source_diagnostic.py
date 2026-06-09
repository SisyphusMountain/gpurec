#!/usr/bin/env python3
"""Pinpoint centered fp32 gradient error by swapping forward states/backward dtype."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.large_dataset_capacity.hogenom_gmres_neumann_family_experiment import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FAMILIES_FILE,
    DEFAULT_SPECIES_TREE,
    family_tree_path,
)
from gpurec import GeneReconModel, SolverOptions
from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.inference.solver import nll_from_root_rows, solve_resident_e_pi
from gpurec.core.parameters.extract_parameters import (
    extract_parameters_uniform_high_low,
    split_fp32_high_low,
)


DEFAULT_OUTPUT_JSON = REPO_ROOT / (
    "benchmarks/large_dataset_capacity/output/"
    "hogenom_centered_production_forward_20260607/error_source_diagnostic.json"
)


def _csv_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species-tree", type=Path, default=DEFAULT_SPECIES_TREE)
    parser.add_argument("--families-file", type=Path, default=DEFAULT_FAMILIES_FILE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--family-indices", type=_csv_ints, default=_csv_ints("799,8173"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--e-iters", type=int, default=256)
    parser.add_argument("--pi-iters", type=int, default=256)
    parser.add_argument("--neumann-terms", type=int, default=8)
    parser.add_argument("--clade-budget", type=int, default=250_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--include-floor-bounds",
        action="store_true",
        help="Run extra fp32 representation lower-bound checks.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


@contextmanager
def temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def solver_options(args: argparse.Namespace) -> SolverOptions:
    return SolverOptions(
        e_max_iter=int(args.e_iters),
        e_tol=1e-10,
        pi_iters=int(args.pi_iters),
        neumann_terms=int(args.neumann_terms),
        bicgstab_max_iter=1000,
        bicgstab_tol=1e-10,
        adjoint_pruning_threshold=0.0,
        use_adjoint_pruning=False,
        pibar_side_threshold=0.0,
    )


def build_model(args: argparse.Namespace, tree_path: Path) -> GeneReconModel:
    return GeneReconModel(
        args.species_tree,
        [tree_path],
        mode="global",
        device=args.device,
        family_chunk_size=1,
        clade_budget=args.clade_budget,
        max_wave_size=args.max_wave_size,
        solver_options=solver_options(args),
    )


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def solve_state(
    args: argparse.Namespace,
    model: GeneReconModel,
    theta_row: torch.Tensor,
    *,
    dtype: torch.dtype,
    centered: bool,
    compensation_dtype: str | None,
) -> dict[str, Any]:
    env = {
        "GPUREC_CENTERED_PI_FORWARD": "1" if centered else None,
        "GPUREC_CENTERED_PI_COMPENSATION_DTYPE": compensation_dtype if centered else None,
    }
    with temporary_env(env):
        static = model.batch_statics[0]
        static.warm_E = None
        theta = theta_row.to(device=args.device, dtype=dtype).contiguous()
        receiver_weights = torch.zeros(
            (int(model.species_helpers["S"]),),
            dtype=dtype,
            device=theta.device,
        )
        start = time.perf_counter()
        (
            E,
            E_s1,
            E_s2,
            Ebar,
            root_rows,
            pi_wave,
            pibar_wave,
            pibar_row_max,
            log_pS,
            log_pD,
            log_pL,
            max_transfer,
            receiver_log_probs,
            use_receiver_weights,
        ) = solve_resident_e_pi(static, theta, receiver_weights)
        cuda_sync(theta.device)
        elapsed_s = time.perf_counter() - start
        centered_state = static.centered_pi_forward_state
        loss = nll_from_root_rows(root_rows, E).detach()
        return {
            "theta": theta,
            "receiver_weights": receiver_weights,
            "E": E.detach(),
            "E_s1": E_s1.detach(),
            "E_s2": E_s2.detach(),
            "Ebar": Ebar.detach(),
            "root_rows": root_rows.detach(),
            "pi": pi_wave.detach(),
            "pibar": pibar_wave.detach(),
            "pibar_row_max": pibar_row_max.detach(),
            "pi_offset": None if centered_state is None else centered_state.pi_offset.detach(),
            "pibar_offset": None if centered_state is None else centered_state.pibar_offset.detach(),
            "log_pS": log_pS.detach(),
            "log_pD": log_pD.detach(),
            "log_pL": log_pL.detach(),
            "max_transfer": max_transfer.detach(),
            "receiver_log_probs": receiver_log_probs.detach(),
            "use_receiver_weights": bool(use_receiver_weights),
            "loss": loss,
            "elapsed_s": elapsed_s,
        }


def integer_center_from_abs(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_max = values.max(dim=1).values
    offset = torch.floor(torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max)))
    local = values - offset.reshape(-1, 1)
    return local.float().contiguous(), offset.float().contiguous()


def centered_abs(state: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state["pi_offset"] is None or state["pibar_offset"] is None:
        raise ValueError("state is not centered")
    pi_abs = state["pi"].double() + state["pi_offset"].double().reshape(-1, 1)
    pibar_abs = state["pibar"].double() + state["pibar_offset"].double().reshape(-1, 1)
    pibar_row_max_abs = state["pibar_row_max"].double() + state["pi_offset"].double()
    return pi_abs.contiguous(), pibar_abs.contiguous(), pibar_row_max_abs.contiguous()


def tensor_summary(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    top_k: int,
    rows_are_2d: bool = True,
) -> dict[str, Any]:
    actual64 = actual.detach().double()
    ref64 = reference.detach().double()
    delta = actual64 - ref64
    ref_norm = torch.linalg.vector_norm(ref64).clamp_min(1e-30)
    rel_l2 = torch.linalg.vector_norm(delta) / ref_norm
    abs_l2 = torch.linalg.vector_norm(delta)
    abs_inf = delta.abs().max()
    out: dict[str, Any] = {
        "rel_l2": float(rel_l2.detach().cpu()),
        "abs_l2": float(abs_l2.detach().cpu()),
        "abs_inf": float(abs_inf.detach().cpu()),
        "ref_l2": float(ref_norm.detach().cpu()),
    }
    if rows_are_2d and actual64.ndim == 2:
        row_l2 = torch.linalg.vector_norm(delta, dim=1)
        ref_row_l2 = torch.linalg.vector_norm(ref64, dim=1).clamp_min(1e-30)
        row_rel = row_l2 / ref_row_l2
        flat_abs = delta.abs().reshape(-1)
        values, flat_idx = torch.topk(flat_abs, k=min(int(top_k), int(flat_abs.numel())))
        cols = actual64.shape[1]
        top_entries = []
        for value, idx in zip(values.detach().cpu().tolist(), flat_idx.detach().cpu().tolist()):
            row = int(idx // cols)
            col = int(idx % cols)
            top_entries.append(
                {
                    "row": row,
                    "species": col,
                    "abs_delta": float(value),
                    "actual": float(actual64[row, col].detach().cpu()),
                    "reference": float(ref64[row, col].detach().cpu()),
                    "row_rel_l2": float(row_rel[row].detach().cpu()),
                }
            )
        worst_rows = torch.topk(row_rel, k=min(int(top_k), int(row_rel.numel()))).indices
        out["top_entries_by_abs_delta"] = top_entries
        out["top_rows_by_rel_l2"] = [
            {
                "row": int(row.detach().cpu()),
                "rel_l2": float(row_rel[row].detach().cpu()),
                "abs_l2": float(row_l2[row].detach().cpu()),
                "ref_l2": float(ref_row_l2[row].detach().cpu()),
            }
            for row in worst_rows
        ]
    return out


def root_seed(root_rows: torch.Tensor) -> torch.Tensor:
    rows = root_rows.double()
    row_max = rows.max(dim=-1, keepdim=True).values
    weights = torch.exp2(rows - row_max)
    probs = weights / weights.sum(dim=-1, keepdim=True)
    return -probs


def run_vjp(
    args: argparse.Namespace,
    model: GeneReconModel,
    state: dict[str, Any],
    *,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    centered: bool,
    pi: torch.Tensor,
    pibar: torch.Tensor,
    pibar_row_max: torch.Tensor,
    pi_offset: torch.Tensor | None = None,
    pibar_offset: torch.Tensor | None = None,
    dtype: torch.dtype,
) -> dict[str, Any]:
    static = model.batch_statics[0]
    target_device = torch.device(args.device)

    def cast(x: torch.Tensor) -> torch.Tensor:
        return x.to(device=target_device, dtype=dtype).contiguous()

    diagnostics: dict[str, Any] = {
        "store_wave_tensors": True,
        "store_pibar_tree_tensors": True,
    }
    start = time.perf_counter()
    grad_theta, _grad_receiver = implicit_grad_loglik_vjp_wave(
        static.wave_layout,
        static.species_helpers,
        Pi_star_wave=cast(pi),
        Pibar_star_wave=cast(pibar),
        Pi_star_wave_offset=None if not centered else cast(pi_offset),  # type: ignore[arg-type]
        Pibar_star_wave_offset=None if not centered else cast(pibar_offset),  # type: ignore[arg-type]
        E_star=cast(state["E"]),
        Ebar=cast(state["Ebar"]),
        E_s1=cast(state["E_s1"]),
        E_s2=cast(state["E_s2"]),
        log_pS=cast(state["log_pS"]),
        log_pD=cast(state["log_pD"]),
        log_pL=cast(state["log_pL"]),
        max_transfer_mat=cast(state["max_transfer"]),
        receiver_log_probs=cast(state["receiver_log_probs"]),
        use_receiver_weights=bool(state["use_receiver_weights"]),
        theta=theta.to(device=target_device, dtype=dtype).contiguous(),
        receiver_weights=receiver_weights.to(device=target_device, dtype=dtype).contiguous(),
        uniform_pibar_row_max=cast(pibar_row_max),
        family_idx=static.rate_family_idx,
        need_receiver_grad=False,
        specieswise=static.specieswise,
        genewise=static.genewise,
        neumann_terms=int(args.neumann_terms),
        self_loop_solver="neumann",
        bicgstab_max_iter=1000,
        bicgstab_tol=1e-10,
        adjoint_pruning_threshold=0.0,
        use_adjoint_pruning=False,
        pibar_side_threshold=0.0,
        diagnostics=diagnostics,
    )
    cuda_sync(target_device)
    return {
        "gradient": grad_theta.detach().cpu().double().reshape(-1),
        "elapsed_s": time.perf_counter() - start,
        "diagnostics": diagnostics,
    }


def gradient_row(
    name: str,
    run: dict[str, Any],
    reference_gradient: torch.Tensor,
) -> dict[str, Any]:
    grad = run["gradient"].double()
    ref = reference_gradient.double()
    delta = grad - ref
    return {
        "variant": name,
        "gradient": [float(x) for x in grad.tolist()],
        "rel_l2_vs_reference": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref).clamp_min(1e-30)
        ),
        "abs_l2_vs_reference": float(torch.linalg.vector_norm(delta)),
        "elapsed_s": float(run["elapsed_s"]),
    }


def state_variant(base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    out = dict(base)
    out.update(updates)
    return out


def _get_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def diagnostic_tensor_comparisons(
    actual: dict[str, Any],
    reference: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    paths = [
        ("root", "root_seed"),
        ("self_loop_sources", "accumulated_rhs"),
        ("self_loop_sources", "grad_E"),
        ("self_loop_sources", "grad_Ebar"),
        ("self_loop_sources", "grad_E_s1"),
        ("self_loop_sources", "grad_E_s2"),
        ("self_loop_sources", "grad_log_pD"),
        ("self_loop_sources", "grad_log_pS"),
        ("self_loop_sources", "grad_max_transfer_mat"),
        ("e_adjoint", "q_E"),
        ("e_adjoint", "wE"),
        ("parameter_vjp", "grad_log_pS_net"),
        ("parameter_vjp", "grad_log_pD_net"),
        ("parameter_vjp", "grad_log_pL_net"),
        ("parameter_vjp", "grad_max_transfer_net"),
        ("parameter_vjp", "log_param_adjoints"),
        ("parameter_vjp", "explicit_theta_grad"),
    ]
    rows = []
    for path in paths:
        a = _get_nested(actual, path)
        r = _get_nested(reference, path)
        if not torch.is_tensor(a) or not torch.is_tensor(r):
            continue
        summary = tensor_summary(a, r, top_k=0, rows_are_2d=False)
        rows.append({"path": ".".join(path), **summary})
    return sorted(rows, key=lambda item: item["rel_l2"], reverse=True)[: int(top_k)]


def wave_tensor_comparisons(
    actual: dict[str, Any],
    reference: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    actual_waves = actual.get("wave_tensors") or []
    ref_waves = reference.get("wave_tensors") or []
    names = ("rhs", "v_k", "aw0", "aw1", "aw2", "aw345", "aw3", "aw4")
    rows = []
    for a_wave, r_wave in zip(actual_waves, ref_waves):
        for name in names:
            a = a_wave.get(name)
            r = r_wave.get(name)
            if not torch.is_tensor(a) or not torch.is_tensor(r):
                continue
            summary = tensor_summary(a, r, top_k=0, rows_are_2d=False)
            rows.append(
                {
                    "reverse_index": int(a_wave.get("reverse_index", -1)),
                    "ws": int(a_wave.get("ws", -1)),
                    "W": int(a_wave.get("W", -1)),
                    "has_splits": bool(a_wave.get("has_splits", False)),
                    "tensor": name,
                    **summary,
                }
            )
    return sorted(rows, key=lambda item: item["rel_l2"], reverse=True)[: int(top_k)]


def pibar_tree_comparisons(
    actual: dict[str, Any],
    reference: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    actual_rows = actual.get("pibar_tree_tensors") or []
    ref_rows = reference.get("pibar_tree_tensors") or []
    names = ("pibar_ud_initial", "pibar_A")
    rows = []
    for a_row, r_row in zip(actual_rows, ref_rows):
        for name in names:
            a = a_row.get(name)
            r = r_row.get(name)
            if not torch.is_tensor(a) or not torch.is_tensor(r):
                continue
            summary = tensor_summary(a, r, top_k=0, rows_are_2d=False)
            rows.append(
                {
                    "reverse_index": int(a_row.get("reverse_index", -1)),
                    "ws": int(a_row.get("ws", -1)),
                    "W": int(a_row.get("W", -1)),
                    "split_count": int(a_row.get("split_count", -1)),
                    "tensor": name,
                    **summary,
                }
            )
    return sorted(rows, key=lambda item: item["rel_l2"], reverse=True)[: int(top_k)]


def parameter_projection_details(
    actual: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any] | None:
    actual_adj = _get_nested(actual, ("parameter_vjp", "log_param_adjoints"))
    ref_adj = _get_nested(reference, ("parameter_vjp", "log_param_adjoints"))
    actual_grad = _get_nested(actual, ("parameter_vjp", "explicit_theta_grad"))
    ref_grad = _get_nested(reference, ("parameter_vjp", "explicit_theta_grad"))
    if not (
        torch.is_tensor(actual_adj)
        and torch.is_tensor(ref_adj)
        and torch.is_tensor(actual_grad)
        and torch.is_tensor(ref_grad)
    ):
        return None
    actual_adj = actual_adj.reshape(-1).double()
    ref_adj = ref_adj.reshape(-1).double()
    actual_grad = actual_grad.reshape(-1).double()
    ref_grad = ref_grad.reshape(-1).double()
    if int(actual_adj.numel()) != 4 or int(ref_adj.numel()) != 4:
        return None
    actual_total = actual_adj.sum()
    ref_total = ref_adj.sum()
    # Infer probabilities from adjoints and gradient: grad_i = adj_i - p_i * total.
    actual_probs = (actual_adj[1:] - actual_grad) / actual_total
    ref_probs = (ref_adj[1:] - ref_grad) / ref_total
    rows = []
    names = ("D", "L", "T")
    for i, name in enumerate(names, start=1):
        actual_subtracted = actual_probs[i - 1] * actual_total
        ref_subtracted = ref_probs[i - 1] * ref_total
        rows.append(
            {
                "theta_component": name,
                "actual_adj_i": float(actual_adj[i]),
                "reference_adj_i": float(ref_adj[i]),
                "adj_i_delta": float(actual_adj[i] - ref_adj[i]),
                "actual_prob_times_total": float(actual_subtracted),
                "reference_prob_times_total": float(ref_subtracted),
                "prob_times_total_delta": float(actual_subtracted - ref_subtracted),
                "actual_theta_grad": float(actual_grad[i - 1]),
                "reference_theta_grad": float(ref_grad[i - 1]),
                "theta_grad_delta": float(actual_grad[i - 1] - ref_grad[i - 1]),
                "cancellation_factor_reference": float(
                    (ref_adj[i].abs() + ref_subtracted.abs())
                    / ref_grad[i - 1].abs().clamp_min(1e-30)
                ),
            }
        )
    return {
        "actual_log_param_adjoints": [float(x) for x in actual_adj.tolist()],
        "reference_log_param_adjoints": [float(x) for x in ref_adj.tolist()],
        "log_param_adjoint_delta": [float(x) for x in (actual_adj - ref_adj).tolist()],
        "actual_total_adjoint": float(actual_total),
        "reference_total_adjoint": float(ref_total),
        "total_adjoint_delta": float(actual_total - ref_total),
        "actual_inferred_probs_DTL": [float(x) for x in actual_probs.tolist()],
        "reference_inferred_probs_DTL": [float(x) for x in ref_probs.tolist()],
        "rows": rows,
    }


def _scalar_from_tensor(value: Any) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    return value.detach().double().reshape(-1).sum()


def log_param_source_breakdown(
    actual: dict[str, Any],
    reference: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any] | None:
    """Split self-loop log-param gradients into uniform-wave and DTS residual sources."""
    actual_waves = actual.get("wave_tensors") or []
    ref_waves = reference.get("wave_tensors") or []
    actual_pD = _scalar_from_tensor(_get_nested(actual, ("self_loop_sources", "grad_log_pD")))
    ref_pD = _scalar_from_tensor(_get_nested(reference, ("self_loop_sources", "grad_log_pD")))
    actual_pS = _scalar_from_tensor(_get_nested(actual, ("self_loop_sources", "grad_log_pS")))
    ref_pS = _scalar_from_tensor(_get_nested(reference, ("self_loop_sources", "grad_log_pS")))
    if actual_pD is None or ref_pD is None or actual_pS is None or ref_pS is None:
        return None

    actual_uniform_pD = torch.zeros((), dtype=torch.float64)
    ref_uniform_pD = torch.zeros((), dtype=torch.float64)
    actual_uniform_pS = torch.zeros((), dtype=torch.float64)
    ref_uniform_pS = torch.zeros((), dtype=torch.float64)
    wave_rows = []
    for actual_wave, ref_wave in zip(actual_waves, ref_waves):
        actual_aw0 = _scalar_from_tensor(actual_wave.get("aw0"))
        ref_aw0 = _scalar_from_tensor(ref_wave.get("aw0"))
        actual_aw345 = _scalar_from_tensor(actual_wave.get("aw345"))
        ref_aw345 = _scalar_from_tensor(ref_wave.get("aw345"))
        if actual_aw0 is None or ref_aw0 is None or actual_aw345 is None or ref_aw345 is None:
            continue
        actual_uniform_pD = actual_uniform_pD + actual_aw0
        ref_uniform_pD = ref_uniform_pD + ref_aw0
        actual_uniform_pS = actual_uniform_pS + actual_aw345
        ref_uniform_pS = ref_uniform_pS + ref_aw345
        wave_rows.append(
            {
                "reverse_index": int(actual_wave.get("reverse_index", -1)),
                "ws": int(actual_wave.get("ws", -1)),
                "W": int(actual_wave.get("W", -1)),
                "has_splits": bool(actual_wave.get("has_splits", False)),
                "uniform_pD_delta": float(actual_aw0 - ref_aw0),
                "uniform_pS_delta": float(actual_aw345 - ref_aw345),
                "uniform_pD_reference": float(ref_aw0),
                "uniform_pS_reference": float(ref_aw345),
            }
        )

    actual_dts_pD = actual_pD - actual_uniform_pD
    ref_dts_pD = ref_pD - ref_uniform_pD
    actual_dts_pS = actual_pS - actual_uniform_pS
    ref_dts_pS = ref_pS - ref_uniform_pS

    def source_row(name: str, actual_value: torch.Tensor, ref_value: torch.Tensor) -> dict[str, float | str]:
        delta = actual_value - ref_value
        return {
            "source": name,
            "actual": float(actual_value),
            "reference": float(ref_value),
            "delta": float(delta),
            "relative_delta": float(delta.abs() / ref_value.abs().clamp_min(1e-30)),
        }

    wave_rows_by_abs_delta = sorted(
        wave_rows,
        key=lambda item: max(abs(float(item["uniform_pD_delta"])), abs(float(item["uniform_pS_delta"]))),
        reverse=True,
    )[: int(top_k)]
    return {
        "rows": [
            source_row("uniform_wave_grad_log_pD", actual_uniform_pD, ref_uniform_pD),
            source_row("dts_residual_grad_log_pD", actual_dts_pD, ref_dts_pD),
            source_row("total_self_loop_grad_log_pD", actual_pD, ref_pD),
            source_row("uniform_wave_grad_log_pS", actual_uniform_pS, ref_uniform_pS),
            source_row("dts_residual_grad_log_pS", actual_dts_pS, ref_dts_pS),
            source_row("total_self_loop_grad_log_pS", actual_pS, ref_pS),
        ],
        "top_uniform_wave_deltas": wave_rows_by_abs_delta,
    }


def wave_error_summary(
    actual_abs: torch.Tensor,
    reference_abs: torch.Tensor,
    wave_metas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    delta = actual_abs.detach().double() - reference_abs.detach().double()
    ref = reference_abs.detach().double()
    for index, meta in enumerate(wave_metas):
        ws = int(meta["start"])
        W = int(meta["W"])
        d = delta[ws : ws + W]
        r = ref[ws : ws + W]
        rows.append(
            {
                "wave_index": index,
                "ws": ws,
                "W": W,
                "has_splits": bool("sl" in meta),
                "rel_l2": float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(r).clamp_min(1e-30)),
                "abs_l2": float(torch.linalg.vector_norm(d)),
                "abs_inf": float(d.abs().max()),
            }
        )
    return sorted(rows, key=lambda item: item["rel_l2"], reverse=True)


def fp32_floor_bound_rows(
    args: argparse.Namespace,
    model: GeneReconModel,
    ref64: dict[str, Any],
    prod32: dict[str, Any],
    theta_row: torch.Tensor,
    reference_gradient: torch.Tensor,
) -> list[dict[str, Any]]:
    """Lower-bound checks for single-fp32 and two-fp32 forward state storage."""
    rows: list[dict[str, Any]] = []

    def single_fp32(t: torch.Tensor) -> torch.Tensor:
        return t.detach().float().double().contiguous()

    def two_fp32(t: torch.Tensor) -> torch.Tensor:
        td = t.detach().double()
        hi = td.float()
        finite = torch.isfinite(td)
        lo = torch.where(finite, td - hi.double(), torch.zeros_like(td)).float()
        return (hi.double() + lo.double()).contiguous()

    rounded_theta = theta_row.float().double()
    rounded_ref64 = solve_state(
        args,
        model,
        rounded_theta,
        dtype=torch.float64,
        centered=False,
        compensation_dtype=None,
    )
    rounded_ref_vjp = run_vjp(
        args,
        model,
        rounded_ref64,
        theta=rounded_ref64["theta"],
        receiver_weights=rounded_ref64["receiver_weights"],
        centered=False,
        pi=rounded_ref64["pi"],
        pibar=rounded_ref64["pibar"],
        pibar_row_max=rounded_ref64["pibar_row_max"],
        dtype=torch.float64,
    )
    rows.append(gradient_row("fp64_reference_at_rounded_fp32_theta", rounded_ref_vjp, reference_gradient))

    single_ref_state = state_variant(
        ref64,
        E=single_fp32(ref64["E"]),
        E_s1=single_fp32(ref64["E_s1"]),
        E_s2=single_fp32(ref64["E_s2"]),
        Ebar=single_fp32(ref64["Ebar"]),
        log_pS=single_fp32(ref64["log_pS"]),
        log_pD=single_fp32(ref64["log_pD"]),
        log_pL=single_fp32(ref64["log_pL"]),
        max_transfer=single_fp32(ref64["max_transfer"]),
        receiver_log_probs=single_fp32(ref64["receiver_log_probs"]),
    )
    single_all_vjp = run_vjp(
        args,
        model,
        single_ref_state,
        theta=single_fp32(ref64["theta"]),
        receiver_weights=single_fp32(ref64["receiver_weights"]),
        centered=False,
        pi=single_fp32(ref64["pi"]),
        pibar=single_fp32(ref64["pibar"]),
        pibar_row_max=single_fp32(ref64["pibar_row_max"]),
        dtype=torch.float64,
    )
    rows.append(gradient_row("single_fp32_all_ref_state", single_all_vjp, reference_gradient))

    two_ref_state = state_variant(
        ref64,
        E=two_fp32(ref64["E"]),
        E_s1=two_fp32(ref64["E_s1"]),
        E_s2=two_fp32(ref64["E_s2"]),
        Ebar=two_fp32(ref64["Ebar"]),
        log_pS=two_fp32(ref64["log_pS"]),
        log_pD=two_fp32(ref64["log_pD"]),
        log_pL=two_fp32(ref64["log_pL"]),
        max_transfer=two_fp32(ref64["max_transfer"]),
        receiver_log_probs=two_fp32(ref64["receiver_log_probs"]),
    )
    two_all_vjp = run_vjp(
        args,
        model,
        two_ref_state,
        theta=two_fp32(ref64["theta"]),
        receiver_weights=two_fp32(ref64["receiver_weights"]),
        centered=False,
        pi=two_fp32(ref64["pi"]),
        pibar=two_fp32(ref64["pibar"]),
        pibar_row_max=two_fp32(ref64["pibar_row_max"]),
        dtype=torch.float64,
    )
    rows.append(gradient_row("two_fp32_all_ref_state", two_all_vjp, reference_gradient))

    def add_abs_ref_vjp_row(
        name: str,
        state: dict[str, Any],
        *,
        theta: torch.Tensor | None = None,
        receiver_weights: torch.Tensor | None = None,
        pi: torch.Tensor | None = None,
        pibar: torch.Tensor | None = None,
        pibar_row_max: torch.Tensor | None = None,
    ) -> None:
        run = run_vjp(
            args,
            model,
            state,
            theta=ref64["theta"] if theta is None else theta,
            receiver_weights=ref64["receiver_weights"] if receiver_weights is None else receiver_weights,
            centered=False,
            pi=ref64["pi"] if pi is None else pi,
            pibar=ref64["pibar"] if pibar is None else pibar,
            pibar_row_max=ref64["pibar_row_max"] if pibar_row_max is None else pibar_row_max,
            dtype=torch.float64,
        )
        rows.append(gradient_row(name, run, reference_gradient))

    add_abs_ref_vjp_row(
        "single_fp32_theta_projection_ref_state",
        ref64,
        theta=single_fp32(ref64["theta"]),
        receiver_weights=single_fp32(ref64["receiver_weights"]),
    )
    add_abs_ref_vjp_row(
        "two_fp32_theta_projection_ref_state",
        ref64,
        theta=two_fp32(ref64["theta"]),
        receiver_weights=two_fp32(ref64["receiver_weights"]),
    )
    add_abs_ref_vjp_row(
        "single_fp32_E_bundle_ref_state",
        state_variant(
            ref64,
            E=single_fp32(ref64["E"]),
            E_s1=single_fp32(ref64["E_s1"]),
            E_s2=single_fp32(ref64["E_s2"]),
            Ebar=single_fp32(ref64["Ebar"]),
        ),
    )
    add_abs_ref_vjp_row(
        "two_fp32_E_bundle_ref_state",
        state_variant(
            ref64,
            E=two_fp32(ref64["E"]),
            E_s1=two_fp32(ref64["E_s1"]),
            E_s2=two_fp32(ref64["E_s2"]),
            Ebar=two_fp32(ref64["Ebar"]),
        ),
    )
    add_abs_ref_vjp_row(
        "single_fp32_param_bundle_ref_state",
        state_variant(
            ref64,
            log_pS=single_fp32(ref64["log_pS"]),
            log_pD=single_fp32(ref64["log_pD"]),
            log_pL=single_fp32(ref64["log_pL"]),
            max_transfer=single_fp32(ref64["max_transfer"]),
            receiver_log_probs=single_fp32(ref64["receiver_log_probs"]),
        ),
    )
    add_abs_ref_vjp_row(
        "two_fp32_param_bundle_ref_state",
        state_variant(
            ref64,
            log_pS=two_fp32(ref64["log_pS"]),
            log_pD=two_fp32(ref64["log_pD"]),
            log_pL=two_fp32(ref64["log_pL"]),
            max_transfer=two_fp32(ref64["max_transfer"]),
            receiver_log_probs=two_fp32(ref64["receiver_log_probs"]),
        ),
    )
    static = model.batch_statics[0]
    theta_hi, theta_lo = split_fp32_high_low(theta_row.to(device=args.device))
    unnorm_hi, unnorm_lo = split_fp32_high_low(
        static.species_helpers["unnorm_row_max"].to(device=args.device, dtype=torch.float64)
    )
    param_hi, param_lo = extract_parameters_uniform_high_low(
        theta_hi,
        theta_lo,
        unnorm_hi,
        unnorm_lo,
        specieswise=static.specieswise,
        genewise=static.genewise,
    )
    add_abs_ref_vjp_row(
        "theta_high_low_helper_param_bundle_ref_state",
        state_variant(
            ref64,
            log_pS=(param_hi[0].double() + param_lo[0].double()).contiguous(),
            log_pD=(param_hi[1].double() + param_lo[1].double()).contiguous(),
            log_pL=(param_hi[2].double() + param_lo[2].double()).contiguous(),
            max_transfer=(param_hi[3].double() + param_lo[3].double()).contiguous(),
        ),
    )
    add_abs_ref_vjp_row(
        "single_fp32_abs_pi_pibar_ref_state",
        ref64,
        pi=single_fp32(ref64["pi"]),
        pibar=single_fp32(ref64["pibar"]),
        pibar_row_max=single_fp32(ref64["pibar_row_max"]),
    )
    add_abs_ref_vjp_row(
        "two_fp32_abs_pi_pibar_ref_state",
        ref64,
        pi=two_fp32(ref64["pi"]),
        pibar=two_fp32(ref64["pibar"]),
        pibar_row_max=two_fp32(ref64["pibar_row_max"]),
    )

    pi_ref_c, pi_ref_offset = integer_center_from_abs(ref64["pi"].double())
    pibar_ref_c, pibar_ref_offset = integer_center_from_abs(ref64["pibar"].double())
    pi_ref_single_abs = pi_ref_c.double() + pi_ref_offset.double().reshape(-1, 1)
    pibar_ref_single_abs = pibar_ref_c.double() + pibar_ref_offset.double().reshape(-1, 1)
    pibar_row_max_ref_single_abs = (
        ref64["pibar_row_max"].double() - pi_ref_offset.double()
    ).float().double() + pi_ref_offset.double()
    single_quant_vjp = run_vjp(
        args,
        model,
        ref64,
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=pi_ref_single_abs,
        pibar=pibar_ref_single_abs,
        pibar_row_max=pibar_row_max_ref_single_abs,
        dtype=torch.float64,
    )
    rows.append(gradient_row("single_fp32_centered_quantized_ref_state", single_quant_vjp, reference_gradient))

    pi_ref_low = (ref64["pi"].double() - pi_ref_single_abs).float()
    pibar_ref_low = (ref64["pibar"].double() - pibar_ref_single_abs).float()
    two_quant_vjp = run_vjp(
        args,
        model,
        ref64,
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=pi_ref_single_abs + pi_ref_low.double(),
        pibar=pibar_ref_single_abs + pibar_ref_low.double(),
        pibar_row_max=ref64["pibar_row_max"].double(),
        dtype=torch.float64,
    )
    rows.append(gradient_row("two_fp32_centered_quantized_ref_state", two_quant_vjp, reference_gradient))

    prod_pi_abs, prod_pibar_abs, _prod_pibar_row_max_abs = centered_abs(prod32)
    pi_prod_low = (ref64["pi"].double() - prod_pi_abs).float()
    pibar_prod_low = (ref64["pibar"].double() - prod_pibar_abs).float()
    prod_pi_pibar_corrected_vjp = run_vjp(
        args,
        model,
        prod32,
        theta=prod32["theta"],
        receiver_weights=prod32["receiver_weights"],
        centered=False,
        pi=prod_pi_abs + pi_prod_low.double(),
        pibar=prod_pibar_abs + pibar_prod_low.double(),
        pibar_row_max=ref64["pibar_row_max"].double(),
        dtype=torch.float64,
    )
    rows.append(
        gradient_row(
            "prod_state_with_ref_pi_pibar_two_fp32_residual",
            prod_pi_pibar_corrected_vjp,
            reference_gradient,
        )
    )
    return rows


def analyze_family(args: argparse.Namespace, family_index: int, family_name: str, theta_row: torch.Tensor) -> dict[str, Any]:
    tree_path = family_tree_path(args.families_file, family_name)
    model = build_model(args, tree_path)

    ref64 = solve_state(args, model, theta_row, dtype=torch.float64, centered=False, compensation_dtype=None)
    prod32 = solve_state(args, model, theta_row, dtype=torch.float32, centered=True, compensation_dtype="fp32")
    abs32 = solve_state(args, model, theta_row, dtype=torch.float32, centered=False, compensation_dtype=None)

    ref_vjp = run_vjp(
        args,
        model,
        ref64,
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=ref64["pi"],
        pibar=ref64["pibar"],
        pibar_row_max=ref64["pibar_row_max"],
        dtype=torch.float64,
    )
    ref_grad = ref_vjp["gradient"]

    prod_vjp = run_vjp(
        args,
        model,
        prod32,
        theta=prod32["theta"],
        receiver_weights=prod32["receiver_weights"],
        centered=True,
        pi=prod32["pi"],
        pibar=prod32["pibar"],
        pi_offset=prod32["pi_offset"],
        pibar_offset=prod32["pibar_offset"],
        pibar_row_max=prod32["pibar_row_max"],
        dtype=torch.float32,
    )

    pi_ref_c, pi_ref_offset = integer_center_from_abs(ref64["pi"].double())
    pibar_ref_c, pibar_ref_offset = integer_center_from_abs(ref64["pibar"].double())
    pibar_row_max_ref_c = (
        ref64["pibar_row_max"].double() - pi_ref_offset.double()
    ).float().contiguous()
    oracle_ref_forward_fp32_backward = run_vjp(
        args,
        model,
        ref64,
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=True,
        pi=pi_ref_c,
        pibar=pibar_ref_c,
        pi_offset=pi_ref_offset,
        pibar_offset=pibar_ref_offset,
        pibar_row_max=pibar_row_max_ref_c,
        dtype=torch.float32,
    )

    prod_pi_abs, prod_pibar_abs, prod_pibar_row_max_abs = centered_abs(prod32)
    prod_state_abs64_backward = run_vjp(
        args,
        model,
        prod32,
        theta=prod32["theta"],
        receiver_weights=prod32["receiver_weights"],
        centered=False,
        pi=prod_pi_abs,
        pibar=prod_pibar_abs,
        pibar_row_max=prod_pibar_row_max_abs,
        dtype=torch.float64,
    )
    prod_state_abs32_backward = run_vjp(
        args,
        model,
        prod32,
        theta=prod32["theta"],
        receiver_weights=prod32["receiver_weights"],
        centered=False,
        pi=prod_pi_abs.float(),
        pibar=prod_pibar_abs.float(),
        pibar_row_max=prod_pibar_row_max_abs.float(),
        dtype=torch.float32,
    )

    prod_e_bundle_on_ref = run_vjp(
        args,
        model,
        state_variant(
            ref64,
            E=prod32["E"],
            E_s1=prod32["E_s1"],
            E_s2=prod32["E_s2"],
            Ebar=prod32["Ebar"],
        ),
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=ref64["pi"],
        pibar=ref64["pibar"],
        pibar_row_max=ref64["pibar_row_max"],
        dtype=torch.float64,
    )
    prod_pi_bundle_on_ref = run_vjp(
        args,
        model,
        ref64,
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=prod_pi_abs,
        pibar=prod_pibar_abs,
        pibar_row_max=prod_pibar_row_max_abs,
        dtype=torch.float64,
    )
    root_ids = model.batch_statics[0].wave_layout["root_clade_ids"].to(
        device=prod_pi_abs.device,
        dtype=torch.long,
    )
    ref_pi_with_prod_roots = ref64["pi"].double().clone()
    ref_pi_with_prod_roots.index_copy_(0, root_ids, prod_pi_abs.index_select(0, root_ids))
    prod_pi_with_ref_roots = prod_pi_abs.double().clone()
    prod_pi_with_ref_roots.index_copy_(0, root_ids, ref64["pi"].double().index_select(0, root_ids))
    prod_root_pi_only_on_ref = run_vjp(
        args,
        model,
        ref64,
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=ref_pi_with_prod_roots,
        pibar=ref64["pibar"],
        pibar_row_max=ref64["pibar_row_max"],
        dtype=torch.float64,
    )
    prod_nonroot_pi_only_on_ref = run_vjp(
        args,
        model,
        ref64,
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=prod_pi_with_ref_roots,
        pibar=ref64["pibar"],
        pibar_row_max=ref64["pibar_row_max"],
        dtype=torch.float64,
    )
    prod_pibar_only_on_ref = run_vjp(
        args,
        model,
        ref64,
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=ref64["pi"],
        pibar=prod_pibar_abs,
        pibar_row_max=ref64["pibar_row_max"],
        dtype=torch.float64,
    )
    prod_pibar_row_max_only_on_ref = run_vjp(
        args,
        model,
        ref64,
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=ref64["pi"],
        pibar=ref64["pibar"],
        pibar_row_max=prod_pibar_row_max_abs,
        dtype=torch.float64,
    )
    prod_param_bundle_on_ref = run_vjp(
        args,
        model,
        state_variant(
            ref64,
            log_pS=prod32["log_pS"],
            log_pD=prod32["log_pD"],
            log_pL=prod32["log_pL"],
            max_transfer=prod32["max_transfer"],
            receiver_log_probs=prod32["receiver_log_probs"],
        ),
        theta=prod32["theta"],
        receiver_weights=prod32["receiver_weights"],
        centered=False,
        pi=ref64["pi"],
        pibar=ref64["pibar"],
        pibar_row_max=ref64["pibar_row_max"],
        dtype=torch.float64,
    )
    prod_e_pi_on_ref = run_vjp(
        args,
        model,
        state_variant(
            ref64,
            E=prod32["E"],
            E_s1=prod32["E_s1"],
            E_s2=prod32["E_s2"],
            Ebar=prod32["Ebar"],
        ),
        theta=ref64["theta"],
        receiver_weights=ref64["receiver_weights"],
        centered=False,
        pi=prod_pi_abs,
        pibar=prod_pibar_abs,
        pibar_row_max=prod_pibar_row_max_abs,
        dtype=torch.float64,
    )

    abs32_vjp = run_vjp(
        args,
        model,
        abs32,
        theta=abs32["theta"],
        receiver_weights=abs32["receiver_weights"],
        centered=False,
        pi=abs32["pi"],
        pibar=abs32["pibar"],
        pibar_row_max=abs32["pibar_row_max"],
        dtype=torch.float32,
    )

    forward_summaries = {
        "current_abs_fp32_vs_ref64": {
            "E": tensor_summary(abs32["E"], ref64["E"], top_k=args.top_k, rows_are_2d=True),
            "Pi_abs": tensor_summary(abs32["pi"], ref64["pi"], top_k=args.top_k, rows_are_2d=True),
            "Pibar_abs": tensor_summary(abs32["pibar"], ref64["pibar"], top_k=args.top_k, rows_are_2d=True),
            "root_rows": tensor_summary(abs32["root_rows"], ref64["root_rows"], top_k=args.top_k, rows_are_2d=True),
            "root_seed": tensor_summary(root_seed(abs32["root_rows"]), root_seed(ref64["root_rows"]), top_k=args.top_k, rows_are_2d=True),
        },
        "centered_integer_fp32_vs_ref64": {
            "E": tensor_summary(prod32["E"], ref64["E"], top_k=args.top_k, rows_are_2d=True),
            "Pi_abs": tensor_summary(prod_pi_abs, ref64["pi"], top_k=args.top_k, rows_are_2d=True),
            "Pibar_abs": tensor_summary(prod_pibar_abs, ref64["pibar"], top_k=args.top_k, rows_are_2d=True),
            "root_rows": tensor_summary(prod32["root_rows"], ref64["root_rows"], top_k=args.top_k, rows_are_2d=True),
            "root_seed": tensor_summary(root_seed(prod32["root_rows"]), root_seed(ref64["root_rows"]), top_k=args.top_k, rows_are_2d=True),
        },
    }

    wave_summaries = {
        "Pi_abs_centered_integer_fp32_vs_ref64": wave_error_summary(
            prod_pi_abs,
            ref64["pi"],
            model.batch_statics[0].wave_layout["wave_metas"],
        )[: int(args.top_k)],
        "Pibar_abs_centered_integer_fp32_vs_ref64": wave_error_summary(
            prod_pibar_abs,
            ref64["pibar"],
            model.batch_statics[0].wave_layout["wave_metas"],
        )[: int(args.top_k)],
    }

    rows = [
        gradient_row("reference_fp64_abs_forward_fp64_backward", ref_vjp, ref_grad),
        gradient_row("current_abs_fp32_forward_backward", abs32_vjp, ref_grad),
        gradient_row("centered_integer_fp32_forward_backward", prod_vjp, ref_grad),
        gradient_row("oracle_ref_forward_centered_int32_fp32_backward", oracle_ref_forward_fp32_backward, ref_grad),
        gradient_row("prod_centered_forward_state_abs_fp64_backward", prod_state_abs64_backward, ref_grad),
        gradient_row("prod_centered_forward_state_abs_fp32_backward", prod_state_abs32_backward, ref_grad),
    ]

    result = {
        "family_index": family_index,
        "family_name": family_name,
        "family_tree": str(tree_path),
        "theta": [float(x) for x in theta_row.tolist()],
        "losses": {
            "reference_fp64_abs": float(ref64["loss"].detach().cpu()),
            "current_abs_fp32": float(abs32["loss"].detach().cpu()),
            "centered_integer_fp32": float(prod32["loss"].detach().cpu()),
        },
        "solve_elapsed_s": {
            "reference_fp64_abs": float(ref64["elapsed_s"]),
            "current_abs_fp32": float(abs32["elapsed_s"]),
            "centered_integer_fp32": float(prod32["elapsed_s"]),
        },
        "gradient_substitution_rows": rows,
        "forward_state_substitution_rows": [
            gradient_row("reference_fp64_abs_forward_fp64_backward", ref_vjp, ref_grad),
            gradient_row("prod_E_bundle_only_abs_fp64_backward", prod_e_bundle_on_ref, ref_grad),
            gradient_row("prod_Pi_bundle_only_abs_fp64_backward", prod_pi_bundle_on_ref, ref_grad),
            gradient_row("prod_root_Pi_only_abs_fp64_backward", prod_root_pi_only_on_ref, ref_grad),
            gradient_row("prod_nonroot_Pi_only_abs_fp64_backward", prod_nonroot_pi_only_on_ref, ref_grad),
            gradient_row("prod_Pibar_only_abs_fp64_backward", prod_pibar_only_on_ref, ref_grad),
            gradient_row("prod_pibar_row_max_only_abs_fp64_backward", prod_pibar_row_max_only_on_ref, ref_grad),
            gradient_row("prod_param_bundle_only_abs_fp64_backward", prod_param_bundle_on_ref, ref_grad),
            gradient_row("prod_E_plus_Pi_bundle_abs_fp64_backward", prod_e_pi_on_ref, ref_grad),
            gradient_row("prod_centered_forward_state_abs_fp64_backward", prod_state_abs64_backward, ref_grad),
        ],
        "fp32_floor_bound_rows": (
            fp32_floor_bound_rows(args, model, ref64, prod32, theta_row, ref_grad)
            if bool(getattr(args, "include_floor_bounds", False))
            else []
        ),
        "forward_tensor_summaries": forward_summaries,
        "top_wave_errors": wave_summaries,
        "backward_same_forward_state_centered_fp32_vs_abs_fp64": {
            "diagnostic_tensor_errors": diagnostic_tensor_comparisons(
                prod_vjp["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
                top_k=args.top_k,
            ),
            "wave_tensor_errors": wave_tensor_comparisons(
                prod_vjp["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
                top_k=args.top_k,
            ),
            "pibar_tree_errors": pibar_tree_comparisons(
                prod_vjp["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
                top_k=args.top_k,
            ),
            "parameter_projection": parameter_projection_details(
                prod_vjp["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
            ),
            "log_param_source_breakdown": log_param_source_breakdown(
                prod_vjp["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
                top_k=args.top_k,
            ),
        },
        "backward_same_forward_state_abs_fp32_vs_abs_fp64": {
            "diagnostic_tensor_errors": diagnostic_tensor_comparisons(
                prod_state_abs32_backward["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
                top_k=args.top_k,
            ),
            "wave_tensor_errors": wave_tensor_comparisons(
                prod_state_abs32_backward["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
                top_k=args.top_k,
            ),
            "pibar_tree_errors": pibar_tree_comparisons(
                prod_state_abs32_backward["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
                top_k=args.top_k,
            ),
            "parameter_projection": parameter_projection_details(
                prod_state_abs32_backward["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
            ),
            "log_param_source_breakdown": log_param_source_breakdown(
                prod_state_abs32_backward["diagnostics"],
                prod_state_abs64_backward["diagnostics"],
                top_k=args.top_k,
            ),
        },
        "backward_same_forward_state_centered_fp32_vs_abs_fp32": {
            "diagnostic_tensor_errors": diagnostic_tensor_comparisons(
                prod_vjp["diagnostics"],
                prod_state_abs32_backward["diagnostics"],
                top_k=args.top_k,
            ),
            "wave_tensor_errors": wave_tensor_comparisons(
                prod_vjp["diagnostics"],
                prod_state_abs32_backward["diagnostics"],
                top_k=args.top_k,
            ),
            "pibar_tree_errors": pibar_tree_comparisons(
                prod_vjp["diagnostics"],
                prod_state_abs32_backward["diagnostics"],
                top_k=args.top_k,
            ),
            "parameter_projection": parameter_projection_details(
                prod_vjp["diagnostics"],
                prod_state_abs32_backward["diagnostics"],
            ),
            "log_param_source_breakdown": log_param_source_breakdown(
                prod_vjp["diagnostics"],
                prod_state_abs32_backward["diagnostics"],
                top_k=args.top_k,
            ),
        },
        "backward_oracle_ref_state_centered_fp32_vs_abs_fp64": {
            "diagnostic_tensor_errors": diagnostic_tensor_comparisons(
                oracle_ref_forward_fp32_backward["diagnostics"],
                ref_vjp["diagnostics"],
                top_k=args.top_k,
            ),
            "wave_tensor_errors": wave_tensor_comparisons(
                oracle_ref_forward_fp32_backward["diagnostics"],
                ref_vjp["diagnostics"],
                top_k=args.top_k,
            ),
            "pibar_tree_errors": pibar_tree_comparisons(
                oracle_ref_forward_fp32_backward["diagnostics"],
                ref_vjp["diagnostics"],
                top_k=args.top_k,
            ),
            "parameter_projection": parameter_projection_details(
                oracle_ref_forward_fp32_backward["diagnostics"],
                ref_vjp["diagnostics"],
            ),
            "log_param_source_breakdown": log_param_source_breakdown(
                oracle_ref_forward_fp32_backward["diagnostics"],
                ref_vjp["diagnostics"],
                top_k=args.top_k,
            ),
        },
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    family_names = list(checkpoint["family_names"])
    theta_cpu = checkpoint["theta"].detach().cpu().double()

    rows = []
    for family_index in args.family_indices:
        family_name = family_names[family_index]
        row = analyze_family(args, family_index, family_name, theta_cpu[family_index])
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "family_done",
                    "family_index": family_index,
                    "family_name": family_name,
                    "gradient_substitution_rows": row["gradient_substitution_rows"],
                    "root_seed_rel": row["forward_tensor_summaries"]["centered_integer_fp32_vs_ref64"]["root_seed"]["rel_l2"],
                    "pi_abs_rel": row["forward_tensor_summaries"]["centered_integer_fp32_vs_ref64"]["Pi_abs"]["rel_l2"],
                }
            ),
            flush=True,
        )

    result = {
        "benchmark": "hogenom_centered_error_source_diagnostic",
        "checkpoint": str(args.checkpoint),
        "species_tree": str(args.species_tree),
        "families_file": str(args.families_file),
        "family_indices": args.family_indices,
        "e_iters": int(args.e_iters),
        "pi_iters": int(args.pi_iters),
        "neumann_terms": int(args.neumann_terms),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "done", "output_json": str(args.output_json)}), flush=True)


if __name__ == "__main__":
    main()
