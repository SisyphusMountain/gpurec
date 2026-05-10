"""Diagnostic helpers for cladewise Pi fixed-point iteration bounds.

This module is intentionally notebook-facing.  It computes the same-clade
linear Pi recurrence after E has converged:

    z_{k+1} = C_gamma + L z_k

with strict subclades fixed at high-pass values.  The diagnostic compares the
``bounds.md`` a priori cladewise scale

    A_gamma = min(1, max_e C_gamma,e / ((I - L)w)_e)

against actual same-clade Pi error for the HOGENOM high-rate families.  It also
computes progressively tighter certified bounds that use only E, L, and
C_gamma: the global c^n envelope, the precomputed vector decay L^n w, and a
finite clade-specific Neumann tail with a certified remainder.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch

from gpurec import GeneReconModel
from gpurec.api.autograd import _extract_parameters
from gpurec.core.forward import _compute_dts_cross, _get_species_wave_helpers
from gpurec.core.likelihood import E_fixed_point
from gpurec.core.log2_utils import logsumexp2


NEG_INF = float("-inf")


DEFAULT_CASES = [
    {
        "case": "family_0223_opt20",
        "family_id": 223,
        "rates": (0.129369, 1.85668, 1.67442),
    },
    {
        "case": "family_0444_x1_5",
        "family_id": 444,
        "rates": (1.5, 1.5, 1.5),
    },
    {
        "case": "family_0952_x1_5",
        "family_id": 952,
        "rates": (1.5, 1.5, 1.5),
    },
    {
        "case": "family_0631_x2",
        "family_id": 631,
        "rates": (2.0, 2.0, 2.0),
    },
]

DEFAULT_PASS_COUNTS = [
    1,
    2,
    3,
    4,
    6,
    8,
    10,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
    29,
    31,
    35,
    40,
    50,
    75,
    100,
    150,
    200,
    300,
]

DEFAULT_TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]


@dataclass
class BoundTerms:
    device: torch.device
    dtype: torch.dtype
    static: object
    S: int
    child1: torch.Tensor
    child2: torch.Tensor
    ancestors_t: torch.Tensor
    log_pS: torch.Tensor
    log_pD: torch.Tensor
    log_pL: torch.Tensor
    max_transfer: torch.Tensor
    pS: torch.Tensor
    pD: torch.Tensor
    pL: torch.Tensor
    transfer_scale: torch.Tensor
    E_log: torch.Tensor
    E_prob: torch.Tensor
    Ebar_log: torch.Tensor
    Ebar_prob: torch.Tensor
    survival: torch.Tensor
    L_survival: torch.Tensor
    s: torch.Tensor
    survival_max: float
    diag_coeff: torch.Tensor
    internal_mask: torch.Tensor
    c: float
    e_iterations: int


def _safe_log2(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, torch.log2(x), torch.full_like(x, NEG_INF))


def _linear_from_log2(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(x), torch.exp2(x), torch.zeros_like(x))


def _as_species_vector(x: torch.Tensor, S: int) -> torch.Tensor:
    if x.ndim == 0:
        return x.expand(S)
    if x.ndim == 1 and x.numel() == 1:
        return x.reshape(()).expand(S)
    if x.ndim == 1 and x.numel() == S:
        return x
    raise ValueError(f"expected scalar or [{S}] tensor, got shape {tuple(x.shape)}")


def _family_gene_path(dataset_dir: Path, family_id: int) -> Path:
    path = dataset_dir / f"g_{family_id}.nwk"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _build_model(
    dataset_dir: Path,
    family_id: int,
    rates: tuple[float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
    preprocess_cache_dir: str | Path,
    max_wave_size: int,
) -> GeneReconModel:
    return GeneReconModel.from_trees(
        species_tree=str(dataset_dir / "sp.nwk"),
        gene_trees=[str(_family_gene_path(dataset_dir, family_id))],
        mode="global",
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
        theta_init_rates=rates,
        preprocess_cache_dir=str(preprocess_cache_dir),
        fixed_iters_E=None,
        fixed_iters_Pi=1,
        max_iters_E=2000,
        tol_E=1e-12,
        max_iters_Pi=1,
        max_wave_size=max_wave_size,
        use_pruning=False,
    )


def _extract_bound_terms(
    model: GeneReconModel,
    *,
    e_tol: float,
    e_max_iters: int,
) -> BoundTerms:
    static = model.static
    device = torch.device(static.device)
    dtype = static.dtype
    S = int(static.species_helpers["S"])

    (
        child1,
        child2,
        _parent,
        _max_depth,
        _child1_cpu,
        _child2_cpu,
        _parent_cpu,
        _ancestor_cols,
        _ancestor_csr_indptr,
        _ancestor_csr_indices,
    ) = _get_species_wave_helpers(static.species_helpers, S, device, True)
    child1 = child1.to(device=device, dtype=torch.long)
    child2 = child2.to(device=device, dtype=torch.long)

    log_pS, log_pD, log_pL, transfer_mat, max_transfer = _extract_parameters(
        model.theta.detach(),
        static,
    )
    max_transfer = max_transfer.squeeze(-1) if max_transfer.ndim > 1 else max_transfer

    E_out = E_fixed_point(
        species_helpers=static.species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=transfer_mat,
        max_transfer_mat=max_transfer,
        max_iters=e_max_iters,
        tolerance=e_tol,
        warm_start_E=None,
        dtype=dtype,
        device=device,
        pibar_mode=static.pibar_mode,
        ancestors_T=static.ancestors_T,
    )

    pS = _as_species_vector(torch.exp2(log_pS), S)
    pD = _as_species_vector(torch.exp2(log_pD), S)
    pL = _as_species_vector(torch.exp2(log_pL), S)
    transfer_scale = _as_species_vector(torch.exp2(max_transfer), S)
    E_prob = _linear_from_log2(E_out["E"])
    Ebar_prob = _linear_from_log2(E_out["E_bar"])
    survival = 1.0 - E_prob

    diag_coeff = 2.0 * pD * E_prob + Ebar_prob
    internal_mask = (child1 < S) & (child2 < S)

    placeholder = BoundTerms(
        device=device,
        dtype=dtype,
        static=static,
        S=S,
        child1=child1,
        child2=child2,
        ancestors_t=static.ancestors_T.to(device=device, dtype=dtype),
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        max_transfer=max_transfer,
        pS=pS,
        pD=pD,
        pL=pL,
        transfer_scale=transfer_scale,
        E_log=E_out["E"],
        E_prob=E_prob,
        Ebar_log=E_out["E_bar"],
        Ebar_prob=Ebar_prob,
        survival=survival,
        L_survival=torch.zeros_like(survival),
        s=torch.zeros_like(survival),
        survival_max=float("nan"),
        diag_coeff=diag_coeff,
        internal_mask=internal_mask,
        c=float("nan"),
        e_iterations=int(E_out["iterations"]),
    )
    Lw = apply_L_linear(placeholder.survival.unsqueeze(0), placeholder).squeeze(0)
    s = placeholder.survival - Lw
    if bool((s <= 0).any()):
        min_s = float(s.min().detach().cpu())
        raise RuntimeError(f"(I - L)w must be positive componentwise, got min={min_s}")
    ratio = Lw / placeholder.survival.clamp_min(torch.finfo(dtype).tiny)
    c = float(ratio.max().detach().cpu())
    if not (0.0 <= c < 1.0):
        raise RuntimeError(f"cladewise contraction constant is outside [0, 1): c={c}")
    placeholder.L_survival = Lw
    placeholder.s = s
    placeholder.survival_max = float(placeholder.survival.max().detach().cpu())
    placeholder.c = c
    return placeholder


def apply_L_linear(z: torch.Tensor, terms: BoundTerms) -> torch.Tensor:
    """Apply the Appendix A same-clade linear operator to row vectors."""
    out = z * terms.diag_coeff

    row_sum = z.sum(dim=1, keepdim=True)
    ancestor_sum = z @ terms.ancestors_t
    eligible_sum = (row_sum - ancestor_sum).clamp_min(0.0)
    out = out + eligible_sum * (terms.E_prob * terms.transfer_scale)

    idx = torch.nonzero(terms.internal_mask, as_tuple=False).flatten()
    if idx.numel() > 0:
        c1 = terms.child1[idx]
        c2 = terms.child2[idx]
        out[:, idx] = out[:, idx] + terms.pS[idx] * (
            terms.E_prob[c2] * z[:, c1] + terms.E_prob[c1] * z[:, c2]
        )
    return out


def pibar_from_linear(z: torch.Tensor, terms: BoundTerms) -> torch.Tensor:
    row_sum = z.sum(dim=1, keepdim=True)
    ancestor_sum = z @ terms.ancestors_t
    eligible_sum = (row_sum - ancestor_sum).clamp_min(0.0)
    return _safe_log2(eligible_sum * terms.transfer_scale)


def weighted_norm_rows(x: torch.Tensor, terms: BoundTerms) -> torch.Tensor:
    denom = terms.survival.clamp_min(torch.finfo(terms.dtype).tiny)
    return (x.abs() / denom).amax(dim=1)


def absolute_norm_rows(x: torch.Tensor) -> torch.Tensor:
    return x.abs().amax(dim=1)


def _cladewise_scale(C_gamma: torch.Tensor, terms: BoundTerms) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A_gamma, max_e C_gamma,e / s_e) for each clade row."""
    denom = terms.s.clamp_min(torch.finfo(terms.dtype).tiny)
    c_over_s = (C_gamma.clamp_min(0.0) / denom).amax(dim=1)
    return torch.minimum(c_over_s, torch.ones_like(c_over_s)), c_over_s


def _decay_curves(terms: BoundTerms, max_passes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute Q_n=max_e (L^n w)_e/w_e and R_n=max_e (L^n w)_e."""
    weighted = torch.empty((max_passes + 1,), device=terms.device, dtype=terms.dtype)
    absolute = torch.empty_like(weighted)
    v = terms.survival.unsqueeze(0)
    for pass_idx in range(max_passes + 1):
        weighted[pass_idx] = weighted_norm_rows(v, terms)[0]
        absolute[pass_idx] = absolute_norm_rows(v)[0]
        if pass_idx < max_passes:
            v = apply_L_linear(v, terms)
    return weighted, absolute


def _ceil_required_from_values(values: torch.Tensor, tol: float) -> int:
    hits = torch.nonzero(values <= tol, as_tuple=False).flatten()
    if hits.numel() == 0:
        return -1
    return int(hits[0].detach().cpu())


def _leaf_rows_for_wave(meta: dict, terms: BoundTerms) -> tuple[torch.Tensor, torch.Tensor]:
    wave_layout = terms.static.wave_layout
    leaf_row_index = wave_layout["leaf_row_index"].to(device=terms.device)
    leaf_col_index = wave_layout["leaf_col_index"].to(device=terms.device)
    ws = int(meta["start"])
    we = int(meta["end"])
    mask = (leaf_row_index >= ws) & (leaf_row_index < we)
    return leaf_row_index[mask] - ws, leaf_col_index[mask]


def initial_wave_state(meta: dict, terms: BoundTerms, *, leaf_observation_init: bool) -> torch.Tensor:
    z = torch.zeros((int(meta["W"]), terms.S), device=terms.device, dtype=terms.dtype)
    if leaf_observation_init:
        rows, cols = _leaf_rows_for_wave(meta, terms)
        if rows.numel() > 0:
            z[rows.long(), cols.long()] = 1.0
    return z


def leaf_constant(meta: dict, terms: BoundTerms) -> torch.Tensor:
    c = torch.zeros((int(meta["W"]), terms.S), device=terms.device, dtype=terms.dtype)
    rows, cols = _leaf_rows_for_wave(meta, terms)
    if rows.numel() > 0:
        c[rows.long(), cols.long()] = terms.pS[cols.long()]
    return c


def _compute_dts_cross_torch(
    Pi_log: torch.Tensor,
    Pibar_log: torch.Tensor,
    meta: dict,
    terms: BoundTerms,
) -> torch.Tensor:
    """Portable fallback for CPU runs.  CUDA runs use GPUREC's fused kernel."""
    W = int(meta["W"])
    S = terms.S
    sl = meta["sl"].long().to(terms.device)
    sr = meta["sr"].long().to(terms.device)
    if sl.numel() == 0:
        return torch.full((W, S), NEG_INF, device=terms.device, dtype=terms.dtype)

    left = Pi_log.index_select(0, sl)
    right = Pi_log.index_select(0, sr)
    left_bar = Pibar_log.index_select(0, sl)
    right_bar = Pibar_log.index_select(0, sr)

    d0 = terms.log_pD + left + right
    d1 = left + right_bar
    d2 = right + left_bar

    c1 = terms.child1.clamp(max=S - 1)
    c2 = terms.child2.clamp(max=S - 1)
    valid = terms.internal_mask
    left_c1 = left.index_select(1, c1)
    left_c2 = left.index_select(1, c2)
    right_c1 = right.index_select(1, c1)
    right_c2 = right.index_select(1, c2)
    d3 = terms.log_pS + left_c1 + right_c2
    d4 = terms.log_pS + right_c1 + left_c2
    d3[:, ~valid] = NEG_INF
    d4[:, ~valid] = NEG_INF

    lsp = meta["log_split_probs"].reshape(-1).to(device=terms.device, dtype=terms.dtype)
    dts_term = lsp[:, None] + logsumexp2(torch.stack([d0, d1, d2, d3, d4], dim=0), dim=0)

    dts_r = torch.full((W, S), NEG_INF, device=terms.device, dtype=terms.dtype)
    reduce_idx = meta["reduce_idx"].long().to(terms.device)
    for split_idx in range(sl.numel()):
        row = int(reduce_idx[split_idx])
        dts_r[row] = torch.logaddexp(dts_r[row], dts_term[split_idx])
    return dts_r


def constant_for_wave(
    Pi_log: torch.Tensor,
    Pibar_log: torch.Tensor,
    meta: dict,
    terms: BoundTerms,
) -> torch.Tensor:
    c = leaf_constant(meta, terms)
    if not bool(meta.get("has_splits", False)):
        return c

    if terms.device.type == "cuda":
        dts_log = _compute_dts_cross(
            Pi_log,
            Pibar_log,
            meta,
            terms.child1,
            terms.child2,
            terms.log_pD,
            terms.log_pS,
            terms.S,
            terms.device,
            terms.dtype,
            parent_reduced=False,
        )
    else:
        dts_log = _compute_dts_cross_torch(Pi_log, Pibar_log, meta, terms)
    return c + _linear_from_log2(dts_log)


def nll_bits_from_root_linear(root_z: torch.Tensor, terms: BoundTerms) -> float:
    numerator = torch.log2(root_z.sum().clamp_min(torch.finfo(terms.dtype).tiny))
    numerator = numerator - math.log2(float(terms.S))
    denominator = torch.log2((1.0 - terms.E_prob.mean()).clamp_min(torch.finfo(terms.dtype).tiny))
    return float((-(numerator - denominator)).detach().cpu())


def nll_error_bound_bits_from_weighted_bound(
    root_z: torch.Tensor,
    weighted_bound: torch.Tensor,
    terms: BoundTerms,
) -> float:
    """Convert a weighted absolute Pi bound into a root NLL bits bound.

    If ``||z_star - z||_{w,inf} <= b``, then
    ``abs(sum(z_star) - sum(z)) <= b * sum(w)``.  The log bound is finite only
    when this absolute mass bound is smaller than the current root mass.
    """
    current_mass = root_z.sum().detach()
    mass_error = weighted_bound.detach() * terms.survival.sum()
    if not bool(torch.isfinite(mass_error)) or float(mass_error.cpu()) < 0.0:
        return float("inf")
    if not bool(current_mass > mass_error):
        return float("inf")
    upper = current_mass + mass_error
    lower = current_mass - mass_error
    hi = torch.log2(upper / current_mass)
    lo = torch.log2(current_mass / lower)
    return float(torch.maximum(hi, lo).cpu())


def _empty_required(W: int, tolerances: Iterable[float], device: torch.device) -> dict[float, torch.Tensor]:
    return {
        float(tol): torch.full((W,), -1, dtype=torch.long, device=device)
        for tol in tolerances
    }


def _update_required(
    required: dict[float, torch.Tensor],
    bound: torch.Tensor,
    pass_idx: int,
) -> None:
    for tol, values in required.items():
        newly_done = (values < 0) & (bound <= tol)
        values[newly_done] = pass_idx


def _root_local_indices(meta: dict, terms: BoundTerms) -> torch.Tensor:
    root_ids = terms.static.wave_layout["root_clade_ids"].to(device=terms.device)
    ws = int(meta["start"])
    we = int(meta["end"])
    mask = (root_ids >= ws) & (root_ids < we)
    return root_ids[mask].long() - ws


def run_case(
    case: dict,
    *,
    dataset_dir: Path,
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    preprocess_cache_dir: str | Path,
    max_passes: int,
    reference_passes: int,
    pass_counts: Iterable[int],
    tolerances: Iterable[float],
    e_tol: float,
    e_max_iters: int,
    max_wave_size: int,
    leaf_observation_init: bool,
    ratio_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if leaf_observation_init:
        raise ValueError(
            "The cladewise a priori tail bounds assume z_0=0. "
            "Set leaf_observation_init=False for this diagnostic."
        )
    if reference_passes < max_passes:
        raise ValueError(
            f"reference_passes ({reference_passes}) must be >= max_passes ({max_passes})"
        )

    pass_counts = sorted({int(p) for p in pass_counts if 1 <= int(p) <= max_passes})
    tolerances = [float(t) for t in tolerances]

    model = _build_model(
        dataset_dir,
        int(case["family_id"]),
        tuple(case["rates"]),
        device=device,
        dtype=dtype,
        preprocess_cache_dir=preprocess_cache_dir,
        max_wave_size=max_wave_size,
    )
    terms = _extract_bound_terms(model, e_tol=e_tol, e_max_iters=e_max_iters)
    wave_layout = terms.static.wave_layout
    wave_metas = wave_layout["wave_metas"]
    C_total = int(wave_layout["ccp_helpers"]["C"])
    root_ids_cpu = {int(x) for x in wave_layout["root_clade_ids"].detach().cpu().tolist()}

    decay_weighted, decay_absolute = _decay_curves(terms, max_passes)
    c_powers = torch.tensor(
        [terms.c ** pass_idx for pass_idx in range(max_passes + 1)],
        device=device,
        dtype=dtype,
    )

    Pi_log = torch.full((C_total, terms.S), NEG_INF, device=device, dtype=dtype)
    Pibar_log = torch.full_like(Pi_log, NEG_INF)

    curve_names = [
        "actual_weighted_error",
        "actual_absolute_error",
        "global_c_bound",
        "global_c_absolute_bound",
        "vector_decay_bound",
        "vector_decay_absolute_bound",
        "finite_tail_weighted_bound",
        "finite_tail_absolute_bound",
        "validation_weighted_ratio",
        "validation_absolute_ratio",
    ]
    worst_curves = {
        name: torch.zeros((max_passes,), device=device, dtype=dtype)
        for name in curve_names
    }
    worst_reference_residual_weighted = torch.zeros((), device=device, dtype=dtype)
    worst_reference_residual_absolute = torch.zeros((), device=device, dtype=dtype)
    validation_weighted_violations = 0
    validation_absolute_violations = 0
    validation_weighted_max_excess = 0.0
    validation_absolute_max_excess = 0.0
    validation_slack = max(float(ratio_floor), 1000.0 * torch.finfo(dtype).eps)

    root_rows: list[dict] = []
    clade_rows: list[dict] = []
    selected_passes = set(pass_counts)
    case_t0 = time.perf_counter()

    for wave_idx, meta in enumerate(wave_metas):
        ws = int(meta["start"])
        we = int(meta["end"])
        W = int(meta["W"])
        C_gamma = constant_for_wave(Pi_log, Pibar_log, meta, terms)
        A_gamma, C_over_s_max = _cladewise_scale(C_gamma, terms)

        z_ref = torch.zeros_like(C_gamma)
        u = C_gamma.clone()
        for _ in range(reference_passes):
            z_ref = z_ref + u
            u = apply_L_linear(u, terms)
        u_H = u.clamp_min(0.0)
        reference_residual_weighted = (u_H / terms.s.clamp_min(torch.finfo(dtype).tiny)).amax(dim=1)
        reference_remainder = reference_residual_weighted.unsqueeze(1) * terms.survival.unsqueeze(0)
        reference_residual_absolute = absolute_norm_rows(reference_remainder)
        worst_reference_residual_weighted = torch.maximum(
            worst_reference_residual_weighted,
            reference_residual_weighted.max(),
        )
        worst_reference_residual_absolute = torch.maximum(
            worst_reference_residual_absolute,
            reference_residual_absolute.max(),
        )

        root_local = _root_local_indices(meta, terms)
        root_ref_nll = [
            nll_bits_from_root_linear(z_ref[int(local)], terms)
            for local in root_local
        ]

        weighted_required = {
            "global_c": _empty_required(W, tolerances, device),
            "vector_decay": _empty_required(W, tolerances, device),
            "finite_tail": _empty_required(W, tolerances, device),
        }
        absolute_required = {
            "global_c": _empty_required(W, tolerances, device),
            "vector_decay": _empty_required(W, tolerances, device),
            "finite_tail": _empty_required(W, tolerances, device),
        }

        def update_required_for_pass(pass_idx: int, finite_weighted: torch.Tensor, finite_absolute: torch.Tensor) -> None:
            global_weighted = A_gamma * c_powers[pass_idx]
            global_absolute = global_weighted * terms.survival_max
            vector_weighted = A_gamma * decay_weighted[pass_idx]
            vector_absolute = A_gamma * decay_absolute[pass_idx]
            _update_required(weighted_required["global_c"], global_weighted, pass_idx)
            _update_required(weighted_required["vector_decay"], vector_weighted, pass_idx)
            _update_required(weighted_required["finite_tail"], finite_weighted, pass_idx)
            _update_required(absolute_required["global_c"], global_absolute, pass_idx)
            _update_required(absolute_required["vector_decay"], vector_absolute, pass_idx)
            _update_required(absolute_required["finite_tail"], finite_absolute, pass_idx)

        initial_tail = z_ref.clamp_min(0.0) + reference_remainder
        update_required_for_pass(
            0,
            weighted_norm_rows(initial_tail, terms),
            absolute_norm_rows(initial_tail),
        )

        z = torch.zeros_like(C_gamma)
        u = C_gamma.clone()
        selected_values: dict[int, dict[str, torch.Tensor]] = {}
        for pass_idx in range(1, max_passes + 1):
            z = z + u
            u = apply_L_linear(u, terms)

            ref_diff = (z_ref - z).clamp_min(0.0)
            actual_weighted = weighted_norm_rows(ref_diff, terms)
            actual_absolute = absolute_norm_rows(ref_diff)
            finite_tail_component = ref_diff + reference_remainder
            finite_tail_weighted = weighted_norm_rows(finite_tail_component, terms)
            finite_tail_absolute = absolute_norm_rows(finite_tail_component)
            global_c_bound = A_gamma * c_powers[pass_idx]
            global_c_absolute_bound = global_c_bound * terms.survival_max
            vector_decay_bound = A_gamma * decay_weighted[pass_idx]
            vector_decay_absolute_bound = A_gamma * decay_absolute[pass_idx]

            update_required_for_pass(pass_idx, finite_tail_weighted, finite_tail_absolute)

            weighted_guard = finite_tail_weighted + reference_residual_weighted
            absolute_guard = finite_tail_absolute + reference_residual_absolute
            weighted_excess = actual_weighted - weighted_guard
            absolute_excess = actual_absolute - absolute_guard
            validation_weighted_max_excess = max(
                validation_weighted_max_excess,
                float(weighted_excess.max().detach().cpu()),
            )
            validation_absolute_max_excess = max(
                validation_absolute_max_excess,
                float(absolute_excess.max().detach().cpu()),
            )
            validation_weighted_violations += int((weighted_excess > validation_slack).sum().detach().cpu())
            validation_absolute_violations += int((absolute_excess > validation_slack).sum().detach().cpu())
            weighted_ratio_mask = (weighted_guard > ratio_floor) | (actual_weighted > ratio_floor)
            absolute_ratio_mask = (absolute_guard > ratio_floor) | (actual_absolute > ratio_floor)
            weighted_ratio = torch.where(
                weighted_ratio_mask & (weighted_guard > 0),
                actual_weighted / weighted_guard,
                torch.zeros_like(actual_weighted),
            )
            absolute_ratio = torch.where(
                absolute_ratio_mask & (absolute_guard > 0),
                actual_absolute / absolute_guard,
                torch.zeros_like(actual_absolute),
            )

            per_pass = {
                "actual_weighted_error": actual_weighted,
                "actual_absolute_error": actual_absolute,
                "global_c_bound": global_c_bound,
                "global_c_absolute_bound": global_c_absolute_bound,
                "vector_decay_bound": vector_decay_bound,
                "vector_decay_absolute_bound": vector_decay_absolute_bound,
                "finite_tail_weighted_bound": finite_tail_weighted,
                "finite_tail_absolute_bound": finite_tail_absolute,
                "validation_weighted_ratio": weighted_ratio,
                "validation_absolute_ratio": absolute_ratio,
            }
            for name, values in per_pass.items():
                worst_curves[name][pass_idx - 1] = torch.maximum(
                    worst_curves[name][pass_idx - 1],
                    values.max(),
                )

            if pass_idx in selected_passes:
                selected_values[pass_idx] = {
                    name: values.detach().cpu()
                    for name, values in per_pass.items()
                    if name not in ("validation_weighted_ratio", "validation_absolute_ratio")
                }
                if root_local.numel() > 0:
                    for root_offset, ref_nll in enumerate(root_ref_nll):
                        local = int(root_local[root_offset])
                        nll = nll_bits_from_root_linear(z[local], terms)
                        root_rows.append(
                            {
                                "case": case["case"],
                                "family_id": int(case["family_id"]),
                                "pass": pass_idx,
                                "root_offset": root_offset,
                                "root_nll_bits": nll,
                                "root_ref_nll_bits": ref_nll,
                                "root_abs_nll_error_bits": abs(nll - ref_nll),
                                "root_actual_weighted_error": float(actual_weighted[local].detach().cpu()),
                                "root_actual_absolute_error": float(actual_absolute[local].detach().cpu()),
                                "root_global_c_bound": float(global_c_bound[local].detach().cpu()),
                                "root_global_c_absolute_bound": float(global_c_absolute_bound[local].detach().cpu()),
                                "root_vector_decay_bound": float(vector_decay_bound[local].detach().cpu()),
                                "root_vector_decay_absolute_bound": float(vector_decay_absolute_bound[local].detach().cpu()),
                                "root_finite_tail_weighted_bound": float(finite_tail_weighted[local].detach().cpu()),
                                "root_finite_tail_absolute_bound": float(finite_tail_absolute[local].detach().cpu()),
                                "root_reference_residual_weighted_bound": float(reference_residual_weighted[local].detach().cpu()),
                                "root_reference_residual_absolute_bound": float(reference_residual_absolute[local].detach().cpu()),
                                "root_weighted_bound": float(finite_tail_weighted[local].detach().cpu()),
                                "root_weighted_actual_ref_error": float(actual_weighted[local].detach().cpu()),
                                "root_nll_bound_bits": nll_error_bound_bits_from_weighted_bound(
                                    z[local],
                                    finite_tail_weighted[local],
                                    terms,
                                ),
                            }
                        )

        weighted_required_cpu = {
            name: {tol: values.detach().cpu() for tol, values in required.items()}
            for name, required in weighted_required.items()
        }
        absolute_required_cpu = {
            name: {tol: values.detach().cpu() for tol, values in required.items()}
            for name, required in absolute_required.items()
        }
        A_gamma_cpu = A_gamma.detach().cpu()
        C_over_s_cpu = C_over_s_max.detach().cpu()
        reference_residual_weighted_cpu = reference_residual_weighted.detach().cpu()
        reference_residual_absolute_cpu = reference_residual_absolute.detach().cpu()
        for local in range(W):
            clade_id = ws + local
            row = {
                "case": case["case"],
                "family_id": int(case["family_id"]),
                "wave": wave_idx,
                "clade_id": clade_id,
                "is_root": clade_id in root_ids_cpu,
                "A_gamma": float(A_gamma_cpu[local]),
                "C_over_s_max": float(C_over_s_cpu[local]),
                "reference_residual_weighted_bound": float(reference_residual_weighted_cpu[local]),
                "reference_residual_absolute_bound": float(reference_residual_absolute_cpu[local]),
            }
            for tol in tolerances:
                # Backward-compatible alias: the tightest diagnostic weighted bound.
                row[f"required_pass_tol_{tol:.0e}"] = int(
                    weighted_required_cpu["finite_tail"][tol][local]
                )
                for bound_name in ("global_c", "vector_decay", "finite_tail"):
                    row[f"{bound_name}_required_weighted_pass_tol_{tol:.0e}"] = int(
                        weighted_required_cpu[bound_name][tol][local]
                    )
                    row[f"{bound_name}_required_absolute_pass_tol_{tol:.0e}"] = int(
                        absolute_required_cpu[bound_name][tol][local]
                    )
            for pass_idx in pass_counts:
                if pass_idx in selected_values:
                    values = selected_values[pass_idx]
                    row[f"actual_weighted_error_pass_{pass_idx}"] = float(
                        values["actual_weighted_error"][local]
                    )
                    row[f"actual_absolute_error_pass_{pass_idx}"] = float(
                        values["actual_absolute_error"][local]
                    )
                    row[f"global_c_bound_pass_{pass_idx}"] = float(
                        values["global_c_bound"][local]
                    )
                    row[f"global_c_absolute_bound_pass_{pass_idx}"] = float(
                        values["global_c_absolute_bound"][local]
                    )
                    row[f"vector_decay_bound_pass_{pass_idx}"] = float(
                        values["vector_decay_bound"][local]
                    )
                    row[f"vector_decay_absolute_bound_pass_{pass_idx}"] = float(
                        values["vector_decay_absolute_bound"][local]
                    )
                    row[f"finite_tail_weighted_bound_pass_{pass_idx}"] = float(
                        values["finite_tail_weighted_bound"][local]
                    )
                    row[f"finite_tail_absolute_bound_pass_{pass_idx}"] = float(
                        values["finite_tail_absolute_bound"][local]
                    )
                    # Compatibility with the first notebook version.
                    row[f"actual_ref_err_pass_{pass_idx}"] = row[
                        f"actual_weighted_error_pass_{pass_idx}"
                    ]
                    row[f"bound_pass_{pass_idx}"] = row[
                        f"finite_tail_weighted_bound_pass_{pass_idx}"
                    ]
            clade_rows.append(row)

        Pi_log[ws:we] = _safe_log2(z_ref)
        Pibar_log[ws:we] = pibar_from_linear(z_ref, terms)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - case_t0
        print(
            f"[{case['case']}] wave {wave_idx + 1}/{len(wave_metas)} "
            f"W={W} elapsed={elapsed:.1f}s",
            flush=True,
        )

    if validation_weighted_violations or validation_absolute_violations:
        raise RuntimeError(
            "finite-tail validation failed: "
            f"weighted violations={validation_weighted_violations}, "
            f"absolute violations={validation_absolute_violations}, "
            f"weighted max excess={validation_weighted_max_excess:.3e}, "
            f"absolute max excess={validation_absolute_max_excess:.3e}"
        )

    pass_rows = []
    worst_cpu = {name: values.detach().cpu() for name, values in worst_curves.items()}
    decay_weighted_cpu = decay_weighted.detach().cpu()
    decay_absolute_cpu = decay_absolute.detach().cpu()
    c_powers_cpu = c_powers.detach().cpu()
    for idx in range(max_passes):
        pass_idx = idx + 1
        pass_rows.append(
            {
                "case": case["case"],
                "family_id": int(case["family_id"]),
                "pass": pass_idx,
                "c_power": float(c_powers_cpu[pass_idx]),
                "Q_weighted": float(decay_weighted_cpu[pass_idx]),
                "Q_absolute": float(decay_absolute_cpu[pass_idx]),
                "actual_weighted_error": float(worst_cpu["actual_weighted_error"][idx]),
                "actual_absolute_error": float(worst_cpu["actual_absolute_error"][idx]),
                "global_c_bound": float(worst_cpu["global_c_bound"][idx]),
                "global_c_absolute_bound": float(worst_cpu["global_c_absolute_bound"][idx]),
                "vector_decay_bound": float(worst_cpu["vector_decay_bound"][idx]),
                "vector_decay_absolute_bound": float(worst_cpu["vector_decay_absolute_bound"][idx]),
                "finite_tail_weighted_bound": float(worst_cpu["finite_tail_weighted_bound"][idx]),
                "finite_tail_absolute_bound": float(worst_cpu["finite_tail_absolute_bound"][idx]),
                "validation_weighted_ratio": float(worst_cpu["validation_weighted_ratio"][idx]),
                "validation_absolute_ratio": float(worst_cpu["validation_absolute_ratio"][idx]),
                # Compatibility aliases.
                "worst_actual_ref_error": float(worst_cpu["actual_weighted_error"][idx]),
                "worst_bound": float(worst_cpu["finite_tail_weighted_bound"][idx]),
                "worst_actual_over_bound": float(worst_cpu["validation_weighted_ratio"][idx]),
            }
        )

    clade_df_case = pd.DataFrame(clade_rows)
    summary = {
        "case": case["case"],
        "family_id": int(case["family_id"]),
        "D": float(case["rates"][0]),
        "L": float(case["rates"][1]),
        "T": float(case["rates"][2]),
        "S": terms.S,
        "C": C_total,
        "c": terms.c,
        "survival_max": terms.survival_max,
        "E_iterations": terms.e_iterations,
        "max_passes": max_passes,
        "reference_passes": reference_passes,
        "A_gamma_min": float(clade_df_case["A_gamma"].min()),
        "A_gamma_median": float(clade_df_case["A_gamma"].median()),
        "A_gamma_mean": float(clade_df_case["A_gamma"].mean()),
        "A_gamma_max": float(clade_df_case["A_gamma"].max()),
        "C_over_s_max": float(clade_df_case["C_over_s_max"].max()),
        "reference_residual_weighted_max": float(worst_reference_residual_weighted.detach().cpu()),
        "reference_residual_absolute_max": float(worst_reference_residual_absolute.detach().cpu()),
        "validation_weighted_violations": validation_weighted_violations,
        "validation_absolute_violations": validation_absolute_violations,
        "validation_weighted_max_excess": validation_weighted_max_excess,
        "validation_absolute_max_excess": validation_absolute_max_excess,
        "worst_actual_over_bound": float(worst_curves["validation_weighted_ratio"].max().detach().cpu()),
        "ratio_floor": float(ratio_floor),
        "runtime_seconds": time.perf_counter() - case_t0,
    }
    for tol in tolerances:
        finite_col = f"finite_tail_required_weighted_pass_tol_{tol:.0e}"
        finite_values = clade_df_case[finite_col]
        summary[f"worst_required_pass_tol_{tol:.0e}"] = int(finite_values.max())
        summary[f"root_required_pass_tol_{tol:.0e}"] = int(
            clade_df_case.loc[lambda df: df["is_root"], finite_col].max()
        )
        for bound_name in ("global_c", "vector_decay", "finite_tail"):
            for norm_name in ("weighted", "absolute"):
                col = f"{bound_name}_required_{norm_name}_pass_tol_{tol:.0e}"
                values = clade_df_case[col]
                summary[f"worst_{bound_name}_required_{norm_name}_pass_tol_{tol:.0e}"] = int(values.max())
                summary[f"root_{bound_name}_required_{norm_name}_pass_tol_{tol:.0e}"] = int(
                    clade_df_case.loc[lambda df: df["is_root"], col].max()
                )

    return (
        pd.DataFrame([summary]),
        pd.DataFrame(pass_rows),
        pd.DataFrame(root_rows),
        clade_df_case,
    )


def _plot_outputs(
    out_dir: Path,
    summary_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    root_df: pd.DataFrame,
    clade_df: pd.DataFrame,
    *,
    histogram_tol: float,
    tolerances: Iterable[float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def positive(values: pd.Series) -> pd.Series:
        return values.where(values > 0)

    cases = list(pass_df["case"].drop_duplicates())
    ncols = 2
    nrows = max(1, math.ceil(len(cases) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4.3 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, case_name in zip(axes.flat, cases):
        group = pass_df[pass_df["case"] == case_name]
        ax.plot(group["pass"], positive(group["actual_weighted_error"]), label="actual")
        ax.plot(group["pass"], positive(group["global_c_bound"]), ":", label="A c^n")
        ax.plot(group["pass"], positive(group["vector_decay_bound"]), "-.", label="A Q_n")
        ax.plot(group["pass"], positive(group["finite_tail_weighted_bound"]), "--", label="finite tail")
        ax.set_yscale("log")
        ax.set_xlabel("Pi passes")
        ax.set_ylabel("worst weighted norm")
        ax.set_title(case_name)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    fig.suptitle("Weighted same-clade Pi error vs certified bounds")
    fig.savefig(out_dir / "bound_vs_actual_weighted_norm.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4.3 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, case_name in zip(axes.flat, cases):
        group = pass_df[pass_df["case"] == case_name]
        ax.plot(group["pass"], positive(group["actual_absolute_error"]), label="actual")
        ax.plot(group["pass"], positive(group["global_c_absolute_bound"]), ":", label="A c^n max(w)")
        ax.plot(group["pass"], positive(group["vector_decay_absolute_bound"]), "-.", label="A max L^n w")
        ax.plot(group["pass"], positive(group["finite_tail_absolute_bound"]), "--", label="finite tail")
        ax.set_yscale("log")
        ax.set_xlabel("Pi passes")
        ax.set_ylabel("worst absolute norm")
        ax.set_title(case_name)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    fig.suptitle("Absolute same-clade Pi error vs certified bounds")
    fig.savefig(out_dir / "bound_vs_actual_absolute_norm.png", dpi=180)
    plt.close(fig)

    if not root_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for case_name, group in root_df.groupby("case"):
            ax.plot(group["pass"], positive(group["root_abs_nll_error_bits"]), label=case_name)
            finite_bound = group[group["root_nll_bound_bits"].map(math.isfinite)]
            if not finite_bound.empty:
                ax.plot(
                    finite_bound["pass"],
                    positive(finite_bound["root_nll_bound_bits"]),
                    ":",
                    label=f"{case_name} bound",
                )
        ax.set_yscale("log")
        ax.set_xlabel("Pi passes")
        ax.set_ylabel("root NLL error to reference (bits)")
        ax.set_title("Root likelihood convergence")
        ax.legend(fontsize=8)
        fig.savefig(out_dir / "root_nll_error_bits.png", dpi=180)
        plt.close(fig)

    for tol in tolerances:
        col = f"finite_tail_required_weighted_pass_tol_{float(tol):.0e}"
        if col not in clade_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for case_name, group in clade_df.groupby("case"):
            values = group[col]
            max_value = max(1, int(values.max()))
            ax.hist(values, bins=range(0, max_value + 3), alpha=0.45, label=case_name)
        ax.set_xlabel(f"finite-tail weighted required passes <= {float(tol):.0e}")
        ax.set_ylabel("clades")
        ax.set_title("Per-clade finite-tail pass bound")
        ax.legend(fontsize=8)
        fig.savefig(out_dir / f"required_pass_hist_tol_{float(tol):.0e}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 4.3 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, case_name in zip(axes.flat, cases):
        group = pass_df[pass_df["case"] == case_name]
        ax.plot(group["pass"], positive(group["c_power"]), label="c^n")
        ax.plot(group["pass"], positive(group["Q_weighted"]), label="Q_n")
        ax.plot(group["pass"], positive(group["finite_tail_weighted_bound"]), label="worst finite tail")
        ax.set_yscale("log")
        ax.set_xlabel("Pi passes")
        ax.set_ylabel("weighted scale")
        ax.set_title(case_name)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    fig.suptitle("Decay refinements")
    fig.savefig(out_dir / "decay_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for case_name, group in clade_df.groupby("case"):
        ax.hist(group["A_gamma"], bins=40, alpha=0.45, label=case_name)
    ax.set_xlabel("A_gamma")
    ax.set_ylabel("clades")
    ax.set_title("Cladewise a priori scale distribution")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "A_gamma_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(summary_df["case"], summary_df["c"])
    ax.set_ylim(0.0, min(1.0, max(0.01, float(summary_df["c"].max()) * 1.05)))
    ax.set_ylabel("c = max_e (Lw)_e / w_e")
    ax.set_title("Same-clade contraction constants")
    ax.tick_params(axis="x", rotation=35)
    fig.savefig(out_dir / "contraction_constants.png", dpi=180)
    plt.close(fig)


def _write_readme(
    out_dir: Path,
    summary_df: pd.DataFrame,
    *,
    max_passes: int,
    reference_passes: int,
    e_tol: float,
    leaf_observation_init: bool,
    ratio_floor: float,
    tolerances: Iterable[float],
) -> None:
    summary_table = summary_df.to_csv(index=False)
    tolerance_list = ", ".join(f"{float(tol):.0e}" for tol in tolerances)
    lines = [
        "# Pi Iteration Bound Diagnostic",
        "",
        "This directory was generated by `notebooks/pi_iteration_bound_diagnostic.ipynb`.",
        "",
        "The diagnostic computes the `bounds.md` cladewise a priori scale",
        "`A_gamma = min(1, max_e C_gamma,e / ((I-L)w)_e)` and compares actual",
        "same-clade Pi error against three certified bounds: `A_gamma c^n`,",
        "`A_gamma max_e (L^n w)_e / w_e`, and the finite clade-specific tail.",
        "Each clade is tested with strict subclades fixed at the high-pass",
        "reference values, so this isolates same-clade Pi convergence rather",
        "than the full fixed-pass DAG error accumulated from lower clades.",
        "",
        "The finite-tail bound is the tightest diagnostic bound.  `global_c`",
        "and `vector_decay` columns are retained to show how much each",
        "refinement buys.  `root_required_pass_*` is still a weighted-Pi-norm",
        "criterion.  Use",
        "`root_curves.csv` for the separate root NLL error and NLL bound.",
        "",
        f"- E tolerance: `{e_tol}`",
        f"- Plotted/search pass horizon: `{max_passes}` Pi passes per clade wave",
        f"- High-pass reference and finite-tail horizon: `{reference_passes}` Pi passes per clade wave",
        f"- Leaf observation initialization: `{leaf_observation_init}`",
        f"- Actual/bound ratio floor: `{ratio_floor}`",
        f"- Tolerances: `{tolerance_list}`",
        "",
        "Main outputs:",
        "",
        "- `case_summary.csv`",
        "- `pass_curves.csv`",
        "- `root_curves.csv`",
        "- `clade_pass_bounds.csv`",
        "- `bound_vs_actual_weighted_norm.png`",
        "- `bound_vs_actual_absolute_norm.png`",
        "- `root_nll_error_bits.png`",
        "- `A_gamma_distribution.png`",
        "- `decay_comparison.png`",
        "- `contraction_constants.png`",
        "- `required_pass_hist_tol_*.png`",
        "",
        "Summary:",
        "",
        "```csv",
        summary_table.rstrip(),
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def run_diagnostic(
    *,
    dataset_dir: str | Path = "tests/data/hogenom_bench",
    out_dir: str | Path = "tests/data/hogenom_bench/diagnostics/pi_iteration_bound_diagnostic",
    cases: list[dict] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    preprocess_cache_dir: str | Path = "/tmp/gpurec_notebook_cache",
    max_passes: int = 300,
    reference_passes: int = 300,
    pass_counts: Iterable[int] = DEFAULT_PASS_COUNTS,
    tolerances: Iterable[float] = DEFAULT_TOLERANCES,
    e_tol: float = 1e-12,
    e_max_iters: int = 2000,
    max_wave_size: int = 8192,
    leaf_observation_init: bool = False,
    histogram_tol: float = 1e-8,
    ratio_floor: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cases is None:
        cases = DEFAULT_CASES
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    summaries = []
    pass_curves = []
    root_curves = []
    clade_bounds = []
    for case in cases:
        print(f"Running {case['case']} on {device} with rates={case['rates']}", flush=True)
        summary, pass_df, root_df, clade_df = run_case(
            case,
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            device=device,
            dtype=dtype,
            preprocess_cache_dir=preprocess_cache_dir,
            max_passes=max_passes,
            reference_passes=reference_passes,
            pass_counts=pass_counts,
            tolerances=tolerances,
            e_tol=e_tol,
            e_max_iters=e_max_iters,
            max_wave_size=max_wave_size,
            leaf_observation_init=leaf_observation_init,
            ratio_floor=ratio_floor,
        )
        summaries.append(summary)
        pass_curves.append(pass_df)
        root_curves.append(root_df)
        clade_bounds.append(clade_df)

    summary_df = pd.concat(summaries, ignore_index=True)
    pass_df = pd.concat(pass_curves, ignore_index=True)
    root_df = pd.concat(root_curves, ignore_index=True)
    clade_df = pd.concat(clade_bounds, ignore_index=True)

    summary_df.to_csv(out_dir / "case_summary.csv", index=False)
    pass_df.to_csv(out_dir / "pass_curves.csv", index=False)
    root_df.to_csv(out_dir / "root_curves.csv", index=False)
    clade_df.to_csv(out_dir / "clade_pass_bounds.csv", index=False)
    _plot_outputs(
        out_dir,
        summary_df,
        pass_df,
        root_df,
        clade_df,
        histogram_tol=histogram_tol,
        tolerances=tolerances,
    )
    _write_readme(
        out_dir,
        summary_df,
        max_passes=max_passes,
        reference_passes=reference_passes,
        e_tol=e_tol,
        leaf_observation_init=leaf_observation_init,
        ratio_floor=ratio_floor,
        tolerances=tolerances,
    )
    return summary_df, pass_df, root_df, clade_df
