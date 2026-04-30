"""Parametrized finite-difference gradient tests for all mode combinations.

Computes central FD dL/dtheta for all 6 valid (genewise, specieswise, pibar_mode)
combos.  Step 1: verify FD is finite and nonzero (forward pass works for each
mode).  Step 2 (added later): compare FD with analytic implicit gradient.

Valid combinations
------------------
  (gw=F, sw=F, uniform), (gw=F, sw=F, dense),
  (gw=F, sw=T, uniform), (gw=F, sw=T, dense),
  (gw=T, sw=F, uniform), (gw=T, sw=T, uniform)
"""

import math
from pathlib import Path

import pytest
import torch

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters, extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import (
    collate_gene_families,
    collate_wave,
    build_wave_layout,
)
from gpurec.optimization.implicit_grad import (
    implicit_grad_loglik_vjp_wave,
    implicit_grad_loglik_vjp_wave_genewise,
)

_INV = 1.0 / math.log(2.0)
_ROOT = Path(__file__).resolve().parent.parent
D, L, T = 0.05, 0.05, 0.05

ALL_MODES = [
    (False, False, "uniform"),
    (False, False, "dense"),
    (False, True, "uniform"),
    (False, True, "dense"),
    (True, False, "uniform"),
    (True, True, "uniform"),
]


# ------------------------------------------------------------------
# Shared data loading
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def tree_data():
    """Load test_trees_20 once: species helpers, 3 gene families, wave layout."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    ext = _load_extension()
    data_dir = _ROOT / "data" / "test_trees_20"
    if not data_dir.exists():
        pytest.skip("test_trees_20 not found")
    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))[:3]
    if len(gene_paths) < 3:
        pytest.skip(f"Need 3 families, found {len(gene_paths)}")

    families = []
    batch_items = []
    sr = None
    for gp in gene_paths:
        raw = ext.preprocess(sp_path, [str(gp)])
        if sr is None:
            sr = raw["species"]
        cr = raw["ccp"]
        ch = {
            "split_leftrights_sorted": cr["split_leftrights_sorted"],
            "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=dtype) * _INV,
            "seg_parent_ids": cr["seg_parent_ids"],
            "ptr_ge2": cr["ptr_ge2"],
            "num_segs_ge2": int(cr["num_segs_ge2"]),
            "num_segs_eq1": int(cr["num_segs_eq1"]),
            "end_rows_ge2": int(cr["end_rows_ge2"]),
            "C": int(cr["C"]),
            "N_splits": int(cr["N_splits"]),
        }
        if "split_parents_sorted" in cr:
            ch["split_parents_sorted"] = cr["split_parents_sorted"]
        item = {
            "ccp": ch,
            "leaf_row_index": raw["leaf_row_index"].long(),
            "leaf_col_index": raw["leaf_col_index"].long(),
            "root_clade_id": int(cr["root_clade_id"]),
        }
        batch_items.append(item)
        families.append({
            "ccp_helpers": ch,
            "leaf_row_index": raw["leaf_row_index"].long(),
            "leaf_col_index": raw["leaf_col_index"].long(),
            "root_clade_id": int(cr["root_clade_id"]),
        })

    sh = {
        "S": int(sr["S"]),
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
    if "ancestors_dense" in sr:
        sh["ancestors_dense"] = sr["ancestors_dense"].to(dtype=dtype, device=device)

    S = sh["S"]
    G = len(families)
    tm_unnorm = torch.log2(sh["Recipients_mat"]).to(device=device, dtype=dtype)
    unnorm_row_max = tm_unnorm.max(dim=-1).values

    # Build batched wave layout (used by non-genewise forward)
    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    root_clade_ids = batched["root_clade_ids"]

    families_waves, families_phases = [], []
    for bi in batch_items:
        w, p = compute_clade_waves(bi["ccp"])
        families_waves.append(w)
        families_phases.append(p)
    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(families_waves, offsets)
    max_n = max(len(p) for p in families_phases)
    cross_phases = [
        max(fp[k] if k < len(fp) else 1 for fp in families_phases)
        for k in range(max_n)
    ]

    wave_layout = build_wave_layout(
        waves=cross_waves,
        phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=root_clade_ids,
        device=device,
        dtype=dtype,
    )

    ancestors_T = None
    if "ancestors_dense" in sh:
        ancestors_T = sh["ancestors_dense"].T.to_sparse_coo()

    return {
        "families": families,
        "batch_items": batch_items,
        "species_helpers": sh,
        "wave_layout": wave_layout,
        "root_clade_ids": root_clade_ids,
        "tm_unnorm": tm_unnorm,
        "unnorm_row_max": unnorm_row_max,
        "ancestors_T": ancestors_T,
        "device": device,
        "dtype": dtype,
        "S": S,
        "G": G,
    }


# ------------------------------------------------------------------
# Forward-pass helpers
# ------------------------------------------------------------------

def _forward_non_genewise(theta, d, specieswise, pibar_mode):
    """theta → params → E → Pi → logL  (shared theta across families)."""
    device, dtype = d["device"], d["dtype"]
    sh = d["species_helpers"]

    if pibar_mode == "uniform":
        log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
            theta, d["unnorm_row_max"], specieswise=specieswise,
        )
        ancestors_T = d["ancestors_T"]
    else:
        log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters(
            theta, d["tm_unnorm"], genewise=False, specieswise=specieswise, pairwise=False,
        )
        if mt.ndim == 2:
            mt = mt.squeeze(-1)
        ancestors_T = None

    E_out = E_fixed_point(
        species_helpers=sh,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device,
        pibar_mode=pibar_mode, ancestors_T=ancestors_T,
    )

    Pi_out = Pi_wave_forward(
        wave_layout=d["wave_layout"], species_helpers=sh,
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode=pibar_mode,
    )

    logL = compute_log_likelihood(Pi_out["Pi"], E_out["E"], d["root_clade_ids"])
    return logL.sum().item()


def _forward_genewise(theta, d, specieswise):
    """theta [G, 3] or [G, S, 3] → per-family forward → sum logL."""
    device, dtype = d["device"], d["dtype"]
    sh = d["species_helpers"]
    families = d["families"]
    pibar_mode = "uniform"

    log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(
        theta, d["unnorm_row_max"], specieswise=specieswise, genewise=True,
    )
    E_out = E_fixed_point(
        species_helpers=sh,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device,
        pibar_mode=pibar_mode, ancestors_T=d["ancestors_T"],
    )

    total = 0.0
    G = theta.shape[0]
    for g in range(G):
        fam = families[g]
        single = {
            "ccp": fam["ccp_helpers"],
            "leaf_row_index": fam["leaf_row_index"],
            "leaf_col_index": fam["leaf_col_index"],
            "root_clade_id": int(fam["root_clade_id"]),
        }
        sb = collate_gene_families([single], dtype=dtype, device=device)
        waves_g, phases_g = compute_clade_waves(fam["ccp_helpers"])
        wl = build_wave_layout(
            waves=waves_g, phases=phases_g,
            ccp_helpers=sb["ccp"],
            leaf_row_index=sb["leaf_row_index"],
            leaf_col_index=sb["leaf_col_index"],
            root_clade_ids=sb["root_clade_ids"],
            device=device, dtype=dtype,
        )
        Pi_out = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"][g], Ebar=E_out["E_bar"][g],
            E_s1=E_out["E_s1"][g], E_s2=E_out["E_s2"][g],
            log_pS=log_pS[g], log_pD=log_pD[g], log_pL=log_pL[g],
            transfer_mat=None, max_transfer_mat=mt[g],
            device=device, dtype=dtype, pibar_mode=pibar_mode,
        )
        logL = compute_log_likelihood(
            Pi_out["Pi"], E_out["E"][g], sb["root_clade_ids"],
        )
        total += logL.sum().item()
    return total


def _forward(theta, d, genewise, specieswise, pibar_mode):
    """Dispatch to the right forward based on mode."""
    if genewise:
        return _forward_genewise(theta, d, specieswise)
    return _forward_non_genewise(theta, d, specieswise, pibar_mode)


def _forward_per_family(theta, d, specieswise, pibar_mode):
    """Per-family forward, summed.  Matches the analytic gradient path exactly."""
    device, dtype = d["device"], d["dtype"]
    sh = d["species_helpers"]

    if pibar_mode == "uniform":
        log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
            theta, d["unnorm_row_max"], specieswise=specieswise,
        )
        ancestors_T = d["ancestors_T"]
    else:
        log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters(
            theta, d["tm_unnorm"], genewise=False, specieswise=specieswise, pairwise=False,
        )
        if mt.ndim == 2:
            mt = mt.squeeze(-1)
        ancestors_T = None

    E_out = E_fixed_point(
        species_helpers=sh,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device,
        pibar_mode=pibar_mode, ancestors_T=ancestors_T,
    )

    total = 0.0
    for bi in d["batch_items"]:
        sb = collate_gene_families([bi], dtype=dtype, device=device)
        w, p = compute_clade_waves(bi["ccp"])
        cw = collate_wave([w], [0])
        wl = build_wave_layout(
            waves=cw, phases=p, ccp_helpers=sb["ccp"],
            leaf_row_index=sb["leaf_row_index"], leaf_col_index=sb["leaf_col_index"],
            root_clade_ids=sb["root_clade_ids"], device=device, dtype=dtype,
        )
        po = Pi_wave_forward(
            wave_layout=wl, species_helpers=sh,
            E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=transfer_mat, max_transfer_mat=mt,
            device=device, dtype=dtype, pibar_mode=pibar_mode,
        )
        total += compute_log_likelihood(po["Pi"], E_out["E"], sb["root_clade_ids"]).sum().item()
    return total


# ------------------------------------------------------------------
# Theta construction and index sampling
# ------------------------------------------------------------------

def _make_theta(S, G, genewise, specieswise, device, dtype):
    """Create theta at (D, L, T) = (0.05, 0.05, 0.05) in log2 space."""
    base = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))
    if genewise and specieswise:
        return base.view(1, 1, 3).expand(G, S, -1).contiguous()
    if genewise:
        return base.unsqueeze(0).expand(G, -1).contiguous()
    if specieswise:
        return base.unsqueeze(0).expand(S, -1).contiguous()
    return base.clone()


def _sample_indices(theta, max_per_dim=5):
    """Pick representative indices for FD perturbation.

    Returns list of tuples that can index into theta.
    """
    if theta.ndim == 1:
        # [3] — all components
        return [(i,) for i in range(theta.shape[0])]

    if theta.ndim == 2:
        # [N, 3] (specieswise or genewise)
        N = theta.shape[0]
        rows = sorted(set([0, N - 1] + torch.randint(1, max(2, N - 1), (min(max_per_dim, N) - 2,)).tolist()))
        return [(r, c) for r in rows for c in range(3)]

    if theta.ndim == 3:
        # [G, S, 3] (genewise + specieswise)
        G, S, _ = theta.shape
        gs = list(range(G))
        ss = sorted(set([0, S - 1] + torch.randint(1, max(2, S - 1), (min(max_per_dim, S) - 2,)).tolist()))
        return [(g, s, c) for g in gs for s in ss for c in range(3)]

    raise ValueError(f"Unexpected theta ndim={theta.ndim}")


# ------------------------------------------------------------------
# FD test
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("genewise,specieswise,pibar_mode", ALL_MODES)
def test_fd_gradient_finite(tree_data, genewise, specieswise, pibar_mode):
    """Central FD gradient is finite and nonzero for every valid mode."""
    d = tree_data
    S, G = d["S"], d["G"]
    device, dtype = d["device"], d["dtype"]
    label = f"gw={genewise}, sw={specieswise}, mode={pibar_mode}"

    theta = _make_theta(S, G, genewise, specieswise, device, dtype)

    # Base forward — must produce finite logL
    logL_base = _forward(theta, d, genewise, specieswise, pibar_mode)
    assert math.isfinite(logL_base), f"[{label}] base logL not finite: {logL_base}"

    # Central FD
    eps = 1e-4
    torch.manual_seed(42)
    indices = _sample_indices(theta)
    fd_grads = {}

    for idx in indices:
        theta_p = theta.clone()
        theta_p[idx] += eps
        logL_p = _forward(theta_p, d, genewise, specieswise, pibar_mode)

        theta_m = theta.clone()
        theta_m[idx] -= eps
        logL_m = _forward(theta_m, d, genewise, specieswise, pibar_mode)

        fd = (logL_p - logL_m) / (2 * eps)
        fd_grads[idx] = fd
        assert math.isfinite(fd), (
            f"[{label}] FD[{idx}] not finite: logL_p={logL_p}, logL_m={logL_m}"
        )

    max_abs = max(abs(v) for v in fd_grads.values())

    # Print summary
    print(f"\n  [{label}] base logL = {logL_base:.8f}")
    print(f"  [{label}] {len(indices)} components checked, max |FD| = {max_abs:.4e}")
    for idx, fd in list(fd_grads.items())[:6]:
        print(f"    theta{list(idx)}: FD = {fd:.8e}")

    assert max_abs > 0, f"[{label}] all FD gradients are zero"


# ------------------------------------------------------------------
# Full-forward helpers (return intermediates for analytic gradient)
# ------------------------------------------------------------------

def _full_forward_non_genewise(theta, d, specieswise, pibar_mode):
    """Forward pass returning all intermediates needed by implicit_grad."""
    device, dtype = d["device"], d["dtype"]
    sh = d["species_helpers"]

    if pibar_mode == "uniform":
        log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
            theta, d["unnorm_row_max"], specieswise=specieswise,
        )
        ancestors_T = d["ancestors_T"]
    else:
        log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters(
            theta, d["tm_unnorm"], genewise=False, specieswise=specieswise, pairwise=False,
        )
        if mt.ndim == 2:
            mt = mt.squeeze(-1)
        ancestors_T = None

    E_out = E_fixed_point(
        species_helpers=sh,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device,
        pibar_mode=pibar_mode, ancestors_T=ancestors_T,
    )

    Pi_out = Pi_wave_forward(
        wave_layout=d["wave_layout"], species_helpers=sh,
        E=E_out["E"], Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode=pibar_mode,
    )

    logL = compute_log_likelihood(Pi_out["Pi"], E_out["E"], d["root_clade_ids"])
    return {
        "logL": logL.sum().item(),
        "Pi_out": Pi_out, "E_out": E_out,
        "log_pS": log_pS, "log_pD": log_pD, "log_pL": log_pL,
        "transfer_mat": transfer_mat, "max_transfer_mat": mt,
        "ancestors_T": ancestors_T,
    }


def _full_forward_genewise(theta, d, specieswise):
    """Forward pass for genewise, returning E quantities and params for implicit_grad."""
    device, dtype = d["device"], d["dtype"]
    sh = d["species_helpers"]
    pibar_mode = "uniform"

    log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(
        theta, d["unnorm_row_max"], specieswise=specieswise, genewise=True,
    )
    E_out = E_fixed_point(
        species_helpers=sh,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=None, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-10, warm_start_E=None,
        dtype=dtype, device=device,
        pibar_mode=pibar_mode, ancestors_T=d["ancestors_T"],
    )
    return {
        "E_out": E_out,
        "log_pS": log_pS, "log_pD": log_pD, "log_pL": log_pL,
        "max_transfer_mat": mt,
    }


# ------------------------------------------------------------------
# Analytic gradient helpers
# ------------------------------------------------------------------

def _analytic_gradient_non_genewise(theta, d, specieswise, pibar_mode, fwd):
    """Compute dL/dtheta via per-family implicit_grad_loglik_vjp_wave, summed.

    The batched multi-family backward has known issues, so we compute
    per-family gradients and sum them (which is mathematically identical
    since theta and E are shared across families).
    """
    device, dtype = d["device"], d["dtype"]

    grad_total = torch.zeros_like(theta)
    for bi in d["batch_items"]:
        sb = collate_gene_families([bi], dtype=dtype, device=device)
        w, p = compute_clade_waves(bi["ccp"])
        cw = collate_wave([w], [0])
        wl = build_wave_layout(
            waves=cw, phases=p,
            ccp_helpers=sb["ccp"],
            leaf_row_index=sb["leaf_row_index"],
            leaf_col_index=sb["leaf_col_index"],
            root_clade_ids=sb["root_clade_ids"],
            device=device, dtype=dtype,
        )
        po = Pi_wave_forward(
            wave_layout=wl, species_helpers=d["species_helpers"],
            E=fwd["E_out"]["E"], Ebar=fwd["E_out"]["E_bar"],
            E_s1=fwd["E_out"]["E_s1"], E_s2=fwd["E_out"]["E_s2"],
            log_pS=fwd["log_pS"], log_pD=fwd["log_pD"], log_pL=fwd["log_pL"],
            transfer_mat=fwd["transfer_mat"], max_transfer_mat=fwd["max_transfer_mat"],
            device=device, dtype=dtype, pibar_mode=pibar_mode,
        )

        kwargs = dict(
            wave_layout=wl,
            species_helpers=d["species_helpers"],
            Pi_star_wave=po["Pi_wave_ordered"],
            Pibar_star_wave=po["Pibar_wave_ordered"],
            E_star=fwd["E_out"]["E"],
            E_s1=fwd["E_out"]["E_s1"],
            E_s2=fwd["E_out"]["E_s2"],
            Ebar=fwd["E_out"]["E_bar"],
            log_pS=fwd["log_pS"], log_pD=fwd["log_pD"], log_pL=fwd["log_pL"],
            max_transfer_mat=fwd["max_transfer_mat"],
            root_clade_ids_perm=wl["root_clade_ids"],
            theta=theta,
            unnorm_row_max=d["unnorm_row_max"],
            specieswise=specieswise,
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
            cg_tol=1e-10, cg_maxiter=1000,
            pibar_mode=pibar_mode,
        )
        if pibar_mode == "uniform":
            kwargs["ancestors_T"] = fwd["ancestors_T"]
        else:
            kwargs["transfer_mat"] = fwd["transfer_mat"]
            kwargs["transfer_mat_unnormalized"] = d["tm_unnorm"]

        g, _ = implicit_grad_loglik_vjp_wave(**kwargs)
        grad_total = grad_total + g

    return grad_total


def _analytic_gradient_genewise(theta, d, specieswise, fwd):
    """Compute dL/dtheta via implicit_grad_loglik_vjp_wave_genewise."""
    device, dtype = d["device"], d["dtype"]

    grad_theta, stats = implicit_grad_loglik_vjp_wave_genewise(
        d["families"], d["species_helpers"],
        E_all=fwd["E_out"]["E"],
        E_s1_all=fwd["E_out"]["E_s1"],
        E_s2_all=fwd["E_out"]["E_s2"],
        Ebar_all=fwd["E_out"]["E_bar"],
        log_pS_all=fwd["log_pS"], log_pD_all=fwd["log_pD"], log_pL_all=fwd["log_pL"],
        mt_all=fwd["max_transfer_mat"],
        theta_stack=theta,
        unnorm_row_max=d["unnorm_row_max"],
        specieswise=specieswise,
        device=device, dtype=dtype,
        pibar_mode="uniform",
        neumann_terms=4, use_pruning=False,
        cg_tol=1e-10, cg_maxiter=1000,
        ancestors_T=d["ancestors_T"],
    )
    return grad_theta


# ------------------------------------------------------------------
# Analytic vs FD test
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("genewise,specieswise,pibar_mode", ALL_MODES)
def test_analytic_matches_fd(tree_data, genewise, specieswise, pibar_mode):
    """Analytic implicit gradient matches central FD for all valid mode combos."""
    d = tree_data
    S, G = d["S"], d["G"]
    device, dtype = d["device"], d["dtype"]
    label = f"gw={genewise}, sw={specieswise}, mode={pibar_mode}"

    theta = _make_theta(S, G, genewise, specieswise, device, dtype)

    # Full forward + analytic gradient
    if genewise:
        fwd = _full_forward_genewise(theta, d, specieswise)
        analytic = _analytic_gradient_genewise(theta, d, specieswise, fwd)
    else:
        fwd = _full_forward_non_genewise(theta, d, specieswise, pibar_mode)
        analytic = _analytic_gradient_non_genewise(theta, d, specieswise, pibar_mode, fwd)

    assert analytic.shape == theta.shape, (
        f"[{label}] grad shape {analytic.shape} != theta shape {theta.shape}"
    )
    assert torch.isfinite(analytic).all(), f"[{label}] analytic gradient has non-finite values"

    # Central FD comparison on sampled indices.
    # Use per-family forward for non-genewise to match the analytic gradient path
    # (the batched wave layout converges to slightly different fixed points).
    if genewise:
        fd_forward = lambda t: _forward_genewise(t, d, specieswise)
    else:
        fd_forward = lambda t: _forward_per_family(t, d, specieswise, pibar_mode)

    # Dense mode is near-exact (<1e-6); uniform mode uses Neumann series for
    # the self-loop VJP which introduces ~1-5% approximation error at small S.
    tol = 1e-3 if pibar_mode == "dense" else 0.15

    eps = 1e-4
    torch.manual_seed(42)
    indices = _sample_indices(theta)
    max_rel_err = 0.0
    n_above_half = 0

    for idx in indices:
        theta_p = theta.clone()
        theta_p[idx] += eps
        logL_p = fd_forward(theta_p)

        theta_m = theta.clone()
        theta_m[idx] -= eps
        logL_m = fd_forward(theta_m)

        fd = (logL_p - logL_m) / (2 * eps)
        ana = analytic[idx].item()
        # Use relative error with a floor to avoid blow-up on near-zero gradients
        rel_err = abs(ana - fd) / max(abs(fd), 1e-6)
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err > tol * 0.5:
            n_above_half += 1
            print(f"  [{label}] theta{list(idx)}: FD={fd:.8e}, analytic={ana:.8e}, "
                  f"rel_err={rel_err:.4e}")
        assert rel_err < tol, (
            f"[{label}] theta{list(idx)} gradient rel error {rel_err:.4e} > {tol}: "
            f"FD={fd:.8e}, analytic={ana:.8e}"
        )

    print(f"\n  [{label}] {len(indices)} components, max rel_err = {max_rel_err:.4e} (tol={tol})")
