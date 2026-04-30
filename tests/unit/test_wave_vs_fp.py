"""Test that Pi_wave_forward matches Pi_fixed_point on real data.

Runs on test_trees_100 (S=199, small-S path) and test_trees_1000
(S=1999, large-S path).  Both paths must produce log-likelihoods
within tolerance of the fixed-point baseline.
"""

import math
from pathlib import Path

import pytest
import torch

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.legacy import Pi_fixed_point
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.batching import build_wave_layout
from gpurec.core.scheduling import compute_clade_waves

_INV = 1.0 / math.log(2.0)
_ROOT = Path(__file__).resolve().parent.parent

D, L, T = 0.05, 0.05, 0.05
TOL = 1e-3  # convergence tolerance (both FP and wave)
LOGL_ATOL = 5e-2  # max acceptable log-likelihood mismatch (wave permutation changes FP ordering)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _preprocess_family(ext, sp_path, gene_path, device, dtype):
    """Preprocess one gene family and return all tensors on device."""
    raw = ext.preprocess(sp_path, [str(gene_path)])
    sr, cr = raw["species"], raw["ccp"]
    S = int(sr["S"])
    C = int(cr["C"])

    sh = {
        "S": S,
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
    ch = {
        "split_leftrights_sorted": cr["split_leftrights_sorted"].to(device=device),
        "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=dtype, device=device) * _INV,
        "seg_parent_ids": cr["seg_parent_ids"].to(device=device),
        "ptr_ge2": cr["ptr_ge2"].to(device=device),
        "num_segs_ge2": int(cr["num_segs_ge2"]),
        "num_segs_eq1": int(cr["num_segs_eq1"]),
        "end_rows_ge2": int(cr["end_rows_ge2"]),
        "C": C,
        "N_splits": int(cr["N_splits"]),
        "split_parents_sorted": cr["split_parents_sorted"].to(device=device),
    }
    if "phased_waves" in cr:
        ch["phased_waves"] = cr["phased_waves"]
        ch["phased_phases"] = cr["phased_phases"]

    li = raw["leaf_row_index"].long().to(device)
    lc = raw["leaf_col_index"].long().to(device)
    root_id = int(cr["root_clade_id"])

    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))
    tm = torch.log2(sh["Recipients_mat"])
    pS, pD, pL, tf, mt = extract_parameters(
        theta, tm, genewise=False, specieswise=False, pairwise=False
    )
    mv = mt.squeeze(-1) if mt.ndim == 2 else mt
    Eo = E_fixed_point(
        species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv, max_iters=2000,
        tolerance=TOL, warm_start_E=None, dtype=dtype, device=device,
    )
    return sh, ch, li, lc, root_id, pS, pD, pL, tf, mv, Eo


def _run_fp(sh, ch, li, lc, pS, pD, pL, tf, mv, Eo, device, dtype):
    return Pi_fixed_point(
        ccp_helpers=ch, species_helpers=sh,
        leaf_row_index=li, leaf_col_index=lc,
        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat_T=tf.T.contiguous(), max_transfer_mat=mv,
        max_iters=2000, tolerance=TOL,
        warm_start_Pi=None, device=device, dtype=dtype,
    )


def _run_wave(sh, ch, li, lc, root_id, pS, pD, pL, tf, mv, Eo, device, dtype):
    waves, phases = compute_clade_waves(ch)
    wave_layout = build_wave_layout(
        waves=waves, phases=phases,
        ccp_helpers=ch,
        leaf_row_index=li, leaf_col_index=lc,
        root_clade_ids=torch.tensor([root_id], dtype=torch.long, device=device),
        device=device, dtype=dtype,
    )
    return Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sh,
        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv,
        device=device, dtype=dtype,
        local_iters=1000, local_tolerance=TOL,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def _dataset_families(ds_name, n_families):
    """Yield (sp_path, gene_path) for n_families from a dataset."""
    data_dir = _ROOT / "data" / ds_name
    if not data_dir.exists():
        pytest.skip(f"{ds_name} not found")
    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))
    for gp in gene_paths[:n_families]:
        yield sp_path, gp


@pytest.fixture(scope="module")
def cpp_ext():
    return _load_extension()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("gi", range(5))
def test_wave_matches_fp_small_s(cpp_ext, gi):
    """Small-S path (S=199): per-wave fused Triton kernel."""
    families = list(_dataset_families("test_trees_100", 5))
    if gi >= len(families):
        pytest.skip("not enough families")
    sp_path, gp = families[gi]

    device = torch.device("cuda")
    dtype = torch.float32

    sh, ch, li, lc, root_id, pS, pD, pL, tf, mv, Eo = _preprocess_family(
        cpp_ext, sp_path, gp, device, dtype
    )
    S = sh["S"]
    assert S <= 256, f"Expected small S, got {S}"

    fp = _run_fp(sh, ch, li, lc, pS, pD, pL, tf, mv, Eo, device, dtype)
    wv = _run_wave(sh, ch, li, lc, root_id, pS, pD, pL, tf, mv, Eo, device, dtype)

    lL_fp = float(compute_log_likelihood(fp["Pi"], Eo["E"], root_id))
    lL_wv = float(compute_log_likelihood(wv["Pi"], Eo["E"], root_id))

    assert abs(lL_fp - lL_wv) < LOGL_ATOL, (
        f"logL mismatch: FP={lL_fp:.6f}, wave={lL_wv:.6f}, "
        f"diff={abs(lL_fp - lL_wv):.2e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("gi", range(5))
def test_wave_matches_fp_large_s(cpp_ext, gi):
    """Large-S path (S=1999): per-wave cuBLAS Pibar."""
    families = list(_dataset_families("test_trees_1000", 5))
    if gi >= len(families):
        pytest.skip("not enough families")
    sp_path, gp = families[gi]

    device = torch.device("cuda")
    dtype = torch.float32

    sh, ch, li, lc, root_id, pS, pD, pL, tf, mv, Eo = _preprocess_family(
        cpp_ext, sp_path, gp, device, dtype
    )
    S = sh["S"]
    assert S > 256, f"Expected large S, got {S}"

    fp = _run_fp(sh, ch, li, lc, pS, pD, pL, tf, mv, Eo, device, dtype)
    wv = _run_wave(sh, ch, li, lc, root_id, pS, pD, pL, tf, mv, Eo, device, dtype)

    lL_fp = float(compute_log_likelihood(fp["Pi"], Eo["E"], root_id))
    lL_wv = float(compute_log_likelihood(wv["Pi"], Eo["E"], root_id))

    assert abs(lL_fp - lL_wv) < LOGL_ATOL, (
        f"logL mismatch: FP={lL_fp:.6f}, wave={lL_wv:.6f}, "
        f"diff={abs(lL_fp - lL_wv):.2e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wave_faster_than_fp_large_s(cpp_ext):
    """Wave should be significantly faster than FP on large-S data."""
    families = list(_dataset_families("test_trees_1000", 1))
    if not families:
        pytest.skip("test_trees_1000 not found")
    sp_path, gp = families[0]

    device = torch.device("cuda")
    dtype = torch.float32

    sh, ch, li, lc, root_id, pS, pD, pL, tf, mv, Eo = _preprocess_family(
        cpp_ext, sp_path, gp, device, dtype
    )

    torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    _run_fp(sh, ch, li, lc, pS, pD, pL, tf, mv, Eo, device, dtype)
    torch.cuda.synchronize()
    t_fp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _run_wave(sh, ch, li, lc, root_id, pS, pD, pL, tf, mv, Eo, device, dtype)
    torch.cuda.synchronize()
    t_wv = time.perf_counter() - t0

    ratio = t_wv / t_fp
    assert ratio < 0.5, (
        f"Wave should be >2x faster than FP, got ratio={ratio:.2f} "
        f"(FP={t_fp*1000:.0f}ms, wave={t_wv*1000:.0f}ms)"
    )
