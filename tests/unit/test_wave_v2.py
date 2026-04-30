"""Test Pi_wave_forward (wave-ordered layout) against FP.

Validates that the zero-copy wave-ordered layout produces the same
per-family log-likelihoods as fixed-point iteration.
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
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import (
    collate_gene_families,
    collate_wave,
    build_wave_layout,
)
from gpurec.core.model import GeneDataset

_INV = 1.0 / math.log(2.0)
_ROOT = Path(__file__).resolve().parent.parent

D, L, T = 0.05, 0.05, 0.05
TOL = 1e-3
LOGL_ATOL = 5e-2  # wave permutation changes FP ordering → slightly larger diffs


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _preprocess_family_raw(ext, sp_path, gene_path, device, dtype):
    raw = ext.preprocess(sp_path, [str(gene_path)])
    sr, cr = raw["species"], raw["ccp"]
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
    if "phased_waves" in cr:
        ch["phased_waves"] = cr["phased_waves"]
        ch["phased_phases"] = cr["phased_phases"]
    batch_item = {
        "ccp": ch,
        "leaf_row_index": raw["leaf_row_index"].long(),
        "leaf_col_index": raw["leaf_col_index"].long(),
        "root_clade_id": int(cr["root_clade_id"]),
    }
    return batch_item, sr


def _compute_shared_params(sr, device, dtype):
    sh = {
        "S": int(sr["S"]),
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
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
    return sh, pS, pD, pL, tf, mv, Eo


def _load_families(ds_name, n, ext, device, dtype):
    data_dir = _ROOT / "data" / ds_name
    if not data_dir.exists():
        pytest.skip(f"{ds_name} not found")
    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))[:n]
    batch_items = []
    sr = None
    for gp in gene_paths:
        item, sr = _preprocess_family_raw(ext, sp_path, gp, device, dtype)
        batch_items.append(item)
    sh, pS, pD, pL, tf, mv, Eo = _compute_shared_params(sr, device, dtype)
    return batch_items, sh, pS, pD, pL, tf, mv, Eo


def _run_batched_wave(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype):
    """Run wave-ordered layout wave forward on batched families."""
    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    ccp = batched["ccp"]
    li = batched["leaf_row_index"]
    lc = batched["leaf_col_index"]
    root_ids = batched["root_clade_ids"]
    family_meta = batched["family_meta"]

    # Use fast per-family scheduling + collate_wave
    families_waves = []
    families_phases = []
    for item in batch_items:
        ch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in item["ccp"].items()}
        waves_i, phases_i = compute_clade_waves(ch_dev)
        families_waves.append(waves_i)
        families_phases.append(phases_i)

    offsets = [m["clade_offset"] for m in family_meta]
    cross_waves = collate_wave(families_waves, offsets)
    max_n_waves = max(len(p) for p in families_phases)
    cross_phases = []
    for k in range(max_n_waves):
        phase_k = 1
        for fp in families_phases:
            if k < len(fp):
                phase_k = max(phase_k, fp[k])
        cross_phases.append(phase_k)

    wave_layout = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=ccp,
        leaf_row_index=li, leaf_col_index=lc,
        root_clade_ids=root_ids,
        device=device, dtype=dtype,
    )

    wv = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sh,
        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv,
        device=device, dtype=dtype,
        local_iters=1000, local_tolerance=TOL,
    )
    logL_vec = compute_log_likelihood(wv["Pi"], Eo["E"], root_ids)
    return [float(lL) for lL in logL_vec]


def _run_per_family_fp(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype):
    logLs = []
    for item in batch_items:
        ch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in item["ccp"].items()}
        li = item["leaf_row_index"].to(device)
        lc = item["leaf_col_index"].to(device)
        root_id = item["root_clade_id"]
        fp_out = Pi_fixed_point(
            ccp_helpers=ch_dev, species_helpers=sh,
            leaf_row_index=li, leaf_col_index=lc,
            E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
            log_pS=pS, log_pD=pD, log_pL=pL,
            transfer_mat_T=tf.T.contiguous(), max_transfer_mat=mv,
            max_iters=2000, tolerance=TOL,
            warm_start_Pi=None, device=device, dtype=dtype,
        )
        lL = float(compute_log_likelihood(fp_out["Pi"], Eo["E"], root_id))
        logLs.append(lL)
    return logLs


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def cpp_ext():
    return _load_extension()


# ------------------------------------------------------------------
# Tests: wave matches FP
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_fam", [2, 5, 10])
def test_wave_matches_fp_small_s(cpp_ext, n_fam):
    """Wave-ordered layout matches per-family FP (small S)."""
    device = torch.device("cuda")
    dtype = torch.float32
    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_100", n_fam, cpp_ext, device, dtype
    )
    logLs_fp = _run_per_family_fp(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype)
    logLs_wave = _run_batched_wave(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype)

    for i, (l_fp, l_wave) in enumerate(zip(logLs_fp, logLs_wave)):
        assert abs(l_fp - l_wave) < LOGL_ATOL, (
            f"Family {i}: FP={l_fp:.6f}, wave={l_wave:.6f}, diff={abs(l_fp - l_wave):.2e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_fam", [2, 5, 10])
def test_wave_matches_fp_large_s(cpp_ext, n_fam):
    """Wave-ordered layout matches per-family FP (large S)."""
    device = torch.device("cuda")
    dtype = torch.float32
    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_1000", n_fam, cpp_ext, device, dtype
    )
    logLs_fp = _run_per_family_fp(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype)
    logLs_wave = _run_batched_wave(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype)

    for i, (l_fp, l_wave) in enumerate(zip(logLs_fp, logLs_wave)):
        assert abs(l_fp - l_wave) < LOGL_ATOL, (
            f"Family {i}: FP={l_fp:.6f}, wave={l_wave:.6f}, diff={abs(l_fp - l_wave):.2e}"
        )


# ------------------------------------------------------------------
# Tests: model API
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_model_api_wave_matches_fp(cpp_ext):
    """GeneDataset.compute_likelihood_batch wave matches per-family FP."""
    data_dir = _ROOT / "data" / "test_trees_1000"
    if not data_dir.exists():
        pytest.skip("test_trees_1000 not found")

    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:10]]

    ds = GeneDataset(sp, genes, genewise=False, specieswise=False, pairwise=False,
                     dtype=torch.float32, device=torch.device("cuda"))

    logLs_wave = ds.compute_likelihood_batch(tol_Pi=TOL)
    logLs_seq = [ds.compute_likelihood(i, tol_Pi=TOL)["log_likelihood"]
                 for i in range(len(genes))]

    for i in range(len(genes)):
        assert math.isfinite(logLs_wave[i]), f"Wave family {i}: logL={logLs_wave[i]}"
        # Use relative tolerance: absolute diffs scale with logL magnitude
        tol = max(LOGL_ATOL, abs(logLs_seq[i]) * 5e-5)
        assert abs(logLs_wave[i] - logLs_seq[i]) < tol, (
            f"Family {i}: wave={logLs_wave[i]:.4f}, seq={logLs_seq[i]:.4f}, "
            f"diff={abs(logLs_wave[i] - logLs_seq[i]):.2e}"
        )


