"""Test cross-family wave batching: Pi_wave_forward on collated families.

Validates that processing N families in a single batched Pi_wave_forward call
produces the same per-family log-likelihoods as processing each family
individually.
"""

import math
import time
from pathlib import Path

import pytest
import torch

from src.core.preprocess_cpp import _load_extension
from src.core.extract_parameters import extract_parameters
from src.core.likelihood import (
    E_fixed_point,
    Pi_fixed_point,
    Pi_wave_forward,
    compute_log_likelihood,
)
from src.core.scheduling import compute_clade_waves
from src.core.batching import collate_gene_families, collate_wave

_INV = 1.0 / math.log(2.0)
_ROOT = Path(__file__).resolve().parent.parent

D, L, T = 0.05, 0.05, 0.05
TOL = 1e-3
LOGL_ATOL = 1e-2


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _preprocess_family_raw(ext, sp_path, gene_path, device, dtype):
    """Preprocess one family, return raw dict (for collation) + helpers."""
    raw = ext.preprocess(sp_path, [str(gene_path)])
    sr, cr = raw["species"], raw["ccp"]
    S = int(sr["S"])
    C = int(cr["C"])

    ch = {
        "split_leftrights_sorted": cr["split_leftrights_sorted"],
        "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=dtype) * _INV,
        "seg_parent_ids": cr["seg_parent_ids"],
        "ptr_ge2": cr["ptr_ge2"],
        "num_segs_ge2": int(cr["num_segs_ge2"]),
        "num_segs_eq1": int(cr["num_segs_eq1"]),
        "end_rows_ge2": int(cr["end_rows_ge2"]),
        "C": C,
        "N_splits": int(cr["N_splits"]),
        "split_parents_sorted": cr["split_parents_sorted"],
    }
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
    """Compute E and parameters (shared across families for same species tree)."""
    sh = {
        "S": int(sr["S"]),
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
    theta = torch.log(torch.tensor([D, L, T], dtype=dtype, device=device))
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


def _run_per_family_wave(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype):
    """Run Pi_wave_forward on each family individually, return per-family logL."""
    logLs = []
    for item in batch_items:
        ch_dev = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in item["ccp"].items()}
        li = item["leaf_row_index"].to(device)
        lc = item["leaf_col_index"].to(device)
        root_id = item["root_clade_id"]

        waves, phases = compute_clade_waves(ch_dev)
        wv = Pi_wave_forward(
            waves=waves, ccp_helpers=ch_dev, species_helpers=sh,
            leaf_row_index=li, leaf_col_index=lc,
            E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
            log_pS=pS, log_pD=pD, log_pL=pL,
            transfer_mat=tf, max_transfer_mat=mv,
            device=device, dtype=dtype, phases=phases,
            local_iters=1000, local_tolerance=TOL,
        )
        lL = float(compute_log_likelihood(wv["Pi"], Eo["E"], root_id))
        logLs.append(lL)
    return logLs


def _run_batched_wave_chunk(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype):
    """Run Pi_wave_forward on a chunk of families batched together."""
    # 1. Collate families
    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    ccp = batched["ccp"]
    li = batched["leaf_row_index"]
    lc = batched["leaf_col_index"]
    root_ids = batched["root_clade_ids"]
    family_meta = batched["family_meta"]

    # 2. Compute per-family waves, then merge into cross-family waves
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

    # Cross-family phases: for each wave k, take max phase across families
    max_n_waves = max(len(p) for p in families_phases)
    cross_phases = []
    for k in range(max_n_waves):
        phase_k = 1
        for fp in families_phases:
            if k < len(fp):
                phase_k = max(phase_k, fp[k])
        cross_phases.append(phase_k)

    # 3. Run wave forward on the merged data
    wv = Pi_wave_forward(
        waves=cross_waves, ccp_helpers=ccp, species_helpers=sh,
        leaf_row_index=li, leaf_col_index=lc,
        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv,
        device=device, dtype=dtype, phases=cross_phases,
        local_iters=1000, local_tolerance=TOL,
    )

    # 4. Extract per-family log-likelihoods
    logL_vec = compute_log_likelihood(wv["Pi"], Eo["E"], root_ids)
    return [float(lL) for lL in logL_vec]


def _run_batched_wave(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
                      chunk_size=None):
    """Run batched wave forward, splitting into chunks if needed."""
    if chunk_size is None or len(batch_items) <= chunk_size:
        return _run_batched_wave_chunk(
            batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
        )

    logLs = []
    for start in range(0, len(batch_items), chunk_size):
        chunk = batch_items[start:start + chunk_size]
        logLs.extend(_run_batched_wave_chunk(
            chunk, sh, pS, pD, pL, tf, mv, Eo, device, dtype
        ))
    return logLs


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

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


@pytest.fixture(scope="module")
def cpp_ext():
    return _load_extension()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_fam", [2, 5, 10])
def test_batched_wave_matches_individual_small_s(cpp_ext, n_fam):
    """Cross-family wave batching matches per-family results (small S)."""
    device = torch.device("cuda")
    dtype = torch.float32

    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_100", n_fam, cpp_ext, device, dtype
    )

    logLs_individual = _run_per_family_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )
    logLs_batched = _run_batched_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )

    for i, (l_ind, l_bat) in enumerate(zip(logLs_individual, logLs_batched)):
        assert abs(l_ind - l_bat) < LOGL_ATOL, (
            f"Family {i}: individual={l_ind:.6f}, batched={l_bat:.6f}, "
            f"diff={abs(l_ind - l_bat):.2e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_fam", [2, 5, 10])
def test_batched_wave_matches_individual_large_s(cpp_ext, n_fam):
    """Cross-family wave batching matches per-family results (large S)."""
    device = torch.device("cuda")
    dtype = torch.float32

    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_1000", n_fam, cpp_ext, device, dtype
    )

    logLs_individual = _run_per_family_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )
    logLs_batched = _run_batched_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )

    for i, (l_ind, l_bat) in enumerate(zip(logLs_individual, logLs_batched)):
        assert abs(l_ind - l_bat) < LOGL_ATOL, (
            f"Family {i}: individual={l_ind:.6f}, batched={l_bat:.6f}, "
            f"diff={abs(l_ind - l_bat):.2e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_wave_100_families_large_s(cpp_ext):
    """Scale test: 100 families in chunks of 20, verify against individual."""
    device = torch.device("cuda")
    dtype = torch.float32

    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_1000", 100, cpp_ext, device, dtype
    )

    logLs_batched = _run_batched_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
        chunk_size=20,
    )

    # Sanity: all log-likelihoods should be finite
    # (values are in log2 space, so they can be positive)
    for i, lL in enumerate(logLs_batched):
        assert math.isfinite(lL), f"Family {i}: logL={lL} is not finite"

    # Spot-check first 5 against individual
    logLs_individual = _run_per_family_wave(
        batch_items[:5], sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )
    for i in range(5):
        assert abs(logLs_individual[i] - logLs_batched[i]) < LOGL_ATOL, (
            f"Family {i}: individual={logLs_individual[i]:.6f}, "
            f"batched={logLs_batched[i]:.6f}, "
            f"diff={abs(logLs_individual[i] - logLs_batched[i]):.2e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_wave_timing_large_s(cpp_ext):
    """Benchmark: batched wave (chunks of 20) vs sequential per-family wave."""
    device = torch.device("cuda")
    dtype = torch.float32
    n_fam = 20

    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_1000", n_fam, cpp_ext, device, dtype
    )

    # Warmup
    _run_batched_wave(batch_items[:2], sh, pS, pD, pL, tf, mv, Eo, device, dtype)
    _run_per_family_wave(batch_items[:2], sh, pS, pD, pL, tf, mv, Eo, device, dtype)

    # Sequential per-family wave
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logLs_seq = _run_per_family_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )
    torch.cuda.synchronize()
    t_seq = time.perf_counter() - t0

    # Batched wave (single chunk)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logLs_bat = _run_batched_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )
    torch.cuda.synchronize()
    t_bat = time.perf_counter() - t0

    # Report
    print(f"\n  {n_fam} families, S=1999:")
    print(f"    Sequential per-family wave: {t_seq*1000:.0f}ms "
          f"({t_seq/n_fam*1000:.0f}ms/family)")
    print(f"    Batched wave:               {t_bat*1000:.0f}ms "
          f"({t_bat/n_fam*1000:.0f}ms/family)")
    print(f"    Speedup: {t_seq/t_bat:.1f}x")

    # Verify correctness
    for i in range(n_fam):
        assert abs(logLs_seq[i] - logLs_bat[i]) < LOGL_ATOL, (
            f"Family {i}: seq={logLs_seq[i]:.6f}, bat={logLs_bat[i]:.6f}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_wave_vs_batched_fp_large_s(cpp_ext):
    """Batched wave vs batched FP (both use collated data)."""
    device = torch.device("cuda")
    dtype = torch.float32
    n_fam = 10

    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_1000", n_fam, cpp_ext, device, dtype
    )

    # Batched wave
    logLs_wv = _run_batched_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )

    # Batched FP (via collate + Pi_fixed_point)
    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    ccp = batched["ccp"]
    li = batched["leaf_row_index"]
    lc = batched["leaf_col_index"]
    root_ids = batched["root_clade_ids"]

    fp_out = Pi_fixed_point(
        ccp_helpers=ccp, species_helpers=sh,
        leaf_row_index=li, leaf_col_index=lc,
        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat_T=tf.T.contiguous(), max_transfer_mat=mv,
        max_iters=2000, tolerance=TOL,
        warm_start_Pi=None, device=device, dtype=dtype,
    )
    logL_fp_vec = compute_log_likelihood(fp_out["Pi"], Eo["E"], root_ids)
    logLs_fp = [float(x) for x in logL_fp_vec]

    for i in range(n_fam):
        assert abs(logLs_fp[i] - logLs_wv[i]) < LOGL_ATOL, (
            f"Family {i}: FP={logLs_fp[i]:.6f}, wave={logLs_wv[i]:.6f}, "
            f"diff={abs(logLs_fp[i] - logLs_wv[i]):.2e}"
        )
