"""Test cross-family wave batching: Pi_wave_forward on collated families.

Validates that processing N families in a single batched Pi_wave_forward call
produces the same per-family log-likelihoods as processing each family
individually, and matches batched fixed-point iteration.
"""

import math
import time
from pathlib import Path

import pytest
import torch

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.legacy import Pi_fixed_point
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
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
    """Preprocess one family, return raw dict (for collation) + helpers."""
    raw = ext.preprocess(sp_path, [str(gene_path)])
    sr, cr = raw["species"], raw["ccp"]
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
    """Compute E and parameters (shared across families for same species tree)."""
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
        wave_layout = build_wave_layout(
            waves=waves, phases=phases,
            ccp_helpers=ch_dev,
            leaf_row_index=li, leaf_col_index=lc,
            root_clade_ids=torch.tensor([root_id], dtype=torch.long, device=device),
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
        lL = float(compute_log_likelihood(wv["Pi"], Eo["E"], root_id))
        logLs.append(lL)
    return logLs


def _run_batched_wave_chunk(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype):
    """Run Pi_wave_forward on a chunk of families batched together."""
    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    ccp = batched["ccp"]
    li = batched["leaf_row_index"]
    lc = batched["leaf_col_index"]
    root_ids = batched["root_clade_ids"]
    family_meta = batched["family_meta"]

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


def _run_per_family_fp(batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype):
    """Run Pi_fixed_point on each family individually, return per-family logL."""
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
def test_batched_wave_vs_sequential_fp_large_s(cpp_ext):
    """Batched wave vs sequential per-family FP (large S)."""
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

    # Sequential per-family FP
    logLs_fp = _run_per_family_fp(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )

    for i in range(n_fam):
        assert abs(logLs_fp[i] - logLs_wv[i]) < LOGL_ATOL, (
            f"Family {i}: FP={logLs_fp[i]:.6f}, wave={logLs_wv[i]:.6f}, "
            f"diff={abs(logLs_fp[i] - logLs_wv[i]):.2e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_wave_vs_fp_100_families_large_s(cpp_ext):
    """Large-scale: 100 families (S=1999), batched wave vs sequential per-family FP.

    Batched wave processes all families in cross-family waves (chunks of 20).
    Spot-checks 10 families against per-family FP, validates all 100 are finite.
    """
    device = torch.device("cuda")
    dtype = torch.float32
    n_fam = 100
    n_spot = 10  # spot-check this many against FP

    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_1000", n_fam, cpp_ext, device, dtype
    )

    # Batched wave (chunks of 20)
    logLs_wv = _run_batched_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
        chunk_size=20,
    )

    # All should be finite
    for i in range(n_fam):
        assert math.isfinite(logLs_wv[i]), f"Wave family {i}: logL={logLs_wv[i]}"

    # Spot-check first n_spot families against sequential per-family FP
    logLs_fp = _run_per_family_fp(
        batch_items[:n_spot], sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )

    diffs = [abs(logLs_fp[i] - logLs_wv[i]) for i in range(n_spot)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    worst_idx = diffs.index(max_diff)

    print(f"\n  {n_fam} families batched wave, {n_spot} spot-checked vs FP:")
    print(f"    Mean diff: {mean_diff:.2e}")
    print(f"    Max diff:  {max_diff:.2e} (family {worst_idx})")
    print(f"    FP[{worst_idx}]={logLs_fp[worst_idx]:.4f}, "
          f"wave[{worst_idx}]={logLs_wv[worst_idx]:.4f}")

    for i in range(n_spot):
        assert diffs[i] < LOGL_ATOL, (
            f"Family {i}: FP={logLs_fp[i]:.6f}, wave={logLs_wv[i]:.6f}, "
            f"diff={diffs[i]:.2e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_wave_vs_fp_small_s_20_families(cpp_ext):
    """Small S: 20 families batched wave vs sequential per-family FP."""
    device = torch.device("cuda")
    dtype = torch.float32

    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_20", 20, cpp_ext, device, dtype
    )

    logLs_wv = _run_batched_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )
    logLs_fp = _run_per_family_fp(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )

    for i in range(len(batch_items)):
        assert math.isfinite(logLs_wv[i]), f"Wave family {i}: logL={logLs_wv[i]}"
        assert abs(logLs_fp[i] - logLs_wv[i]) < LOGL_ATOL, (
            f"Family {i}: FP={logLs_fp[i]:.6f}, wave={logLs_wv[i]:.6f}, "
            f"diff={abs(logLs_fp[i] - logLs_wv[i]):.2e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_wave_vs_fp_100_families_small_s(cpp_ext):
    """100 families from test_trees_100 (small S), wave vs per-family FP."""
    device = torch.device("cuda")
    dtype = torch.float32

    batch_items, sh, pS, pD, pL, tf, mv, Eo = _load_families(
        "test_trees_100", 100, cpp_ext, device, dtype
    )

    logLs_wv = _run_batched_wave(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype,
        chunk_size=50,
    )
    logLs_fp = _run_per_family_fp(
        batch_items, sh, pS, pD, pL, tf, mv, Eo, device, dtype
    )

    diffs = [abs(logLs_fp[i] - logLs_wv[i]) for i in range(len(batch_items))]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    print(f"\n  100 families (small S), wave vs FP:")
    print(f"    Mean diff: {mean_diff:.2e}")
    print(f"    Max diff:  {max_diff:.2e}")

    for i in range(len(batch_items)):
        assert math.isfinite(logLs_wv[i]), f"Wave family {i}: logL={logLs_wv[i]}"
        assert diffs[i] < LOGL_ATOL, (
            f"Family {i}: FP={logLs_fp[i]:.6f}, wave={logLs_wv[i]:.6f}, "
            f"diff={diffs[i]:.2e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_model_api_wave_vs_sequential(cpp_ext):
    """End-to-end: GeneDataset.compute_likelihood_batch wave matches per-family."""
    data_dir = _ROOT / "data" / "test_trees_1000"
    if not data_dir.exists():
        pytest.skip("test_trees_1000 not found")

    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:10]]

    ds = GeneDataset(sp, genes, genewise=False, specieswise=False, pairwise=False,
                     dtype=torch.float32, device=torch.device("cuda"))

    # Wave batched
    logLs_wv = ds.compute_likelihood_batch(tol_Pi=TOL)
    # Per-family (uses FP internally)
    logLs_seq = [ds.compute_likelihood(i, tol_Pi=TOL)["log_likelihood"]
                 for i in range(len(genes))]

    for i in range(len(genes)):
        assert math.isfinite(logLs_wv[i]), f"Wave family {i}: logL={logLs_wv[i]}"
        assert abs(logLs_wv[i] - logLs_seq[i]) < LOGL_ATOL, (
            f"Family {i}: wave={logLs_wv[i]:.4f}, seq={logLs_seq[i]:.4f}, "
            f"diff={abs(logLs_wv[i] - logLs_seq[i]):.2e}"
        )


# ------------------------------------------------------------------
# ALeRax comparison tests
# ------------------------------------------------------------------

# ALeRax reference data: (dataset, gene_tree, output_dir, D, L, T, alerax_logL_nats)
_ALERAX_REFS = [
    ("test_trees_1", "g.nwk", "output", 1e-10, 1e-10, 1e-10, -2.56495),
    ("test_trees_2", "g.nwk", "output", 1e-10, 1e-10, 0.0517229, -8.72486),
    ("test_trees_3", "g.nwk", "output", 0.0555539, 1e-10, 1e-10, -6.75086),
    ("test_mixed_200", "g.nwk", "output", 0.16103, 1e-10, 0.156391, -6215.73),
]


def _run_wave_single(ext, sp_path, gene_path, D_val, L_val, T_val, device, dtype):
    """Run wave forward for a single family with given DTL params."""
    raw = ext.preprocess(sp_path, [str(gene_path)])
    sr, cr = raw["species"], raw["ccp"]

    sh = {
        "S": int(sr["S"]),
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }

    ch = {k: (v.to(device) if torch.is_tensor(v) else v)
          for k, v in cr.items()}
    ch["log_split_probs_sorted"] = cr["log_split_probs_sorted"].to(dtype=dtype, device=device) * _INV

    li = raw["leaf_row_index"].long().to(device)
    lc = raw["leaf_col_index"].long().to(device)
    root_id = int(cr["root_clade_id"])

    theta = torch.log2(torch.tensor([D_val, L_val, T_val], dtype=dtype, device=device))
    tm = torch.log2(sh["Recipients_mat"])
    pS, pD, pL, tf, mt = extract_parameters(
        theta, tm, genewise=False, specieswise=False, pairwise=False
    )
    mv = mt.squeeze(-1) if mt.ndim == 2 else mt

    Eo = E_fixed_point(
        species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv, max_iters=2000,
        tolerance=1e-12, warm_start_E=None, dtype=dtype, device=device,
    )

    waves, phases = compute_clade_waves(ch)
    wave_layout = build_wave_layout(
        waves=waves, phases=phases,
        ccp_helpers=ch,
        leaf_row_index=li, leaf_col_index=lc,
        root_clade_ids=torch.tensor([root_id], dtype=torch.long, device=device),
        device=device, dtype=dtype,
    )
    wv = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sh,
        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mv,
        device=device, dtype=dtype,
        local_iters=2000, local_tolerance=1e-6,
    )

    logL_bits = float(compute_log_likelihood(wv["Pi"], Eo["E"], root_id))
    LN2 = math.log(2.0)
    logL_nats = -logL_bits * LN2
    return logL_nats, sh["S"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ref", _ALERAX_REFS,
                         ids=[r[0] for r in _ALERAX_REFS])
def test_wave_matches_alerax(cpp_ext, ref):
    """Wave forward matches ALeRax per-family likelihood at ALeRax's optimized parameters.

    ALeRax 1.3.0 per_fam_likelihoods.txt omits the uniform origination prior
    1/S from per-family log-likelihoods, while our compute_log_likelihood
    includes it as -log2(S). We correct for this by adding ln(S) to our
    result before comparison.
    """
    ds_name, gene_file, out_dir, D_val, L_val, T_val, alerax_nats = ref
    device = torch.device("cuda")
    dtype = torch.float32  # Triton kernels require float32

    data_dir = _ROOT / "data" / ds_name
    if not data_dir.exists():
        pytest.skip(f"{ds_name} not found")

    sp_path = str(data_dir / "sp.nwk")
    gene_path = str(data_dir / gene_file)

    wave_nats, S = _run_wave_single(
        cpp_ext, sp_path, gene_path, D_val, L_val, T_val, device, dtype
    )

    # ALeRax 1.3.0 omits the 1/S origination prior from per-family reporting.
    # Our formula includes -log2(S) in bits = -ln(S) in nats. Add it back.
    wave_nats_no_orig = wave_nats + math.log(S)

    diff = abs(wave_nats_no_orig - alerax_nats)
    print(f"\n  {ds_name} (S={S}): ALeRax={alerax_nats:.5f}, "
          f"wave={wave_nats:.5f}, wave(no orig)={wave_nats_no_orig:.5f}, "
          f"diff={diff:.2e}")

    # ALeRax likelihoods are reported to ~5 decimal places
    assert diff < 0.05, (
        f"{ds_name}: ALeRax={alerax_nats}, wave(no orig)={wave_nats_no_orig}, diff={diff:.2e}"
    )
