"""Tests for optimize_theta_genewise (genewise L-BFGS)."""

import math
from pathlib import Path

import pytest
import torch

from gpurec.core.model import GeneDataset
from gpurec.optimization.theta_optimizer import optimize_theta_genewise, optimize_theta_wave

_ROOT = Path(__file__).resolve().parent.parent
TOL_PI = 1e-3
TOL_E = 1e-8


@pytest.fixture(scope="module")
def data_dir():
    d = _ROOT / "data" / "test_trees_100"
    if not d.exists():
        pytest.skip("test_trees_100 not found")
    return d


# ------------------------------------------------------------------
# Test: NLL decreases over L-BFGS steps
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nll_decreases(data_dir):
    """Verify that NLL decreases over L-BFGS steps for 3 families."""
    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:3]]

    device = torch.device("cuda")
    dtype = torch.float32

    ds = GeneDataset(sp, genes, genewise=True, specieswise=False, pairwise=False,
                     dtype=dtype, device=device)

    G = len(ds.families)
    theta_init = torch.randn(G, 3, dtype=dtype, device=device) * 0.3 - 5.0

    result = optimize_theta_genewise(
        families=ds.families,
        species_helpers=ds.species_helpers,
        unnorm_row_max=ds.unnorm_row_max,
        theta_init=theta_init,
        max_steps=5,
        lbfgs_m=5,
        grad_tol=1e-8,  # don't converge early, run all steps
        device=device, dtype=dtype,
        pibar_mode='uniform',
    )

    history = result['history']
    assert len(history) >= 2, f"Expected at least 2 history entries, got {len(history)}"

    # NLL should decrease (or stay flat) for each gene across steps
    nll_start = history[0]['nll']
    nll_end = history[-1]['nll']
    for g in range(G):
        assert nll_end[g] <= nll_start[g] + 1e-2, (
            f"Gene {g}: NLL increased from {nll_start[g]:.4f} to {nll_end[g]:.4f}"
        )

    # Total NLL should decrease
    assert nll_end.sum() < nll_start.sum() + 1e-2, (
        f"Total NLL increased: {nll_start.sum():.4f} -> {nll_end.sum():.4f}"
    )


# ------------------------------------------------------------------
# Test: matches per-family scipy L-BFGS-B
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_matches_scipy_per_family(data_dir):
    """Final rates from genewise L-BFGS ~= per-family optimize_theta_wave(lbfgs)."""
    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:2]]

    device = torch.device("cuda")
    dtype = torch.float32

    ds = GeneDataset(sp, genes, genewise=True, specieswise=False, pairwise=False,
                     dtype=dtype, device=device)

    G = len(ds.families)
    # Use same init for both methods
    torch.manual_seed(42)
    theta_init = torch.randn(G, 3, dtype=dtype, device=device) * 0.3 - 5.0

    # Genewise L-BFGS (our new function)
    result_genewise = optimize_theta_genewise(
        families=ds.families,
        species_helpers=ds.species_helpers,
        unnorm_row_max=ds.unnorm_row_max,
        theta_init=theta_init,
        max_steps=25,
        lbfgs_m=10,
        grad_tol=1e-6,
        device=device, dtype=dtype,
        pibar_mode='uniform',
    )

    # Per-family scipy L-BFGS-B (reference)
    from gpurec.core.batching import collate_gene_families, build_wave_layout
    from gpurec.core.scheduling import compute_clade_waves

    for g in range(G):
        fam = ds.families[g]
        single_item = {
            'ccp': fam['ccp_helpers'],
            'leaf_row_index': fam['leaf_row_index'],
            'leaf_col_index': fam['leaf_col_index'],
            'root_clade_id': int(fam['root_clade_id']),
        }
        single_batched = collate_gene_families([single_item], dtype=dtype, device=device)
        waves_g, phases_g = compute_clade_waves(fam['ccp_helpers'])
        wl_g = build_wave_layout(
            waves=waves_g, phases=phases_g,
            ccp_helpers=single_batched['ccp'],
            leaf_row_index=single_batched['leaf_row_index'],
            leaf_col_index=single_batched['leaf_col_index'],
            root_clade_ids=single_batched['root_clade_ids'],
            device=device, dtype=dtype,
        )

        # Move species_helpers to device for optimize_theta_wave
        sp_helpers_gpu = {
            k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else
                v.to(device=device) if torch.is_tensor(v) else v)
            for k, v in ds.species_helpers.items()
        }
        result_scipy = optimize_theta_wave(
            wave_layout=wl_g,
            species_helpers=sp_helpers_gpu,
            root_clade_ids=single_batched['root_clade_ids'],
            unnorm_row_max=ds.unnorm_row_max.to(device=device, dtype=dtype),
            theta_init=theta_init[g],
            steps=30,
            optimizer='lbfgs',
            device=device, dtype=dtype,
            pibar_mode='uniform',
        )

        nll_gw = float(result_genewise['nll'][g])
        nll_sc = float(result_scipy['negative_log_likelihood'])

        # Genewise NLL should be close to scipy NLL (within 2% relative)
        # Genewise may find equal or slightly better minimum
        rel_diff = (nll_gw - nll_sc) / max(abs(nll_sc), 1.0)
        assert rel_diff < 0.02, (
            f"Gene {g}: genewise NLL={nll_gw:.4f}, scipy NLL={nll_sc:.4f}, "
            f"rel_diff={rel_diff:.4f}"
        )


# ------------------------------------------------------------------
# Test: convergence masking reduces n_active
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_convergence_masking(data_dir):
    """With varying difficulty, n_active should decrease over steps."""
    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:5]]

    device = torch.device("cuda")
    dtype = torch.float32

    ds = GeneDataset(sp, genes, genewise=True, specieswise=False, pairwise=False,
                     dtype=dtype, device=device)

    G = len(ds.families)
    # Some genes start near optimum (small theta), others far away
    theta_init = torch.zeros(G, 3, dtype=dtype, device=device)
    # Put first 2 genes near a reasonable optimum
    theta_init[:2] = torch.tensor([-3.0, -3.0, -3.0], dtype=dtype, device=device)
    # Put others far from optimum
    theta_init[2:] = torch.tensor([-8.0, -1.0, -8.0], dtype=dtype, device=device)

    result = optimize_theta_genewise(
        families=ds.families,
        species_helpers=ds.species_helpers,
        unnorm_row_max=ds.unnorm_row_max,
        theta_init=theta_init,
        max_steps=10,
        lbfgs_m=5,
        grad_tol=1e-3,  # relatively loose tolerance to see convergence masking
        device=device, dtype=dtype,
        pibar_mode='uniform',
    )

    history = result['history']
    # Should have run at least a few steps
    assert len(history) >= 2, f"Expected at least 2 steps, got {len(history)}"

    # n_active should be non-increasing
    n_actives = [h['n_active'] for h in history]
    for i in range(1, len(n_actives)):
        assert n_actives[i] <= n_actives[i - 1], (
            f"n_active increased at step {i}: {n_actives[i-1]} -> {n_actives[i]}"
        )
