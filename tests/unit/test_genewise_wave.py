"""Test genewise wave support.

Validates that genewise + uniform/uniform pibar modes use the wave
per-family loop path and produce results consistent with the genewise FP fallback.
"""

import math
from pathlib import Path

import pytest
import torch

from gpurec.core.model import GeneDataset
from gpurec.core.extract_parameters import extract_parameters_uniform

_ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-3
# uniform vs dense produces small diffs; genewise adds per-gene variation
LOGL_ATOL = 5e-2


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_dir():
    d = _ROOT / "data" / "test_trees_100"
    if not d.exists():
        pytest.skip("test_trees_100 not found")
    return d


@pytest.fixture(scope="module")
def data_dir_large():
    d = _ROOT / "data" / "test_trees_1000"
    if not d.exists():
        pytest.skip("test_trees_1000 not found")
    return d


# ------------------------------------------------------------------
# Test: extract_parameters_uniform genewise shapes
# ------------------------------------------------------------------

def test_extract_params_uniform_genewise_shapes():
    """Verify shapes for genewise uniform extraction."""
    G, S = 5, 199
    device = torch.device("cpu")
    dtype = torch.float32
    unnorm_row_max = torch.randn(S, dtype=dtype, device=device)

    # genewise, non-specieswise: theta [G, 3]
    theta_g = torch.randn(G, 3, dtype=dtype, device=device)
    pS, pD, pL, tm, mt = extract_parameters_uniform(
        theta_g, unnorm_row_max, specieswise=False, genewise=True,
    )
    assert pS.shape == (G,), f"pS shape: {pS.shape}"
    assert pD.shape == (G,), f"pD shape: {pD.shape}"
    assert pL.shape == (G,), f"pL shape: {pL.shape}"
    assert tm is None
    assert mt.shape == (G, S), f"mt shape: {mt.shape}"

    # genewise + specieswise: theta [G, S, 3]
    theta_gs = torch.randn(G, S, 3, dtype=dtype, device=device)
    pS2, pD2, pL2, tm2, mt2 = extract_parameters_uniform(
        theta_gs, unnorm_row_max, specieswise=True, genewise=True,
    )
    assert pS2.shape == (G, S), f"pS shape: {pS2.shape}"
    assert pD2.shape == (G, S), f"pD shape: {pD2.shape}"
    assert pL2.shape == (G, S), f"pL shape: {pL2.shape}"
    assert tm2 is None
    assert mt2.shape == (G, S), f"mt shape: {mt2.shape}"


def test_extract_params_uniform_genewise_consistency():
    """Per-family extraction == batched genewise extraction (non-specieswise)."""
    G, S = 4, 50
    dtype = torch.float32
    unnorm_row_max = torch.randn(S, dtype=dtype)

    thetas = [torch.randn(3, dtype=dtype) for _ in range(G)]
    # Per-family
    pS_list, mt_list = [], []
    for t in thetas:
        pS, pD, pL, _, mt = extract_parameters_uniform(t, unnorm_row_max, specieswise=False)
        pS_list.append(pS)
        mt_list.append(mt)

    # Batched genewise
    theta_stack = torch.stack(thetas, dim=0)
    pS_g, pD_g, pL_g, _, mt_g = extract_parameters_uniform(
        theta_stack, unnorm_row_max, specieswise=False, genewise=True,
    )

    for i in range(G):
        assert torch.allclose(pS_g[i], pS_list[i], atol=1e-6), f"pS mismatch at gene {i}"
        assert torch.allclose(mt_g[i], mt_list[i], atol=1e-6), f"mt mismatch at gene {i}"


# ------------------------------------------------------------------
# Test: genewise wave vs genewise FP (uniform)
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_genewise_wave_vs_fp_uniform(data_dir):
    """Genewise wave (uniform) matches genewise FP (dense)."""
    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:5]]

    device = torch.device("cuda")
    dtype = torch.float32

    # Create genewise dataset with different theta per family
    ds = GeneDataset(sp, genes, genewise=True, specieswise=False, pairwise=False,
                     dtype=dtype, device=device)

    # Assign different rates per family
    for i, fam in enumerate(ds.families):
        fam['theta'] = torch.randn(3, dtype=dtype, device=device) * 0.5 - 5.0

    # Wave path (genewise + uniform)
    logLs_wave = ds.compute_likelihood_batch(
        pibar_mode='uniform', tol_Pi=TOL, tol_E=TOL,
    )

    # FP fallback (genewise + dense)
    logLs_fp = ds.compute_likelihood_batch(
        pibar_mode='dense', tol_Pi=TOL, tol_E=TOL,
    )

    for i, (lw, lf) in enumerate(zip(logLs_wave, logLs_fp)):
        assert math.isfinite(lw), f"Wave family {i}: logL={lw}"
        assert math.isfinite(lf), f"FP family {i}: logL={lf}"
        assert abs(lw - lf) < LOGL_ATOL, (
            f"Family {i}: wave={lw:.4f}, fp={lf:.4f}, diff={abs(lw - lf):.2e}"
        )


# ------------------------------------------------------------------
# Test: genewise wave vs genewise FP (specieswise + uniform)
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_genewise_wave_vs_fp_specieswise_uniform(data_dir):
    """Genewise + specieswise wave (uniform) matches FP (dense)."""
    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:3]]

    device = torch.device("cuda")
    dtype = torch.float32

    ds = GeneDataset(sp, genes, genewise=True, specieswise=True, pairwise=False,
                     dtype=dtype, device=device)

    S = ds.S
    for i, fam in enumerate(ds.families):
        fam['theta'] = torch.randn(S, 3, dtype=dtype, device=device) * 0.3 - 5.0

    logLs_wave = ds.compute_likelihood_batch(
        pibar_mode='uniform', tol_Pi=TOL, tol_E=TOL,
    )
    logLs_fp = ds.compute_likelihood_batch(
        pibar_mode='dense', tol_Pi=TOL, tol_E=TOL,
    )

    for i, (lw, lf) in enumerate(zip(logLs_wave, logLs_fp)):
        assert math.isfinite(lw), f"Wave family {i}: logL={lw}"
        assert math.isfinite(lf), f"FP family {i}: logL={lf}"
        # Specieswise has slightly higher uniform error
        tol = max(LOGL_ATOL, abs(lf) * 5e-4)
        assert abs(lw - lf) < tol, (
            f"Family {i}: wave={lw:.4f}, fp={lf:.4f}, diff={abs(lw - lf):.2e}"
        )


# ------------------------------------------------------------------
# Test: genewise wave batch == sequential single-family wave
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_genewise_wave_batch_vs_per_family(data_dir):
    """Batched genewise wave == per-family genewise wave (single-index batches)."""
    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:4]]

    device = torch.device("cuda")
    dtype = torch.float32

    ds = GeneDataset(sp, genes, genewise=True, specieswise=False, pairwise=False,
                     dtype=dtype, device=device)

    for i, fam in enumerate(ds.families):
        fam['theta'] = torch.randn(3, dtype=dtype, device=device) * 0.5 - 5.0

    # Full batch
    logLs_batch = ds.compute_likelihood_batch(
        pibar_mode='uniform', tol_Pi=TOL, tol_E=TOL,
    )

    # Per-family (single-index batches, same code path)
    logLs_seq = []
    for i in range(len(genes)):
        logL_i = ds.compute_likelihood_batch(
            [i], pibar_mode='uniform', tol_Pi=TOL, tol_E=TOL,
        )
        logLs_seq.append(logL_i[0])

    for i, (lb, ls) in enumerate(zip(logLs_batch, logLs_seq)):
        assert math.isfinite(lb), f"Batch family {i}: logL={lb}"
        assert math.isfinite(ls), f"Seq family {i}: logL={ls}"
        # Same code path, only difference is E batching — should be very close
        assert abs(lb - ls) < 1e-2, (
            f"Family {i}: batch={lb:.4f}, seq={ls:.4f}, diff={abs(lb - ls):.2e}"
        )


# ------------------------------------------------------------------
# Test: genewise uniform mode (exact, with ancestor correction)
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_genewise_wave_large_s(data_dir_large):
    """Genewise wave works on larger species trees (S~2000)."""
    sp = str(data_dir_large / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir_large.glob("g_*.nwk"))[:3]]

    device = torch.device("cuda")
    dtype = torch.float32

    ds = GeneDataset(sp, genes, genewise=True, specieswise=False, pairwise=False,
                     dtype=dtype, device=device)

    for i, fam in enumerate(ds.families):
        fam['theta'] = torch.randn(3, dtype=dtype, device=device) * 0.5 - 5.0

    # wave (genewise + uniform)
    logLs_wave = ds.compute_likelihood_batch(
        pibar_mode='uniform', tol_Pi=TOL, tol_E=TOL,
    )

    # dense FP (reference)
    logLs_fp = ds.compute_likelihood_batch(
        pibar_mode='dense', tol_Pi=TOL, tol_E=TOL,
    )

    for i, (lw, lf) in enumerate(zip(logLs_wave, logLs_fp)):
        assert math.isfinite(lw), f"Wave family {i}: logL={lw}"
        assert math.isfinite(lf), f"FP family {i}: logL={lf}"
        # Use relative tolerance: absolute diffs scale with logL magnitude
        tol = max(LOGL_ATOL, abs(lf) * 5e-4)
        assert abs(lw - lf) < tol, (
            f"Family {i}: wave={lw:.4f}, fp={lf:.4f}, diff={abs(lw - lf):.2e}"
        )
