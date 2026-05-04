"""Test genewise wave support.

Validates that genewise + uniform/uniform pibar modes use the wave
per-family loop path and produce results consistent with the genewise FP fallback.
"""

import math
from pathlib import Path

import pytest
import torch

from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import compute_log_likelihood_root_rows
from gpurec.core.model import GeneDataset
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.scheduling import compute_clade_waves
from gpurec.api.model import GeneReconModel

_ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-3
# uniform vs dense produces small diffs; genewise adds per-gene variation
LOGL_ATOL = 5e-2


def _set_deterministic_genewise_theta(ds, *, specieswise, scale=0.25):
    gen = torch.Generator(device=ds.device)
    gen.manual_seed(7)
    for fam in ds.families:
        shape = (ds.S, 3) if specieswise else (3,)
        fam["theta"] = (
            torch.randn(shape, generator=gen, dtype=ds.dtype, device=ds.device)
            * scale
            - 5.0
        )


def _build_genewise_wave_inputs(ds, indices, *, device, dtype):
    batch_items = [
        {
            "ccp": ds.families[idx]["ccp_helpers"],
            "leaf_row_index": ds.families[idx]["leaf_row_index"],
            "leaf_col_index": ds.families[idx]["leaf_col_index"],
            "root_clade_id": int(ds.families[idx]["root_clade_id"]),
        }
        for idx in indices
    ]
    batched = collate_gene_families(batch_items, dtype=dtype, device=device)

    families_waves = []
    families_phases = []
    for idx in indices:
        waves_i, phases_i = compute_clade_waves(ds.families[idx]["ccp_helpers"])
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

    wave_layout = build_wave_layout(
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

    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = ds._extract_batch_params(
        indices,
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
    )
    species_helpers, ancestors_T = ds._species_helpers_for_mode(
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
    )
    E_out = ds._solve_e_fixed_point(
        species_helpers=species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=transfer_mat,
        max_transfer_vec=max_transfer_vec,
        max_iters_E=200,
        tol_E=TOL,
        device=device,
        dtype=dtype,
        pibar_mode="uniform",
        ancestors_T=ancestors_T,
    )

    return wave_layout, species_helpers, E_out, log_pS, log_pD, log_pL, max_transfer_vec


def _run_genewise_uniform_wave(ds, wave_layout, species_helpers, E_out,
                               log_pS, log_pD, log_pL, max_transfer_vec):
    return Pi_wave_forward(
        wave_layout=wave_layout,
        species_helpers=species_helpers,
        E=E_out["E"],
        Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"],
        E_s2=E_out["E_s2"],
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=None,
        max_transfer_mat=max_transfer_vec,
        device=ds.device,
        dtype=ds.dtype,
        local_iters=200,
        local_tolerance=TOL,
        fixed_iters=3,
        pibar_mode="uniform",
        family_idx=wave_layout["family_idx"],
        return_original=True,
        need_pibar=False,
        return_root_rows=True,
    )


def _nll_from_root_rows(Pi_out, E_out):
    return compute_log_likelihood_root_rows(Pi_out["Pi_root_rows"], E_out["E"])


def _make_theta_init(G, S, *, specieswise, dtype, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(11)
    shape = (G, S, 3) if specieswise else (G, 3)
    return torch.randn(shape, generator=gen, dtype=dtype, device=device) * 0.2 - 5.0


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
# Test: Proposal 7 genewise uniform leaf-index path
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("specieswise", [False, True], ids=["log_pS_G", "log_pS_GS"])
def test_genewise_uniform_leaf_index_path_matches_dense_leaf_fallback(data_dir, specieswise):
    """Proposal 7: batched genewise uniform leaf-index path matches dense leaf terms.

    The fallback layout deletes ``leaf_species_index`` so a Proposal 7
    implementation must use the compact leaf-index path for the normal layout
    and the existing dense ``[C, S]`` leaf term for the reference layout.
    """
    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:3]]
    device = torch.device("cuda")
    dtype = torch.float32

    ds = GeneDataset(
        sp,
        genes,
        genewise=True,
        specieswise=specieswise,
        pairwise=False,
        dtype=dtype,
        device=device,
    )
    _set_deterministic_genewise_theta(ds, specieswise=specieswise)

    inputs = _build_genewise_wave_inputs(ds, list(range(len(genes))), device=device, dtype=dtype)
    wave_layout, species_helpers, E_out, log_pS, log_pD, log_pL, max_transfer_vec = inputs
    assert log_pS.shape == ((len(genes), ds.S) if specieswise else (len(genes),))

    fallback_layout = dict(wave_layout)
    fallback_layout["leaf_species_index"] = None

    dense_out = _run_genewise_uniform_wave(
        ds, fallback_layout, species_helpers, E_out,
        log_pS, log_pD, log_pL, max_transfer_vec,
    )
    leaf_index_out = _run_genewise_uniform_wave(
        ds, wave_layout, species_helpers, E_out,
        log_pS, log_pD, log_pL, max_transfer_vec,
    )

    if leaf_index_out["clade_species_map"] is not None:
        pytest.xfail("Proposal 7 is not active: batched genewise uniform still materializes dense leaf terms")

    assert dense_out["clade_species_map"] is not None
    dense_nll = _nll_from_root_rows(dense_out, E_out)
    leaf_index_nll = _nll_from_root_rows(leaf_index_out, E_out)

    assert torch.allclose(leaf_index_nll, dense_nll, atol=1e-4, rtol=1e-5)
    assert torch.allclose(
        leaf_index_out["Pi_root_rows"],
        dense_out["Pi_root_rows"],
        atol=1e-4,
        rtol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("specieswise", [False, True], ids=["log_pS_G", "log_pS_GS"])
def test_genewise_uniform_family_indexed_constants_match_row_expanded(
    data_dir, specieswise, monkeypatch
):
    """Proposal 0: [G,S] in-kernel constants match old per-wave [W,S] expansion."""
    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:3]]
    device = torch.device("cuda")
    dtype = torch.float32

    ds = GeneDataset(
        sp,
        genes,
        genewise=True,
        specieswise=specieswise,
        pairwise=False,
        dtype=dtype,
        device=device,
    )
    _set_deterministic_genewise_theta(ds, specieswise=specieswise)

    inputs = _build_genewise_wave_inputs(ds, list(range(len(genes))), device=device, dtype=dtype)
    wave_layout, species_helpers, E_out, log_pS, log_pD, log_pL, max_transfer_vec = inputs

    monkeypatch.setenv("GPUREC_FORWARD_FAMILY_INDEXED_CONSTS", "0")
    row_expanded_out = _run_genewise_uniform_wave(
        ds, wave_layout, species_helpers, E_out,
        log_pS, log_pD, log_pL, max_transfer_vec,
    )

    monkeypatch.setenv("GPUREC_FORWARD_FAMILY_INDEXED_CONSTS", "1")
    family_indexed_out = _run_genewise_uniform_wave(
        ds, wave_layout, species_helpers, E_out,
        log_pS, log_pD, log_pL, max_transfer_vec,
    )

    row_nll = _nll_from_root_rows(row_expanded_out, E_out)
    family_nll = _nll_from_root_rows(family_indexed_out, E_out)

    assert torch.allclose(family_nll, row_nll, atol=1e-4, rtol=1e-5)
    assert torch.allclose(
        family_indexed_out["Pi_root_rows"],
        row_expanded_out["Pi_root_rows"],
        atol=1e-4,
        rtol=1e-5,
    )
    assert torch.allclose(
        family_indexed_out["Pi"],
        row_expanded_out["Pi"],
        atol=1e-4,
        rtol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_genewise_uniform_leaf_index_gradient_bridge_matches_dense_leaf_fallback(data_dir):
    """Proposal 7 gradient bridge parity for genewise uniform leaf-index leaves."""
    sp = str(data_dir / "sp.nwk")
    genes = [str(g) for g in sorted(data_dir.glob("g_*.nwk"))[:2]]
    device = torch.device("cuda")
    dtype = torch.float32
    specieswise = False

    probe_ds = GeneDataset(
        sp,
        genes,
        genewise=True,
        specieswise=specieswise,
        pairwise=False,
        dtype=dtype,
        device=device,
    )
    _set_deterministic_genewise_theta(probe_ds, specieswise=specieswise)
    inputs = _build_genewise_wave_inputs(probe_ds, list(range(len(genes))), device=device, dtype=dtype)
    probe_out = _run_genewise_uniform_wave(probe_ds, *inputs)
    if probe_out["clade_species_map"] is not None:
        pytest.xfail("Proposal 7 is not active: batched genewise uniform still materializes dense leaf terms")

    theta_init = _make_theta_init(
        len(genes), probe_ds.S, specieswise=specieswise, dtype=dtype, device=device
    )
    kwargs = dict(
        mode="genewise",
        pibar_mode="uniform",
        max_iters_E=200,
        tol_E=TOL,
        max_iters_Pi=200,
        tol_Pi=TOL,
        fixed_iters_Pi=3,
        neumann_terms=2,
        use_pruning=False,
        theta_init=theta_init,
    )

    fast_ds = GeneDataset(
        sp,
        genes,
        genewise=True,
        specieswise=specieswise,
        pairwise=False,
        dtype=dtype,
        device=device,
    )
    dense_ds = GeneDataset(
        sp,
        genes,
        genewise=True,
        specieswise=specieswise,
        pairwise=False,
        dtype=dtype,
        device=device,
    )
    fast_model = GeneReconModel(dataset=fast_ds, **kwargs)
    dense_model = GeneReconModel(dataset=dense_ds, **kwargs)

    dense_layout = dict(dense_model._static.wave_layout)
    dense_layout["leaf_species_index"] = None
    dense_model._static.wave_layout = dense_layout

    fast_nll = fast_model()
    dense_nll = dense_model()
    fast_nll.backward()
    dense_nll.backward()

    assert torch.allclose(fast_nll, dense_nll, atol=1e-4, rtol=1e-5)
    assert torch.allclose(
        fast_model.theta.grad,
        dense_model.theta.grad,
        atol=2e-3,
        rtol=2e-3,
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
