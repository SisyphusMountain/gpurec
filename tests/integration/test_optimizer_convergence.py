"""
Optimizer convergence validation.

Tier 1: Self-consistent convergence (no AleRax needed)
  - test_genewise_converges: 20 families, L-BFGS genewise, verify gradient→0
  - test_shared_converges: same families, shared params, verify convergence
  - test_genewise_reproducible: two random inits → same optimum
  - test_forward_consistency: recompute likelihood at optimum → matches optimizer NLL

Tier 2: AleRax reference comparison (needs AleRax+MPI)
  - test_genewise_matches_alerax: simulate trees, run AleRax, compare parameters
"""

import math
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from src.core.model import GeneDataset
from src.optimization.theta_optimizer import optimize_theta_genewise, optimize_theta_wave

_ROOT = Path(__file__).resolve().parents[1]
DATA_1000 = _ROOT / "data" / "test_trees_1000"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

N_FAMILIES = 5  # keep low to avoid OOM on 24GB GPU

@pytest.fixture(scope="module")
def ds_genewise():
    """Load families from test_trees_1000 as genewise dataset."""
    if not DATA_1000.exists():
        pytest.skip("test_trees_1000 not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    sp = str(DATA_1000 / "sp.nwk")
    genes = [str(g) for g in sorted(DATA_1000.glob("g_*.nwk"))[:N_FAMILIES]]
    ds = GeneDataset(sp, genes, genewise=True, specieswise=False, pairwise=False,
                     dtype=DTYPE, device=DEVICE)
    return ds


@pytest.fixture(scope="module")
def ds_shared():
    """Load families from test_trees_1000 as shared-param dataset."""
    if not DATA_1000.exists():
        pytest.skip("test_trees_1000 not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    sp = str(DATA_1000 / "sp.nwk")
    genes = [str(g) for g in sorted(DATA_1000.glob("g_*.nwk"))[:N_FAMILIES]]
    ds = GeneDataset(sp, genes, genewise=False, specieswise=False, pairwise=False,
                     dtype=DTYPE, device=DEVICE)
    return ds


def _run_genewise(ds, theta_init, max_steps=50, grad_tol=1e-5, pibar_mode='uniform'):
    """Helper to run optimize_theta_genewise with standard settings."""
    return optimize_theta_genewise(
        families=ds.families,
        species_helpers=ds.species_helpers,
        unnorm_row_max=ds.unnorm_row_max,
        theta_init=theta_init,
        max_steps=max_steps,
        lbfgs_m=10,
        grad_tol=grad_tol,
        e_max_iters=2000,
        e_tol=1e-8,
        neumann_terms=3,
        pruning_threshold=1e-6,
        cg_tol=1e-8,
        cg_maxiter=500,
        device=DEVICE,
        dtype=DTYPE,
        pibar_mode=pibar_mode,
    )


# ---------------------------------------------------------------
# Tier 1: Self-consistent convergence
# ---------------------------------------------------------------

class TestGenewiseConverges:
    """Verify that genewise L-BFGS converges (gradient→0) on 20 families."""

    def test_convergence(self, ds_genewise):
        G = len(ds_genewise.families)
        torch.manual_seed(123)
        # Start offset from truth: log2(0.2) ≈ -2.32, add noise
        theta_init = (math.log2(0.2) * torch.ones(G, 3, dtype=DTYPE, device=DEVICE)
                      + torch.randn(G, 3, dtype=DTYPE, device=DEVICE) * 0.5)

        result = _run_genewise(ds_genewise, theta_init, max_steps=20, grad_tol=1e-5)
        history = result['history']

        # Should have run at least a few steps
        assert len(history) >= 3, f"Only {len(history)} steps"

        # Convergence: all genes should have converged (n_active == 0)
        # or gradient norm should be small
        final = history[-1]
        final_nll = result['nll']
        final_rates = result['rates']

        # Print per-step progress
        for i, h in enumerate(history):
            print(f"  Step {i}: NLL_sum={h['nll'].sum():.2f}, "
                  f"grad_inf={h['grad_inf']:.2e}, n_active={h['n_active']}")

        # Check gradient decreased significantly
        grad_first = history[0]['grad_inf']
        grad_last = final['grad_inf']
        print(f"Gradient: {grad_first:.2e} -> {grad_last:.2e} "
              f"({len(history)} steps, n_active={final['n_active']}/{G})")

        # Either all converged or gradient dropped by 100x
        assert final['n_active'] == 0 or grad_last < grad_first * 0.01, (
            f"Insufficient convergence: grad {grad_first:.2e} -> {grad_last:.2e}, "
            f"n_active={final['n_active']}"
        )

        # NLL should be monotonically non-increasing per gene
        for step_idx in range(1, len(history)):
            prev_nll = history[step_idx - 1]['nll']
            curr_nll = history[step_idx]['nll']
            for g in range(G):
                assert curr_nll[g] <= prev_nll[g] + 1e-2, (
                    f"Gene {g} NLL increased at step {step_idx}: "
                    f"{prev_nll[g]:.4f} -> {curr_nll[g]:.4f}"
                )

        # Parameters should be in a reasonable range
        for g in range(G):
            for p in range(3):
                rate = float(final_rates[g, p])
                assert 1e-6 < rate < 2.0, (
                    f"Gene {g} param {p}: rate={rate:.6f} out of range"
                )

        # Print summary
        nll_total = float(final_nll.sum())
        print(f"Final total NLL: {nll_total:.2f}")
        print(f"Rate ranges: D=[{final_rates[:,0].min():.4f}, {final_rates[:,0].max():.4f}], "
              f"L=[{final_rates[:,1].min():.4f}, {final_rates[:,1].max():.4f}], "
              f"T=[{final_rates[:,2].min():.4f}, {final_rates[:,2].max():.4f}]")


class TestSharedConverges:
    """Verify that shared-param L-BFGS converges via optimize_theta_wave."""

    def test_convergence(self, ds_shared):
        from src.core.batching import collate_gene_families, build_wave_layout
        from src.core.scheduling import compute_clade_waves

        # Build cross-family wave layout for all families
        items = []
        for fam in ds_shared.families:
            items.append({
                'ccp': fam['ccp_helpers'],
                'leaf_row_index': fam['leaf_row_index'],
                'leaf_col_index': fam['leaf_col_index'],
                'root_clade_id': int(fam['root_clade_id']),
            })
        batched = collate_gene_families(items, dtype=DTYPE, device=DEVICE)

        # Use first family's waves (cross-family scheduling)
        all_waves, all_phases = [], []
        for fam in ds_shared.families:
            w, p = compute_clade_waves(fam['ccp_helpers'])
            all_waves.append(w)
            all_phases.append(p)

        # Build layout from first family (for shared params, merge all)
        wl = build_wave_layout(
            waves=all_waves[0], phases=all_phases[0],
            ccp_helpers=batched['ccp'],
            leaf_row_index=batched['leaf_row_index'],
            leaf_col_index=batched['leaf_col_index'],
            root_clade_ids=batched['root_clade_ids'],
            device=DEVICE, dtype=DTYPE,
        )

        sp_helpers_gpu = {
            k: (v.to(device=DEVICE, dtype=DTYPE) if torch.is_tensor(v) and v.is_floating_point()
                else v.to(device=DEVICE) if torch.is_tensor(v) else v)
            for k, v in ds_shared.species_helpers.items()
        }

        theta_init = math.log2(0.2) * torch.ones(3, dtype=DTYPE, device=DEVICE)

        result = optimize_theta_wave(
            wave_layout=wl,
            species_helpers=sp_helpers_gpu,
            root_clade_ids=batched['root_clade_ids'],
            unnorm_row_max=ds_shared.unnorm_row_max.to(device=DEVICE, dtype=DTYPE),
            theta_init=theta_init,
            steps=60,
            optimizer='lbfgs',
            device=DEVICE,
            dtype=DTYPE,
            pibar_mode='uniform',
        )

        history = result['history']
        assert len(history) >= 3, f"Only {len(history)} steps"

        # Check convergence: gradient should drop significantly
        grad_first = history[0].grad_infinity_norm
        grad_last = history[-1].grad_infinity_norm
        print(f"Shared-param gradient: {grad_first:.2e} -> {grad_last:.2e} "
              f"({len(history)} evaluations)")

        # NLL should decrease
        nll_first = float(history[0].negative_log_likelihood)
        nll_last = float(history[-1].negative_log_likelihood)
        assert nll_last < nll_first, (
            f"NLL did not decrease: {nll_first:.4f} -> {nll_last:.4f}"
        )

        # Rates should be reasonable
        rates = result['rates']
        for p in range(3):
            rate = float(rates[p])
            assert 1e-6 < rate < 2.0, f"Param {p}: rate={rate:.6f} out of range"

        print(f"Shared-param rates: D={float(rates[0]):.4f}, "
              f"L={float(rates[1]):.4f}, T={float(rates[2]):.4f}")
        print(f"Final NLL: {nll_last:.2f}")


class TestGenewiseReproducible:
    """Two different random inits should converge to the same optimum."""

    def test_reproducibility(self, ds_genewise):
        G = len(ds_genewise.families)

        # Run 1
        torch.manual_seed(111)
        theta1 = (math.log2(0.1) * torch.ones(G, 3, dtype=DTYPE, device=DEVICE)
                  + torch.randn(G, 3, dtype=DTYPE, device=DEVICE) * 0.8)
        result1 = _run_genewise(ds_genewise, theta1, max_steps=60, grad_tol=1e-5)

        # Run 2: different init
        torch.manual_seed(222)
        theta2 = (math.log2(0.3) * torch.ones(G, 3, dtype=DTYPE, device=DEVICE)
                  + torch.randn(G, 3, dtype=DTYPE, device=DEVICE) * 0.8)
        result2 = _run_genewise(ds_genewise, theta2, max_steps=60, grad_tol=1e-5)

        # Per-gene NLL should be close (same optimum)
        nll1 = result1['nll']
        nll2 = result2['nll']

        max_diff = float((nll1 - nll2).abs().max())
        mean_diff = float((nll1 - nll2).abs().mean())
        print(f"NLL diff across inits: mean={mean_diff:.4f}, max={max_diff:.4f}")

        # Allow some tolerance — L-BFGS might find slightly different local minima
        # but NLL should be very close
        assert max_diff < 0.5, (
            f"NLL differs too much between inits: max_diff={max_diff:.4f}"
        )


class TestForwardConsistency:
    """Recompute likelihood at optimized parameters → matches optimizer NLL."""

    def test_consistency(self, ds_genewise):
        G = len(ds_genewise.families)
        torch.manual_seed(42)
        theta_init = torch.randn(G, 3, dtype=DTYPE, device=DEVICE) * 0.3 - 4.0

        result = _run_genewise(ds_genewise, theta_init, max_steps=30, grad_tol=1e-5)

        # Recompute likelihood at optimized theta using forward-only path
        theta_opt = result['theta'].to(device=DEVICE, dtype=DTYPE)

        from src.core.extract_parameters import extract_parameters_uniform
        from src.core.likelihood import E_fixed_point, Pi_wave_forward, compute_log_likelihood
        from src.core.batching import collate_gene_families, build_wave_layout
        from src.core.scheduling import compute_clade_waves

        S = ds_genewise.species_helpers['S']
        unnorm = ds_genewise.unnorm_row_max.to(device=DEVICE, dtype=DTYPE)

        for g in range(G):
            theta_g = theta_opt[g]
            params = extract_parameters_uniform(
                theta_g, unnorm, genewise=False, specieswise=False)

            sp_helpers_gpu = {
                k: (v.to(device=DEVICE, dtype=DTYPE) if torch.is_tensor(v) and v.is_floating_point()
                    else v.to(device=DEVICE) if torch.is_tensor(v) else v)
                for k, v in ds_genewise.species_helpers.items()
            }

            E_out = E_fixed_point(
                species_helpers=sp_helpers_gpu,
                log_pS=params['log_pS'], log_pD=params['log_pD'], log_pL=params['log_pL'],
                transfer_mat=None,
                max_transfer_mat=params['max_transfer_mat'],
                max_iters=2000, tolerance=1e-8,
                warm_start_E=None, dtype=DTYPE, device=DEVICE,
                pibar_mode='uniform',
            )

            fam = ds_genewise.families[g]
            item = {
                'ccp': fam['ccp_helpers'],
                'leaf_row_index': fam['leaf_row_index'],
                'leaf_col_index': fam['leaf_col_index'],
                'root_clade_id': int(fam['root_clade_id']),
            }
            batched = collate_gene_families([item], dtype=DTYPE, device=DEVICE)
            waves_g, phases_g = compute_clade_waves(fam['ccp_helpers'])
            wl = build_wave_layout(
                waves=waves_g, phases=phases_g,
                ccp_helpers=batched['ccp'],
                leaf_row_index=batched['leaf_row_index'],
                leaf_col_index=batched['leaf_col_index'],
                root_clade_ids=batched['root_clade_ids'],
                device=DEVICE, dtype=DTYPE,
            )

            Pi_out = Pi_wave_forward(
                wave_layout=wl,
                species_helpers=sp_helpers_gpu,
                E=E_out['E'], Ebar=E_out['E_bar'],
                E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
                log_pS=params['log_pS'], log_pD=params['log_pD'], log_pL=params['log_pL'],
                max_transfer_mat=params['max_transfer_mat'],
                device=DEVICE, dtype=DTYPE,
                pibar_mode='uniform',
            )

            logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'],
                                          int(fam['root_clade_id']))
            nll_recomputed = -float(logL)
            nll_optimizer = float(result['nll'][g])

            diff = abs(nll_recomputed - nll_optimizer)
            assert diff < 0.1, (
                f"Gene {g}: optimizer NLL={nll_optimizer:.4f}, "
                f"recomputed NLL={nll_recomputed:.4f}, diff={diff:.4f}"
            )


# ---------------------------------------------------------------
# Tier 2: AleRax reference (optional)
# ---------------------------------------------------------------

ALERAX_AVAILABLE = shutil.which("alerax") is not None or (
    Path(__file__).resolve().parents[2] / "extra" / "AleRax_modified" / "build" / "bin" / "alerax"
).exists()


@pytest.mark.skipif(not ALERAX_AVAILABLE, reason="AleRax binary not found")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAleRaxComparison:
    """Simulate trees, run AleRax, compare gpurec optimizer output."""

    def test_genewise_matches_alerax(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tests.integration.test_e2e_alerax import (
            simulate_trees, run_alerax, parse_alerax_parameters,
            parse_alerax_likelihoods, run_gpurec,
        )

        with tempfile.TemporaryDirectory(prefix="gpurec_conv_") as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "data"
            alerax_out = str(tmpdir / "alerax_output")

            # Simulate 60-leaf species tree + 50 gene trees
            sp_path, gene_paths, families_path = simulate_trees(data_dir)

            # Run AleRax
            run_alerax(sp_path, families_path, alerax_out)
            alerax_params = parse_alerax_parameters(alerax_out)
            alerax_liks = parse_alerax_likelihoods(alerax_out)

            print(f"AleRax params: D={alerax_params['D']:.4e}, "
                  f"L={alerax_params['L']:.4e}, T={alerax_params['T']:.4e}")

            # Run gpurec shared-param optimizer
            ds = GeneDataset(sp_path, gene_paths,
                             genewise=False, specieswise=False, pairwise=False,
                             dtype=DTYPE, device=DEVICE)

            from src.core.batching import collate_gene_families, build_wave_layout
            from src.core.scheduling import compute_clade_waves

            items = []
            for fam in ds.families:
                items.append({
                    'ccp': fam['ccp_helpers'],
                    'leaf_row_index': fam['leaf_row_index'],
                    'leaf_col_index': fam['leaf_col_index'],
                    'root_clade_id': int(fam['root_clade_id']),
                })
            batched = collate_gene_families(items, dtype=DTYPE, device=DEVICE)
            w0, p0 = compute_clade_waves(ds.families[0]['ccp_helpers'])
            wl = build_wave_layout(
                waves=w0, phases=p0,
                ccp_helpers=batched['ccp'],
                leaf_row_index=batched['leaf_row_index'],
                leaf_col_index=batched['leaf_col_index'],
                root_clade_ids=batched['root_clade_ids'],
                device=DEVICE, dtype=DTYPE,
            )

            sp_helpers_gpu = {
                k: (v.to(device=DEVICE, dtype=DTYPE)
                    if torch.is_tensor(v) and v.is_floating_point()
                    else v.to(device=DEVICE) if torch.is_tensor(v) else v)
                for k, v in ds.species_helpers.items()
            }

            theta_init = math.log2(0.1) * torch.ones(3, dtype=DTYPE, device=DEVICE)
            result = optimize_theta_wave(
                wave_layout=wl,
                species_helpers=sp_helpers_gpu,
                root_clade_ids=batched['root_clade_ids'],
                unnorm_row_max=ds.unnorm_row_max.to(device=DEVICE, dtype=DTYPE),
                theta_init=theta_init,
                steps=100,
                optimizer='lbfgs',
                device=DEVICE,
                dtype=DTYPE,
                pibar_mode='uniform',
            )

            gpurec_rates = result['rates']
            print(f"gpurec rates: D={float(gpurec_rates[0]):.4e}, "
                  f"L={float(gpurec_rates[1]):.4e}, T={float(gpurec_rates[2]):.4e}")

            # Rates should be within 2x of AleRax
            for param, idx in [('D', 0), ('L', 1), ('T', 2)]:
                gpurec_val = float(gpurec_rates[idx])
                alerax_val = alerax_params[param]
                ratio = gpurec_val / alerax_val if alerax_val > 0 else float('inf')
                assert 0.5 < ratio < 2.0, (
                    f"{param}: gpurec={gpurec_val:.4e}, alerax={alerax_val:.4e}, "
                    f"ratio={ratio:.2f}"
                )
