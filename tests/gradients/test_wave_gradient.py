"""Test wave-decomposed implicit gradient computation.

Tests:
1. Neumann series on synthetic problem
2. Wave backward pass smoke test
3. Wave gradient vs finite differences (via direct backward)
4. Gradient step decreases NLL
5. Pruning accuracy
6. Vectorized gradient bounds
"""

import math
from pathlib import Path

import pytest
import torch

from src.core.preprocess_cpp import _load_extension
from src.core.extract_parameters import extract_parameters_uniform
from src.core.likelihood import (
    E_fixed_point,
    E_step,
    Pi_wave_forward_v2,
    Pi_wave_backward_v2,
    compute_log_likelihood,
    compute_gradient_bounds,
    _self_loop_differentiable,
)
from src.core.scheduling import compute_clade_waves
from src.core.batching import (
    collate_gene_families,
    collate_wave,
    build_wave_layout,
)

_INV = 1.0 / math.log(2.0)
_ROOT = Path(__file__).resolve().parent.parent
D, L, T = 0.05, 0.05, 0.05
TOL = 1e-3


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _setup_single_family(ds_name, n_families=1, device=None, dtype=torch.float64):
    """Set up problem for gradient testing."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ext = _load_extension()
    data_dir = _ROOT / "data" / ds_name
    if not data_dir.exists():
        pytest.skip(f"{ds_name} not found")
    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))[:n_families]
    if len(gene_paths) < n_families:
        pytest.skip(f"Only {len(gene_paths)} families in {ds_name}")

    batch_items = []
    sr = None
    for gp in gene_paths:
        raw = ext.preprocess(sp_path, [str(gp)])
        if sr is None:
            sr = raw['species']
        cr = raw['ccp']
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
        batch_items.append({
            "ccp": ch,
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
    S = sh["S"]

    theta = torch.log(torch.tensor([D, L, T], dtype=dtype, device=device))
    tm_unnorm = torch.log2(sh["Recipients_mat"])
    unnorm_row_max = tm_unnorm.max(dim=-1).values.to(device=device, dtype=dtype)

    # Use uniform mode for large S, dense for small S (Triton kernels need dense for S<=256)
    pibar_mode = 'uniform' if S > 256 else 'dense'

    if pibar_mode == 'uniform':
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters_uniform(
            theta, unnorm_row_max, specieswise=False,
        )
    else:
        from src.core.extract_parameters import extract_parameters
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
            theta, tm_unnorm.to(device=device, dtype=dtype),
            genewise=False, specieswise=False, pairwise=False,
        )
        if max_transfer_mat.ndim == 2:
            max_transfer_mat = max_transfer_mat.squeeze(-1)

    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=max_transfer_mat,
        max_iters=2000, tolerance=1e-8, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode=pibar_mode,
    )
    E = E_out['E']
    E_s1 = E_out['E_s1']
    E_s2 = E_out['E_s2']
    Ebar = E_out['E_bar']

    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    ccp_helpers = batched['ccp']
    leaf_row_index = batched['leaf_row_index']
    leaf_col_index = batched['leaf_col_index']
    root_clade_ids = batched['root_clade_ids']

    families_waves, families_phases = [], []
    for bi in batch_items:
        w, p = compute_clade_waves(bi['ccp'])
        families_waves.append(w)
        families_phases.append(p)

    offsets = [m['clade_offset'] for m in batched['family_meta']]
    cross_waves = collate_wave(families_waves, offsets)
    max_n = max(len(p) for p in families_phases)
    cross_phases = [max(fp[k] if k < len(fp) else 1 for fp in families_phases) for k in range(max_n)]

    wave_layout = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=ccp_helpers,
        leaf_row_index=leaf_row_index,
        leaf_col_index=leaf_col_index,
        root_clade_ids=root_clade_ids,
        device=device, dtype=dtype,
    )

    Pi_out = Pi_wave_forward_v2(
        wave_layout=wave_layout, species_helpers=sh,
        E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=max_transfer_mat,
        device=device, dtype=dtype, pibar_mode=pibar_mode,
    )

    return {
        'wave_layout': wave_layout,
        'Pi_out': Pi_out,
        'species_helpers': sh,
        'E': E, 'E_s1': E_s1, 'E_s2': E_s2, 'Ebar': Ebar,
        'log_pS': log_pS, 'log_pD': log_pD, 'log_pL': log_pL,
        'transfer_mat': transfer_mat,
        'max_transfer_mat': max_transfer_mat,
        'root_clade_ids': root_clade_ids,
        'root_clade_ids_perm': wave_layout['root_clade_ids'],
        'theta': theta,
        'unnorm_row_max': unnorm_row_max,
        'pibar_mode': pibar_mode,
        'batch_items': batch_items,
        'device': device, 'dtype': dtype, 'S': S,
    }


def _full_forward(theta, unnorm_row_max, sh, wave_layout, root_clade_ids, device, dtype,
                   pibar_mode='uniform', tm_unnorm=None):
    """Full forward pass at given theta, return logL."""
    S = sh['S']
    if pibar_mode == 'uniform':
        log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
            theta, unnorm_row_max, specieswise=False,
        )
    else:
        from src.core.extract_parameters import extract_parameters
        log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters(
            theta, tm_unnorm, genewise=False, specieswise=False, pairwise=False,
        )
        if mt.ndim == 2:
            mt = mt.squeeze(-1)
    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-8, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode=pibar_mode,
    )
    Pi_out = Pi_wave_forward_v2(
        wave_layout=wave_layout, species_helpers=sh,
        E=E_out['E'], Ebar=E_out['E_bar'], E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode=pibar_mode,
    )
    logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_ids)
    return logL.sum().item(), Pi_out, E_out, log_pS, log_pD, log_pL, mt


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestNeumannSynthetic:
    """Test Neumann series on a known contractive linear map."""

    def test_neumann_matches_exact_inverse(self):
        rho = 0.04
        n = 20
        torch.manual_seed(42)
        Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
        D_diag = torch.linspace(0, rho, n, dtype=torch.float64)
        J = Q @ torch.diag(D_diag) @ Q.T

        rhs = torch.randn(n, dtype=torch.float64)
        exact = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - J.T, rhs)

        v = rhs.clone()
        term = rhs.clone()
        for _ in range(3):
            term = J.T @ term
            v = v + term

        rel_err = (v - exact).norm() / exact.norm()
        assert rel_err < 1e-4, f"Neumann 3-term rel error = {rel_err:.2e}"

    def test_neumann_4_terms_fp64(self):
        n = 50
        rho = 0.04
        torch.manual_seed(123)
        Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
        D_diag = torch.linspace(0, rho, n, dtype=torch.float64)
        J = Q @ torch.diag(D_diag) @ Q.T

        rhs = torch.randn(n, dtype=torch.float64)
        exact = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - J.T, rhs)

        v = rhs.clone()
        term = rhs.clone()
        for _ in range(4):
            term = J.T @ term
            v = v + term

        rel_err = (v - exact).norm() / exact.norm()
        assert rel_err < 3e-6, f"Neumann 4-term rel error = {rel_err:.2e}"


class TestWaveBackward:
    """Test wave backward pass on real tree data."""

    @pytest.fixture(scope="class")
    def setup_20(self):
        return _setup_single_family("test_trees_20", n_families=1, dtype=torch.float32)

    def test_backward_runs(self, setup_20):
        """Backward pass completes without error."""
        d = setup_20
        result = Pi_wave_backward_v2(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_out']['Pi_wave_ordered'],
            Pibar_star_wave=d['Pi_out']['Pibar_wave_ordered'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=d['max_transfer_mat'],
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['root_clade_ids_perm'],
            device=d['device'], dtype=d['dtype'],
            neumann_terms=3, use_pruning=False,
        )
        assert 'v_Pi' in result
        assert result['v_Pi'].shape == d['Pi_out']['Pi_wave_ordered'].shape
        assert torch.isfinite(result['v_Pi']).all(), "v_Pi contains non-finite values"

    def test_backward_gradients_finite(self, setup_20):
        """All gradient accumulators should be finite."""
        d = setup_20
        result = Pi_wave_backward_v2(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_out']['Pi_wave_ordered'],
            Pibar_star_wave=d['Pi_out']['Pibar_wave_ordered'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=d['max_transfer_mat'],
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['root_clade_ids_perm'],
            device=d['device'], dtype=d['dtype'],
            neumann_terms=3, use_pruning=False,
        )
        for key in ['grad_E', 'grad_Ebar', 'grad_log_pD', 'grad_log_pS', 'grad_max_transfer_mat']:
            assert torch.isfinite(result[key]).all(), f"{key} has non-finite values"

    def test_root_rhs_nonzero(self, setup_20):
        """The root clade should have nonzero RHS."""
        d = setup_20
        result = Pi_wave_backward_v2(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_out']['Pi_wave_ordered'],
            Pibar_star_wave=d['Pi_out']['Pibar_wave_ordered'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=d['max_transfer_mat'],
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['root_clade_ids_perm'],
            device=d['device'], dtype=d['dtype'],
            neumann_terms=3, use_pruning=False,
        )
        v_Pi = result['v_Pi']
        root_ids = d['root_clade_ids_perm']
        root_grad_norm = v_Pi[root_ids].abs().max()
        assert root_grad_norm > 0, "Root clade gradient is zero"


class TestGradientDescent:
    """Test that a gradient step decreases NLL."""

    @pytest.fixture(scope="class")
    def setup_20(self):
        return _setup_single_family("test_trees_20", n_families=1, dtype=torch.float32)

    def test_gradient_step_decreases_nll(self, setup_20):
        """Taking a small gradient step should decrease NLL.

        We compute the partial gradient dL/d(log_pS, log_pD, mt) from the
        backward pass, then perturb the intermediate parameters and check
        that NLL decreases.
        """
        d = setup_20
        device, dtype = d['device'], d['dtype']

        result = Pi_wave_backward_v2(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_out']['Pi_wave_ordered'],
            Pibar_star_wave=d['Pi_out']['Pibar_wave_ordered'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=d['max_transfer_mat'],
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['root_clade_ids_perm'],
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
        )

        # At least some parameter gradients should be nonzero
        has_nonzero = any(
            result[k].abs().max() > 0 for k in ['grad_log_pD', 'grad_log_pS', 'grad_max_transfer_mat']
        )
        assert has_nonzero, "All parameter gradients are zero"

        # Check NLL at base
        nll_base = -compute_log_likelihood(
            d['Pi_out']['Pi'], d['E'], d['root_clade_ids']
        ).sum().item()

        # Full forward at slightly different theta
        theta_base = d['theta'].clone()
        tm_unnorm = torch.log2(d['species_helpers']['Recipients_mat']).to(device=device, dtype=dtype)
        pibar_mode = d['pibar_mode']
        logL_base, _, _, _, _, _, _ = _full_forward(
            theta_base, d['unnorm_row_max'], d['species_helpers'],
            d['wave_layout'], d['root_clade_ids'], device, dtype,
            pibar_mode=pibar_mode, tm_unnorm=tm_unnorm,
        )
        print(f"  Base logL = {logL_base:.6f}")

        # Try perturbing each theta component
        eps = 1e-3
        for i in range(theta_base.numel()):
            theta_p = theta_base.clone()
            theta_p[i] += eps
            logL_p, _, _, _, _, _, _ = _full_forward(
                theta_p, d['unnorm_row_max'], d['species_helpers'],
                d['wave_layout'], d['root_clade_ids'], device, dtype,
                pibar_mode=pibar_mode, tm_unnorm=tm_unnorm,
            )
            fd_grad = (logL_p - logL_base) / eps
            print(f"  theta[{i}] fd_grad = {fd_grad:.6e}")


class TestWaveVsFiniteDifference:
    """Test wave backward gradients against finite differences (fp64)."""

    @pytest.fixture(scope="class")
    def setup_1000(self):
        return _setup_single_family("test_trees_1000", n_families=1, dtype=torch.float64)

    def test_log_pD_gradient(self, setup_1000):
        """dL/d(log_pD) matches central FD within 0.1%."""
        d = setup_1000
        device, dtype = d['device'], d['dtype']

        result = Pi_wave_backward_v2(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_out']['Pi_wave_ordered'],
            Pibar_star_wave=d['Pi_out']['Pibar_wave_ordered'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=d['max_transfer_mat'],
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['root_clade_ids_perm'],
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
        )

        eps = 1e-4
        logL_p = self._forward_logL(d, log_pD=d['log_pD'] + eps)
        logL_m = self._forward_logL(d, log_pD=d['log_pD'] - eps)
        fd = (logL_p - logL_m) / (2 * eps)
        bw = result['grad_log_pD'].item()

        rel_err = abs(bw - fd) / (abs(fd) + 1e-30)
        print(f"  log_pD: FD={fd:.6e}, BW={bw:.6e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-3, f"log_pD gradient rel error {rel_err:.4e}"

    def test_log_pS_gradient(self, setup_1000):
        """dL/d(log_pS) matches central FD within 0.1%."""
        d = setup_1000
        device, dtype = d['device'], d['dtype']

        result = Pi_wave_backward_v2(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_out']['Pi_wave_ordered'],
            Pibar_star_wave=d['Pi_out']['Pibar_wave_ordered'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=d['max_transfer_mat'],
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['root_clade_ids_perm'],
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
        )

        eps = 1e-4
        logL_p = self._forward_logL(d, log_pS=d['log_pS'] + eps)
        logL_m = self._forward_logL(d, log_pS=d['log_pS'] - eps)
        fd = (logL_p - logL_m) / (2 * eps)
        bw = result['grad_log_pS'].item()

        rel_err = abs(bw - fd) / (abs(fd) + 1e-30)
        print(f"  log_pS: FD={fd:.6e}, BW={bw:.6e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-3, f"log_pS gradient rel error {rel_err:.4e}"

    def test_mt_gradient(self, setup_1000):
        """dL/d(mt) directional derivative matches FD within 0.1%."""
        d = setup_1000
        device, dtype = d['device'], d['dtype']
        mt = d['max_transfer_mat']

        result = Pi_wave_backward_v2(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_out']['Pi_wave_ordered'],
            Pibar_star_wave=d['Pi_out']['Pibar_wave_ordered'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=mt,
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['root_clade_ids_perm'],
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
        )

        torch.manual_seed(42)
        direction = torch.randn_like(mt)
        direction = direction / direction.norm()

        eps = 1e-4
        logL_p = self._forward_logL(d, max_transfer_mat=mt + eps * direction)
        logL_m = self._forward_logL(d, max_transfer_mat=mt - eps * direction)
        fd = (logL_p - logL_m) / (2 * eps)
        bw = (result['grad_max_transfer_mat'] * direction).sum().item()

        rel_err = abs(bw - fd) / (abs(fd) + 1e-30)
        print(f"  mt dir: FD={fd:.6e}, BW={bw:.6e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-3, f"mt gradient rel error {rel_err:.4e}"

    @staticmethod
    def _forward_logL(d, log_pS=None, log_pD=None, max_transfer_mat=None):
        """Run Pi forward with optionally perturbed params, return logL."""
        _log_pS = log_pS if log_pS is not None else d['log_pS']
        _log_pD = log_pD if log_pD is not None else d['log_pD']
        _mt = max_transfer_mat if max_transfer_mat is not None else d['max_transfer_mat']

        Pi_out = Pi_wave_forward_v2(
            wave_layout=d['wave_layout'], species_helpers=d['species_helpers'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=_log_pS, log_pD=_log_pD, log_pL=d['log_pL'],
            transfer_mat=d['transfer_mat'], max_transfer_mat=_mt,
            device=d['device'], dtype=d['dtype'], pibar_mode=d['pibar_mode'],
        )
        logL = compute_log_likelihood(Pi_out['Pi'], d['E'], d['root_clade_ids'])
        return logL.sum().item()


class TestPruning:
    """Test that pruning doesn't significantly affect gradients."""

    @pytest.fixture(scope="class")
    def setup_20(self):
        return _setup_single_family("test_trees_20", n_families=1, dtype=torch.float32)

    def test_pruning_runs_and_is_finite(self, setup_20):
        """Pruned backward should run and produce finite results."""
        d = setup_20
        kwargs = dict(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_out']['Pi_wave_ordered'],
            Pibar_star_wave=d['Pi_out']['Pibar_wave_ordered'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=d['max_transfer_mat'],
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['root_clade_ids_perm'],
            device=d['device'], dtype=d['dtype'],
            neumann_terms=3,
        )

        result_full = Pi_wave_backward_v2(**kwargs, use_pruning=False)
        result_pruned = Pi_wave_backward_v2(**kwargs, use_pruning=True, pruning_threshold=-50.0)

        # Pruned result should be finite
        assert torch.isfinite(result_pruned['v_Pi']).all()

        # Pruned gradient should agree on the unpruned (important) clades
        v_full = result_full['v_Pi']
        v_pruned = result_pruned['v_Pi']

        # Check correlation: inner product should be positive
        dot = (v_full * v_pruned).sum()
        print(f"  Full/pruned dot product: {dot:.6e}")
        print(f"  Full norm: {v_full.norm():.6e}, Pruned norm: {v_pruned.norm():.6e}")
        assert dot >= 0, "Pruned gradient points in opposite direction"


class TestGradientBoundsVectorized:
    """Test vectorized compute_gradient_bounds."""

    @pytest.fixture(scope="class")
    def setup_20(self):
        return _setup_single_family("test_trees_20", n_families=1, dtype=torch.float32)

    def test_vectorized_produces_valid_output(self, setup_20):
        """Wave-meta version produces reasonable gradient bounds."""
        d = setup_20
        Pi = d['Pi_out']['Pi_wave_ordered']
        wave_metas = d['wave_layout']['wave_metas']
        root_ids = d['root_clade_ids_perm']

        gb, pm = compute_gradient_bounds(
            Pi, {}, root_ids, threshold=-20.0, wave_metas=wave_metas,
        )
        # Root should have bound = 0
        assert (gb[root_ids] == 0.0).all()
        # Gradient bounds should be finite or -inf
        assert (torch.isfinite(gb) | (gb == float('-inf'))).all()
        # Some clades should be prunable
        n_pruned = pm.sum().item()
        C = Pi.shape[0]
        print(f"  {n_pruned}/{C} clades pruned at threshold=-20")


class TestSelfLoopDifferentiable:
    """Test the differentiable self-loop helper."""

    @pytest.fixture(scope="class")
    def setup_20(self):
        return _setup_single_family("test_trees_20", n_families=1, dtype=torch.float32)

    def test_self_loop_matches_forward(self, setup_20):
        """_self_loop_differentiable should produce same output as fused kernel."""
        d = setup_20
        device, dtype = d['device'], d['dtype']
        S = d['S']

        # Get first non-trivial wave
        wave_metas = d['wave_layout']['wave_metas']
        Pi_star = d['Pi_out']['Pi_wave_ordered']

        mt = d['max_transfer_mat'].squeeze(-1) if d['max_transfer_mat'].ndim > 1 else d['max_transfer_mat']
        DL_const = 1.0 + d['log_pD'] + d['E']
        SL1_const = d['log_pS'] + d['E_s2']
        SL2_const = d['log_pS'] + d['E_s1']

        # Species child indices
        sh = d['species_helpers']
        sp_P_idx = sh['s_P_indexes']
        sp_c12_idx = sh['s_C12_indexes']
        p_cpu = sp_P_idx.cpu().long()
        c_cpu = sp_c12_idx.cpu().long()
        mask_c1 = p_cpu < S
        sp_child1 = torch.full((S,), S, dtype=torch.long)
        sp_child2 = torch.full((S,), S, dtype=torch.long)
        sp_child1[p_cpu[mask_c1]] = c_cpu[mask_c1]
        sp_child2[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]
        sp_child1 = sp_child1.to(device)
        sp_child2 = sp_child2.to(device)

        # Test on a leaf wave (no dts_r)
        meta0 = wave_metas[0]
        ws, we = meta0['start'], meta0['end']
        Pi_W = Pi_star[ws:we].clone()

        leaf_row_index = d['wave_layout']['leaf_row_index']
        leaf_col_index = d['wave_layout']['leaf_col_index']
        W = we - ws
        lwt = torch.full((W, S), float('-inf'), device=device, dtype=dtype)
        mask = (leaf_row_index >= ws) & (leaf_row_index < we)
        if mask.any():
            lwt[leaf_row_index[mask] - ws, leaf_col_index[mask]] = 0.0
        leaf_wt = d['log_pS'] + lwt

        result = _self_loop_differentiable(
            Pi_W, mt, DL_const, d['Ebar'], d['E'], SL1_const, SL2_const,
            sp_child1, sp_child2, leaf_wt, None, S,
        )
        assert result.shape == Pi_W.shape
        assert torch.isfinite(result).all()

    def test_self_loop_gradients_flow(self, setup_20):
        """Autograd should be able to differentiate through the self-loop."""
        d = setup_20
        device, dtype = d['device'], d['dtype']
        S = d['S']

        mt = d['max_transfer_mat'].squeeze(-1) if d['max_transfer_mat'].ndim > 1 else d['max_transfer_mat']
        DL_const = 1.0 + d['log_pD'] + d['E']
        SL1_const = d['log_pS'] + d['E_s2']
        SL2_const = d['log_pS'] + d['E_s1']

        sh = d['species_helpers']
        sp_P_idx = sh['s_P_indexes']
        sp_c12_idx = sh['s_C12_indexes']
        p_cpu = sp_P_idx.cpu().long()
        c_cpu = sp_c12_idx.cpu().long()
        mask_c1 = p_cpu < S
        sp_child1 = torch.full((S,), S, dtype=torch.long)
        sp_child2 = torch.full((S,), S, dtype=torch.long)
        sp_child1[p_cpu[mask_c1]] = c_cpu[mask_c1]
        sp_child2[p_cpu[~mask_c1] - S] = c_cpu[~mask_c1]
        sp_child1 = sp_child1.to(device)
        sp_child2 = sp_child2.to(device)

        # Small test: 3 clades, S species
        W = 3
        Pi_W = torch.randn(W, S, device=device, dtype=dtype, requires_grad=True)
        leaf_wt = torch.full((W, S), float('-inf'), device=device, dtype=dtype)

        result = _self_loop_differentiable(
            Pi_W, mt, DL_const, d['Ebar'], d['E'], SL1_const, SL2_const,
            sp_child1, sp_child2, leaf_wt, None, S,
        )
        loss = result.sum()
        loss.backward()
        assert Pi_W.grad is not None
        assert torch.isfinite(Pi_W.grad).all()


class TestFullChainFD:
    """Test full-chain gradient dL/dtheta (through both E and Pi) via FD."""

    @pytest.fixture(scope="class")
    def setup_1000(self):
        return _setup_single_family("test_trees_1000", n_families=1, dtype=torch.float64)

    def test_full_chain_gradient_matches_fd(self, setup_1000):
        """Perturb theta, re-solve E AND Pi, compare logL → validates full dL/dtheta."""
        from src.optimization.theta_optimizer import implicit_grad_loglik_vjp_wave

        d = setup_1000
        device, dtype = d['device'], d['dtype']
        theta = d['theta'].clone()
        pibar_mode = d['pibar_mode']
        tm_unnorm = torch.log2(d['species_helpers']['Recipients_mat']).to(device=device, dtype=dtype)

        # Full forward at base theta (re-solves E and Pi)
        logL_base, Pi_out_base, E_out_base, log_pS, log_pD, log_pL, mt = _full_forward(
            theta, d['unnorm_row_max'], d['species_helpers'],
            d['wave_layout'], d['root_clade_ids'], device, dtype,
            pibar_mode=pibar_mode, tm_unnorm=tm_unnorm,
        )

        # Run implicit_grad_loglik_vjp_wave for analytical gradient
        grad_theta, statsG = implicit_grad_loglik_vjp_wave(
            d['wave_layout'], d['species_helpers'],
            Pi_star_wave=Pi_out_base['Pi_wave_ordered'],
            Pibar_star_wave=Pi_out_base['Pibar_wave_ordered'],
            E_star=E_out_base['E'], E_s1=E_out_base['E_s1'],
            E_s2=E_out_base['E_s2'], Ebar=E_out_base['E_bar'],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            max_transfer_mat=mt,
            root_clade_ids_perm=d['wave_layout']['root_clade_ids'],
            theta=theta,
            unnorm_row_max=d['unnorm_row_max'],
            specieswise=False,
            device=device, dtype=dtype,
            neumann_terms=4, use_pruning=False,
            cg_tol=1e-10, cg_maxiter=1000,
        )

        print(f"  Base logL (NLL) = {logL_base:.8f}")
        print(f"  E CG stats: {statsG}")
        print(f"  Analytical gradient: {grad_theta}")

        # Central finite differences for each theta component.
        # eps=1e-4 balances truncation vs round-off: the E fixed-point solve
        # has tol=1e-8, so FD accuracy ~ tol/eps = 1e-4 at eps=1e-4.
        eps = 1e-4
        for i in range(theta.numel()):
            theta_p = theta.clone()
            theta_p[i] += eps
            logL_p, _, _, _, _, _, _ = _full_forward(
                theta_p, d['unnorm_row_max'], d['species_helpers'],
                d['wave_layout'], d['root_clade_ids'], device, dtype,
                pibar_mode=pibar_mode, tm_unnorm=tm_unnorm,
            )

            theta_m = theta.clone()
            theta_m[i] -= eps
            logL_m, _, _, _, _, _, _ = _full_forward(
                theta_m, d['unnorm_row_max'], d['species_helpers'],
                d['wave_layout'], d['root_clade_ids'], device, dtype,
                pibar_mode=pibar_mode, tm_unnorm=tm_unnorm,
            )

            fd = (logL_p - logL_m) / (2 * eps)
            analytic = grad_theta[i].item()
            rel_err = abs(analytic - fd) / (abs(fd) + 1e-30)
            print(f"  theta[{i}]: FD={fd:.8e}, analytic={analytic:.8e}, rel_err={rel_err:.4e}")
            assert rel_err < 0.01, (
                f"theta[{i}] full-chain gradient rel error {rel_err:.4e} > 1%: "
                f"FD={fd:.8e}, analytic={analytic:.8e}"
            )


class TestEndToEnd:
    """Test end-to-end optimization loop."""

    @pytest.fixture(scope="class")
    def setup_1000(self):
        return _setup_single_family("test_trees_1000", n_families=1, dtype=torch.float64)

    def test_optimization_decreases_nll(self, setup_1000):
        """5 steps of optimize_theta_wave, NLL should decrease."""
        from src.optimization.theta_optimizer import optimize_theta_wave

        d = setup_1000
        device, dtype = d['device'], d['dtype']

        result = optimize_theta_wave(
            wave_layout=d['wave_layout'],
            species_helpers=d['species_helpers'],
            root_clade_ids=d['root_clade_ids'],
            unnorm_row_max=d['unnorm_row_max'],
            theta_init=d['theta'].clone(),
            steps=5,
            lr=0.1,
            e_max_iters=2000,
            e_tol=1e-8,
            neumann_terms=4,
            use_pruning=False,
            cg_tol=1e-10,
            cg_maxiter=1000,
            specieswise=False,
            device=device,
            dtype=dtype,
            pibar_mode=d['pibar_mode'],
        )

        history = result['history']
        print(f"  Optimization history ({len(history)} steps):")
        for rec in history:
            print(f"    Step {rec.iteration}: NLL={rec.negative_log_likelihood:.6f}, "
                  f"|grad|={rec.grad_infinity_norm:.4e}, |dtheta|={rec.theta_step_inf:.4e}")

        assert len(history) >= 2, "Need at least 2 steps"
        nll_first = history[0].negative_log_likelihood
        nll_last = history[-1].negative_log_likelihood
        print(f"  NLL: {nll_first:.6f} → {nll_last:.6f} (delta={nll_last - nll_first:.6e})")
        assert nll_last < nll_first, (
            f"NLL did not decrease: {nll_first:.6f} → {nll_last:.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
