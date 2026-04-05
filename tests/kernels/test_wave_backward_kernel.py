"""Test fused Triton backward kernel against PyTorch analytical path.

Compares wave_backward_uniform_fused (Triton) vs
_self_loop_vjp_precompute + _self_loop_Jt_apply + param VJP (PyTorch)
on real data, per wave.
"""

import math
from pathlib import Path

import pytest
import torch

from gpurec.core.preprocess_cpp import _load_extension
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.likelihood import E_fixed_point
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.backward import _self_loop_vjp_precompute, _self_loop_Jt_apply
from gpurec.core._helpers import _safe_exp2_ratio
from gpurec.core.log2_utils import logsumexp2, logaddexp2
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import (
    collate_gene_families,
    collate_wave,
    build_wave_layout,
)
from gpurec.core.kernels.wave_backward import wave_backward_uniform_fused

_INV = 1.0 / math.log(2.0)
_ROOT = Path(__file__).resolve().parent.parent
NEG_INF = float('-inf')
D, L, T = 0.05, 0.05, 0.05


def _setup(ds_name, n_families=1, dtype=torch.float32):
    """Set up a single-family test problem."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ext = _load_extension()
    data_dir = _ROOT / "data" / ds_name
    if not data_dir.exists():
        pytest.skip(f"{ds_name} not found")
    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))[:n_families]
    if len(gene_paths) < n_families:
        pytest.skip(f"Only {len(gene_paths)} families")

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
    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))
    tm_unnorm = torch.log2(sh["Recipients_mat"])
    unnorm_row_max = tm_unnorm.max(dim=-1).values.to(device=device, dtype=dtype)

    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
        theta, unnorm_row_max, specieswise=False,
    )

    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-8, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode='uniform',
    )

    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    ccp_helpers = batched['ccp']

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
        leaf_row_index=batched['leaf_row_index'],
        leaf_col_index=batched['leaf_col_index'],
        root_clade_ids=batched['root_clade_ids'],
        device=device, dtype=dtype,
    )

    Pi_out = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sh,
        E=E_out['E'], Ebar=E_out['E_bar'], E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        device=device, dtype=dtype, pibar_mode='uniform',
    )

    # Species child lookup
    sp_P_idx = sh['s_P_indexes'].cpu().long()
    sp_c12_idx = sh['s_C12_indexes'].cpu().long()
    mask_c1 = sp_P_idx < S
    sp_child1_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child2_cpu = torch.full((S,), S, dtype=torch.long)
    sp_child1_cpu[sp_P_idx[mask_c1]] = sp_c12_idx[mask_c1]
    sp_child2_cpu[sp_P_idx[~mask_c1] - S] = sp_c12_idx[~mask_c1]
    sp_child1 = sp_child1_cpu.to(device)
    sp_child2 = sp_child2_cpu.to(device)

    return {
        'wave_layout': wave_layout,
        'Pi_star_wave': Pi_out['Pi_wave_ordered'],
        'Pibar_star_wave': Pi_out['Pibar_wave_ordered'],
        'species_helpers': sh,
        'E': E_out['E'], 'E_s1': E_out['E_s1'], 'E_s2': E_out['E_s2'],
        'Ebar': E_out['E_bar'],
        'log_pS': log_pS, 'log_pD': log_pD, 'log_pL': log_pL,
        'max_transfer_mat': mt,
        'sp_child1': sp_child1, 'sp_child2': sp_child2,
        'device': device, 'dtype': dtype, 'S': S,
    }


def _get_leaf_wt(wave_layout, ws, we, S, log_pS, device, dtype):
    """Compute leaf term [W, S] for a wave."""
    W = we - ws
    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']
    lwt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    mask = (leaf_row_index >= ws) & (leaf_row_index < we)
    if mask.any():
        lwt[leaf_row_index[mask] - ws, leaf_col_index[mask]] = 0.0
    return log_pS + lwt


def _pytorch_single_wave_backward(
    Pi_star, Pibar_star, ws, W, S,
    dts_r, rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_wt,
    neumann_terms,
):
    """Run the PyTorch analytical path for a single wave.

    Returns v_k and per-element param contributions.
    """
    Pi_W = Pi_star[ws:ws+W]
    Pibar_W = Pibar_star[ws:ws+W]

    ingredients = _self_loop_vjp_precompute(
        Pi_W, Pibar_W, dts_r,
        mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2, leaf_wt, S,
        pibar_mode='uniform', transfer_mat_T=None, ancestors_T=None,
    )

    # Neumann series
    v_k = rhs.clone()
    term = rhs
    for _ in range(neumann_terms):
        term = _self_loop_Jt_apply(
            term, ingredients, sp_child1, sp_child2, S, W,
            pibar_mode='uniform', transfer_mat_T=None, ancestors_T=None,
        )
        v_k = v_k + term

    # Param VJP
    alpha_full = v_k * ingredients['w_L']
    wt = ingredients['w_terms']
    aw0 = alpha_full * wt[0]
    aw1 = alpha_full * wt[1]
    aw2 = alpha_full * wt[2]
    aw3 = alpha_full * wt[3]
    aw4 = alpha_full * wt[4]
    aw5 = alpha_full * wt[5]
    aw345 = aw3 + aw4 + aw5

    return v_k, aw0, aw1, aw2, aw345, aw3, aw4


@pytest.fixture(scope="class")
def setup_100():
    return _setup("test_trees_100", n_families=1, dtype=torch.float32)


@pytest.fixture(scope="class")
def setup_1000():
    return _setup("test_trees_1000", n_families=1, dtype=torch.float32)


class TestWaveBackwardKernel:
    """Compare Triton backward kernel vs PyTorch analytical for each wave."""

    @pytest.fixture(autouse=True)
    def _setup(self, setup_100):
        self.data = setup_100

    def _run_one_wave(self, wave_idx, neumann_terms=3):
        """Run both paths on a single wave and return outputs."""
        d = self.data
        wl = d['wave_layout']
        meta = wl['wave_metas'][wave_idx]
        ws = meta['start']
        we = meta['end']
        W = meta['W']
        S = d['S']
        device = d['device']
        dtype = d['dtype']

        # DTS cross (reuse forward's computation via _compute_dts_cross)
        if meta['has_splits']:
            from gpurec.core.forward import _compute_dts_cross
            dts_r = _compute_dts_cross(
                d['Pi_star_wave'], d['Pibar_star_wave'], meta,
                d['sp_child1'], d['sp_child2'],
                d['log_pD'], d['log_pS'], S, device, dtype,
            )
        else:
            dts_r = None

        DL_const = 1.0 + d['log_pD'] + d['E']
        SL1_const = d['log_pS'] + d['E_s2']
        SL2_const = d['log_pS'] + d['E_s1']
        mt = d['max_transfer_mat'].squeeze(-1) if d['max_transfer_mat'].ndim > 1 else d['max_transfer_mat']
        leaf_wt = _get_leaf_wt(wl, ws, we, S, d['log_pS'], device, dtype)

        # Random RHS (simulating adjoint from later waves)
        torch.manual_seed(42 + wave_idx)
        rhs = torch.randn(W, S, device=device, dtype=dtype) * 0.01

        # PyTorch reference
        v_k_ref, aw0_ref, aw1_ref, aw2_ref, aw345_ref, aw3_ref, aw4_ref = \
            _pytorch_single_wave_backward(
                d['Pi_star_wave'], d['Pibar_star_wave'], ws, W, S,
                dts_r, rhs,
                mt, DL_const, d['Ebar'], d['E'], SL1_const, SL2_const,
                d['sp_child1'], d['sp_child2'], leaf_wt,
                neumann_terms,
            )

        # Triton kernel (needs fp32, contiguous data)
        rhs_triton = rhs.clone()  # kernel overwrites rhs as scratch
        v_k_tri, aw0_tri, aw1_tri, aw2_tri, aw345_tri, aw3_tri, aw4_tri = \
            wave_backward_uniform_fused(
                d['Pi_star_wave'].float().contiguous(),
                d['Pibar_star_wave'].float().contiguous(),
                ws, W, S,
                dts_r.float().contiguous() if dts_r is not None else None,
                rhs_triton.float().contiguous(),
                mt.float().contiguous(),
                DL_const.float().contiguous(),
                d['Ebar'].float().contiguous(),
                d['E'].float().contiguous(),
                SL1_const.float().contiguous(),
                SL2_const.float().contiguous(),
                d['sp_child1'].long().contiguous(),
                d['sp_child2'].long().contiguous(),
                leaf_wt.float().contiguous(),
                neumann_terms=neumann_terms,
            )

        return {
            'ref': (v_k_ref.float(), aw0_ref.float(), aw1_ref.float(), aw2_ref.float(),
                    aw345_ref.float(), aw3_ref.float(), aw4_ref.float()),
            'tri': (v_k_tri, aw0_tri, aw1_tri, aw2_tri, aw345_tri, aw3_tri, aw4_tri),
            'W': W, 'S': S, 'has_splits': meta['has_splits'],
        }

    def test_leaf_wave(self):
        """Test first wave (leaves, no splits)."""
        result = self._run_one_wave(0)
        names = ['v_k', 'aw0', 'aw1', 'aw2', 'aw345', 'aw3', 'aw4']
        for i, name in enumerate(names):
            ref = result['ref'][i]
            tri = result['tri'][i]
            max_diff = (ref - tri).abs().max().item()
            rel = max_diff / (ref.abs().max().item() + 1e-12)
            assert rel < 1e-3, (
                f"Wave 0 (leaf) {name}: max_diff={max_diff:.2e}, "
                f"rel={rel:.2e}, ref_max={ref.abs().max():.2e}"
            )

    def test_internal_wave_with_splits(self):
        """Test an internal wave that has splits."""
        wl = self.data['wave_layout']
        # Find a wave with splits
        for k in range(len(wl['wave_metas']) - 1, -1, -1):
            if wl['wave_metas'][k]['has_splits']:
                result = self._run_one_wave(k)
                names = ['v_k', 'aw0', 'aw1', 'aw2', 'aw345', 'aw3', 'aw4']
                for i, name in enumerate(names):
                    ref = result['ref'][i]
                    tri = result['tri'][i]
                    max_diff = (ref - tri).abs().max().item()
                    ref_scale = ref.abs().max().item() + 1e-12
                    rel = max_diff / ref_scale
                    assert rel < 1e-3, (
                        f"Wave {k} (splits) {name}: max_diff={max_diff:.2e}, "
                        f"rel={rel:.2e}, ref_max={ref_scale:.2e}"
                    )
                return
        pytest.skip("No waves with splits found")

    def test_neumann_convergence(self):
        """More Neumann terms should not change result significantly (ρ(A) << 1)."""
        result_3 = self._run_one_wave(0, neumann_terms=3)
        result_6 = self._run_one_wave(0, neumann_terms=6)
        diff = (result_3['tri'][0] - result_6['tri'][0]).abs().max().item()
        ref_scale = result_3['tri'][0].abs().max().item() + 1e-12
        # Spectral radius varies by tree size (~0.04 for S=1999, ~0.3 for S=199).
        # With 3 extra terms, diff ~ ρ^4, so allow up to 5%.
        assert diff / ref_scale < 0.05, f"Neumann not converged: diff={diff:.2e}"

    def test_all_waves(self):
        """Test all waves in the backward sweep."""
        wl = self.data['wave_layout']
        K = len(wl['wave_metas'])
        names = ['v_k', 'aw0', 'aw1', 'aw2', 'aw345', 'aw3', 'aw4']
        max_rels = {n: 0.0 for n in names}

        for k in range(K):
            result = self._run_one_wave(k)
            for i, name in enumerate(names):
                ref = result['ref'][i]
                tri = result['tri'][i]
                max_diff = (ref - tri).abs().max().item()
                ref_scale = ref.abs().max().item() + 1e-12
                rel = max_diff / ref_scale
                max_rels[name] = max(max_rels[name], rel)
                assert rel < 5e-3, (
                    f"Wave {k} {name}: rel={rel:.2e}, max_diff={max_diff:.2e}"
                )

        print(f"\nMax relative errors across {K} waves:")
        for name, rel in max_rels.items():
            print(f"  {name}: {rel:.2e}")


class TestWaveBackwardKernelLargeS:
    """Test at S=1999 (test_trees_1000) — the production scale."""

    @pytest.fixture(autouse=True)
    def _setup(self, setup_1000):
        self.data = setup_1000

    def test_leaf_wave(self):
        """Test leaf wave at large S."""
        d = self.data
        wl = d['wave_layout']
        meta = wl['wave_metas'][0]
        ws, we, W, S = meta['start'], meta['end'], meta['W'], d['S']
        device, dtype = d['device'], d['dtype']

        DL_const = 1.0 + d['log_pD'] + d['E']
        SL1_const = d['log_pS'] + d['E_s2']
        SL2_const = d['log_pS'] + d['E_s1']
        mt = d['max_transfer_mat'].squeeze(-1) if d['max_transfer_mat'].ndim > 1 else d['max_transfer_mat']
        leaf_wt = _get_leaf_wt(wl, ws, we, S, d['log_pS'], device, dtype)

        torch.manual_seed(123)
        rhs = torch.randn(W, S, device=device, dtype=dtype) * 0.01

        v_ref, aw0_ref, aw1_ref, aw2_ref, aw345_ref, aw3_ref, aw4_ref = \
            _pytorch_single_wave_backward(
                d['Pi_star_wave'], d['Pibar_star_wave'], ws, W, S,
                None, rhs, mt, DL_const, d['Ebar'], d['E'], SL1_const, SL2_const,
                d['sp_child1'], d['sp_child2'], leaf_wt, 3,
            )

        v_tri, *aw_tri = wave_backward_uniform_fused(
            d['Pi_star_wave'].float().contiguous(),
            d['Pibar_star_wave'].float().contiguous(),
            ws, W, S, None,
            rhs.float().clone().contiguous(),
            mt.float().contiguous(), DL_const.float().contiguous(),
            d['Ebar'].float().contiguous(), d['E'].float().contiguous(),
            SL1_const.float().contiguous(), SL2_const.float().contiguous(),
            d['sp_child1'].long().contiguous(), d['sp_child2'].long().contiguous(),
            leaf_wt.float().contiguous(), neumann_terms=3,
        )

        names = ['v_k', 'aw0', 'aw1', 'aw2', 'aw345', 'aw3', 'aw4']
        refs = [v_ref.float(), aw0_ref.float(), aw1_ref.float(), aw2_ref.float(),
                aw345_ref.float(), aw3_ref.float(), aw4_ref.float()]
        tris = [v_tri] + aw_tri
        for i, name in enumerate(names):
            rel = (refs[i] - tris[i]).abs().max().item() / (refs[i].abs().max().item() + 1e-12)
            assert rel < 1e-3, f"Large-S leaf {name}: rel={rel:.2e}"
            print(f"  {name}: rel={rel:.2e}")

    def test_root_wave(self):
        """Test root wave (last wave, has splits) at large S."""
        d = self.data
        wl = d['wave_layout']
        K = len(wl['wave_metas'])
        meta = wl['wave_metas'][K - 1]
        ws, we, W, S = meta['start'], meta['end'], meta['W'], d['S']
        device, dtype = d['device'], d['dtype']

        if meta['has_splits']:
            from gpurec.core.forward import _compute_dts_cross
            dts_r = _compute_dts_cross(
                d['Pi_star_wave'], d['Pibar_star_wave'], meta,
                d['sp_child1'], d['sp_child2'],
                d['log_pD'], d['log_pS'], S, device, dtype,
            )
        else:
            dts_r = None

        DL_const = 1.0 + d['log_pD'] + d['E']
        SL1_const = d['log_pS'] + d['E_s2']
        SL2_const = d['log_pS'] + d['E_s1']
        mt = d['max_transfer_mat'].squeeze(-1) if d['max_transfer_mat'].ndim > 1 else d['max_transfer_mat']
        leaf_wt = _get_leaf_wt(wl, ws, we, S, d['log_pS'], device, dtype)

        torch.manual_seed(456)
        rhs = torch.randn(W, S, device=device, dtype=dtype) * 0.01

        v_ref, aw0_ref, aw1_ref, aw2_ref, aw345_ref, aw3_ref, aw4_ref = \
            _pytorch_single_wave_backward(
                d['Pi_star_wave'], d['Pibar_star_wave'], ws, W, S,
                dts_r, rhs, mt, DL_const, d['Ebar'], d['E'], SL1_const, SL2_const,
                d['sp_child1'], d['sp_child2'], leaf_wt, 3,
            )

        v_tri, *aw_tri = wave_backward_uniform_fused(
            d['Pi_star_wave'].float().contiguous(),
            d['Pibar_star_wave'].float().contiguous(),
            ws, W, S,
            dts_r.float().contiguous() if dts_r is not None else None,
            rhs.float().clone().contiguous(),
            mt.float().contiguous(), DL_const.float().contiguous(),
            d['Ebar'].float().contiguous(), d['E'].float().contiguous(),
            SL1_const.float().contiguous(), SL2_const.float().contiguous(),
            d['sp_child1'].long().contiguous(), d['sp_child2'].long().contiguous(),
            leaf_wt.float().contiguous(), neumann_terms=3,
        )

        names = ['v_k', 'aw0', 'aw1', 'aw2', 'aw345', 'aw3', 'aw4']
        refs = [v_ref.float(), aw0_ref.float(), aw1_ref.float(), aw2_ref.float(),
                aw345_ref.float(), aw3_ref.float(), aw4_ref.float()]
        tris = [v_tri] + aw_tri
        for i, name in enumerate(names):
            rel = (refs[i] - tris[i]).abs().max().item() / (refs[i].abs().max().item() + 1e-12)
            assert rel < 1e-3, f"Large-S root {name}: rel={rel:.2e}"
            print(f"  {name}: rel={rel:.2e}")


class TestFusedBackwardE2EFiniteDifference:
    """End-to-end FD test that exercises the fused Triton kernel via Pi_wave_backward.

    Uses fp32 + S=1999 + uniform → fused kernel is activated.
    Compares analytical gradient against central finite differences.
    """

    @pytest.fixture(scope="class")
    def setup_fp32_1000(self):
        """fp32 setup at S=1999 — this hits the fused kernel path."""
        return _setup("test_trees_1000", n_families=1, dtype=torch.float32)

    def test_grad_log_pD_vs_fd(self, setup_fp32_1000):
        d = setup_fp32_1000
        from gpurec.core.forward import Pi_wave_forward
        from gpurec.core.backward import Pi_wave_backward
        from gpurec.core.likelihood import compute_log_likelihood

        result = Pi_wave_backward(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_star_wave'],
            Pibar_star_wave=d['Pibar_star_wave'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=d['max_transfer_mat'],
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['wave_layout']['root_clade_ids'],
            device=d['device'], dtype=d['dtype'],
            neumann_terms=3, use_pruning=False,
            pibar_mode='uniform',
        )

        eps = 1e-2  # large eps for fp32 (logL~2349, need eps*grad >> fp32_noise)
        logL_p = self._forward_logL(d, log_pD=d['log_pD'] + eps)
        logL_m = self._forward_logL(d, log_pD=d['log_pD'] - eps)
        fd = (logL_p - logL_m) / (2 * eps)
        bw = result['grad_log_pD'].item()

        rel_err = abs(bw - fd) / (abs(fd) + 1e-30)
        print(f"  log_pD: FD={fd:.6e}, BW={bw:.6e}, rel_err={rel_err:.2e}")
        assert rel_err < 2e-2, f"log_pD gradient rel error {rel_err:.4e}"

    def test_grad_log_pS_vs_fd(self, setup_fp32_1000):
        d = setup_fp32_1000
        from gpurec.core.backward import Pi_wave_backward

        result = Pi_wave_backward(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_star_wave'],
            Pibar_star_wave=d['Pibar_star_wave'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=d['max_transfer_mat'],
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['wave_layout']['root_clade_ids'],
            device=d['device'], dtype=d['dtype'],
            neumann_terms=3, use_pruning=False,
            pibar_mode='uniform',
        )

        eps = 1e-2
        logL_p = self._forward_logL(d, log_pS=d['log_pS'] + eps)
        logL_m = self._forward_logL(d, log_pS=d['log_pS'] - eps)
        fd = (logL_p - logL_m) / (2 * eps)
        bw = result['grad_log_pS'].item()

        rel_err = abs(bw - fd) / (abs(fd) + 1e-30)
        print(f"  log_pS: FD={fd:.6e}, BW={bw:.6e}, rel_err={rel_err:.2e}")
        assert rel_err < 2e-2, f"log_pS gradient rel error {rel_err:.4e}"

    def test_grad_mt_vs_fd(self, setup_fp32_1000):
        d = setup_fp32_1000
        from gpurec.core.backward import Pi_wave_backward
        mt = d['max_transfer_mat']

        result = Pi_wave_backward(
            wave_layout=d['wave_layout'],
            Pi_star_wave=d['Pi_star_wave'],
            Pibar_star_wave=d['Pibar_star_wave'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=d['log_pS'], log_pD=d['log_pD'], log_pL=d['log_pL'],
            max_transfer_mat=mt,
            species_helpers=d['species_helpers'],
            root_clade_ids_perm=d['wave_layout']['root_clade_ids'],
            device=d['device'], dtype=d['dtype'],
            neumann_terms=3, use_pruning=False,
            pibar_mode='uniform',
        )

        # Use all-ones direction (large signal) instead of random (which partially cancels).
        direction = torch.ones_like(mt)
        direction = direction / direction.norm()

        eps = 1e-2
        logL_p = self._forward_logL(d, max_transfer_mat=mt + eps * direction)
        logL_m = self._forward_logL(d, max_transfer_mat=mt - eps * direction)
        fd = (logL_p - logL_m) / (2 * eps)
        bw = (result['grad_max_transfer_mat'] * direction).sum().item()

        rel_err = abs(bw - fd) / (abs(fd) + 1e-30)
        print(f"  mt dir: FD={fd:.6e}, BW={bw:.6e}, rel_err={rel_err:.2e}")
        assert rel_err < 2e-2, f"mt gradient rel error {rel_err:.4e}"

    @staticmethod
    def _forward_logL(d, log_pS=None, log_pD=None, max_transfer_mat=None):
        from gpurec.core.forward import Pi_wave_forward
        from gpurec.core.likelihood import compute_log_likelihood
        from gpurec.core.batching import build_wave_layout

        _log_pS = log_pS if log_pS is not None else d['log_pS']
        _log_pD = log_pD if log_pD is not None else d['log_pD']
        _mt = max_transfer_mat if max_transfer_mat is not None else d['max_transfer_mat']

        Pi_out = Pi_wave_forward(
            wave_layout=d['wave_layout'], species_helpers=d['species_helpers'],
            E=d['E'], Ebar=d['Ebar'], E_s1=d['E_s1'], E_s2=d['E_s2'],
            log_pS=_log_pS, log_pD=_log_pD, log_pL=d['log_pL'],
            transfer_mat=None, max_transfer_mat=_mt,
            device=d['device'], dtype=d['dtype'], pibar_mode='uniform',
        )
        # Need root clade IDs in original space for compute_log_likelihood
        perm = d['wave_layout']['perm']
        root_ids_perm = d['wave_layout']['root_clade_ids']
        logL = compute_log_likelihood(Pi_out['Pi_wave_ordered'], d['E'], root_ids_perm)
        return logL.sum().item()
