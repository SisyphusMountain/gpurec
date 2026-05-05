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
        "ancestors_dense": sr["ancestors_dense"].to(dtype=dtype, device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }
    S = sh["S"]
    ancestors_T = sh["ancestors_dense"].T.to_sparse_coo()
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
        ancestors_T=ancestors_T,
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
        'ancestors_T': ancestors_T,
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


def _species_parent_from_children(sp_child1, sp_child2, S):
    parent = torch.full((S,), -1, device=sp_child1.device, dtype=torch.long)
    species = torch.arange(S, device=sp_child1.device, dtype=torch.long)
    c1 = sp_child1.long()
    c2 = sp_child2.long()
    mask1 = c1 < S
    mask2 = c2 < S
    parent[c1[mask1]] = species[mask1]
    parent[c2[mask2]] = species[mask2]
    return parent.contiguous()


def _balanced_self_loop_topology(device, dtype):
    """Small binary species tree with internal nodes that expose subtree bugs."""
    S = 7
    sp_child1 = torch.tensor([1, 3, 5, S, S, S, S], device=device, dtype=torch.long)
    sp_child2 = torch.tensor([2, 4, 6, S, S, S, S], device=device, dtype=torch.long)
    sp_parent = _species_parent_from_children(sp_child1, sp_child2, S)

    ancestors_dense = torch.zeros((S, S), device=device, dtype=dtype)
    for desc in range(S):
        cur = desc
        while cur >= 0:
            ancestors_dense[desc, cur] = 1.0
            cur = int(sp_parent[cur].item())
    ancestors_T = ancestors_dense.T.contiguous().to_sparse_coo()

    level_parents = torch.tensor(
        [[1, 2], [0, -1]], device=device, dtype=torch.long,
    )
    compact_level_ptr = torch.tensor([0, 2, 3], device=device, dtype=torch.long)
    compact_level_parents = torch.tensor([1, 2, 0], device=device, dtype=torch.int32)
    compact_level_child1 = torch.tensor([3, 5, 1], device=device, dtype=torch.int32)
    compact_level_child2 = torch.tensor([4, 6, 2], device=device, dtype=torch.int32)

    return {
        "S": S,
        "sp_child1": sp_child1,
        "sp_child2": sp_child2,
        "sp_parent": sp_parent,
        "ancestors_dense": ancestors_dense,
        "ancestors_T": ancestors_T,
        "level_parents": level_parents,
        "compact_level_ptr": compact_level_ptr,
        "compact_level_parents": compact_level_parents,
        "compact_level_child1": compact_level_child1,
        "compact_level_child2": compact_level_child2,
    }


def _subtree_sums_from_children(values, sp_child1, sp_child2):
    """Reference subtree sums in species-id order for [W, S] values."""
    result = values.clone()
    S = values.shape[1]

    def visit(node):
        c1 = int(sp_child1[node].item())
        c2 = int(sp_child2[node].item())
        if c1 < S:
            result[:, node] += visit(c1)
        if c2 < S:
            result[:, node] += visit(c2)
        return result[:, node].clone()

    has_parent = torch.zeros(S, dtype=torch.bool, device=values.device)
    for child in torch.cat([sp_child1.long(), sp_child2.long()]):
        if int(child.item()) < S:
            has_parent[int(child.item())] = True
    for root in torch.nonzero(~has_parent, as_tuple=False).flatten().tolist():
        visit(int(root))
    return result


def _uniform_pibar_from_pi(Pi_star, mt, ancestors_dense):
    row_max = Pi_star.max(dim=1, keepdim=True).values
    p_prime = torch.exp2(Pi_star - row_max)
    ancestor_sum = p_prime @ ancestors_dense.T
    denom = p_prime.sum(dim=1, keepdim=True) - ancestor_sum
    Pibar_star = torch.where(
        denom > 0,
        torch.log2(denom) + row_max + mt.unsqueeze(0),
        torch.full_like(Pi_star, NEG_INF),
    )
    return Pibar_star.contiguous(), row_max.squeeze(1).contiguous()


def _synthetic_self_loop_case(device, dtype, *, W=3):
    torch.manual_seed(20260505)
    topo = _balanced_self_loop_topology(device, dtype)
    S = topo["S"]

    Pi_star = (torch.randn(W, S, device=device, dtype=dtype) * 0.25 - 2.0).contiguous()
    mt = torch.linspace(-4.25, -3.75, S, device=device, dtype=dtype).contiguous()
    Pibar_star, pibar_row_max = _uniform_pibar_from_pi(
        Pi_star, mt, topo["ancestors_dense"],
    )
    rhs = (torch.randn(W, S, device=device, dtype=dtype) * 0.01).contiguous()

    E = (torch.randn(S, device=device, dtype=dtype) * 0.02 - 0.55).contiguous()
    Ebar = (torch.randn(S, device=device, dtype=dtype) * 0.02 - 0.75).contiguous()
    DL_const = (torch.randn(S, device=device, dtype=dtype) * 0.02 - 4.0).contiguous()
    SL1_const = (torch.randn(S, device=device, dtype=dtype) * 0.02 - 5.0).contiguous()
    SL2_const = (torch.randn(S, device=device, dtype=dtype) * 0.02 - 5.1).contiguous()
    log_pS = torch.linspace(-5.2, -4.8, S, device=device, dtype=dtype).contiguous()

    leaf_species_idx = torch.tensor([3, 5, -1], device=device, dtype=torch.int32)
    if W != 3:
        leaf_species_idx = torch.full((W,), -1, device=device, dtype=torch.int32)
        leaf_species_idx[:min(W, 2)] = torch.tensor(
            [3, 5][:min(W, 2)], device=device, dtype=torch.int32,
        )
    leaf_wt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    for w, s in enumerate(leaf_species_idx.tolist()):
        if 0 <= s < S:
            leaf_wt[w, s] = log_pS[s]

    topo.update({
        "Pi_star": Pi_star,
        "Pibar_star": Pibar_star,
        "pibar_row_max": pibar_row_max,
        "rhs": rhs,
        "mt": mt,
        "DL_const": DL_const,
        "Ebar": Ebar,
        "E": E,
        "SL1_const": SL1_const,
        "SL2_const": SL2_const,
        "leaf_wt": leaf_wt.contiguous(),
        "leaf_species_idx": leaf_species_idx.contiguous(),
        "leaf_logp": log_pS,
    })
    return topo


def _accum_from_per_element_refs(refs):
    _v, aw0, aw1, aw2, aw345, aw3, aw4 = refs
    return (
        aw0.sum().reshape(1).contiguous(),
        aw345.sum().reshape(1).contiguous(),
        (aw0 + aw2).sum(dim=0).contiguous(),
        aw1.sum(dim=0).contiguous(),
        aw4.sum(dim=0).contiguous(),
        aw3.sum(dim=0).contiguous(),
        aw2.sum(dim=0).contiguous(),
    )


def _compute_wave_dts_r(d, meta, S, device, dtype):
    if not meta['has_splits']:
        return None
    from gpurec.core.forward import _compute_dts_cross
    return _compute_dts_cross(
        d['Pi_star_wave'], d['Pibar_star_wave'], meta,
        d['sp_child1'], d['sp_child2'],
        d['log_pD'], d['log_pS'], S, device, dtype,
    )


def _zero_accum_param_grads(S, device, dtype):
    return (
        torch.zeros(1, device=device, dtype=dtype),
        torch.zeros(1, device=device, dtype=dtype),
        torch.zeros(S, device=device, dtype=dtype),
        torch.zeros(S, device=device, dtype=dtype),
        torch.zeros(S, device=device, dtype=dtype),
        torch.zeros(S, device=device, dtype=dtype),
        torch.zeros(S, device=device, dtype=dtype),
    )


def _run_wave_scatter_or_gather(d, wave_idx, rhs, *, use_gather, monkeypatch):
    wl = d['wave_layout']
    meta = wl['wave_metas'][wave_idx]
    ws, we, W, S = meta['start'], meta['end'], meta['W'], d['S']
    device, dtype = d['device'], d['dtype']
    dts_r = _compute_wave_dts_r(d, meta, S, device, dtype)

    DL_const = 1.0 + d['log_pD'] + d['E']
    SL1_const = d['log_pS'] + d['E_s2']
    SL2_const = d['log_pS'] + d['E_s1']
    mt = d['max_transfer_mat'].squeeze(-1) if d['max_transfer_mat'].ndim > 1 else d['max_transfer_mat']
    leaf_wt = _get_leaf_wt(wl, ws, we, S, d['log_pS'], device, dtype)
    accum = _zero_accum_param_grads(S, device, dtype)
    sp_parent = _species_parent_from_children(d['sp_child1'], d['sp_child2'], S)

    monkeypatch.setenv("GPUREC_WAVE_SPEC_GATHER", "1" if use_gather else "0")
    v_k, *_ = wave_backward_uniform_fused(
        d['Pi_star_wave'].to(dtype=dtype).contiguous(),
        d['Pibar_star_wave'].to(dtype=dtype).contiguous(),
        ws, W, S,
        dts_r.to(dtype=dtype).contiguous() if dts_r is not None else None,
        rhs.to(dtype=dtype).clone().contiguous(),
        mt.to(dtype=dtype).contiguous(),
        DL_const.to(dtype=dtype).contiguous(),
        d['Ebar'].to(dtype=dtype).contiguous(),
        d['E'].to(dtype=dtype).contiguous(),
        SL1_const.to(dtype=dtype).contiguous(),
        SL2_const.to(dtype=dtype).contiguous(),
        d['sp_child1'].long().contiguous(),
        d['sp_child2'].long().contiguous(),
        leaf_wt.to(dtype=dtype).contiguous(),
        neumann_terms=3,
        accum_param_grads=accum,
        sp_parent=sp_parent,
    )
    return v_k, accum


def _pytorch_single_wave_backward(
    Pi_star, Pibar_star, ws, W, S,
    dts_r, rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    sp_child1, sp_child2, leaf_wt,
    ancestors_T,
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
        pibar_mode='uniform', transfer_mat_T=None, ancestors_T=ancestors_T,
    )

    # Neumann series
    v_k = rhs.clone()
    term = rhs
    for _ in range(neumann_terms):
        term = _self_loop_Jt_apply(
            term, ingredients, sp_child1, sp_child2, S, W,
            pibar_mode='uniform', transfer_mat_T=None, ancestors_T=ancestors_T,
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
                d['ancestors_T'],
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wave_speciation_gather_matches_scatter(monkeypatch):
    """The exact scatter path is used even if the old gather flag is set."""
    d = _setup("test_trees_100", n_families=1, dtype=torch.float64)
    wl = d['wave_layout']
    wave_indices = [0]
    for k in range(len(wl['wave_metas']) - 1, -1, -1):
        if wl['wave_metas'][k]['has_splits']:
            wave_indices.append(k)
            break

    for wave_idx in wave_indices:
        meta = wl['wave_metas'][wave_idx]
        W, S = meta['W'], d['S']
        torch.manual_seed(8800 + wave_idx)
        rhs = torch.randn(W, S, device=d['device'], dtype=d['dtype']) * 0.01

        v_scatter, accum_scatter = _run_wave_scatter_or_gather(
            d, wave_idx, rhs, use_gather=False, monkeypatch=monkeypatch
        )
        v_gather, accum_gather = _run_wave_scatter_or_gather(
            d, wave_idx, rhs, use_gather=True, monkeypatch=monkeypatch
        )

        torch.testing.assert_close(v_gather, v_scatter, rtol=1e-10, atol=1e-10)
        for got, ref in zip(accum_gather, accum_scatter):
            torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wave_backward_skip_inactive_zero_stores_keeps_stale_rows_masked(monkeypatch):
    """Inactive v_k rows may stay stale when every consumer receives the mask."""
    monkeypatch.setenv("GPUREC_WAVE_SPEC_GATHER", "0")
    device = torch.device("cuda")
    dtype = torch.float32
    W, S = 3, 8
    ws = 0
    torch.manual_seed(909)

    Pi_star = (torch.randn(W, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    Pibar_star = (torch.randn(W, S, device=device, dtype=dtype) * 0.2 - 2.0).contiguous()
    rhs = (torch.randn(W, S, device=device, dtype=dtype) * 0.01).contiguous()
    active_mask = torch.tensor([True, False, True], device=device)
    mt = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 4.0).contiguous()
    E = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 1.0).contiguous()
    Ebar = (torch.randn(S, device=device, dtype=dtype) * 0.01 - 1.0).contiguous()
    DL_const = (1.0 - 4.0 + E).contiguous()
    SL1_const = (-5.0 + E).contiguous()
    SL2_const = (-5.0 + Ebar).contiguous()
    leaf_wt = torch.full((W, S), NEG_INF, device=device, dtype=dtype)
    sp_child1 = torch.tensor([1, 3, 5, S, S, S, S, S], device=device)
    sp_child2 = torch.tensor([2, 4, 6, S, S, S, S, S], device=device)

    def _scratch(fill):
        return {
            name: torch.full((W, S), fill, device=device, dtype=dtype)
            for name in (
                "v_k", "aw0", "aw1", "aw2", "aw345", "aw3", "aw4",
                "spec_buf", "term_buf",
            )
        }

    accum_zero = _zero_accum_param_grads(S, device, dtype)
    v_zero, *_ = wave_backward_uniform_fused(
        Pi_star, Pibar_star, ws, W, S, None, rhs,
        mt, DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2, leaf_wt,
        neumann_terms=3,
        accum_param_grads=accum_zero,
        active_mask=active_mask,
        skip_inactive_zero_stores=False,
        scratch=_scratch(37.0),
    )

    sentinel = 123.0
    accum_skip = _zero_accum_param_grads(S, device, dtype)
    v_skip, *_ = wave_backward_uniform_fused(
        Pi_star, Pibar_star, ws, W, S, None, rhs,
        mt, DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2, leaf_wt,
        neumann_terms=3,
        accum_param_grads=accum_skip,
        active_mask=active_mask,
        skip_inactive_zero_stores=True,
        scratch=_scratch(sentinel),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(v_skip[active_mask], v_zero[active_mask], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(v_zero[~active_mask], torch.zeros_like(v_zero[~active_mask]))
    torch.testing.assert_close(
        v_skip[~active_mask],
        torch.full_like(v_skip[~active_mask], sentinel),
    )
    for got, ref in zip(accum_skip, accum_zero):
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wave_backward_uniform_fused_supports_fp64_synthetic():
    """Exercise the fused self-loop kernel in fp64 on an exact small reference."""
    device = torch.device("cuda")
    dtype = torch.float64
    W, S = 3, 8
    ws = 0
    torch.manual_seed(1234)

    Pi_star = torch.randn(W, S, device=device, dtype=dtype).contiguous()
    Pibar_star = torch.randn(W, S, device=device, dtype=dtype).contiguous()
    dts_r = torch.randn(W, S, device=device, dtype=dtype).contiguous()
    rhs = torch.randn(W, S, device=device, dtype=dtype) * 0.01

    mt = torch.zeros(S, device=device, dtype=dtype)
    DL_const = torch.randn(S, device=device, dtype=dtype) * 0.1
    Ebar = torch.randn(S, device=device, dtype=dtype) * 0.1
    E = torch.randn(S, device=device, dtype=dtype) * 0.1
    SL1_const = torch.randn(S, device=device, dtype=dtype) * 0.1
    SL2_const = torch.randn(S, device=device, dtype=dtype) * 0.1
    leaf_wt = torch.randn(W, S, device=device, dtype=dtype) * 0.1

    sp_child1 = torch.tensor([1, 3, 5, S, S, S, S, S], device=device, dtype=torch.long)
    sp_child2 = torch.tensor([2, 4, 6, S, S, S, S, S], device=device, dtype=torch.long)
    sp_parent = _species_parent_from_children(sp_child1, sp_child2, S)
    ancestors_T_dense = torch.zeros((S, S), device=device, dtype=dtype)
    for desc in range(S):
        cur = desc
        while cur >= 0:
            ancestors_T_dense[cur, desc] = 1.0
            cur = int(sp_parent[cur].item())
    ancestors_T = ancestors_T_dense.to_sparse_coo()
    pi_max = Pi_star.max(dim=1, keepdim=True).values
    p_prime = torch.exp2(Pi_star - pi_max)
    denom = p_prime.sum(dim=1, keepdim=True) - p_prime @ ancestors_T_dense
    Pibar_star = (torch.log2(denom) + pi_max + mt.unsqueeze(0)).contiguous()

    for neumann_terms in (3, 20):
        refs = _pytorch_single_wave_backward(
            Pi_star, Pibar_star, ws, W, S, dts_r, rhs,
            mt, DL_const, Ebar, E, SL1_const, SL2_const,
            sp_child1, sp_child2, leaf_wt, ancestors_T,
            neumann_terms=neumann_terms,
        )
        tris = wave_backward_uniform_fused(
            Pi_star, Pibar_star, ws, W, S, dts_r, rhs.clone(),
            mt, DL_const, Ebar, E, SL1_const, SL2_const,
            sp_child1, sp_child2, leaf_wt,
            neumann_terms=neumann_terms,
            sp_parent=sp_parent,
        )

        for ref, tri in zip(refs, tris):
            torch.testing.assert_close(tri, ref, rtol=1e-10, atol=1e-10)


def test_wave_backward_uniform_fused_supports_bf16_synthetic():
    """Exercise the bf16 fused self-loop path against an fp32 reference."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    device = torch.device("cuda")
    W, S = 3, 8
    ws = 0
    torch.manual_seed(5678)

    ref_dtype = torch.float32
    Pi_star = (torch.randn(W, S, device=device, dtype=ref_dtype) * 0.1).contiguous()
    Pibar_star = (torch.randn(W, S, device=device, dtype=ref_dtype) * 0.1).contiguous()
    dts_r = (torch.randn(W, S, device=device, dtype=ref_dtype) * 0.1).contiguous()
    rhs = (torch.randn(W, S, device=device, dtype=ref_dtype) * 0.01).contiguous()

    mt = torch.zeros(S, device=device, dtype=ref_dtype)
    DL_const = torch.randn(S, device=device, dtype=ref_dtype) * 0.05
    Ebar = torch.randn(S, device=device, dtype=ref_dtype) * 0.05
    E = torch.randn(S, device=device, dtype=ref_dtype) * 0.05
    SL1_const = torch.randn(S, device=device, dtype=ref_dtype) * 0.05
    SL2_const = torch.randn(S, device=device, dtype=ref_dtype) * 0.05
    leaf_wt = torch.randn(W, S, device=device, dtype=ref_dtype) * 0.05

    sp_child1 = torch.tensor([1, 3, 5, S, S, S, S, S], device=device, dtype=torch.long)
    sp_child2 = torch.tensor([2, 4, 6, S, S, S, S, S], device=device, dtype=torch.long)
    sp_parent = _species_parent_from_children(sp_child1, sp_child2, S)
    ancestors_T_dense = torch.zeros((S, S), device=device, dtype=ref_dtype)
    for desc in range(S):
        cur = desc
        while cur >= 0:
            ancestors_T_dense[cur, desc] = 1.0
            cur = int(sp_parent[cur].item())
    ancestors_T = ancestors_T_dense.to_sparse_coo()
    pi_max = Pi_star.max(dim=1, keepdim=True).values
    p_prime = torch.exp2(Pi_star - pi_max)
    denom = p_prime.sum(dim=1, keepdim=True) - p_prime @ ancestors_T_dense
    Pibar_star = (torch.log2(denom) + pi_max + mt.unsqueeze(0)).contiguous()

    refs = _pytorch_single_wave_backward(
        Pi_star, Pibar_star, ws, W, S, dts_r, rhs,
        mt, DL_const, Ebar, E, SL1_const, SL2_const,
        sp_child1, sp_child2, leaf_wt, ancestors_T,
        neumann_terms=3,
    )
    bf16 = torch.bfloat16
    tris = wave_backward_uniform_fused(
        Pi_star.to(bf16).contiguous(),
        Pibar_star.to(bf16).contiguous(),
        ws, W, S,
        dts_r.to(bf16).contiguous(),
        rhs.to(bf16).contiguous(),
        mt.to(bf16).contiguous(),
        DL_const.to(bf16).contiguous(),
        Ebar.to(bf16).contiguous(),
        E.to(bf16).contiguous(),
        SL1_const.to(bf16).contiguous(),
        SL2_const.to(bf16).contiguous(),
        sp_child1,
        sp_child2,
        leaf_wt.to(bf16).contiguous(),
        neumann_terms=3,
        sp_parent=sp_parent,
    )

    for tri in tris:
        assert tri.dtype == torch.bfloat16
        assert torch.isfinite(tri.float()).all()
    for ref, tri in zip(refs, tris):
        torch.testing.assert_close(tri.float(), ref.float(), rtol=6e-2, atol=6e-3)


def test_uniform_pibar_jt_correction_is_subtree_sum_not_self_only():
    """The uniform-Pibar VJP correction is a descendant subtree sum."""
    device = torch.device("cpu")
    dtype = torch.float64
    topo = _balanced_self_loop_topology(device, dtype)
    W, S = 2, topo["S"]

    p_prime = torch.linspace(0.2, 1.4, W * S, device=device, dtype=dtype).reshape(W, S)
    pibar_coeff = torch.linspace(1.1, 2.4, W * S, device=device, dtype=dtype).reshape(W, S)
    v = torch.linspace(-0.05, 0.08, W * S, device=device, dtype=dtype).reshape(W, S)
    u_d = v * pibar_coeff

    ingredients = {
        "w_L": torch.ones(W, S, device=device, dtype=dtype),
        "w_terms": torch.zeros(6, W, S, device=device, dtype=dtype),
        "p_prime": p_prime,
        "pibar_inv_denom": pibar_coeff,
    }
    ingredients["w_terms"][2] = 1.0

    actual = _self_loop_Jt_apply(
        v, ingredients,
        topo["sp_child1"], topo["sp_child2"], S, W,
        pibar_mode="uniform", transfer_mat_T=None, ancestors_T=topo["ancestors_T"],
    )
    subtree = _subtree_sums_from_children(u_d, topo["sp_child1"], topo["sp_child2"])
    expected = p_prime * (u_d.sum(dim=1, keepdim=True) - subtree)
    self_only = p_prime * (u_d.sum(dim=1, keepdim=True) - u_d)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert (expected - self_only).abs()[:, :3].max().item() > 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "case_name,env",
    [
        ("proposal0_2d_triton", {
            "GPUREC_SELF_LOOP_2D_TRITON": "1",
            "GPUREC_SELF_LOOP_2D_BLOCK_W": "1",
        }),
        ("proposal1_tree_staged", {
            "GPUREC_SELF_LOOP_TREE_STAGED": "1",
            "GPUREC_SELF_LOOP_TREE_TILE_W": "2",
            "GPUREC_SELF_LOOP_TREE_TILE_S": "128",
        }),
        ("proposal2_tree_transposed", {
            "GPUREC_SELF_LOOP_TREE_TRANSPOSED": "1",
            "GPUREC_SELF_LOOP_TREE_TILE_W": "16",
        }),
        ("proposal3_hybrid_router", {
            "GPUREC_SELF_LOOP_TREE_STAGED": "1",
            "GPUREC_SELF_LOOP_TREE_MIN_W": "0",
            "GPUREC_SELF_LOOP_TREE_SPLIT_WAVES": "1",
        }),
    ],
)
def test_documented_self_loop_prototype_flags_match_synthetic_reference(
    monkeypatch,
    case_name,
    env,
):
    """Documented self-loop prototype flags preserve the synthetic reference."""
    device = torch.device("cuda")
    dtype = torch.float32
    d = _synthetic_self_loop_case(device, dtype)
    S = d["S"]

    refs = _pytorch_single_wave_backward(
        d["Pi_star"], d["Pibar_star"], 0, d["Pi_star"].shape[0], S,
        None, d["rhs"],
        d["mt"], d["DL_const"], d["Ebar"], d["E"], d["SL1_const"], d["SL2_const"],
        d["sp_child1"], d["sp_child2"], d["leaf_wt"], d["ancestors_T"],
        neumann_terms=4,
    )

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    tris = wave_backward_uniform_fused(
        d["Pi_star"], d["Pibar_star"], 0, d["Pi_star"].shape[0], S,
        None, d["rhs"].clone(),
        d["mt"], d["DL_const"], d["Ebar"], d["E"], d["SL1_const"], d["SL2_const"],
        d["sp_child1"], d["sp_child2"], d["leaf_wt"],
        neumann_terms=4,
        sp_parent=d["sp_parent"],
        pibar_row_max=d["pibar_row_max"],
    )

    for ref, tri in zip(refs, tris):
        torch.testing.assert_close(
            tri, ref, rtol=2e-5, atol=2e-6,
            msg=f"{case_name} does not match synthetic self-loop reference",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_nosplit_tree_correction_matches_synthetic_reference():
    """Proposal 5 tree correction matches reference and differs from self-only."""
    pytest.importorskip("cuda.bindings")
    from gpurec.core.kernels.wave_backward_cuda import wave_backward_uniform_nosplit_cuda

    device = torch.device("cuda")
    dtype = torch.float32
    d = _synthetic_self_loop_case(device, dtype)
    S = d["S"]
    W = d["Pi_star"].shape[0]
    neumann_terms = 4

    refs = _pytorch_single_wave_backward(
        d["Pi_star"], d["Pibar_star"], 0, W, S,
        None, d["rhs"],
        d["mt"], d["DL_const"], d["Ebar"], d["E"], d["SL1_const"], d["SL2_const"],
        d["sp_child1"], d["sp_child2"], d["leaf_wt"], d["ancestors_T"],
        neumann_terms=neumann_terms,
    )
    expected_accum = _accum_from_per_element_refs(refs)

    accum_tree = _zero_accum_param_grads(S, device, dtype)
    v_tree, *_ = wave_backward_uniform_nosplit_cuda(
        d["Pi_star"], d["Pibar_star"], 0, W, S,
        d["rhs"].clone(),
        d["mt"], d["DL_const"], d["Ebar"], d["E"], d["SL1_const"], d["SL2_const"],
        d["sp_child1"].to(torch.int32).contiguous(),
        d["sp_child2"].to(torch.int32).contiguous(),
        d["sp_parent"].to(torch.int32).contiguous(),
        d["leaf_species_idx"],
        d["leaf_logp"],
        d["pibar_row_max"],
        d["compact_level_ptr"],
        d["compact_level_parents"],
        d["compact_level_child1"],
        d["compact_level_child2"],
        accum_tree,
        neumann_terms=neumann_terms,
        correction_mode="tree",
    )

    accum_self = _zero_accum_param_grads(S, device, dtype)
    v_self, *_ = wave_backward_uniform_nosplit_cuda(
        d["Pi_star"], d["Pibar_star"], 0, W, S,
        d["rhs"].clone(),
        d["mt"], d["DL_const"], d["Ebar"], d["E"], d["SL1_const"], d["SL2_const"],
        d["sp_child1"].to(torch.int32).contiguous(),
        d["sp_child2"].to(torch.int32).contiguous(),
        d["sp_parent"].to(torch.int32).contiguous(),
        d["leaf_species_idx"],
        d["leaf_logp"],
        d["pibar_row_max"],
        d["compact_level_ptr"],
        d["compact_level_parents"],
        d["compact_level_child1"],
        d["compact_level_child2"],
        accum_self,
        neumann_terms=neumann_terms,
        correction_mode="self",
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(v_tree, refs[0], rtol=2e-4, atol=2e-6)
    for got, ref in zip(accum_tree, expected_accum):
        torch.testing.assert_close(got, ref, rtol=2e-4, atol=2e-6)
    assert (v_self - v_tree).abs().max().item() > 1e-6


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
                d['sp_child1'], d['sp_child2'], leaf_wt, d['ancestors_T'], 3,
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
                d['sp_child1'], d['sp_child2'], leaf_wt, d['ancestors_T'], 3,
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
            assert rel < 1e-2, f"Large-S root {name}: rel={rel:.2e}"
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
            ancestors_T=d['ancestors_T'],
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
            ancestors_T=d['ancestors_T'],
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
            ancestors_T=d['ancestors_T'],
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
