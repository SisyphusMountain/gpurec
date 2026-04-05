"""Diagnostic script to reproduce and trace the NaN NLL bug in optimize_theta_genewise.

Usage:
    python experiments/debug_nan_genewise.py

Three passes:
  Pass 1: Batched run (families 0-4). Detects which family goes NaN and at which step.
           Saves state to /tmp/nan_debug_state.pt at first NaN detection.
  Pass 2: Isolated run of the NaN family. Verifies it converges normally.
  Pass 3: Replay: re-run the NaN family in isolation using the exact theta from the
           batched run's step BEFORE the NaN to reproduce it, with anomaly detection.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import copy

import torch

# Ensure the gpurec package is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpurec.core.model import GeneDataset
from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.backward import Pi_wave_backward
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.batching import collate_gene_families, build_wave_layout, collate_wave
from gpurec.core.scheduling import compute_clade_waves
from gpurec.optimization.genewise_optimizer import optimize_theta_genewise
from gpurec.optimization.implicit_grad import _e_adjoint_and_theta_vjp

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "test_trees_100"
STATE_PATH = Path("/tmp/nan_debug_state.pt")

N_FAMILIES = 5
MAX_STEPS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
_THETA_MIN = math.log2(1e-10)


def load_dataset(family_indices):
    sp = str(DATA_DIR / "sp.nwk")
    all_genes = sorted(DATA_DIR.glob("g_*.nwk"))
    genes = [str(all_genes[i]) for i in family_indices]
    ds = GeneDataset(
        sp, genes, genewise=True, specieswise=False, pairwise=False,
        dtype=DTYPE, device=DEVICE,
    )
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Instrumented optimizer (thin wrapper that hooks _nll_and_grad)
# ──────────────────────────────────────────────────────────────────────────────

def run_batched_instrumented(ds, theta_init, max_steps=MAX_STEPS):
    """Run optimize_theta_genewise but intercept each step's NLL.

    Returns the full result plus a `nan_info` dict if NaN was detected, else None.
    """
    # We monkey-patch the inner _nll_and_grad by wrapping optimize_theta_genewise
    # internals through a custom per-step callback.
    # Since _nll_and_grad is a closure, we duplicate the outer loop logic here.

    import math as _math
    from gpurec.optimization.genewise_optimizer import _lbfgs_two_loop

    families = ds.families
    species_helpers = ds.species_helpers
    unnorm_row_max = ds.unnorm_row_max
    pibar_mode = 'uniform'
    specieswise = False
    lbfgs_m = 10
    grad_tol = 1e-5
    e_max_iters = 2000
    e_tol = 1e-8
    neumann_terms = 3
    pruning_threshold = 1e-6
    cg_tol = 1e-8
    cg_maxiter = 500
    local_iters = 2000
    local_tolerance = 1e-3

    device = DEVICE
    dtype = DTYPE

    G = len(families)
    unnorm_row_max = unnorm_row_max.to(device=device, dtype=dtype)

    _skip_keys = {'Recipients_mat'}
    def _move_tensor(t):
        if t.dtype.is_floating_point:
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)
    sh = {
        k: (_move_tensor(v) if torch.is_tensor(v) and k not in _skip_keys else v)
        for k, v in species_helpers.items()
    }

    # Build wave layouts
    wave_layouts = []
    root_clade_ids_list = []
    per_family_waves = []
    per_family_phases = []
    batch_items = []
    for g in range(G):
        fam = families[g]
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
        wave_layouts.append(wl_g)
        root_clade_ids_list.append(single_batched['root_clade_ids'])
        per_family_waves.append(waves_g)
        per_family_phases.append(phases_g)
        batch_items.append(single_item)

    all_batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    family_meta = all_batched['family_meta']
    offsets = [m['clade_offset'] for m in family_meta]
    sizes = [m['C'] for m in family_meta]
    cross_waves = collate_wave(per_family_waves, offsets)
    cross_phases = [max(per_family_phases[g][k] for g in range(G) if k < len(per_family_phases[g]))
                    for k in range(len(cross_waves))]
    merged_layout = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=all_batched['ccp'],
        leaf_row_index=all_batched['leaf_row_index'],
        leaf_col_index=all_batched['leaf_col_index'],
        root_clade_ids=all_batched['root_clade_ids'],
        device=device, dtype=dtype,
        family_clade_counts=sizes,
        family_clade_offsets=offsets,
    )
    C_total = int(all_batched['ccp']['C'])
    S = sh['S']

    anc_dense = sh['ancestors_dense'].to(device=device, dtype=dtype)
    _ancestors_T = anc_dense.T.to_sparse_coo()

    def _eval_E(theta_t, warm_E):
        log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(
            theta_t, unnorm_row_max, specieswise=specieswise, genewise=True,
        )
        E_out = E_fixed_point(
            species_helpers=sh,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            max_iters=e_max_iters, tolerance=e_tol,
            warm_start_E=warm_E,
            dtype=dtype, device=device, pibar_mode=pibar_mode,
            ancestors_T=_ancestors_T,
        )
        return log_pS, log_pD, log_pL, mt, E_out

    def _nll_and_grad(theta_t, warm_E, active_mask):
        log_pS, log_pD, log_pL, mt, E_out = _eval_E(theta_t, warm_E)

        nll = torch.full((G,), float('nan'), device=device, dtype=dtype)
        grad = torch.zeros_like(theta_t)
        active_genes = active_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        Pi_orig = {}
        Pibar_orig = {}
        for g in active_genes:
            E_g = E_out['E'][g]
            Ebar_g = E_out['E_bar'][g]
            E_s1_g = E_out['E_s1'][g]
            E_s2_g = E_out['E_s2'][g]
            mt_g = mt[g]
            pS_g = log_pS[g] if log_pS.ndim >= 1 else log_pS
            pD_g = log_pD[g] if log_pD.ndim >= 1 else log_pD
            pL_g = log_pL[g] if log_pL.ndim >= 1 else log_pL

            Pi_out_g = Pi_wave_forward(
                wave_layout=wave_layouts[g], species_helpers=sh,
                E=E_g, Ebar=Ebar_g, E_s1=E_s1_g, E_s2=E_s2_g,
                log_pS=pS_g, log_pD=pD_g, log_pL=pL_g,
                transfer_mat=None, max_transfer_mat=mt_g,
                device=device, dtype=dtype,
                local_iters=local_iters, local_tolerance=local_tolerance,
                pibar_mode=pibar_mode,
            )
            logL_g = compute_log_likelihood(Pi_out_g['Pi'], E_g, root_clade_ids_list[g])
            nll[g] = logL_g.sum()

            perm_g = wave_layouts[g]['perm']
            Pi_orig[g] = Pi_out_g['Pi_wave_ordered'][perm_g]
            Pibar_orig[g] = Pi_out_g['Pibar_wave_ordered'][perm_g]

        merged_perm = merged_layout['perm']
        Pi_star_merged = torch.full((C_total, S), float('-inf'), device=device, dtype=dtype)
        Pibar_star_merged = torch.full((C_total, S), float('-inf'), device=device, dtype=dtype)
        for g in active_genes:
            o, c = offsets[g], sizes[g]
            Pi_star_merged[merged_perm[o:o+c]] = Pi_orig[g]
            Pibar_star_merged[merged_perm[o:o+c]] = Pibar_orig[g]

        pi_bwd = Pi_wave_backward(
            wave_layout=merged_layout,
            Pi_star_wave=Pi_star_merged,
            Pibar_star_wave=Pibar_star_merged,
            E=E_out['E'], Ebar=E_out['E_bar'],
            E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            max_transfer_mat=mt,
            species_helpers=sh,
            root_clade_ids_perm=merged_layout['root_clade_ids'],
            device=device, dtype=dtype,
            neumann_terms=neumann_terms,
            use_pruning=True,
            pruning_threshold=pruning_threshold,
            pibar_mode=pibar_mode,
            ancestors_T=_ancestors_T,
            family_idx=merged_layout['family_idx'],
        )

        for g in active_genes:
            E_g = E_out['E'][g]
            Ebar_g = E_out['E_bar'][g]
            mt_g = mt[g]
            pS_g = log_pS[g] if log_pS.ndim >= 1 else log_pS
            pD_g = log_pD[g] if log_pD.ndim >= 1 else log_pD
            pL_g = log_pL[g] if log_pL.ndim >= 1 else log_pL

            pi_bwd_g = {
                'grad_E': pi_bwd['grad_E'][g],
                'grad_Ebar': pi_bwd['grad_Ebar'][g],
                'grad_E_s1': pi_bwd['grad_E_s1'][g],
                'grad_E_s2': pi_bwd['grad_E_s2'][g],
                'grad_log_pD': pi_bwd['grad_log_pD'][g],
                'grad_log_pS': pi_bwd['grad_log_pS'][g],
                'grad_max_transfer_mat': pi_bwd['grad_max_transfer_mat'][g],
            }

            grad_theta_g, _ = _e_adjoint_and_theta_vjp(
                pi_bwd_g, E_g, Ebar_g, E_out['E_s1'][g], E_out['E_s2'][g],
                pS_g, pD_g, pL_g, mt_g,
                sh, wave_layouts[g]['root_clade_ids'],
                theta_t[g], unnorm_row_max, specieswise,
                device, dtype,
                cg_tol=cg_tol, cg_maxiter=cg_maxiter,
                pibar_mode=pibar_mode,
                ancestors_T=_ancestors_T,
            )
            grad[g] = grad_theta_g

        return nll, grad, E_out

    def _nll_only(theta_t, warm_E, active_mask):
        log_pS, log_pD, log_pL, mt, E_out = _eval_E(theta_t, warm_E)
        nll = torch.full((G,), float('nan'), device=device, dtype=dtype)
        for g in active_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            E_g = E_out['E'][g]
            mt_g = mt[g]
            pS_g = log_pS[g] if log_pS.ndim >= 1 else log_pS
            pD_g = log_pD[g] if log_pD.ndim >= 1 else log_pD
            pL_g = log_pL[g] if log_pL.ndim >= 1 else log_pL
            Pi_out_g = Pi_wave_forward(
                wave_layout=wave_layouts[g], species_helpers=sh,
                E=E_g, Ebar=E_out['E_bar'][g],
                E_s1=E_out['E_s1'][g], E_s2=E_out['E_s2'][g],
                log_pS=pS_g, log_pD=pD_g, log_pL=pL_g,
                transfer_mat=None, max_transfer_mat=mt_g,
                device=device, dtype=dtype,
                local_iters=local_iters, local_tolerance=local_tolerance,
                pibar_mode=pibar_mode,
            )
            logL_g = compute_log_likelihood(Pi_out_g['Pi'], E_g, root_clade_ids_list[g])
            nll[g] = logL_g.sum()
        return nll, E_out

    # --- L-BFGS loop ---
    theta = theta_init.to(device=device, dtype=dtype).clone()
    active = torch.ones(G, dtype=torch.bool, device=device)
    nll, grad, E_out = _nll_and_grad(theta, None, active)

    theta_shape = theta.shape
    flat_P = theta[0].numel()
    S_hist = torch.zeros(G, lbfgs_m, flat_P, device=device, dtype=dtype)
    Y_hist = torch.zeros(G, lbfgs_m, flat_P, device=device, dtype=dtype)
    history_len = 0
    history = [{'step': 0, 'nll': nll.detach().cpu().clone(),
                 'grad_inf': float(grad.abs().max().item()), 'n_active': int(active.sum().item())}]

    nan_info = None
    prev_theta = theta.clone()
    prev_E = E_out['E'].clone()

    for step in range(1, max_steps + 1):
        # --- NaN check ---
        nan_mask = nll.isnan()
        if nan_mask.any():
            nan_genes = nan_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
            print(f"\n[PASS 1] NaN detected at step={step} for family local_idx={nan_genes}")
            print(f"         (global family indices: {nan_genes})")
            for g in nan_genes:
                print(f"  Family {g}: theta={theta[g].cpu().tolist()}")
            # Save state: theta at current (NaN) step AND theta/E from previous step
            torch.save({
                'nan_step': step,
                'nan_genes': nan_genes,
                'theta_nan': theta.cpu().clone(),
                'theta_prev': prev_theta.cpu().clone(),
                'warm_E_prev': prev_E.cpu().clone(),
                'nll': nll.cpu().clone(),
                'nll_prev': history[-1]['nll'],
            }, STATE_PATH)
            print(f"[PASS 1] Saved debug state to {STATE_PATH}")
            nan_info = {
                'step': step,
                'nan_genes': nan_genes,
                'theta_nan': theta.cpu().clone(),
                'theta_prev': prev_theta.cpu().clone(),
                'warm_E_prev': prev_E.cpu().clone(),
            }
            break

        # Save state before this step for NaN diagnosis
        prev_theta = theta.clone()
        prev_E = E_out['E'].clone()

        grad_flat = grad.reshape(G, -1)
        active = active & (grad_flat.abs().max(dim=-1).values >= grad_tol)
        if not active.any():
            print(f"[PASS 1] Converged at step={step} (all genes below grad_tol)")
            break

        if history_len == 0:
            direction = -grad_flat
        else:
            direction = _lbfgs_two_loop(grad_flat, S_hist, Y_hist, history_len)
        direction[~active] = 0

        slope = (grad_flat * direction).sum(-1)
        bad_dir = active & (slope >= 0)
        if bad_dir.any():
            direction[bad_dir] = -grad_flat[bad_dir]
            slope[bad_dir] = (grad_flat[bad_dir] * direction[bad_dir]).sum(-1)

        alpha = torch.ones(G, device=device, dtype=dtype)
        ls_pending = active.clone()
        E_ls = None
        for ls_iter in range(10):
            theta_try = (theta.reshape(G, -1) + alpha.unsqueeze(-1) * direction).reshape(theta_shape)
            theta_try = torch.where(theta_try < _THETA_MIN, _THETA_MIN, theta_try)
            nll_try, E_try = _nll_only(theta_try, E_out['E'].detach(), ls_pending)
            E_ls = E_try

            # --- Check NaN in line search ---
            nan_ls = ls_pending & nll_try.isnan()
            if nan_ls.any():
                bad_genes = nan_ls.nonzero(as_tuple=False).squeeze(-1).tolist()
                print(f"\n[PASS 1] NaN in LINE SEARCH at step={step}, ls_iter={ls_iter}")
                print(f"         affected families: {bad_genes}")
                for g in bad_genes:
                    print(f"  Family {g}: theta_try={theta_try[g].cpu().tolist()}, alpha={alpha[g].item():.6f}")
                    # Diagnose: print components
                    log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(
                        theta_try, unnorm_row_max, specieswise=specieswise, genewise=True,
                    )
                    print(f"    log_pS={log_pS[g].item():.4f}, log_pD={log_pD[g].item():.4f}, "
                          f"log_pL={log_pL[g].item():.4f}")
                    print(f"    mt[g] min={mt[g].min().item():.4f}, max={mt[g].max().item():.4f}")
                    E_g = E_ls['E'][g]
                    print(f"    E[g] min={E_g.min().item():.4f}, max={E_g.max().item():.4f}, "
                          f"has_nan={E_g.isnan().any().item()}")

            armijo_ok = ls_pending & (nll_try <= nll + 1e-4 * alpha * slope)
            ls_pending = ls_pending & ~armijo_ok
            if not ls_pending.any():
                break
            alpha[ls_pending] *= 0.5

        alpha[ls_pending] = 0.0

        theta_accepted = (theta.reshape(G, -1) + alpha.unsqueeze(-1) * direction).reshape(theta_shape)
        theta_accepted = torch.where(theta_accepted < _THETA_MIN, _THETA_MIN, theta_accepted)

        # Verbose: print per-family info for this step
        print(f"[PASS 1] step={step:2d} | NLL: {['%.4f' % v for v in nll.cpu().tolist()]} | "
              f"alpha={['%.4f' % v for v in alpha.cpu().tolist()]} | "
              f"active={active.cpu().tolist()}")

        nll_new, grad_new, E_new = _nll_and_grad(
            theta_accepted, E_ls['E'].detach(), active,
        )

        s_k = (theta_accepted - theta).reshape(G, -1)
        y_k = (grad_new - grad).reshape(G, -1)
        bad_curv = (s_k * y_k).sum(-1) <= 1e-10
        s_k[bad_curv] = 0
        y_k[bad_curv] = 0
        if history_len < lbfgs_m:
            S_hist[:, history_len] = s_k
            Y_hist[:, history_len] = y_k
            history_len += 1
        else:
            S_hist[:, :-1] = S_hist[:, 1:].clone()
            S_hist[:, -1] = s_k
            Y_hist[:, :-1] = Y_hist[:, 1:].clone()
            Y_hist[:, -1] = y_k

        nll_new[~active] = nll[~active]
        theta, nll, grad, E_out = theta_accepted, nll_new, grad_new, E_new

        history.append({
            'step': step,
            'nll': nll.detach().cpu().clone(),
            'grad_inf': float(grad.abs().max().item()),
            'n_active': int(active.sum().item()),
            'alpha': alpha.detach().cpu().clone(),
        })

    return history, nan_info


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2: Isolated run of the NaN family
# ──────────────────────────────────────────────────────────────────────────────

def run_isolated(ds_single, theta_init_single, max_steps=MAX_STEPS, label="isolated"):
    """Run optimize_theta_genewise on a single family and report."""
    result = optimize_theta_genewise(
        families=ds_single.families,
        species_helpers=ds_single.species_helpers,
        unnorm_row_max=ds_single.unnorm_row_max,
        theta_init=theta_init_single,
        max_steps=max_steps,
        lbfgs_m=10,
        grad_tol=1e-5,
        device=DEVICE, dtype=DTYPE,
        pibar_mode='uniform',
        specieswise=False,
    )
    history = result['history']
    nll_vals = [h['nll'][0].item() for h in history]
    print(f"\n[{label}] NLL trajectory: {['%.4f' % v for v in nll_vals]}")
    print(f"[{label}] Final theta: {result['theta'][0].tolist()}")
    print(f"[{label}] Final NLL: {result['nll'][0].item():.6f}")
    has_nan = any(math.isnan(v) for v in nll_vals)
    print(f"[{label}] Any NaN in history: {has_nan}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Pass 3: Anomaly-detected replay with exact theta from batched run
# ──────────────────────────────────────────────────────────────────────────────

def run_anomaly_replay(ds_single, theta_saved, warm_E_saved, nan_gene_idx):
    """Replay the NaN family's forward+backward at the exact theta that caused NaN.

    Uses torch.autograd.set_detect_anomaly(True) to find the first NaN.
    `theta_saved` is shape [G, 3] from the batched run; we extract gene `nan_gene_idx`.
    `warm_E_saved` is shape [G, S] from the batched run.
    """
    theta_g = theta_saved[nan_gene_idx:nan_gene_idx+1].to(device=DEVICE, dtype=DTYPE)
    warm_E_g = warm_E_saved[nan_gene_idx:nan_gene_idx+1].to(device=DEVICE, dtype=DTYPE)

    print(f"\n[PASS 3] Replaying family {nan_gene_idx} with anomaly detection")
    print(f"  theta = {theta_g[0].tolist()}")
    print(f"  warm_E min={warm_E_g.min().item():.4f}, max={warm_E_g.max().item():.4f}")

    sh = ds_single.species_helpers
    unnorm_row_max = ds_single.unnorm_row_max.to(device=DEVICE, dtype=DTYPE)
    _skip_keys = {'Recipients_mat'}
    def _move_tensor(t):
        if t.dtype.is_floating_point:
            return t.to(device=DEVICE, dtype=DTYPE)
        return t.to(device=DEVICE)
    sh = {
        k: (_move_tensor(v) if torch.is_tensor(v) and k not in _skip_keys else v)
        for k, v in sh.items()
    }
    anc_dense = sh['ancestors_dense'].to(device=DEVICE, dtype=DTYPE)
    _ancestors_T = anc_dense.T.to_sparse_coo()

    fam = ds_single.families[0]
    single_item = {
        'ccp': fam['ccp_helpers'],
        'leaf_row_index': fam['leaf_row_index'],
        'leaf_col_index': fam['leaf_col_index'],
        'root_clade_id': int(fam['root_clade_id']),
    }
    single_batched = collate_gene_families([single_item], dtype=DTYPE, device=DEVICE)
    waves_g, phases_g = compute_clade_waves(fam['ccp_helpers'])
    wave_layout_g = build_wave_layout(
        waves=waves_g, phases=phases_g,
        ccp_helpers=single_batched['ccp'],
        leaf_row_index=single_batched['leaf_row_index'],
        leaf_col_index=single_batched['leaf_col_index'],
        root_clade_ids=single_batched['root_clade_ids'],
        device=DEVICE, dtype=DTYPE,
    )
    root_clade_ids_g = single_batched['root_clade_ids']

    with torch.autograd.set_detect_anomaly(True):
        log_pS, log_pD, log_pL, _, mt = extract_parameters_uniform(
            theta_g, unnorm_row_max, specieswise=False, genewise=True,
        )
        print(f"  log_pS={log_pS[0].item():.4f}, log_pD={log_pD[0].item():.4f}, "
              f"log_pL={log_pL[0].item():.4f}, mt min={mt[0].min().item():.4f}")

        E_out = E_fixed_point(
            species_helpers=sh,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=None, max_transfer_mat=mt,
            max_iters=2000, tolerance=1e-8,
            warm_start_E=warm_E_g,
            dtype=DTYPE, device=DEVICE, pibar_mode='uniform',
            ancestors_T=_ancestors_T,
        )
        E_g = E_out['E'][0]
        print(f"  E min={E_g.min().item():.4f}, max={E_g.max().item():.4f}, "
              f"has_nan={E_g.isnan().any().item()}")

        Pi_out_g = Pi_wave_forward(
            wave_layout=wave_layout_g, species_helpers=sh,
            E=E_g, Ebar=E_out['E_bar'][0],
            E_s1=E_out['E_s1'][0], E_s2=E_out['E_s2'][0],
            log_pS=log_pS[0], log_pD=log_pD[0], log_pL=log_pL[0],
            transfer_mat=None, max_transfer_mat=mt[0],
            device=DEVICE, dtype=DTYPE,
            local_iters=2000, local_tolerance=1e-3,
            pibar_mode='uniform',
        )
        Pi_val = Pi_out_g['Pi']
        print(f"  Pi has_nan={Pi_val.isnan().any().item()}, min={Pi_val[~Pi_val.isinf()].min().item():.4f}")

        logL_g = compute_log_likelihood(Pi_val, E_g, root_clade_ids_g)
        nll_g = logL_g.sum()
        print(f"  NLL (replayed) = {nll_g.item():.6f}")

        if nll_g.isnan():
            print("  [CONFIRMED] NaN reproduced in isolation with saved theta!")
        else:
            print("  [NOT REPRODUCED] NaN not triggered with saved theta in isolation.")
            print("  This suggests the issue is in the BACKWARD path or theta trajectory,")
            print("  not in the forward pass at this exact theta.")

        # --- Now try the backward pass too ---
        try:
            pi_bwd = Pi_wave_backward(
                wave_layout=wave_layout_g,
                Pi_star_wave=Pi_out_g['Pi_wave_ordered'],
                Pibar_star_wave=Pi_out_g['Pibar_wave_ordered'],
                E=E_g, Ebar=E_out['E_bar'][0],
                E_s1=E_out['E_s1'][0], E_s2=E_out['E_s2'][0],
                log_pS=log_pS[0], log_pD=log_pD[0], log_pL=log_pL[0],
                max_transfer_mat=mt[0],
                species_helpers=sh,
                root_clade_ids_perm=wave_layout_g['root_clade_ids'],
                device=DEVICE, dtype=DTYPE,
                neumann_terms=3,
                use_pruning=True,
                pruning_threshold=1e-6,
                pibar_mode='uniform',
                ancestors_T=_ancestors_T,
            )
            grad_E = pi_bwd['grad_E']
            grad_pS = pi_bwd['grad_log_pS']
            print(f"  Backward grad_E has_nan={grad_E.isnan().any().item()}")
            print(f"  Backward grad_pS has_nan={grad_pS.isnan().any().item()}")

            grad_theta_g, _ = _e_adjoint_and_theta_vjp(
                pi_bwd, E_g, E_out['E_bar'][0], E_out['E_s1'][0], E_out['E_s2'][0],
                log_pS[0], log_pD[0], log_pL[0], mt[0],
                sh, wave_layout_g['root_clade_ids'],
                theta_g[0], unnorm_row_max, False,
                DEVICE, DTYPE,
                cg_tol=1e-8, cg_maxiter=500,
                pibar_mode='uniform',
                ancestors_T=_ancestors_T,
            )
            print(f"  grad_theta has_nan={grad_theta_g.isnan().any().item()}: {grad_theta_g.tolist()}")
        except Exception as e:
            print(f"  Backward raised exception: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading {N_FAMILIES} families from {DATA_DIR}")

    # Use fixed seed for reproducibility
    torch.manual_seed(0)

    ds = load_dataset(list(range(N_FAMILIES)))
    G = len(ds.families)
    theta_init = torch.randn(G, 3, dtype=DTYPE, device=DEVICE) * 0.3 - 5.0
    print(f"theta_init:\n{theta_init.cpu()}")

    # ── Pass 1: Batched run ──
    print(f"\n{'='*60}")
    print("PASS 1: Batched run (families 0-4)")
    print('='*60)
    history, nan_info = run_batched_instrumented(ds, theta_init.clone(), max_steps=MAX_STEPS)

    if nan_info is None:
        print(f"\n[PASS 1] No NaN detected in {len(history)-1} steps. Bug not triggered with this seed/init.")
        print("Try with more families or different seed.")
        return

    nan_genes = nan_info['nan_genes']
    nan_gene = nan_genes[0]  # Focus on first NaN gene
    print(f"\n[PASS 1] First NaN: family {nan_gene} at step {nan_info['step']}")

    # ── Pass 2: Isolated run of NaN family ──
    print(f"\n{'='*60}")
    print(f"PASS 2: Isolated run of family {nan_gene} with random init")
    print('='*60)

    ds_single = load_dataset([nan_gene])
    # Use the same initial theta for this family as in the batched run
    theta_init_single = theta_init[nan_gene:nan_gene+1].cpu()
    run_isolated(ds_single, theta_init_single, max_steps=MAX_STEPS, label=f"isolated fam{nan_gene}")

    # ── Pass 3: Anomaly-detected replay with exact saved theta ──
    print(f"\n{'='*60}")
    print(f"PASS 3: Anomaly-detected replay for family {nan_gene}")
    print(f"        using theta from batched run at step {nan_info['step']-1} (the step BEFORE NaN)")
    print('='*60)

    # The theta at step `nan_info['step']` is what caused NaN.
    # We re-run with theta_prev (the accepted theta from the step before).
    run_anomaly_replay(ds_single, nan_info['theta_prev'], nan_info['warm_E_prev'], nan_gene_idx=nan_gene)

    # Also try with the NaN-step theta itself
    print(f"\n--- Also replaying with theta AT the NaN step ---")
    run_anomaly_replay(ds_single, nan_info['theta_nan'], nan_info['warm_E_prev'], nan_gene_idx=nan_gene)

    print(f"\n{'='*60}")
    print("DIAGNOSIS COMPLETE")
    print('='*60)


if __name__ == "__main__":
    main()
