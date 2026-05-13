"""Backward pass: retained fused CUDA path for Pi adjoints."""

import os

import torch

from .log2_utils import logsumexp2
from ._helpers import _safe_exp2_ratio  # noqa: F401
from .memory_policy import proposal0_memory_gate
from .extract_parameters import as_family_param, as_family_species
from .species import species_wave_topology

_SUPPORTED_BACKWARD_FLOAT_DTYPES = (torch.float32, torch.float64)


# ---------------------------------------------------------------------------
# Pi wave backward
# ---------------------------------------------------------------------------

@torch.no_grad()
def Pi_wave_backward(
    wave_layout,
    Pi_star_wave,
    Pibar_star_wave,
    E, Ebar, E_s1, E_s2,
    log_pS, log_pD, log_pL,
    max_transfer_mat,
    species_helpers,
    root_clade_ids_perm,
    device, dtype,
    *,
    neumann_terms=3,
    pruning_threshold=1e-6,
    use_pruning=True,
    ancestors_T=None,
    family_idx=None,
    uniform_pibar_row_max=None,
):
    """Wave-decomposed backward pass for implicit gradient computation.

    Computes dL/dPi via Neumann series per wave (root→leaves), then
    accumulates parameter gradients.  Always operates in batched mode
    internally; a single gene tree (family_idx=None) is handled as G=1.

    Args:
        wave_layout: dict from build_wave_layout()
        Pi_star_wave: [C, S] converged Pi in wave-ordered space
        Pibar_star_wave: [C, S] converged Pibar in wave-ordered space
        E, Ebar, E_s1, E_s2: [S] or [G, S] species extinction
        log_pS, log_pD, log_pL: scalar/[S] or [G]/[G, S] event probabilities
        max_transfer_mat: [S] or [G, S] log2-space
        species_helpers: species tree helpers
        root_clade_ids_perm: Long[F] root clade IDs in wave-ordered space
        device, dtype: target device/dtype
        neumann_terms: number of Neumann series terms (default 3)
        pruning_threshold: linear-space adjoint magnitude threshold for pruning
        use_pruning: whether to prune waves with negligible adjoint gradient
        ancestors_T: [S, S] sparse CSR = ancestors.T
        family_idx: Long[C] clade→family mapping. None → auto-wrapped as G=1.
        uniform_pibar_row_max: optional [C] final forward-side row max values
            for uniform Pibar. Used only by opt-in fused cross-Pibar VJP paths.

    Returns:
        dict with:
            'v_Pi': [C, S] adjoint vector for Pi (wave-ordered)
            'grad_E': [S] or [G, S] gradient contribution from Pi adjoint to E
            'grad_log_pS': [S] or [G, S] gradient wrt log_pS
            'grad_log_pD': [S] or [G, S] gradient wrt log_pD
            'grad_max_transfer_mat': [S] or [G, S] gradient wrt max_transfer_mat
    """
    from .kernels.wave_backward import (
        wave_backward_uniform_fused,
        dts_cross_backward_accum_fused,
        uniform_cross_pibar_vjp_tree_from_ud_fused,
        active_mask_from_rhs_absmax_fused,
    )
    from .forward import _compute_dts_cross as _compute_dts_cross_for_backward

    wave_metas = wave_layout['wave_metas']
    C, S = Pi_star_wave.shape
    K = len(wave_metas)

    target_device = torch.device(device)
    if target_device.type == 'cuda' and target_device.index is None:
        target_device = torch.device('cuda', torch.cuda.current_device())
    device = target_device
    if target_device.type != 'cuda':
        raise RuntimeError("Pi_wave_backward only retains the fused CUDA fast path")
    if dtype not in _SUPPORTED_BACKWARD_FLOAT_DTYPES:
        raise RuntimeError("Pi_wave_backward fused path requires float32 or float64")
    if S <= 256:
        raise RuntimeError("Pi_wave_backward fused path requires S > 256")

    dts_grad_mt_two_stage_tile_splits = 128

    species_topology = species_wave_topology(
        species_helpers,
        S=S,
        device=target_device,
    )
    sp_child1 = species_topology["sp_child1"]
    sp_child2 = species_topology["sp_child2"]
    sp_child1_wave = sp_child1
    sp_child2_wave = sp_child2
    sp_parent_wave = species_topology["sp_parent"]
    max_ancestor_depth = int(species_topology["max_ancestor_depth"])
    compact_level_ptr = species_topology["compact_level_ptr"]
    compact_level_parents = species_topology["compact_level_parents"]
    compact_level_child1 = species_topology["compact_level_child1"]
    compact_level_child2 = species_topology["compact_level_child2"]

    # Auto-wrap single-family inputs into batched format (G=1).
    _auto_wrapped = family_idx is None
    if _auto_wrapped:
        family_idx = torch.zeros(C, dtype=torch.long, device=device)
        E = E.unsqueeze(0)
        Ebar = Ebar.unsqueeze(0)
        E_s1 = E_s1.unsqueeze(0)
        E_s2 = E_s2.unsqueeze(0)
        log_pS = log_pS.unsqueeze(0)
        log_pD = log_pD.unsqueeze(0)
        log_pL = log_pL.unsqueeze(0)
        max_transfer_mat = max_transfer_mat.unsqueeze(0)
    else:
        family_idx = family_idx.to(device=device, dtype=torch.long).contiguous()

    mt_squeezed = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 2 else max_transfer_mat

    G = int(E.shape[0]) if E.ndim == 2 else 1

    def _family_species_shape_ok(p):
        return (
            torch.is_tensor(p)
            and p.ndim == 2
            and int(p.shape[0]) == int(G)
            and int(p.shape[1]) == int(S)
        )

    param_kwargs = dict(S=S, device=target_device, dtype=dtype, family_rows=G)
    mt_family = as_family_species(mt_squeezed, name="max_transfer_mat", **param_kwargs)
    E_family = as_family_species(E, name="E", **param_kwargs)
    Ebar_family = as_family_species(Ebar, name="Ebar", **param_kwargs)
    E_s1_family = as_family_species(E_s1, name="E_s1", **param_kwargs)
    E_s2_family = as_family_species(E_s2, name="E_s2", **param_kwargs)
    log_pD_param = as_family_param(log_pD, name="log_pD", **param_kwargs)
    log_pS_param = as_family_param(log_pS, name="log_pS", **param_kwargs)
    log_pD_family = as_family_species(log_pD, name="log_pD", **param_kwargs)
    log_pS_family = as_family_species(log_pS, name="log_pS", **param_kwargs)
    if _auto_wrapped:
        mt_shared = mt_squeezed[0] if mt_squeezed.ndim == 2 else mt_squeezed
        E_shared = E[0]
        Ebar_shared = Ebar[0]
        E_s1_shared = E_s1[0]
        E_s2_shared = E_s2[0]
        log_pD_shared = log_pD[0]
        log_pS_shared = log_pS[0]
        DL_shared = (1.0 + log_pD_shared + E_shared).contiguous()
        SL1_shared = (log_pS_shared + E_s2_shared).contiguous()
        SL2_shared = (log_pS_shared + E_s1_shared).contiguous()
    else:
        mt_shared = E_shared = Ebar_shared = E_s1_shared = E_s2_shared = None
        log_pD_shared = log_pS_shared = None
        DL_shared = SL1_shared = SL2_shared = None

    family_indexed_self_loop_supported = (
        _family_species_shape_ok(E_family)
        and _family_species_shape_ok(Ebar_family)
        and _family_species_shape_ok(E_s1_family)
        and _family_species_shape_ok(E_s2_family)
        and _family_species_shape_ok(mt_family)
        and _family_species_shape_ok(log_pD_family)
        and _family_species_shape_ok(log_pS_family)
    )
    if not (_auto_wrapped or family_indexed_self_loop_supported):
        raise RuntimeError("Pi_wave_backward requires shared or family-indexed species constants")

    max_wave_W = max((int(meta.get('W', 0)) for meta in wave_metas), default=0)

    DL_family = (1.0 + log_pD_family + E_family).contiguous()
    SL1_family = (log_pS_family + E_s2_family).contiguous()
    SL2_family = (log_pS_family + E_s1_family).contiguous()

    def _wave_consts(ws, we, *, family_indexed):
        """Return constants for the current self-loop wave."""
        if _auto_wrapped:
            return mt_shared, DL_shared, E_shared, Ebar_shared, SL1_shared, SL2_shared
        if family_indexed:
            return mt_family, DL_family, E_family, Ebar_family, SL1_family, SL2_family
        fi_w = family_idx[ws:we]
        return (
            mt_family[fi_w],
            DL_family[fi_w],
            E_family[fi_w],
            Ebar_family[fi_w],
            SL1_family[fi_w],
            SL2_family[fi_w],
        )

    leaf_species_index = wave_layout.get('leaf_species_index')
    if leaf_species_index is None:
        raise RuntimeError("Pi_wave_backward fused path requires leaf_species_index")
    if _auto_wrapped:
        uniform_leaf_logp = (
            log_pS_shared.expand(S).contiguous()
            if log_pS_shared.ndim == 0
            else log_pS_shared.contiguous()
        )
    else:
        uniform_leaf_logp = log_pS_family

    self_loop_2d_memory_ok, required_bytes, budget_bytes = proposal0_memory_gate(
        max(1, int(max_wave_W)),
        S,
        dtype,
        device=device,
    )
    if not self_loop_2d_memory_ok:
        raise RuntimeError(
            "Pi_wave_backward fused path requires 2D self-loop scratch "
            f"({required_bytes / (1024 ** 3):.2f} GiB requested, "
            f"{(budget_bytes or 0) / (1024 ** 3):.2f} GiB budget)"
        )

    leaf_species_index_wave = leaf_species_index.to(device=device, dtype=torch.int32).contiguous()

    n_waves_total = K
    n_waves_skipped = 0
    n_clades_total = C
    n_clades_skipped = 0

    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    if torch.is_tensor(root_clade_ids_perm):
        root_ids_iter = root_clade_ids_perm.detach()
        if root_ids_iter.device.type != "cpu":
            root_ids_iter = root_ids_iter.cpu()
        root_ids_iter = root_ids_iter.tolist()
    else:
        root_ids_iter = root_clade_ids_perm
    for r in root_ids_iter:
        r = int(r)
        root_Pi = Pi_star_wave[r]
        lse = logsumexp2(root_Pi, dim=0)
        accumulated_rhs[r] = -_safe_exp2_ratio(root_Pi, lse)

    grad_log_pD = torch.zeros_like(log_pD)
    grad_log_pS = torch.zeros_like(log_pS)
    grad_mt = torch.zeros_like(mt_squeezed)
    grad_E_acc = torch.zeros_like(E)
    grad_Ebar_acc = torch.zeros_like(Ebar)
    grad_E_s1_acc = torch.zeros_like(E_s1)
    grad_E_s2_acc = torch.zeros_like(E_s2)
    has_forward_pibar_row_max = (
        uniform_pibar_row_max is not None
        and torch.is_tensor(uniform_pibar_row_max)
        and uniform_pibar_row_max.numel() == C
    )
    if not has_forward_pibar_row_max:
        raise RuntimeError("Pi_wave_backward fused path requires uniform_pibar_row_max from forward")
    forward_pibar_row_max = uniform_pibar_row_max.to(device=device, dtype=dtype).contiguous()
    active_mask_threshold = pruning_threshold if use_pruning else 0.0

    def _compute_active_mask(rhs):
        return active_mask_from_rhs_absmax_fused(
            rhs, active_mask_threshold, use_pruning=use_pruning
        )

    no_cpu_pruning = (
        os.environ.get("GPUREC_BACKWARD_NO_CPU_PRUNING", "1").strip().lower()
        not in ("", "0", "false", "no", "off")
    )
    skip_inactive_pibar_zero = (
        os.environ.get("GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO", "1").strip().lower()
        not in ("", "0", "false", "no", "off")
    )
    cuda_self_loop_nosplit_mode = os.environ.get(
        "GPUREC_CUDA_SELF_LOOP_NOSPLIT",
        "auto",
    ).strip().lower()
    cuda_self_loop_nosplit_enabled = (
        cuda_self_loop_nosplit_mode not in ("", "0", "false", "no", "off")
    )
    cuda_self_loop_nosplit_required = cuda_self_loop_nosplit_mode in (
        "1",
        "true",
        "yes",
        "on",
        "force",
        "required",
    )
    cuda_self_loop_nosplit_correction = os.environ.get(
        "GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION",
        "tree",
    )

    for k in range(K - 1, -1, -1):
        meta = wave_metas[k]
        ws = meta['start']
        we = meta['end']
        W = meta['W']

        # The fused uniform kernel treats rhs as read-only, and this wave's
        # later cross-DTS/Pibar adjoints accumulate into child rows.
        rhs_k = accumulated_rhs[ws:we]
        if no_cpu_pruning:
            active_mask = (
                _compute_active_mask(rhs_k).contiguous() if use_pruning else None
            )
        else:
            active_mask = _compute_active_mask(rhs_k).contiguous()
            wave_active = bool(active_mask.any())
            if not wave_active:
                n_waves_skipped += 1
                n_clades_skipped += W
                continue

            n_clades_skipped += W - int(active_mask.sum().item())

        leaf_wt = None

        if meta['has_splits']:
            reduce_idx = meta['reduce_idx']
            log_pD_dts = log_pD_shared if _auto_wrapped else log_pD_param
            log_pS_dts = log_pS_shared if _auto_wrapped else log_pS_param
            with torch.no_grad():
                dts_r = _compute_dts_cross_for_backward(
                    Pi_star_wave.detach(), Pibar_star_wave.detach(), meta,
                    sp_child1, sp_child2, log_pD_dts, log_pS_dts, S, device, dtype,
                    active_mask=active_mask,
                    family_idx=None if _auto_wrapped else family_idx,
                    family_offset=0 if _auto_wrapped else ws,
                )
        else:
            dts_r = None

        use_family_indexed_self_loop = not _auto_wrapped
        mt_w, DL_w, E_w, Ebar_w, SL1_w, SL2_w = _wave_consts(
            ws, we, family_indexed=use_family_indexed_self_loop
        )

        # Per-wave family indices for scatter accumulation.
        fi_w = family_idx[ws:we]
        fi_expand = fi_w.unsqueeze(1).expand(W, S)

        def _scatter_accum(acc, contrib):
            if contrib.dtype != acc.dtype:
                contrib = contrib.to(dtype=acc.dtype)
            if G == 1:
                if acc.ndim == 1:
                    acc[0] += contrib.sum()
                else:
                    acc[0] += contrib.sum(dim=0)
                return
            if acc.ndim == 1:
                acc.scatter_add_(0, fi_w, contrib.sum(dim=1))
            else:
                acc.scatter_add_(0, fi_expand, contrib)

        use_cuda_nosplit = (
            cuda_self_loop_nosplit_enabled
            and _auto_wrapped
            and dts_r is None
            and dtype == torch.float32
            and torch.is_tensor(uniform_leaf_logp)
            and int(uniform_leaf_logp.numel()) == S
            and compact_level_ptr is not None
            and compact_level_parents is not None
            and compact_level_child1 is not None
            and compact_level_child2 is not None
        )
        self_loop_grads_accumulated = False
        if use_cuda_nosplit:
            try:
                from .kernels.wave_backward_cuda import wave_backward_uniform_nosplit_cuda

                v_k = wave_backward_uniform_nosplit_cuda(
                    Pi_star_wave,
                    Pibar_star_wave,
                    ws,
                    W,
                    S,
                    rhs_k,
                    mt_w,
                    DL_w,
                    Ebar_w,
                    E_w,
                    SL1_w,
                    SL2_w,
                    sp_child1_wave,
                    sp_child2_wave,
                    sp_parent_wave,
                    leaf_species_index_wave,
                    uniform_leaf_logp,
                    forward_pibar_row_max,
                    compact_level_ptr,
                    compact_level_parents,
                    compact_level_child1,
                    compact_level_child2,
                    (
                        grad_log_pD[0],
                        grad_log_pS[0],
                        grad_E_acc[0],
                        grad_Ebar_acc[0],
                        grad_E_s1_acc[0],
                        grad_E_s2_acc[0],
                        grad_mt[0] if grad_mt.ndim == 2 else grad_mt,
                    ),
                    active_mask=active_mask,
                    neumann_terms=neumann_terms,
                    correction_mode=cuda_self_loop_nosplit_correction,
                )
                aw0 = aw1 = aw2 = aw345 = aw3 = aw4 = None
                self_loop_grads_accumulated = True
            except (ImportError, RuntimeError):
                if cuda_self_loop_nosplit_required:
                    raise
                cuda_self_loop_nosplit_enabled = False
                use_cuda_nosplit = False
                self_loop_grads_accumulated = False
        if not use_cuda_nosplit:
            v_k, aw0, aw1, aw2, aw345, aw3, aw4 = wave_backward_uniform_fused(
                Pi_star_wave, Pibar_star_wave, ws, W, S,
                dts_r, rhs_k,
                mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                sp_child1_wave, sp_child2_wave, leaf_wt,
                neumann_terms=neumann_terms,
                leaf_species_idx=leaf_species_index_wave,
                leaf_logp=uniform_leaf_logp,
                active_mask=active_mask,
                sp_parent=sp_parent_wave,
                max_ancestor_depth=max_ancestor_depth,
                pibar_row_max=forward_pibar_row_max,
                family_idx=family_idx if use_family_indexed_self_loop else None,
                family_indexed_consts=use_family_indexed_self_loop,
                compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents,
                compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2,
            )

        if not self_loop_grads_accumulated:
            if (
                G == 1
                and grad_log_pD.ndim == 2
                and grad_log_pS.ndim == 2
                and grad_E_acc.ndim == 2
                and grad_Ebar_acc.ndim == 2
                and grad_E_s1_acc.ndim == 2
                and grad_E_s2_acc.ndim == 2
                and grad_mt.ndim == 2
            ):
                aw0_sum = aw0.sum(dim=0)
                aw2_sum = aw2.sum(dim=0)
                grad_log_pD[0] += aw0_sum.to(dtype=grad_log_pD.dtype)
                grad_log_pS[0] += aw345.sum(dim=0).to(dtype=grad_log_pS.dtype)
                grad_E_acc[0] += (aw0_sum + aw2_sum).to(dtype=grad_E_acc.dtype)
                grad_Ebar_acc[0] += aw1.sum(dim=0).to(dtype=grad_Ebar_acc.dtype)
                grad_E_s1_acc[0] += aw4.sum(dim=0).to(dtype=grad_E_s1_acc.dtype)
                grad_E_s2_acc[0] += aw3.sum(dim=0).to(dtype=grad_E_s2_acc.dtype)
                grad_mt[0] += aw2_sum.to(dtype=grad_mt.dtype)
            else:
                _scatter_accum(grad_log_pD, aw0)
                _scatter_accum(grad_log_pS, aw345)
                _scatter_accum(grad_E_acc, aw0 + aw2)
                _scatter_accum(grad_Ebar_acc, aw1)
                _scatter_accum(grad_E_s1_acc, aw4)
                _scatter_accum(grad_E_s2_acc, aw3)
                _scatter_accum(grad_mt, aw2)

        if meta['has_splits'] and dts_r is not None:
            sl = meta['sl']
            sr = meta['sr']
            wlsp = meta['log_split_probs']
            reduce_idx = meta['reduce_idx']

            if _auto_wrapped:
                dts_log_pD = log_pD_shared
                dts_log_pS = log_pS_shared
                dts_grad_log_pD = grad_log_pD[0]
                dts_grad_log_pS = grad_log_pS[0]
                dts_grad_mt = grad_mt[0] if grad_mt.ndim == 2 else grad_mt
                dts_mt = mt_shared
                dts_family_idx = None
            else:
                dts_log_pD = log_pD_param
                dts_log_pS = log_pS_param
                dts_grad_log_pD = grad_log_pD
                dts_grad_log_pS = grad_log_pS
                dts_grad_mt = grad_mt
                dts_mt = mt_family
                dts_family_idx = family_idx
            dts_accum_result = dts_cross_backward_accum_fused(
                Pi_star_wave, Pibar_star_wave, v_k, ws,
                sl, sr, reduce_idx, wlsp,
                dts_log_pD, dts_log_pS,
                sp_child1, sp_child2, accumulated_rhs, S,
                active_mask=active_mask,
                merge_s_term=True,
                grad_log_pD=dts_grad_log_pD,
                grad_log_pS=dts_grad_log_pS,
                grad_mt=dts_grad_mt,
                accum_param_reductions=True,
                accum_mt_reduction=True,
                output_pibar_ud=True,
                output_pibar_side_active=True,
                pibar_side_threshold=0.0,
                mt_squeezed=dts_mt,
                pibar_row_max=forward_pibar_row_max,
                grad_mt_two_stage=(
                    dts_grad_mt.ndim == 1
                    and int(dts_grad_mt.numel()) == S
                ),
                grad_mt_two_stage_tile_splits=dts_grad_mt_two_stage_tile_splits,
                skip_inactive_pibar_output_zero=skip_inactive_pibar_zero,
                family_idx=dts_family_idx,
            )
            grad_Pibar_l, grad_Pibar_r, pibar_side_active, _param_pD, _param_pS = dts_accum_result

            uniform_cross_pibar_vjp_tree_from_ud_fused(
                Pi_star_wave,
                grad_Pibar_l,
                grad_Pibar_r,
                sl,
                sr,
                accumulated_rhs,
                S,
                active_mask=active_mask,
                reduce_idx=reduce_idx,
                pibar_row_max=forward_pibar_row_max,
                skip_zero_sides=True,
                side_active=pibar_side_active,
                compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents,
                compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2,
                side_active_threshold=0.0,
            )

    result = {
        'v_Pi': accumulated_rhs,
        'grad_E': grad_E_acc,
        'grad_Ebar': grad_Ebar_acc,
        'grad_E_s1': grad_E_s1_acc,
        'grad_E_s2': grad_E_s2_acc,
        'grad_log_pD': grad_log_pD,
        'grad_log_pS': grad_log_pS,
        'grad_max_transfer_mat': grad_mt,
        'n_waves_total': n_waves_total,
        'n_waves_skipped': n_waves_skipped,
        'n_waves_processed': n_waves_total - n_waves_skipped,
        'n_clades_total': n_clades_total,
        'n_clades_skipped': n_clades_skipped,
        'n_clades_active': n_clades_total - n_clades_skipped,
    }
    # Unwrap G=1 results back to original shapes.
    if _auto_wrapped:
        for key in ('grad_E', 'grad_Ebar', 'grad_E_s1', 'grad_E_s2',
                     'grad_log_pD', 'grad_log_pS', 'grad_max_transfer_mat'):
            result[key] = result[key][0]

    return result
