"""Backward pass: retained Triton CUDA path for Pi adjoints.

The production Pi backward/gradient path is the retained fused Triton path for
``float32``/``float64`` tensors and currently requires ``S > 256`` species
nodes.  Tiny species trees are useful for parser/config tests but need a
small-species backward fallback before they can be end-to-end optimizer smokes.
"""

import math

import torch

from .log2_utils import logsumexp2
from .likelihood import prepare_origination_probs
from ._helpers import _safe_exp2_ratio  # noqa: F401
from .memory_policy import proposal0_memory_gate
from .extract_parameters import as_family_param, as_family_species
from .species import species_wave_topology
from .backward_pruning_policy import (
    backward_pruning_policy,
)

_SUPPORTED_BACKWARD_FLOAT_DTYPES = (torch.float32, torch.float64)


def _auto_wrap_backward_inputs(
    *,
    C,
    device,
    family_idx,
    E,
    Ebar,
    E_s1,
    E_s2,
    log_pS,
    log_pD,
    log_pL,
    max_transfer_mat,
):
    """Apply the current ``Pi_wave_backward`` shared-input layout policy.

    This records the legacy MODE-02 behavior: ``family_idx=None`` means a
    single shared family row (``G=1``) and every parameter tensor is unsqueezed
    once.  Passing an explicit ``family_idx`` preserves the caller's parameter
    tensor ranks for downstream shape disambiguation.
    """
    auto_wrapped = family_idx is None
    if auto_wrapped:
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

    return (
        auto_wrapped,
        family_idx,
        E,
        Ebar,
        E_s1,
        E_s2,
        log_pS,
        log_pD,
        log_pL,
        max_transfer_mat,
    )


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
    family_idx=None,
    uniform_pibar_row_max=None,
    origination_probs=None,
    origination_probs_prepared: bool = False,
    initial_v_pi=None,
    return_residual_stats: bool = False,
    fixed_point_relaxation: float = 1.0,
):
    """Wave-decomposed backward pass for implicit gradient computation.

    Computes dL/dPi via Neumann series per wave (root→leaves), then
    accumulates parameter gradients.  Always operates in batched mode
    internally; a single gene tree (family_idx=None) is handled as G=1.  The
    retained fused path requires CUDA, ``float32``/``float64``, and ``S > 256``.

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
        family_idx: Long[C] clade→family mapping. None → auto-wrapped as G=1.
        uniform_pibar_row_max: optional [C] final forward-side row max values
            for uniform Pibar. Required by fused cross-Pibar VJP paths.
        origination_probs: optional [S] or [F, S] root-species origination
            probabilities. ``None`` keeps the historical uniform distribution.
        origination_probs_prepared: when True, skip validation/renormalization
            for model-owned origination probabilities already prepared at
            construction time.
        initial_v_pi: optional [C, S] previous Pi adjoint used to warm-start
            each wave's self-loop fixed-point solve.
        return_residual_stats: when True, apply one extra self-loop step per
            wave after the solve and return aggregate fixed-point residual
            diagnostics.
        fixed_point_relaxation: positive Richardson relaxation factor for
            warmstarted fixed-point updates. ``1.0`` preserves the standard
            update ``v <- rhs + J^T v``.

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
    if isinstance(fixed_point_relaxation, bool) or (
        torch.is_tensor(fixed_point_relaxation)
        and fixed_point_relaxation.dtype == torch.bool
    ):
        raise ValueError("fixed_point_relaxation must be a positive finite number")
    try:
        fixed_point_relaxation = float(fixed_point_relaxation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fixed_point_relaxation must be a positive finite number"
        ) from exc
    if not math.isfinite(fixed_point_relaxation) or fixed_point_relaxation <= 0.0:
        raise ValueError("fixed_point_relaxation must be a positive finite number")
    if S <= 256:
        raise RuntimeError("Pi_wave_backward fused path requires S > 256")
    if initial_v_pi is not None:
        if not torch.is_tensor(initial_v_pi):
            raise TypeError("initial_v_pi must be a tensor when provided")
        if tuple(initial_v_pi.shape) != (int(C), int(S)):
            raise ValueError(
                f"initial_v_pi shape {tuple(initial_v_pi.shape)} does not match "
                f"Pi shape {(int(C), int(S))}"
            )
        initial_v_pi = initial_v_pi.to(device=device, dtype=dtype).contiguous()

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

    (
        _auto_wrapped,
        family_idx,
        E,
        Ebar,
        E_s1,
        E_s2,
        log_pS,
        log_pD,
        log_pL,
        max_transfer_mat,
    ) = _auto_wrap_backward_inputs(
        C=C,
        device=device,
        family_idx=family_idx,
        E=E,
        Ebar=Ebar,
        E_s1=E_s1,
        E_s2=E_s2,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        max_transfer_mat=max_transfer_mat,
    )

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
    residual_absmax_values = []
    residual_relmax_values = []

    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    solved_v_pi = torch.zeros_like(accumulated_rhs)
    if torch.is_tensor(root_clade_ids_perm):
        root_ids_device = root_clade_ids_perm.to(device=device, dtype=torch.long)
    else:
        root_ids_device = torch.as_tensor(
            root_clade_ids_perm,
            device=device,
            dtype=torch.long,
        )
    origin_probs = prepare_origination_probs(
        origination_probs,
        S=S,
        device=device,
        dtype=dtype,
        family_count=(
            int(root_ids_device.numel()) if origination_probs is not None else None
        ),
        assume_prepared=origination_probs_prepared,
    )
    root_Pi = Pi_star_wave.index_select(0, root_ids_device)
    if origin_probs is None:
        root_terms = root_Pi
    elif origin_probs.ndim == 1:
        root_terms = root_Pi + torch.log2(origin_probs).view(1, S)
    else:
        root_terms = root_Pi + torch.log2(origin_probs)
    lse = logsumexp2(root_terms, dim=-1, keepdim=True)
    root_rhs = -_safe_exp2_ratio(root_terms, lse)
    accumulated_rhs.index_copy_(0, root_ids_device, root_rhs)

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
    pruning_policy = backward_pruning_policy(
        use_pruning=use_pruning,
        pruning_threshold=pruning_threshold,
    )
    active_mask_threshold = pruning_policy.active_mask_threshold

    def _compute_active_mask(rhs):
        return active_mask_from_rhs_absmax_fused(
            rhs, active_mask_threshold, use_pruning=use_pruning
        )

    skip_inactive_pibar_zero = pruning_policy.skip_inactive_pibar_zero
    specialize_nonleaf_leaf_term = True

    for k in range(K - 1, -1, -1):
        meta = wave_metas[k]
        ws = meta['start']
        we = meta['end']
        W = meta['W']

        # The fused uniform kernel treats rhs as read-only, and this wave's
        # later cross-DTS/Pibar adjoints accumulate into child rows.
        rhs_k = accumulated_rhs[ws:we]
        active_mask = _compute_active_mask(rhs_k).contiguous() if use_pruning else None

        leaf_wt = None
        wave_has_leaf_term = (
            not specialize_nonleaf_leaf_term
            or int(meta.get('phase', 1)) == 1
        )

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

        param_grad_vector = (
            grad_log_pD.ndim == 2
            and grad_log_pS.ndim == 2
            and int(grad_log_pD.shape[0]) == 1
            and int(grad_log_pS.shape[0]) == 1
            and int(grad_log_pD.shape[1]) == S
            and int(grad_log_pS.shape[1]) == S
        )
        param_grad_scalar = (
            grad_log_pD.ndim == 1
            and grad_log_pS.ndim == 1
            and int(grad_log_pD.numel()) == 1
            and int(grad_log_pS.numel()) == 1
        )
        triton_accum_self_loop_grads = (
            _auto_wrapped
            and dtype == torch.float32
            and G == 1
            and grad_E_acc.ndim == 2
            and grad_Ebar_acc.ndim == 2
            and grad_E_s1_acc.ndim == 2
            and grad_E_s2_acc.ndim == 2
            and grad_mt.ndim == 2
            and int(grad_E_acc.shape[0]) == 1
            and int(grad_Ebar_acc.shape[0]) == 1
            and int(grad_E_s1_acc.shape[0]) == 1
            and int(grad_E_s2_acc.shape[0]) == 1
            and int(grad_mt.shape[0]) == 1
            and int(grad_E_acc.shape[1]) == S
            and int(grad_Ebar_acc.shape[1]) == S
            and int(grad_E_s1_acc.shape[1]) == S
            and int(grad_E_s2_acc.shape[1]) == S
            and int(grad_mt.shape[1]) == S
            and (param_grad_vector or param_grad_scalar)
        )
        self_loop_grad_targets = None
        if triton_accum_self_loop_grads:
            self_loop_grad_targets = (
                grad_log_pD[0] if param_grad_vector else grad_log_pD,
                grad_log_pS[0] if param_grad_vector else grad_log_pS,
                grad_E_acc[0],
                grad_Ebar_acc[0],
                grad_E_s1_acc[0],
                grad_E_s2_acc[0],
                grad_mt[0],
                param_grad_vector,
            )
        wave_result = wave_backward_uniform_fused(
            Pi_star_wave, Pibar_star_wave, ws, W, S,
            dts_r, rhs_k,
            mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
            sp_child1_wave, sp_child2_wave, leaf_wt,
            neumann_terms=neumann_terms,
            leaf_species_idx=leaf_species_index_wave,
            leaf_logp=uniform_leaf_logp,
            has_leaf_term=wave_has_leaf_term,
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
            self_loop_grad_targets=self_loop_grad_targets,
            initial_v=None if initial_v_pi is None else initial_v_pi[ws:we],
            return_residual_stats=return_residual_stats,
            fixed_point_relaxation=(
                fixed_point_relaxation if initial_v_pi is not None else 1.0
            ),
        )
        if return_residual_stats:
            v_k, aw0, aw1, aw2, aw345, aw3, aw4, residual_stats = wave_result
            if residual_stats is not None:
                residual_absmax_values.append(
                    float(residual_stats["residual_absmax"])
                )
                residual_relmax_values.append(
                    float(residual_stats["residual_relmax"])
                )
        else:
            v_k, aw0, aw1, aw2, aw345, aw3, aw4 = wave_result
        solved_slice = solved_v_pi[ws:we]
        if active_mask is None:
            solved_slice.copy_(v_k)
        else:
            solved_slice.copy_(torch.where(active_mask[:, None], v_k, 0.0))
        self_loop_grads_accumulated = triton_accum_self_loop_grads

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
        'v_Pi': solved_v_pi,
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
        'used_pi_initial_guess': initial_v_pi is not None,
    }
    if return_residual_stats:
        result.update(
            {
                'pi_adjoint_residual_absmax': max(
                    residual_absmax_values,
                    default=0.0,
                ),
                'pi_adjoint_residual_relmax': max(
                    residual_relmax_values,
                    default=0.0,
                ),
                'pi_adjoint_residual_wave_count': len(residual_absmax_values),
            }
        )
    # Unwrap G=1 results back to original shapes.
    if _auto_wrapped:
        for key in ('grad_E', 'grad_Ebar', 'grad_E_s1', 'grad_E_s2',
                     'grad_log_pD', 'grad_log_pS', 'grad_max_transfer_mat'):
            result[key] = result[key][0]

    return result
