"""Fused Triton kernels for the retained wave-backward fast path.

This module also contains private standalone diagnostics/helpers used by that
path.  In particular, ``active_mask_from_rhs_absmax_fused()`` accepts bf16
inputs for standalone row-mask experiments, but the retained public
``Pi_wave_backward`` path rejects bf16 before this helper is reached.
"""

import torch
import triton
import triton.language as tl

from gpurec.core.kernels._dts_layout_contract import dts_backward_param_layout
from gpurec.core.memory_policy import proposal0_memory_gate

from gpurec.core.kernels.wave_backward_kernels import (
    _active_mask_from_rhs_absmax_kernel,
    _wave_backward_uniform_2d_precompute_kernel,
    _wave_backward_uniform_2d_jt_kernel,
    _receiver_grad_from_pibar_self_loop_kernel,
    _wave_backward_uniform_param_store_kernel,
    _dts_cross_backward_accum_kernel,
    _dts_grad_mt_two_stage_reduce_kernel,
    _pibar_ud_side_active_kernel,
    _uniform_cross_pibar_vjp_tree_from_ud_compact_kernel,
)

_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64, torch.bfloat16)


def _tl_float_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _device_scalar_param(param, *, device, dtype):
    """Return a one-element device tensor without extracting CUDA scalars."""
    if torch.is_tensor(param):
        if param.numel() != 1:
            raise ValueError("fused DTS backward scalar parameters must have one element")
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        return param.reshape(1).contiguous()
    return torch.tensor([param], device=device, dtype=dtype)


def _dts_layout_param_args(log_pD, log_pS, *, family_idx, S, device, dtype):
    """Return DTS parameter tensors plus a Triton addressing layout.

    With ``family_idx`` present, retained backward treats a one-dimensional
    tensor as family scalar rows before considering a shared ``[S]`` species
    vector.  Direct callers that need forward/backward parity when ``G == S``
    should use ``[G, 1]`` for family scalar rows or ``[G, S]`` for
    family/species rows.

    Layouts:
      0: shared scalar, tensor [1]
      1: shared species vector, tensor [S]
      2: family scalar, tensor [G] addressed by family_idx[parent]
      3: family species, tensor [G, S] addressed by family_idx[parent], s
    """

    def _normalize(param):
        if not torch.is_tensor(param):
            return _device_scalar_param(param, device=device, dtype=dtype), 0
        if param.device != device or param.dtype != dtype:
            param = param.to(device=device, dtype=dtype)
        try:
            layout = dts_backward_param_layout(
                param,
                S=S,
                family_indexed=family_idx is not None,
            )
        except ValueError as exc:
            raise ValueError(
                "DTS parameters must be scalar, [S], [G], [G, 1], or [G, S] "
                "for the fused DTS backward path"
            ) from exc
        layout_code = int(layout.code)
        if layout_code == 0:
            return param.reshape(1).contiguous(), 0
        if layout_code == 1:
            return param.contiguous(), 1
        if layout_code == 2:
            if param.ndim == 2:
                return param.reshape(int(param.shape[0])).contiguous(), 2
            return param.contiguous(), 2
        if layout_code == 3:
            return param.contiguous(), 3
        raise AssertionError("validated DTS backward layout reached unreachable branch")

    pD, layout_D = _normalize(log_pD)
    pS, layout_S = _normalize(log_pS)
    if layout_D != layout_S:
        raise ValueError("log_pD/log_pS must use the same DTS parameter layout")
    return pD, pS, layout_D


def _dts_grad_layout(grad, *, family_idx, S):
    """Return gradient addressing layout matching _dts_layout_param_args."""
    try:
        return int(
            dts_backward_param_layout(
                grad,
                S=S,
                family_indexed=family_idx is not None,
            ).code
        )
    except ValueError as exc:
        raise ValueError("unsupported DTS gradient layout") from exc


def _uniform_backward_const_layout(const_tensor, family_idx, family_indexed):
    """Return addressing mode for self-loop constants.

    Modes:
      0: shared [S]
      1: row-expanded [W, S]
      2: family-indexed [G, S] addressed through family_idx[C]
    """
    if family_indexed:
        if family_idx is None:
            raise ValueError("family-indexed backward constants require family_idx")
        if const_tensor.ndim != 2:
            raise ValueError("family-indexed backward constants require [G, S] tensors")
        return 2
    if const_tensor.ndim == 2:
        return 1
    return 0


def _uniform_backward_leaf_logp_mode(use_leaf_index, leaf_logp, family_idx, family_indexed):
    """Return addressing mode for leaf log-probabilities in the self-loop."""
    if not use_leaf_index:
        return 0
    if family_indexed:
        if family_idx is None:
            raise ValueError("family-indexed leaf log-probabilities require family_idx")
        if leaf_logp.ndim == 1:
            return 1
        if leaf_logp.ndim == 2:
            if int(leaf_logp.shape[1]) == 1:
                raise ValueError("family-indexed [G, 1] leaf_logp should be expanded to [G, S]")
            return 2
        raise ValueError("family-indexed leaf_logp must have shape [G] or [G, S]")
    if leaf_logp.numel() == 1:
        return 3
    return 0


def active_mask_from_rhs_absmax_fused(rhs, threshold, *, use_pruning=True):
    """Build the row activity mask for backward pruning in one Triton launch.

    This is a private retained-kernel helper, not a public dtype policy.  The
    helper accepts fp32/fp64/bf16 CUDA tensors for standalone mask experiments;
    the public ``Pi_wave_backward`` path still supports only fp32/fp64 and
    rejects bf16 before calling this helper.
    """
    if rhs.ndim != 2:
        raise ValueError("rhs must be a 2D tensor")
    if rhs.device.type != "cuda":
        raise ValueError("active_mask_from_rhs_absmax_fused requires a CUDA tensor")
    if rhs.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise ValueError(
            "active_mask_from_rhs_absmax_fused supports fp32/fp64/bf16 tensors"
        )

    W, S = rhs.shape
    active_mask = torch.empty((W,), device=rhs.device, dtype=torch.bool)
    if W == 0:
        return active_mask

    BLOCK_S = min(256, triton.next_power_of_2(S))
    _active_mask_from_rhs_absmax_kernel[(W,)](
        rhs,
        active_mask,
        float(threshold),
        S,
        rhs.stride(0),
        BLOCK_S,
        STRICT_GT=bool(not use_pruning),
        DTYPE=_tl_float_dtype(rhs.dtype),
    )
    return active_mask


@torch.no_grad()
def _gmres_solve_wave_self_loop(
    apply_a,
    rhs: torch.Tensor,
    *,
    max_iter: int,
) -> torch.Tensor:
    """Solve ``A v = rhs`` for one wave with fixed-iteration unrestarted GMRES."""
    max_iter = int(max_iter)
    if max_iter < 1:
        return torch.zeros_like(rhs)

    b_norm_t = torch.linalg.vector_norm(rhs)
    if float(b_norm_t.detach().cpu()) == 0.0:
        return torch.zeros_like(rhs)

    return _gmres_solve_wave_self_loop_fixed_cgs2(
        apply_a,
        rhs,
        max_iter=max_iter,
        b_norm_t=b_norm_t,
    )


def _gmres_solve_wave_self_loop_fixed_cgs2(
    apply_a,
    rhs: torch.Tensor,
    *,
    max_iter: int,
    b_norm_t: torch.Tensor,
) -> torch.Tensor:
    """Fixed-m GMRES Arnoldi using batched CGS with one reorthogonalization."""
    basis = torch.empty(
        (max_iter + 1, *rhs.shape),
        dtype=rhs.dtype,
        device=rhs.device,
    )
    basis_2d = basis.reshape(max_iter + 1, -1)
    basis_2d[0].copy_(rhs.reshape(-1) / b_norm_t)
    hessenberg = torch.zeros(
        (max_iter + 1, max_iter),
        dtype=rhs.dtype,
        device=rhs.device,
    )
    e1 = torch.zeros((max_iter + 1,), dtype=rhs.dtype, device=rhs.device)
    e1[0] = b_norm_t
    coeff_buf = torch.empty((max_iter,), dtype=rhs.dtype, device=rhs.device)
    coeff2_buf = torch.empty((max_iter,), dtype=rhs.dtype, device=rhs.device)
    work = torch.empty_like(rhs).reshape(-1)
    work2 = torch.empty_like(rhs).reshape(-1)

    effective_iter = max_iter
    # Happy-breakdown tolerance, purely relative to ||rhs|| and the working precision. ``rhs == 0``
    # is already handled by the caller, so ``b_norm_t > 0`` here and the tol is strictly positive.
    # (The old ``clamp(b_norm_t, min=1.0)`` floored the scale at a magic 1.0; dropping it only makes
    # breakdown *less* eager for small ||rhs||, i.e. it runs more Arnoldi steps and is more accurate,
    # never less -- so it cannot truncate the Krylov space prematurely.)
    breakdown_tol = torch.finfo(rhs.dtype).eps * b_norm_t
    for j in range(max_iter):
        w = apply_a(basis[j]).reshape(-1)
        q = basis_2d[: j + 1]
        coeff = coeff_buf[: j + 1]
        torch.mv(q, w, out=coeff)
        hessenberg[: j + 1, j].copy_(coeff)
        torch.addmv(w, q.t(), coeff, beta=1.0, alpha=-1.0, out=work)

        coeff2 = coeff2_buf[: j + 1]
        torch.mv(q, work, out=coeff2)
        hessenberg[: j + 1, j].add_(coeff2)
        torch.addmv(work, q.t(), coeff2, beta=1.0, alpha=-1.0, out=work2)

        next_norm_t = torch.linalg.vector_norm(work2)
        hessenberg[j + 1, j] = next_norm_t
        if bool((next_norm_t <= breakdown_tol).detach().cpu()):
            effective_iter = j + 1
            break
        if j + 1 < max_iter:
            # Reached only when the breakdown check above did NOT fire, i.e.
            # ``next_norm_t > breakdown_tol > 0`` -- so the divisor is strictly positive and no
            # floor is needed (the old ``clamp(next_norm_t, min=tiny)`` was a dead no-op here).
            torch.div(work2, next_norm_t, out=basis_2d[j + 1])

    h_sub = hessenberg[: effective_iter + 1, :effective_iter]
    rhs_sub = e1[: effective_iter + 1]
    y = torch.linalg.lstsq(h_sub, rhs_sub).solution
    out = torch.empty_like(rhs)
    torch.mv(basis_2d[:effective_iter].t(), y, out=out.reshape(-1))
    return out


def _wave_backward_uniform_2d(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    receiver_log_probs,
    sp_child1, sp_child2, leaf_term_wt,
    *,
    neumann_terms,
    leaf_species_idx,
    leaf_logp,
    has_leaf_term,
    active_mask,
    sp_parent,
    max_ancestor_depth,
    pibar_row_max,
    family_idx,
    const_layout,
    leaf_logp_mode,
    use_leaf_index,
    compact_level_ptr,
    compact_level_parents,
    compact_level_child1,
    compact_level_child2,
    grad_receiver_log_probs=None,
    use_receiver_weights=True,
    self_loop_grad_targets=None,
    initial_v=None,
    self_loop_solver="neumann",
    return_last_increment=False,
    reserved_scratch_bytes=None,
):
    """Retained 2D row-block/full-species tree-reduction self-loop."""
    if Pi_star.device.type != "cuda":
        raise RuntimeError("GPUREC self-loop 2D fast path requires CUDA tensors")
    if Pi_star.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "2D self-loop fast path currently supports fp32/fp64 only",
        )
    ok, required_bytes, budget_bytes = proposal0_memory_gate(
        W,
        S,
        Pi_star.dtype,
        device=Pi_star.device,
        reserved_scratch_bytes=reserved_scratch_bytes,
    )
    if not ok:
        raise RuntimeError(
            "2D self-loop fast path estimated scratch "
            f"{required_bytes / (1024 ** 3):.2f} GiB above memory budget "
            f"{(budget_bytes or 0) / (1024 ** 3):.2f} GiB",
        )
    if const_layout not in (0, 1, 2):
        raise RuntimeError("unsupported self-loop constant layout")
    if use_leaf_index and leaf_logp_mode not in (0, 1, 2, 3):
        raise RuntimeError("unsupported leaf log-probability layout")

    device = Pi_star.device
    dtype = Pi_star.dtype
    receiver_log_probs = receiver_log_probs.to(device=device, dtype=dtype).contiguous()
    if sp_parent is None:
        raise ValueError("sp_parent is required for the retained 2D self-loop path")
    sp_parent = sp_parent.to(device=device, dtype=torch.int32).contiguous()
    if max_ancestor_depth is None:
        raise ValueError("max_ancestor_depth is required for the retained 2D self-loop path")
    max_ancestor_depth = max(1, int(max_ancestor_depth))

    if (
        compact_level_ptr is None
        or compact_level_parents is None
        or compact_level_child1 is None
        or compact_level_child2 is None
    ):
        raise ValueError("compact species levels are required for the retained 2D self-loop path")
    compact_level_ptr = compact_level_ptr.to(device=device, dtype=torch.long).contiguous()
    compact_level_parents = compact_level_parents.to(device=device, dtype=torch.int32).contiguous()
    compact_level_child1 = compact_level_child1.to(device=device, dtype=torch.int32).contiguous()
    compact_level_child2 = compact_level_child2.to(device=device, dtype=torch.int32).contiguous()

    block_w = 1
    block_s = triton.next_power_of_2(S)
    block_nodes = 128
    n_row_blocks = triton.cdiv(W, block_w)
    scratch_shape = (W, S)

    v_k = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw0 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw1 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw2 = torch.empty(scratch_shape, device=device, dtype=dtype)
    accum_self_loop_grads = self_loop_grad_targets is not None
    aw345 = None if accum_self_loop_grads else torch.empty(scratch_shape, device=device, dtype=dtype)
    aw3 = torch.empty(scratch_shape, device=device, dtype=dtype)
    aw4 = torch.empty(scratch_shape, device=device, dtype=dtype)
    spec_buf = torch.empty(scratch_shape, device=device, dtype=dtype)
    term_buf = torch.empty(scratch_shape, device=device, dtype=dtype)
    pibar_corr = torch.empty(scratch_shape, device=device, dtype=dtype)

    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for the retained 2D self-loop path")
    pibar_row_max = pibar_row_max.to(device=device, dtype=dtype).contiguous()
    skip_inactive_scratch_zero = True
    if family_idx is not None:
        family_idx = family_idx.to(device=device, dtype=torch.long).contiguous()
    else:
        family_idx = sp_parent
    requested_has_leaf_term = bool(has_leaf_term)
    use_leaf_index = bool(use_leaf_index and requested_has_leaf_term)
    has_materialized_leaf_term = leaf_term_wt is not None
    if leaf_term_wt is None:
        leaf_term_wt = leaf_logp if use_leaf_index else Pi_star
    has_leaf_term = bool(
        requested_has_leaf_term
        and (use_leaf_index or has_materialized_leaf_term)
    )
    leaf_species_arg = leaf_species_idx if use_leaf_index else sp_child1
    leaf_logp_arg = leaf_logp if use_leaf_index else leaf_term_wt
    use_child_edge_self_loop = True

    launch_options = {"num_warps": 8}

    _wave_backward_uniform_2d_precompute_kernel[(n_row_blocks,)](
        Pi_star,
        Pibar_star,
        pibar_row_max,
        dts_r if dts_r is not None else Pi_star,
        dts_r is not None,
        rhs,
        active_mask if active_mask is not None else rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        receiver_log_probs,
        sp_child1,
        sp_child2,
        sp_parent,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        family_idx,
        v_k,
        aw0,
        aw1,
        aw2,
        aw3,
        aw4,
        ws,
        W,
        S,
        Pi_star.stride(0),
        block_w,
        block_s,
        max_ancestor_depth,
        USE_LEAF_INDEX=bool(use_leaf_index),
        HAS_LEAF_TERM=bool(has_leaf_term),
        LEAF_LOGP_MODE=int(leaf_logp_mode),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
        CONST_LAYOUT=int(const_layout),
        DTYPE=_tl_float_dtype(dtype),
        USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        **launch_options,
    )

    self_loop_solver = str(self_loop_solver).strip().lower()
    jt_options = {"num_warps": 2}
    if self_loop_solver == "gmres":
        if initial_v is not None:
            raise ValueError("GMRES self-loop solve does not support initial_v")
        gmres_a_buf = torch.empty_like(v_k)
        gmres_rhs = rhs
        gmres_active_mask = active_mask
        if active_mask is not None:
            gmres_active_mask = active_mask.to(device=device, dtype=torch.bool).contiguous()
            gmres_rhs = rhs * gmres_active_mask[:, None].to(dtype=dtype)

        def _apply_a(term_in: torch.Tensor) -> torch.Tensor:
            _wave_backward_uniform_2d_jt_kernel[(n_row_blocks,)](
                term_in,
                gmres_a_buf,
                rhs,
                gmres_active_mask if gmres_active_mask is not None else rhs,
                aw0,
                aw1,
                aw2,
                aw3,
                aw4,
                sp_child1,
                sp_child2,
                sp_parent,
                compact_level_ptr,
                compact_level_parents,
                compact_level_child1,
                compact_level_child2,
                pibar_corr,
                v_k,
                W,
                S,
                block_w,
                block_s,
                block_nodes,
                compact_level_ptr.numel() - 1,
                USE_ACTIVE_MASK=bool(gmres_active_mask is not None),
                SKIP_INACTIVE_SCRATCH_ZERO=False,
                FIXED_POINT_UPDATE=False,
                DTYPE=_tl_float_dtype(dtype),
                USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
                OUTPUT_A=True,
                ACCUMULATE_V=False,
                **jt_options,
            )
            return gmres_a_buf

        v_k.copy_(
            _gmres_solve_wave_self_loop(
                _apply_a,
                gmres_rhs,
                max_iter=int(neumann_terms),
            )
        )
    elif self_loop_solver == "neumann" and initial_v is not None:
        if tuple(initial_v.shape) != scratch_shape:
            raise ValueError(
                f"initial_v shape {tuple(initial_v.shape)} does not match "
                f"wave scratch shape {scratch_shape}"
            )
        v_k.copy_(initial_v.to(device=device, dtype=dtype).contiguous())
        for _n in range(int(neumann_terms)):
            _wave_backward_uniform_2d_jt_kernel[(n_row_blocks,)](
                v_k,
                spec_buf,
                rhs,
                active_mask if active_mask is not None else rhs,
                aw0,
                aw1,
                aw2,
                aw3,
                aw4,
                sp_child1,
                sp_child2,
                sp_parent,
                compact_level_ptr,
                compact_level_parents,
                compact_level_child1,
                compact_level_child2,
                pibar_corr,
                v_k,
                W,
                S,
                block_w,
                block_s,
                block_nodes,
                compact_level_ptr.numel() - 1,
                USE_ACTIVE_MASK=bool(active_mask is not None),
                SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
                FIXED_POINT_UPDATE=True,
                DTYPE=_tl_float_dtype(dtype),
                USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
                OUTPUT_A=False,
                ACCUMULATE_V=True,
                **jt_options,
            )
    elif self_loop_solver == "neumann":
        for n in range(int(neumann_terms)):
            term_in = rhs if n == 0 else (spec_buf if n % 2 == 1 else term_buf)
            term_out = spec_buf if n % 2 == 0 else term_buf
            _wave_backward_uniform_2d_jt_kernel[(n_row_blocks,)](
                term_in,
                term_out,
                rhs,
                active_mask if active_mask is not None else rhs,
                aw0,
                aw1,
                aw2,
                aw3,
                aw4,
                sp_child1,
                sp_child2,
                sp_parent,
                compact_level_ptr,
                compact_level_parents,
                compact_level_child1,
                compact_level_child2,
                pibar_corr,
                v_k,
                W,
                S,
                block_w,
                block_s,
                block_nodes,
                compact_level_ptr.numel() - 1,
                USE_ACTIVE_MASK=bool(active_mask is not None),
                SKIP_INACTIVE_SCRATCH_ZERO=bool(skip_inactive_scratch_zero),
                FIXED_POINT_UPDATE=False,
                DTYPE=_tl_float_dtype(dtype),
                USE_CHILD_EDGE_SELF_LOOP=bool(use_child_edge_self_loop),
                OUTPUT_A=False,
                ACCUMULATE_V=True,
                **jt_options,
            )
    else:
        raise ValueError(f"unsupported self-loop solver {self_loop_solver!r}")

    # Per-row relative size of the last Neumann increment = validated stiffness predictor.
    # Computed before the param-store kernel runs (which may reuse scratch buffers).
    last_increment_relres = None
    if (
        return_last_increment
        and self_loop_solver == "neumann"
        and initial_v is None
        and int(neumann_terms) > 0
    ):
        last_buf = spec_buf if (int(neumann_terms) - 1) % 2 == 0 else term_buf
        eps = torch.finfo(torch.float32).tiny
        num = last_buf.float().norm(dim=1)
        den = v_k.float().norm(dim=1).clamp_min(eps)
        relres = num / den
        # Inactive (pruned) rows hold uninitialized scratch — their adjoint is negligible,
        # so treat them as converged (0) rather than letting garbage pollute the per-family max.
        if active_mask is not None:
            row_active = active_mask.reshape(active_mask.shape[0], -1).ne(0).any(dim=1)
            relres = torch.where(row_active, relres, torch.zeros_like(relres))
        last_increment_relres = relres

    if accum_self_loop_grads:
        (
            grad_log_pD_ptr,
            grad_log_pS_ptr,
            grad_E_ptr,
            grad_Ebar_ptr,
            grad_E_s1_ptr,
            grad_E_s2_ptr,
            grad_mt_ptr,
            param_grad_vector,
        ) = self_loop_grad_targets
        aw345_ptr = aw0
    else:
        grad_log_pD_ptr = aw0
        grad_log_pS_ptr = aw0
        grad_E_ptr = aw0
        grad_Ebar_ptr = aw0
        grad_E_s1_ptr = aw0
        grad_E_s2_ptr = aw0
        grad_mt_ptr = aw0
        param_grad_vector = False
        aw345_ptr = aw345

    if grad_receiver_log_probs is not None:
        _receiver_grad_from_pibar_self_loop_kernel[(n_row_blocks,)](
            v_k,
            active_mask if active_mask is not None else rhs,
            aw1,
            aw2,
            compact_level_ptr,
            compact_level_parents,
            compact_level_child1,
            compact_level_child2,
            pibar_corr,
            grad_receiver_log_probs,
            W,
            S,
            block_w,
            block_s,
            block_nodes,
            compact_level_ptr.numel() - 1,
            USE_ACTIVE_MASK=bool(active_mask is not None),
            DTYPE=_tl_float_dtype(dtype),
            **jt_options,
        )

    _wave_backward_uniform_param_store_kernel[(n_row_blocks,)](
        Pi_star,
        Pibar_star,
        dts_r if dts_r is not None else Pi_star,
        dts_r is not None,
        v_k,
        active_mask if active_mask is not None else rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        sp_child1,
        sp_child2,
        leaf_term_wt,
        leaf_species_arg,
        leaf_logp_arg,
        family_idx,
        grad_log_pD_ptr,
        grad_log_pS_ptr,
        grad_E_ptr,
        grad_Ebar_ptr,
        grad_E_s1_ptr,
        grad_E_s2_ptr,
        grad_mt_ptr,
        aw0,
        aw1,
        aw2,
        aw345_ptr,
        aw3,
        aw4,
        ws,
        W,
        S,
        Pi_star.stride(0),
        block_w,
        block_s,
        USE_LEAF_INDEX=bool(use_leaf_index),
        HAS_LEAF_TERM=bool(has_leaf_term),
        LEAF_LOGP_MODE=int(leaf_logp_mode),
        USE_ACTIVE_MASK=bool(active_mask is not None),
        CONST_LAYOUT=int(const_layout),
        ACCUM_GRADS=bool(accum_self_loop_grads),
        PARAM_GRAD_VECTOR=bool(param_grad_vector),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    if accum_self_loop_grads:
        base = (v_k, None, None, None, None, None, None)
    else:
        base = (v_k, aw0, aw1, aw2, aw345, aw3, aw4)
    if return_last_increment:
        return (*base, last_increment_relres)
    return base


def wave_backward_uniform_fused(
    Pi_star, Pibar_star, ws, W, S,
    dts_r,
    rhs,
    mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const,
    receiver_log_probs,
    sp_child1, sp_child2, leaf_term_wt,
    neumann_terms=3,
    leaf_species_idx=None,
    leaf_logp=None,
    has_leaf_term=True,
    active_mask=None,
    sp_parent=None,
    max_ancestor_depth=None,
    pibar_row_max=None,
    family_idx=None,
    family_indexed_consts=False,
    compact_level_ptr=None,
    compact_level_parents=None,
    compact_level_child1=None,
    compact_level_child2=None,
    grad_receiver_log_probs=None,
    use_receiver_weights=True,
    self_loop_grad_targets=None,
    initial_v=None,
    self_loop_solver="neumann",
    return_last_increment=False,
    reserved_scratch_bytes=None,
):
    """Fused backward: precompute + Neumann + param VJP in one kernel per wave.

    Args:
        Pi_star: [C, S] converged Pi
        Pibar_star: [C, S] converged Pibar
        ws: wave start offset
        W: wave size
        S: number of species
        dts_r: [W, S] or None
        rhs: [W, S] incoming adjoint
        mt_squeezed, DL_const, Ebar, E, SL1_const, SL2_const:
            [S], [W, S], or [G, S] when family_indexed_consts=True
        sp_child1, sp_child2: [S] long
        leaf_term_wt: [W, S]
        neumann_terms: int
        leaf_species_idx: optional [C] row -> species leaf index, -1 for non-leaves
        leaf_logp: optional [S], [G], or [G, S] log_pS values used with
            leaf_species_idx

    Returns:
        v_k: [W, S] Neumann-solved adjoint
        aw0, aw1, aw2, aw345, aw3, aw4: [W, S] per-element param grad contributions
    """
    requested_has_leaf_term = bool(has_leaf_term)
    use_leaf_index = (
        requested_has_leaf_term
        and leaf_species_idx is not None
        and leaf_logp is not None
    )
    const_layout = _uniform_backward_const_layout(
        DL_const, family_idx, bool(family_indexed_consts)
    )
    if bool(family_indexed_consts) and use_leaf_index:
        if leaf_logp.ndim == 1:
            leaf_logp = leaf_logp.unsqueeze(-1).expand(-1, S).contiguous()
        elif leaf_logp.ndim == 2 and int(leaf_logp.shape[1]) == 1:
            leaf_logp = leaf_logp.expand(-1, S).contiguous()
        else:
            leaf_logp = leaf_logp.contiguous()
    leaf_logp_mode = _uniform_backward_leaf_logp_mode(
        use_leaf_index, leaf_logp, family_idx, bool(family_indexed_consts)
    )
    if family_idx is not None:
        family_idx = family_idx.to(device=Pi_star.device, dtype=torch.long).contiguous()
    if sp_parent is None:
        raise ValueError("sp_parent is required for the retained backward fast path")
    sp_parent = sp_parent.to(device=Pi_star.device).contiguous()
    if Pi_star.device.type == "cuda" and sp_parent.dtype != torch.int32:
        sp_parent = sp_parent.to(dtype=torch.int32)
    if max_ancestor_depth is None:
        raise ValueError("max_ancestor_depth is required for the retained backward fast path")
    max_ancestor_depth = max(1, int(max_ancestor_depth))
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for the retained backward fast path")
    pibar_row_max = pibar_row_max.to(device=Pi_star.device, dtype=Pi_star.dtype).contiguous()

    return _wave_backward_uniform_2d(
        Pi_star,
        Pibar_star,
        ws,
        W,
        S,
        dts_r,
        rhs,
        mt_squeezed,
        DL_const,
        Ebar,
        E,
        SL1_const,
        SL2_const,
        receiver_log_probs,
        sp_child1,
        sp_child2,
        leaf_term_wt,
        neumann_terms=neumann_terms,
        leaf_species_idx=leaf_species_idx,
        leaf_logp=leaf_logp,
        has_leaf_term=requested_has_leaf_term,
        active_mask=active_mask,
        sp_parent=sp_parent,
        max_ancestor_depth=max_ancestor_depth,
        pibar_row_max=pibar_row_max,
        family_idx=family_idx,
        const_layout=const_layout,
        leaf_logp_mode=leaf_logp_mode,
        use_leaf_index=use_leaf_index,
        compact_level_ptr=compact_level_ptr,
        compact_level_parents=compact_level_parents,
        compact_level_child1=compact_level_child1,
        compact_level_child2=compact_level_child2,
        grad_receiver_log_probs=grad_receiver_log_probs,
        use_receiver_weights=use_receiver_weights,
        self_loop_grad_targets=self_loop_grad_targets,
        initial_v=initial_v,
        self_loop_solver=self_loop_solver,
        return_last_increment=return_last_increment,
        reserved_scratch_bytes=reserved_scratch_bytes,
    )


# =========================================================================
# Cross-clade DTS backward kernel
# =========================================================================


def dts_cross_backward_accum_fused(
    Pi_star, Pibar_star, v_k, ws,
    sl, sr, reduce_idx, wlsp,
    log_pD, log_pS,
    sp_child1, sp_child2,
    accumulated_rhs,
    S,
    active_mask=None,
    use_atomics=True,
    merge_s_term=False,
    grad_log_pD=None,
    grad_log_pS=None,
    grad_mt=None,
    accum_param_reductions=False,
    accum_mt_reduction=False,
    output_pibar_ud=False,
    output_pibar_side_active=False,
    pibar_side_threshold=0.0,
    mt_squeezed=None,
    pibar_row_max=None,
    grad_mt_two_stage=False,
    grad_mt_two_stage_tile_splits=128,
    skip_inactive_pibar_output_zero=False,
    family_idx=None,
):
    """Fused DTS backward with direct Pi-adjoint accumulation."""
    n_ws = sl.shape[0]
    device = Pi_star.device
    dtype = Pi_star.dtype

    wlsp_flat = wlsp.squeeze(-1) if wlsp.ndim > 1 else wlsp
    family_idx_arg = None
    if family_idx is not None:
        family_idx_arg = family_idx.to(device=device, dtype=torch.long).contiguous()
    log_pD_arg, log_pS_arg, param_layout = _dts_layout_param_args(
        log_pD, log_pS, family_idx=family_idx_arg, S=S, device=device, dtype=dtype
    )
    device_scalar_params = False
    if param_layout == 0:
        device_scalar_params = True

    if accum_param_reductions and (grad_log_pD is None or grad_log_pS is None):
        raise ValueError("grad_log_pD/grad_log_pS are required when accumulating DTS scalar reductions")
    if accum_param_reductions:
        param_grad_layout = _dts_grad_layout(grad_log_pD, family_idx=family_idx_arg, S=S)
        if param_grad_layout != _dts_grad_layout(grad_log_pS, family_idx=family_idx_arg, S=S):
            raise ValueError("grad_log_pD/grad_log_pS must use the same DTS gradient layout")
    else:
        param_grad_layout = 0
    if accum_mt_reduction and grad_mt is None:
        raise ValueError("grad_mt is required when accumulating DTS mt reductions")
    if accum_mt_reduction:
        if grad_mt.numel() == 1 or (grad_mt.ndim == 1 and int(grad_mt.shape[0]) == int(S)):
            grad_mt_layout = 0
        elif family_idx_arg is not None and grad_mt.ndim == 2 and int(grad_mt.shape[1]) == int(S):
            grad_mt_layout = 1
        else:
            raise ValueError("DTS mt reduction target must be scalar, [S], or [G, S]")
    else:
        grad_mt_layout = 0
    if output_pibar_ud and (mt_squeezed is None or pibar_row_max is None):
        raise ValueError("mt_squeezed and pibar_row_max are required when outputting Pibar u_d")
    if output_pibar_ud:
        if mt_squeezed.ndim == 1 and int(mt_squeezed.shape[0]) == int(S):
            mt_layout = 0
        elif family_idx_arg is not None and mt_squeezed.ndim == 2 and int(mt_squeezed.shape[1]) == int(S):
            mt_layout = 1
        else:
            raise ValueError("mt_squeezed must have shape [S] or [G, S] when outputting Pibar u_d")
    else:
        mt_layout = 0
    if output_pibar_ud and pibar_row_max.numel() < Pi_star.shape[0]:
        raise ValueError("pibar_row_max must contain one row-max value per Pi row")
    if output_pibar_side_active and not output_pibar_ud:
        raise ValueError("output_pibar_side_active requires output_pibar_ud")
    if torch.is_tensor(pibar_side_threshold):
        if pibar_side_threshold.numel() != 1:
            raise ValueError("pibar_side_threshold tensor must contain one value")
        side_threshold_enabled = bool(output_pibar_side_active)
        side_active_threshold_arg = _device_scalar_param(
            pibar_side_threshold, device=device, dtype=dtype
        )
    else:
        pibar_side_threshold = float(pibar_side_threshold)
        if pibar_side_threshold < 0.0:
            raise ValueError("pibar_side_threshold must be non-negative")
        side_threshold_enabled = bool(output_pibar_side_active and pibar_side_threshold > 0.0)
        side_active_threshold_arg = (
            torch.tensor([pibar_side_threshold], device=device, dtype=dtype)
            if side_threshold_enabled
            else None
        )

    if output_pibar_ud:
        grad_Pibar_l = None
        grad_Pibar_r = None
    else:
        grad_Pibar_l = torch.empty((n_ws, S), device=device, dtype=dtype)
        grad_Pibar_r = torch.empty((n_ws, S), device=device, dtype=dtype)
    if output_pibar_ud:
        pibar_ud = torch.empty((2 * n_ws, S), device=device, dtype=dtype)
        pibar_A = torch.empty((2 * n_ws,), device=device, dtype=dtype)
    else:
        pibar_ud = None
        pibar_A = None
    pibar_side_active = (
        torch.empty((2 * n_ws,), device=device, dtype=torch.bool)
        if output_pibar_side_active
        else None
    )
    if accum_param_reductions:
        param_pD = None
        param_pS = None
    else:
        param_pD = torch.empty(n_ws, device=device, dtype=dtype)
        param_pS = torch.empty(n_ws, device=device, dtype=dtype)
    param_pD_arg = grad_log_pD if accum_param_reductions else param_pD
    param_pS_arg = grad_log_pS if accum_param_reductions else param_pS
    dummy = pibar_ud if output_pibar_ud else grad_Pibar_l
    grad_log_pD_arg = grad_log_pD if accum_param_reductions else dummy
    grad_log_pS_arg = grad_log_pS if accum_param_reductions else dummy
    grad_mt_arg = grad_mt if accum_mt_reduction else dummy
    pibar_ud_arg = pibar_ud if output_pibar_ud else dummy
    pibar_A_arg = pibar_A if output_pibar_ud else dummy
    pibar_side_active_arg = pibar_side_active if output_pibar_side_active else dummy
    mt_arg = mt_squeezed.contiguous() if output_pibar_ud and not mt_squeezed.is_contiguous() else mt_squeezed
    pibar_row_max_arg = (
        pibar_row_max.contiguous()
        if output_pibar_ud and not pibar_row_max.is_contiguous()
        else pibar_row_max
    )
    mt_arg = mt_arg if output_pibar_ud else dummy
    pibar_row_max_arg = pibar_row_max_arg if output_pibar_ud else dummy
    side_active_threshold_arg = side_active_threshold_arg if side_threshold_enabled else dummy
    family_idx_kernel_arg = family_idx_arg if family_idx_arg is not None else sl
    grad_mt_scalar = bool(accum_mt_reduction and grad_mt.numel() == 1)
    use_grad_mt_two_stage = bool(
        grad_mt_two_stage
        and accum_mt_reduction
        and grad_mt_layout == 0
        and not grad_mt_scalar
        and grad_mt.numel() == S
    )
    grad_mt_two_stage_tile_splits = max(1, int(grad_mt_two_stage_tile_splits))
    n_grad_mt_tiles = triton.cdiv(n_ws, grad_mt_two_stage_tile_splits)
    if use_grad_mt_two_stage:
        grad_mt_partial = torch.empty((n_grad_mt_tiles, S), device=device, dtype=dtype)
        grad_mt_partial.zero_()
    else:
        grad_mt_partial = dummy

    stride_C = Pi_star.stride(0)
    BLOCK_S = min(256, triton.next_power_of_2(S))
    launch_options = {"num_warps": 8}

    _dts_cross_backward_accum_kernel[(n_ws,)](
        Pi_star, Pibar_star,
        v_k,
        active_mask if active_mask is not None else v_k,
        sl, sr, reduce_idx, wlsp_flat,
        log_pD_arg, log_pS_arg, family_idx_kernel_arg,
        sp_child1, sp_child2,
        accumulated_rhs,
        grad_Pibar_l if grad_Pibar_l is not None else pibar_ud,
        grad_Pibar_r if grad_Pibar_r is not None else pibar_ud,
        param_pD_arg, param_pS_arg,
        grad_log_pD_arg, grad_log_pS_arg, grad_mt_arg,
        grad_mt_partial,
        pibar_ud_arg, pibar_A_arg, pibar_side_active_arg, mt_arg, pibar_row_max_arg,
        side_active_threshold_arg,
        ws, S, stride_C, BLOCK_S,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_ATOMICS=bool(use_atomics),
        MERGE_S_TERM=bool(merge_s_term),
        DEVICE_SCALAR_PARAMS=bool(device_scalar_params),
        PARAM_LAYOUT=int(param_layout),
        PARAM_GRAD_LAYOUT=int(param_grad_layout),
        MT_LAYOUT=int(mt_layout),
        GRAD_MT_LAYOUT=int(grad_mt_layout),
        ACCUM_PARAM_REDUCTIONS=bool(accum_param_reductions),
        ACCUM_MT_REDUCTION=bool(accum_mt_reduction),
        GRAD_MT_SCALAR=bool(grad_mt_scalar),
        GRAD_MT_TWO_STAGE=bool(use_grad_mt_two_stage),
        GRAD_MT_TILE_SPLITS=grad_mt_two_stage_tile_splits,
        OUTPUT_PIBAR_UD=bool(output_pibar_ud),
        OUTPUT_SIDE_ACTIVE=bool(output_pibar_side_active),
        SIDE_ACTIVE_THRESHOLD_ENABLED=side_threshold_enabled,
        SKIP_INACTIVE_PIBAR_OUTPUT_ZERO=bool(skip_inactive_pibar_output_zero),
        DTYPE=_tl_float_dtype(dtype),
        **launch_options,
    )

    if use_grad_mt_two_stage:
        mt_block_s = min(64, triton.next_power_of_2(S))
        mt_block_tiles = 16
        _dts_grad_mt_two_stage_reduce_kernel[(triton.cdiv(S, mt_block_s),)](
            grad_mt_partial,
            grad_mt,
            n_grad_mt_tiles,
            S,
            mt_block_tiles,
            mt_block_s,
            DTYPE=_tl_float_dtype(dtype),
            num_warps=4,
        )

    if output_pibar_ud:
        if output_pibar_side_active:
            return pibar_ud, pibar_A, pibar_side_active, param_pD, param_pS
        return pibar_ud, pibar_A, param_pD, param_pS
    return grad_Pibar_l, grad_Pibar_r, param_pD, param_pS


# =========================================================================
# Uniform Pibar VJP for cross-clade gradients
# =========================================================================


def uniform_cross_pibar_vjp_tree_from_ud_fused(
    Pi_star,
    receiver_log_probs,
    pibar_ud,
    pibar_A,
    sl,
    sr,
    accumulated_rhs,
    S,
    active_mask=None,
    reduce_idx=None,
    pibar_row_max=None,
    skip_zero_sides=False,
    side_active=None,
    compact_level_ptr=None,
    compact_level_parents=None,
    compact_level_child1=None,
    compact_level_child2=None,
    grad_receiver_log_probs=None,
    use_receiver_weights=True,
    side_active_threshold=0.0,
):
    """Uniform-Pibar VJP tree correction from DTS-staged u_d."""
    n_ws = sl.shape[0]
    if n_ws == 0:
        return
    if active_mask is not None and reduce_idx is None:
        raise ValueError("reduce_idx is required when active_mask is provided")
    if pibar_row_max is None:
        raise ValueError("pibar_row_max is required for DTS-staged Pibar VJP")
    if torch.is_tensor(side_active_threshold):
        if side_active_threshold.numel() != 1:
            raise ValueError("side_active_threshold tensor must contain one value")
        side_active_threshold_enabled = True
        side_active_threshold_arg = _device_scalar_param(
            side_active_threshold, device=Pi_star.device, dtype=Pi_star.dtype
        )
    else:
        side_active_threshold = float(side_active_threshold)
        if side_active_threshold < 0.0:
            raise ValueError("side_active_threshold must be non-negative")
        side_active_threshold_enabled = side_active_threshold > 0.0
        side_active_threshold_arg = (
            torch.tensor([side_active_threshold], device=Pi_star.device, dtype=Pi_star.dtype)
            if side_active_threshold_enabled
            else None
        )

    BLOCK_S = min(256, triton.next_power_of_2(S))
    launch_options = {"num_warps": 4}
    stride_C = Pi_star.stride(0)
    if side_active is not None:
        if side_active.numel() != 2 * n_ws:
            raise ValueError("side_active must have one entry per split side")
        side_active = side_active.contiguous()
    elif skip_zero_sides:
        side_active = torch.empty((2 * n_ws,), device=Pi_star.device, dtype=torch.bool)
        _pibar_ud_side_active_kernel[(2 * n_ws,)](
            pibar_ud,
            side_active,
            side_active_threshold_arg if side_active_threshold_enabled else pibar_ud,
            S,
            BLOCK_S,
            SIDE_ACTIVE_THRESHOLD_ENABLED=bool(side_active_threshold_enabled),
            DTYPE=_tl_float_dtype(Pi_star.dtype),
            **launch_options,
        )

    if (
        compact_level_ptr is None
        or compact_level_parents is None
        or compact_level_child1 is None
        or compact_level_child2 is None
    ):
        raise ValueError("compact species levels are required for Pibar VJP")
    if compact_level_ptr.numel() < 2:
        raise ValueError("compact_level_ptr must contain at least start and end offsets")
    compact_level_ptr = compact_level_ptr.contiguous()
    compact_level_parents = compact_level_parents.contiguous()
    compact_level_child1 = compact_level_child1.contiguous()
    compact_level_child2 = compact_level_child2.contiguous()
    receiver_log_probs = receiver_log_probs.to(device=Pi_star.device, dtype=Pi_star.dtype).contiguous()
    receiver_grad_arg = (
        grad_receiver_log_probs
        if grad_receiver_log_probs is not None
        else pibar_A
    )
    _uniform_cross_pibar_vjp_tree_from_ud_compact_kernel[(2 * n_ws,)](
        Pi_star,
        receiver_log_probs,
        pibar_ud,
        pibar_A,
        side_active if side_active is not None else pibar_A,
        sl,
        sr,
        reduce_idx if reduce_idx is not None else sl,
        active_mask if active_mask is not None else pibar_ud,
        pibar_row_max,
        compact_level_ptr,
        compact_level_parents,
        compact_level_child1,
        compact_level_child2,
        accumulated_rhs,
        receiver_grad_arg,
        n_ws,
        S,
        stride_C,
        BLOCK_S,
        N_LEVELS=compact_level_ptr.numel() - 1,
        USE_ACTIVE_MASK=bool(active_mask is not None),
        USE_SIDE_ACTIVE=bool(side_active is not None),
        ACCUM_RECEIVER_GRAD=bool(grad_receiver_log_probs is not None),
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(Pi_star.dtype),
        **launch_options,
    )
    return side_active
