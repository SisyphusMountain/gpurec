import warnings

import torch

from gpurec.api.solver_options import SolverOptions
from gpurec.config import dtype_rel_tol_default as _e_adjoint_rel_tol_default, dtype_rel_tol_floor as _e_adjoint_rel_tol_floor
from gpurec.core.inference.logspace import logsumexp2 as _logsumexp2, log2_survival as _log2_survival
from gpurec.core.kernels.pi_forward import _select_log_split_probs, compute_dts_forward
from gpurec.core.kernels.wave_backward import (
    compute_active_adjoint_row_mask,
    accumulate_gene_split_event_vjp,
    accumulate_transfer_complement_vjp_from_donor_adjoint,
    solve_reconciliation_wave_vjp,
)
from gpurec.core.parameters.extract_parameters import (
    as_family_param,
    as_family_species,
    extract_parameters_weighted_receivers,
    resolve_accumulator_dtype,
)
from gpurec.core.kernels.e_step import e_step_triton_autograd

_NEG_INF = float("-inf")


def _safe_exp2_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    neg_inf = a == _NEG_INF
    a_safe = torch.where(neg_inf, torch.zeros_like(a), a)
    b_safe = torch.where(neg_inf, torch.zeros_like(b), b)
    return torch.where(neg_inf, torch.zeros_like(a), torch.exp2(a_safe - b_safe))


def _likelihood_root_seed(
    root_pi: torch.Tensor,
    origination_log_probs: torch.Tensor | None,
    *,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the configured-head root cotangent rounded to state dtype."""

    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=root_pi.dtype,
    )
    head = root_pi.to(dtype=accumulator_dtype)
    if origination_log_probs is not None:
        head = head + origination_log_probs.to(device=head.device, dtype=head.dtype)
    root_lse = _logsumexp2(head, dim=-1, keepdim=True)
    return -_safe_exp2_ratio(head, root_lse).to(dtype=root_pi.dtype)


def _likelihood_log2_survival(
    extinction: torch.Tensor,
    origination_probs: torch.Tensor | None,
    *,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate the likelihood head's survival normalizer."""

    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=extinction.dtype,
    )
    probabilities = (
        None
        if origination_probs is None
        else origination_probs.to(
            device=extinction.device,
            dtype=accumulator_dtype,
        )
    )
    return _log2_survival(extinction.to(dtype=accumulator_dtype), probabilities)


# `_e_adjoint_rel_tol_default` / `_e_adjoint_rel_tol_floor` moved to
# `gpurec.config.gpurec_config` (as `dtype_rel_tol_default` / `dtype_rel_tol_floor`)
# and re-exported above under these names: tests and other callers still import
# them from here.


@torch.no_grad()
def _neumann_e_adjoint(Av, b: torch.Tensor, *, max_iter: int = 128, tol=None):
    """Solve ``(I - J) x = b`` by Neumann series ``x = sum_k J^k b``, where ``J w = w - Av(w)``.

    The E-adjoint / GGN linear solve. No orthogonalization -> no fp32 Arnoldi-style residual
    floor. Valid because the E-step self-map Jacobian ``J`` is a contraction (the forward E
    fixed point converges), so ``(I-J)^{-1} = sum_k J^k`` converges.

    By telescoping, the true relative residual after summing terms ``k=0..N`` is
    ``||J^{N+1} b|| / ||b||``; since ``||J|| < 1`` this is bounded above by
    ``||J^N b|| / ||b||`` -- the quantity actually tracked below (``rel``) -- so ``rel`` is a
    conservative (upper-bound) proxy for the true residual, off by one power of ``J``.

    ``tol`` is a RELATIVE residual target: ``None`` -> the dtype-matched default
    (:func:`_e_adjoint_rel_tol_default`); a value below the dtype floor
    (:func:`_e_adjoint_rel_tol_floor`) is clamped up with a warning. Returns the best (smallest
    conservative-residual) iterate; raises ``RuntimeError`` only if that residual never reaches
    the acceptance floor within ``max_iter`` terms -- a genuinely non-contractive operator, never
    a solver artefact.
    """
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    dtype = b.dtype
    floor = _e_adjoint_rel_tol_floor(dtype)
    default = _e_adjoint_rel_tol_default(dtype)
    if tol is None:
        target = default
    else:
        target = float(tol)
        if target <= 0.0:
            raise ValueError("tol must be positive")
        if target < floor:
            warnings.warn(
                f"neumann tol={target:.2e} is below the {dtype} finite-precision "
                f"residual floor {floor:.2e}; clamping to the floor. A tighter "
                f"relative residual is unreachable in this precision -- use fp64.",
                RuntimeWarning, stacklevel=2,
            )
            target = floor
    # Any iterate reaching the dtype's natural floor counts as converged-to-precision.
    accept = max(target, default)

    bnorm = float(torch.linalg.vector_norm(b.reshape(-1)).detach().cpu())
    if bnorm == 0.0:
        return b.clone()

    x = b.clone()
    term = b.clone()
    best_x = x
    best_rel = float("inf")
    for _ in range(max_iter):
        term = term - Av(term)  # == J @ term
        x = x + term
        rel = float(torch.linalg.vector_norm(term.reshape(-1)).detach().cpu()) / bnorm
        if rel < best_rel:
            best_rel = rel
            best_x = x
        if rel <= target:
            return x

    if best_rel <= accept:
        return best_x
    raise RuntimeError(
        f"E-adjoint Neumann series failed to converge at conservative relative residual "
        f"{best_rel:.3e} after {max_iter} terms (target {target:.3e}, dtype {dtype}); the "
        f"self-map J is likely not a contraction (spectral radius >= 1) or needs more than "
        f"{max_iter} terms to solve."
    )


@torch.no_grad()
def implicit_grad_loglik_vjp_wave(
    wave_layout, species_helpers, *, Pi_star_wave: torch.Tensor,
    Pibar_star_wave: torch.Tensor, E_star: torch.Tensor, E_s1: torch.Tensor,
    E_s2: torch.Tensor, Ebar: torch.Tensor, log_pS: torch.Tensor,
    log_pD: torch.Tensor, log_pL: torch.Tensor, max_transfer_mat: torch.Tensor,
    receiver_log_probs: torch.Tensor,
    use_receiver_weights: bool,
    theta: torch.Tensor, receiver_weights: torch.Tensor, uniform_pibar_row_max: torch.Tensor,
    family_idx: torch.Tensor,
    leaf_fm_log: torch.Tensor | None = None,
    specieswise: bool = False,
    genewise: bool = False,
    neumann_terms: int | None = None,
    neumann_term_tol: float | None = None,
    e_adjoint_max_iter: int | None = None,
    e_adjoint_tol=None,
    adjoint_pruning_threshold: float | None = None,
    use_adjoint_pruning: bool = True,
    pibar_side_threshold: float | None = None,
    collect_backward_relres: bool = False,
    warm_v: dict | None = None,
    reserved_scratch_bytes: int | None = None,
    seed_root: torch.Tensor | None = None,
    drop_norm: bool = False,
    cache: dict | None = None,
    origination_log_probs: torch.Tensor | None = None,
    origination_probs: torch.Tensor | None = None,
    accumulator_dtype: torch.dtype | None = None,
    pi_offset: torch.Tensor,
    pibar_offset: torch.Tensor,
):
    if neumann_terms is None:
        neumann_terms = SolverOptions().neumann_terms
    neumann_terms = int(neumann_terms)
    if neumann_terms < 0:
        raise ValueError("neumann_terms must be non-negative")
    if neumann_term_tol is None:
        neumann_term_tol = SolverOptions().neumann_term_tol
    neumann_term_tol = float(neumann_term_tol)
    if neumann_term_tol < 0.0:
        raise ValueError("neumann_term_tol must be non-negative")
    if e_adjoint_max_iter is None:
        e_adjoint_max_iter = SolverOptions().e_adjoint_max_iter
    if adjoint_pruning_threshold is None:
        adjoint_pruning_threshold = SolverOptions().adjoint_pruning_threshold
    adjoint_pruning_threshold = float(adjoint_pruning_threshold)
    if adjoint_pruning_threshold < 0.0:
        raise ValueError("adjoint_pruning_threshold must be non-negative")
    if pibar_side_threshold is None:
        pibar_side_threshold = SolverOptions().pibar_side_threshold
    pibar_side_threshold = float(pibar_side_threshold)
    if pibar_side_threshold < 0.0:
        raise ValueError("pibar_side_threshold must be non-negative")

    C, S = Pi_star_wave.shape
    device = Pi_star_wave.device
    dtype = Pi_star_wave.dtype
    offset_dtype = pi_offset.dtype
    if offset_dtype not in (torch.float32, torch.float64):
        raise TypeError("pi_offset must use torch.float32 or torch.float64")
    for name, value in (("pi_offset", pi_offset), ("pibar_offset", pibar_offset)):
        if value.ndim != 1 or int(value.shape[0]) != int(C):
            raise ValueError(f"{name} must have shape [{int(C)}]")
        if value.dtype != offset_dtype:
            raise TypeError("Pi/Pibar offsets must share one accumulator dtype")
        if value.device != device:
            raise ValueError(f"{name} must be on the Pi device")
    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=offset_dtype,
    )
    if accumulator_dtype != offset_dtype:
        raise TypeError("accumulator_dtype must match the Pi/Pibar offset dtype")
    pi_offset = pi_offset.contiguous()
    pibar_offset = pibar_offset.contiguous()
    family_rows = int(E_star.shape[0])
    E_family, Ebar_family, log_pS_family, log_pD_family, max_transfer_family = (
        as_family_species(x, S, family_rows)
        for x in (E_star, Ebar, log_pS, log_pD, max_transfer_mat)
    )
    log_pD_param, log_pS_param = (as_family_param(x, family_rows, S) for x in (log_pD, log_pS))
    DL_family = 1.0 + log_pD_family + E_family
    SL1_family = log_pS_family + as_family_species(E_s2, S, family_rows)
    SL2_family = log_pS_family + as_family_species(E_s1, S, family_rows)
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_log_pD, grad_log_pS = (torch.zeros_like(x) for x in (log_pD_param, log_pS_param))
    grad_max_transfer_mat = torch.zeros_like(max_transfer_family)
    grad_receiver_log_probs = torch.zeros((S,), device=device, dtype=dtype)
    grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc = (
        torch.zeros_like(x) for x in (E_star, Ebar, E_star, E_star)
    )
    root_ids = wave_layout["root_clade_ids"]
    root_Pi = Pi_star_wave.index_select(0, root_ids)
    # Numerator seed = -softmax over the origination-weighted root rows. Uniform default
    # (origination_log_probs is None) is the plain softmax(root_Pi); a constant origination_log_probs
    # is only a shift, which softmax ignores, so this is identical at uniform.
    # seed_root=None -> the loss seed -softmax2(root_Pi_w) (production); a caller-supplied root
    # cotangent (the GGN/J^T path) is used verbatim.
    seed = (
        _likelihood_root_seed(
            root_Pi,
            origination_log_probs,
            accumulator_dtype=accumulator_dtype,
        )
        if seed_root is None
        else seed_root.to(dtype)
    )
    # The dense adjoint state retains the configured dtype.
    seed = seed.to(device=device, dtype=dtype)
    accumulated_rhs.index_copy_(0, root_ids, seed)
    def _scatter_accum(acc: torch.Tensor, family_rows_for_wave: torch.Tensor, contrib: torch.Tensor) -> None:
        if contrib.dtype != acc.dtype:
            contrib = contrib.to(dtype=acc.dtype)
        if int(family_rows) == 1:
            if acc.ndim == 1:
                acc[0] += contrib.sum()
            elif int(acc.shape[1]) == 1:
                acc[0, 0] += contrib.sum()
            else:
                acc[0] += contrib.sum(dim=0)
            return
        if acc.ndim == 1:
            acc.index_add_(0, family_rows_for_wave, contrib.sum(dim=1))
        elif int(acc.shape[1]) == 1:
            acc[:, 0].index_add_(0, family_rows_for_wave, contrib.sum(dim=1))
        else:
            acc.index_add_(0, family_rows_for_wave, contrib)

    sp_child1 = species_helpers["sp_child1"]
    sp_child2 = species_helpers["sp_child2"]
    compact_level_ptr = species_helpers["compact_level_ptr"]
    compact_level_parents = species_helpers["compact_level_parents"]
    compact_level_child1 = species_helpers["compact_level_child1"]
    compact_level_child2 = species_helpers["compact_level_child2"]
    leaf_species_idx = wave_layout["leaf_species_index"].to(device=device, dtype=torch.int32).contiguous()

    # Diagnostic: per-family max relative size of the last Neumann increment (stiffness).
    # Uses the true per-clade family map (batch-local), independent of the rate `family_idx`.
    backward_relres = None
    backward_vk_mag = None
    if collect_backward_relres:
        clade_family = wave_layout["family_idx"].to(device=device, dtype=torch.long)
        n_fam = int(clade_family.max().item()) + 1 if clade_family.numel() else 0
        backward_relres = torch.zeros(n_fam, device=device, dtype=torch.float32)
        backward_vk_mag = torch.zeros(n_fam, device=device, dtype=torch.float32)

    for meta in reversed(wave_layout["wave_metas"]):
        ws = int(meta["start"])
        W = int(meta["W"])
        init_v = warm_v.get(ws) if warm_v is not None else None   # per-wave adjoint warm-start
        rhs_k = accumulated_rhs[ws : ws + W]
        active_mask = compute_active_adjoint_row_mask(
            rhs_k,
            adjoint_pruning_threshold,
            use_pruning=bool(use_adjoint_pruning),
        ).contiguous()
        has_splits = bool(meta.get("has_splits", "sl" in meta))
        has_leaf_term = int(meta.get("phase", 1 if not has_splits else 2)) == 1
        if has_splits:
            dts_r, dts_offset = compute_dts_forward(
                Pi_star_wave.detach(),
                pi_offset,
                Pibar_star_wave.detach(),
                pibar_offset,
                meta["sl"],
                meta["sr"],
                sp_child1,
                sp_child2,
                W,
                meta["reduce_idx"],
                log_pD_param,
                log_pS_param,
                family_idx=family_idx,
                log_split_probs=_select_log_split_probs(meta, Pi_star_wave.dtype),
                n_single_split_parents=meta.get("n_eq1"),
                single_split_parent_rows=meta.get("eq1_reduce_idx"),
                multiple_split_group_ptr=meta.get("ge2_ptr"),
                multiple_split_parent_rows=meta.get("ge2_parent_ids"),
                max_splits_per_multiple_parent=meta.get("ge2_max_fanout"),
                active_parent_rows=active_mask,
                family_offset=ws,
            )
        else:
            dts_r = None
            dts_offset = None
        backward_out = solve_reconciliation_wave_vjp(
            Pi_star_wave,
            Pibar_star_wave,
            ws,
            W,
            S,
            dts_r,
            rhs_k,
            max_transfer_family,
            DL_family,
            Ebar_family,
            E_family,
            SL1_family,
            SL2_family,
            receiver_log_probs,
            sp_child1,
            sp_child2,
            None,
            neumann_terms=neumann_terms,
            neumann_term_tol=neumann_term_tol,
            leaf_species_idx=leaf_species_idx,
            leaf_logp=log_pS_family,
            # E-only fraction-missing (AleRax v1.4.0 model): the Pi backward gets
            # NO fraction-missing leaf term, matching the Pi forward. Fraction-missing
            # flows only through the E-step gradient (_e_adjoint_and_theta_vjp below,
            # which forwards `leaf_fm_log` to its e_step_triton_autograd calls).
            leaf_fm_log=None,
            has_leaf_term=has_leaf_term,
            active_mask=active_mask,
            species_parent=species_helpers["sp_parent"],
            max_ancestor_depth=int(species_helpers["max_ancestor_depth"]),
            pibar_row_max=uniform_pibar_row_max,
            family_idx=family_idx,
            family_indexed_consts=True,
            compact_level_ptr=species_helpers["compact_level_ptr"],
            compact_level_parents=compact_level_parents,
            compact_level_child1=compact_level_child1,
            compact_level_child2=compact_level_child2,
            grad_receiver_log_probs=grad_receiver_log_probs,
            use_receiver_weights=use_receiver_weights,
            return_last_increment=collect_backward_relres,
            initial_v=init_v,
            reserved_scratch_bytes=reserved_scratch_bytes,
            pi_offset=pi_offset,
            pibar_offset=pibar_offset,
            gene_split_offset=dts_offset,
        )
        if collect_backward_relres:
            (
                v_k,
                duplication_loss_event_vjp,
                transfer_loss_event_vjp,
                transfer_event_vjp,
                speciation_leaf_event_vjp,
                speciation_child1_event_vjp,
                speciation_child2_event_vjp,
                last_relres,
            ) = backward_out
            wave_family = clade_family[ws : ws + W]
            row_active = active_mask.reshape(active_mask.shape[0], -1).ne(0).any(dim=1)
            vk_norm = torch.where(
                row_active, v_k.float().norm(dim=1), torch.zeros(W, device=device, dtype=torch.float32)
            )
            backward_vk_mag.scatter_reduce_(
                0,
                wave_family,
                vk_norm,
                reduce="amax",
                include_self=True,
            )
            if last_relres is not None:
                backward_relres.scatter_reduce_(
                    0,
                    wave_family,
                    last_relres.to(dtype=torch.float32),
                    reduce="amax",
                    include_self=True,
                )
        else:
            (
                v_k,
                duplication_loss_event_vjp,
                transfer_loss_event_vjp,
                transfer_event_vjp,
                speciation_leaf_event_vjp,
                speciation_child1_event_vjp,
                speciation_child2_event_vjp,
            ) = backward_out
        if warm_v is not None:
            # Cache the solved adjoint for next call's warm-start, but ZERO the pruned/inactive rows first
            # -- they hold uninitialized scratch, and reusing that garbage as initial_v poisons the next
            # solve (NaN in the downstream E-adjoint), especially when the active set shifts between thetas.
            # NaN-safe select (NOT multiply: inactive rows hold uninitialized scratch = NaN/inf, and
            # 0.0 * NaN = NaN, which would poison the next warm-start). torch.where drops them cleanly.
            _row_active = active_mask.reshape(active_mask.shape[0], -1).ne(0).any(dim=1)
            warm_v[ws] = torch.where(
                _row_active.unsqueeze(-1), v_k, torch.zeros((), dtype=v_k.dtype, device=v_k.device)
            ).detach()
        if cache is not None:
            # per-wave adjoint state for the exact-HVP tangent sweep (theta fixed across CG). Pruning
            # leaves inactive v_k rows uninitialized; the primal never reads them but the second-order
            # contraction does -> sanitize with the row mask.
            row_active = (active_mask.reshape(W, -1) != 0).any(dim=1)
            v_clean = torch.where(row_active.unsqueeze(1), v_k, torch.zeros_like(v_k))
            cache.setdefault("waves", []).append(dict(
                ws=ws, W=W, v=v_clean, dts_r=dts_r, dts_offset=dts_offset,
                active_mask=active_mask,
                has_splits=has_splits, has_leaf_term=has_leaf_term, meta=meta,
            ))
        family_rows_for_wave = family_idx[ws : ws + W]
        _scatter_accum(grad_log_pD, family_rows_for_wave, duplication_loss_event_vjp)
        _scatter_accum(grad_log_pS, family_rows_for_wave, speciation_leaf_event_vjp)
        _scatter_accum(
            grad_E_acc,
            family_rows_for_wave,
            duplication_loss_event_vjp + transfer_event_vjp,
        )
        _scatter_accum(grad_Ebar_acc, family_rows_for_wave, transfer_loss_event_vjp)
        _scatter_accum(grad_E_s1_acc, family_rows_for_wave, speciation_child2_event_vjp)
        _scatter_accum(grad_E_s2_acc, family_rows_for_wave, speciation_child1_event_vjp)
        _scatter_accum(grad_max_transfer_mat, family_rows_for_wave, transfer_event_vjp)
        if has_splits and dts_r is not None:
            split_left_rows = meta["sl"]
            split_right_rows = meta["sr"]
            (
                donor_adjoint,
                total_donor_adjoint,
                active_donor_side,
                _duplication_parameter_vjp,
                _speciation_parameter_vjp,
            ) = accumulate_gene_split_event_vjp(
                Pi_star_wave,
                Pibar_star_wave,
                v_k,
                ws,
                split_left_rows,
                split_right_rows,
                meta["reduce_idx"],
                _select_log_split_probs(meta, Pi_star_wave.dtype),
                log_pD_param,
                log_pS_param,
                sp_child1,
                sp_child2,
                accumulated_rhs,
                S,
                active_mask=active_mask,
                merge_s_term=True,
                grad_log_pD=grad_log_pD,
                grad_log_pS=grad_log_pS,
                grad_max_transfer=grad_max_transfer_mat,
                accum_param_reductions=True,
                accum_max_transfer_reduction=True,
                output_donor_adjoint=True,
                output_active_donor_sides=True,
                pibar_side_threshold=pibar_side_threshold,
                max_transfer=max_transfer_family,
                pibar_row_max=uniform_pibar_row_max,
                grad_max_transfer_two_stage=bool(
                    grad_max_transfer_mat.ndim == 2
                    and int(grad_max_transfer_mat.shape[0]) == 1
                ),
                grad_max_transfer_two_stage_tile_splits=128,
                skip_inactive_pibar_output_zero=True,
                family_idx=family_idx,
                pi_offset=pi_offset,
                pibar_offset=pibar_offset,
            )
            accumulate_transfer_complement_vjp_from_donor_adjoint(
                Pi_star_wave,
                receiver_log_probs,
                donor_adjoint,
                total_donor_adjoint,
                split_left_rows,
                split_right_rows,
                accumulated_rhs,
                S,
                active_mask=active_mask,
                reduce_idx=meta["reduce_idx"],
                pibar_row_max=uniform_pibar_row_max,
                skip_zero_donor_sides=True,
                active_donor_side=active_donor_side,
                compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents,
                compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2,
                grad_receiver_log_probs=grad_receiver_log_probs,
                use_receiver_weights=use_receiver_weights,
                side_active_threshold=pibar_side_threshold,
            )
    if collect_backward_relres:
        # Diagnostic short-circuit: the per-family backward residual is fully accumulated
        # from the per-wave self-loop solves; the E-adjoint solve is not needed here.
        return backward_relres, backward_vk_mag
    if cache is not None:
        cache["accum"] = dict(
            grad_E=grad_E_acc, grad_Ebar=grad_Ebar_acc, grad_E_s1=grad_E_s1_acc,
            grad_E_s2=grad_E_s2_acc, grad_log_pD=grad_log_pD, grad_log_pS=grad_log_pS,
            grad_mc=grad_max_transfer_mat, grad_col=grad_receiver_log_probs,
        )
    return _e_adjoint_and_theta_vjp(
        E_star, log_pS, log_pD, log_pL, max_transfer_mat,
        receiver_log_probs,
        use_receiver_weights,
        grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc,
        grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
        int(root_ids.numel()), theta, receiver_weights, species_helpers,
        specieswise=specieswise,
        genewise=genewise,
        leaf_fm_log=leaf_fm_log,
        drop_norm=drop_norm,
        e_adjoint_max_iter=e_adjoint_max_iter,
        e_adjoint_tol=e_adjoint_tol,
        cache=cache,
        origination_probs=origination_probs,
        accumulator_dtype=accumulator_dtype,
    )


def _e_adjoint_and_theta_vjp(
    E_star, log_pS, log_pD, log_pL, max_transfer_mat, receiver_log_probs, use_receiver_weights,
    grad_E, grad_Ebar, grad_E_s1, grad_E_s2,
    grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
    n_fam, theta, receiver_weights, species_helpers, *, specieswise, genewise,
    leaf_fm_log: torch.Tensor | None = None,
    drop_norm: bool = False,
    e_adjoint_max_iter: int | None = None,
    e_adjoint_tol=None,
    cache=None,
    origination_probs=None,
    accumulator_dtype: torch.dtype | None = None,
):
    if e_adjoint_max_iter is None:
        e_adjoint_max_iter = SolverOptions().e_adjoint_max_iter
    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=E_star.dtype,
    )
    topology_args = (
        species_helpers["sp_parent"],
        species_helpers["sp_child1"],
        species_helpers["sp_child2"],
        int(species_helpers["max_ancestor_depth"]),
    )

    E_req = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        triton_E_from_E, E_s1_from_E, E_s2_from_E, Ebar_from_E = e_step_triton_autograd(
            E_req,
            log_pS,
            log_pD,
            log_pL,
            max_transfer_mat,
            receiver_log_probs,
            *topology_args,
            use_receiver_weights=use_receiver_weights,
            leaf_fm_log=leaf_fm_log,
        )
        # ``drop_norm`` (GGN/J^T use) skips the loss's explicit E-normalization term, which is not
        # part of d(Pi_root)/dtheta. Default False -> the full real gradient (production path).
        aux_outputs = (E_s1_from_E, E_s2_from_E, Ebar_from_E)
        aux_grads = (grad_E_s1, grad_E_s2, grad_Ebar)
        if not drop_norm:
            denom = _likelihood_log2_survival(
                E_req,
                origination_probs,
                accumulator_dtype=accumulator_dtype,
            )
            direct_obj = denom.sum() if E_req.shape[0] == n_fam else (n_fam * denom).sum()
            aux_outputs = (direct_obj, *aux_outputs)
            aux_grads = (torch.ones_like(direct_obj), *aux_grads)
        (aux_to_e,) = torch.autograd.grad(aux_outputs, E_req, grad_outputs=aux_grads, retain_graph=True)
    q_E = grad_E + aux_to_e

    E_shape = E_star.shape
    q_flat = q_E.reshape(-1)

    def AG_flat(w_flat):
        wE = w_flat.view(E_shape).contiguous()
        with torch.enable_grad():
            (gE,) = torch.autograd.grad(
                triton_E_from_E,
                E_req,
                # autograd.grad treats grad_outputs as read-only, so the
                # defensive clone is unnecessary; passing wE directly avoids a
                # full E-vector copy on every E-adjoint matvec.
                grad_outputs=wE,
                retain_graph=True,
            )
        return (wE - gE).reshape(-1)

    # Linear E-adjoint solve ``(I - J) wE = q`` via Neumann series (see _neumann_e_adjoint).
    wE = _neumann_e_adjoint(
        AG_flat,
        q_flat,
        max_iter=e_adjoint_max_iter,
        tol=e_adjoint_tol,
    ).view(E_shape)
    if cache is not None:
        cache["e_side"] = dict(q_E=q_E, wE=wE, aux_to_e=aux_to_e)

    theta_req = theta.detach().requires_grad_(True)
    receiver_req = receiver_weights.detach().requires_grad_(True)
    with torch.enable_grad():
        log_pS_r, log_pD_r, log_pL_r, mt_r, receiver_log_probs_r = extract_parameters_weighted_receivers(
            theta_req,
            receiver_req,
            species_helpers,
            specieswise=specieswise,
            genewise=genewise,
            uniform_fast=not use_receiver_weights,
            accumulator_dtype=accumulator_dtype,
        )
        S = int(species_helpers["S"])
        family_rows = int(E_star.shape[0])
        log_pS_param = as_family_param(log_pS_r, family_rows, S)
        log_pD_param = as_family_param(log_pD_r, family_rows, S)
        param_loss = (
            (log_pS_param * grad_log_pS).sum()
            + (log_pD_param * grad_log_pD).sum()
            + (mt_r * grad_max_transfer_mat).sum()
            + (receiver_log_probs_r * grad_receiver_log_probs).sum()
        )
        E_from_params, _, _, Ebar_from_params = e_step_triton_autograd(
            E_star.detach(),
            log_pS_r,
            log_pD_r,
            log_pL_r,
            mt_r,
            receiver_log_probs_r,
            *topology_args,
            use_receiver_weights=use_receiver_weights,
            leaf_fm_log=leaf_fm_log,
        )
        grad_theta, grad_receiver = torch.autograd.grad(
            (param_loss, Ebar_from_params, E_from_params),
            (theta_req, receiver_req),
            grad_outputs=(torch.ones_like(param_loss), grad_Ebar, wE),
        )
    return grad_theta, grad_receiver
