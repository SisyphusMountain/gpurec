import torch

from gpurec.core.inference.logspace import logsumexp2 as _logsumexp2
from gpurec.core.kernels.dts_fused import compute_dts_forward
from gpurec.core.kernels.wave_backward import (
    active_mask_from_rhs_absmax_fused,
    dts_cross_backward_accum_fused,
    uniform_cross_pibar_vjp_tree_from_ud_fused,
    wave_backward_uniform_fused,
)
from gpurec.core.memory_policy import cuda_memory_budget_bytes
from gpurec.core.parameters.extract_parameters import (
    as_family_param,
    as_family_species,
    extract_parameters_weighted_receivers,
)
from gpurec.core.kernels.e_step import e_step_triton_autograd

_NEG_INF = float("-inf")


def _safe_exp2_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    neg_inf = a == _NEG_INF
    a_safe = torch.where(neg_inf, torch.zeros_like(a), a)
    b_safe = torch.where(neg_inf, torch.zeros_like(b), b)
    return torch.where(neg_inf, torch.zeros_like(a), torch.exp2(a_safe - b_safe))


@torch.no_grad()
def _bicgstab(
    Av,
    b: torch.Tensor,
    *,
    max_iter: int = 500,
    tol: float = 1e-7,
    breakdown_tol: float = 1e-30,
):
    max_iter = int(max_iter)
    tol = float(tol)
    breakdown_tol = float(breakdown_tol)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if breakdown_tol <= 0.0:
        raise ValueError("breakdown_tol must be positive")

    x = torch.zeros_like(b)
    r = b - Av(x)
    bnorm = max(float(torch.linalg.vector_norm(b).detach().cpu()), 1.0)
    rel_res = float(torch.linalg.vector_norm(r).detach().cpu()) / bnorm
    if rel_res <= tol:
        return x

    r_hat = r.clone()
    rho_old = torch.ones((), dtype=b.dtype, device=b.device)
    alpha = torch.ones((), dtype=b.dtype, device=b.device)
    omega = torch.ones((), dtype=b.dtype, device=b.device)
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)

    for k in range(1, max_iter + 1):
        rho = torch.dot(r_hat, r)
        if float(rho.abs().detach().cpu()) <= breakdown_tol:
            break

        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = Av(p)
        denom = torch.dot(r_hat, v)
        if float(denom.abs().detach().cpu()) <= breakdown_tol:
            break

        alpha = rho / denom
        s = r - alpha * v
        rel_s = float(torch.linalg.vector_norm(s).detach().cpu()) / bnorm
        if rel_s <= tol:
            return x + alpha * p

        t = Av(s)
        tt = torch.dot(t, t)
        if float(tt.abs().detach().cpu()) <= breakdown_tol:
            break

        omega = torch.dot(t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        rel_res = float(torch.linalg.vector_norm(r).detach().cpu()) / bnorm
        if rel_res <= tol:
            return x
        if float(omega.abs().detach().cpu()) <= breakdown_tol:
            break
        rho_old = rho

    raise RuntimeError(f"E-adjoint BiCGSTAB solve failed after {k} iterations (relative residual {rel_res:.3e})")


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
    specieswise: bool = False,
    genewise: bool = False,
    neumann_terms: int = 3,
    self_loop_solver: str = "neumann",
    gmres_tol: float = 1e-10,
    gmres_check_interval: int = 1,
    gmres_check_schedule: list[int] | None = None,
    gmres_validate_check_schedule: bool = True,
    gmres_trust_check_schedule: bool = False,
    gmres_trusted_schedule_safety_margin: int = 0,
    gmres_solution_cache: list[torch.Tensor | None] | None = None,
    gmres_solution_cache_min_iterations: int = 2,
    gmres_preconditioner: str = "none",
    gmres_diagonal_preconditioner_floor: float = 1e-4,
    bicgstab_max_iter: int = 500,
    bicgstab_tol: float = 1e-7,
    bicgstab_breakdown_tol: float = 1e-30,
    adjoint_pruning_threshold: float = 1e-6,
    use_adjoint_pruning: bool = True,
    pibar_side_threshold: float = 0.0,
):
    neumann_terms = int(neumann_terms)
    if neumann_terms < 0:
        raise ValueError("neumann_terms must be non-negative")
    self_loop_solver = str(self_loop_solver).strip().lower()
    if self_loop_solver not in ("neumann", "gmres", "gmres_fixed"):
        raise ValueError("self_loop_solver must be one of: neumann, gmres, gmres_fixed")
    gmres_tol = float(gmres_tol)
    if gmres_tol <= 0.0:
        raise ValueError("gmres_tol must be positive")
    gmres_check_interval = int(gmres_check_interval)
    if gmres_check_interval < 1:
        raise ValueError("gmres_check_interval must be at least 1")
    adjoint_pruning_threshold = float(adjoint_pruning_threshold)
    if adjoint_pruning_threshold < 0.0:
        raise ValueError("adjoint_pruning_threshold must be non-negative")
    pibar_side_threshold = float(pibar_side_threshold)
    if pibar_side_threshold < 0.0:
        raise ValueError("pibar_side_threshold must be non-negative")

    C, S = Pi_star_wave.shape
    device = Pi_star_wave.device
    dtype = Pi_star_wave.dtype
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
    root_lse = _logsumexp2(root_Pi, dim=-1, keepdim=True)
    accumulated_rhs.index_copy_(
        0,
        root_ids,
        -_safe_exp2_ratio(root_Pi, root_lse),
    )
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
    wave_metas = wave_layout["wave_metas"]
    use_gmres_check_schedule = (
        gmres_check_schedule is not None
        and self_loop_solver == "gmres"
        and neumann_terms > 0
    )
    previous_gmres_check_schedule = (
        list(gmres_check_schedule)
        if use_gmres_check_schedule
        and len(gmres_check_schedule) == len(wave_metas)
        else None
    )
    next_gmres_check_schedule: list[int] | None = [] if use_gmres_check_schedule else None
    use_trusted_gmres_check_schedule = (
        bool(gmres_trust_check_schedule)
        and not bool(gmres_validate_check_schedule)
        and previous_gmres_check_schedule is not None
        and gmres_solution_cache is None
    )
    use_gmres_solution_cache = (
        gmres_solution_cache is not None
        and self_loop_solver == "gmres"
        and neumann_terms > 0
    )
    gmres_solution_cache_min_iterations = max(1, int(gmres_solution_cache_min_iterations))
    previous_gmres_solution_cache = (
        list(gmres_solution_cache)
        if use_gmres_solution_cache
        and len(gmres_solution_cache) == len(wave_metas)
        else None
    )
    next_gmres_solution_cache: list[torch.Tensor | None] | None = (
        [] if use_gmres_solution_cache else None
    )
    wave_memory_budget_bytes = cuda_memory_budget_bytes(device) if device.type == "cuda" else None

    for wave_rev_index, meta in enumerate(reversed(wave_metas)):
        ws = int(meta["start"])
        W = int(meta["W"])
        rhs_k = accumulated_rhs[ws : ws + W]
        active_mask = active_mask_from_rhs_absmax_fused(
            rhs_k,
            adjoint_pruning_threshold,
            use_pruning=bool(use_adjoint_pruning),
        ).contiguous()
        has_splits = bool(meta.get("has_splits", "sl" in meta))
        has_leaf_term = int(meta.get("phase", 1 if not has_splits else 2)) == 1
        dts_r = (
            compute_dts_forward(
                Pi_star_wave.detach(), Pibar_star_wave.detach(), meta["sl"], meta["sr"],
                sp_child1,
                sp_child2,
                W,
                meta["reduce_idx"],
                log_pD_param,
                log_pS_param,
                family_idx=family_idx,
                log_split_probs=meta.get("log_split_probs"),
                n_eq1=meta.get("n_eq1"),
                eq1_reduce_idx=meta.get("eq1_reduce_idx"),
                ge2_ptr=meta.get("ge2_ptr"),
                ge2_parent_ids=meta.get("ge2_parent_ids"),
                ge2_max_fanout=meta.get("ge2_max_fanout"),
                active_parent_rows=active_mask,
                family_offset=ws,
            )
            if has_splits
            else None
        )
        gmres_min_check_iter = 1
        gmres_wave_stats: dict[str, float | int | str] | None = None
        if use_gmres_check_schedule or use_gmres_solution_cache:
            if previous_gmres_check_schedule is not None:
                scheduled_iterations = int(previous_gmres_check_schedule[wave_rev_index])
                if use_trusted_gmres_check_schedule:
                    scheduled_iterations += int(gmres_trusted_schedule_safety_margin)
                gmres_min_check_iter = max(1, min(scheduled_iterations, neumann_terms))
            gmres_wave_stats = {}
        initial_v = None
        if previous_gmres_solution_cache is not None:
            cached_v = previous_gmres_solution_cache[wave_rev_index]
            if (
                cached_v is not None
                and tuple(cached_v.shape) == (W, S)
                and cached_v.device == device
                and cached_v.dtype == dtype
            ):
                initial_v = cached_v

        v_k, aw0, aw1, aw2, aw345, aw3, aw4 = wave_backward_uniform_fused(
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
            initial_v=initial_v,
            leaf_species_idx=leaf_species_idx,
            leaf_logp=log_pS_family,
            has_leaf_term=has_leaf_term,
            active_mask=active_mask,
            sp_parent=species_helpers["sp_parent"],
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
            self_loop_solver=self_loop_solver,
            gmres_tol=gmres_tol,
            gmres_check_interval=gmres_check_interval,
            gmres_min_check_iter=gmres_min_check_iter,
            gmres_trust_min_check_iter=use_trusted_gmres_check_schedule,
            gmres_stats_out=gmres_wave_stats,
            gmres_preconditioner=gmres_preconditioner,
            gmres_diagonal_preconditioner_floor=gmres_diagonal_preconditioner_floor,
            memory_budget_bytes=wave_memory_budget_bytes,
        )
        if next_gmres_check_schedule is not None:
            if use_trusted_gmres_check_schedule and previous_gmres_check_schedule is not None:
                observed_iterations = int(previous_gmres_check_schedule[wave_rev_index])
            else:
                observed_iterations = (
                    int(gmres_wave_stats["iterations"])
                    if gmres_wave_stats is not None and "iterations" in gmres_wave_stats
                    else 1
                )
            next_gmres_check_schedule.append(max(1, min(observed_iterations, neumann_terms)))
        if next_gmres_solution_cache is not None:
            observed_iterations = (
                int(gmres_wave_stats["iterations"])
                if gmres_wave_stats is not None and "iterations" in gmres_wave_stats
                else 0
            )
            warm_start_used = bool(
                gmres_wave_stats is not None
                and bool(gmres_wave_stats.get("warm_start_used", False))
            )
            warm_start_accepted = bool(
                gmres_wave_stats is not None
                and bool(gmres_wave_stats.get("warm_start_accepted", False))
            )
            keep_solution = warm_start_accepted or (
                not warm_start_used
                and observed_iterations >= gmres_solution_cache_min_iterations
            )
            next_gmres_solution_cache.append(v_k.detach().clone() if keep_solution else None)
        family_rows_for_wave = family_idx[ws : ws + W]
        _scatter_accum(grad_log_pD, family_rows_for_wave, aw0)
        _scatter_accum(grad_log_pS, family_rows_for_wave, aw345)
        _scatter_accum(grad_E_acc, family_rows_for_wave, aw0 + aw2)
        _scatter_accum(grad_Ebar_acc, family_rows_for_wave, aw1)
        _scatter_accum(grad_E_s1_acc, family_rows_for_wave, aw4)
        _scatter_accum(grad_E_s2_acc, family_rows_for_wave, aw3)
        _scatter_accum(grad_max_transfer_mat, family_rows_for_wave, aw2)
        if has_splits and dts_r is not None:
            sl = meta["sl"]
            sr = meta["sr"]
            grad_Pibar_l, grad_Pibar_r, pibar_side_active, _param_pD, _param_pS = dts_cross_backward_accum_fused(
                Pi_star_wave,
                Pibar_star_wave,
                v_k,
                ws,
                sl,
                sr,
                meta["reduce_idx"],
                meta.get("log_split_probs", sl.new_zeros((int(sl.numel()),), dtype=Pi_star_wave.dtype)),
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
                grad_mt=grad_max_transfer_mat,
                accum_param_reductions=True,
                accum_mt_reduction=True,
                output_pibar_ud=True,
                output_pibar_side_active=True,
                pibar_side_threshold=pibar_side_threshold,
                mt_squeezed=max_transfer_family,
                pibar_row_max=uniform_pibar_row_max,
                grad_mt_two_stage=bool(grad_max_transfer_mat.ndim == 2 and int(grad_max_transfer_mat.shape[0]) == 1),
                grad_mt_two_stage_tile_splits=128,
                skip_inactive_pibar_output_zero=True,
                family_idx=family_idx,
            )
            uniform_cross_pibar_vjp_tree_from_ud_fused(
                Pi_star_wave,
                receiver_log_probs,
                grad_Pibar_l,
                grad_Pibar_r,
                sl,
                sr,
                accumulated_rhs,
                S,
                active_mask=active_mask,
                reduce_idx=meta["reduce_idx"],
                pibar_row_max=uniform_pibar_row_max,
                skip_zero_sides=True,
                side_active=pibar_side_active,
                compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents,
                compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2,
                grad_receiver_log_probs=grad_receiver_log_probs,
                use_receiver_weights=use_receiver_weights,
                side_active_threshold=pibar_side_threshold,
            )
    if gmres_check_schedule is not None and next_gmres_check_schedule is not None:
        gmres_check_schedule[:] = next_gmres_check_schedule
    if gmres_solution_cache is not None and next_gmres_solution_cache is not None:
        gmres_solution_cache[:] = next_gmres_solution_cache
    return _e_adjoint_and_theta_vjp(
        E_star, log_pS, log_pD, log_pL, max_transfer_mat,
        receiver_log_probs,
        use_receiver_weights,
        grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc,
        grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
        int(root_ids.numel()), theta, receiver_weights, species_helpers,
        specieswise=specieswise,
        genewise=genewise,
        bicgstab_max_iter=bicgstab_max_iter,
        bicgstab_tol=bicgstab_tol,
        bicgstab_breakdown_tol=bicgstab_breakdown_tol,
    )


def _e_adjoint_and_theta_vjp(
    E_star, log_pS, log_pD, log_pL, max_transfer_mat, receiver_log_probs, use_receiver_weights,
    grad_E, grad_Ebar, grad_E_s1, grad_E_s2,
    grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
    n_fam, theta, receiver_weights, species_helpers, *, specieswise, genewise,
    bicgstab_max_iter: int = 500,
    bicgstab_tol: float = 1e-7,
    bicgstab_breakdown_tol: float = 1e-30,
):
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
        )
        survival = (1 - torch.exp2(E_req).mean(dim=-1)).clamp_min(torch.finfo(E_req.dtype).tiny)
        denom = torch.log2(survival)
        direct_obj = denom.sum() if E_req.shape[0] == n_fam else (n_fam * denom).sum()
        (aux_to_e,) = torch.autograd.grad(
            (direct_obj, E_s1_from_E, E_s2_from_E, Ebar_from_E),
            E_req,
            grad_outputs=(
                torch.ones_like(direct_obj),
                grad_E_s1,
                grad_E_s2,
                grad_Ebar,
            ),
            retain_graph=True,
        )
    q_E = grad_E + aux_to_e

    E_shape = E_star.shape
    q_flat = q_E.reshape(-1)

    def AG_flat(w_flat):
        wE = w_flat.view(E_shape).contiguous()
        with torch.enable_grad():
            (gE,) = torch.autograd.grad(
                triton_E_from_E,
                E_req,
                grad_outputs=wE.clone(),
                retain_graph=True,
            )
        return (wE - gE).reshape(-1)

    wE = _bicgstab(
        AG_flat,
        q_flat,
        max_iter=bicgstab_max_iter,
        tol=bicgstab_tol,
        breakdown_tol=bicgstab_breakdown_tol,
    ).view(E_shape)

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
        )
        grad_theta, grad_receiver = torch.autograd.grad(
            (param_loss, Ebar_from_params, E_from_params),
            (theta_req, receiver_req),
            grad_outputs=(torch.ones_like(param_loss), grad_Ebar, wE),
        )
    return grad_theta, grad_receiver
