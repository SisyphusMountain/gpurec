import torch

from gpurec.core.kernels.dts_fused import compute_dts_forward
from gpurec.core.kernels.wave_backward import (
    accumulate_split_dts_vjp,
    accumulate_split_pibar_vjp,
    compute_wave_adjoint,
)
from gpurec.core.parameters.extract_parameters import (
    as_family_param,
    as_family_species,
    extract_parameters_uniform,
)
from gpurec.core.kernels.e_step import e_step_triton_autograd


@torch.no_grad()
def _bicgstab(Av, b: torch.Tensor):
    tol = 1e-7
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

    for k in range(1, 501):
        rho = torch.dot(r_hat, r)
        if float(rho.abs().detach().cpu()) <= 1e-30:
            break

        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = Av(p)
        denom = torch.dot(r_hat, v)
        if float(denom.abs().detach().cpu()) <= 1e-30:
            break

        alpha = rho / denom
        s = r - alpha * v
        rel_s = float(torch.linalg.vector_norm(s).detach().cpu()) / bnorm
        if rel_s <= tol:
            return x + alpha * p

        t = Av(s)
        tt = torch.dot(t, t)
        if float(tt.abs().detach().cpu()) <= 1e-30:
            break

        omega = torch.dot(t, s) / tt
        x = x + alpha * p + omega * s
        r = s - omega * t
        rel_res = float(torch.linalg.vector_norm(r).detach().cpu()) / bnorm
        if rel_res <= tol:
            return x
        if float(omega.abs().detach().cpu()) <= 1e-30:
            break
        rho_old = rho

    raise RuntimeError(f"E-adjoint BiCGSTAB solve failed after {k} iterations (relative residual {rel_res:.3e})")


@torch.no_grad()
def implicit_grad_loglik_vjp_wave(
    wave_layout, species_helpers, *, Pi_star_wave: torch.Tensor,
    Pibar_star_wave: torch.Tensor, E_star: torch.Tensor, E_s1: torch.Tensor,
    E_s2: torch.Tensor, Ebar: torch.Tensor, log_pS: torch.Tensor,
    log_pD: torch.Tensor, log_pL: torch.Tensor, max_transfer_mat: torch.Tensor,
    theta: torch.Tensor, uniform_pibar_row_max: torch.Tensor,
    family_idx: torch.Tensor,
):
    C, S = Pi_star_wave.shape
    device = Pi_star_wave.device
    dtype = Pi_star_wave.dtype
    family_rows = int(E_star.shape[0])
    E_family, Ebar_family, log_pS_family, log_pD_family, max_transfer_family = (
        as_family_species(x, S, family_rows)
        for x in (E_star, Ebar, log_pS, log_pD, max_transfer_mat)
    )
    log_pD_param, log_pS_param = (as_family_param(x, family_rows) for x in (log_pD, log_pS))
    DL_family = 1.0 + log_pD_family + E_family
    SL1_family = log_pS_family + as_family_species(E_s2, S, family_rows)
    SL2_family = log_pS_family + as_family_species(E_s1, S, family_rows)

    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_log_pD, grad_log_pS = (torch.zeros_like(x.reshape(-1)) for x in (log_pD_param, log_pS_param))
    grad_max_transfer_mat = torch.zeros_like(max_transfer_family)
    grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc = (
        torch.zeros_like(x) for x in (E_star, Ebar, E_star, E_star)
    )
    root_ids = wave_layout["root_clade_ids"]
    root_Pi = Pi_star_wave.index_select(0, root_ids)
    accumulated_rhs.index_copy_(
        0,
        root_ids,
        -torch.softmax(root_Pi * 0.6931471805599453, dim=-1).nan_to_num_(0.0),
    )
    for meta in reversed(wave_layout["wave_metas"]):
        ws = int(meta["start"])
        W = int(meta["W"])
        rhs_k = accumulated_rhs[ws : ws + W]
        active_parent_rows = (rhs_k.abs().amax(dim=1) >= 1e-6).contiguous()
        dts_r = (
            compute_dts_forward(
                Pi_star_wave.detach(), Pibar_star_wave.detach(), meta["sl"], meta["sr"],
                species_helpers["sp_child1"],
                species_helpers["sp_child2"],
                W,
                meta["reduce_idx"],
                log_pD_param,
                log_pS_param,
                family_idx=family_idx,
                active_parent_rows=active_parent_rows,
                family_offset=ws,
            )
            if "sl" in meta
            else None
        )
        v_k = compute_wave_adjoint(
            Pi_star_wave,
            Pibar_star_wave,
            ws,
            W,
            S,
            dts_r,
            rhs_k,
            DL_family,
            Ebar_family,
            E_family,
            SL1_family,
            SL2_family,
            species_helpers["sp_child1"],
            species_helpers["sp_child2"],
            leaf_species_idx=wave_layout["leaf_species_index"],
            leaf_logp=log_pS_family,
            has_leaf_term="sl" not in meta,
            active_parent_rows=active_parent_rows,
            sp_parent=species_helpers["sp_parent"],
            max_ancestor_depth=int(species_helpers["max_ancestor_depth"]),
            pibar_row_max=uniform_pibar_row_max,
            family_idx=family_idx,
            compact_level_ptr=species_helpers["compact_level_ptr"],
            compact_level_parents=species_helpers["compact_level_parents"],
            compact_level_child1=species_helpers["compact_level_child1"],
            compact_level_child2=species_helpers["compact_level_child2"],
            self_loop_grad_targets=(
                grad_log_pD,
                grad_log_pS,
                grad_E_acc,
                grad_Ebar_acc,
                grad_E_s1_acc,
                grad_E_s2_acc,
                grad_max_transfer_mat,
            ),
        )
        if dts_r is not None:
            sl = meta["sl"]
            sr = meta["sr"]
            pibar_ud, pibar_A = accumulate_split_dts_vjp(
                Pi_star_wave,
                Pibar_star_wave,
                v_k,
                ws,
                sl,
                sr,
                meta["reduce_idx"],
                log_pD_param,
                log_pS_param,
                species_helpers["sp_child1"],
                species_helpers["sp_child2"],
                accumulated_rhs,
                S,
                active_parent_rows=active_parent_rows,
                grad_log_pD=grad_log_pD,
                grad_log_pS=grad_log_pS,
                grad_max_transfer_mat=grad_max_transfer_mat,
                max_transfer_mat=max_transfer_family,
                pibar_row_max=uniform_pibar_row_max,
                family_idx=family_idx,
            )
            accumulate_split_pibar_vjp(
                Pi_star_wave,
                pibar_ud,
                pibar_A,
                sl,
                sr,
                accumulated_rhs,
                S,
                active_parent_rows=active_parent_rows,
                reduce_idx=meta["reduce_idx"],
                pibar_row_max=uniform_pibar_row_max,
                compact_level_ptr=species_helpers["compact_level_ptr"],
                compact_level_parents=species_helpers["compact_level_parents"],
                compact_level_child1=species_helpers["compact_level_child1"],
                compact_level_child2=species_helpers["compact_level_child2"],
            )
    return _e_adjoint_and_theta_vjp(
        E_star, log_pS, log_pD, log_pL, max_transfer_mat,
        grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc,
        grad_log_pD, grad_log_pS, grad_max_transfer_mat,
        int(root_ids.numel()), theta, species_helpers,
    )


def _e_adjoint_and_theta_vjp(
    E_star, log_pS, log_pD, log_pL, max_transfer_mat,
    grad_E, grad_Ebar, grad_E_s1, grad_E_s2,
    grad_log_pD, grad_log_pS, grad_max_transfer_mat,
    n_fam, theta, species_helpers,
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
            *topology_args,
        )
        denom = torch.log2(1 - torch.exp2(E_req).mean(dim=-1))
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

    wE = _bicgstab(AG_flat, q_flat).view(E_shape)

    theta_req = theta.detach().requires_grad_(True)
    with torch.enable_grad():
        log_pS_r, log_pD_r, log_pL_r, mt_r = extract_parameters_uniform(
            theta_req, species_helpers["unnorm_row_max"],
        )
        param_loss = (
            (log_pS_r * grad_log_pS).sum()
            + (log_pD_r * grad_log_pD).sum()
            + (mt_r * (grad_max_transfer_mat + grad_Ebar)).sum()
        )
        E_from_theta = e_step_triton_autograd(
            E_star.detach(),
            log_pS_r,
            log_pD_r,
            log_pL_r,
            mt_r,
            *topology_args,
        )[0]
        grad_theta = torch.autograd.grad(
            (param_loss, E_from_theta),
            theta_req,
            grad_outputs=(torch.ones_like(param_loss), wE),
        )[0]
    return grad_theta
