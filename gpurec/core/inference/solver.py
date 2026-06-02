import math

import torch

from gpurec.core.inference.forward import pi_wave_forward
from gpurec.core.inference.logspace import logsumexp2
from gpurec.core.kernels.e_step import e_fixed_point_triton
from gpurec.core.parameters.extract_parameters import extract_parameters_uniform


def solve_resident_e_pi(static, theta: torch.Tensor, *, warm_start_E: torch.Tensor | None = None):
    solver_options = static.solver_options
    solver_options.validate()
    log_p_s, log_p_d, log_p_l, max_transfer = extract_parameters_uniform(
        theta.detach(),
        static.species_helpers["unnorm_row_max"],
        specieswise=static.specieswise,
        genewise=static.genewise,
    )
    S = int(static.species_helpers["S"])
    e_shape = (int(static.wave_layout["root_clade_ids"].numel()) if static.genewise else 1, S)
    E0 = (
        warm_start_E.detach().to(theta).contiguous()
        if warm_start_E is not None
        else theta.new_full(e_shape, float(solver_options.e_init))
    )
    E, E_s1, E_s2, Ebar = e_fixed_point_triton(
        E0,
        log_pS=log_p_s,
        log_pD=log_p_d,
        log_pL=log_p_l,
        max_transfer=max_transfer,
        sp_parent=static.species_helpers["sp_parent"],
        sp_child1=static.species_helpers["sp_child1"],
        sp_child2=static.species_helpers["sp_child2"],
        max_ancestor_depth=int(static.species_helpers["max_ancestor_depth"]),
        max_iter=solver_options.e_max_iter,
        tol=solver_options.e_tol,
    )
    root_rows, pi_wave, pibar_wave, pibar_row_max = pi_wave_forward(
        wave_layout=static.wave_layout,
        species_helpers=static.species_helpers,
        e=E,
        e_bar=Ebar,
        e_s1=E_s1,
        e_s2=E_s2,
        log_p_s=log_p_s,
        log_p_d=log_p_d,
        max_transfer_mat=max_transfer,
        family_idx=static.rate_family_idx,
        pi_iters=solver_options.pi_iters,
    )
    return E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max, log_p_s, log_p_d, log_p_l, max_transfer


def nll_from_root_rows(root_rows: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    survival = (1 - torch.exp2(E).mean(dim=-1)).clamp_min(torch.finfo(E.dtype).tiny)
    return -(
        logsumexp2(root_rows, dim=-1)
        - math.log2(root_rows.shape[-1])
        - torch.log2(survival)
    ).sum()
