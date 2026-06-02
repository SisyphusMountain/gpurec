import math

import torch

from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.inference.forward import Pi_wave_forward
from gpurec.core.kernels.e_step import e_fixed_point_triton
from gpurec.core.parameters.extract_parameters import extract_parameters_uniform
from gpurec.core.scheduling.batching import build_wave_layout, preprocess_dataset

def solve_resident_e_pi(static, theta: torch.Tensor, *, warm_start_E: torch.Tensor | None = None):
    log_p_s, log_p_d, log_p_l, max_transfer = extract_parameters_uniform(
        theta.detach(),
        static.species_helpers["unnorm_row_max"],
    )
    S = int(static.species_helpers["S"])
    e_shape = (int(static.wave_layout["root_clade_ids"].numel()) if static.genewise else 1, S)
    E0 = warm_start_E.detach().to(theta).contiguous() if warm_start_E is not None else theta.new_full(e_shape, -1.0)
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
    )
    root_rows, pi_wave, pibar_wave, pibar_row_max = Pi_wave_forward(
        wave_layout=static.wave_layout,
        species_helpers=static.species_helpers,
        E=E,
        Ebar=Ebar,
        E_s1=E_s1,
        E_s2=E_s2,
        log_pS=log_p_s,
        log_pD=log_p_d,
        max_transfer_mat=max_transfer,
        family_idx=static.rate_family_idx,
    )
    return E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max, log_p_s, log_p_d, log_p_l, max_transfer


class _GeneReconFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, static):
        with torch.no_grad():
            E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max, log_pS, log_pD, log_pL, max_transfer_vec = solve_resident_e_pi(
                static, theta, warm_start_E=static.warm_E
            )
            loss = -(
                torch.logsumexp(root_rows * math.log(2.0), dim=-1) / math.log(2.0)
                - math.log2(root_rows.shape[-1])
                - torch.log2(1 - torch.exp2(E).mean(dim=-1))
            ).sum()

        ctx.save_for_backward(
            theta,
            pi_wave,
            pibar_wave,
            E,
            E_s1,
            E_s2,
            Ebar,
            log_pS,
            log_pD,
            log_pL,
            max_transfer_vec,
            pibar_row_max,
        )
        ctx.static = static
        static.warm_E = E.detach()
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (
            theta,
            Pi_star_wave,
            Pibar_star_wave,
            E_star,
            E_s1,
            E_s2,
            Ebar,
            log_pS,
            log_pD,
            log_pL,
            max_transfer_vec,
            uniform_pibar_row_max,
        ) = ctx.saved_tensors
        static = ctx.static
        return (
            implicit_grad_loglik_vjp_wave(
                static.wave_layout,
                static.species_helpers,
                Pi_star_wave=Pi_star_wave,
                Pibar_star_wave=Pibar_star_wave,
                E_star=E_star,
                Ebar=Ebar,
                E_s1=E_s1,
                E_s2=E_s2,
                log_pS=log_pS,
                log_pD=log_pD,
                log_pL=log_pL,
                max_transfer_mat=max_transfer_vec,
                theta=theta,
                family_idx=static.rate_family_idx,
                uniform_pibar_row_max=uniform_pibar_row_max,
            ) * grad_output,
            None,
        )


class GeneReconModel(torch.nn.Module):
    def __init__(
        self,
        species_tree,
        gene_trees,
        *,
        mode: str = "global",
        device="cuda",
    ):
        super().__init__()
        device = torch.device(device)
        genewise = mode == "genewise"

        raw = preprocess_dataset(str(species_tree), gene_trees)
        species_raw = raw["species"]
        families = raw["families"]
        species_helpers = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in species_raw.items()}
        wave_layout = build_wave_layout(families, device=device)
        self.families = families
        self.theta = torch.nn.Parameter(
            torch.full((3,) if not genewise else (len(families), 3), math.log2(1e-10), dtype=torch.float32, device=device)
        )
        self.wave_layout = wave_layout
        self.species_helpers = species_helpers
        self.genewise = genewise
        self.rate_family_idx = wave_layout["family_idx"] if genewise else torch.zeros_like(wave_layout["family_idx"])
        self.warm_E = None

    def forward(self) -> torch.Tensor:
        return _GeneReconFunction.apply(self.theta, self)
