import math
from dataclasses import dataclass

import torch

from gpurec.api._implicit_grad import _logsumexp2, implicit_grad_loglik_vjp_wave
from gpurec.api.solver_options import SolverOptions
from gpurec.core.inference.forward import Pi_wave_forward
from gpurec.core.kernels.e_step import e_fixed_point_triton
from gpurec.core.parameters.extract_parameters import extract_parameters_uniform
from gpurec.core.scheduling.batching import (
    build_wave_layout,
    build_wave_layout_from_plan,
    plan_batch_wave_layouts,
    preprocess_dataset,
)


_MODE_FLAGS = {
    "global": (False, False),
    "specieswise": (False, True),
    "genewise": (True, False),
}
_UNSET = object()


def _mode_flags(mode: str) -> tuple[bool, bool]:
    try:
        return _MODE_FLAGS[str(mode).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"mode must be one of {sorted(_MODE_FLAGS)}, got {mode!r}") from exc


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
        pi_iters=solver_options.pi_iters,
    )
    return E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max, log_p_s, log_p_d, log_p_l, max_transfer


def _nll_from_root_rows(root_rows: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    return -(
        _logsumexp2(root_rows, dim=-1)
        - math.log2(root_rows.shape[-1])
        - torch.log2(1 - torch.exp2(E).mean(dim=-1))
    ).sum()


@dataclass
class _BatchStatic:
    wave_layout: dict
    species_helpers: dict
    genewise: bool
    specieswise: bool
    rate_family_idx: torch.Tensor
    family_indices: list[int]
    family_index_tensor: torch.Tensor
    solver_options: SolverOptions
    warm_E: torch.Tensor | None = None


class _GeneReconFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, static):
        with torch.no_grad():
            E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max, log_pS, log_pD, log_pL, max_transfer_vec = solve_resident_e_pi(
                static, theta, warm_start_E=static.warm_E
            )
            loss = _nll_from_root_rows(root_rows, E)

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
                specieswise=static.specieswise,
                genewise=static.genewise,
                neumann_terms=static.solver_options.neumann_terms,
                bicgstab_max_iter=static.solver_options.bicgstab_max_iter,
                bicgstab_tol=static.solver_options.bicgstab_tol,
                bicgstab_breakdown_tol=static.solver_options.bicgstab_breakdown_tol,
                adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
                use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
                pibar_side_threshold=static.solver_options.pibar_side_threshold,
            ) * grad_output,
            None,
        )


def _evaluate_static_loss_grad(static: _BatchStatic, theta: torch.Tensor, *, need_grad: bool):
    with torch.no_grad():
        E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max, log_pS, log_pD, log_pL, max_transfer_vec = solve_resident_e_pi(
            static, theta, warm_start_E=static.warm_E
        )
        loss = _nll_from_root_rows(root_rows, E).detach()
        static.warm_E = E.detach()
        if not need_grad:
            return loss, None
        grad = implicit_grad_loglik_vjp_wave(
            static.wave_layout,
            static.species_helpers,
            Pi_star_wave=pi_wave,
            Pibar_star_wave=pibar_wave,
            E_star=E,
            Ebar=Ebar,
            E_s1=E_s1,
            E_s2=E_s2,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            max_transfer_mat=max_transfer_vec,
            theta=theta,
            family_idx=static.rate_family_idx,
            uniform_pibar_row_max=pibar_row_max,
            specieswise=static.specieswise,
            genewise=static.genewise,
            neumann_terms=static.solver_options.neumann_terms,
            bicgstab_max_iter=static.solver_options.bicgstab_max_iter,
            bicgstab_tol=static.solver_options.bicgstab_tol,
            bicgstab_breakdown_tol=static.solver_options.bicgstab_breakdown_tol,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
        ).detach()
    return loss, grad


class _GeneReconFullLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, model):
        loss, grad = model._stream_batches(theta, need_grad=True)
        if grad is None:
            raise RuntimeError("missing streamed gradient")
        ctx.save_for_backward(grad.to(device=theta.device, dtype=theta.dtype))
        return loss.to(device=theta.device, dtype=theta.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (grad,) = ctx.saved_tensors
        return grad * grad_output.to(device=grad.device, dtype=grad.dtype), None


class GeneReconModel(torch.nn.Module):
    def __init__(
        self,
        species_tree,
        gene_trees,
        *,
        mode: str = "global",
        device="cuda",
        family_chunk_size: int | None = 300,
        clade_budget: int | None = 315_000,
        batch_packing: str = "depth_first_fit",
        max_wave_size: int = 8192,
        solver_options: SolverOptions | dict | None = None,
    ):
        super().__init__()
        device = torch.device(device)
        mode = str(mode).strip().lower()
        genewise, specieswise = _mode_flags(mode)
        self.solver_options = solver_options

        raw = preprocess_dataset(
            str(species_tree),
            gene_trees,
            family_chunk_size=family_chunk_size,
            clade_budget=clade_budget,
            batch_packing=batch_packing,
            max_wave_size=max_wave_size,
        )
        species_raw = raw["species"]
        families = raw["families"]
        species_helpers = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in species_raw.items()}
        self.families = families
        batches = raw.get("batches") or [list(range(len(families)))]
        if not batches:
            batches = [list(range(len(families)))]
        batch_wave_layouts = raw.get("batch_wave_layouts") or [None] * len(batches)
        theta_shape = (len(families), 3) if genewise else ((int(species_helpers["S"]), 3) if specieswise else (3,))
        self.theta = torch.nn.Parameter(
            torch.full(theta_shape, math.log2(1e-10), dtype=torch.float32, device=device)
        )
        self.species_helpers = species_helpers
        self.mode = mode
        self.genewise = genewise
        self.specieswise = specieswise
        self.family_chunk_size = family_chunk_size
        self.clade_budget = clade_budget
        self.batch_packing = batch_packing
        self.max_wave_size = max_wave_size
        self.family_batches = [list(batch) for batch in batches]
        self.batch_wave_layouts = batch_wave_layouts
        self.batch_statics = [
            self._make_batch_static(batch, plan, device=device, max_wave_size=max_wave_size)
            for batch, plan in zip(self.family_batches, self.batch_wave_layouts)
        ]
        first = self.batch_statics[0]
        self.wave_layout = first.wave_layout
        self.rate_family_idx = first.rate_family_idx
        self.warm_E = first.warm_E

    def _make_batch_static(self, batch, plan=None, *, device: torch.device, max_wave_size: int) -> _BatchStatic:
        families = [self.families[index] for index in batch]
        wave_layout = (
            build_wave_layout_from_plan(plan, device=device)
            if plan is not None
            else build_wave_layout(families, device=device, max_wave_size=max_wave_size)
        )
        rate_family_idx = wave_layout["family_idx"] if self.genewise else torch.zeros_like(wave_layout["family_idx"])
        return _BatchStatic(
            wave_layout=wave_layout,
            species_helpers=self.species_helpers,
            genewise=self.genewise,
            specieswise=self.specieswise,
            rate_family_idx=rate_family_idx,
            family_indices=list(batch),
            family_index_tensor=torch.tensor(batch, dtype=torch.long, device=device),
            solver_options=self.solver_options,
        )

    @property
    def solver_options(self) -> SolverOptions:
        return self._solver_options

    @solver_options.setter
    def solver_options(self, options: SolverOptions | dict | None) -> None:
        if options is None:
            options = SolverOptions()
        elif isinstance(options, dict):
            options = SolverOptions(**options)
        elif not isinstance(options, SolverOptions):
            raise TypeError("solver_options must be a SolverOptions instance, dict, or None")

        options.validate()
        self._solver_options = options
        for static in getattr(self, "batch_statics", []):
            static.solver_options = options

    def configure_solver(self, **kwargs) -> SolverOptions:
        for name, value in kwargs.items():
            if not hasattr(self.solver_options, name):
                raise AttributeError(f"unknown solver option {name!r}")
            setattr(self.solver_options, name, value)
        self.solver_options.validate()
        return self.solver_options

    def clear_warm_starts(self) -> None:
        for static in self.batch_statics:
            static.warm_E = None
        self.warm_E = None

    def _theta_for_static(self, static: _BatchStatic, theta: torch.Tensor) -> torch.Tensor:
        return theta.index_select(0, static.family_index_tensor) if self.genewise else theta

    def _stream_batches(self, theta: torch.Tensor, *, need_grad: bool):
        total = torch.zeros((), dtype=theta.dtype, device=theta.device)
        grad_total = torch.zeros_like(theta) if need_grad else None
        for static in self.batch_statics:
            theta_batch = self._theta_for_static(static, theta)
            loss_i, grad_i = _evaluate_static_loss_grad(static, theta_batch, need_grad=need_grad)
            total = total + loss_i.to(device=theta.device, dtype=theta.dtype)
            if need_grad:
                if grad_i is None or grad_total is None:
                    raise RuntimeError("missing batch gradient")
                if self.genewise:
                    grad_total.index_add_(0, static.family_index_tensor, grad_i.to(device=theta.device, dtype=theta.dtype))
                else:
                    grad_total.add_(grad_i.to(device=theta.device, dtype=theta.dtype))
        return total.detach(), None if grad_total is None else grad_total.detach()

    def replan_batches(
        self,
        *,
        family_chunk_size=_UNSET,
        clade_budget=_UNSET,
        batch_packing=_UNSET,
        max_wave_size=_UNSET,
    ):
        self.family_chunk_size = self.family_chunk_size if family_chunk_size is _UNSET else family_chunk_size
        self.clade_budget = self.clade_budget if clade_budget is _UNSET else clade_budget
        self.batch_packing = self.batch_packing if batch_packing is _UNSET else batch_packing
        self.max_wave_size = self.max_wave_size if max_wave_size is _UNSET else max_wave_size
        raw_plan = plan_batch_wave_layouts(
            self.families,
            family_chunk_size=self.family_chunk_size,
            clade_budget=self.clade_budget,
            batch_packing=self.batch_packing,
            max_wave_size=self.max_wave_size,
        )
        self.family_batches = raw_plan["batches"]
        self.batch_wave_layouts = raw_plan["batch_wave_layouts"]
        self.batch_statics = [
            self._make_batch_static(batch, plan, device=self.theta.device, max_wave_size=self.max_wave_size)
            for batch, plan in zip(self.family_batches, self.batch_wave_layouts)
        ]
        self.wave_layout = self.batch_statics[0].wave_layout
        self.rate_family_idx = self.batch_statics[0].rate_family_idx
        self.warm_E = self.batch_statics[0].warm_E
        return self.family_batches

    def forward(self) -> torch.Tensor:
        if len(self.batch_statics) == 1:
            return _GeneReconFunction.apply(self.theta, self.batch_statics[0])
        if torch.is_grad_enabled() and self.theta.requires_grad:
            return _GeneReconFullLossFunction.apply(self.theta, self)
        loss, _ = self._stream_batches(self.theta, need_grad=False)
        return loss
