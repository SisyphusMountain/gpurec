import math

import torch

from gpurec.api._autograd import _GeneReconFullLossFunction, _GeneReconFunction
from gpurec.api._batch_state import _BatchStatic, build_batch_static
from gpurec.api._execution import stream_batches, theta_for_static
from gpurec.api.solver_options import SolverOptions
from gpurec.core.scheduling.batching import plan_batch_wave_layouts, preprocess_dataset


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
        self.receiver_weights = torch.nn.Parameter(
            torch.zeros((int(species_helpers["S"]),), dtype=torch.float32, device=device)
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
        return build_batch_static(
            self.families,
            batch,
            plan,
            species_helpers=self.species_helpers,
            genewise=self.genewise,
            specieswise=self.specieswise,
            solver_options=self.solver_options,
            device=device,
            max_wave_size=max_wave_size,
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
            static.gmres_check_schedule = None
            static.gmres_check_schedule_key = None
            static.gmres_check_schedule_pass_count = 0
            static.gmres_solution_cache = None
            static.gmres_solution_cache_key = None
        self.warm_E = None

    def _theta_for_static(self, static: _BatchStatic, theta: torch.Tensor) -> torch.Tensor:
        return theta_for_static(static, theta, genewise=self.genewise)

    def _stream_batches(self, theta: torch.Tensor, receiver_weights: torch.Tensor, *, need_grad: bool):
        return stream_batches(
            self.batch_statics,
            theta,
            receiver_weights,
            genewise=self.genewise,
            need_grad=need_grad,
        )

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

    def forward(
        self,
        theta: torch.Tensor | None = None,
        receiver_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        theta = self.theta if theta is None else theta
        receiver_weights = self.receiver_weights if receiver_weights is None else receiver_weights
        if len(self.batch_statics) == 1:
            static = self.batch_statics[0]
            return _GeneReconFunction.apply(self._theta_for_static(static, theta), receiver_weights, static)
        if torch.is_grad_enabled() and (theta.requires_grad or receiver_weights.requires_grad):
            return _GeneReconFullLossFunction.apply(theta, receiver_weights, self)
        loss, _, _ = self._stream_batches(theta, receiver_weights, need_grad=False)
        return loss
