from dataclasses import dataclass

import torch

from gpurec.api.solver_options import SolverOptions
from gpurec.core.scheduling.batching import build_wave_layout, build_wave_layout_from_plan


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
    gmres_check_schedule: list[int] | None = None
    gmres_check_schedule_key: tuple | None = None
    gmres_solution_cache: list[torch.Tensor | None] | None = None
    gmres_solution_cache_key: tuple | None = None


def _gmres_wave_signature(static: _BatchStatic) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(meta["start"]), int(meta["W"]))
        for meta in static.wave_layout["wave_metas"]
    )


def _gmres_adaptive_cache_key(static: _BatchStatic) -> tuple:
    options = static.solver_options
    species_nodes = int(static.species_helpers["sp_parent"].numel())
    return (
        str(options.self_loop_solver).strip().lower(),
        int(options.neumann_terms),
        float(options.gmres_tol),
        int(options.gmres_check_interval),
        str(options.gmres_preconditioner).strip().lower(),
        float(options.gmres_diagonal_preconditioner_floor),
        bool(options.use_adjoint_pruning),
        float(options.adjoint_pruning_threshold),
        float(options.pibar_side_threshold),
        bool(static.genewise),
        bool(static.specieswise),
        tuple(int(index) for index in static.family_indices),
        species_nodes,
        str(static.family_index_tensor.device),
        _gmres_wave_signature(static),
    )


def gmres_check_schedule_for_static(static: _BatchStatic) -> list[int] | None:
    options = static.solver_options
    if (
        not bool(options.gmres_reuse_check_schedule)
        or str(options.self_loop_solver).strip().lower() != "gmres"
    ):
        static.gmres_check_schedule = None
        static.gmres_check_schedule_key = None
        return None

    key = _gmres_adaptive_cache_key(static)
    if static.gmres_check_schedule_key != key:
        static.gmres_check_schedule = []
        static.gmres_check_schedule_key = key
    if static.gmres_check_schedule is None:
        static.gmres_check_schedule = []
    return static.gmres_check_schedule


def gmres_solution_cache_for_static(static: _BatchStatic) -> list[torch.Tensor | None] | None:
    options = static.solver_options
    if (
        not bool(options.gmres_reuse_solution)
        or str(options.self_loop_solver).strip().lower() != "gmres"
    ):
        static.gmres_solution_cache = None
        static.gmres_solution_cache_key = None
        return None

    key = _gmres_adaptive_cache_key(static) + (
        int(options.gmres_solution_cache_min_iterations),
    )
    if static.gmres_solution_cache_key != key:
        static.gmres_solution_cache = []
        static.gmres_solution_cache_key = key
    if static.gmres_solution_cache is None:
        static.gmres_solution_cache = []
    return static.gmres_solution_cache


def build_batch_static(
    families,
    batch,
    plan,
    *,
    species_helpers: dict,
    genewise: bool,
    specieswise: bool,
    solver_options: SolverOptions,
    device: torch.device,
    max_wave_size: int,
) -> _BatchStatic:
    batch_families = [families[index] for index in batch]
    wave_layout = (
        build_wave_layout_from_plan(plan, device=device)
        if plan is not None
        else build_wave_layout(batch_families, device=device, max_wave_size=max_wave_size)
    )
    rate_family_idx = wave_layout["family_idx"] if genewise else torch.zeros_like(wave_layout["family_idx"])
    return _BatchStatic(
        wave_layout=wave_layout,
        species_helpers=species_helpers,
        genewise=genewise,
        specieswise=specieswise,
        rate_family_idx=rate_family_idx,
        family_indices=list(batch),
        family_index_tensor=torch.tensor(batch, dtype=torch.long, device=device),
        solver_options=solver_options,
    )
