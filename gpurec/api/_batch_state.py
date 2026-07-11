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
    warm_v: dict | None = None   # per-wave backward Pi-adjoint warm-start cache (keyed by wave-start ws)
    warm_adjoint_ok: bool = True  # memory gate: False -> ignore GPUREC_WARM_ADJOINT (cache won't fit), run cold
    # Transient per-wave self-loop scratch headroom (bytes) the warm-adjoint fit decision already
    # reserved at build time (``warm_adjoint_fits``'s max-batch scratch). Passed to the forward
    # self-loop gate so it trusts the build reservation instead of re-reading the depleted post-cache
    # free memory. None -> cold path (gate reads current free memory as usual). See memory_policy.
    warm_scratch_reserved_bytes: int | None = None


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
