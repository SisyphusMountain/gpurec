from dataclasses import dataclass

import torch

from gpurec.api.solver_options import SolverOptions
from gpurec.config.precision import PrecisionOptions
from gpurec.core.pi_state import PiState
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
    precision_options: PrecisionOptions
    accumulator_dtype: torch.dtype
    pi_forward_state: PiState | None = None
    warm_E: torch.Tensor | None = None
    # Per-species fraction-missing leaf boundary (log2(fraction_missing_s), -inf
    # off-leaf/observed). Shared across batches; None => every gene observed.
    leaf_fm_log: torch.Tensor | None = None
    # Memory gate: may the forward keep its per-wave gene-split (DTS) rows so the backward reads
    # them instead of recomputing them? Resolved once per (re)build by
    # GeneReconModel._resolve_forward_gene_split_gate. False here is the historical behaviour
    # (always recompute), which is what a static built outside the model gets.
    forward_gene_split_ok: bool = False
    # The kept rows themselves, {wave start: (rows [W,S], row offsets [W])}. The GRADIENT entry
    # points install an empty dict before the forward solve (which is the request: a dict means
    # "fill me"), read it in the backward, and drop it again straight after -- a pure-forward call
    # leaves it None and the forward frees each wave's block as it always did.
    forward_gene_split: dict | None = None


def build_batch_static(
    families,
    batch,
    plan,
    *,
    species_helpers: dict,
    genewise: bool,
    specieswise: bool,
    solver_options: SolverOptions,
    precision_options: PrecisionOptions,
    accumulator_dtype: torch.dtype,
    device: torch.device,
    max_wave_size: int,
    leaf_fm_log: torch.Tensor | None = None,
) -> _BatchStatic:
    batch_families = [families[index] for index in batch]
    wave_layout = (
        build_wave_layout_from_plan(
            plan,
            device=device,
            model_dtype=precision_options.model_torch_dtype,
            accumulator_dtype=accumulator_dtype,
        )
        if plan is not None
        else build_wave_layout(
            batch_families,
            device=device,
            max_wave_size=max_wave_size,
            model_dtype=precision_options.model_torch_dtype,
            accumulator_dtype=accumulator_dtype,
        )
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
        precision_options=precision_options,
        accumulator_dtype=accumulator_dtype,
        leaf_fm_log=leaf_fm_log,
    )
