from __future__ import annotations

from typing import Literal

import torch

from gpurec.api.model import GeneReconModel

from .config import RunConfig


def build_alerax_workflow_model(
    config: RunConfig,
    *,
    refresh_preprocess_cache: bool | None = None,
    prefetch_batches: Literal["all"] | int = "all",
) -> GeneReconModel:
    if not str(config.device).startswith("cuda"):
        raise RuntimeError("gpurec production workflow currently requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if refresh_preprocess_cache is None:
        refresh_preprocess_cache = config.refresh_preprocess_cache
    return GeneReconModel.from_alerax_families(
        str(config.species_tree),
        config.families_file,
        mode=config.mode,
        start=config.start,
        max_families=config.max_families,
        device=config.device,
        dtype=config.torch_dtype,
        theta_init_rates=config.theta_init_rates,
        preprocess_cache_dir=config.preprocess_cache,
        refresh_preprocess_cache=refresh_preprocess_cache,
        fixed_iters_E=config.fixed_iters_e,
        max_iters_E=config.max_iters_e,
        tol_E=config.tol_e,
        fixed_iters_Pi=config.fixed_iters_pi,
        neumann_terms=config.neumann_terms,
        adaptive_iters=config.adaptive_iters,
        convergence_check_interval=config.convergence_check_interval,
        e_logsumexp_tol=config.e_logsumexp_tol,
        pi_max_diff_tol=config.pi_max_diff_tol,
        gradient_change_tol=config.gradient_change_tol,
        gradient_change_rtol=config.gradient_change_rtol,
        family_chunk_size=config.family_chunk_size,
        clade_budget=config.clade_budget,
        batch_packing=config.batch_packing,
        max_wave_size=config.max_wave_size,
        lazy_preprocess=True,
        prefetch_batches=prefetch_batches,
    )
