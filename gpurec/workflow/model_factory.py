from __future__ import annotations

from typing import Literal

from gpurec.api.model import GeneReconModel
from gpurec.api._validation import require_cuda_device

from .config import RunConfig


def build_alerax_workflow_model(
    config: RunConfig,
    *,
    prefetch_batches: Literal["all"] | int = "all",
) -> GeneReconModel:
    require_cuda_device(config.device, owner="gpurec production workflow")
    pi_adjoint_kwargs = (
        {
            "pi_adjoint_warmstart": True,
            "pi_adjoint_cache_update_mode": "stage",
        }
        if config.hessian_sgd_pi_adjoint_warmstart
        else {}
    )
    model = GeneReconModel.from_alerax_families(
        str(config.species_tree),
        config.families_file,
        mode=config.mode,
        start=config.start,
        max_families=config.max_families,
        device=config.device,
        dtype=config.torch_dtype,
        theta_init_rates=config.theta_init_rates,
        preprocess_cpu_cores=config.preprocess_cpu_cores,
        fixed_iters_E=config.fixed_iters_e,
        max_iters_E=config.max_iters_e,
        tol_E=config.tol_e,
        fixed_iters_Pi=config.fixed_iters_pi,
        neumann_terms=config.neumann_terms,
        adaptive_iters=config.adaptive_iters,
        adaptive_neumann_terms=config.adaptive_neumann_terms,
        convergence_check_interval=config.convergence_check_interval,
        e_logsumexp_tol=config.e_logsumexp_tol,
        pi_max_diff_tol=config.pi_max_diff_tol,
        gradient_change_tol=config.gradient_change_tol,
        gradient_change_rtol=config.gradient_change_rtol,
        family_chunk_size=config.family_chunk_size,
        clade_budget=config.clade_budget,
        batch_packing=config.batch_packing,
        max_wave_size=config.max_wave_size,
        small_family_max_leaves=config.small_family_max_leaves,
        lazy_preprocess=True,
        prefetch_batches=prefetch_batches,
        **pi_adjoint_kwargs,
    )
    return model
