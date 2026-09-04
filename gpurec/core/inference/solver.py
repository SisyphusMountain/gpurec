import math

import torch

from gpurec.core.inference.forward import pi_wave_forward
from gpurec.core.inference.logspace import (
    logsumexp2,
    log2_survival,
    survival_by_species_from_E,
)
from gpurec.core.kernels.e_step import e_fixed_point_triton
from gpurec.core.parameters.extract_parameters import (
    extract_parameters_uniform,
    extract_parameters_weighted_receivers,
    origination_log_probs_from_weights,
    resolve_accumulator_dtype,
)


def receiver_weights_are_uniform(receiver_weights: torch.Tensor) -> bool:
    flat = receiver_weights.detach().reshape(-1)
    return bool(torch.all(flat == flat[0]).item())


def origination_weights_are_uniform(origination_weights: torch.Tensor) -> bool:
    flat = origination_weights.detach().reshape(-1)
    return bool(torch.all(flat == flat[0]).item())


def solve_resident_e_pi(
    static,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    warm_start_E: torch.Tensor | None = None,
    pi_iters: int | None = None,
    pi_residual_out: torch.Tensor | None = None,
):
    solver_options = static.solver_options
    solver_options.validate()
    accumulator_dtype = resolve_accumulator_dtype(
        getattr(static, "accumulator_dtype", None),
        fallback=theta.dtype,
    )
    use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
    S = int(static.species_helpers["S"])
    if use_receiver_weights:
        log_p_s, log_p_d, log_p_l, max_transfer, receiver_log_probs = extract_parameters_weighted_receivers(
            theta.detach(),
            receiver_weights.detach(),
            static.species_helpers,
            specieswise=static.specieswise,
            genewise=static.genewise,
            accumulator_dtype=accumulator_dtype,
        )
    else:
        log_p_s, log_p_d, log_p_l, max_transfer = extract_parameters_uniform(
            theta.detach(),
            static.species_helpers["unnorm_row_max"].to(device=theta.device, dtype=theta.dtype),
            specieswise=static.specieswise,
            genewise=static.genewise,
            accumulator_dtype=accumulator_dtype,
        )
        receiver_log_probs = theta.new_full((S,), -math.log2(S))
    e_shape = (int(static.wave_layout["root_clade_ids"].numel()) if static.genewise else 1, S)
    E0 = (
        warm_start_E.detach().to(theta).contiguous()
        if warm_start_E is not None
        else theta.new_full(e_shape, torch.finfo(theta.dtype).min)
    )
    E, E_s1, E_s2, Ebar = e_fixed_point_triton(
        E0,
        log_pS=log_p_s,
        log_pD=log_p_d,
        log_pL=log_p_l,
        max_transfer=max_transfer,
        receiver_log_probs=receiver_log_probs,
        use_receiver_weights=use_receiver_weights,
        species_parent=static.species_helpers["sp_parent"],
        species_child1=static.species_helpers["sp_child1"],
        species_child2=static.species_helpers["sp_child2"],
        # Each species' height (0 at a leaf) and the tree's height: the extinction complement sums
        # the valid receivers level by level rather than subtracting the ancestor chain.
        species_height=static.species_helpers["sp_height"],
        species_levels=int(static.species_helpers["compact_level_ptr"].numel()) - 1,
        max_iter=solver_options.e_max_iter,
        tol=solver_options.e_tol,
        leaf_fm_log=getattr(static, "leaf_fm_log", None),
    )
    pi_forward_result = pi_wave_forward(
        wave_layout=static.wave_layout,
        species_helpers=static.species_helpers,
        e=E,
        e_bar=Ebar,
        e_s1=E_s1,
        e_s2=E_s2,
        log_p_s=log_p_s,
        log_p_d=log_p_d,
        max_transfer_mat=max_transfer,
        receiver_log_probs=receiver_log_probs,
        use_receiver_weights=use_receiver_weights,
        family_idx=static.rate_family_idx,
        pi_iters=solver_options.pi_iters if pi_iters is None else int(pi_iters),
        pi_residual_out=pi_residual_out,
        accumulator_dtype=accumulator_dtype,
        self_loop_mode=solver_options.forward_self_loop,
        exact_range_log2=solver_options.exact_range_log2,
        # Exact-solve pivot-guard counts are a benchmarking probe, not production state; the
        # benchmark wraps ``pi_wave_forward`` to pass its own tensor.
        exact_guard_trips_out=None,
        # E-only fraction-missing (AleRax v1.4.0 model): fraction-missing enters
        # ONLY the extinction E-step above; the Pi/reconciliation numerator gets
        # no fraction-missing leaf term, so None here (never `static.leaf_fm_log`).
        leaf_fm_log=None,
    )
    root_rows, pi_wave, pibar_wave, pibar_row_max, pi_state = pi_forward_result
    static.pi_forward_state = pi_state
    return (
        E,
        E_s1,
        E_s2,
        Ebar,
        root_rows,
        pi_wave,
        pibar_wave,
        pibar_row_max,
        log_p_s,
        log_p_d,
        log_p_l,
        max_transfer,
        receiver_log_probs,
    )


def solve_forward_residual(
    static,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    pi_iters: int,
    warm_start_E: torch.Tensor | None = None,
):
    """Per-family forward Pi convergence residual at a (high) ``pi_iters``.

    Runs the forward solve and captures the final iteration's per-clade
    ``max_s |Pi_new - Pi_old|`` (= the size of the last Pi update). Returns a 1-D
    float tensor of length ``n_families_in_batch`` (batch-local index), holding the
    max residual over each family's clades. Diagnostic only; meaningful only at a
    converged forward, hence the caller supplies a high ``pi_iters``.
    """
    C = int(static.wave_layout["leaf_species_index"].numel())
    pi_residual = torch.zeros(C, device=theta.device, dtype=torch.float32)
    solve_resident_e_pi(
        static,
        theta,
        receiver_weights,
        warm_start_E=warm_start_E,
        pi_iters=pi_iters,
        pi_residual_out=pi_residual,
    )
    fam_local = static.wave_layout["family_idx"].to(device=pi_residual.device, dtype=torch.long)
    n_fam = int(static.family_index_tensor.numel())
    per_family = torch.zeros(n_fam, device=pi_residual.device, dtype=torch.float32)
    per_family.scatter_reduce_(0, fam_local, pi_residual, reduce="amax", include_self=True)
    return per_family


def nll_vector_from_root_rows(
    root_rows: torch.Tensor,
    E: torch.Tensor,
    *,
    origination_log_probs: torch.Tensor | None = None,
    origination_probs: torch.Tensor | None = None,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Per-family negative log-likelihood from the root Pi-rows and extinction E.

    Default (``origination_log_probs is None``): the uniform origination prior — every candidate
    origination branch carries equal weight ``1/S`` (the ``- log2(S)`` term) and survival is the
    unweighted mean ``1 - mean_s 2^{E_s}``. When ``origination_log_probs`` (= base-2 log of the
    softmax over the S species nodes) and ``origination_probs`` (= ``2**origination_log_probs``) are
    supplied, each branch ``s`` instead carries weight ``origination_probs[s]`` in BOTH the
    origination prior (numerator) and the survival normalizer. The weighted form reduces exactly to
    the uniform form when the weights are equal.

    The likelihood head is evaluated in ``accumulator_dtype``. Runtime callers
    pass the configured dtype explicitly; standalone callers default to the
    root-row dtype.
    """
    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=root_rows.dtype,
    )
    root_rows = root_rows.to(dtype=accumulator_dtype)
    E = E.to(device=root_rows.device, dtype=accumulator_dtype)
    if origination_log_probs is not None:
        origination_log_probs = origination_log_probs.to(
            device=root_rows.device, dtype=accumulator_dtype
        )
    if origination_probs is not None:
        origination_probs = origination_probs.to(
            device=root_rows.device, dtype=accumulator_dtype
        )
    if origination_log_probs is None:
        return -(
            logsumexp2(root_rows, dim=-1)
            - math.log2(root_rows.shape[-1])
            - log2_survival(E)
        )
    return -(
        logsumexp2(root_rows + origination_log_probs, dim=-1)
        - log2_survival(E, origination_probs)
    )


def nll_from_root_rows(
    root_rows: torch.Tensor,
    E: torch.Tensor,
    *,
    origination_log_probs: torch.Tensor | None = None,
    origination_probs: torch.Tensor | None = None,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return nll_vector_from_root_rows(
        root_rows,
        E,
        origination_log_probs=origination_log_probs,
        origination_probs=origination_probs,
        accumulator_dtype=accumulator_dtype,
    ).sum()


def origination_grad_from_root_rows(
    root_rows: torch.Tensor,
    E: torch.Tensor,
    origination_weights: torch.Tensor,
    *,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``d(sum_families NLL)/d(origination_weights)`` at fixed ``root_rows``, ``E``.

    Origination weights enter ONLY this aggregation (never the fixed-point solve or the kernels).
    The gradient is the closed-form difference between the survival and root responsibilities,
    divided by ``ln(2)``. ``root_rows`` and ``E`` are held constant.
    Returns a length-S tensor in ``origination_weights``' dtype for the global/specieswise case; for
    genewise per-family origination weights ``[G,S]``, this returns ``[G,S]`` instead.
    """
    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=root_rows.dtype,
    )
    # Reproduce the configured likelihood head without constructing a temporary
    # autograd graph. The probabilities are materialized through log-softmax2,
    # exactly as they are in the forward head.
    root_rows = root_rows.detach().to(dtype=accumulator_dtype)
    E = E.detach().to(device=root_rows.device, dtype=accumulator_dtype)
    ow = origination_weights.detach().to(
        device=root_rows.device, dtype=accumulator_dtype
    )
    log_probs = origination_log_probs_from_weights(
        ow,
        accumulator_dtype=accumulator_dtype,
    )
    probabilities = torch.exp2(log_probs)

    # Root responsibility: p_s 2**Pi_s / sum_j p_j 2**Pi_j. Subtracting
    # the row maximum before exp2 prevents underflow for very negative rows.
    root_log_mass = root_rows + log_probs
    root_row_max = root_log_mass.max(dim=-1, keepdim=True).values
    root_row_max = torch.where(
        root_row_max == float("-inf"),
        torch.zeros_like(root_row_max),
        root_row_max,
    )
    root_mass = torch.exp2(root_log_mass - root_row_max)
    root_total = root_mass.sum(dim=-1, keepdim=True)
    root_responsibility = root_mass / torch.where(
        root_total > 0,
        root_total,
        torch.ones_like(root_total),
    )

    # Survival responsibility: p_s (1 - 2**E_s) / sum_j p_j
    # (1 - 2**E_j). expm1 keeps 1 - 2**E accurate as E approaches zero.
    survival_mass = probabilities * survival_by_species_from_E(E)
    survival_total = survival_mass.sum(dim=-1, keepdim=True)
    survival_responsibility = survival_mass / torch.where(
        survival_total > 0,
        survival_total,
        torch.ones_like(survival_total),
    )

    per_family_grad = (survival_responsibility - root_responsibility) / math.log(2.0)
    grad = per_family_grad.sum_to_size(ow.shape)
    return grad.to(dtype=origination_weights.dtype)
