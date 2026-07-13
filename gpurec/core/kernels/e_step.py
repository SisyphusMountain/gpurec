import torch
import triton
import triton.language as tl

from gpurec.api.solver_options import SolverOptions
from gpurec.core.parameters.extract_parameters import as_family_species


def _tl_float_dtype(dtype):
    """Map a supported PyTorch model dtype to its Triton scalar dtype."""
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float64:
        return tl.float64
    raise TypeError(f"kernel state must use torch.float32 or torch.float64, got {dtype}")


@triton.jit
def _update_extinction_log_probabilities_kernel(
    E_ptr,
    E_new_ptr,
    E_s1_out_ptr,
    E_s2_out_ptr,
    Ebar_out_ptr,
    max_diff_out_ptr,
    log_pS_ptr,
    log_pD_ptr,
    log_pL_ptr,
    max_transfer_ptr,
    receiver_log_probs_ptr,
    species_parent_ptr,
    species_child1_ptr,
    species_child2_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    neg_inf = -float("inf")
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    E = tl.load(E_ptr + base + offs, mask=mask, other=neg_inf)
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(receiver_log_probs_ptr + offs, mask=mask, other=neg_inf)
        receiver_weighted_extinction_log_probability = receiver_log_probability + E
    else:
        receiver_weighted_extinction_log_probability = E
    row_max = tl.max(receiver_weighted_extinction_log_probability, axis=0)
    row_max_safe = tl.where(row_max != neg_inf, row_max, tl.zeros([1], dtype=DTYPE))
    total_receiver_mass = tl.sum(tl.exp2(receiver_weighted_extinction_log_probability - row_max_safe), axis=0)

    ancestor_species = offs
    excluded_ancestor_mass = zero
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        ancestor_valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
        ancestor_extinction_log_probability = tl.load(E_ptr + base + ancestor_species, mask=ancestor_valid, other=neg_inf)
        if USE_RECEIVER_WEIGHTS:
            ancestor_receiver_log_probability = tl.load(receiver_log_probs_ptr + ancestor_species, mask=ancestor_valid, other=neg_inf)
            excluded_ancestor_mass += tl.where(ancestor_valid, tl.exp2(ancestor_receiver_log_probability + ancestor_extinction_log_probability - row_max_safe), zero)
        else:
            excluded_ancestor_mass += tl.where(ancestor_valid, tl.exp2(ancestor_extinction_log_probability - row_max_safe), zero)
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=ancestor_valid, other=-1).to(tl.int32)

    c1 = tl.load(species_child1_ptr + offs, mask=mask, other=-1)
    c2 = tl.load(species_child2_ptr + offs, mask=mask, other=-1)
    c1_valid = mask & (c1 >= 0) & (c1 < S)
    c2_valid = mask & (c2 >= 0) & (c2 < S)
    E_s1 = tl.load(E_ptr + base + c1, mask=c1_valid, other=neg_inf)
    E_s2 = tl.load(E_ptr + base + c2, mask=c2_valid, other=neg_inf)

    max_transfer = tl.load(max_transfer_ptr + base + offs, mask=mask, other=0.0)
    valid_receiver_mass = total_receiver_mass - excluded_ancestor_mass
    Ebar = tl.where(valid_receiver_mass > 0.0, tl.log2(valid_receiver_mass) + row_max + max_transfer, neg_inf)

    speciation_log_probability = tl.load(log_pS_ptr + base + offs, mask=mask, other=neg_inf)
    duplication_log_probability = tl.load(log_pD_ptr + base + offs, mask=mask, other=neg_inf)
    loss_log_probability = tl.load(log_pL_ptr + base + offs, mask=mask, other=neg_inf)

    speciation_log_term = speciation_log_probability + E_s1 + E_s2
    duplication_log_term = duplication_log_probability + 2.0 * E
    transfer_log_term = E + Ebar
    loss_log_term = loss_log_probability
    logsumexp_max = tl.maximum(
        tl.maximum(speciation_log_term, duplication_log_term),
        tl.maximum(transfer_log_term, loss_log_term),
    )
    logsumexp_max_safe = tl.where(logsumexp_max == neg_inf, zero, logsumexp_max)
    extinction_event_scaled_mass = (
        tl.exp2(speciation_log_term - logsumexp_max_safe)
        + tl.exp2(duplication_log_term - logsumexp_max_safe)
        + tl.exp2(transfer_log_term - logsumexp_max_safe)
        + tl.exp2(loss_log_term - logsumexp_max_safe)
    )
    E_new = tl.log2(extinction_event_scaled_mass) + logsumexp_max

    tl.store(E_new_ptr + base + offs, E_new, mask=mask)
    tl.store(E_s1_out_ptr + base + offs, E_s1, mask=mask)
    tl.store(E_s2_out_ptr + base + offs, E_s2, mask=mask)
    tl.store(Ebar_out_ptr + base + offs, Ebar, mask=mask)
    if COMPUTE_DIFF:
        diff = tl.where(mask, tl.abs(E_new - E), zero)
        tl.store(max_diff_out_ptr + g, tl.max(diff, axis=0))


@triton.jit
def _stage_extinction_and_transfer_complement_vjp_kernel(
    E_ptr,
    E_new_ptr,
    E_s1_ptr,
    E_s2_ptr,
    Ebar_ptr,
    log_pS_ptr,
    log_pD_ptr,
    log_pL_ptr,
    receiver_log_probs_ptr,
    species_parent_ptr,
    species_child1_ptr,
    species_child2_ptr,
    grad_E_new_ptr,
    grad_E_s1_out_ptr,
    grad_E_s2_out_ptr,
    grad_Ebar_out_ptr,
    grad_E_ptr,
    grad_log_pS_ptr,
    grad_log_pD_ptr,
    grad_log_pL_ptr,
    grad_max_transfer_ptr,
    grad_receiver_log_probs_ptr,
    receiver_mass_ptr,
    excluded_donor_adjoint_ptr,
    total_donor_adjoint_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_ANCESTOR_DEPTH: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    DTYPE: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    neg_inf = -float("inf")
    zero = tl.zeros([BLOCK_S], dtype=DTYPE)

    E = tl.load(E_ptr + base + offs, mask=mask, other=neg_inf)
    if USE_RECEIVER_WEIGHTS:
        receiver_log_probability = tl.load(receiver_log_probs_ptr + offs, mask=mask, other=neg_inf)
        receiver_weighted_extinction_log_probability = receiver_log_probability + E
    else:
        receiver_weighted_extinction_log_probability = E
    E_new = tl.load(E_new_ptr + base + offs, mask=mask, other=neg_inf)
    E_s1 = tl.load(E_s1_ptr + base + offs, mask=mask, other=neg_inf)
    E_s2 = tl.load(E_s2_ptr + base + offs, mask=mask, other=neg_inf)
    Ebar = tl.load(Ebar_ptr + base + offs, mask=mask, other=neg_inf)
    speciation_log_probability = tl.load(log_pS_ptr + base + offs, mask=mask, other=neg_inf)
    duplication_log_probability = tl.load(log_pD_ptr + base + offs, mask=mask, other=neg_inf)
    loss_log_probability = tl.load(log_pL_ptr + base + offs, mask=mask, other=neg_inf)

    row_max = tl.max(receiver_weighted_extinction_log_probability, axis=0)
    row_max_safe = tl.where(row_max != neg_inf, row_max, tl.zeros([1], dtype=DTYPE))
    receiver_mass = tl.exp2(
        receiver_weighted_extinction_log_probability - row_max_safe
    )
    receiver_mass = tl.where(mask, receiver_mass, zero)
    total_receiver_mass = tl.sum(receiver_mass, axis=0)
    tl.store(receiver_mass_ptr + base + offs, receiver_mass, mask=mask)
    tl.store(excluded_donor_adjoint_ptr + base + offs, zero, mask=mask)

    extinction_update_adjoint = tl.load(grad_E_new_ptr + base + offs, mask=mask, other=0.0)
    child1_extinction_output_adjoint = tl.load(grad_E_s1_out_ptr + base + offs, mask=mask, other=0.0)
    child2_extinction_output_adjoint = tl.load(grad_E_s2_out_ptr + base + offs, mask=mask, other=0.0)
    transfer_complement_output_adjoint = tl.load(grad_Ebar_out_ptr + base + offs, mask=mask, other=0.0)

    speciation_log_term = speciation_log_probability + E_s1 + E_s2
    duplication_log_term = duplication_log_probability + 2.0 * E
    transfer_log_term = E + Ebar
    loss_log_term = loss_log_probability
    speciation_event_vjp = tl.where(
        mask, extinction_update_adjoint * tl.exp2(speciation_log_term - E_new), zero
    )
    duplication_event_vjp = tl.where(
        mask, extinction_update_adjoint * tl.exp2(duplication_log_term - E_new), zero
    )
    transfer_event_vjp = tl.where(
        mask, extinction_update_adjoint * tl.exp2(transfer_log_term - E_new), zero
    )
    loss_event_vjp = tl.where(mask, extinction_update_adjoint * tl.exp2(loss_log_term - E_new), zero)

    transfer_complement_vjp = transfer_event_vjp + transfer_complement_output_adjoint
    tl.store(grad_log_pS_ptr + base + offs, speciation_event_vjp, mask=mask)
    tl.store(grad_log_pD_ptr + base + offs, duplication_event_vjp, mask=mask)
    tl.store(grad_log_pL_ptr + base + offs, loss_event_vjp, mask=mask)
    tl.store(grad_max_transfer_ptr + base + offs, transfer_complement_vjp, mask=mask)

    tl.store(
        grad_E_ptr + base + offs,
        2.0 * duplication_event_vjp + transfer_event_vjp,
        mask=mask,
    )

    # Cross-warp lost-update fix: the plain stores above (grad_E here, excluded_donor_adjoint earlier)
    # and the atomic_adds below target overlapping row addresses (a state's children/
    # ancestors are other states handled by other warps of the same CTA). Without a barrier
    # a warp's atomic_add can land before another warp's initializing store, which then
    # overwrites it -> dropped gradient contribution. Order stores-then-atomics.
    tl.debug_barrier()

    c1 = tl.load(species_child1_ptr + offs, mask=mask, other=-1)
    c2 = tl.load(species_child2_ptr + offs, mask=mask, other=-1)
    c1_valid = mask & (c1 >= 0) & (c1 < S)
    c2_valid = mask & (c2 >= 0) & (c2 < S)
    tl.atomic_add(
        grad_E_ptr + base + c1,
        speciation_event_vjp + child1_extinction_output_adjoint,
        sem="relaxed",
        mask=c1_valid,
    )
    tl.atomic_add(
        grad_E_ptr + base + c2,
        speciation_event_vjp + child2_extinction_output_adjoint,
        sem="relaxed",
        mask=c2_valid,
    )

    ancestor_species = offs
    excluded_ancestor_mass = tl.zeros([BLOCK_S], dtype=DTYPE)
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
        ancestor_extinction_log_probability = tl.load(E_ptr + base + ancestor_species, mask=valid, other=neg_inf)
        if USE_RECEIVER_WEIGHTS:
            ancestor_receiver_log_probability = tl.load(receiver_log_probs_ptr + ancestor_species, mask=valid, other=neg_inf)
            excluded_ancestor_mass += tl.where(valid, tl.exp2(ancestor_receiver_log_probability + ancestor_extinction_log_probability - row_max_safe), zero)
        else:
            excluded_ancestor_mass += tl.where(valid, tl.exp2(ancestor_extinction_log_probability - row_max_safe), zero)
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=valid, other=-1).to(tl.int32)

    valid_receiver_mass = total_receiver_mass - excluded_ancestor_mass
    donor_adjoint = tl.where(
        mask & (valid_receiver_mass > 0.0),
        transfer_complement_vjp / valid_receiver_mass,
        zero,
    )
    tl.store(
        total_donor_adjoint_ptr + g, tl.sum(donor_adjoint, axis=0)
    )

    ancestor_species = offs
    for _ in range(0, MAX_ANCESTOR_DEPTH):
        valid = mask & (ancestor_species >= 0) & (ancestor_species < S)
        tl.atomic_add(
            excluded_donor_adjoint_ptr + base + ancestor_species,
            donor_adjoint,
            sem="relaxed",
            mask=valid,
        )
        ancestor_species = tl.load(species_parent_ptr + ancestor_species, mask=valid, other=-1).to(tl.int32)


@triton.jit
def _finalize_extinction_transfer_complement_vjp_kernel(
    grad_E_ptr,
    grad_receiver_log_probs_ptr,
    receiver_mass_ptr,
    excluded_donor_adjoint_ptr,
    total_donor_adjoint_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
):
    g = tl.program_id(0)
    base = g * S
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    receiver_mass = tl.load(
        receiver_mass_ptr + base + offs, mask=mask, other=0.0
    )
    excluded_donor_adjoint = tl.load(
        excluded_donor_adjoint_ptr + base + offs, mask=mask, other=0.0
    )
    total_donor_adjoint = tl.load(total_donor_adjoint_ptr + g)
    current_extinction_vjp = tl.load(
        grad_E_ptr + base + offs, mask=mask, other=0.0
    )
    transfer_complement_vjp = receiver_mass * (
        total_donor_adjoint - excluded_donor_adjoint
    )
    tl.store(
        grad_E_ptr + base + offs,
        current_extinction_vjp + transfer_complement_vjp,
        mask=mask,
    )
    if USE_RECEIVER_WEIGHTS:
        tl.atomic_add(
            grad_receiver_log_probs_ptr + offs,
            transfer_complement_vjp,
            sem="relaxed",
            mask=mask,
        )


def _launch_e_step_forward_2d(
    E: torch.Tensor,
    log_pS_mat: torch.Tensor,
    log_pD_mat: torch.Tensor,
    log_pL_mat: torch.Tensor,
    max_transfer_mat: torch.Tensor,
    receiver_log_probs: torch.Tensor,
    species_parent: torch.Tensor,
    species_child1: torch.Tensor,
    species_child2: torch.Tensor,
    max_ancestor_depth: int,
    *,
    max_diff_out: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    use_receiver_weights: bool = True,
):
    G = int(E.shape[0])
    S = int(E.shape[1])
    block_s = int(triton.next_power_of_2(S))
    E_new, E_s1, E_s2, Ebar = (torch.empty_like(E) for _ in range(4)) if out is None else out
    _update_extinction_log_probabilities_kernel[(G,)](
        E,
        E_new,
        E_s1,
        E_s2,
        Ebar,
        E_new if max_diff_out is None else max_diff_out,
        log_pS_mat,
        log_pD_mat,
        log_pL_mat,
        max_transfer_mat,
        receiver_log_probs,
        species_parent,
        species_child1,
        species_child2,
        S,
        BLOCK_S=block_s,
        MAX_ANCESTOR_DEPTH=int(max_ancestor_depth),
        COMPUTE_DIFF=max_diff_out is not None,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        DTYPE=_tl_float_dtype(E.dtype),
        num_warps=8,
    )
    return E_new, E_s1, E_s2, Ebar


class _TritonEStep2D(torch.autograd.Function):
    @staticmethod
    def forward(
        E,
        log_pS_mat,
        log_pD_mat,
        log_pL_mat,
        max_transfer_mat,
        receiver_log_probs,
        species_parent,
        species_child1,
        species_child2,
        max_ancestor_depth: int,
        use_receiver_weights: bool,
    ):
        return _launch_e_step_forward_2d(
            E,
            log_pS_mat,
            log_pD_mat,
            log_pL_mat,
            max_transfer_mat,
            receiver_log_probs,
            species_parent,
            species_child1,
            species_child2,
            int(max_ancestor_depth),
            use_receiver_weights=bool(use_receiver_weights),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], *output, *inputs[1:4], inputs[5], *inputs[6:9])
        ctx.max_ancestor_depth = int(inputs[9])
        ctx.use_receiver_weights = bool(inputs[10])

    @staticmethod
    def backward(ctx, grad_E_new, grad_E_s1_out, grad_E_s2_out, grad_Ebar_out):
        (
            E,
            E_new,
            E_s1,
            E_s2,
            Ebar,
            log_pS_mat,
            log_pD_mat,
            log_pL_mat,
            receiver_log_probs,
            species_parent,
            species_child1,
            species_child2,
        ) = ctx.saved_tensors
        G = int(E.shape[0])
        S = int(E.shape[1])
        block_s = int(triton.next_power_of_2(S))
        grad_E_new, grad_E_s1_out, grad_E_s2_out, grad_Ebar_out = (
            torch.zeros_like(E) if grad is None else grad.contiguous()
            for grad in (grad_E_new, grad_E_s1_out, grad_E_s2_out, grad_Ebar_out)
        )
        (
            grad_E,
            grad_log_pS,
            grad_log_pD,
            grad_log_pL,
            grad_max_transfer,
            receiver_mass,
            excluded_donor_adjoint,
        ) = (
            torch.empty_like(E) for _ in range(7)
        )
        grad_receiver_log_probs = torch.zeros_like(receiver_log_probs)
        total_donor_adjoint = torch.empty((G,), dtype=E.dtype, device=E.device)

        _stage_extinction_and_transfer_complement_vjp_kernel[(G,)](
            E,
            E_new,
            E_s1,
            E_s2,
            Ebar,
            log_pS_mat,
            log_pD_mat,
            log_pL_mat,
            receiver_log_probs,
            species_parent,
            species_child1,
            species_child2,
            grad_E_new,
            grad_E_s1_out,
            grad_E_s2_out,
            grad_Ebar_out,
            grad_E,
            grad_log_pS,
            grad_log_pD,
            grad_log_pL,
            grad_max_transfer,
            grad_receiver_log_probs,
            receiver_mass,
            excluded_donor_adjoint,
            total_donor_adjoint,
            S,
            BLOCK_S=block_s,
            MAX_ANCESTOR_DEPTH=int(ctx.max_ancestor_depth),
            USE_RECEIVER_WEIGHTS=bool(ctx.use_receiver_weights),
            DTYPE=_tl_float_dtype(E.dtype),
            num_warps=8,
        )
        _finalize_extinction_transfer_complement_vjp_kernel[(G,)](
            grad_E,
            grad_receiver_log_probs,
            receiver_mass,
            excluded_donor_adjoint,
            total_donor_adjoint,
            S,
            BLOCK_S=block_s,
            USE_RECEIVER_WEIGHTS=bool(ctx.use_receiver_weights),
            num_warps=8,
        )
        return (
            grad_E,
            grad_log_pS,
            grad_log_pD,
            grad_log_pL,
            grad_max_transfer,
            grad_receiver_log_probs,
            None,
            None,
            None,
            None,
            None,
        )


def e_step_triton_autograd(
    E: torch.Tensor,
    log_pS: torch.Tensor,
    log_pD: torch.Tensor,
    log_pL: torch.Tensor,
    max_transfer: torch.Tensor,
    receiver_log_probs: torch.Tensor,
    species_parent: torch.Tensor,
    species_child1: torch.Tensor,
    species_child2: torch.Tensor,
    max_ancestor_depth: int,
    use_receiver_weights: bool = True,
):
    E_arg = E.contiguous()
    return _TritonEStep2D.apply(
        E_arg,
        *(as_family_species(param, int(E_arg.shape[1]), int(E_arg.shape[0])) for param in (log_pS, log_pD, log_pL, max_transfer)),
        receiver_log_probs.contiguous(),
        species_parent,
        species_child1,
        species_child2,
        int(max_ancestor_depth),
        bool(use_receiver_weights),
    )


def e_fixed_point_triton(
    E0: torch.Tensor,
    log_pS: torch.Tensor,
    log_pD: torch.Tensor,
    log_pL: torch.Tensor,
    max_transfer: torch.Tensor,
    receiver_log_probs: torch.Tensor,
    species_parent: torch.Tensor,
    species_child1: torch.Tensor,
    species_child2: torch.Tensor,
    max_ancestor_depth: int,
    *,
    max_iter: int | None = None,
    tol: float | None = None,
    use_receiver_weights: bool = True,
):
    if max_iter is None:
        max_iter = SolverOptions().e_max_iter
    if tol is None:
        tol = SolverOptions().e_tol
    max_iter = int(max_iter)
    tol = float(tol)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    E_a = E0.contiguous().clone()
    G = int(E_a.shape[0])
    log_pS_mat, log_pD_mat, log_pL_mat, max_transfer_mat = (
        as_family_species(param, int(E_a.shape[1]), int(E_a.shape[0]))
        for param in (log_pS, log_pD, log_pL, max_transfer)
    )
    forward_args = (
        log_pS_mat,
        log_pD_mat,
        log_pL_mat,
        max_transfer_mat,
        receiver_log_probs.contiguous(),
        species_parent,
        species_child1,
        species_child2,
        int(max_ancestor_depth),
    )

    E_b, E_s1, E_s2, Ebar = (torch.empty_like(E_a) for _ in range(4))
    max_diff_out = torch.empty((G,), dtype=E_a.dtype, device=E_a.device)

    for _ in range(max_iter):
        _launch_e_step_forward_2d(
            E_a,
            *forward_args,
            max_diff_out=max_diff_out,
            out=(E_b, E_s1, E_s2, Ebar),
            use_receiver_weights=bool(use_receiver_weights),
        )
        E_a, E_b = E_b, E_a
        max_diff = float(max_diff_out.max().item())
        if max_diff < tol:
            break

    _, E_s1, E_s2, Ebar = _launch_e_step_forward_2d(
        E_a,
        *forward_args,
        use_receiver_weights=bool(use_receiver_weights),
    )

    return E_a, E_s1, E_s2, Ebar
