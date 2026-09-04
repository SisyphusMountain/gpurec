import torch
import triton
import triton.language as tl

from gpurec.api.solver_options import SolverOptions
from gpurec.core.parameters.extract_parameters import as_family_species
# The additive species-tree sums; see that module for why the old "row total minus the
# excluded few" construction had to go.
from gpurec.core.kernels.species_tree_sums import (
    off_subtree_sum,
    species_neighbourhood,
    valid_receiver_sum,
)


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
    species_height_ptr,
    leaf_fm_log_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
    COMPUTE_DIFF: tl.constexpr,
    USE_RECEIVER_WEIGHTS: tl.constexpr,
    USE_FRACTION_MISSING: tl.constexpr,
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
    receiver_mass = tl.where(
        mask, tl.exp2(receiver_weighted_extinction_log_probability - row_max_safe), zero
    )

    (
        species_height, c1_valid, c1, c2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    ) = species_neighbourhood(
        species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
        offs, mask, S,
    )
    E_s1 = tl.load(E_ptr + base + c1, mask=c1_valid, other=neg_inf)
    E_s2 = tl.load(E_ptr + base + c2, mask=c2_valid, other=neg_inf)
    if USE_FRACTION_MISSING:
        # At a leaf species (child sentinels) with fraction_missing>0, the terminal
        # speciation term is the single factor p^S * fm_l: E_s1=log2(fm_l), E_s2=0.
        fm_log = tl.load(leaf_fm_log_ptr + offs, mask=mask, other=neg_inf)
        is_missing_leaf = mask & (fm_log > neg_inf)
        E_s1 = tl.where(is_missing_leaf, fm_log, E_s1)
        E_s2 = tl.where(is_missing_leaf, tl.zeros([BLOCK_S], dtype=DTYPE), E_s2)

    max_transfer = tl.load(max_transfer_ptr + base + offs, mask=mask, other=0.0)
    # An extinct lineage in species s can transfer into every species that is neither s nor an
    # ancestor of s. Summed by ADDITION over the tree, never as the whole row's mass minus the
    # ancestor chain: see gpurec/core/kernels/species_tree_sums.py for why that difference is
    # rounding noise whenever one lane holds the row.
    valid_receiver_mass = valid_receiver_sum(
        receiver_mass, mask, zero, species_height,
        c1_valid, c1, c2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )
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
    species_height_ptr,
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
    off_subtree_donor_adjoint_ptr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    N_LEVELS: tl.constexpr,
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
    tl.store(receiver_mass_ptr + base + offs, receiver_mass, mask=mask)

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

    # Cross-warp lost-update fix: the plain store above (grad_E) and the atomic_adds below target
    # overlapping row addresses (a state's children are other states handled by other warps of the
    # same CTA). Without a barrier a warp's atomic_add can land before another warp's initializing
    # store, which then overwrites it -> dropped gradient contribution. Order stores-then-atomics.
    tl.debug_barrier()

    (
        species_height, c1_valid, c1, c2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe,
    ) = species_neighbourhood(
        species_child1_ptr, species_child2_ptr, species_parent_ptr, species_height_ptr,
        offs, mask, S,
    )
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

    # The same valid receiver mass the forward built, by the same additive walk.
    valid_receiver_mass = valid_receiver_sum(
        receiver_mass, mask, zero, species_height,
        c1_valid, c1, c2_valid, c2,
        has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
    )
    donor_adjoint = tl.where(
        mask & (valid_receiver_mass > 0.0),
        transfer_complement_vjp / valid_receiver_mass,
        zero,
    )

    # Extinction probability E[r] enters the transfer complement of every donor that may reach r,
    # that is every donor OUTSIDE r's own subtree, so the finalize kernel needs the donor adjoint
    # summed off each lane's subtree. Built by ADDITION in registers: subtree sums bottom-up, then
    # off-subtree(child) = off-subtree(parent) + parent's own term + sibling's subtree, top-down.
    # It used to be the row total minus an ancestor walk of scattered atomic adds.
    tl.store(
        off_subtree_donor_adjoint_ptr + base + offs,
        off_subtree_sum(
            donor_adjoint, mask, zero, species_height,
            c1_valid, c1, c2_valid, c2,
            has_parent, parent_safe, parent_height, has_sibling, sibling_safe, N_LEVELS,
        ),
        mask=mask,
    )


@triton.jit
def _finalize_extinction_transfer_complement_vjp_kernel(
    grad_E_ptr,
    grad_receiver_log_probs_ptr,
    receiver_mass_ptr,
    off_subtree_donor_adjoint_ptr,
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
    off_subtree_donor_adjoint = tl.load(
        off_subtree_donor_adjoint_ptr + base + offs, mask=mask, other=0.0
    )
    current_extinction_vjp = tl.load(
        grad_E_ptr + base + offs, mask=mask, other=0.0
    )
    # The staging kernel already summed the donor adjoint off each lane's subtree by addition.
    transfer_complement_vjp = receiver_mass * off_subtree_donor_adjoint
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
    species_height: torch.Tensor,
    species_levels: int,
    *,
    max_diff_out: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    use_receiver_weights: bool = True,
    leaf_fm_log: torch.Tensor | None = None,
):
    G = int(E.shape[0])
    S = int(E.shape[1])
    block_s = int(triton.next_power_of_2(S))
    E_new, E_s1, E_s2, Ebar = (torch.empty_like(E) for _ in range(4)) if out is None else out
    use_fraction_missing = leaf_fm_log is not None
    # When there is no fraction-missing tensor the constexpr short-circuits the
    # kernel load, so a valid-but-unused 1-element placeholder is enough.
    leaf_fm_log_arg = (
        leaf_fm_log.contiguous()
        if use_fraction_missing
        else torch.empty(1, device=E.device, dtype=E.dtype)
    )
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
        species_height,
        leaf_fm_log_arg,
        S,
        BLOCK_S=block_s,
        N_LEVELS=int(species_levels),
        COMPUTE_DIFF=max_diff_out is not None,
        USE_RECEIVER_WEIGHTS=bool(use_receiver_weights),
        USE_FRACTION_MISSING=use_fraction_missing,
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
        species_height,
        species_levels: int,
        use_receiver_weights: bool,
        leaf_fm_log,
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
            species_height,
            int(species_levels),
            use_receiver_weights=bool(use_receiver_weights),
            leaf_fm_log=leaf_fm_log,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], *output, *inputs[1:4], inputs[5], *inputs[6:10])
        ctx.species_levels = int(inputs[10])
        ctx.use_receiver_weights = bool(inputs[11])

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
            species_height,
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
            off_subtree_donor_adjoint,
        ) = (
            torch.empty_like(E) for _ in range(7)
        )
        grad_receiver_log_probs = torch.zeros_like(receiver_log_probs)

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
            species_height,
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
            off_subtree_donor_adjoint,
            S,
            BLOCK_S=block_s,
            N_LEVELS=int(ctx.species_levels),
            USE_RECEIVER_WEIGHTS=bool(ctx.use_receiver_weights),
            DTYPE=_tl_float_dtype(E.dtype),
            num_warps=8,
        )
        _finalize_extinction_transfer_complement_vjp_kernel[(G,)](
            grad_E,
            grad_receiver_log_probs,
            receiver_mass,
            off_subtree_donor_adjoint,
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
            None,  # species_parent
            None,  # species_child1
            None,  # species_child2
            None,  # species_height
            None,  # species_levels
            None,  # use_receiver_weights
            None,  # leaf_fm_log (non-differentiable, fixed input)
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
    species_height: torch.Tensor,
    species_levels: int,
    use_receiver_weights: bool = True,
    leaf_fm_log: torch.Tensor | None = None,
):
    """``species_height`` is 0 at a leaf and 1 + the taller child above; ``species_levels`` is the
    species tree's height. Both drive the additive valid-receiver-mass walk (see
    ``gpurec/core/kernels/species_tree_sums.py``) that replaced the old ancestor-chain walk."""
    E_arg = E.contiguous()
    return _TritonEStep2D.apply(
        E_arg,
        *(as_family_species(param, int(E_arg.shape[1]), int(E_arg.shape[0])) for param in (log_pS, log_pD, log_pL, max_transfer)),
        receiver_log_probs.contiguous(),
        species_parent,
        species_child1,
        species_child2,
        species_height,
        int(species_levels),
        bool(use_receiver_weights),
        leaf_fm_log,
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
    species_height: torch.Tensor,
    species_levels: int,
    *,
    max_iter: int | None = None,
    tol: float | None = None,
    use_receiver_weights: bool = True,
    leaf_fm_log: torch.Tensor | None = None,
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
        species_height,
        int(species_levels),
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
            leaf_fm_log=leaf_fm_log,
        )
        E_a, E_b = E_b, E_a
        max_diff = float(max_diff_out.max().item())
        if max_diff < tol:
            break

    _, E_s1, E_s2, Ebar = _launch_e_step_forward_2d(
        E_a,
        *forward_args,
        use_receiver_weights=bool(use_receiver_weights),
        leaf_fm_log=leaf_fm_log,
    )

    return E_a, E_s1, E_s2, Ebar
