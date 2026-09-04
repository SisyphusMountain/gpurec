import math
import warnings

import torch

from gpurec.api.solver_options import SolverOptions
from gpurec.config import dtype_rel_tol_default as _e_adjoint_rel_tol_default, dtype_rel_tol_floor as _e_adjoint_rel_tol_floor
from gpurec.core.inference.logspace import logsumexp2 as _logsumexp2, log2_survival as _log2_survival
from gpurec.core.kernels.pi_forward import _select_log_split_probs, compute_dts_forward
from gpurec.core.kernels.wave_backward import (
    compute_active_adjoint_row_mask,
    accumulate_gene_split_event_vjp,
    accumulate_transfer_complement_vjp_from_donor_adjoint,
    solve_reconciliation_wave_vjp,
)
from gpurec.core.memory_policy import wave_scratch_budget_bytes
from gpurec.core.parameters.extract_parameters import (
    as_family_param,
    as_family_species,
    extract_parameters_weighted_receivers,
    resolve_accumulator_dtype,
)
from gpurec.core.kernels.e_step import e_step_triton_autograd

_NEG_INF = float("-inf")


def _safe_exp2_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    neg_inf = a == _NEG_INF
    a_safe = torch.where(neg_inf, torch.zeros_like(a), a)
    b_safe = torch.where(neg_inf, torch.zeros_like(b), b)
    return torch.where(neg_inf, torch.zeros_like(a), torch.exp2(a_safe - b_safe))


def _likelihood_root_seed(
    root_pi: torch.Tensor,
    origination_log_probs: torch.Tensor | None,
    *,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the configured-head root cotangent rounded to state dtype."""

    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=root_pi.dtype,
    )
    head = root_pi.to(dtype=accumulator_dtype)
    if origination_log_probs is not None:
        head = head + origination_log_probs.to(device=head.device, dtype=head.dtype)
    root_lse = _logsumexp2(head, dim=-1, keepdim=True)
    return -_safe_exp2_ratio(head, root_lse).to(dtype=root_pi.dtype)


def _likelihood_log2_survival(
    extinction: torch.Tensor,
    origination_probs: torch.Tensor | None,
    *,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate the likelihood head's survival normalizer."""

    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=extinction.dtype,
    )
    probabilities = (
        None
        if origination_probs is None
        else origination_probs.to(
            device=extinction.device,
            dtype=accumulator_dtype,
        )
    )
    return _log2_survival(extinction.to(dtype=accumulator_dtype), probabilities)


# `_e_adjoint_rel_tol_default` / `_e_adjoint_rel_tol_floor` moved to
# `gpurec.config.gpurec_config` (as `dtype_rel_tol_default` / `dtype_rel_tol_floor`)
# and re-exported above under these names: tests and other callers still import
# them from here.


# How many consecutive Neumann terms may fail to improve on the smallest term seen so far before
# the series is declared non-contracting and the GMRES fallback below takes over. Not a setting:
# it only decides how long a solve that is already failing keeps failing before a solver that
# cannot fail takes over, so a wrong value costs time, never accuracy. 16 is chosen so that a
# series whose terms merely OSCILLATE on the way down (complex-eigenvalue tail) is not mistaken
# for one that has stopped converging: at the slowest rate that still finishes inside a 128-term
# budget (about 0.81 per term) 16 terms shrink the envelope by 0.81^16 = 0.035, so an oscillation
# would have to swing by a factor of 28 to hide the decay.
_E_ADJOINT_NEUMANN_STALL_PATIENCE = 16

# Krylov vectors per GMRES restart cycle in the fallback. Not a setting: it is a
# memory-versus-orthogonalization-cost tradeoff inside a solve that only runs when the Neumann
# series has already failed. Each basis vector is one full E vector, so 48 of them cost
# 48 * families * species * 8 bytes -- about 150 MB for a 100-family batch on a 2013-leaf species
# tree (4025 species-tree nodes), which such a batch can afford; a longer cycle costs more memory
# and makes every Gram-Schmidt pass longer, a shorter one restarts more often and converges slower.
_E_ADJOINT_GMRES_RESTART = 48

# Below this many unknowns the fallback assembles ``I - J`` column by column and factorizes it
# instead of running GMRES. Not a setting: it is the size at which building the matrix (one matvec
# per column) stops being cheaper and smaller than iterating. 512 columns of a 512-long vector is
# a 2 MB float64 matrix and 512 matvecs, both trivial, and the answer is then exact rather than
# iterative -- which matters because restarted GMRES can stagnate on a strongly non-contracting
# operator, exactly the regime the fallback exists for.
_E_ADJOINT_DENSE_SOLVE_LIMIT = 512


@torch.no_grad()
def _dense_e_adjoint(Av, b):
    """Solve ``(I - J) x = b`` by building ``I - J`` from the matvec and factorizing it.

    For a system small enough that ``b.numel()`` matvecs are affordable this is both exact and
    immune to the stagnation a restarted Krylov method can hit on a strongly non-contracting
    operator. Assembly and the LU solve run in float64 whatever ``b``'s dtype is; the matvec
    itself is still called in ``b``'s dtype.

    Returns ``(x, relative_residual)`` with ``x`` in ``b``'s dtype and shape.
    """
    shape = b.shape
    dtype = b.dtype
    work = torch.float64
    flat = b.reshape(-1).to(work)
    n = flat.numel()
    bnorm = float(torch.linalg.vector_norm(flat))
    if bnorm == 0.0:
        return torch.zeros_like(b), 0.0

    operator = torch.empty(n, n, dtype=work, device=flat.device)
    basis = torch.zeros(n, dtype=dtype, device=b.device)
    for column in range(n):
        basis.zero_()
        basis[column] = 1
        operator[:, column] = Av(basis.reshape(shape)).reshape(-1).to(work)

    solution = torch.linalg.solve(operator, flat)
    rel = float(torch.linalg.vector_norm(operator @ solution - flat)) / bnorm
    return solution.reshape(shape).to(dtype), rel


@torch.no_grad()
def _gmres_e_adjoint(Av, b, *, x0, tol, max_matvecs, restart):
    """Solve ``(I - J) x = b`` by restarted GMRES on the matvec ``Av(w) == (I - J) w``.

    This is the fallback for :func:`_neumann_e_adjoint`: GMRES minimizes the residual over the
    Krylov subspace, so unlike the Neumann series it needs no contraction assumption and cannot
    break down on a nonsingular operator -- it only needs ``I - J`` to be invertible.

    ``x0`` is the initial guess (the best Neumann iterate). Arnoldi runs in float64 regardless of
    ``b``'s dtype -- the Gram-Schmidt inner products and the Givens least-squares are where an
    fp32 Arnoldi loses orthogonality, and the vectors are only ``families x species`` long, so
    float64 here costs little; the matvec itself is still called in ``b``'s dtype, so the residual
    floor is set by the model's precision and not by the orthogonalization.

    Returns ``(x, relative_residual)`` with ``x`` in ``b``'s dtype and shape; the caller decides
    whether that residual is good enough.
    """
    restart = int(restart)
    max_matvecs = int(max_matvecs)
    if restart < 1:
        raise ValueError("restart must be at least 1")
    if max_matvecs < 1:
        raise ValueError("max_matvecs must be at least 1")

    shape = b.shape
    dtype = b.dtype
    work = torch.float64
    flat = b.reshape(-1).to(work)
    bnorm = float(torch.linalg.vector_norm(flat))
    if bnorm == 0.0:
        return torch.zeros_like(b), 0.0

    def apply(v):
        return Av(v.reshape(shape).to(dtype)).reshape(-1).to(work)

    x = x0.reshape(-1).to(work).clone()
    spent = 0
    residual = flat - apply(x)
    spent += 1
    rel = float(torch.linalg.vector_norm(residual)) / bnorm
    best_x, best_rel = x, rel

    while rel > tol and spent < max_matvecs:
        cycle = min(restart, max_matvecs - spent)
        if cycle < 1:
            break
        beta = float(torch.linalg.vector_norm(residual))
        if beta == 0.0:
            break
        basis = [residual / beta]
        hessenberg = torch.zeros(cycle + 1, cycle, dtype=work)
        cosines = torch.zeros(cycle, dtype=work)
        sines = torch.zeros(cycle, dtype=work)
        rhs = torch.zeros(cycle + 1, dtype=work)
        rhs[0] = beta
        used = 0
        for j in range(cycle):
            w = apply(basis[j])
            spent += 1
            # Two Gram-Schmidt passes: the second one removes the components the first pass
            # leaves behind when the new direction is nearly inside the existing span, which is
            # exactly the regime a restarted cycle ends in.
            for _ in range(2):
                for i in range(j + 1):
                    coeff = torch.dot(basis[i], w)
                    hessenberg[i, j] += coeff.cpu()
                    w = w - coeff * basis[i]
            subdiagonal = float(torch.linalg.vector_norm(w))
            hessenberg[j + 1, j] = subdiagonal
            for i in range(j):
                rotated = cosines[i] * hessenberg[i, j] + sines[i] * hessenberg[i + 1, j]
                hessenberg[i + 1, j] = (
                    -sines[i] * hessenberg[i, j] + cosines[i] * hessenberg[i + 1, j]
                )
                hessenberg[i, j] = rotated
            magnitude = math.hypot(float(hessenberg[j, j]), float(hessenberg[j + 1, j]))
            if magnitude == 0.0:
                # The Krylov space is exhausted and carries no new direction: stop with what the
                # first ``j`` vectors already give.
                break
            cosines[j] = float(hessenberg[j, j]) / magnitude
            sines[j] = float(hessenberg[j + 1, j]) / magnitude
            hessenberg[j, j] = magnitude
            hessenberg[j + 1, j] = 0.0
            rhs[j + 1] = -sines[j] * rhs[j]
            rhs[j] = cosines[j] * rhs[j]
            used = j + 1
            if abs(float(rhs[j + 1])) / bnorm <= tol or subdiagonal == 0.0:
                break
            basis.append(w / subdiagonal)
        if used == 0:
            break
        y = torch.linalg.solve_triangular(
            hessenberg[:used, :used], rhs[:used].unsqueeze(1), upper=True
        ).squeeze(1)
        for i in range(used):
            x = x + float(y[i]) * basis[i]
        residual = flat - apply(x)
        spent += 1
        rel = float(torch.linalg.vector_norm(residual)) / bnorm
        if rel < best_rel:
            best_rel = rel
            best_x = x

    return best_x.reshape(shape).to(dtype), best_rel


@torch.no_grad()
def _neumann_e_adjoint(Av, b: torch.Tensor, *, max_iter: int = 128, tol=None):
    """Solve ``(I - J) x = b`` by Neumann series ``x = sum_k J^k b``, where ``J w = w - Av(w)``.

    The E-adjoint / GGN linear solve. No orthogonalization -> no fp32 Arnoldi-style residual
    floor. Valid because the E-step self-map Jacobian ``J`` is a contraction (the forward E
    fixed point converges), so ``(I-J)^{-1} = sum_k J^k`` converges.

    By telescoping, the true relative residual after summing terms ``k=0..N`` is
    ``||J^{N+1} b|| / ||b||``; since ``||J|| < 1`` this is bounded above by
    ``||J^N b|| / ||b||`` -- the quantity actually tracked below (``rel``) -- so ``rel`` is a
    conservative (upper-bound) proxy for the true residual, off by one power of ``J``.

    The contraction holds only when ``J`` really is the Jacobian of the map the forward solved:
    a caller that hands over a mismatched operator gets terms that grow instead of shrink. That is
    not hypothetical -- ``gpurec/api/_execution.py`` used to build this system from the E-step
    WITHOUT ``fraction_missing`` while the forward had solved the one WITH it, and the resulting
    operator had spectral radius 2.0 instead of 0.05 (measured on one Coleman family, 2013 species,
    fraction_missing up to 0.4998, float64), so the terms climbed by about 1.45x each and reached
    inf. Rather than sum a divergent series, this function detects the failure -- the term norm
    stops improving for ``_E_ADJOINT_NEUMANN_STALL_PATIENCE`` terms, or goes non-finite, while the
    sum is still short of the acceptance floor -- and hands the SAME system to a solver that needs
    no contraction: :func:`_dense_e_adjoint` when the system is small enough to assemble,
    :func:`_gmres_e_adjoint` otherwise. That is a safety net, not a licence: an operator that is
    not a contraction here means something upstream is inconsistent and is worth finding.

    The fast path is untouched: the fallback can only trigger where the series would otherwise
    have raised, so every solve that converges returns exactly what it did before.

    ``tol`` is a RELATIVE residual target: ``None`` -> the dtype-matched default
    (:func:`_e_adjoint_rel_tol_default`); a value below the dtype floor
    (:func:`_e_adjoint_rel_tol_floor`) is clamped up with a warning. Returns the best (smallest
    conservative-residual) iterate; raises ``RuntimeError`` only if neither the series nor the
    GMRES fallback reaches the acceptance floor within ``max_iter`` matvecs.
    """
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    dtype = b.dtype
    floor = _e_adjoint_rel_tol_floor(dtype)
    default = _e_adjoint_rel_tol_default(dtype)
    if tol is None:
        target = default
    else:
        target = float(tol)
        if target <= 0.0:
            raise ValueError("tol must be positive")
        if target < floor:
            warnings.warn(
                f"neumann tol={target:.2e} is below the {dtype} finite-precision "
                f"residual floor {floor:.2e}; clamping to the floor. A tighter "
                f"relative residual is unreachable in this precision -- use fp64.",
                RuntimeWarning, stacklevel=2,
            )
            target = floor
    # Any iterate reaching the dtype's natural floor counts as converged-to-precision.
    accept = max(target, default)

    bnorm = float(torch.linalg.vector_norm(b.reshape(-1)).detach().cpu())
    if bnorm == 0.0:
        return b.clone()

    x = b.clone()
    term = b.clone()
    best_x = x
    best_rel = float("inf")
    stalled = 0
    terms = 0
    for _ in range(max_iter):
        term = term - Av(term)  # == J @ term
        x = x + term
        terms += 1
        rel = float(torch.linalg.vector_norm(term.reshape(-1)).detach().cpu()) / bnorm
        if rel < best_rel:
            best_rel = rel
            best_x = x
            stalled = 0
        else:
            stalled += 1
        if rel <= target:
            return x
        if best_rel > accept and (
            stalled >= _E_ADJOINT_NEUMANN_STALL_PATIENCE or not math.isfinite(rel)
        ):
            # The terms have stopped shrinking while the sum is still short of the acceptance
            # floor, so no number of further terms will get there: J is not a contraction. Stop
            # here rather than burning the rest of the budget (and overflowing to inf) and hand
            # the same system to the fallback below.
            break

    if best_rel <= accept:
        return best_x

    # Fallback: the series is not contracting, so sum it no further -- solve the same system
    # ``(I - J) x = b`` with a method that needs no contraction. Small enough to assemble ->
    # build the operator and factorize it, which is exact and cannot stagnate; otherwise GMRES,
    # warm-started from the best Neumann iterate so a series that merely ran out of terms starts
    # from a good guess.
    if b.numel() <= _E_ADJOINT_DENSE_SOLVE_LIMIT:
        fallback, fallback_rel = _dense_e_adjoint(Av, b)
        used = f"the assembled {b.numel()}x{b.numel()} direct solve"
    else:
        fallback, fallback_rel = _gmres_e_adjoint(
            Av,
            b,
            x0=best_x,
            tol=target,
            max_matvecs=max_iter,
            restart=_E_ADJOINT_GMRES_RESTART,
        )
        used = f"the GMRES fallback (within the same {max_iter}-matvec budget)"
    if fallback_rel <= accept:
        return fallback
    raise RuntimeError(
        f"E-adjoint solve failed to converge: the Neumann series stopped at conservative "
        f"relative residual {best_rel:.3e} after {terms} terms (target {target:.3e}, dtype "
        f"{dtype}) because the self-map J is not a contraction, and {used} reached only "
        f"{fallback_rel:.3e}. A non-contracting J usually means the operator does not match the "
        f"map the forward solved (see the fraction-missing case in this function's docstring); "
        f"failing that, raise e_adjoint_max_iter or loosen e_adjoint_tol."
    )


@torch.no_grad()
def implicit_grad_loglik_vjp_wave(
    wave_layout, species_helpers, *, Pi_star_wave: torch.Tensor,
    Pibar_star_wave: torch.Tensor, E_star: torch.Tensor, E_s1: torch.Tensor,
    E_s2: torch.Tensor, Ebar: torch.Tensor, log_pS: torch.Tensor,
    log_pD: torch.Tensor, log_pL: torch.Tensor, max_transfer_mat: torch.Tensor,
    receiver_log_probs: torch.Tensor,
    use_receiver_weights: bool,
    # Whether the caller is going to USE the returned receiver-weight gradient. Every caller says
    # so explicitly (no default) because getting it wrong is silent: the receiver gradient still
    # comes back with the right shape, only missing the reconciliation (Pi) half of its value.
    # False skips two per-wave Triton launches -- the receiver-log-probability VJP kernel and the
    # receiver-gradient accumulation inside the transfer-subtree VJP -- which together were 6.4 %
    # of one gradient on the 200-family Coleman batch. A genewise rate fit freezes the receiver
    # weights, so it passes False; anything that trains them (map_cv, the specieswise joint
    # theta+alpha solves) passes True.
    need_receiver_grad: bool,
    # The forward's kept gene-split (DTS) rows, {wave start: (rows, row offsets)}, or None to
    # recompute them here. Every caller says which (no default): reading rows that do NOT come
    # from the matching forward solve would be silently wrong, so there is no safe fallback to
    # guess at. See gpurec/core/inference/forward.pi_wave_forward's ``gene_split_out``.
    forward_gene_split: dict | None,
    theta: torch.Tensor, receiver_weights: torch.Tensor, uniform_pibar_row_max: torch.Tensor,
    family_idx: torch.Tensor,
    leaf_fm_log: torch.Tensor | None = None,
    specieswise: bool = False,
    genewise: bool = False,
    neumann_terms: int | None = None,
    neumann_term_tol: float | None = None,
    adjoint_self_loop: str | None = None,
    wide_row: torch.Tensor | None = None,
    e_adjoint_max_iter: int | None = None,
    e_adjoint_tol=None,
    adjoint_pruning_threshold: float | None = None,
    use_adjoint_pruning: bool = True,
    pibar_side_threshold: float | None = None,
    collect_backward_relres: bool = False,
    warm_v: dict | None = None,
    reserved_scratch_bytes: int | None = None,
    seed_root: torch.Tensor | None = None,
    drop_norm: bool = False,
    cache: dict | None = None,
    origination_log_probs: torch.Tensor | None = None,
    origination_probs: torch.Tensor | None = None,
    accumulator_dtype: torch.dtype | None = None,
    pi_offset: torch.Tensor,
    pibar_offset: torch.Tensor,
):
    if neumann_terms is None:
        neumann_terms = SolverOptions().neumann_terms
    neumann_terms = int(neumann_terms)
    if neumann_terms < 0:
        raise ValueError("neumann_terms must be non-negative")
    if neumann_term_tol is None:
        neumann_term_tol = SolverOptions().neumann_term_tol
    neumann_term_tol = float(neumann_term_tol)
    if neumann_term_tol < 0.0:
        raise ValueError("neumann_term_tol must be non-negative")
    if adjoint_self_loop is None:
        adjoint_self_loop = SolverOptions().adjoint_self_loop
    if e_adjoint_max_iter is None:
        e_adjoint_max_iter = SolverOptions().e_adjoint_max_iter
    if adjoint_pruning_threshold is None:
        adjoint_pruning_threshold = SolverOptions().adjoint_pruning_threshold
    adjoint_pruning_threshold = float(adjoint_pruning_threshold)
    if adjoint_pruning_threshold < 0.0:
        raise ValueError("adjoint_pruning_threshold must be non-negative")
    if pibar_side_threshold is None:
        pibar_side_threshold = SolverOptions().pibar_side_threshold
    pibar_side_threshold = float(pibar_side_threshold)
    if pibar_side_threshold < 0.0:
        raise ValueError("pibar_side_threshold must be non-negative")

    C, S = Pi_star_wave.shape
    device = Pi_star_wave.device
    dtype = Pi_star_wave.dtype
    offset_dtype = pi_offset.dtype
    if offset_dtype not in (torch.float32, torch.float64):
        raise TypeError("pi_offset must use torch.float32 or torch.float64")
    for name, value in (("pi_offset", pi_offset), ("pibar_offset", pibar_offset)):
        if value.ndim != 1 or int(value.shape[0]) != int(C):
            raise ValueError(f"{name} must have shape [{int(C)}]")
        if value.dtype != offset_dtype:
            raise TypeError("Pi/Pibar offsets must share one accumulator dtype")
        if value.device != device:
            raise ValueError(f"{name} must be on the Pi device")
    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=offset_dtype,
    )
    if accumulator_dtype != offset_dtype:
        raise TypeError("accumulator_dtype must match the Pi/Pibar offset dtype")
    pi_offset = pi_offset.contiguous()
    pibar_offset = pibar_offset.contiguous()
    family_rows = int(E_star.shape[0])
    E_family, Ebar_family, log_pS_family, log_pD_family, max_transfer_family = (
        as_family_species(x, S, family_rows)
        for x in (E_star, Ebar, log_pS, log_pD, max_transfer_mat)
    )
    log_pD_param, log_pS_param = (as_family_param(x, family_rows, S) for x in (log_pD, log_pS))
    DL_family = 1.0 + log_pD_family + E_family
    SL1_family = log_pS_family + as_family_species(E_s2, S, family_rows)
    SL2_family = log_pS_family + as_family_species(E_s1, S, family_rows)
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_log_pD, grad_log_pS = (torch.zeros_like(x) for x in (log_pD_param, log_pS_param))
    grad_max_transfer_mat = torch.zeros_like(max_transfer_family)
    # None means "nobody asked for this gradient": every kernel below reads that as a switch and
    # skips the receiver-side work entirely, and the theta VJP at the end drops the term that
    # would have consumed it.
    grad_receiver_log_probs = (
        torch.zeros((S,), device=device, dtype=dtype) if need_receiver_grad else None
    )
    grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc = (
        torch.zeros_like(x) for x in (E_star, Ebar, E_star, E_star)
    )
    root_ids = wave_layout["root_clade_ids"]
    root_Pi = Pi_star_wave.index_select(0, root_ids)
    # Numerator seed = -softmax over the origination-weighted root rows. Uniform default
    # (origination_log_probs is None) is the plain softmax(root_Pi); a constant origination_log_probs
    # is only a shift, which softmax ignores, so this is identical at uniform.
    # seed_root=None -> the loss seed -softmax2(root_Pi_w) (production); a caller-supplied root
    # cotangent (the GGN/J^T path) is used verbatim.
    seed = (
        _likelihood_root_seed(
            root_Pi,
            origination_log_probs,
            accumulator_dtype=accumulator_dtype,
        )
        if seed_root is None
        else seed_root.to(dtype)
    )
    # The dense adjoint state retains the configured dtype.
    seed = seed.to(device=device, dtype=dtype)
    accumulated_rhs.index_copy_(0, root_ids, seed)
    def _scatter_accum(acc: torch.Tensor, family_rows_for_wave: torch.Tensor, contrib: torch.Tensor) -> None:
        if contrib.dtype != acc.dtype:
            contrib = contrib.to(dtype=acc.dtype)
        if int(family_rows) == 1:
            if acc.ndim == 1:
                acc[0] += contrib.sum()
            elif int(acc.shape[1]) == 1:
                acc[0, 0] += contrib.sum()
            else:
                acc[0] += contrib.sum(dim=0)
            return
        if acc.ndim == 1:
            acc.index_add_(0, family_rows_for_wave, contrib.sum(dim=1))
        elif int(acc.shape[1]) == 1:
            acc[:, 0].index_add_(0, family_rows_for_wave, contrib.sum(dim=1))
        else:
            acc.index_add_(0, family_rows_for_wave, contrib)

    sp_child1 = species_helpers["sp_child1"]
    sp_child2 = species_helpers["sp_child2"]
    compact_level_ptr = species_helpers["compact_level_ptr"]
    compact_level_parents = species_helpers["compact_level_parents"]
    compact_level_child1 = species_helpers["compact_level_child1"]
    compact_level_child2 = species_helpers["compact_level_child2"]
    leaf_species_idx = wave_layout["leaf_species_index"].to(device=device, dtype=torch.int32).contiguous()

    # Diagnostic: per-family max relative size of the last Neumann increment (stiffness).
    # Uses the true per-clade family map (batch-local), independent of the rate `family_idx`.
    backward_relres = None
    backward_vk_mag = None
    if collect_backward_relres:
        clade_family = wave_layout["family_idx"].to(device=device, dtype=torch.long)
        n_fam = int(clade_family.max().item()) + 1 if clade_family.numel() else 0
        backward_relres = torch.zeros(n_fam, device=device, dtype=torch.float32)
        backward_vk_mag = torch.zeros(n_fam, device=device, dtype=torch.float32)

    # One free-memory reading for the whole reverse sweep instead of one per wave. Each wave's
    # self-loop scratch is allocated and freed inside that wave, so every wave used to read the
    # same number back at the cost of a blocking cudaMemGetInfo and two memory_stats() dict
    # builds -- 1977 of them per 200-family Coleman gradient. See memory_policy.
    wave_scratch_budget = wave_scratch_budget_bytes(reserved_scratch_bytes, device=device)

    for meta in reversed(wave_layout["wave_metas"]):
        ws = int(meta["start"])
        W = int(meta["W"])
        init_v = warm_v.get(ws) if warm_v is not None else None   # per-wave adjoint warm-start
        rhs_k = accumulated_rhs[ws : ws + W]
        active_mask = compute_active_adjoint_row_mask(
            rhs_k,
            adjoint_pruning_threshold,
            use_pruning=bool(use_adjoint_pruning),
        ).contiguous()
        has_splits = bool(meta.get("has_splits", "sl" in meta))
        has_leaf_term = int(meta.get("phase", 1 if not has_splits else 2)) == 1
        kept_gene_split = None if forward_gene_split is None else forward_gene_split.get(ws)
        if has_splits and kept_gene_split is not None:
            # The forward already reduced these rows from the very same Pi/Pibar child rows this
            # wave reads (a child's rows are written by its own, earlier wave and never touched
            # again), so recomputing them here reproduces them exactly. The forward's copy covers
            # every parent row while this wave's may prune some; the extra rows are never read,
            # because every consumer below is masked by ``active_mask``.
            dts_r, dts_offset = kept_gene_split
        elif has_splits:
            dts_r, dts_offset = compute_dts_forward(
                Pi_star_wave.detach(),
                pi_offset,
                Pibar_star_wave.detach(),
                pibar_offset,
                meta["sl"],
                meta["sr"],
                sp_child1,
                sp_child2,
                W,
                meta["reduce_idx"],
                log_pD_param,
                log_pS_param,
                family_idx=family_idx,
                log_split_probs=_select_log_split_probs(meta, Pi_star_wave.dtype),
                n_single_split_parents=meta.get("n_eq1"),
                single_split_parent_rows=meta.get("eq1_reduce_idx"),
                multiple_split_group_ptr=meta.get("ge2_ptr"),
                multiple_split_parent_rows=meta.get("ge2_parent_ids"),
                max_splits_per_multiple_parent=meta.get("ge2_max_fanout"),
                active_parent_rows=active_mask,
                family_offset=ws,
            )
        else:
            dts_r = None
            dts_offset = None
        backward_out = solve_reconciliation_wave_vjp(
            Pi_star_wave,
            Pibar_star_wave,
            ws,
            W,
            S,
            dts_r,
            rhs_k,
            max_transfer_family,
            DL_family,
            Ebar_family,
            E_family,
            SL1_family,
            SL2_family,
            receiver_log_probs,
            sp_child1,
            sp_child2,
            None,
            neumann_terms=neumann_terms,
            neumann_term_tol=neumann_term_tol,
            adjoint_self_loop=adjoint_self_loop,
            wide_row=wide_row,
            leaf_species_idx=leaf_species_idx,
            leaf_logp=log_pS_family,
            # E-only fraction-missing (AleRax v1.4.0 model): the Pi backward gets
            # NO fraction-missing leaf term, matching the Pi forward. Fraction-missing
            # flows only through the E-step gradient (_e_adjoint_and_theta_vjp below,
            # which forwards `leaf_fm_log` to its e_step_triton_autograd calls).
            leaf_fm_log=None,
            has_leaf_term=has_leaf_term,
            active_mask=active_mask,
            species_parent=species_helpers["sp_parent"],
            species_height=species_helpers["sp_height"],
            species_subtree_start=species_helpers["sp_subtree_start"],
            species_subtree_end=species_helpers["sp_subtree_end"],
            pibar_row_max=uniform_pibar_row_max,
            family_idx=family_idx,
            family_indexed_consts=True,
            compact_level_ptr=species_helpers["compact_level_ptr"],
            compact_level_parents=compact_level_parents,
            compact_level_child1=compact_level_child1,
            compact_level_child2=compact_level_child2,
            grad_receiver_log_probs=grad_receiver_log_probs,
            use_receiver_weights=use_receiver_weights,
            return_last_increment=collect_backward_relres,
            initial_v=init_v,
            reserved_scratch_bytes=wave_scratch_budget,
            pi_offset=pi_offset,
            pibar_offset=pibar_offset,
            gene_split_offset=dts_offset,
        )
        if collect_backward_relres:
            (
                v_k,
                duplication_loss_event_vjp,
                transfer_loss_event_vjp,
                transfer_event_vjp,
                speciation_leaf_event_vjp,
                speciation_child1_event_vjp,
                speciation_child2_event_vjp,
                last_relres,
            ) = backward_out
            wave_family = clade_family[ws : ws + W]
            row_active = active_mask.reshape(active_mask.shape[0], -1).ne(0).any(dim=1)
            vk_norm = torch.where(
                row_active, v_k.float().norm(dim=1), torch.zeros(W, device=device, dtype=torch.float32)
            )
            backward_vk_mag.scatter_reduce_(
                0,
                wave_family,
                vk_norm,
                reduce="amax",
                include_self=True,
            )
            if last_relres is not None:
                backward_relres.scatter_reduce_(
                    0,
                    wave_family,
                    last_relres.to(dtype=torch.float32),
                    reduce="amax",
                    include_self=True,
                )
        else:
            (
                v_k,
                duplication_loss_event_vjp,
                transfer_loss_event_vjp,
                transfer_event_vjp,
                speciation_leaf_event_vjp,
                speciation_child1_event_vjp,
                speciation_child2_event_vjp,
            ) = backward_out
        if warm_v is not None:
            # Cache the solved adjoint for next call's warm-start, but ZERO the pruned/inactive rows first
            # -- they hold uninitialized scratch, and reusing that garbage as initial_v poisons the next
            # solve (NaN in the downstream E-adjoint), especially when the active set shifts between thetas.
            # NaN-safe select (NOT multiply: inactive rows hold uninitialized scratch = NaN/inf, and
            # 0.0 * NaN = NaN, which would poison the next warm-start). torch.where drops them cleanly.
            _row_active = active_mask.reshape(active_mask.shape[0], -1).ne(0).any(dim=1)
            warm_v[ws] = torch.where(
                _row_active.unsqueeze(-1), v_k, torch.zeros((), dtype=v_k.dtype, device=v_k.device)
            ).detach()
        if cache is not None:
            # per-wave adjoint state for the exact-HVP tangent sweep (theta fixed across CG). Pruning
            # leaves inactive v_k rows uninitialized; the primal never reads them but the second-order
            # contraction does -> sanitize with the row mask.
            row_active = (active_mask.reshape(W, -1) != 0).any(dim=1)
            v_clean = torch.where(row_active.unsqueeze(1), v_k, torch.zeros_like(v_k))
            cache.setdefault("waves", []).append(dict(
                ws=ws, W=W, v=v_clean, dts_r=dts_r, dts_offset=dts_offset,
                active_mask=active_mask,
                has_splits=has_splits, has_leaf_term=has_leaf_term, meta=meta,
            ))
        family_rows_for_wave = family_idx[ws : ws + W]
        _scatter_accum(grad_log_pD, family_rows_for_wave, duplication_loss_event_vjp)
        _scatter_accum(grad_log_pS, family_rows_for_wave, speciation_leaf_event_vjp)
        _scatter_accum(
            grad_E_acc,
            family_rows_for_wave,
            duplication_loss_event_vjp + transfer_event_vjp,
        )
        _scatter_accum(grad_Ebar_acc, family_rows_for_wave, transfer_loss_event_vjp)
        _scatter_accum(grad_E_s1_acc, family_rows_for_wave, speciation_child2_event_vjp)
        _scatter_accum(grad_E_s2_acc, family_rows_for_wave, speciation_child1_event_vjp)
        _scatter_accum(grad_max_transfer_mat, family_rows_for_wave, transfer_event_vjp)
        if has_splits and dts_r is not None:
            split_left_rows = meta["sl"]
            split_right_rows = meta["sr"]
            (
                donor_adjoint,
                active_donor_side,
                _duplication_parameter_vjp,
                _speciation_parameter_vjp,
            ) = accumulate_gene_split_event_vjp(
                Pi_star_wave,
                Pibar_star_wave,
                v_k,
                ws,
                split_left_rows,
                split_right_rows,
                meta["reduce_idx"],
                _select_log_split_probs(meta, Pi_star_wave.dtype),
                log_pD_param,
                log_pS_param,
                sp_child1,
                sp_child2,
                accumulated_rhs,
                S,
                active_mask=active_mask,
                merge_s_term=True,
                grad_log_pD=grad_log_pD,
                grad_log_pS=grad_log_pS,
                grad_max_transfer=grad_max_transfer_mat,
                accum_param_reductions=True,
                accum_max_transfer_reduction=True,
                output_donor_adjoint=True,
                output_active_donor_sides=True,
                pibar_side_threshold=pibar_side_threshold,
                max_transfer=max_transfer_family,
                pibar_row_max=uniform_pibar_row_max,
                grad_max_transfer_two_stage=bool(
                    grad_max_transfer_mat.ndim == 2
                    and int(grad_max_transfer_mat.shape[0]) == 1
                ),
                grad_max_transfer_two_stage_tile_splits=128,
                skip_inactive_pibar_output_zero=True,
                family_idx=family_idx,
                pi_offset=pi_offset,
                pibar_offset=pibar_offset,
            )
            accumulate_transfer_complement_vjp_from_donor_adjoint(
                Pi_star_wave,
                receiver_log_probs,
                donor_adjoint,
                split_left_rows,
                split_right_rows,
                accumulated_rhs,
                S,
                species_helpers["sp_parent"],
                active_mask=active_mask,
                reduce_idx=meta["reduce_idx"],
                pibar_row_max=uniform_pibar_row_max,
                skip_zero_donor_sides=True,
                active_donor_side=active_donor_side,
                compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents,
                compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2,
                grad_receiver_log_probs=grad_receiver_log_probs,
                use_receiver_weights=use_receiver_weights,
                side_active_threshold=pibar_side_threshold,
            )
    if collect_backward_relres:
        # Diagnostic short-circuit: the per-family backward residual is fully accumulated
        # from the per-wave self-loop solves; the E-adjoint solve is not needed here.
        return backward_relres, backward_vk_mag
    if cache is not None:
        cache["accum"] = dict(
            grad_E=grad_E_acc, grad_Ebar=grad_Ebar_acc, grad_E_s1=grad_E_s1_acc,
            grad_E_s2=grad_E_s2_acc, grad_log_pD=grad_log_pD, grad_log_pS=grad_log_pS,
            grad_mc=grad_max_transfer_mat, grad_col=grad_receiver_log_probs,
        )
    return _e_adjoint_and_theta_vjp(
        E_star, log_pS, log_pD, log_pL, max_transfer_mat,
        receiver_log_probs,
        use_receiver_weights,
        grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc,
        grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
        int(root_ids.numel()), theta, receiver_weights, species_helpers,
        specieswise=specieswise,
        genewise=genewise,
        leaf_fm_log=leaf_fm_log,
        drop_norm=drop_norm,
        e_adjoint_max_iter=e_adjoint_max_iter,
        e_adjoint_tol=e_adjoint_tol,
        cache=cache,
        origination_probs=origination_probs,
        accumulator_dtype=accumulator_dtype,
    )


def _e_adjoint_and_theta_vjp(
    E_star, log_pS, log_pD, log_pL, max_transfer_mat, receiver_log_probs, use_receiver_weights,
    grad_E, grad_Ebar, grad_E_s1, grad_E_s2,
    grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
    n_fam, theta, receiver_weights, species_helpers, *, specieswise, genewise,
    leaf_fm_log: torch.Tensor | None = None,
    drop_norm: bool = False,
    e_adjoint_max_iter: int | None = None,
    e_adjoint_tol=None,
    cache=None,
    origination_probs=None,
    accumulator_dtype: torch.dtype | None = None,
):
    if e_adjoint_max_iter is None:
        e_adjoint_max_iter = SolverOptions().e_adjoint_max_iter
    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=E_star.dtype,
    )
    topology_args = (
        species_helpers["sp_parent"],
        species_helpers["sp_child1"],
        species_helpers["sp_child2"],
        # Each species' height (0 at a leaf) and the tree's height: the extinction complement adds
        # up the valid receivers level by level instead of subtracting the ancestor chain.
        species_helpers["sp_height"],
        int(species_helpers["compact_level_ptr"].numel()) - 1,
    )

    E_req = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        triton_E_from_E, E_s1_from_E, E_s2_from_E, Ebar_from_E = e_step_triton_autograd(
            E_req,
            log_pS,
            log_pD,
            log_pL,
            max_transfer_mat,
            receiver_log_probs,
            *topology_args,
            use_receiver_weights=use_receiver_weights,
            leaf_fm_log=leaf_fm_log,
        )
        # ``drop_norm`` (GGN/J^T use) skips the loss's explicit E-normalization term, which is not
        # part of d(Pi_root)/dtheta. Default False -> the full real gradient (production path).
        aux_outputs = (E_s1_from_E, E_s2_from_E, Ebar_from_E)
        aux_grads = (grad_E_s1, grad_E_s2, grad_Ebar)
        if not drop_norm:
            denom = _likelihood_log2_survival(
                E_req,
                origination_probs,
                accumulator_dtype=accumulator_dtype,
            )
            direct_obj = denom.sum() if E_req.shape[0] == n_fam else (n_fam * denom).sum()
            aux_outputs = (direct_obj, *aux_outputs)
            aux_grads = (torch.ones_like(direct_obj), *aux_grads)
        (aux_to_e,) = torch.autograd.grad(aux_outputs, E_req, grad_outputs=aux_grads, retain_graph=True)
    q_E = grad_E + aux_to_e

    E_shape = E_star.shape
    q_flat = q_E.reshape(-1)

    def AG_flat(w_flat):
        wE = w_flat.view(E_shape).contiguous()
        with torch.enable_grad():
            (gE,) = torch.autograd.grad(
                triton_E_from_E,
                E_req,
                # autograd.grad treats grad_outputs as read-only, so the
                # defensive clone is unnecessary; passing wE directly avoids a
                # full E-vector copy on every E-adjoint matvec.
                grad_outputs=wE,
                retain_graph=True,
            )
        return (wE - gE).reshape(-1)

    # Linear E-adjoint solve ``(I - J) wE = q`` via Neumann series (see _neumann_e_adjoint).
    wE = _neumann_e_adjoint(
        AG_flat,
        q_flat,
        max_iter=e_adjoint_max_iter,
        tol=e_adjoint_tol,
    ).view(E_shape)
    if cache is not None:
        cache["e_side"] = dict(q_E=q_E, wE=wE, aux_to_e=aux_to_e)

    theta_req = theta.detach().requires_grad_(True)
    receiver_req = receiver_weights.detach().requires_grad_(True)
    with torch.enable_grad():
        log_pS_r, log_pD_r, log_pL_r, mt_r, receiver_log_probs_r = extract_parameters_weighted_receivers(
            theta_req,
            receiver_req,
            species_helpers,
            specieswise=specieswise,
            genewise=genewise,
            uniform_fast=not use_receiver_weights,
            accumulator_dtype=accumulator_dtype,
        )
        S = int(species_helpers["S"])
        family_rows = int(E_star.shape[0])
        log_pS_param = as_family_param(log_pS_r, family_rows, S)
        log_pD_param = as_family_param(log_pD_r, family_rows, S)
        param_loss = (
            (log_pS_param * grad_log_pS).sum()
            + (log_pD_param * grad_log_pD).sum()
            + (mt_r * grad_max_transfer_mat).sum()
        )
        if grad_receiver_log_probs is not None:
            # Only present when the caller asked for the receiver-weight gradient; without it the
            # reconciliation half of dNLL/d(receiver logits) was never accumulated, so there is
            # nothing to contract here. theta itself is unaffected either way: the receiver
            # log-probabilities are a softmax of the receiver weights alone, so this term
            # contributes to grad_receiver and to nothing else.
            param_loss = param_loss + (receiver_log_probs_r * grad_receiver_log_probs).sum()
        E_from_params, _, _, Ebar_from_params = e_step_triton_autograd(
            E_star.detach(),
            log_pS_r,
            log_pD_r,
            log_pL_r,
            mt_r,
            receiver_log_probs_r,
            *topology_args,
            use_receiver_weights=use_receiver_weights,
            leaf_fm_log=leaf_fm_log,
        )
        # ``grad_receiver_log_probs is None`` is the caller's "I will not use the receiver
        # gradient": ask autograd for theta alone (which also skips the receiver branch of this
        # graph) and hand back None, so a caller that uses it anyway fails at once instead of
        # silently reading a value that is missing its reconciliation half.
        wanted = (theta_req,) if grad_receiver_log_probs is None else (theta_req, receiver_req)
        grads = torch.autograd.grad(
            (param_loss, Ebar_from_params, E_from_params),
            wanted,
            grad_outputs=(torch.ones_like(param_loss), grad_Ebar, wE),
        )
    if grad_receiver_log_probs is None:
        return grads[0], None
    return grads[0], grads[1]
