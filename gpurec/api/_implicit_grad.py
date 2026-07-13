import warnings

import torch

from gpurec.api.solver_options import SolverOptions
from gpurec.config import dtype_rel_tol_default as _bicgstab_rel_tol_default, dtype_rel_tol_floor as _bicgstab_rel_tol_floor
from gpurec.core.inference.logspace import logsumexp2 as _logsumexp2, log2_survival as _log2_survival
from gpurec.core.kernels.pi_forward import compute_dts_forward
from gpurec.core.kernels.wave_backward import (
    active_mask_from_rhs_absmax_fused,
    dts_cross_backward_accum_fused,
    uniform_cross_pibar_vjp_tree_from_ud_fused,
    wave_backward_uniform_fused,
)
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


# `_bicgstab_rel_tol_default` / `_bicgstab_rel_tol_floor` moved to
# `gpurec.config.gpurec_config` (as `dtype_rel_tol_default` / `dtype_rel_tol_floor`)
# and re-exported above under their original names for back-compat: tests and
# other callers still import them from here.


@torch.no_grad()
def _bicgstab(
    Av,
    b: torch.Tensor,
    *,
    max_iter: int | None = None,
    tol=None,
    breakdown_tol=None,
):
    """BiCGSTAB for the E-adjoint / GGN linear solve, dtype-aware and fail-safe.

    ``tol`` is a RELATIVE residual target (``||r||/max(||b||,1)``). ``None`` ->
    the dtype-matched default (:func:`_bicgstab_rel_tol_default`). A caller value
    below the dtype floor (:func:`_bicgstab_rel_tol_floor`) is clamped up with a
    warning -- a tighter relative residual is unreachable in that precision.

    ``max_iter`` -- ``None`` -> ``SolverOptions().bicgstab_max_iter`` (the single
    source of truth for the E-adjoint Krylov cap).

    ``breakdown_tol`` is a dimensionless RELATIVE factor: each Krylov inner
    product is tested against this factor times its operand norms, so a breakdown
    fires only on genuine loss-of-orthogonality / a near-null direction -- never
    on the small-but-valid inner products that occur near convergence (the old
    absolute ``1e-30`` either never tripped in fp32, or tripped spuriously near
    convergence and the solve then raised on an essentially-converged iterate).
    ``None`` -> ``eps``.

    On exit (max_iter or breakdown) the BEST iterate seen is returned whenever it
    reached the working-precision floor; only a genuinely non-converged solve
    raises ``RuntimeError``.
    """
    if max_iter is None:
        max_iter = SolverOptions().bicgstab_max_iter
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    eps = float(torch.finfo(b.dtype).eps)
    floor = _bicgstab_rel_tol_floor(b.dtype)
    if tol is None:
        target = _bicgstab_rel_tol_default(b.dtype)
    else:
        target = float(tol)
        if target <= 0.0:
            raise ValueError("tol must be positive")
        if target < floor:
            warnings.warn(
                f"bicgstab tol={target:.2e} is below the {b.dtype} finite-precision "
                f"residual floor {floor:.2e}; clamping to the floor. A tighter "
                f"relative residual is unreachable in this precision -- use fp64.",
                RuntimeWarning, stacklevel=2,
            )
            target = floor
    # Accept any iterate that reaches the dtype's natural floor as 'converged to
    # working precision', so a near-floor stall/breakdown is success, not a crash.
    accept = max(target, _bicgstab_rel_tol_default(b.dtype))
    bd = eps if breakdown_tol is None else float(breakdown_tol)
    if bd <= 0.0:
        raise ValueError("breakdown_tol must be positive")

    x = torch.zeros_like(b)
    # x0 = 0 and A is linear, so r0 = b - Av(0) = b exactly; skip the wasted
    # operator apply on the zero vector.
    r = b.clone()
    bnorm = max(float(torch.linalg.vector_norm(b).detach().cpu()), 1.0)
    rhat_norm = float(torch.linalg.vector_norm(r).detach().cpu())
    r_norm = rhat_norm
    rel_res = r_norm / bnorm
    if rel_res <= target:
        return x

    best_x = x
    best_res = rel_res

    r_hat = r.clone()
    rho_old = torch.ones((), dtype=b.dtype, device=b.device)
    alpha = torch.ones((), dtype=b.dtype, device=b.device)
    omega = torch.ones((), dtype=b.dtype, device=b.device)
    v = torch.zeros_like(b)
    p = torch.zeros_like(b)

    broke = False
    for k in range(1, max_iter + 1):
        rho = torch.dot(r_hat, r)
        # Relative breakdown: r_hat and r have lost their inner product (cosine
        # below machine precision) -> classic BiCGSTAB breakdown.
        if float(rho.abs().detach().cpu()) <= bd * rhat_norm * max(r_norm, eps):
            broke = True
            break

        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = Av(p)
        v_norm = float(torch.linalg.vector_norm(v).detach().cpu())
        denom = torch.dot(r_hat, v)
        if float(denom.abs().detach().cpu()) <= bd * rhat_norm * max(v_norm, eps):
            broke = True
            break

        alpha = rho / denom
        s = r - alpha * v
        s_norm = float(torch.linalg.vector_norm(s).detach().cpu())
        rel_s = s_norm / bnorm
        xs = x + alpha * p
        if rel_s < best_res:
            best_res = rel_s
            best_x = xs
        if rel_s <= target:
            return xs

        t = Av(s)
        tt = torch.dot(t, t)
        # Relative breakdown: A s ~ 0 with s != 0 (near-null direction); also
        # guards the omega = <t,s>/tt division below.
        if float(tt.detach().cpu()) <= (bd * max(s_norm, eps)) ** 2:
            broke = True
            break

        omega = torch.dot(t, s) / tt
        x = xs + omega * s
        r = s - omega * t
        r_norm = float(torch.linalg.vector_norm(r).detach().cpu())
        rel_res = r_norm / bnorm
        if rel_res < best_res:
            best_res = rel_res
            best_x = x
        if rel_res <= target:
            return x
        # omega is the divisor of the next iteration's beta; guard against the
        # alpha/omega blow-up when omega collapses relative to alpha.
        if float(omega.abs().detach().cpu()) <= bd * max(float(alpha.abs().detach().cpu()), eps):
            broke = True
            break
        rho_old = rho

    if best_res <= accept:
        return best_x
    why = "broke down" if broke else f"hit max_iter={max_iter}"
    raise RuntimeError(
        f"E-adjoint BiCGSTAB solve {why} at relative residual {best_res:.3e} "
        f"(target {target:.3e}, dtype {b.dtype}); the operator is likely singular "
        f"or too ill-conditioned to solve in this precision."
    )


@torch.no_grad()
def _gmres(Av, b: torch.Tensor, *, max_iter: int = 128, tol=None):
    """Matrix-free GMRES for the E-adjoint / GGN linear solve ``(I - J_E^T) w = b``.

    GMRES minimizes the residual over the growing Krylov subspace and -- unlike
    BiCGSTAB -- CANNOT break down on a nonsingular operator: no shadow-residual
    inner product to lose, so a well-conditioned but moderately non-symmetric
    E-adjoint (the case that trips BiCGSTAB's biorthogonality guard and makes it
    raise a spurious "singular/ill-conditioned" error) is solved robustly. The
    Arnoldi step uses batched classical Gram-Schmidt with one reorthogonalization
    (CGS2) -- one ``mv`` per step, matching the proven wave self-loop GMRES
    (:func:`gpurec.core.kernels.wave_backward._gmres_solve_wave_self_loop_fixed_cgs2`)
    -- and the small Hessenberg least-squares gives the true minimal residual for
    early stopping.

    ``tol`` is a RELATIVE residual target (``||r|| / max(||b||, 1)``); ``None`` ->
    the dtype-matched default (:func:`_bicgstab_rel_tol_default`, fp32 1e-6 /
    fp64 1e-12). A value below the dtype floor (:func:`_bicgstab_rel_tol_floor`)
    is clamped up with a warning -- unreachable in that precision. Returns the
    minimal-residual iterate; raises ``RuntimeError`` only if the residual never
    reaches the acceptance floor within ``max_iter`` Arnoldi steps -- a genuinely
    singular operator, never a solver artefact.
    """
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    dtype = b.dtype
    device = b.device
    floor = _bicgstab_rel_tol_floor(dtype)
    default = _bicgstab_rel_tol_default(dtype)
    if tol is None:
        target = default
    else:
        target = float(tol)
        if target <= 0.0:
            raise ValueError("tol must be positive")
        if target < floor:
            warnings.warn(
                f"gmres tol={target:.2e} is below the {dtype} finite-precision "
                f"residual floor {floor:.2e}; clamping to the floor. A tighter "
                f"relative residual is unreachable in this precision -- use fp64.",
                RuntimeWarning, stacklevel=2,
            )
            target = floor
    # Any iterate reaching the dtype's natural floor counts as converged-to-precision.
    accept = max(target, default)

    b_flat = b.reshape(-1)
    n = int(b_flat.numel())
    b_norm_t = torch.linalg.vector_norm(b_flat)
    b_norm = float(b_norm_t.detach().cpu())
    if b_norm == 0.0:
        return torch.zeros_like(b)
    scale = max(b_norm, 1.0)

    m = min(max_iter, n)  # the Krylov dimension cannot exceed the space dimension
    eps = float(torch.finfo(dtype).eps)
    tiny = float(torch.finfo(dtype).tiny)

    basis = torch.empty((m + 1, n), dtype=dtype, device=device)
    basis[0].copy_(b_flat / b_norm_t)          # x0 = 0  ->  r0 = b
    H = torch.zeros((m + 1, m), dtype=dtype, device=device)
    e1 = torch.zeros((m + 1,), dtype=dtype, device=device)
    e1[0] = b_norm_t
    coeff = torch.empty((m,), dtype=dtype, device=device)
    coeff2 = torch.empty((m,), dtype=dtype, device=device)

    best_res = b_norm / scale  # relative residual of x = 0
    y_best = None
    k_best = 0
    for j in range(m):
        w = Av(basis[j].view(b.shape)).reshape(-1)
        q = basis[: j + 1]
        c = coeff[: j + 1]
        torch.mv(q, w, out=c)
        H[: j + 1, j].copy_(c)
        w = torch.addmv(w, q.t(), c, beta=1.0, alpha=-1.0)   # CGS pass 1
        c2 = coeff2[: j + 1]
        torch.mv(q, w, out=c2)
        H[: j + 1, j].add_(c2)
        w = torch.addmv(w, q.t(), c2, beta=1.0, alpha=-1.0)  # CGS pass 2 (reorthogonalize)
        h_next_t = torch.linalg.vector_norm(w)
        H[j + 1, j] = h_next_t

        # Minimal residual of the current Krylov iterate via the (j+2, j+1) Hessenberg LS.
        y = torch.linalg.lstsq(H[: j + 2, : j + 1], e1[: j + 2]).solution
        res = float(torch.linalg.vector_norm(e1[: j + 2] - H[: j + 2, : j + 1] @ y).detach().cpu())
        rel = res / scale
        if rel <= best_res:
            best_res = rel
            y_best = y
            k_best = j + 1

        h_next = float(h_next_t.detach().cpu())
        if rel <= target or h_next <= eps * scale:  # converged, or happy breakdown (exact)
            break
        if j + 1 < m:
            basis[j + 1].copy_(w / max(h_next, tiny))

    if y_best is not None and best_res <= accept:
        return torch.mv(basis[:k_best].t(), y_best).view(b.shape)
    raise RuntimeError(
        f"E-adjoint GMRES failed to converge at relative residual {best_res:.3e} "
        f"after {m} Arnoldi steps (target {target:.3e}, dtype {dtype}); the "
        f"operator is singular or needs more than {m} iterations to solve."
    )


@torch.no_grad()
def _neumann_e_adjoint(Av, b: torch.Tensor, *, max_iter: int = 128, tol=None):
    """Solve ``(I - J) x = b`` by Neumann series ``x = sum_k J^k b``, where ``J w = w - Av(w)``.

    Alternative to :func:`_gmres` for the E-adjoint / GGN linear solve. No orthogonalization
    -> no fp32 Arnoldi residual floor (unlike GMRES, which stalls at a theta-dependent ~1e-6 to
    5.5e-6 relative residual at large species counts). Valid because the E-step self-map
    Jacobian ``J`` is a contraction (the forward E fixed point converges), so
    ``(I-J)^{-1} = sum_k J^k`` converges.

    By telescoping, the true relative residual after summing terms ``k=0..N`` is
    ``||J^{N+1} b|| / ||b||``; since ``||J|| < 1`` this is bounded above by
    ``||J^N b|| / ||b||`` -- the quantity actually tracked below (``rel``) -- so ``rel`` is a
    conservative (upper-bound) proxy for the true residual, off by one power of ``J``.

    ``tol`` is a RELATIVE residual target, matching :func:`_gmres` exactly: ``None`` -> the
    dtype-matched default (:func:`_bicgstab_rel_tol_default`); a value below the dtype floor
    (:func:`_bicgstab_rel_tol_floor`) is clamped up with a warning. Returns the best (smallest
    conservative-residual) iterate; raises ``RuntimeError`` only if that residual never reaches
    the acceptance floor within ``max_iter`` terms -- a genuinely non-contractive operator, never
    a solver artefact.
    """
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    dtype = b.dtype
    floor = _bicgstab_rel_tol_floor(dtype)
    default = _bicgstab_rel_tol_default(dtype)
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
    for _ in range(max_iter):
        term = term - Av(term)  # == J @ term
        x = x + term
        rel = float(torch.linalg.vector_norm(term.reshape(-1)).detach().cpu()) / bnorm
        if rel < best_rel:
            best_rel = rel
            best_x = x
        if rel <= target:
            return x

    if best_rel <= accept:
        return best_x
    raise RuntimeError(
        f"E-adjoint Neumann series failed to converge at conservative relative residual "
        f"{best_rel:.3e} after {max_iter} terms (target {target:.3e}, dtype {dtype}); the "
        f"self-map J is likely not a contraction (spectral radius >= 1) or needs more than "
        f"{max_iter} terms to solve."
    )


@torch.no_grad()
def implicit_grad_loglik_vjp_wave(
    wave_layout, species_helpers, *, Pi_star_wave: torch.Tensor,
    Pibar_star_wave: torch.Tensor, E_star: torch.Tensor, E_s1: torch.Tensor,
    E_s2: torch.Tensor, Ebar: torch.Tensor, log_pS: torch.Tensor,
    log_pD: torch.Tensor, log_pL: torch.Tensor, max_transfer_mat: torch.Tensor,
    receiver_log_probs: torch.Tensor,
    theta: torch.Tensor, receiver_weights: torch.Tensor, uniform_pibar_row_max: torch.Tensor,
    family_idx: torch.Tensor,
    specieswise: bool = False,
    genewise: bool = False,
    neumann_terms: int | None = None,
    self_loop_solver: str = "neumann",
    bicgstab_max_iter: int | None = None,
    bicgstab_tol=None,
    bicgstab_breakdown_tol=None,
    e_adjoint_solver: str | None = None,
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
    self_loop_solver = str(self_loop_solver).strip().lower()
    if self_loop_solver not in ("neumann", "gmres"):
        raise ValueError("self_loop_solver must be one of: neumann, gmres")
    if bicgstab_max_iter is None:
        bicgstab_max_iter = SolverOptions().bicgstab_max_iter
    if e_adjoint_solver is None:
        e_adjoint_solver = SolverOptions().e_adjoint_solver
    e_adjoint_solver = str(e_adjoint_solver).strip().lower()
    if e_adjoint_solver not in ("gmres", "neumann"):
        raise ValueError("e_adjoint_solver must be one of: gmres, neumann")
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
    grad_receiver_log_probs = torch.zeros((S,), device=device, dtype=dtype)
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

    for meta in reversed(wave_layout["wave_metas"]):
        ws = int(meta["start"])
        W = int(meta["W"])
        init_v = warm_v.get(ws) if warm_v is not None else None   # per-wave adjoint warm-start
        rhs_k = accumulated_rhs[ws : ws + W]
        active_mask = active_mask_from_rhs_absmax_fused(
            rhs_k,
            adjoint_pruning_threshold,
            use_pruning=bool(use_adjoint_pruning),
        ).contiguous()
        has_splits = bool(meta.get("has_splits", "sl" in meta))
        has_leaf_term = int(meta.get("phase", 1 if not has_splits else 2)) == 1
        if has_splits:
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
                log_split_probs=meta.get("log_split_probs"),
                n_eq1=meta.get("n_eq1"),
                eq1_reduce_idx=meta.get("eq1_reduce_idx"),
                ge2_ptr=meta.get("ge2_ptr"),
                ge2_parent_ids=meta.get("ge2_parent_ids"),
                ge2_max_fanout=meta.get("ge2_max_fanout"),
                active_parent_rows=active_mask,
                family_offset=ws,
            )
        else:
            dts_r = None
            dts_offset = None
        backward_out = wave_backward_uniform_fused(
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
            leaf_species_idx=leaf_species_idx,
            leaf_logp=log_pS_family,
            has_leaf_term=has_leaf_term,
            active_mask=active_mask,
            sp_parent=species_helpers["sp_parent"],
            max_ancestor_depth=int(species_helpers["max_ancestor_depth"]),
            pibar_row_max=uniform_pibar_row_max,
            family_idx=family_idx,
            family_indexed_consts=True,
            compact_level_ptr=species_helpers["compact_level_ptr"],
            compact_level_parents=compact_level_parents,
            compact_level_child1=compact_level_child1,
            compact_level_child2=compact_level_child2,
            grad_receiver_log_probs=grad_receiver_log_probs,
            self_loop_solver=self_loop_solver,
            return_last_increment=collect_backward_relres,
            initial_v=init_v,
            reserved_scratch_bytes=reserved_scratch_bytes,
            pi_offset=pi_offset,
            pibar_offset=pibar_offset,
            dts_offset=dts_offset,
        )
        if collect_backward_relres:
            v_k, aw0, aw1, aw2, aw345, aw3, aw4, last_relres = backward_out
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
            v_k, aw0, aw1, aw2, aw345, aw3, aw4 = backward_out
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
        _scatter_accum(grad_log_pD, family_rows_for_wave, aw0)
        _scatter_accum(grad_log_pS, family_rows_for_wave, aw345)
        _scatter_accum(grad_E_acc, family_rows_for_wave, aw0 + aw2)
        _scatter_accum(grad_Ebar_acc, family_rows_for_wave, aw1)
        _scatter_accum(grad_E_s1_acc, family_rows_for_wave, aw4)
        _scatter_accum(grad_E_s2_acc, family_rows_for_wave, aw3)
        _scatter_accum(grad_max_transfer_mat, family_rows_for_wave, aw2)
        if has_splits and dts_r is not None:
            sl = meta["sl"]
            sr = meta["sr"]
            grad_Pibar_l, grad_Pibar_r, pibar_side_active, _param_pD, _param_pS = dts_cross_backward_accum_fused(
                Pi_star_wave,
                Pibar_star_wave,
                v_k,
                ws,
                sl,
                sr,
                meta["reduce_idx"],
                meta.get("log_split_probs", sl.new_zeros((int(sl.numel()),), dtype=Pi_star_wave.dtype)),
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
                grad_mt=grad_max_transfer_mat,
                accum_param_reductions=True,
                accum_mt_reduction=True,
                output_pibar_ud=True,
                output_pibar_side_active=True,
                pibar_side_threshold=pibar_side_threshold,
                mt_squeezed=max_transfer_family,
                pibar_row_max=uniform_pibar_row_max,
                grad_mt_two_stage=bool(grad_max_transfer_mat.ndim == 2 and int(grad_max_transfer_mat.shape[0]) == 1),
                grad_mt_two_stage_tile_splits=128,
                skip_inactive_pibar_output_zero=True,
                family_idx=family_idx,
                pi_offset=pi_offset,
                pibar_offset=pibar_offset,
            )
            uniform_cross_pibar_vjp_tree_from_ud_fused(
                Pi_star_wave,
                receiver_log_probs,
                grad_Pibar_l,
                grad_Pibar_r,
                sl,
                sr,
                accumulated_rhs,
                S,
                active_mask=active_mask,
                reduce_idx=meta["reduce_idx"],
                pibar_row_max=uniform_pibar_row_max,
                skip_zero_sides=True,
                side_active=pibar_side_active,
                compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents,
                compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2,
                grad_receiver_log_probs=grad_receiver_log_probs,
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
        grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc,
        grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
        int(root_ids.numel()), theta, receiver_weights, species_helpers,
        specieswise=specieswise,
        genewise=genewise,
        drop_norm=drop_norm,
        bicgstab_max_iter=bicgstab_max_iter,
        bicgstab_tol=bicgstab_tol,
        bicgstab_breakdown_tol=bicgstab_breakdown_tol,
        e_adjoint_solver=e_adjoint_solver,
        cache=cache,
        origination_probs=origination_probs,
        accumulator_dtype=accumulator_dtype,
    )


def _e_adjoint_and_theta_vjp(
    E_star, log_pS, log_pD, log_pL, max_transfer_mat, receiver_log_probs,
    grad_E, grad_Ebar, grad_E_s1, grad_E_s2,
    grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
    n_fam, theta, receiver_weights, species_helpers, *, specieswise, genewise,
    drop_norm: bool = False,
    bicgstab_max_iter: int | None = None,
    bicgstab_tol=None,
    bicgstab_breakdown_tol=None,
    e_adjoint_solver: str | None = None,
    cache=None,
    origination_probs=None,
    accumulator_dtype: torch.dtype | None = None,
):
    if bicgstab_max_iter is None:
        bicgstab_max_iter = SolverOptions().bicgstab_max_iter
    if e_adjoint_solver is None:
        e_adjoint_solver = SolverOptions().e_adjoint_solver
    e_adjoint_solver = str(e_adjoint_solver).strip().lower()
    if e_adjoint_solver not in ("gmres", "neumann"):
        raise ValueError("e_adjoint_solver must be one of: gmres, neumann")
    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=E_star.dtype,
    )
    topology_args = (
        species_helpers["sp_parent"],
        species_helpers["sp_child1"],
        species_helpers["sp_child2"],
        int(species_helpers["max_ancestor_depth"]),
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

    # Linear E-adjoint solve ``(I - J) wE = q``. ``bicgstab_breakdown_tol`` is a
    # BiCGSTAB-only guard and has no analogue in GMRES/Neumann, so it is not threaded here.
    # Default "gmres" (breakdown-free); "neumann" avoids GMRES's fp32 Arnoldi
    # orthogonalization floor at large species counts (see _neumann_e_adjoint).
    solver = _neumann_e_adjoint if e_adjoint_solver == "neumann" else _gmres
    wE = solver(
        AG_flat,
        q_flat,
        max_iter=bicgstab_max_iter,
        tol=bicgstab_tol,
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
            + (receiver_log_probs_r * grad_receiver_log_probs).sum()
        )
        E_from_params, _, _, Ebar_from_params = e_step_triton_autograd(
            E_star.detach(),
            log_pS_r,
            log_pD_r,
            log_pL_r,
            mt_r,
            receiver_log_probs_r,
            *topology_args,
        )
        grad_theta, grad_receiver = torch.autograd.grad(
            (param_loss, Ebar_from_params, E_from_params),
            (theta_req, receiver_req),
            grad_outputs=(torch.ones_like(param_loss), grad_Ebar, wE),
        )
    return grad_theta, grad_receiver
