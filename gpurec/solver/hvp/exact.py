"""Analytic exact-Hessian HVP (forward-over-reverse) — orchestrator.

Per outer Newton point: ``build_point_cache`` runs the production backward ONCE (the verified
``vjp_root_to_theta`` loop) while caching each wave adjoint, gene-split
likelihood, and the E-side adjoint ``wE`` — theta is fixed across all CG
iterations, so the cache amortizes. Each
``hvp(u)`` then costs one tangent-forward sweep + one tangent-adjoint sweep (same solve
operators, modified seeds) + the second-order contraction kernels (e_step_so / wave_so / dts_so).

Status: point-cache + gradient reproduction (build step 2). The tangent-adjoint sweep
(steps 3-5) composes on top of this cache.
"""

from __future__ import annotations

import torch

from gpurec.api._implicit_grad import _neumann_e_adjoint, _safe_exp2_ratio
from gpurec.config.memory import MemoryOptions
from gpurec.core.inference.logspace import logsumexp2 as _logsumexp2, survival_from_E as _survival_from_E
from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.core.kernels.dts_so import dts_backward_so
from gpurec.core.kernels.e_step import e_step_triton_autograd
from gpurec.core.kernels.e_step_so import e_step_backward_so
from gpurec.core.kernels.pi_forward import _select_log_split_probs
from gpurec.core.kernels.wave_backward import (
    accumulate_gene_split_event_vjp, accumulate_transfer_complement_vjp_from_donor_adjoint,
    solve_reconciliation_wave_vjp,
)
from gpurec.core.kernels.wave_so import wave_backward_so
from gpurec.core.memory_policy import wave_scratch_budget_bytes
from gpurec.core.parameters.extract_parameters import (
    as_family_param,
    as_family_species,
    extract_parameters_weighted_receivers,
    resolve_accumulator_dtype,
)
from gpurec.solver.hvp.forward_tangent import jvp_root_scores, wave_step_constants
from gpurec.solver.hvp.gauss_newton import vjp_root_to_theta

_LN2 = 0.6931471805599453


def _single_static(static):
    """The exact HVP is single-batch; accept either a single static or a ``batch_statics`` list."""
    if isinstance(static, (list, tuple)):
        if len(static) != 1:
            raise NotImplementedError(
                "exact HVP is single-batch; multi-batch HVP is not ported (Phase 3 is single-batch)"
            )
        return static[0]
    return static


def _warm_reserved_scratch_bytes(static):
    """Mirror ``_execution.py``'s memory-gate sourcing: the warm-adjoint cache
    (``static.warm_v``) is only resident -- and thus only depletes the free-memory budget the
    gated fast path re-reads -- when ``GPUREC_WARM_ADJOINT`` is set AND the build-time gate
    (``static.warm_adjoint_ok``) allowed it. Same condition as ``evaluate_static_loss_grad``'s
    ``_warm_v is not None``; returns ``None`` (cold, unchanged) otherwise.
    """
    import os
    if os.environ.get("GPUREC_WARM_ADJOINT") and getattr(static, "warm_adjoint_ok", True):
        return static.warm_scratch_reserved_bytes
    return None


@torch.no_grad()
def build_point_cache(static, theta, col_weights, sv, *, origination_log_probs=None,
                      origination_probs=None, warm_v=None):
    """Cache each wave adjoint, split likelihood, activity mask, and the E adjoint.

    Returns ``(grad_theta, grad_receiver_weights, cache)``.
    """
    static = _single_static(static)
    cache: dict = {}
    grad_theta, grad_col = vjp_root_to_theta(
        static, sv, None, theta, col_weights, drop_norm=False, cache=cache,
        origination_log_probs=origination_log_probs, origination_probs=origination_probs,
        reserved_scratch_bytes=_warm_reserved_scratch_bytes(static),
        warm_v=warm_v,
        # Fraction-missing is E-only and fixed; thread it so the cached first-order
        # adjoint (wE) + cotangents match the fraction-missing forward. Without this the
        # HVP's point cache would be fraction-missing-wrong (its first-order gradient would
        # not match the production autograd gradient), corrupting every second-order term
        # built on the cache. vjp_root_to_theta already accepts leaf_fm_log (see gauss_newton.py).
        leaf_fm_log=getattr(static, "leaf_fm_log", None),
    )
    return grad_theta, grad_col, cache


def _scatter_accum(acc, family_rows_for_wave, contrib, family_rows):
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


def _head_seed_tangents(
    root_Pi,
    E_star,
    omega,
    t_root,
    dE,
    u_omega,
    dtype,
    accumulator_dtype,
):
    """Origination-weighted head-HVP seed tangents (double-backward on the small NLL aggregation head
    ``h = sum_fam nll(root_rows, E, omega)``; correct by construction and consistent with the kernels'
    seed convention -- the hand-coded uniform ``d_seed`` equals ``ds_root`` here at uniform omega).

    Returns ``(ds_root, ds_E_survival, Hv_omega)``:
      ds_root        = d(dnll/droot) along (t_root, u_omega)  -> root-row HVP seed (with the omega cross-term),
      ds_E_survival  = d(dnll/dE)   along (dE, u_omega)       -> survival E-seed tangent (replaces dg_norm),
      Hv_omega       = d(dnll/domega) along (t_root, dE, u_omega) -> the omega row of ``H u``.
    """
    from gpurec.core.inference.solver import nll_vector_from_root_rows
    from gpurec.core.parameters.extract_parameters import origination_log_probs_from_weights

    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=root_Pi.dtype,
    )
    rP = root_Pi.detach().to(dtype=accumulator_dtype).requires_grad_(True)
    Ev = E_star.detach().to(dtype=accumulator_dtype).requires_grad_(True)
    om = omega.detach().to(device=rP.device, dtype=accumulator_dtype).requires_grad_(True)
    with torch.enable_grad():
        olp = origination_log_probs_from_weights(
            om,
            accumulator_dtype=accumulator_dtype,
        )
        op = torch.exp2(olp)
        nll = nll_vector_from_root_rows(
            rP,
            Ev,
            origination_log_probs=olp,
            origination_probs=op,
            accumulator_dtype=accumulator_dtype,
        ).sum()
        s_root, s_E, s_om = torch.autograd.grad(nll, (rP, Ev, om), create_graph=True)
        inner = (
            (s_root * t_root.to(dtype=accumulator_dtype)).sum()
            + (s_E * dE.to(dtype=accumulator_dtype)).sum()
            + (s_om * u_omega.to(s_om.dtype)).sum()
        )
        ds_root, ds_E, Hv_om = torch.autograd.grad(inner, (rP, Ev, om))
    return ds_root.to(dtype), ds_E.to(dtype), Hv_om.to(dtype)


def make_exact_hvp(static, theta, col_weights, sv, *, cache=None, debug_out=None,
                   tangent_self_iters=None, origination_log_probs=None, origination_probs=None,
                   origination_weights=None):
    """Analytic exact-Hessian HVP over the FULL dataset ``H u = sum_b H_b u``.

    Dispatches on the number of batches:
      * single static / length-1 ``batch_statics`` list -> the single-batch primitive
        (``make_exact_hvp_single``), behaviour-identical to the historical code;
      * ``batch_statics`` list of length > 1 -> the STREAMING multi-batch wrapper
        (``_make_exact_hvp_streaming``), which per ``hvp(u)`` call loops the batches, rebuilds
        each batch's forward saved-intermediates + point cache, accumulates ``H_b u``, and frees
        the batch before the next (memory-bounded; caches are NOT held resident across batches).

    ``sv`` is used only on the single-batch path; the streaming path obtains each ``sv_b`` itself
    via ``forward_solve`` on the length-1 batch (multi-batch ``forward_solve`` returns ``sv=None``).
    """
    if isinstance(static, (list, tuple)) and len(static) > 1:
        return _make_exact_hvp_streaming(
            list(static), theta, col_weights, debug_out=debug_out,
            tangent_self_iters=tangent_self_iters, origination_log_probs=origination_log_probs,
            origination_probs=origination_probs, origination_weights=origination_weights,
        )
    return make_exact_hvp_single(
        static, theta, col_weights, sv, cache=cache, debug_out=debug_out,
        tangent_self_iters=tangent_self_iters, origination_log_probs=origination_log_probs,
        origination_probs=origination_probs, origination_weights=origination_weights,
    )


def make_exact_hvp_single(static, theta, col_weights, sv, *, cache=None, debug_out=None,
                          tangent_self_iters=None, origination_log_probs=None, origination_probs=None,
                          origination_weights=None):
    """Analytic exact-Hessian HVP for a SINGLE batch. Builds the per-point adjoint cache once
    (if not given) and returns ``hvp(u_vec) -> H u`` (flat 3S). Runs in the dtype of ``theta``/``sv``.

    ``tangent_self_iters`` sets the FIXED per-wave self-loop iteration count for the tangent
    forward sweep (sync-free; see ``jvp_root_scores``). Resolution order: this argument, then
    the ``NEWTON_TANGENT_SELF_ITERS`` env var, then ``solver_options.pi_iters`` (matching the
    primal forward truncation). Not hardcoded — change it per run via the env var or the arg.
    """
    import os

    static = _single_static(static)
    so = static.solver_options
    if tangent_self_iters is None:
        _env = os.environ.get("NEWTON_TANGENT_SELF_ITERS")
        if _env:
            tangent_self_iters = int(_env)
        elif getattr(so, "pi_iters", None):
            # Match the primal forward's truncation so the HVP is the exact
            # Hessian of the *truncated* objective.
            tangent_self_iters = int(so.pi_iters)
        else:
            # No primal pi_iters to match (ad-hoc caller). 16 is documented as
            # non-convergent on the representative fixture (+33 NLL, ~2.6x grad
            # bias), so fall back to the validated floor of 64 and warn rather
            # than silently returning a wrong curvature.
            import warnings
            tangent_self_iters = 64
            warnings.warn(
                "hvp_exact: no tangent_self_iters / NEWTON_TANGENT_SELF_ITERS / "
                "solver_options.pi_iters provided; defaulting to 64 (the validated "
                "convergence floor). Pass tangent_self_iters to match your primal "
                "forward truncation.",
                RuntimeWarning, stacklevel=2,
            )
    # How often to run free_cuda_cache_if_tight() in the reverse sweep. It is a blocking
    # cudaMemGetInfo round-trip (~7.9us) and on a large-free GPU never empties the pool, so
    # firing once per wave is ~142 wasted driver calls/HVP. Gate it to every K waves; it stays
    # load-bearing on the big fixtures (memory pressure builds gradually, so checking every K
    # waves still trips the gate in time). 1 = every wave (old behaviour); change per run via env.
    _fc_env = os.environ.get("NEWTON_FREE_CACHE_EVERY")
    free_cache_every = int(_fc_env) if _fc_env else MemoryOptions().free_cache_every
    free_cache_every = max(1, free_cache_every)
    sh, wl = static.species_helpers, static.wave_layout
    S = int(sh["S"])
    # S7: derive the weighted-receiver flag from the base alpha (kill the False hardcodes).
    # uniform base -> uniform_fast path (the legacy theta-only behaviour, bit-for-bit);
    # non-uniform base -> weighted paths LIVE so the backward/cache + receiver-weight
    # cotangent are finite.
    use_receiver_weights = not receiver_weights_are_uniform(col_weights)
    # Fixed fraction-missing leaf boundary (log2, [S]); None => no fraction-missing (fast default).
    # It is E-ONLY: threaded into the E-step primal recomputations + the E-step tangent fixed point
    # so the second-order theta curvature is correct with fraction_missing>0. fraction_missing is a
    # constant input -> no gradient / curvature is accumulated with respect to it.
    leaf_fm_log = getattr(static, "leaf_fm_log", None)
    family_idx = static.rate_family_idx
    species_child1 = sh["sp_child1"]
    species_child2 = sh["sp_child2"]
    species_parent = sh["sp_parent"]
    # Each species' height (0 at a leaf) and the tree's height. The additive valid-receiver and
    # off-subtree sums walk one level per pass and need both; the level count is read off the
    # compact level tables' shape, so no device-to-host copy.
    species_height = sh["sp_height"]
    species_levels = int(sh["compact_level_ptr"].numel()) - 1
    leaf_species_idx = wl["leaf_species_index"].to(device=theta.device, dtype=torch.int32).contiguous()
    root_ids = wl["root_clade_ids"]
    n_fam = int(root_ids.numel())
    dtype = sv["pi_wave"].dtype
    accumulator_dtype = resolve_accumulator_dtype(
        getattr(static, "accumulator_dtype", None),
        fallback=dtype,
    )
    C = int(sv["pi_wave"].shape[0])
    E_star = sv["E"]
    G = int(E_star.shape[0])
    pi_state = sv["pi_state"]
    pi_state.validate(
        sv["pi_wave"], sv["pibar_wave"], sv["pibar_row_max"],
        check_values=False,
    )
    pi_offset = pi_state.pi_offset
    pibar_offset = pi_state.pibar_offset
    # The forward's own per-row decision, so this HVP's adjoint and tangent split the same way it
    # did. None whenever nothing was flagged, which leaves every path as it was.
    wide_row = pi_state.wide_row if pi_state.wide_row_total > 0 else None

    # Turn ON the origination head whenever origination_weights are supplied (even UNIFORM omega=0):
    # the omega curvature at uniform omega is nonzero and is exactly what the joint gate must capture.
    # Derive the log-probs from the weights (shape-generic: [S] specieswise/global, [G,S] genewise) so
    # build_point_cache runs the same weighted head as the tangent sweep -- at uniform omega this is
    # numerically identical to the default uniform forward, so the point cache stays consistent; we are
    # only enabling the head double-backward that produces Hv_omega.
    if origination_weights is not None and origination_log_probs is None:
        from gpurec.core.parameters.extract_parameters import origination_log_probs_from_weights
        origination_log_probs = origination_log_probs_from_weights(
            origination_weights.to(device=theta.device, dtype=accumulator_dtype),
            accumulator_dtype=accumulator_dtype,
        )
        origination_probs = torch.exp2(origination_log_probs)

    if cache is None:
        if getattr(static, "warm_adjoint_ok", True) and static.solver_options.use_hvp_warm_start:
            if static.warm_v is None:
                static.warm_v = {}
            _warm_v = static.warm_v
        else:
            _warm_v = None
        _, _, cache = build_point_cache(static, theta, col_weights, sv,
                                        origination_log_probs=origination_log_probs,
                                        origination_probs=origination_probs,
                                        warm_v=_warm_v)
    acc = cache["accum"]
    wE = cache["e_side"]["wE"]
    # One free-memory reading for this whole reverse sweep. On the cold path (no resident warm
    # cache) ``_warm_reserved_scratch_bytes`` returns None, and the per-wave gate inside
    # ``solve_reconciliation_wave_vjp`` then read free memory itself once per wave -- a blocking
    # cudaMemGetInfo plus two memory_stats() dict builds each time, for a number that cannot move
    # between waves (the scratch is allocated and freed inside one wave). Same fix as the gradient
    # path in gpurec/api/_implicit_grad.py.
    reserved_scratch_bytes = wave_scratch_budget_bytes(
        _warm_reserved_scratch_bytes(static), device=sv["pi_wave"].device
    )

    wave_constants = wave_step_constants(sv, S)
    pibar_row_max = sv["pibar_row_max"]
    receiver_log_probs = sv["receiver_log_probs"]
    as_family_matrix = lambda tensor: as_family_species(tensor, S, G)
    pS_m, pD_m, pL_m = (
        as_family_matrix(sv["log_pS"]),
        as_family_matrix(sv["log_pD"]),
        as_family_matrix(sv["log_pL"]),
    )

    # e-step autograd graph at (E*, P): reused for all linear-in-cotangent transposed products
    E_req = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        E_new_g, E_s1_g, E_s2_g, Ebar_g = e_step_triton_autograd(
            E_req, sv["log_pS"], sv["log_pD"], sv["log_pL"], sv["max_transfer"],
            receiver_log_probs, species_parent, species_child1, species_child2,
            species_height, species_levels, use_receiver_weights=use_receiver_weights,
            leaf_fm_log=leaf_fm_log,
        )

    def jt_E(g_new):
        with torch.enable_grad():
            (out,) = torch.autograd.grad(E_new_g, E_req, grad_outputs=g_new.contiguous(),
                                         retain_graph=True)
        return out

    def aux_T(g_s1, g_s2, g_ebar):
        with torch.enable_grad():
            (out,) = torch.autograd.grad((E_s1_g, E_s2_g, Ebar_g), E_req,
                                         grad_outputs=(g_s1, g_s2, g_ebar), retain_graph=True)
        return out

    # Uniform survival head state, used only by the origination-uniform tangent.
    norm = _survival_from_E(
        E_star.to(dtype=accumulator_dtype),
        keepdim=True,
    )
    fam_factor = 1.0 if G == n_fam else float(n_fam)

    zeros_state = lambda: torch.zeros_like(E_star)

    # e-step head VJP at fixed cotangents (used both for the u-independent primal cotangents
    # below and, per-u, for the tangent cotangents with g_new=dwE)
    def e_bwd_params(g_new, g_ebar):
        with torch.enable_grad():
            pS_r = pS_m.detach().requires_grad_(True)
            pD_r = pD_m.detach().requires_grad_(True)
            pL_r = pL_m.detach().requires_grad_(True)
            max_transfer_req = (
                as_family_matrix(sv["max_transfer"].squeeze(-1))
                .detach()
                .requires_grad_(True)
            )
            receiver_log_probs_req = receiver_log_probs.detach().requires_grad_(True)
            En, _, _, Eb = e_step_triton_autograd(
                E_star.detach(), pS_r, pD_r, pL_r, max_transfer_req,
                receiver_log_probs_req, species_parent, species_child1,
                species_child2, species_height, species_levels,
                use_receiver_weights=use_receiver_weights,
                leaf_fm_log=leaf_fm_log,
            )
            outs = torch.autograd.grad(
                (En, Eb),
                (pS_r, pD_r, pL_r, max_transfer_req, receiver_log_probs_req),
                                       grad_outputs=(g_new, g_ebar), allow_unused=True)
        return tuple(torch.zeros_like(z) if o is None else o
                     for o, z in zip(
                         outs,
                         (pS_m, pD_m, pL_m, pS_m, receiver_log_probs),
                     ))

    # ---- u-INDEPENDENT setup (theta fixed across all CG iterations): primal cotangents and the
    # smooth head graph + first-order grad g1 are built ONCE here, not per hvp(u). The head's
    # forward graph is retained (create_graph) so each hvp(u) only adds phi2 + one backward. ----
    base_p = e_bwd_params(wE, acc["grad_Ebar"])
    # Genewise: log_pS/log_pD are per-family scalars broadcast across all S species, so their e-step
    # cotangent base_p[.] ([G,S]) must be summed to per-family [G,1] BEFORE adding the already-per-family
    # DTS cotangent acc[...] ([G,1]) -- otherwise the [G,1] term broadcasts over S and the head
    # contraction (pS_hp[G,1] * cot_pS).sum(), which sums the species axis, multiplies it by S (the ~Sx
    # HVP bug). Specieswise/global keep the per-species form BIT-FOR-BIT (there S IS the parameter axis;
    # everything is [1,S]). pL has no accumulated term and the cached
    # max-transfer cotangent is genuinely per-species [G,S], so both are
    # correct unchanged in every mode.
    if static.genewise:
        cot_pS = acc["grad_log_pS"] + base_p[0].sum(dim=-1, keepdim=True)
        cot_pD = acc["grad_log_pD"] + base_p[1].sum(dim=-1, keepdim=True)
    else:
        cot_pS = acc["grad_log_pS"] + as_family_param(base_p[0], G, S)
        cot_pD = acc["grad_log_pD"] + as_family_param(base_p[1], G, S)
    cot_pL = base_p[2]
    cot_max_transfer = acc["grad_mc"] + base_p[3]
    cot_receiver_log_probs = acc["grad_col"] + base_p[4]
    theta_req = theta.detach().requires_grad_(True)
    receiver_weights_req = col_weights.detach().requires_grad_(True)
    _head_grad_ctx = torch.enable_grad()
    _head_grad_ctx.__enter__()
    (
        pS_h,
        pD_h,
        pL_h,
        max_transfer_h,
        receiver_log_probs_h,
    ) = extract_parameters_weighted_receivers(
        theta_req, receiver_weights_req, sh,
        specieswise=static.specieswise, genewise=static.genewise,
        uniform_fast=not use_receiver_weights,
        accumulator_dtype=accumulator_dtype,
    )
    pS_hp = as_family_param(pS_h, G, S)
    pD_hp = as_family_param(pD_h, G, S)
    pL_hi = as_family_species(pL_h, S, G)
    max_transfer_hi = as_family_species(
        max_transfer_h.squeeze(-1)
        if max_transfer_h.ndim == pS_h.ndim + 1
        else max_transfer_h,
        S,
        G,
    )
    phi1 = ((pS_hp * cot_pS).sum() + (pD_hp * cot_pD).sum() + (pL_hi * cot_pL).sum()
            + (max_transfer_hi * cot_max_transfer).sum()
            + (receiver_log_probs_h * cot_receiver_log_probs).sum())
    # S8: grad phi1 w.r.t. BOTH theta and receiver weights.
    # g1_receiver_weights carries the receiver-weight row of the first-order
    # head VJP through log-softmax and the valid-receiver normalizer. Autograd
    # returns each partial independently, so g1_theta is bit-for-bit the legacy single-target grad
    # (the u_alpha=0 regression is preserved). create_graph so the second grad below can pass
    # through the softmax/receiver_norm curvature ONCE (no double-count with the kernels' dreceiver_log_probs-linear
    # cotangent). receiver_weights_req always requires grad here, so
    # allow_unused is False.
    g1_theta, g1_receiver_weights = torch.autograd.grad(
        phi1, (theta_req, receiver_weights_req), create_graph=True
    )
    _head_grad_ctx.__exit__(None, None, None)

    # Each probe's tangent forward starts by reducing every split wave's PRIMAL gene-split (DTS)
    # rows out of Pi/Pibar. Those rows depend on theta only, and theta is fixed for the life of
    # this closure, so the three probes of one Hessian were reducing the identical rows three
    # times -- 17 ms of the 502 ms a probe costs on the 200-family Coleman batch. Keep them after
    # the first probe and hand them back on the next two. The cost is one more
    # [batch clades x species] tensor (0.75 GiB on that batch, on top of a 6.5 GiB probe peak),
    # which is the same tensor the gradient path already budgets for, so reuse its build-time
    # memory decision: a card that cannot afford it keeps recomputing, exactly as before.
    primal_gene_split = {} if bool(getattr(static, "forward_gene_split_ok", False)) else None

    def hvp(u_vec, probe_id=None):
        # Joint split: u = [u_theta (theta_numel); u_alpha (S)]. The theta-milestone harness still
        # passes a length-(theta_numel) vector (u_alpha implicitly 0); accept both. theta_shape is
        # explicit (do NOT assume [S,3]).
        u_vec = u_vec.to(theta.dtype)
        theta_numel = theta.numel()
        u = u_vec[:theta_numel].reshape(theta.shape)
        # BASE layout: [u_theta (theta_numel); u_alpha (S)?; u_omega (omega_numel)?]. Alpha BEFORE omega
        # -- the codebase convention (curvature/origination.py consumes z=[theta;alpha;omega]). The omega
        # size is keyed off the origination parameter, so this is uniform across modes: genewise
        # omega_numel=G*S -> u_omega reshapes to [G,S]; specieswise/global omega_numel=S -> [S].
        #   n_tail == 0                 -> theta-only, returns theta_numel BIT-FOR-BIT (_verify_hvp gate);
        #   n_tail == S                 -> [theta; alpha], omega implicitly 0 (_verify_hvp_recv path);
        #   n_tail == S + omega_numel   -> full [theta; alpha; omega].
        omega_numel = origination_weights.numel() if origination_weights is not None else 0
        n_tail = u_vec.numel() - theta_numel
        has_omega = (omega_numel > 0 and n_tail == S + omega_numel)
        joint = n_tail >= S
        if has_omega:
            u_alpha = u_vec[theta_numel:theta_numel + S].contiguous()
            u_omega = u_vec[theta_numel + S:theta_numel + S + omega_numel].reshape(origination_weights.shape)
        else:
            u_alpha = (u_vec[theta_numel:theta_numel + S].contiguous() if joint
                       else torch.zeros(S, device=theta.device, dtype=theta.dtype))
            # u_omega is only consumed when the origination head is active; keep its shape matched to
            # origination_weights ([G,S] genewise / [S] otherwise) so _head_seed_tangents contracts cleanly.
            u_omega = (torch.zeros_like(origination_weights) if origination_weights is not None
                       else torch.zeros(S, device=theta.device, dtype=theta.dtype))
        with torch.no_grad():
            # S3/S7: at a NON-UNIFORM base the tangent forward MUST go through the weighted path
            # (param_jvp_weighted + use_receiver_weights), consistent with the weighted primal fixed
            # point, or the tangent E-adjoint diverges (1e18 / NaN). u_alpha=0 then gives dreceiver_log_probs=0 ->
            # the pure-theta tangent at the non-uniform base. At a UNIFORM base keep alpha=None so
            # the legacy uniform theta-only tangent is reproduced BIT-FOR-BIT (regression guard).
            _alpha = col_weights if use_receiver_weights else None
            _u_alpha = u_alpha if use_receiver_weights else None
            t_root, full = jvp_root_scores(static, theta, u, sv, return_full=True,
                                           keep_d_dts=False, self_iters=tangent_self_iters,
                                           primal_gene_split=primal_gene_split,
                                           alpha=_alpha, u_alpha=_u_alpha,
                                           leaf_fm_log=leaf_fm_log)
            tangent_constants = full["tangent_constants"]
            dPi, dPibar = full["dPi"], full["dPibar"]
            dpS_m, dpD_m, dpL_m = (
                as_family_matrix(full["dlog_pS"]),
                as_family_matrix(full["dlog_pD"]),
                as_family_matrix(full["dlog_pL"]),
            )
            d_max_transfer_m = as_family_matrix(full["d_max_transfer"].squeeze(-1))
            # S4: the receiver-weight tangent seed = softmax-Jacobian . u_alpha,
            # exposed by S3's
            # weighted jvp. At a uniform base the weighted path is off and there is no dreceiver_log_probs key
            # -> dreceiver_log_probs=None (the e_step_backward_so dreceiver_log_probs slot then zero-fills; bit-for-bit legacy).
            dreceiver_log_probs = full.get("dreceiver_log_probs") if use_receiver_weights else None
            dE = full["d_extinction"]
            dEbar_e = full["d_extinction_complement"]
            dE_s1 = full["d_extinction_child1"]
            dE_s2 = full["d_extinction_child2"]

            # tangent of the loss seed -q on root rows. Uniform origination: hand-coded (bit-for-bit
            # legacy). Weighted origination: the autograd head gives the weighted root-seed tangent
            # (incl. the omega cross-term), the survival E-seed tangent, and the omega row of H u.
            root_Pi = sv["pi_wave"].index_select(0, root_ids)
            Hv_omega = ds_E_surv = None
            if origination_log_probs is None:
                root_head = root_Pi.to(dtype=accumulator_dtype)
                t_root_head = t_root.to(dtype=accumulator_dtype)
                q = _safe_exp2_ratio(
                    root_head,
                    _logsumexp2(root_head, dim=-1, keepdim=True),
                )
                d_seed = -_LN2 * q * (
                    t_root_head - (q * t_root_head).sum(dim=-1, keepdim=True)
                )
            else:
                d_seed, ds_E_surv, Hv_omega = _head_seed_tangents(
                    root_Pi,
                    E_star,
                    origination_weights,
                    t_root,
                    dE,
                    u_omega,
                    dtype,
                    accumulator_dtype,
                )

            d_rhs = torch.zeros((C, S), device=theta.device, dtype=dtype)
            d_rhs.index_copy_(0, root_ids, d_seed.to(dtype))

            d_grad_log_pD = torch.zeros_like(acc["grad_log_pD"])
            d_grad_log_pS = torch.zeros_like(acc["grad_log_pS"])
            d_grad_extinction, d_grad_extinction_complement, d_grad_extinction_child1, d_grad_extinction_child2 = (zeros_state() for _ in range(4))
            d_grad_max_transfer = torch.zeros_like(acc["grad_mc"])
            d_grad_receiver_log_probs = torch.zeros((S,), device=theta.device, dtype=dtype)
            # The receiver-weight ROW of ``H u`` is only returned on the joint contract. The
            # receiver log-probabilities come from the receiver weights alone
            # (extract_parameters_weighted_receivers: receiver_log_probs_from_weights(receiver_weights)),
            # never from theta, so on the theta-only contract the whole receiver cotangent is
            # multiplied by something with no theta in it and contributes EXACTLY zero to
            # out_theta. Handing the two wave-backward entry points ``None`` instead of a buffer
            # is how they are told nobody wants it (same switch the gradient path's
            # ``need_receiver_grad`` throws): it drops the per-wave receiver-log-probability VJP
            # kernel and the receiver half of the transfer-complement kernel, 12 ms of the 502 ms
            # a probe costs on the 200-family Coleman batch. ``d_grad_receiver_log_probs`` stays
            # allocated because the second-order kernels below need a pointer to aim at when the
            # receiver weights are non-uniform; on this branch nothing reads it back.

            from gpurec.solver.value_and_grad import free_cuda_cache_if_tight

            for _wi, wave in enumerate(cache["waves"]):  # already reverse order
                if _wi % free_cache_every == 0:
                    free_cuda_cache_if_tight()
                ws, W = wave["ws"], wave["W"]
                _tangent_warm = (
                    probe_id is not None
                    and getattr(static, "warm_adjoint_ok", True)
                    and static.solver_options.use_hvp_warm_start
                )
                _probe_cache = None
                _init_v = None
                if _tangent_warm:
                    if static.warm_v_tangent is None:
                        static.warm_v_tangent = {}
                    _probe_cache = static.warm_v_tangent.setdefault(probe_id, {})
                    _init_v = _probe_cache.get(ws)
                meta = wave["meta"]
                v_k = wave["v"]
                gene_split_log_likelihood = wave["dts_r"]
                gene_split_offset = wave["dts_offset"]
                # Recompute the split-likelihood tangent from the cached, pruned
                # split likelihood. Storing every tangent would cost another
                # Pi-sized buffer; one tangent launch per wave is cheap.
                if gene_split_log_likelihood is not None:
                    from gpurec.core.kernels.dts_tangent import compute_dts_tangent
                    d_gene_split_log_likelihood = compute_dts_tangent(
                        sv["pi_wave"], sv["pibar_wave"], dPi, dPibar, meta["sl"], meta["sr"],
                        species_child1, species_child2, W, meta["reduce_idx"],
                        wave_constants["duplication_log_probability_param"],
                        wave_constants["speciation_log_probability_param"],
                        tangent_constants["d_duplication_log_probability_param"],
                        tangent_constants["d_speciation_log_probability_param"],
                        gene_split_log_likelihood, family_idx,
                        log_split_probs=_select_log_split_probs(meta, sv["pi_wave"].dtype), family_offset=ws,
                        pi_offset=pi_offset,
                        pibar_offset=pibar_offset,
                        gene_split_offset=gene_split_offset,
                    )
                else:
                    d_gene_split_log_likelihood = None
                has_leaf = wave["has_leaf_term"]
                # (a) second-order contraction at fixed v_k; d_rhs folds the wave's rhs cotangent
                # into d_Av so it IS the solve seed (= d_rhs[ws:ws+W] + d_Av), no host add
                (
                    d_Av,
                    contraction_duplication_loss_vjp,
                    contraction_transfer_loss_vjp,
                    contraction_transfer_vjp,
                    contraction_speciation_leaf_vjp,
                    contraction_speciation_child1_vjp,
                    contraction_speciation_child2_vjp,
                    wave_d_grad_receiver_log_probs,
                ) = wave_backward_so(
                    sv["pi_wave"], dPi, sv["pibar_wave"], dPibar, v_k, ws, W, S,
                    pibar_row_max,
                    wave_constants["duplication_loss_const"],
                    tangent_constants["d_duplication_loss_const"],
                    wave_constants["extinction_complement"],
                    tangent_constants["d_extinction_complement"],
                    wave_constants["extinction"], tangent_constants["d_extinction"],
                    wave_constants["speciation_child1_const"],
                    tangent_constants["d_speciation_child1_const"],
                    wave_constants["speciation_child2_const"],
                    tangent_constants["d_speciation_child2_const"],
                    receiver_log_probs, species_child1, species_child2,
                    species_parent,
                    gene_split_log_likelihood, d_gene_split_log_likelihood,
                    species_height=species_height, species_levels=species_levels,
                    leaf_species_idx=leaf_species_idx,
                    leaf_logp=wave_constants["leaf_log_probability"],
                    d_leaf_logp=tangent_constants["d_leaf_log_probability"],
                    family_idx=family_idx, has_leaf_term=has_leaf, use_receiver_weights=use_receiver_weights,
                    d_rhs=d_rhs, dreceiver_log_probs=dreceiver_log_probs,
                    pi_offset=pi_offset,
                    pibar_offset=pibar_offset,
                    gene_split_offset=gene_split_offset,
                )
                # S5: accumulate the wave-SO receiver-log-probability cotangent
                # (the tangent of the wave self-loop receiver gradient). It is
                # zero when receiver weights are disabled.
                if use_receiver_weights:
                    d_grad_receiver_log_probs = d_grad_receiver_log_probs + wave_d_grad_receiver_log_probs
                # Re-read each wave: the line above REBINDS the accumulator to a fresh tensor, so a
                # binding taken once before the loop would hand the kernels a stale buffer.
                wave_grad_receiver_log_probs = d_grad_receiver_log_probs if joint else None
                # (b) tangent-adjoint solve with the SAME operator and cached mask
                seed = d_Av
                (
                    dv,
                    linear_duplication_loss_vjp,
                    linear_transfer_loss_vjp,
                    linear_transfer_vjp,
                    linear_speciation_leaf_vjp,
                    linear_speciation_child1_vjp,
                    linear_speciation_child2_vjp,
                ) = solve_reconciliation_wave_vjp(
                    sv["pi_wave"], sv["pibar_wave"], ws, W, S,
                    gene_split_log_likelihood, seed, wave_constants["max_transfer"],
                    wave_constants["duplication_loss_const"],
                    wave_constants["extinction_complement"], wave_constants["extinction"],
                    wave_constants["speciation_child1_const"],
                    wave_constants["speciation_child2_const"], receiver_log_probs,
                    species_child1, species_child2, None, neumann_terms=int(so.neumann_terms),
                    neumann_term_tol=float(so.neumann_term_tol),
                    adjoint_self_loop=so.adjoint_self_loop,
                    wide_row=wide_row,
                    leaf_species_idx=leaf_species_idx,
                    leaf_logp=wave_constants["leaf_log_probability"],
                    has_leaf_term=has_leaf,
                    active_mask=wave["active_mask"], species_parent=species_parent,
                    species_subtree_start=sh["sp_subtree_start"],
                    species_subtree_end=sh["sp_subtree_end"],
                    pibar_row_max=pibar_row_max, family_idx=family_idx,
                    family_indexed_consts=True,
                    compact_level_ptr=sh["compact_level_ptr"],
                    compact_level_parents=sh["compact_level_parents"],
                    compact_level_child1=sh["compact_level_child1"],
                    compact_level_child2=sh["compact_level_child2"],
                    grad_receiver_log_probs=wave_grad_receiver_log_probs,
                    use_receiver_weights=use_receiver_weights,
                    initial_v=_init_v,
                    return_last_increment=False,
                    reserved_scratch_bytes=reserved_scratch_bytes,
                    pi_offset=pi_offset,
                    pibar_offset=pibar_offset,
                    gene_split_offset=gene_split_offset,
                )
                if _tangent_warm:
                    _mask = wave.get("active_mask")
                    if _mask is not None:
                        _row_active = _mask.reshape(_mask.shape[0], -1).ne(0).any(dim=1)
                        _cached_v = torch.where(
                            _row_active.unsqueeze(-1), dv, torch.zeros((), dtype=dv.dtype, device=dv.device)
                        )
                    else:
                        _cached_v = dv
                    _probe_cache[ws] = _cached_v.detach()
                duplication_loss_event_vjp = (
                    contraction_duplication_loss_vjp + linear_duplication_loss_vjp
                )
                transfer_loss_event_vjp = contraction_transfer_loss_vjp + linear_transfer_loss_vjp
                transfer_event_vjp = contraction_transfer_vjp + linear_transfer_vjp
                speciation_leaf_event_vjp = contraction_speciation_leaf_vjp + linear_speciation_leaf_vjp
                speciation_child1_event_vjp = (
                    contraction_speciation_child1_vjp + linear_speciation_child1_vjp
                )
                speciation_child2_event_vjp = (
                    contraction_speciation_child2_vjp + linear_speciation_child2_vjp
                )
                if debug_out is not None:
                    debug_out.setdefault("wave_trace", []).append(
                        (ws, float(d_Av.abs().max()), float(dv.abs().max()),
                         float(d_rhs.abs().max())))
                rows_i = family_idx[ws:ws + W]
                _scatter_accum(d_grad_log_pD, rows_i, duplication_loss_event_vjp, G)
                _scatter_accum(d_grad_log_pS, rows_i, speciation_leaf_event_vjp, G)
                _scatter_accum(
                    d_grad_extinction,
                    rows_i,
                    duplication_loss_event_vjp + transfer_event_vjp,
                    G,
                )
                _scatter_accum(d_grad_extinction_complement, rows_i, transfer_loss_event_vjp, G)
                _scatter_accum(d_grad_extinction_child1, rows_i, speciation_child2_event_vjp, G)
                _scatter_accum(d_grad_extinction_child2, rows_i, speciation_child1_event_vjp, G)
                _scatter_accum(d_grad_max_transfer, rows_i, transfer_event_vjp, G)
                if gene_split_log_likelihood is not None:
                    # C^T dv via the frozen kernels (linear in v)
                    (
                        donor_adjoint,
                        active_donor_side,
                        _duplication_parameter_vjp,
                        _speciation_parameter_vjp,
                    ) = accumulate_gene_split_event_vjp(
                        sv["pi_wave"], sv["pibar_wave"], dv, ws, meta["sl"], meta["sr"],
                        meta["reduce_idx"],
                        _select_log_split_probs(meta, dtype),
                        wave_constants["duplication_log_probability_param"],
                        wave_constants["speciation_log_probability_param"],
                        species_child1, species_child2, d_rhs, S,
                        active_mask=wave["active_mask"], merge_s_term=True,
                        grad_log_pD=d_grad_log_pD,
                        grad_log_pS=d_grad_log_pS,
                        grad_max_transfer=d_grad_max_transfer,
                        accum_param_reductions=True,
                        accum_max_transfer_reduction=True,
                        output_donor_adjoint=True,
                        output_active_donor_sides=True,
                        pibar_side_threshold=so.pibar_side_threshold,
                        max_transfer=wave_constants["max_transfer"],
                        pibar_row_max=pibar_row_max,
                        grad_max_transfer_two_stage=bool(
                            d_grad_max_transfer.ndim == 2 and int(d_grad_max_transfer.shape[0]) == 1
                        ),
                        grad_max_transfer_two_stage_tile_splits=128,
                        skip_inactive_pibar_output_zero=True,
                        family_idx=family_idx,
                        pi_offset=pi_offset,
                        pibar_offset=pibar_offset,
                    )
                    accumulate_transfer_complement_vjp_from_donor_adjoint(
                        sv["pi_wave"],
                        receiver_log_probs,
                        donor_adjoint,
                        meta["sl"],
                        meta["sr"],
                        d_rhs,
                        S,
                        species_parent,
                        active_mask=wave["active_mask"], reduce_idx=meta["reduce_idx"],
                        pibar_row_max=pibar_row_max,
                        skip_zero_donor_sides=True,
                        active_donor_side=active_donor_side,
                        compact_level_ptr=sh["compact_level_ptr"],
                        compact_level_parents=sh["compact_level_parents"],
                        compact_level_child1=sh["compact_level_child1"],
                        compact_level_child2=sh["compact_level_child2"],
                        grad_receiver_log_probs=wave_grad_receiver_log_probs,
                        use_receiver_weights=use_receiver_weights,
                        side_active_threshold=so.pibar_side_threshold,
                    )
                    # d(C^T) v_k contraction at fixed v_k
                    dts_backward_so(
                        sv["pi_wave"], dPi, sv["pibar_wave"], dPibar, v_k, ws, meta, S,
                        wave_constants["duplication_log_probability_param"],
                        wave_constants["speciation_log_probability_param"],
                        tangent_constants["d_duplication_log_probability_param"],
                        tangent_constants["d_speciation_log_probability_param"],
                        wave_constants["max_transfer"], tangent_constants["d_max_transfer"],
                        receiver_log_probs, species_child1, species_child2,
                        pibar_row_max, family_idx,
                        d_rhs, d_grad_log_pD, d_grad_log_pS, d_grad_max_transfer, d_grad_receiver_log_probs,
                        species_parent=species_parent,
                        compact_level_ptr=sh["compact_level_ptr"],
                        compact_level_parents=sh["compact_level_parents"],
                        compact_level_child1=sh["compact_level_child1"],
                        compact_level_child2=sh["compact_level_child2"],
                        use_receiver_weights=use_receiver_weights, dreceiver_log_probs=dreceiver_log_probs,
                        pi_offset=pi_offset,
                        pibar_offset=pibar_offset,
                    )

            # ---- E-side ---- (the big tangent buffers are no longer needed)
            del dPi, dPibar
            full.clear()
            free_cuda_cache_if_tight()
            x_args = (E_star.contiguous(), E_star.contiguous(), sv["E_s1"], sv["E_s2"],
                      sv["Ebar"], pS_m, pD_m, pL_m, receiver_log_probs.contiguous(),
                      species_parent, species_child1, species_child2)
            # S4: the 9th dx slot is dreceiver_log_probs (e_step_backward_so signature ...,dlog_pL, dreceiver_log_probs). Was
            # formerly hardcoded None; thread the S3 seed so the E-step SO contraction
            # carries the receiver-weight dependence. None at a uniform base preserves
            # the legacy uniform path exactly.
            dx = (dE, dE, dE_s1, dE_s2, dEbar_e, dpS_m, dpD_m, dpL_m, dreceiver_log_probs)
            zero_g = zeros_state()
            # tangent of aux_to_e: linear part + contraction + norm-term closed form
            aux_lin = aux_T(d_grad_extinction_child1, d_grad_extinction_child2, d_grad_extinction_complement)
            so_aux = e_step_backward_so(
                *x_args, zero_g, acc["grad_Ebar"], *dx,
                species_height=species_height, species_levels=species_levels,
                use_receiver_weights=use_receiver_weights,
            )
            if origination_log_probs is None:
                e2E = torch.exp2(E_star.to(dtype=accumulator_dtype))
                dE_head = dE.to(dtype=accumulator_dtype)
                dnorm = -_LN2 * (e2E * dE_head).mean(dim=-1, keepdim=True)
                dg_norm = fam_factor * (
                    -_LN2 * e2E * dE_head / (S * norm)
                    + e2E * dnorm / (S * norm * norm)
                )
                dg_norm = dg_norm.to(dtype=dtype)
            else:
                dg_norm = ds_E_surv  # weighted + omega-coupled survival tangent (autograd head)
            dq_E = d_grad_extinction + aux_lin + so_aux[0] + dg_norm
            # tangent E-adjoint solve: same operator, new rhs
            so_w = e_step_backward_so(*x_args, wE, zero_g, *dx,
                                      species_height=species_height, species_levels=species_levels,
                                      use_receiver_weights=use_receiver_weights)
            rhs_E = (dq_E + so_w[0]).reshape(-1)
            E_shape = E_star.shape

            def AG_flat(w_flat):
                gE = jt_E(w_flat.view(E_shape))
                return (w_flat.view(E_shape) - gE).reshape(-1)

            # Same linear E-adjoint operator ``(I - J)``, new rhs. Neumann series (see
            # _neumann_e_adjoint): no orthogonalization, so no fp32 residual floor at large
            # species counts.
            dwE = _neumann_e_adjoint(AG_flat, rhs_E, max_iter=so.e_adjoint_max_iter,
                                      tol=so.e_adjoint_tol).view(E_shape)
            if debug_out is not None:
                debug_out.update(
                    d_grad_extinction=d_grad_extinction.clone(), d_grad_log_pD=d_grad_log_pD.clone(), d_grad_log_pS=d_grad_log_pS.clone(),
                    d_grad_max_transfer=d_grad_max_transfer.clone(), d_grad_extinction_complement=d_grad_extinction_complement.clone(), d_grad_extinction_child1=d_grad_extinction_child1.clone(),
                    d_grad_extinction_child2=d_grad_extinction_child2.clone(), dq_E=dq_E.clone(), dwE=dwE.clone(),
                    d_grad_receiver_log_probs=d_grad_receiver_log_probs.clone(),
                )

            # tangent param-cotangents from the e-step head: linear (tangent cotangents,
            # g_new=dwE) + contraction at fixed cotangents (g_new=wE, g_ebar=grad_Ebar_acc).
            # e_bwd_params and the primal cotangents/head graph are hoisted (u-independent).
            lin_p = e_bwd_params(dwE, d_grad_extinction_complement)
            so_p = e_step_backward_so(*x_args, wE, acc["grad_Ebar"], *dx,
                                      species_height=species_height, species_levels=species_levels,
                                      use_receiver_weights=use_receiver_weights)
            # so_p outputs: (d_grad_E, d_grad_pS, d_grad_pD, d_grad_pL, d_grad_mc, d_grad_receiver_log_probs)

            if static.genewise:  # per-family log_pS/pD: sum the per-species tangent cotangents (see phi1 above)
                d_cot_pS = d_grad_log_pS + (lin_p[0] + so_p[1]).sum(dim=-1, keepdim=True)
                d_cot_pD = d_grad_log_pD + (lin_p[1] + so_p[2]).sum(dim=-1, keepdim=True)
            else:
                d_cot_pS = d_grad_log_pS + as_family_param(lin_p[0] + so_p[1], G, S)
                d_cot_pD = d_grad_log_pD + as_family_param(lin_p[1] + so_p[2], G, S)
            d_cot_pL = lin_p[2] + so_p[3]
            d_cot_max_transfer = d_grad_max_transfer + lin_p[3] + so_p[4]
            # Only assembled on the joint contract: on the theta-only contract the wave sweep was
            # told not to fill ``d_grad_receiver_log_probs`` (see above), and the head term below
            # that would consume it is an exact zero for out_theta anyway.
            d_cot_receiver_log_probs = (
                (d_grad_receiver_log_probs + lin_p[4] + so_p[5]) if joint else None
            )

        # ---- smooth parameter head (autograd; forward graph + g1 hoisted, retained) ----
        with torch.enable_grad():
            phi2 = (
                (pS_hp * d_cot_pS).sum()
                + (pD_hp * d_cot_pD).sum()
                + (pL_hi * d_cot_pL).sum()
                + (max_transfer_hi * d_cot_max_transfer).sum()
            )
            if joint:
                # ``receiver_log_probs_h`` is a function of the receiver weights only, so this term
                # carries the receiver row of ``H u`` and adds exactly nothing to the theta row.
                phi2 = phi2 + (receiver_log_probs_h * d_cot_receiver_log_probs).sum()
            # S8: grad w.r.t. BOTH theta and receiver weights of
            # (g1_theta * u_theta).sum()
            # + (g1_receiver_weights * u_alpha).sum() + phi2.
            # - out_theta = H_tt u_theta + H_ta u_alpha (theta row of H u).
            # - out_receiver_weights = H_at u_theta + H_aa u_alpha.
            # The alpha-alpha softmax curvature + receiver_norm->max_transfer curvature are captured
            # HERE ONCE (the receiver log-probabilities and max-transfer terms
            # are differentiated twice through g1's graph); the kernels carry
            # only the dreceiver_log_probs-linear cotangent, so there is no
            # double count. With u_alpha=0 the receiver-weight head term
            # vanishes and out_theta is bit-for-bit the legacy grad (regression guard). retain_graph
            # so the hoisted forward graph + g1 survive for the next hvp(u) call.
            if not joint:
                # legacy theta-only contract: EXACT legacy graph -- grad of (g1_theta*u).sum()+phi2
                # w.r.t. theta_req ONLY (no receiver-weight head term or grad target). g1_theta is
                # autograd's independent partial of phi1, so it equals the old single-target g1
                # bit-for-bit -> out reproduces the theta-only HVP BIT-FOR-BIT (the _verify_hvp
                # uniform gate + the non-uniform milestone depend on this).
                (out_theta,) = torch.autograd.grad((g1_theta * u).sum() + phi2, theta_req,
                                                   retain_graph=True)
                return out_theta.reshape(-1)
            # Joint contract: differentiate both theta and receiver-weight rows.
            head = (
                (g1_theta * u).sum()
                + (g1_receiver_weights * u_alpha).sum()
                + phi2
            )
            out_theta, out_receiver_weights = torch.autograd.grad(
                head, (theta_req, receiver_weights_req), retain_graph=True
            )
        if has_omega:
            # Full [theta; alpha; omega] contract (BASE order, alpha BEFORE omega -- matches
            # curvature/origination.py's z=[theta;alpha;omega]): the omega row is the head omega-Hessian .
            # (t_root, dE, u_omega) from _head_seed_tangents (omega is head-only -> no kernel/adjoint
            # term). Hv_omega is [G,S] genewise / [S] otherwise; flatten in place.
            return torch.cat([
                out_theta.reshape(-1),
                out_receiver_weights.reshape(-1),
                Hv_omega.reshape(-1),
            ])
        return torch.cat([
            out_theta.reshape(-1), out_receiver_weights.reshape(-1)
        ])

    return hvp


def _make_exact_hvp_streaming(batch_statics, theta, col_weights, *, debug_out=None,
                              tangent_self_iters=None, origination_log_probs=None,
                              origination_probs=None, origination_weights=None):
    """STREAMING multi-batch exact HVP: ``H u = sum_b H_b u``.

    The total NLL is a sum over gene families and the batches partition the families into DISJOINT
    subsets, so ``H = d2L/dtheta2 = sum_b H_b`` exactly (specieswise theta / global theta are SHARED
    across families; genewise theta is per-family, so each batch's Hessian only touches its own
    families -- the E-survival term is weighted by each batch's family count, and summing recovers
    the full-dataset curvature). Nothing couples across batches except the shared read-only ``theta``
    / ``col_weights`` and the summed output.

    Per ``hvp(u)`` call this loops the batches, rebuilds each batch's forward saved-intermediates
    ``sv_b`` (via ``forward_solve`` on the length-1 batch -- multi-batch ``forward_solve`` returns
    ``None``) and its point cache, evaluates ``H_b u``, accumulates, and frees the batch cache
    before the next. This rebuilds per-batch caches each CG/Lanczos iteration (no cross-iteration
    amortization) -- the correct memory/robustness tradeoff at scale, where holding every batch's
    ~GB saved-intermediates + cache resident would OOM.

    Genewise: ``theta`` is the FULL ``[G,3]``; ``forward_solve`` re-selects each batch's families
    internally, while the per-batch primitive consumes the batch-local ``theta_b`` (gathered by
    ``static.family_index_tensor``). The batch-local ``H_b u_b`` is scattered back into the full
    theta output by ``index_add_`` on the disjoint family rows. Specieswise/global: ``theta`` is
    shared, so each ``H_b u`` is simply summed.
    """
    from gpurec.solver.value_and_grad import forward_solve, free_cuda_cache_if_tight

    genewise = bool(batch_statics[0].genewise)
    dev = theta.device
    dtype = theta.dtype
    theta_numel = int(theta.numel())
    S = int(col_weights.numel())
    per_family_orig = (origination_weights is not None and origination_weights.ndim == 2)

    def hvp(u_vec, probe_id=None):
        u_vec = u_vec.to(device=dev, dtype=dtype)
        n_tail = int(u_vec.numel()) - theta_numel
        if not genewise:
            # SHARED theta (specieswise/global): H u = sum_b H_b u. theta / col_weights / any
            # shared tail (alpha [S], omega [S]) are identical across batches, so a straight sum
            # of the per-batch primitive outputs is the full-dataset H u.
            out = None
            for static_b in batch_statics:
                _loss, sv_b = forward_solve([static_b], theta, col_weights)
                hvp_b = make_exact_hvp_single(
                    static_b, theta, col_weights, sv_b, tangent_self_iters=tangent_self_iters,
                    origination_log_probs=origination_log_probs,
                    origination_probs=origination_probs, origination_weights=origination_weights,
                )
                contrib = hvp_b(u_vec, probe_id=probe_id)
                out = contrib if out is None else out + contrib
                del hvp_b, sv_b
                free_cuda_cache_if_tight()
            return out

        # GENEWISE: per-family (disjoint) theta blocks. Gather each batch's family rows, apply the
        # batch-local primitive, scatter back. Supports theta-only u (the ridge/Newton path) and the
        # joint [theta (3G); alpha (S); omega (G*S)] layout (alpha shared, omega per-family).
        u_theta = u_vec[:theta_numel].reshape(theta.shape)
        omega_numel = int(origination_weights.numel()) if origination_weights is not None else 0
        has_alpha = n_tail >= S
        has_omega = (omega_numel > 0 and n_tail == S + omega_numel)
        u_alpha = u_vec[theta_numel:theta_numel + S].contiguous() if has_alpha else None
        u_omega_full = (u_vec[theta_numel + S:theta_numel + S + omega_numel].reshape(origination_weights.shape)
                        if has_omega else None)

        out_theta = torch.zeros(theta.shape, device=dev, dtype=dtype)
        out_alpha = torch.zeros(S, device=dev, dtype=dtype) if has_alpha else None
        out_omega = torch.zeros_like(origination_weights) if has_omega else None
        for static_b in batch_statics:
            fam_b = static_b.family_index_tensor.to(device=dev)
            G_b = int(fam_b.numel())
            theta_b = theta.index_select(0, fam_b).contiguous()  # [G_b, 3]
            orig_b = (origination_weights.index_select(0, fam_b).contiguous() if per_family_orig
                      else origination_weights)
            # FULL theta to forward_solve (re-selects the batch families internally); batch-local
            # theta_b to the primitive (consumes it directly). Both see the identical theta_b.
            _loss, sv_b = forward_solve([static_b], theta, col_weights)
            hvp_b = make_exact_hvp_single(
                static_b, theta_b, col_weights, sv_b, tangent_self_iters=tangent_self_iters,
                origination_weights=orig_b,
            )
            # assemble batch-local u_b in the primitive's [theta_b; alpha; omega_b] layout
            u_theta_b = u_theta.index_select(0, fam_b).reshape(-1)
            parts = [u_theta_b]
            if has_alpha:
                parts.append(u_alpha)
            if has_omega:
                parts.append(u_omega_full.index_select(0, fam_b).reshape(-1))
            o_b = hvp_b(
                torch.cat(parts) if len(parts) > 1 else u_theta_b, probe_id=probe_id
            ).to(dtype=dtype)
            out_theta.index_add_(0, fam_b, o_b[:3 * G_b].reshape(G_b, 3))
            if has_alpha:
                out_alpha = out_alpha + o_b[3 * G_b:3 * G_b + S]
            if has_omega:
                out_omega.index_add_(0, fam_b, o_b[3 * G_b + S:3 * G_b + S + G_b * S].reshape(G_b, S))
            del hvp_b, sv_b
            free_cuda_cache_if_tight()

        result = [out_theta.reshape(-1)]
        if has_alpha:
            result.append(out_alpha.reshape(-1))
        if has_omega:
            result.append(out_omega.reshape(-1))
        return result[0] if len(result) == 1 else torch.cat(result)

    return hvp
