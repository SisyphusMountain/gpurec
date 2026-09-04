"""Full forward-mode tangent (Jvp) of the root scores w.r.t. theta.

``jvp_root_scores(static, theta, v, sv)`` returns ``d(Pi_root)/dtheta . v`` by threading tangents
through the whole forward solve, mirroring ``solve_e_pi`` + ``pi_wave_forward``:

  1. parameter tangent ``dparams = d(extract_parameters)/dtheta . v`` (forward-mode autodiff);
  2. E-step tangent fixed point (``e_tangent_fixed_point``);
  3. Pi-wave tangent: per wave (topological order), the cross-wave ``dts`` tangent then the
     self-loop tangent solved to convergence (the same true fixed point the adjoint differentiates).

This is the ``J`` of the Gauss-Newton operator ``M = J^T B J``; the matching ``J^T`` reuses the
existing backward (see ``gauss_newton.py``).
"""

from __future__ import annotations

import warnings

import torch
from torch.func import jvp

from gpurec.config import dtype_rel_tol_default
from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.core.parameters.extract_parameters import (
    as_family_param, as_family_species, extract_parameters_uniform,
    extract_parameters_weighted_receivers,
)
from gpurec.core.kernels.pi_forward import (
    _select_log_split_probs,
    compute_dts_forward,
)
from gpurec.core.kernels.dts_tangent import compute_dts_tangent
from gpurec.core.kernels.e_step_tangent import e_tangent_fixed_point
from gpurec.core.kernels.wave_tangent import (
    compute_wave_step_tangent, compute_wave_step_tangent_selfloop,
)

DEFAULT_SELF_MAX_ITER = 200


def param_jvp_uniform(static, theta, v):
    """Forward-mode tangent of extract_parameters_uniform along v (use_receiver_weights=False path)."""
    unnorm_row_max = static.species_helpers["unnorm_row_max"].to(device=theta.device, dtype=theta.dtype)

    def f(th):
        return extract_parameters_uniform(
            th,
            unnorm_row_max,
            specieswise=static.specieswise,
            genewise=static.genewise,
            accumulator_dtype=getattr(static, "accumulator_dtype", None),
        )

    primals, tangents = jvp(f, (theta,), (v,))
    # (log_pS, log_pD, log_pL, max_transfer)
    return tangents


def param_jvp_weighted(static, theta, alpha, u_theta, u_alpha):
    """Forward-mode tangent of ``extract_parameters_weighted_receivers`` along (u_theta, u_alpha).

    Returns ``(dlog_pS, dlog_pD, dlog_pL, dmax_transfer, dreceiver_log_probs)`` where the new alpha coupling lives
    in TWO outputs that ``param_jvp_uniform`` lacked:

    * ``dmax_transfer`` now carries the ``alpha -> receiver_log_probs -> receiver_valid_log_normalizer
      (receiver_norm) -> max_transfer`` coupling (the DOMINANT alpha->rate sensitivity; the uniform
      extractor dropped ``receiver_norm`` entirely);
    * ``dreceiver_log_probs`` is the softmax-Jacobian applied to ``u_alpha``,
      ``(diag(w) - w w^T) u_alpha / ln2`` in log2-space — autograd computes it, do NOT hand-roll.
      This is the alpha tangent SEED threaded into the e-step / wave / dts tangents.

    ``uniform_fast=True`` keeps the ``- log2(S)`` shift so the tangent's ``max_transfer`` matches the
    primal forward's; the JVP differentiates through ``receiver_norm`` regardless.
    """
    sh = static.species_helpers
    al = alpha.to(device=theta.device, dtype=theta.dtype)
    ua = u_alpha.to(device=theta.device, dtype=theta.dtype)

    def f(th, a):
        return extract_parameters_weighted_receivers(
            th, a, sh, specieswise=static.specieswise, genewise=static.genewise,
            uniform_fast=True,
            accumulator_dtype=getattr(static, "accumulator_dtype", None),
        )

    primals, tangents = jvp(f, (theta, al), (u_theta, ua))
    # (dlog_pS, dlog_pD, dlog_pL, dmax_transfer, dreceiver_log_probs)
    return tangents


def wave_step_constants(sv, S):
    """Base per-family wave-step constants (mirrors pi_wave_forward)."""
    family_rows = int(sv["E"].shape[0])
    extinction = as_family_species(sv["E"], S, family_rows)
    extinction_complement = as_family_species(sv["Ebar"], S, family_rows)
    extinction_child1 = as_family_species(sv["E_s1"], S, family_rows)
    extinction_child2 = as_family_species(sv["E_s2"], S, family_rows)
    max_transfer = as_family_species(
        sv["max_transfer"].squeeze(-1), S, family_rows
    )
    duplication_log_probability = as_family_species(
        sv["log_pD"], S, family_rows
    )
    speciation_log_probability = as_family_species(
        sv["log_pS"], S, family_rows
    )
    return {
        "duplication_loss_const": 1.0
        + duplication_log_probability
        + extinction,
        "extinction_complement": extinction_complement,
        "extinction": extinction,
        "speciation_child1_const": speciation_log_probability
        + extinction_child2,
        "speciation_child2_const": speciation_log_probability
        + extinction_child1,
        "max_transfer": max_transfer,
        "leaf_log_probability": speciation_log_probability,
        "duplication_log_probability_param": as_family_param(
            sv["log_pD"], family_rows, S
        ),
        "speciation_log_probability_param": as_family_param(
            sv["log_pS"], family_rows, S
        ),
    }


def _default_tol(dtype):
    return dtype_rel_tol_default(dtype)


def _wave_tangent_constants(static, theta, v, sv, S, e_tol, raw_out=None,
                            alpha=None, u_alpha=None, use_receiver_weights=False,
                            leaf_fm_log=None):
    """E + parameter tangents assembled into the wave-step tangent constants.

    ``alpha``/``u_alpha`` (and ``use_receiver_weights``) turn on the WEIGHTED forward tangent: the
    parameter JVP differentiates through ``receiver_norm`` (so ``dmax_transfer`` carries the alpha
    coupling) and the softmax-Jacobian seed ``dreceiver_log_probs`` is threaded into the
    E-step tangent fixed point ALONGSIDE the wave tangent (consistency required: an inconsistent
    dreceiver_log_probs -> O(1) FD failure). With ``alpha=None`` this is the legacy uniform theta-only tangent
    (bit-for-bit)."""
    sh = static.species_helpers
    if alpha is None:
        dlog_pS, dlog_pD, dlog_pL, dmax_transfer = param_jvp_uniform(static, theta, v)
        dreceiver_log_probs = None
    else:
        dlog_pS, dlog_pD, dlog_pL, dmax_transfer, dreceiver_log_probs = param_jvp_weighted(
            static, theta, alpha, v, u_alpha,
        )
    dE, dE_s1, dE_s2, dEbar = e_tangent_fixed_point(
        sv["E"], dlog_pS, dlog_pD, dlog_pL, dmax_transfer,
        sv["log_pS"], sv["log_pD"], sv["log_pL"], sv["max_transfer"], sv["receiver_log_probs"],
        sh["sp_parent"], sh["sp_child1"], sh["sp_child2"],
        species_height=sh["sp_height"],
        species_levels=int(sh["compact_level_ptr"].numel()) - 1,
        max_iter=int(static.solver_options.e_max_iter), tol=e_tol,
        use_receiver_weights=bool(use_receiver_weights), dreceiver_log_probs=dreceiver_log_probs,
        leaf_fm_log=leaf_fm_log,
    )
    if raw_out is not None:
        raw_out.update(
            dlog_pS=dlog_pS,
            dlog_pD=dlog_pD,
            dlog_pL=dlog_pL,
            d_max_transfer=dmax_transfer,
            d_extinction=dE,
            d_extinction_child1=dE_s1,
            d_extinction_child2=dE_s2,
            d_extinction_complement=dEbar,
            dreceiver_log_probs=dreceiver_log_probs,
        )
    S_ = S
    family_rows = int(sv["E"].shape[0])
    d_extinction = as_family_species(dE, S_, family_rows)
    d_extinction_complement = as_family_species(dEbar, S_, family_rows)
    d_extinction_child1 = as_family_species(dE_s1, S_, family_rows)
    d_extinction_child2 = as_family_species(dE_s2, S_, family_rows)
    d_duplication_log_probability = as_family_species(
        dlog_pD, S_, family_rows
    )
    d_speciation_log_probability = as_family_species(
        dlog_pS, S_, family_rows
    )
    d_max_transfer = as_family_species(
        dmax_transfer.squeeze(-1), S_, family_rows
    )
    return {
        "d_duplication_loss_const": d_duplication_log_probability
        + d_extinction,
        "d_extinction_complement": d_extinction_complement,
        "d_extinction": d_extinction,
        "d_speciation_child1_const": d_speciation_log_probability
        + d_extinction_child2,
        "d_speciation_child2_const": d_speciation_log_probability
        + d_extinction_child1,
        "d_max_transfer": d_max_transfer,
        "d_leaf_log_probability": d_speciation_log_probability,
        "d_duplication_log_probability_param": as_family_param(
            dlog_pD, family_rows, S_
        ),
        "d_speciation_log_probability_param": as_family_param(
            dlog_pS, family_rows, S_
        ),
        "_dreceiver_log_probs": dreceiver_log_probs,
    }


def jvp_root_scores(static, theta, v, sv, *, self_tol=None, self_max_iter=DEFAULT_SELF_MAX_ITER, e_tol=None,
                    self_iters=None, return_full=False, keep_d_dts=True, fused_selfloop=True,
                    alpha=None, u_alpha=None, leaf_fm_log=None):
    """d(Pi_root)/d[theta;alpha] . [v;u_alpha]  -> tensor [n_root_rows, S].

    ``self_iters`` (int): run the per-wave self-loop for a FIXED number of Jacobi steps with
    no per-iteration host sync — this matches the primal forward's ``pi_iters`` truncation
    (N Jacobi steps from a zero tangent == the N-term Neumann partial sum the primal uses) and
    streams the tangent sweep without CPU<->GPU stalls. ``self_iters=None`` (default) keeps the
    adaptive converge-to-``self_tol`` loop used by the fp64 verification gates.

    ``alpha``/``u_alpha`` turn on the WEIGHTED forward tangent (S3): the parameter JVP goes through
    ``extract_parameters_weighted_receivers`` so ``dmax_transfer`` carries the alpha->receiver_norm coupling,
    and the softmax-Jacobian seed ``dreceiver_log_probs`` is threaded identically into the
    E-step tangent fixed point AND the wave-step tangent (use_receiver_weights=True). At a non-uniform
    base this is what makes the tangent FINITE and self-consistent with the weighted primal (the
    legacy uniform tangent NaNs there). ``alpha=None`` -> legacy uniform theta-only tangent.

    ``leaf_fm_log`` ([S], log2) is the fixed fraction-missing leaf boundary threaded into the
    E-step tangent fixed point so the PRIMAL E_s1/E_s2 recomputed there match the fraction-missing
    forward (the tangents dE_s1/dE_s2 stay 0 -- fraction_missing is a constant input, no curvature).
    ``None`` (default) reproduces the no-fraction-missing tangent BIT-FOR-BIT.

    With ``return_full=True`` returns (root_tangents, full) where ``full`` carries everything the
    exact-HVP tangent-adjoint sweep needs: dPi/dPibar [C,S] buffers, per-wave
    gene-split tangents keyed by wave start, equation-named wave constants,
    raw parameter tangents, extinction tangents, and ``dreceiver_log_probs``.
    """
    sh, wl = static.species_helpers, static.wave_layout
    S = int(sh["S"])
    if self_tol is None:
        self_tol = _default_tol(theta.dtype)
    if e_tol is None:
        e_tol = _default_tol(theta.dtype)
    family_idx = static.rate_family_idx
    leaf_species_idx = wl["leaf_species_index"].to(torch.int32)
    species_child1 = sh["sp_child1"]
    species_child2 = sh["sp_child2"]
    species_parent = sh["sp_parent"]
    species_height = sh.get("sp_height")
    # The species tree's height. One bucket per height in the compact level tables, so this is a
    # shape rather than a value: no device-to-host copy, unlike reducing species_height itself.
    species_levels = int(sh["compact_level_ptr"].numel()) - 1
    # The tangent is the forward tree system with a different right-hand side, so
    # SolverOptions.adjoint_self_loop -- which already says "solve the wave's linear system
    # exactly rather than iterating it" for the adjoint -- selects it here too.
    exact_selfloop = getattr(static, "solver_options", None) is not None and (
        static.solver_options.adjoint_self_loop == "exact"
    )
    if species_height is None:
        raise ValueError(
            "the wave-tangent self-loop needs species_helpers['sp_height']; rebuild the "
            "model so the species payload carries it"
        )

    # S3: weighted tangent iff a non-uniform base alpha is supplied (matches the primal's
    # use_receiver_weights derivation). dreceiver_log_probs is the softmax-Jacobian seed; col is the live
    # receiver_log_probs already stored in sv.
    weighted = alpha is not None
    use_receiver_weights = bool(weighted) and not receiver_weights_are_uniform(alpha)
    receiver_log_probs = sv["receiver_log_probs"]

    base = wave_step_constants(sv, S)
    raw = {} if return_full else None
    tangent_constants = _wave_tangent_constants(
        static, theta, v, sv, S, e_tol, raw_out=raw,
        alpha=alpha if weighted else None, u_alpha=u_alpha, use_receiver_weights=use_receiver_weights,
        leaf_fm_log=leaf_fm_log,
    )
    dreceiver_log_probs = tangent_constants.pop("_dreceiver_log_probs")

    pi = sv["pi_wave"]
    pibar = sv["pibar_wave"]
    pi_state = sv["pi_state"]
    pi_state.validate(pi, pibar, sv["pibar_row_max"], check_values=False)
    pi_offset = pi_state.pi_offset
    pibar_offset = pi_state.pibar_offset
    # The forward's own record of the rows it could not hold under one row scale. The tangent's
    # elimination underflows on the same lanes, so those rows take the sweeps instead. None
    # whenever nothing was flagged, which leaves the tangent exactly as it was.
    tangent_wide_row = pi_state.wide_row if pi_state.wide_row_total > 0 else None
    C = int(pi.shape[0])
    dpi = torch.zeros((C, S), device=pi.device, dtype=pi.dtype)
    dpibar = torch.zeros((C, S), device=pi.device, dtype=pi.dtype)
    d_gene_split_by_wave_start = {} if return_full else None

    def step(
        dPi_out,
        gene_split_log_likelihood,
        d_gene_split_log_likelihood,
        gene_split_offset,
        ws,
        W,
        has_leaf,
        store,
    ):
        compute_wave_step_tangent(
            pi, dpi, dPi_out, ws, W, S,
            base["max_transfer"], tangent_constants["d_max_transfer"],
            base["duplication_loss_const"], tangent_constants["d_duplication_loss_const"],
            base["extinction_complement"], tangent_constants["d_extinction_complement"],
            base["extinction"], tangent_constants["d_extinction"],
            base["speciation_child1_const"], tangent_constants["d_speciation_child1_const"],
            base["speciation_child2_const"], tangent_constants["d_speciation_child2_const"],
            receiver_log_probs, species_child1, species_child2, species_parent,
            gene_split_log_likelihood, d_gene_split_log_likelihood,
            leaf_species_idx=leaf_species_idx,
            leaf_logp=base["leaf_log_probability"],
            d_leaf_logp=tangent_constants["d_leaf_log_probability"],
            family_idx=family_idx, dPibar_out=(dpibar if store else None),
            has_leaf_term=has_leaf, input_ws=None, use_receiver_weights=use_receiver_weights,
            dreceiver_log_probs=dreceiver_log_probs,
            species_height=species_height, species_levels=species_levels,
            pi_offset=pi_offset, gene_split_offset=gene_split_offset,
        )

    for meta in wl["wave_metas"]:
        ws, W = int(meta["start"]), int(meta["W"])
        has_splits = "sl" in meta
        has_leaf = not has_splits
        if has_splits:
            gene_split_log_likelihood, gene_split_offset = compute_dts_forward(
                pi, pi_offset, pibar, pibar_offset,
                meta["sl"], meta["sr"], species_child1, species_child2,
                W, meta["reduce_idx"],
                base["duplication_log_probability_param"],
                base["speciation_log_probability_param"], family_idx=family_idx,
                log_split_probs=_select_log_split_probs(meta, pi.dtype), n_single_split_parents=meta.get("n_eq1"),
                single_split_parent_rows=meta.get("eq1_reduce_idx"), multiple_split_group_ptr=meta.get("ge2_ptr"),
                multiple_split_parent_rows=meta.get("ge2_parent_ids"),
                max_splits_per_multiple_parent=meta.get("ge2_max_fanout"), family_offset=ws,
            )
            d_gene_split_log_likelihood = compute_dts_tangent(
                pi, pibar, dpi, dpibar, meta["sl"], meta["sr"],
                species_child1, species_child2, W, meta["reduce_idx"],
                base["duplication_log_probability_param"],
                base["speciation_log_probability_param"],
                tangent_constants["d_duplication_log_probability_param"],
                tangent_constants["d_speciation_log_probability_param"],
                gene_split_log_likelihood, family_idx,
                log_split_probs=_select_log_split_probs(meta, pi.dtype), family_offset=ws,
                pi_offset=pi_offset, pibar_offset=pibar_offset,
                gene_split_offset=gene_split_offset,
            )
        else:
            gene_split_log_likelihood = None
            d_gene_split_log_likelihood = None
            gene_split_offset = None
        if (
            return_full
            and d_gene_split_log_likelihood is not None
            and keep_d_dts
        ):
            d_gene_split_by_wave_start[ws] = d_gene_split_log_likelihood

        if self_iters is not None and fused_selfloop:
            # fixed-count, sync-free Jacobi matching the primal forward's pi_iters truncation.
            # Fused into ONE launch: the n_it-step in-place self-loop runs register-resident
            # (primal weights/r/constants are loop-invariant -> loaded once), collapsing n_it
            # launches -> 1 and the invariant global traffic ~n_it x. Numerically identical to
            # looping `step` n_it times in-place (last step writes dpibar).
            compute_wave_step_tangent_selfloop(
                pi, dpi, ws, W, S, max(int(self_iters), 1),
                base["max_transfer"], tangent_constants["d_max_transfer"],
                base["duplication_loss_const"], tangent_constants["d_duplication_loss_const"],
                base["extinction_complement"], tangent_constants["d_extinction_complement"],
                base["extinction"], tangent_constants["d_extinction"],
                base["speciation_child1_const"], tangent_constants["d_speciation_child1_const"],
                base["speciation_child2_const"], tangent_constants["d_speciation_child2_const"],
                receiver_log_probs, species_child1, species_child2, species_parent,
                gene_split_log_likelihood, d_gene_split_log_likelihood,
                leaf_species_idx=leaf_species_idx,
                leaf_logp=base["leaf_log_probability"],
                d_leaf_logp=tangent_constants["d_leaf_log_probability"],
                family_idx=family_idx, dPibar_out=dpibar, has_leaf_term=has_leaf,
                use_receiver_weights=use_receiver_weights, dreceiver_log_probs=dreceiver_log_probs,
                pi_offset=pi_offset, gene_split_offset=gene_split_offset,
                species_height=species_height, species_levels=species_levels,
                exact=exact_selfloop,
                wide_row=tangent_wide_row if exact_selfloop else None,
            )
        elif self_iters is not None:
            # reference (unfused) fixed-count path: one launch per Jacobi step
            n_it = max(int(self_iters), 1)
            for _ in range(n_it - 1):
                step(
                    dpi,
                    gene_split_log_likelihood,
                    d_gene_split_log_likelihood,
                    gene_split_offset,
                    ws,
                    W,
                    has_leaf,
                    store=False,
                )
            step(
                dpi,
                gene_split_log_likelihood,
                d_gene_split_log_likelihood,
                gene_split_offset,
                ws,
                W,
                has_leaf,
                store=True,
            )
        else:
            prev = dpi.narrow(0, ws, W).clone()
            converged = False
            for _ in range(int(self_max_iter)):
                step(
                    dpi,
                    gene_split_log_likelihood,
                    d_gene_split_log_likelihood,
                    gene_split_offset,
                    ws,
                    W,
                    has_leaf,
                    store=False,
                )
                cur = dpi.narrow(0, ws, W)
                diff = float((cur - prev).abs().max())
                scale = float(cur.abs().max())
                if diff <= self_tol * max(1.0, scale):
                    converged = True
                    break
                prev = cur.clone()
            if not converged:
                # Silent truncation here returns a NON-converged tangent -> the Jvp
                # (hence the HVP / GGN curvature, and any PD certificate built on it)
                # is wrong. Fail loud. Constant message so warnings dedupes to once.
                warnings.warn(
                    "jvp_root_scores tangent self-loop hit self_max_iter without "
                    "converging; the tangent (Jvp/HVP/GGN curvature) is truncated and "
                    "may be inaccurate. Raise self_max_iter (default 200) or self_tol.",
                    RuntimeWarning, stacklevel=2,
                )
            step(
                dpi,
                gene_split_log_likelihood,
                d_gene_split_log_likelihood,
                gene_split_offset,
                ws,
                W,
                has_leaf,
                store=True,
            )

    roots = dpi.index_select(0, wl["root_clade_ids"])
    if return_full:
        return roots, dict(
            dPi=dpi,
            dPibar=dpibar,
            d_gene_split_by_wave_start=d_gene_split_by_wave_start,
            tangent_constants=tangent_constants,
            **raw,
        )
    return roots
