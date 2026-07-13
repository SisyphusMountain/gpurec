"""Full forward-mode tangent (Jvp) of the root scores w.r.t. theta.

``jvp_root_scores(static, theta, v, sv)`` returns ``d(Pi_root)/dtheta . v`` by threading tangents
through the whole forward solve, mirroring ``solve_e_pi`` + ``pi_wave_forward``:

  1. parameter tangent ``dparams = d(extract_parameters)/dtheta . v`` (forward-mode autodiff);
  2. E-step tangent fixed point (``e_tangent_fixed_point``);
  3. Pi-wave tangent: per wave (topological order), the cross-wave ``dts`` tangent then the
     self-loop tangent solved to convergence (the same true fixed point the adjoint differentiates).

This is the ``J`` of the Gauss-Newton operator ``M = J^T B J``; the matching ``J^T`` reuses the
existing backward (see ``ggn.py``).
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
from gpurec.core.kernels.dts_fused import compute_dts_forward
from gpurec.core.kernels.centered_pi_forward import compute_dts_forward_centered
from gpurec.core.kernels.dts_tangent import compute_dts_tangent
from gpurec.core.kernels.e_step_tangent import e_tangent_fixed_point
from gpurec.core.kernels.wave_tangent import (
    compute_wave_step_tangent, compute_wave_step_tangent_selfloop,
)

DEFAULT_SELF_MAX_ITER = 200


def param_jvp_uniform(static, theta, v):
    """Forward-mode tangent of extract_parameters_uniform along v (use_col_weights=False path)."""
    unnorm_row_max = static.species_helpers["unnorm_row_max"].to(device=theta.device, dtype=theta.dtype)

    def f(th):
        return extract_parameters_uniform(
            th, unnorm_row_max, specieswise=static.specieswise, genewise=static.genewise
        )

    primals, tangents = jvp(f, (theta,), (v,))
    # (log_pS, log_pD, log_pL, max_coupling)
    return tangents


def param_jvp_weighted(static, theta, alpha, u_theta, u_alpha):
    """Forward-mode tangent of ``extract_parameters_weighted_receivers`` along (u_theta, u_alpha).

    Returns ``(dlog_pS, dlog_pD, dlog_pL, dmax_transfer, dcol)`` where the new alpha coupling lives
    in TWO outputs that ``param_jvp_uniform`` lacked:

    * ``dmax_transfer`` now carries the ``alpha -> receiver_log_probs -> receiver_valid_log_normalizer
      (receiver_norm) -> max_transfer`` coupling (the DOMINANT alpha->rate sensitivity; the uniform
      extractor dropped ``receiver_norm`` entirely);
    * ``dcol = dreceiver_log_probs`` is the softmax-Jacobian applied to ``u_alpha``,
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
        )

    primals, tangents = jvp(f, (theta, al), (u_theta, ua))
    # (dlog_pS, dlog_pD, dlog_pL, dmax_transfer, dreceiver_log_probs)
    return tangents


def wave_step_constants(sv, S):
    """Base per-item wave-step constants (mirrors pi_wave_forward)."""
    item_rows = int(sv["E"].shape[0])
    e_item = as_family_species(sv["E"], S, item_rows)
    ebar_item = as_family_species(sv["Ebar"], S, item_rows)
    e_s1_item = as_family_species(sv["E_s1"], S, item_rows)
    e_s2_item = as_family_species(sv["E_s2"], S, item_rows)
    mc_item = as_family_species(sv["max_transfer"].squeeze(-1), S, item_rows)
    pd_item = as_family_species(sv["log_pD"], S, item_rows)
    ps_item = as_family_species(sv["log_pS"], S, item_rows)
    return {
        "dl": 1.0 + pd_item + e_item, "ebar": ebar_item, "e": e_item,
        "sl1": ps_item + e_s2_item, "sl2": ps_item + e_s1_item,
        "mc": mc_item, "leaf": ps_item,
        "pd_param": as_family_param(sv["log_pD"], item_rows, S),
        "ps_param": as_family_param(sv["log_pS"], item_rows, S),
    }


def _default_tol(dtype):
    return dtype_rel_tol_default(dtype)


def _wave_tangent_constants(static, theta, v, sv, S, e_tol, raw_out=None,
                            alpha=None, u_alpha=None, use_col_weights=False):
    """E + parameter tangents assembled into the wave-step tangent constants.

    ``alpha``/``u_alpha`` (and ``use_col_weights``) turn on the WEIGHTED forward tangent: the
    parameter JVP differentiates through ``receiver_norm`` (so ``dmax_coupling`` carries the alpha
    coupling) and the softmax-Jacobian seed ``dcol = dreceiver_log_probs`` is threaded into the
    E-step tangent fixed point ALONGSIDE the wave tangent (consistency required: an inconsistent
    dcol -> O(1) FD failure). With ``alpha=None`` this is the legacy uniform theta-only tangent
    (bit-for-bit)."""
    sh = static.species_helpers
    if alpha is None:
        dlog_pS, dlog_pD, dlog_pL, dmax_coupling = param_jvp_uniform(static, theta, v)
        dcol = None
    else:
        dlog_pS, dlog_pD, dlog_pL, dmax_coupling, dcol = param_jvp_weighted(
            static, theta, alpha, v, u_alpha,
        )
    dE, dE_s1, dE_s2, dEbar = e_tangent_fixed_point(
        sv["E"], dlog_pS, dlog_pD, dlog_pL, dmax_coupling,
        sv["log_pS"], sv["log_pD"], sv["log_pL"], sv["max_transfer"], sv["receiver_log_probs"],
        sh["sp_parent"], sh["sp_child1"], sh["sp_child2"], int(sh["max_ancestor_depth"]),
        max_iter=int(static.solver_options.e_max_iter), tol=e_tol,
        use_col_weights=bool(use_col_weights), dcol_log_probs=dcol,
    )
    if raw_out is not None:
        raw_out.update(dlog_pS=dlog_pS, dlog_pD=dlog_pD, dlog_pL=dlog_pL,
                       dmax_coupling=dmax_coupling, dE=dE, dE_s1=dE_s1, dE_s2=dE_s2, dEbar=dEbar,
                       dreceiver_log_probs=dcol)
    S_ = S
    item_rows = int(sv["E"].shape[0])
    de_item = as_family_species(dE, S_, item_rows)
    debar_item = as_family_species(dEbar, S_, item_rows)
    de_s1_item = as_family_species(dE_s1, S_, item_rows)
    de_s2_item = as_family_species(dE_s2, S_, item_rows)
    dpd_item = as_family_species(dlog_pD, S_, item_rows)
    dps_item = as_family_species(dlog_pS, S_, item_rows)
    dmc_item = as_family_species(dmax_coupling.squeeze(-1), S_, item_rows)
    return {
        "dDL": dpd_item + de_item, "dEbar": debar_item, "dE": de_item,
        "dSL1": dps_item + de_s2_item, "dSL2": dps_item + de_s1_item,
        "dMC": dmc_item, "dleaf": dps_item,
        "dpd_param": as_family_param(dlog_pD, item_rows, S_),
        "dps_param": as_family_param(dlog_pS, item_rows, S_),
        "_dcol": dcol,  # softmax-Jacobian alpha seed (None on the legacy uniform path)
    }


def jvp_root_scores(static, theta, v, sv, *, self_tol=None, self_max_iter=DEFAULT_SELF_MAX_ITER, e_tol=None,
                    self_iters=None, return_full=False, keep_d_dts=True, fused_selfloop=True,
                    alpha=None, u_alpha=None):
    """d(Pi_root)/d[theta;alpha] . [v;u_alpha]  -> tensor [n_root_rows, S].

    ``self_iters`` (int): run the per-wave self-loop for a FIXED number of Jacobi steps with
    no per-iteration host sync — this matches the primal forward's ``pi_iters`` truncation
    (N Jacobi steps from a zero tangent == the N-term Neumann partial sum the primal uses) and
    streams the tangent sweep without CPU<->GPU stalls. ``self_iters=None`` (default) keeps the
    adaptive converge-to-``self_tol`` loop used by the fp64 verification gates.

    ``alpha``/``u_alpha`` turn on the WEIGHTED forward tangent (S3): the parameter JVP goes through
    ``extract_parameters_weighted_receivers`` so ``dMC`` carries the alpha->receiver_norm coupling,
    and the softmax-Jacobian seed ``dcol = dreceiver_log_probs`` is threaded IDENTICALLY into the
    E-step tangent fixed point AND the wave-step tangent (use_col_weights=True). At a non-uniform
    base this is what makes the tangent FINITE and self-consistent with the weighted primal (the
    legacy uniform tangent NaNs there). ``alpha=None`` -> legacy uniform theta-only tangent.

    With ``return_full=True`` returns (root_tangents, full) where ``full`` carries everything the
    exact-HVP tangent-adjoint sweep needs: dPi/dPibar [C,S] buffers, per-wave d_dts (dict keyed by
    wave start), the tangent constants dict (dDL/dEbar/dE/dSL1/dSL2/dMC/dleaf/dpd_param/
    dps_param), the raw parameter tangents (dlog_pS, dlog_pD, dlog_pL, dmax_coupling), the E
    tangents (dE*, dE_s1, dE_s2, dEbar), and ``dreceiver_log_probs`` (= dcol, the alpha seed).
    """
    sh, wl = static.species_helpers, static.wave_layout
    S = int(sh["S"])
    if self_tol is None:
        self_tol = _default_tol(theta.dtype)
    if e_tol is None:
        e_tol = _default_tol(theta.dtype)
    item_idx = static.rate_family_idx
    leaf_state_idx = wl["leaf_species_index"].to(torch.int32)
    c1, c2, parent = sh["sp_child1"], sh["sp_child2"], sh["sp_parent"]
    mad = int(sh["max_ancestor_depth"])

    # S3: weighted tangent iff a non-uniform base alpha is supplied (matches the primal's
    # use_receiver_weights derivation). dcol is the softmax-Jacobian seed; col is the live
    # receiver_log_probs already stored in sv.
    weighted = alpha is not None
    use_col_weights = bool(weighted) and not receiver_weights_are_uniform(alpha)
    col_logp = sv["receiver_log_probs"]

    base = wave_step_constants(sv, S)
    raw = {} if return_full else None
    dcst = _wave_tangent_constants(
        static, theta, v, sv, S, e_tol, raw_out=raw,
        alpha=alpha if weighted else None, u_alpha=u_alpha, use_col_weights=use_col_weights,
    )
    dcol = dcst.pop("_dcol")

    pi = sv["pi_wave"]
    pibar = sv["pibar_wave"]
    centered_state = sv.get("centered_pi_state")
    centered = centered_state is not None
    configured_centered = (
        str(getattr(static.solver_options, "pi_representation", "absolute")).strip().lower()
        == "centered"
    )
    if configured_centered and not centered:
        raise RuntimeError(
            "centered JVP requires saved['centered_pi_state'] from the matching forward_solve"
        )
    pi_offset = centered_state.pi_offset if centered else None
    pibar_offset = centered_state.pibar_offset if centered else None
    C = int(pi.shape[0])
    dpi = torch.zeros((C, S), device=pi.device, dtype=pi.dtype)
    dpibar = torch.zeros((C, S), device=pi.device, dtype=pi.dtype)
    d_dts_by_ws = {} if return_full else None

    def step(dPi_out, dts_r, d_dts, dts_offset, ws, W, has_leaf, store):
        compute_wave_step_tangent(
            pi, dpi, dPi_out, ws, W, S,
            base["mc"], dcst["dMC"], base["dl"], dcst["dDL"], base["ebar"], dcst["dEbar"],
            base["e"], dcst["dE"], base["sl1"], dcst["dSL1"], base["sl2"], dcst["dSL2"],
            col_logp, c1, c2, parent, mad, dts_r, d_dts,
            leaf_state_idx=leaf_state_idx, leaf_logp=base["leaf"], dleaf_logp=dcst["dleaf"],
            item_idx=item_idx, dPibar_out=(dpibar if store else None),
            has_leaf_term=has_leaf, input_ws=None, use_col_weights=use_col_weights,
            dcol_log_probs=dcol,
            pi_offset=pi_offset, dts_offset=dts_offset,
        )

    for meta in wl["wave_metas"]:
        ws, W = int(meta["start"]), int(meta["W"])
        has_splits = "sl" in meta
        has_leaf = not has_splits
        if has_splits:
            if centered:
                dts_r, dts_offset = compute_dts_forward_centered(
                    pi, pi_offset, pibar, pibar_offset,
                    meta["sl"], meta["sr"], c1, c2, W, meta["reduce_idx"],
                    base["pd_param"], base["ps_param"], family_idx=item_idx,
                    log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
                    eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
                    ge2_parent_ids=meta.get("ge2_parent_ids"),
                    ge2_max_fanout=meta.get("ge2_max_fanout"), family_offset=ws,
                )
            else:
                dts_r = compute_dts_forward(
                    pi, pibar, meta["sl"], meta["sr"], c1, c2, W, meta["reduce_idx"],
                    base["pd_param"], base["ps_param"], family_idx=item_idx,
                    log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
                    eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
                    ge2_parent_ids=meta.get("ge2_parent_ids"),
                    ge2_max_fanout=meta.get("ge2_max_fanout"), family_offset=ws,
                )
                dts_offset = None
            d_dts = compute_dts_tangent(
                pi, pibar, dpi, dpibar, meta["sl"], meta["sr"], c1, c2, W, meta["reduce_idx"],
                base["pd_param"], base["ps_param"], dcst["dpd_param"], dcst["dps_param"], dts_r, item_idx,
                log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
                eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
                ge2_parent_ids=meta.get("ge2_parent_ids"), ge2_max_fanout=meta.get("ge2_max_fanout"),
                item_offset=ws,
                pi_offset=pi_offset, pibar_offset=pibar_offset, dts_offset=dts_offset,
            )
        else:
            dts_r = d_dts = dts_offset = None
        if return_full and d_dts is not None and keep_d_dts:
            d_dts_by_ws[ws] = d_dts

        if self_iters is not None and fused_selfloop:
            # fixed-count, sync-free Jacobi matching the primal forward's pi_iters truncation.
            # Fused into ONE launch: the n_it-step in-place self-loop runs register-resident
            # (primal weights/r/constants are loop-invariant -> loaded once), collapsing n_it
            # launches -> 1 and the invariant global traffic ~n_it x. Numerically identical to
            # looping `step` n_it times in-place (last step writes dpibar).
            compute_wave_step_tangent_selfloop(
                pi, dpi, ws, W, S, max(int(self_iters), 1),
                base["mc"], dcst["dMC"], base["dl"], dcst["dDL"], base["ebar"], dcst["dEbar"],
                base["e"], dcst["dE"], base["sl1"], dcst["dSL1"], base["sl2"], dcst["dSL2"],
                col_logp, c1, c2, parent, mad, dts_r, d_dts,
                leaf_state_idx=leaf_state_idx, leaf_logp=base["leaf"], dleaf_logp=dcst["dleaf"],
                item_idx=item_idx, dPibar_out=dpibar, has_leaf_term=has_leaf,
                use_col_weights=use_col_weights, dcol_log_probs=dcol,
                pi_offset=pi_offset, dts_offset=dts_offset,
            )
        elif self_iters is not None:
            # reference (unfused) fixed-count path: one launch per Jacobi step
            n_it = max(int(self_iters), 1)
            for _ in range(n_it - 1):
                step(dpi, dts_r, d_dts, dts_offset, ws, W, has_leaf, store=False)  # in-place Jacobi
            step(dpi, dts_r, d_dts, dts_offset, ws, W, has_leaf, store=True)  # last step writes dpibar
        else:
            prev = dpi.narrow(0, ws, W).clone()
            converged = False
            for _ in range(int(self_max_iter)):
                step(dpi, dts_r, d_dts, dts_offset, ws, W, has_leaf, store=False)  # in-place Jacobi on dpi[ws:ws+W]
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
            step(dpi, dts_r, d_dts, dts_offset, ws, W, has_leaf, store=True)  # write converged dpibar[ws:ws+W]

    roots = dpi.index_select(0, wl["root_clade_ids"])
    if return_full:
        return roots, dict(dPi=dpi, dPibar=dpibar, d_dts=d_dts_by_ws, dcst=dcst, **raw)
    return roots
