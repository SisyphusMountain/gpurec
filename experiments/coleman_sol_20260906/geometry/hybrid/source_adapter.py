"""Fail-closed experiment adapter for the production genewise continuation.

This module does not mutate :mod:`gpurec.fit.genewise_fit`.  It reads the current function source,
asserts every narrow continuation site, compiles a private hierarchical variant in memory, and
leaves the original callable as the native control.
"""
from __future__ import annotations

import hashlib
import inspect

import torch

import gpurec.fit.genewise_fit as _production
from hierarchical_adapter import (
    phi_to_theta as _hier_phi_to_theta,
    working_set_tangent_direction as _hier_working_direction,
    retract_feasible_ray_cpu64 as _hier_retract,
    theta_to_phi as _hier_theta_to_phi,
    transform_gradient as _hier_gradient,
    transform_hessian as _hier_hessian,
    predicted_decrease as _hier_predicted,
)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"hierarchical source adapter expected one {label} site, found {count}")
    return source.replace(old, new)


def _compile_hierarchical_fit(trust_metric: str):
    if trust_metric not in ("native", "coordinate"):
        raise ValueError(trust_metric)
    source = inspect.getsource(_production.fit_genewise)
    source_hash = hashlib.sha256(source.encode()).hexdigest()

    source = _replace_once(source,
'''    prev_free = torch.zeros(F_all, 3, device=dev, dtype=dtype)
''',
'''    prev_free = torch.zeros(F_all, 3, device=dev, dtype=dtype)
    # Experiment-only ambient hierarchical secant state. ``prev_phi_seen`` replaces the native
    # free-mask's accidental first-iterate guard without masking any u/v/w coordinate.
    prev_phi = torch.zeros(F_all, 3, device=dev, dtype=dtype)
    prev_g_phi = torch.zeros(F_all, 3, device=dev, dtype=dtype)
    prev_phi_seen = torch.zeros(F_all, dtype=torch.bool, device=dev)
    # Diagnose ray truncation stalls without altering the working set or step.
    hier_zero_step_count = torch.zeros(F_all, dtype=torch.long, device=dev)
    hier_nearzero_step_count = torch.zeros(F_all, dtype=torch.long, device=dev)
    hier_boundary_zero_count = torch.zeros(F_all, dtype=torch.long, device=dev)
    hier_zero_direction_count = torch.zeros(F_all, dtype=torch.long, device=dev)
    hier_native_zero_count = torch.zeros(F_all, dtype=torch.long, device=dev)
    hier_min_ray_fraction = torch.ones(F_all, dtype=dtype, device=dev)
''', "ambient secant state")

    source = _replace_once(source,
'''                _track_best(active, lv, sub, live, n_steps)
''',
'''                _track_best(active, lv, sub, live, n_steps)
                phi = _hier_theta_to_phi(sub)
                g_phi = _hier_gradient(phi, g)
''', "point gradient transform")

    source = _replace_once(source,
'''                if not refresh_due:   # carry the curvature forward from the step just taken
                    both = free * prev_free.index_select(0, active)
                    B_fam.index_copy_(0, active, _carry_curvature(
                        B_fam.index_select(0, active),
                        sub - prev_theta.index_select(0, active),
                        g - prev_g.index_select(0, active), both, active))
''',
'''                if not refresh_due:   # ambient BFGS in true phi, with gradients at own points
                    seen_phi = prev_phi_seen.index_select(0, active).to(dtype).unsqueeze(1)
                    B_fam.index_copy_(0, active, _carry_curvature(
                        B_fam.index_select(0, active),
                        phi - prev_phi.index_select(0, active),
                        g_phi - prev_g_phi.index_select(0, active),
                        torch.ones_like(g_phi) * seen_phi, active))
''', "ambient BFGS update")

    source = _replace_once(source,
'''                    free = staged_free((~fixed).to(dtype), n_steps)
                prev_theta.index_copy_(0, active, sub)
                prev_g.index_copy_(0, active, g)
                prev_free.index_copy_(0, active, free)
''',
'''                    free = staged_free((~fixed).to(dtype), n_steps)
                    phi = torch.where(keep, prev_phi.index_select(0, active), phi)
                    g_phi = torch.where(keep, prev_g_phi.index_select(0, active), g_phi)
                prev_theta.index_copy_(0, active, sub)
                prev_g.index_copy_(0, active, g)
                prev_free.index_copy_(0, active, free)
                prev_phi.index_copy_(0, active, phi)
                prev_g_phi.index_copy_(0, active, g_phi)
                prev_phi_seen.index_fill_(0, active, True)
''', "rollback and previous hierarchical state")

    source = _replace_once(source,
'''                            B_fam.index_copy_(0, got, H_s[ref_s])
''',
'''                            H_s_phi = _hier_hessian(
                                _hier_theta_to_phi(sub[stuck][ref_s]), g[stuck][ref_s], H_s[ref_s])
                            B_fam.index_copy_(0, got, H_s_phi)
''', "targeted exact Hessian transform")

    source = _replace_once(source,
'''                    B_fam.index_copy_(0, active[refreshed], H_new[refreshed])
''',
'''                    H_new_phi = _hier_hessian(phi[refreshed], g[refreshed], H_new[refreshed])
                    B_fam.index_copy_(0, active[refreshed], H_new_phi)
''', "full exact Hessian transform")

    old_step = '''                r_step = radius.index_select(0, active).unsqueeze(1)
                Hred, delta = convexified_step(B_fam.index_select(0, active), g, free,
                                               radius.index_select(0, active))
                delta = delta * live.unsqueeze(1).to(dtype)   # settled rows keep their frozen theta
                reshaped = step_model == "rate_affine" and not exact_curvature_now
                if reshaped:
                    # The quadratic step assumes the gradient is affine in the LOG rate; a count
                    # likelihood's gradient is affine in the RATE. log2(1 + ln2 * delta) is the
                    # exact minimizer of that model, coordinate by coordinate. A coordinate whose
                    # 1 + ln2 * delta is not positive is asking for a negative rate: send it down by
                    # the trust radius instead (the box clamp below stops it at the rate floor).
                    inner = 1.0 + _LN2 * delta
                    positive = inner > 0
                    delta = torch.where(
                        positive,
                        torch.log2(torch.where(positive, inner, torch.ones_like(inner))),
                        -r_step.expand_as(inner))
                    delta = delta * live.unsqueeze(1).to(dtype)
                if step_extrapolation != 1.0:
                    # Lengthen a step that continues a well-judged one in the same direction.
                    prev_s = prev_step.index_select(0, active)
                    denom = delta.norm(dim=1) * prev_s.norm(dim=1)
                    turning = denom > 0
                    cosine = torch.where(turning, (delta * prev_s).sum(dim=1)
                                         / torch.where(turning, denom, torch.ones_like(denom)),
                                         torch.zeros_like(denom))
                    lengthen = extrapolate_ok.index_select(0, active) & (cosine > 0.9) & live
                    delta = torch.where(lengthen.unsqueeze(1), delta * step_extrapolation, delta)
                dn = delta.norm(dim=1, keepdim=True)
                capped = dn > r_step
                step = delta * torch.where(capped, r_step / torch.where(capped, dn, torch.ones_like(dn)),
                                           torch.ones_like(dn))
                new_sub = clamp_(sub + step)
                applied = new_sub - sub   # what the box left of the step
                if reshaped:
                    # The decrease the rate-affine model predicts at the step actually applied:
                    # k ln2 d - a (2**d - 1) per coordinate, with a = Hred_jj / ln2**2 the rate
                    # curvature and k = a - g_j / ln2 the linear-in-rate coefficient, plus the
                    # quadratic model's off-diagonal cross terms (the rate model is separable).
                    h_diag = torch.diagonal(Hred, dim1=1, dim2=2)
                    a_rate = h_diag / (_LN2 * _LN2)
                    k_rate = a_rate - g / _LN2
                    diagonal_gain = (k_rate * _LN2 * applied
                                     - a_rate * (torch.exp2(applied) - 1.0)).sum(dim=1)
                    quad_full = (applied.unsqueeze(1) @ Hred @ applied.unsqueeze(2)).reshape(-1)
                    quad_diag = (h_diag * applied * applied).sum(dim=1)
                    predicted = diagonal_gain - 0.5 * (quad_full - quad_diag)
                else:
                    predicted = -((g * applied).sum(dim=1)
                                  + 0.5 * (applied.unsqueeze(1) @ Hred @ applied.unsqueeze(2)).reshape(-1))
                pred_dec.index_copy_(0, active, torch.where(live, predicted, torch.zeros_like(predicted)))
                last_capped.index_copy_(0, active, capped.squeeze(1) & live)
                if step_extrapolation != 1.0:
                    prev_step.index_copy_(0, active, applied)
                prev_nll.index_copy_(0, active, lv)
                sub = new_sub
'''
    new_step = '''                radius_active = radius.index_select(0, active)
                direction, Hred, g_face, working_fixed, _eigen_limited = _hier_working_direction(
                    phi, sub, g, g_phi, B_fam.index_select(0, active), fixed,
                    radius_active, mu, lo, hi, bounds.bound_active_eps,
                    metric=_hier_trust_metric)
                direction = direction * live.unsqueeze(1).to(dtype)
                phi_new, new_sub, applied_tangent, chord, capped = _hier_retract(
                    phi, sub, direction, working_fixed, lo, hi, radius_active,
                    metric=_hier_trust_metric)
                proposal_norm = direction.norm(dim=1)
                applied_norm = applied_tangent.norm(dim=1)
                proposed = proposal_norm > 0
                ray_fraction = torch.where(
                    proposed, applied_norm / torch.where(
                        proposed, proposal_norm, torch.ones_like(proposal_norm)),
                    torch.ones_like(proposal_norm))
                nonconverged = pgmax(sub, g) >= tol
                zero_ray = live & nonconverged & proposed & (applied_norm <= 1e-10)
                nearzero_ray = live & nonconverged & proposed & (ray_fraction <= 1e-6)
                zero_direction = live & nonconverged & (proposal_norm <= 1e-10)
                native_zero = live & nonconverged & ((new_sub - sub).norm(dim=1) <= 1e-10)
                at_native_face = ((sub <= lo + bounds.bound_active_eps)
                                  | (sub >= hi - bounds.bound_active_eps)).any(dim=1)
                hier_zero_step_count.index_add_(0, active, zero_ray.to(torch.long))
                hier_nearzero_step_count.index_add_(0, active, nearzero_ray.to(torch.long))
                hier_boundary_zero_count.index_add_(
                    0, active, (zero_ray & at_native_face).to(torch.long))
                hier_zero_direction_count.index_add_(0, active, zero_direction.to(torch.long))
                hier_native_zero_count.index_add_(0, active, native_zero.to(torch.long))
                old_fraction = hier_min_ray_fraction.index_select(0, active)
                hier_min_ray_fraction.index_copy_(
                    0, active, torch.minimum(old_fraction, ray_fraction))
                # Prediction is in the actually scaled tangent variable. The ambient chord is
                # deliberately reserved for the next BFGS secant through ``phi_new``.
                predicted = _hier_predicted(g_face, Hred, applied_tangent)
                pred_dec.index_copy_(0, active, torch.where(live, predicted, torch.zeros_like(predicted)))
                last_capped.index_copy_(0, active, capped & live)
                prev_nll.index_copy_(0, active, lv)
                sub = new_sub
'''
    source = _replace_once(source, old_step, new_step, "hierarchical trust step")

    source = _replace_once(source,
'''        gradient_work=gradient_work,
''',
'''        gradient_work=gradient_work,
        hierarchical_ray_diagnostics=dict(
            zero_step_count=hier_zero_step_count,
            nearzero_step_count=hier_nearzero_step_count,
            boundary_zero_count=hier_boundary_zero_count,
            zero_direction_nonconverged_count=hier_zero_direction_count,
            actual_native_zero_nonconverged_count=hier_native_zero_count,
            min_ray_fraction=hier_min_ray_fraction,
        ),
''', "ray diagnostic result")

    namespace = dict(vars(_production))
    namespace.update(
        _hier_theta_to_phi=_hier_theta_to_phi,
        _hier_gradient=_hier_gradient,
        _hier_hessian=_hier_hessian,
        _hier_working_direction=_hier_working_direction,
        _hier_retract=_hier_retract,
        _hier_predicted=_hier_predicted,
        _hier_trust_metric=trust_metric,
    )
    exec(compile(source, f"<hierarchical-fit:{source_hash[:12]}>", "exec"), namespace)
    return namespace["fit_genewise"], source_hash


_HIERARCHICAL_FITS = {}
for _metric in ("native", "coordinate"):
    _fit, PRODUCTION_SOURCE_SHA256 = _compile_hierarchical_fit(_metric)
    _HIERARCHICAL_FITS[_metric] = _fit


def fit_genewise_hierarchical(*args, **kwargs):
    """Run the experimental hierarchical continuation from a caller-paid endpoint seed.

    The bounded campaign intentionally supports only the common post-EM diagnostic configuration:
    Adam is disabled, curvature is a supplied hierarchical ``[F,3,3]`` tensor, and optional
    continuation features that change the step or coordinate mask are disabled.
    """
    trust_metric = kwargs.pop("hierarchical_trust_metric", "native")
    if trust_metric not in _HIERARCHICAL_FITS:
        raise ValueError("hierarchical_trust_metric must be 'native' or 'coordinate'")
    unsupported = {
        "warmup_method": kwargs.get("warmup_method") != "adam",
        "adam_steps": kwargs.get("adam_steps") != 0,
        "init_curvature": not isinstance(kwargs.get("init_curvature"), torch.Tensor),
        "curvature_update": kwargs.get("curvature_update") != "bfgs",
        "step_model": kwargs.get("step_model") != "quadratic",
        "step_extrapolation": kwargs.get("step_extrapolation") != 1.0,
        "stop_nll_bits": kwargs.get("stop_nll_bits") != 0.0,
        "coordinate_staging": kwargs.get("coordinate_staging") != (0, 0),
        "targeted_hessian": kwargs.get("targeted_hessian") != (0, 0.0),
    }
    bad = [name for name, value in unsupported.items() if value]
    if bad:
        raise ValueError("hierarchical experiment adapter requires the fixed post-EM recipe; "
                         f"unsupported settings: {', '.join(bad)}")
    result = _HIERARCHICAL_FITS[trust_metric](*args, **kwargs)
    result["coordinate_system"] = "hierarchical_event_tree"
    result["hierarchical_trust_metric"] = trust_metric
    result["production_source_sha256"] = PRODUCTION_SOURCE_SHA256
    return result


def fit_genewise_native(*args, **kwargs):
    """Unmodified production callable used as the control arm."""
    return _production.fit_genewise(*args, **kwargs)
