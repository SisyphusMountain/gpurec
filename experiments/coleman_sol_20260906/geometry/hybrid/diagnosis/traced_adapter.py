"""Fail-closed, callback-instrumented native and hierarchical fit compilers.

No production callable or model method is replaced. Both variants are compiled in memory from the
current production function. The hierarchical variant delegates its algorithm transformation to
the already-audited experiment adapter, adding trace sites only after that transformation finishes.
"""
from __future__ import annotations

import hashlib
import inspect

import torch

import gpurec.fit.genewise_fit as _production
from hierarchical_adapter import theta_to_phi as _trace_phi_impl
from hierarchical_adapter import transform_gradient as _trace_gradient_phi_impl


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"trace adapter expected one {label} site, found {count}")
    return source.replace(old, new)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return value


class FitTrace:
    def __init__(self, coordinate: str):
        self.coordinate = coordinate
        self.evaluations: list[dict] = []
        self.proposals: list[dict] = []

    def evaluation(self, **row):
        self.evaluations.append({key: _cpu(value) for key, value in row.items()})

    def proposal(self, **row):
        self.proposals.append({key: _cpu(value) for key, value in row.items()})

    def state_dict(self) -> dict:
        return {
            "coordinate": self.coordinate,
            "evaluations": self.evaluations,
            "proposals": self.proposals,
        }


def _instrument(source: str, coordinate: str) -> str:
    source = _replace_once(source,
'''                pred_a = pred_dec.index_select(0, active)
''',
'''                pred_a = pred_dec.index_select(0, active)
                _trace_radius_before = radius.index_select(0, active).clone()
''', "pre-ratio radius")

    source = _replace_once(source,
'''                free = staged_free((~fixed).to(dtype), n_steps)

                refresh_due = refresh_due or since_exact >= hessian_refresh
''',
'''                free = staged_free((~fixed).to(dtype), n_steps)
                _trace_callback.evaluation(
                    coordinate=_trace_coordinate, global_family_ids=active, live=live,
                    settled=settled, family_clades=clades, theta=sub, gradient_theta=g,
                    phi=_trace_phi(sub), gradient_phi=_trace_gradient_phi(_trace_phi(sub), g),
                    loss_bits=lv, curvature=B_fam.index_select(0, active),
                    radius_before=_trace_radius_before, radius_after=r_act,
                    fixed=fixed, free=free, predicted_previous=pred_a, pending=pending,
                    actual_decrease=actual, ratio=ratio, reject=reject,
                    refresh_due_before=bool(refresh_due), since_exact=int(since_exact),
                    pi=int(pi_cur), tier_index=int(pi_idx), iteration=int(it),
                    global_step=int(n_steps), gradient_pass=int(n_passes),
                    model_builds=int(n_builds), model_rebuilds=int(n_rebuilds))

                refresh_due = refresh_due or since_exact >= hessian_refresh
''', "evaluation callback")

    if coordinate == "native":
        source = _replace_once(source,
'''                pred_dec.index_copy_(0, active, torch.where(live, predicted, torch.zeros_like(predicted)))
                last_capped.index_copy_(0, active, capped.squeeze(1) & live)
''',
'''                _trace_callback.proposal(
                    coordinate=_trace_coordinate, global_family_ids=active, live=live,
                    settled=settled, family_clades=clades, theta=sub, gradient_theta=g,
                    phi=_trace_phi(sub), gradient_phi=_trace_gradient_phi(_trace_phi(sub), g),
                    curvature=B_fam.index_select(0, active), radius=radius.index_select(0, active),
                    fixed=fixed, working_fixed=fixed, free=free,
                    proposed_direction=delta, radius_scaled_direction=step,
                    applied_tangent=step, applied_theta=new_sub, applied_native=applied,
                    regularized_curvature=Hred, predicted=predicted,
                    radius_capped=capped.squeeze(1), exact_refresh=bool(exact_curvature_now),
                    pi=int(pi_cur), tier_index=int(pi_idx), iteration=int(it),
                    global_step=int(n_steps), gradient_pass=int(n_passes), mu=float(mu))
                pred_dec.index_copy_(0, active, torch.where(live, predicted, torch.zeros_like(predicted)))
                last_capped.index_copy_(0, active, capped.squeeze(1) & live)
''', "native proposal callback")
    elif coordinate == "hierarchical_native_metric":
        source = _replace_once(source,
'''                pred_dec.index_copy_(0, active, torch.where(live, predicted, torch.zeros_like(predicted)))
                last_capped.index_copy_(0, active, capped & live)
''',
'''                _trace_callback.proposal(
                    coordinate=_trace_coordinate, global_family_ids=active, live=live,
                    settled=settled, family_clades=clades, theta=sub, gradient_theta=g,
                    phi=phi, gradient_phi=g_phi,
                    curvature=B_fam.index_select(0, active), radius=radius.index_select(0, active),
                    fixed=fixed, working_fixed=working_fixed,
                    proposed_direction=direction, radius_scaled_direction=applied_tangent,
                    applied_tangent=applied_tangent, lifted_chord=chord,
                    applied_theta=new_sub, applied_native=new_sub-sub,
                    regularized_curvature=Hred, projected_gradient=g_face,
                    predicted=predicted, radius_capped=capped,
                    exact_refresh=bool(exact_curvature_now),
                    pi=int(pi_cur), tier_index=int(pi_idx), iteration=int(it),
                    global_step=int(n_steps), gradient_pass=int(n_passes), mu=float(mu))
                pred_dec.index_copy_(0, active, torch.where(live, predicted, torch.zeros_like(predicted)))
                last_capped.index_copy_(0, active, capped & live)
''', "hierarchical proposal callback")
    else:
        raise ValueError(coordinate)
    return source


def compile_traced_native() -> tuple[object, FitTrace, str]:
    source = inspect.getsource(_production.fit_genewise)
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    source = _instrument(source, "native")
    trace = FitTrace("native")
    namespace = dict(vars(_production))
    namespace.update(
        _trace_callback=trace, _trace_coordinate="native",
        _trace_phi=_trace_phi_impl, _trace_gradient_phi=_trace_gradient_phi_impl,
    )
    exec(compile(source, f"<traced-native:{source_hash[:12]}>", "exec"), namespace)
    return namespace["fit_genewise"], trace, source_hash


def compile_traced_hierarchical_native_metric() -> tuple[object, FitTrace, str]:
    # Import by path through the experiment directory; run_trace.py installs that path explicitly.
    import source_adapter as hierarchy

    original_replace = hierarchy._replace_once

    def tracing_replace(source, old, new, label):
        result = original_replace(source, old, new, label)
        if label == "ray diagnostic result":  # final transformation site before compilation
            result = _instrument(result, "hierarchical_native_metric")
        return result

    hierarchy._replace_once = tracing_replace
    try:
        fit, source_hash = hierarchy._compile_hierarchical_fit("native")
    finally:
        hierarchy._replace_once = original_replace
    trace = FitTrace("hierarchical_native_metric")
    fit.__globals__["_trace_callback"] = trace
    fit.__globals__["_trace_coordinate"] = "hierarchical_native_metric"
    fit.__globals__["_trace_phi"] = _trace_phi_impl
    fit.__globals__["_trace_gradient_phi"] = _trace_gradient_phi_impl
    return fit, trace, source_hash
