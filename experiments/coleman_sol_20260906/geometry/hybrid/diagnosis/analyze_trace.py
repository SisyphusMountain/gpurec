"""Summarize scheduling, bound, trust, rejection, and eigen-regularization trace events."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


THIS = Path(__file__).resolve().parent
HYBRID = THIS.parent
if str(HYBRID) not in sys.path:
    sys.path.insert(0, str(HYBRID))

from hierarchical_adapter import face_step_model, jacobian, phi_to_theta


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float | None:
    values, weights = values.double().flatten(), weights.double().flatten()
    valid = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if not bool(valid.any()):
        return None
    values, weights = values[valid], weights[valid]
    order = values.argsort()
    values, weights = values[order], weights[order]
    threshold = q * weights.sum()
    index = int(torch.searchsorted(weights.cumsum(0), threshold, right=False).clamp(max=len(values)-1))
    return float(values[index])


def _event_summary(mask: torch.Tensor, clades: torch.Tensor, full_clades: int) -> dict:
    mask = mask.bool()
    return {
        "events": int(mask.sum()),
        "family_clades": int(clades[mask].sum()),
        "full_clade_equivalents": float(clades[mask].sum()) / full_clades,
        "families": int(torch.unique(torch.nonzero(mask, as_tuple=True)[0]).numel()) if mask.ndim > 1 else None,
    }


def _analyse_arm(name: str, arm: dict, family_clades: torch.Tensor, paths: list[str]) -> dict:
    evaluations = arm["trace"]["evaluations"]
    proposals = arm["trace"]["proposals"]
    full_clades = int(family_clades.sum())

    resident_work = live_work = settled_work = 0
    same_theta_live_work = changed_theta_live_work = 0
    previous_evaluated_theta: dict[int, torch.Tensor] = {}
    per_pass = []
    last_live_pass = torch.full((len(paths),), -1, dtype=torch.long)
    for row in evaluations:
        ids, clades = row["global_family_ids"].long(), row["family_clades"].long()
        live, settled = row["live"].bool(), row["settled"].bool()
        resident = int(clades.sum())
        live_clades = int(clades[live].sum())
        settled_clades = int(clades[settled].sum())
        resident_work += resident
        live_work += live_clades
        settled_work += settled_clades
        for local, global_id in enumerate(ids.tolist()):
            if not bool(live[local]):
                continue
            current_theta = row["theta"][local]
            if global_id in previous_evaluated_theta and torch.equal(
                    current_theta, previous_evaluated_theta[global_id]):
                same_theta_live_work += int(clades[local])
            else:
                changed_theta_live_work += int(clades[local])
            previous_evaluated_theta[global_id] = current_theta.clone()
        last_live_pass[ids[live]] = int(row["gradient_pass"])
        per_pass.append({
            "gradient_pass": int(row["gradient_pass"]), "iteration": int(row["iteration"]),
            "resident_families": int(ids.numel()), "live_families": int(live.sum()),
            "resident_clades": resident, "live_clades": live_clades,
            "settled_resident_clades": settled_clades,
            "settled_resident_fraction": settled_clades / resident if resident else 0.0,
            "model_builds": int(row["model_builds"]), "model_rebuilds": int(row["model_rebuilds"]),
        })

    freeze_valid = last_live_pass >= 0
    freeze_values = last_live_pass[freeze_valid]
    freeze_weights = family_clades[freeze_valid]
    per_family_freeze = [
        {"index": i, "family": paths[i], "clades": int(family_clades[i]),
         "last_live_gradient_pass": int(last_live_pass[i])}
        for i in range(len(paths))
    ]

    working_events = authoritative_events = ray_cut_events = radius_cut_events = box_cut_events = 0
    working_clades = authoritative_clades = ray_cut_clades = radius_cut_clades = box_cut_clades = 0
    both_cut_events = both_cut_clades = 0
    coordinate_working = torch.zeros(3, dtype=torch.long)
    coordinate_working_clades = torch.zeros(3, dtype=torch.long)
    negative_eigenvalues = sign_floor_eigenvalues = trust_floor_eigenvalues = 0
    negative_family_events = sign_floor_family_events = trust_floor_family_events = 0
    negative_clades = sign_floor_clades = trust_floor_clades = 0
    ray_fractions, ray_weights = [], []
    adjusted_ratios, adjusted_weights = [], []
    limited_positive_ratios, limited_positive_weights = [], []
    live_proposal_events = live_proposal_clades = 0

    for row in proposals:
        live = row["live"].bool()
        clades = row["family_clades"].long()
        fixed = row["fixed"].bool()
        working = row["working_fixed"].bool()
        additions = working & ~fixed & live[:, None]
        live_proposal_events += int(live.sum())
        live_proposal_clades += int(clades[live].sum())
        authoritative = fixed & live[:, None]
        working_events += int(additions.sum())
        working_clades += int((clades[:, None] * additions).sum())
        authoritative_events += int(authoritative.sum())
        authoritative_clades += int((clades[:, None] * authoritative).sum())
        coordinate_working += additions.sum(dim=0)
        coordinate_working_clades += (clades[:, None] * additions).sum(dim=0)

        proposal = row["proposed_direction"]
        applied_tangent = row["applied_tangent"]
        proposed_norm = proposal.norm(dim=1)
        applied_norm = applied_tangent.norm(dim=1)
        ray_fraction = torch.where(
            proposed_norm > 0, applied_norm / proposed_norm.clamp_min(torch.finfo(proposal.dtype).tiny),
            torch.ones_like(proposed_norm))
        cut = live & (ray_fraction < 1.0 - 2e-6)
        radius_cut = live & row["radius_capped"].bool()
        if name == "native":
            box_cut = live & ((row["applied_native"] - row["radius_scaled_direction"]).norm(dim=1) > 2e-6)
        else:
            raw = phi_to_theta(row["phi"] + proposal)
            held = torch.where(working, row["theta"], raw)
            lo, hi = math.log2(1e-6), 1.0
            box_cut = live & (((held < lo - 2e-6) | (held > hi + 2e-6)).any(dim=1))
        both = radius_cut & box_cut
        ray_cut_events += int(cut.sum()); ray_cut_clades += int(clades[cut].sum())
        radius_cut_events += int(radius_cut.sum()); radius_cut_clades += int(clades[radius_cut].sum())
        box_cut_events += int(box_cut.sum()); box_cut_clades += int(clades[box_cut].sum())
        both_cut_events += int(both.sum()); both_cut_clades += int(clades[both].sum())
        ray_fractions.append(ray_fraction[live]); ray_weights.append(clades[live])

        curvature = row["curvature"].double()
        radius = row["radius"].double()
        if name == "native":
            model = curvature
            gradient = row["gradient_theta"].double() * row["free"].double()
            scales = None
        else:
            phi = row["phi"].double()
            gtheta = row["gradient_theta"].double()
            gphi = row["gradient_phi"].double()
            _p, model, gradient = face_step_model(
                phi, gtheta, gphi, curvature, working,
            )
        eigenvalues, vectors = torch.linalg.eigh(model)
        gradient_eigen = torch.einsum("bij,bi->bj", vectors, gradient)
        if name == "native":
            scales = torch.ones_like(eigenvalues)
        else:
            scales = torch.einsum("bki,bij->bkj", jacobian(phi), vectors).norm(dim=1)
        sign_floor = float(row["mu"]) * scales.square()
        trust_floor = gradient_eigen.abs() * scales / radius[:, None]
        base = torch.maximum(eigenvalues, sign_floor)
        adjusted = torch.maximum(base, trust_floor)
        tol = 1e-10 * torch.maximum(torch.ones_like(adjusted), adjusted.abs())
        negative = (eigenvalues < 0) & live[:, None]
        sign_limited = (sign_floor > eigenvalues + tol) & live[:, None]
        trust_limited = (trust_floor > base + tol) & live[:, None]
        negative_eigenvalues += int(negative.sum())
        sign_floor_eigenvalues += int(sign_limited.sum())
        trust_floor_eigenvalues += int(trust_limited.sum())
        neg_family = negative.any(dim=1); sign_family = sign_limited.any(dim=1); trust_family = trust_limited.any(dim=1)
        negative_family_events += int(neg_family.sum()); negative_clades += int(clades[neg_family].sum())
        sign_floor_family_events += int(sign_family.sum()); sign_floor_clades += int(clades[sign_family].sum())
        trust_floor_family_events += int(trust_family.sum()); trust_floor_clades += int(clades[trust_family].sum())
        positive = live[:, None] & (eigenvalues > 1e-12)
        adjusted_ratios.append((adjusted / eigenvalues.clamp_min(1e-12))[positive])
        adjusted_weights.append(clades[:, None].expand_as(adjusted)[positive])
        limited_positive = positive & trust_limited
        limited_positive_ratios.append(
            (adjusted / eigenvalues.clamp_min(1e-12))[limited_positive])
        limited_positive_weights.append(
            clades[:, None].expand_as(adjusted)[limited_positive])

    pending_count = reject_count = shrink_count = grow_count = 0
    pending_clades = reject_clades = shrink_clades = grow_clades = 0
    ratios, ratio_weights = [], []
    for row in evaluations:
        live, clades = row["live"].bool(), row["family_clades"].long()
        pending = row["pending"].bool() & live
        reject = row["reject"].bool() & live
        shrink = live & (row["radius_after"] < row["radius_before"] - 1e-7)
        grow = live & (row["radius_after"] > row["radius_before"] + 1e-7)
        pending_count += int(pending.sum()); pending_clades += int(clades[pending].sum())
        reject_count += int(reject.sum()); reject_clades += int(clades[reject].sum())
        shrink_count += int(shrink.sum()); shrink_clades += int(clades[shrink].sum())
        grow_count += int(grow.sum()); grow_clades += int(clades[grow].sum())
        ratios.append(row["ratio"][pending]); ratio_weights.append(clades[pending])

    concat = lambda rows, dtype=torch.float64: torch.cat(rows) if rows else torch.empty(0, dtype=dtype)
    ratio_values, ratio_w = concat(ratios), concat(ratio_weights, torch.long)
    ray_values, ray_w = concat(ray_fractions), concat(ray_weights, torch.long)
    adjust_values, adjust_w = concat(adjusted_ratios), concat(adjusted_weights, torch.long)
    limited_values, limited_w = concat(limited_positive_ratios), concat(
        limited_positive_weights, torch.long)
    return {
        "gradient_work": {
            "resident_clades": resident_work,
            "live_clades_immediate_removal_lower_bound": live_work,
            "settled_resident_clades": settled_work,
            "live_same_theta_remeasurement_clades": same_theta_live_work,
            "live_changed_point_clades": changed_theta_live_work,
            "resident_full_clade_equivalents": resident_work / full_clades,
            "immediate_removal_full_clade_equivalents": live_work / full_clades,
            "avoidable_settled_resident_equivalents_lower_bound": settled_work / full_clades,
            "same_theta_remeasurement_equivalents": same_theta_live_work / full_clades,
            "changed_point_equivalents": changed_theta_live_work / full_clades,
            "per_pass": per_pass,
        },
        "last_live_pass": {
            "clade_weighted_median": _weighted_quantile(freeze_values, freeze_weights, 0.5),
            "clade_weighted_p90": _weighted_quantile(freeze_values, freeze_weights, 0.9),
            "per_family": per_family_freeze,
        },
        "bounds": {
            "live_proposal_family_events": live_proposal_events,
            "live_proposal_clades": live_proposal_clades,
            "authoritative_fixed_coordinate_events": authoritative_events,
            "authoritative_fixed_coordinate_clades": authoritative_clades,
            "working_set_added_coordinate_events": working_events,
            "working_set_added_coordinate_clades": working_clades,
            "working_set_added_coordinate_equivalents": working_clades / full_clades,
            "working_set_additions_D_L_T": coordinate_working.tolist(),
            "working_set_addition_clades_D_L_T": coordinate_working_clades.tolist(),
            "ray_cut_family_events": ray_cut_events,
            "ray_cut_clades": ray_cut_clades,
            "radius_cut_family_events": radius_cut_events,
            "radius_cut_clades": radius_cut_clades,
            "box_cut_family_events": box_cut_events,
            "box_cut_clades": box_cut_clades,
            "both_radius_and_box_family_events": both_cut_events,
            "both_radius_and_box_clades": both_cut_clades,
            "ray_fraction_clade_weighted_median": _weighted_quantile(ray_values, ray_w, 0.5),
            "ray_fraction_clade_weighted_p10": _weighted_quantile(ray_values, ray_w, 0.1),
            "ray_fraction_min": float(ray_values.min()) if ray_values.numel() else None,
        },
        "trust_acceptance": {
            "pending_family_events": pending_count, "pending_clades": pending_clades,
            "rejected_family_events": reject_count, "rejected_clades": reject_clades,
            "radius_shrink_family_events": shrink_count, "radius_shrink_clades": shrink_clades,
            "radius_grow_family_events": grow_count, "radius_grow_clades": grow_clades,
            "ratio_clade_weighted_p10": _weighted_quantile(ratio_values, ratio_w, 0.1),
            "ratio_clade_weighted_median": _weighted_quantile(ratio_values, ratio_w, 0.5),
            "ratio_clade_weighted_p90": _weighted_quantile(ratio_values, ratio_w, 0.9),
        },
        "convexification": {
            "negative_eigenvalue_events": negative_eigenvalues,
            "negative_curvature_family_events": negative_family_events,
            "negative_curvature_family_clades": negative_clades,
            "sign_floor_eigenvalue_events": sign_floor_eigenvalues,
            "sign_floor_family_events": sign_floor_family_events,
            "sign_floor_family_clades": sign_floor_clades,
            "directional_trust_floor_eigenvalue_events": trust_floor_eigenvalues,
            "directional_trust_floor_family_events": trust_floor_family_events,
            "directional_trust_floor_family_clades": trust_floor_clades,
            "positive_eigen_adjustment_clade_weighted_median": _weighted_quantile(
                adjust_values, adjust_w, 0.5),
            "positive_eigen_adjustment_clade_weighted_p90": _weighted_quantile(
                adjust_values, adjust_w, 0.9),
            "trust_limited_positive_adjustment_clade_weighted_median": _weighted_quantile(
                limited_values, limited_w, 0.5),
            "trust_limited_positive_adjustment_clade_weighted_p90": _weighted_quantile(
                limited_values, limited_w, 0.9),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    trace = torch.load(args.trace, map_location="cpu", weights_only=False)
    family_clades = trace["family_clades"].long()
    report = {
        "schema": "gpurec.hybrid_diagnostic_analysis.v1",
        "trace": args.trace,
        "full_clades": int(family_clades.sum()),
        "shared_em_full_clade_equivalents": float(trace["shared_em_gradient_work"].__len__()),
        "arms": {},
    }
    for name, arm in trace["arms"].items():
        report["arms"][name] = _analyse_arm(name, arm, family_clades, trace["paths"])
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        name: {
            "work": row["gradient_work"], "bounds": row["bounds"],
            "trust": row["trust_acceptance"], "convexification": row["convexification"],
        } for name, row in report["arms"].items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
