"""Offline CPU analysis of the saved 200-family native/hierarchical step trace."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[4]
HYBRID = REPO / "experiments/coleman_sol_20260906/geometry/hybrid"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_endpoint_geometry import exact_linear_native_trust_step, quantiles  # noqa: E402
from hierarchical_adapter import jacobian, phi_to_theta  # noqa: E402


def quadratic_gain(
    gradient: torch.Tensor, curvature: torch.Tensor, direction: torch.Tensor,
) -> torch.Tensor:
    return -(
        (gradient * direction).sum(dim=1)
        + 0.5 * torch.einsum("fi,fij,fj->f", direction, curvature, direction)
    )


def nonlinear_radius_scale(
    phi: torch.Tensor, theta: torch.Tensor, direction: torch.Tensor, radius: torch.Tensor,
) -> torch.Tensor:
    """Largest alpha in [0,1] satisfying the exact native displacement norm on this ray."""
    full_norm = (phi_to_theta(phi + direction) - theta).norm(dim=1)
    low = torch.where(full_norm <= radius, torch.ones_like(radius), torch.zeros_like(radius))
    high = torch.ones_like(radius)
    for _ in range(64):
        middle = 0.5 * (low + high)
        trial_norm = (phi_to_theta(phi + middle[:, None] * direction) - theta).norm(dim=1)
        feasible = trial_norm <= radius
        low = torch.where(feasible, middle, low)
        high = torch.where(feasible, high, middle)
    return low


def exact_step_with_equalities(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    metric: torch.Tensor,
    jac: torch.Tensor,
    equalities: torch.Tensor,
    radius: float,
    mu: float,
) -> torch.Tensor:
    """Exact generalized trust step in the nullspace of selected native faces."""
    rows = jac[equalities]
    if len(rows):
        _u, singular, vh = torch.linalg.svd(rows, full_matrices=True)
        rank = int((singular > 1e-10).sum())
        null = vh[rank:].T
    else:
        null = torch.eye(3, dtype=torch.float64)
    dimension = null.shape[1]
    if dimension == 0:
        return torch.zeros(3, dtype=torch.float64)
    reduced_h = null.T @ hessian @ null
    reduced_g = null.T @ gradient
    reduced_metric = null.T @ metric @ null
    values_g, vectors_g = torch.linalg.eigh(reduced_metric)
    metric_inverse_half = (
        vectors_g * values_g.rsqrt().unsqueeze(0)
    ) @ vectors_g.T
    whitened_h = metric_inverse_half @ reduced_h @ metric_inverse_half
    values_h, vectors_h = torch.linalg.eigh(whitened_h)
    whitened_h = (
        vectors_h * values_h.clamp_min(mu).unsqueeze(0)
    ) @ vectors_h.T
    whitened_g = metric_inverse_half @ reduced_g
    identity = torch.eye(dimension, dtype=torch.float64)

    def solve(multiplier: float) -> torch.Tensor:
        return -torch.linalg.solve(
            whitened_h + multiplier * identity, whitened_g,
        )

    step = solve(0.0)
    if float(step.norm()) > radius:
        low, high = 0.0, 1.0
        while float(solve(high).norm()) > radius:
            high *= 2.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            if float(solve(middle).norm()) > radius:
                low = middle
            else:
                high = middle
        step = solve(high)
    return null @ metric_inverse_half @ step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        default=REPO / "experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200.pt",
        type=Path,
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    trace = torch.load(args.trace, map_location="cpu", weights_only=False)
    proposals = trace["arms"]["hierarchical_native_metric"]["trace"]["proposals"]
    evaluations = trace["arms"]["hierarchical_native_metric"]["trace"]["evaluations"]

    all_cosine = []
    all_gain_ratio = []
    all_actual_gain_ratio = []
    all_gain_ratio_nonlinear = []
    all_metric_condition = []
    all_metric_correlation = []
    all_current_alpha = []
    current_linear_radius_excess = 0
    all_records = []
    counts = {
        "live_proposals": 0,
        "current_face_rows": 0,
        "working_face_promotions": 0,
        "working_face_promoted_rows": 0,
        "current_full_direction_box_cross_rows": 0,
        "interior_no_current_box_cross_rows": 0,
        "counterfactual_exact_box_cross_rows": 0,
        "comparison_rows": 0,
        "promoted_rows_where_exact_G_step_is_already_tangent_cone_feasible": 0,
    }
    by_step = []
    lo, hi = math.log2(1e-6), 1.0

    for proposal in proposals:
        live = proposal["live"].bool()
        fixed = proposal["fixed"].bool()
        working = proposal["working_fixed"].bool()
        extra = working & ~fixed
        phi = proposal["phi"].double()
        theta = proposal["theta"].double()
        gradient = proposal["gradient_phi"].double()
        curvature = proposal["curvature"].double()
        radius = proposal["radius"].double()
        current_full = proposal["proposed_direction"].double()
        current_applied = proposal["applied_tangent"].double()
        metric = jacobian(phi).transpose(-1, -2) @ jacobian(phi)

        current_full_theta = phi_to_theta(phi + current_full)
        current_box_cross = ((current_full_theta < lo) | (current_full_theta > hi)).any(dim=1)
        interior = ~fixed.any(dim=1) & ~working.any(dim=1)
        eligible = live & interior & ~current_box_cross

        counts["live_proposals"] += int(live.sum())
        counts["current_face_rows"] += int((live & fixed.any(dim=1)).sum())
        counts["working_face_promotions"] += int((extra & live[:, None]).sum())
        counts["working_face_promoted_rows"] += int((live & extra.any(dim=1)).sum())
        counts["current_full_direction_box_cross_rows"] += int((live & current_box_cross).sum())
        counts["interior_no_current_box_cross_rows"] += int(eligible.sum())

        # Check whether changing only the metric/trust solve would have made each observed
        # step-only promotion unnecessary. Authoritative KKT faces stay equalities; other current
        # faces retain their one-sided tangent-cone sign.
        for row in (live & extra.any(dim=1)).nonzero(as_tuple=True)[0].tolist():
            exact_face_step = exact_step_with_equalities(
                curvature[row], gradient[row], metric[row], jacobian(phi[row]), fixed[row],
                float(radius[row]), float(proposal["mu"]),
            )
            native_linear = jacobian(phi[row]) @ exact_face_step
            at_lower = theta[row] <= lo + 1e-6
            at_upper = theta[row] >= hi - 1e-6
            feasible = bool(
                ((~at_lower) | (native_linear >= -1e-10)).all()
                and ((~at_upper) | (native_linear <= 1e-10)).all()
            )
            counts[
                "promoted_rows_where_exact_G_step_is_already_tangent_cone_feasible"
            ] += int(feasible)

        if not bool(eligible.any()):
            continue
        rows = eligible.nonzero(as_tuple=True)[0]
        # The endpoint artifact has one common radius, but the live fit adapts it per family.
        exact_rows, convexified_rows = [], []
        for row in rows.tolist():
            d, hc = exact_linear_native_trust_step(
                curvature[row:row + 1], gradient[row:row + 1], metric[row:row + 1],
                radius=float(radius[row]), mu=float(proposal["mu"]),
            )
            exact_rows.append(d[0])
            convexified_rows.append(hc[0])
        exact = torch.stack(exact_rows)
        convexified = torch.stack(convexified_rows)
        exact_theta = phi_to_theta(phi[rows] + exact)
        exact_box_cross = ((exact_theta < lo) | (exact_theta > hi)).any(dim=1)
        counts["counterfactual_exact_box_cross_rows"] += int(exact_box_cross.sum())
        take = ~exact_box_cross
        if not bool(take.any()):
            continue
        rows = rows[take]
        exact = exact[take]
        convexified = convexified[take]
        cur = current_applied[rows]
        met = metric[rows]
        grad = gradient[rows]
        rad = radius[rows]
        ph = phi[rows]
        th = theta[rows]

        cur_norm = torch.sqrt(torch.einsum("fi,fij,fj->f", cur, met, cur))
        exact_norm = torch.sqrt(torch.einsum("fi,fij,fj->f", exact, met, exact))
        cosine = torch.einsum("fi,fij,fj->f", cur, met, exact) / (
            cur_norm * exact_norm
        ).clamp_min(1e-30)
        exact_gain = quadratic_gain(grad, convexified, exact)
        # The recorded step obeys the exact nonlinear native radius, not necessarily its local
        # linearization. Scale it only for the strict same-feasible-set full-G comparison.
        current_linear_scale = torch.minimum(
            torch.ones_like(cur_norm), rad / cur_norm.clamp_min(1e-30),
        )
        current_linear_feasible = cur * current_linear_scale[:, None]
        current_linear_radius_excess += int((cur_norm > rad * (1.0 + 1e-10)).sum())
        current_gain = quadratic_gain(grad, convexified, current_linear_feasible)
        current_actual_gain = quadratic_gain(grad, convexified, cur)
        gain_ratio = current_gain / exact_gain.clamp_min(1e-30)
        actual_gain_ratio = current_actual_gain / exact_gain.clamp_min(1e-30)
        exact_nonlinear_scale = nonlinear_radius_scale(ph, th, exact, rad)
        exact_nonlinear = exact * exact_nonlinear_scale[:, None]
        exact_nonlinear_gain = quadratic_gain(grad, convexified, exact_nonlinear)
        gain_ratio_nonlinear = current_actual_gain / exact_nonlinear_gain.clamp_min(1e-30)

        values_g = torch.linalg.eigvalsh(met)
        values_h, vectors_h = torch.linalg.eigh(curvature[rows])
        metric_h = vectors_h.transpose(-1, -2) @ met @ vectors_h
        diagonal_h = torch.diagonal(metric_h, dim1=-2, dim2=-1)
        normalized_h = metric_h / torch.sqrt(
            diagonal_h.unsqueeze(-1) * diagonal_h.unsqueeze(-2)
        )
        mask = ~torch.eye(3, dtype=torch.bool)
        correlation = normalized_h[:, mask].abs().amax(dim=1)
        proposed_norm = proposal["proposed_direction"][rows].double().norm(dim=1)
        applied_norm = proposal["applied_tangent"][rows].double().norm(dim=1)
        current_alpha = applied_norm / proposed_norm.clamp_min(1e-30)

        ids = proposal["global_family_ids"][rows]
        for local, global_id, c, gr, grn, ca in zip(
            rows.tolist(), ids.tolist(), cosine.tolist(), gain_ratio.tolist(),
            gain_ratio_nonlinear.tolist(), current_alpha.tolist(),
        ):
            all_records.append({
                "global_family_id": global_id,
                "global_step": int(proposal["global_step"]),
                "pi": int(proposal["pi"]),
                "row": local,
                "physical_metric_cosine": c,
                "gain_ratio_linear_exact": gr,
                "gain_ratio_nonlinear_rescaled_exact": grn,
                "current_ray_alpha": ca,
            })
        all_cosine.append(cosine)
        all_gain_ratio.append(gain_ratio)
        all_actual_gain_ratio.append(actual_gain_ratio)
        all_gain_ratio_nonlinear.append(gain_ratio_nonlinear)
        all_metric_condition.append(values_g[:, -1] / values_g[:, 0])
        all_metric_correlation.append(correlation)
        all_current_alpha.append(current_alpha)
        counts["comparison_rows"] += len(rows)
        by_step.append({
            "global_step": int(proposal["global_step"]),
            "pi": int(proposal["pi"]),
            "rows": len(rows),
            "gain_ratio_median": float(gain_ratio.median()),
            "gain_ratio_min": float(gain_ratio.min()),
            "metric_cosine_median": float(cosine.median()),
            "current_ray_alpha_min": float(current_alpha.min()),
        })

    cosine = torch.cat(all_cosine)
    gain_ratio = torch.cat(all_gain_ratio)
    actual_gain_ratio = torch.cat(all_actual_gain_ratio)
    gain_ratio_nonlinear = torch.cat(all_gain_ratio_nonlinear)
    metric_condition = torch.cat(all_metric_condition)
    metric_correlation = torch.cat(all_metric_correlation)
    current_alpha = torch.cat(all_current_alpha)
    worst = sorted(all_records, key=lambda row: row["gain_ratio_linear_exact"])[:12]
    family_rows: dict[int, list[dict]] = {}
    for row in all_records:
        family_rows.setdefault(row["global_family_id"], []).append(row)
    family_summary = []
    for global_id, rows in family_rows.items():
        ratios = [row["gain_ratio_linear_exact"] for row in rows]
        family_summary.append({
            "global_family_id": global_id,
            "min_gain_ratio": min(ratios),
            "rows_below_0_99": sum(value < 0.99 for value in ratios),
            "rows_below_0_90": sum(value < 0.90 for value in ratios),
            "affected_steps": [
                row["global_step"] for row in rows
                if row["gain_ratio_linear_exact"] < 0.99
            ],
        })
    family_summary.sort(key=lambda row: (row["min_gain_ratio"], -row["rows_below_0_99"]))
    focus_ids = {16, 83, 101, 169, 177}

    pending_ratio = []
    rejects = 0
    pending = 0
    for evaluation in evaluations:
        live = evaluation["live"].bool()
        judged = live & evaluation["pending"].bool()
        pending += int(judged.sum())
        rejects += int((judged & evaluation["reject"].bool()).sum())
        if bool(judged.any()):
            pending_ratio.append(evaluation["ratio"][judged].double())

    output = {
        "schema": "gpurec.em_hybrid_trace_geometry_diagnosis.v1",
        "trace": str(args.trace),
        "scope": (
            "Counterfactual exact-G comparison uses live rows with no current fixed/working face "
            "and neither the recorded full proposal nor exact linearized proposal crossing the box."
        ),
        "counts": counts,
        "actual_saved_step_metric": {
            "G_condition": quantiles(metric_condition),
            "max_normalized_G_offdiagonal_in_raw_B_eigenbasis": quantiles(metric_correlation),
            "physical_metric_cosine_current_vs_exact_G": quantiles(cosine),
            "raw_B_convexified_quadratic_gain_ratio_current_vs_exact_G": quantiles(gain_ratio),
            "recorded_actual_step_gain_ratio_vs_exact_linear_G": quantiles(actual_gain_ratio),
            "gain_ratio_vs_exact_G_after_exact_step_nonlinear_radius_rescale": quantiles(
                gain_ratio_nonlinear
            ),
            "current_ray_alpha": quantiles(current_alpha),
            "recorded_steps_outside_local_linear_G_radius": current_linear_radius_excess,
            "rows_below_0_90_exact_gain": int((gain_ratio < 0.90).sum()),
            "rows_below_0_75_exact_gain": int((gain_ratio < 0.75).sum()),
        },
        "recorded_trust_ratio": {
            "pending_judgments": pending,
            "rejections": rejects,
            "actual_over_recorded_predicted": quantiles(torch.cat(pending_ratio)),
        },
        "by_step": by_step,
        "worst_exact_G_counterfactual_rows": worst,
        "families_most_affected_by_metric_approximation": family_summary[:20],
        "root_identified_heavy_slower_families": [
            row for row in family_summary if row["global_family_id"] in focus_ids
        ],
        "working_set_note": (
            "Only counts promotions. A release-optimality test needs the inequality face QP; "
            "the existing trace establishes whether this can be material by its frequency."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
