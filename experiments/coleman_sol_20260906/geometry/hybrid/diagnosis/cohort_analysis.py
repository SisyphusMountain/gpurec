"""Interior/bound cohort and BFGS-guard decomposition for the saved diagnostic trace."""
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

from hierarchical_adapter import theta_to_phi, transform_gradient


def _at_bound(theta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    lo, hi = math.log2(1e-6), 1.0
    return ((theta <= lo + eps) | (theta >= hi - eps)).any(dim=1)


def _changed_point_work(arm: dict, family_clades: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    changed_clades = torch.zeros_like(family_clades, dtype=torch.long)
    changed_evaluations = torch.zeros_like(family_clades, dtype=torch.long)
    previous = {}
    for row in arm["trace"]["evaluations"]:
        for local, gid in enumerate(row["global_family_ids"].tolist()):
            if not bool(row["live"][local]):
                continue
            theta = row["theta"][local]
            changed = gid not in previous or not torch.equal(theta, previous[gid])
            if changed:
                changed_evaluations[gid] += 1
                changed_clades[gid] += family_clades[gid]
            previous[gid] = theta.clone()
    return changed_clades, changed_evaluations


def _event_cohort(arm: dict, cohort: torch.Tensor) -> dict:
    radius_events = radius_clades = rejection_events = rejection_clades = 0
    shrink_events = shrink_clades = 0
    for row in arm["trace"]["proposals"]:
        ids, clades = row["global_family_ids"].long(), row["family_clades"].long()
        select = row["live"].bool() & cohort[ids]
        radius = select & row["radius_capped"].bool()
        radius_events += int(radius.sum()); radius_clades += int(clades[radius].sum())
    for row in arm["trace"]["evaluations"]:
        ids, clades = row["global_family_ids"].long(), row["family_clades"].long()
        select = row["live"].bool() & cohort[ids]
        reject = select & row["reject"].bool()
        shrink = select & (row["radius_after"] < row["radius_before"] - 1e-7)
        rejection_events += int(reject.sum()); rejection_clades += int(clades[reject].sum())
        shrink_events += int(shrink.sum()); shrink_clades += int(clades[shrink].sum())
    return {
        "radius_cut_events": radius_events, "radius_cut_clades": radius_clades,
        "rejection_events": rejection_events, "rejection_clades": rejection_clades,
        "radius_shrink_events": shrink_events, "radius_shrink_clades": shrink_clades,
    }


def _bfgs_guards(name: str, arm: dict, cohort: torch.Tensor) -> dict:
    previous: dict[int, dict] = {}
    totals = {key: 0 for key in (
        "attempt_events", "attempt_clades", "zero_secant_events", "zero_secant_clades",
        "curvature_guard_fail_events", "curvature_guard_fail_clades",
        "accepted_events", "accepted_clades",
    )}
    for row in arm["trace"]["evaluations"]:
        scheduled_refresh = bool(row["refresh_due_before"]) or int(row["since_exact"]) >= 15
        ids, clades = row["global_family_ids"].long(), row["family_clades"].long()
        for local, gid in enumerate(ids.tolist()):
            if gid not in previous or scheduled_refresh or not bool(cohort[gid]):
                continue
            old = previous[gid]
            if name == "native":
                point = row["theta"][local]
                gradient = row["gradient_theta"][local]
                free = row["free"][local] * old["free"]
            else:
                point = theta_to_phi(row["theta"][local:local+1])[0]
                gradient = transform_gradient(
                    point, row["gradient_theta"][local],
                )
                free = torch.ones_like(point)
            s = (point - old["point"]) * free
            y = (gradient - old["gradient"]) * free
            b = row["curvature"][local]
            bs = (b @ s) * free
            sy = s @ y
            sbs = s @ bs
            zero = float(s.norm()) == 0.0
            ok = bool(sy > 1e-10 * s.norm() * y.norm()) and bool(sbs > 0)
            prefix = "zero_secant" if zero else ("accepted" if ok else "curvature_guard_fail")
            totals["attempt_events"] += 1; totals["attempt_clades"] += int(clades[local])
            totals[f"{prefix}_events"] += 1; totals[f"{prefix}_clades"] += int(clades[local])

        # Production stores the evaluated point unless the ratio test rejects it, in which case
        # it restores the previous state. This update mirrors that state transition.
        for local, gid in enumerate(ids.tolist()):
            if bool(row["reject"][local]) and gid in previous:
                continue
            if name == "native":
                point = row["theta"][local].clone()
                gradient = row["gradient_theta"][local].clone()
                free = row["free"][local].clone()
            else:
                point = theta_to_phi(row["theta"][local:local+1])[0]
                gradient = transform_gradient(point, row["gradient_theta"][local])
                free = torch.ones_like(point)
            previous[gid] = {"point": point, "gradient": gradient, "free": free}
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    trace = torch.load(args.trace, map_location="cpu", weights_only=False)
    clades = trace["family_clades"].long()
    n = len(clades)
    ever_bound = torch.zeros(n, dtype=torch.bool)
    for arm in trace["arms"].values():
        for row in arm["trace"]["evaluations"]:
            ids = row["global_family_ids"].long()
            ever_bound[ids] |= _at_bound(row["theta"])
    interior = ~ever_bound

    changed = {}
    changed_evals = {}
    for name, arm in trace["arms"].items():
        changed[name], changed_evals[name] = _changed_point_work(arm, clades)
    delta = changed["hierarchical_native_metric"] - changed["native"]
    delta_evals = changed_evals["hierarchical_native_metric"] - changed_evals["native"]
    heavy = torch.argsort(delta, descending=True)[:10]
    report = {
        "schema": "gpurec.hybrid_cohort_analysis.v1",
        "trace": args.trace,
        "ever_bound_families": int(ever_bound.sum()),
        "always_interior_families": int(interior.sum()),
        "changed_point_penalty_clades": int(delta.sum()),
        "interior_changed_point_penalty_clades": int(delta[interior].sum()),
        "interior_penalty_fraction": float(delta[interior].sum()) / float(delta.sum()),
        "heaviest_hierarchical_penalties": [
            {"index": int(i), "family": trace["paths"][int(i)], "clades": int(clades[i]),
             "extra_changed_evaluations": int(delta_evals[i]),
             "extra_changed_point_clades": int(delta[i]), "ever_bound": bool(ever_bound[i])}
            for i in heavy
        ],
        "arms": {},
    }
    for name, arm in trace["arms"].items():
        report["arms"][name] = {
            "always_interior_events": _event_cohort(arm, interior),
            "ever_bound_events": _event_cohort(arm, ever_bound),
            "bfgs_guards_all": _bfgs_guards(name, arm, torch.ones(n, dtype=torch.bool)),
            "bfgs_guards_interior": _bfgs_guards(name, arm, interior),
        }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
