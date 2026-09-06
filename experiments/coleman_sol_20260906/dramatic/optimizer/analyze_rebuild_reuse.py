"""Quantify the optimistic ceiling for reusing an unchanged-point gradient after replanning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[4]


def q(value: list[float]) -> dict[str, float] | None:
    if not value:
        return None
    tensor = torch.tensor(value, dtype=torch.float64)
    return {
        name: float(torch.quantile(tensor, point))
        for name, point in (("min", 0.0), ("p50", 0.5), ("p90", 0.9), ("max", 1.0))
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", type=Path,
        default=REPO / "experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200.pt",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.trace, map_location="cpu", weights_only=False)
    output = {
        "schema": "gpurec.rebuild_gradient_reuse_ceiling.v1",
        "trace": str(args.trace), "arms": {},
    }
    for name, arm in saved["arms"].items():
        previous: dict[int, dict] = {}
        clades = {"moved_or_first": 0, "same_point_after_same_tier_build": 0,
                  "settled_resident": 0}
        gradient_max_differences = []
        nll_differences = []
        same_rows = 0
        for evaluation in arm["trace"]["evaluations"]:
            ids = evaluation["global_family_ids"].tolist()
            for row, global_id in enumerate(ids):
                row_clades = int(evaluation["family_clades"][row])
                if bool(evaluation["settled"][row]):
                    clades["settled_resident"] += row_clades
                    continue
                current = {
                    "theta": evaluation["theta"][row],
                    "gradient": evaluation["gradient_theta"][row],
                    "nll": evaluation["loss_bits"][row],
                    "builds": int(evaluation["model_builds"]),
                    "pi": int(evaluation["pi"]),
                }
                old = previous.get(global_id)
                reusable = (
                    old is not None
                    and current["builds"] > old["builds"]
                    and current["pi"] == old["pi"]
                    and torch.equal(current["theta"], old["theta"])
                )
                if reusable:
                    clades["same_point_after_same_tier_build"] += row_clades
                    same_rows += 1
                    gradient_max_differences.append(float(
                        (current["gradient"] - old["gradient"]).abs().max()
                    ))
                    nll_differences.append(float(abs(current["nll"] - old["nll"])))
                else:
                    clades["moved_or_first"] += row_clades
                previous[global_id] = current
        total = sum(clades.values())
        shared = 2 * int(saved["family_clades"].sum())
        output["arms"][name] = {
            "continuation_clades": total,
            **clades,
            "same_point_rows": same_rows,
            "same_point_fraction_of_continuation_gradient_clades": (
                clades["same_point_after_same_tier_build"] / total
            ),
            "same_point_fraction_including_two_shared_EM_passes": (
                clades["same_point_after_same_tier_build"] / (total + shared)
            ),
            "same_point_gradient_abs_max_difference": q(gradient_max_differences),
            "same_point_nll_abs_difference_bits": q(nll_differences),
        }
    output["h100_projection"] = {
        "reference_total_seconds": 396.305,
        "reference_gradient_seconds": 307.0,
        "note": (
            "Optimistic projection applies the 200-family native same-point clade fraction to the "
            "entire H100 gradient phase. It is a ceiling, not a measured full-data saving."
        ),
    }
    native_fraction = output["arms"]["native"][
        "same_point_fraction_of_continuation_gradient_clades"
    ]
    projected = native_fraction * output["h100_projection"]["reference_gradient_seconds"]
    output["h100_projection"].update({
        "optimistic_seconds_saved": projected,
        "optimistic_total_seconds_after_reuse": 396.305 - projected,
        "optimistic_total_speedup_percent": 100.0 * projected / 396.305,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
