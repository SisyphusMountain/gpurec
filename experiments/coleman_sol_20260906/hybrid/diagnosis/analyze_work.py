"""Disaggregate diagnostic gradient work without changing either optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def analyze(saved: dict) -> dict:
    full = int(saved["family_clades"].sum())
    result = {"full_dataset_clades": full, "arms": {}}
    for name, arm in saved["arms"].items():
        last = {}
        work = dict(frozen_resident=0, live_same_point=0, live_new_point=0)
        after_build = 0
        evaluations = arm["trace"]["evaluations"]
        frozen_by_pass = []
        for row in evaluations:
            frozen_by_pass.append({
                "iteration": row["iteration"],
                "resident_clades": int(row["family_clades"].sum()),
                "frozen_clades": int(row["family_clades"][~row["live"]].sum()),
                "resident_families": len(row["global_family_ids"]),
                "live_families": int(row["live"].sum()),
            })
            for local, family in enumerate(row["global_family_ids"].tolist()):
                clades = int(row["family_clades"][local])
                theta = row["theta"][local]
                if not bool(row["live"][local]):
                    work["frozen_resident"] += clades
                elif family in last and torch.equal(theta, last[family][0]):
                    work["live_same_point"] += clades
                    if row["model_builds"] > last[family][1]:
                        after_build += clades
                else:
                    work["live_new_point"] += clades
                last[family] = (theta.clone(), row["model_builds"])
        charged = sum(int(row["clades"]) for row in arm["result"]["gradient_work"])
        assert sum(work.values()) == charged, (work, charged)
        proposals = arm["trace"]["proposals"]
        result["arms"][name] = {
            "gradient_clades": work,
            "gradient_equivalents": {key: value / full for key, value in work.items()},
            "same_point_after_rebuild_clades": after_build,
            "continuation_total_clades": charged,
            "including_common_em_equivalents": charged / full + 2,
            "live_family_proposals": sum(int(row["live"].sum()) for row in proposals),
            "frozen_residency_by_pass": frozen_by_pass,
        }
    native = result["arms"]["native"]["gradient_clades"]
    hierarchy = result["arms"]["hierarchical_native_metric"]["gradient_clades"]
    result["hierarchical_minus_native_clades"] = {
        key: hierarchy[key] - native[key] for key in native
    }
    result["note"] = (
        "Traced diagnostic trajectories, not new timing benchmarks. New-point includes the "
        "first endpoint evaluation and rejected trials; same-point is exact equality with the "
        "previous evaluation for that family, not a claim that repeated FP32 gradients are "
        "bit-identical. Shared EM is identical, and Hessian probes are separate."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = analyze(torch.load(args.trace, map_location="cpu", weights_only=False))
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for name, arm in report["arms"].items():
        print(name, arm["gradient_clades"], arm["gradient_equivalents"])
    print("difference", report["hierarchical_minus_native_clades"])


if __name__ == "__main__":
    main()
