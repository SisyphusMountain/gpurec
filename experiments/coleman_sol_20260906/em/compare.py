"""Compare a candidate fit with a baseline, including per-family basin changes."""
from __future__ import annotations

import argparse
import json

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--material-bits", required=True, type=float)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    if baseline["paths"] != candidate["paths"]:
        raise ValueError("family lists differ")
    delta = candidate["nll_per_family"].double() - baseline["nll_per_family"].double()
    material = delta.abs() > args.material_bits
    rows = []
    for index in material.nonzero(as_tuple=True)[0]:
        i = int(index)
        rows.append({
            "index": i,
            "family": baseline["paths"][i].split("/")[-1].split(".")[0],
            "delta_nll_bits": float(delta[i]),
            "baseline_theta": baseline["theta"][i].tolist(),
            "candidate_theta": candidate["theta"][i].tolist(),
        })
    report = {
        "baseline": baseline["row"],
        "candidate": candidate["row"],
        "candidate_minus_baseline_nll_bits": float(delta.sum()),
        "per_family_delta_median_bits": float(delta.median()),
        "per_family_delta_min_bits": float(delta.min()),
        "per_family_delta_max_bits": float(delta.max()),
        "material_threshold_bits": args.material_bits,
        "material_families": rows,
    }
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
