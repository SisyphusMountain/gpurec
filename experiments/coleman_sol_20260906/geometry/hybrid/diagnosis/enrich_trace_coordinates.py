"""CPU-only addition of derivable phi/gradient-phi fields to an existing trace."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


HYBRID = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HYBRID))
from hierarchical_adapter import theta_to_phi, transform_gradient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    saved = torch.load(args.trace, map_location="cpu", weights_only=False)
    for arm in saved["arms"].values():
        for category in ("evaluations", "proposals"):
            for row in arm["trace"][category]:
                if "phi" not in row:
                    row["phi"] = theta_to_phi(row["theta"])
                if "gradient_phi" not in row:
                    row["gradient_phi"] = transform_gradient(row["phi"], row["gradient_theta"])
    saved["schema"] = "gpurec.hybrid_diagnostic_trace.v1.coordinate_complete"
    saved["coordinate_enrichment"] = (
        "CPU-derived deterministically from each recorded theta and native gradient; no model call"
    )
    torch.save(saved, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
