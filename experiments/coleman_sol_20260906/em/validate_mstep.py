"""Validate the recovered and exact active-set M-steps on saved Coleman counts."""
from __future__ import annotations

import argparse
import importlib.util

import torch

from mstep import HI, LO, m_step, surrogate


def load_recovered(path: str):
    spec = importlib.util.spec_from_file_location("recovered_fraction", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", required=True)
    parser.add_argument("--recovered-fraction", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    saved = torch.load(args.counts, map_location="cpu", weights_only=False)
    recovered = load_recovered(args.recovered_fraction)
    report = {}
    for tag in ("start", "adam", "opt"):
        counts = -saved[f"a_total_{tag}"].double()
        theta, status, residual = m_step(counts)
        old_theta, old_pinned, old_bad = recovered.m_step(counts, LO, HI)
        dq = surrogate(theta, counts) - surrogate(old_theta, counts)
        changed = (theta - old_theta).abs().amax(dim=1) > 1e-10
        row = {
            "families": int(counts.shape[0]),
            "old_kkt_bad_coordinates": int(old_bad.sum()),
            "changed_families": int(changed.sum()),
            "new_max_scaled_kkt_residual": float(residual.max()),
            "new_pinned_coordinates": int((status != 0).sum()),
            "min_q_improvement_nats": float(dq.min()),
            "max_q_improvement_nats": float(dq.max()),
        }
        report[tag] = row
        print(f"[mstep] {tag}: {row}", flush=True)
    torch.save(report, args.out)
    print(f"[mstep] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
