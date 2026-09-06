"""Read-only likelihood probes of first-step hypotheses, not optimizer fits."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from gpurec.api.model import GeneReconModel
from gpurec.config import GpurecConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    proposals = torch.load(args.proposals, map_location="cpu", weights_only=False)
    shared = torch.load(proposals["artifact"], map_location="cpu", weights_only=False)
    config = GpurecConfig.genewise_reference()
    solver = replace(config.solver, pi_iters=16, neumann_terms=16)
    start = proposals["theta_start"].float()
    lo, hi = [float(torch.tensor(v, dtype=torch.float32))
              for v in shared["metadata"]["rate_box_log2"]]
    n = len(start)
    paths = shared["paths"][:n]
    clades = shared["family_clades"][:n]
    started = time.perf_counter()
    model = GeneReconModel(
        shared["metadata"]["species"], paths, mode="genewise", device="cuda",
        config=config, solver_options=solver,
        clade_budget=int(shared["metadata"]["clade_budget"]),
    )
    model.receiver_weights.requires_grad_(False)

    def value(theta):
        assert theta.shape == start.shape and bool(torch.isfinite(theta).all())
        assert bool(((theta >= lo) & (theta <= hi)).all())
        with torch.no_grad():
            loss = model.genewise_loss_vector(theta=theta.to(device="cuda", dtype=torch.float32))
        return loss.detach().double().cpu()

    baseline = value(start)
    vectors = {"start": baseline}
    metrics = {}
    for name, candidate in proposals["candidates"].items():
        loss = value(candidate["theta"].float())
        vectors[name] = loss
        gain = baseline - loss
        gain_fp32 = baseline.float() - loss.float()
        predicted = candidate["predicted"].float().cpu()
        pending = predicted > 0.05
        rejected = pending & (gain_fp32 < -0.05)
        shrunk = pending & (gain_fp32 < 0.25 * predicted)
        ratio = gain_fp32[pending] / predicted[pending]
        metrics[name] = {
            "nll_bits": float(loss.sum()),
            "total_actual_gain_bits": float(gain.sum()),
            "median_actual_gain_bits": float(gain.median()),
            "families_worse_above_0_05_bits": int((gain < -0.05).sum()),
            "pending_ratio_tests": int(pending.sum()),
            "would_reject_under_existing_rule": int(rejected.sum()),
            "would_shrink_under_existing_rule": int(shrunk.sum()),
            "rejected_clade_share": float(clades[rejected].sum() / clades.sum()),
            "retained_gain_after_existing_rejection_rule_bits": float(
                torch.where(rejected, torch.zeros_like(gain), gain).sum()),
            "median_ratio_when_pending": float(ratio.median()) if len(ratio) else None,
            "max_family_regression_bits": float((-gain).clamp_min(0).max()),
        }
        print(name, json.dumps(metrics[name]), flush=True)
    torch.cuda.synchronize()
    report = {
        "proposals": args.proposals, "n_families": n,
        "start_nll_bits": float(baseline.sum()), "candidates": metrics,
        "probe_seconds": time.perf_counter() - started,
        "note": "Common-point forward-only counterfactual probes. Not full fits or demonstrated runtime gains.",
    }
    destination = Path(args.out)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"paths": paths, "nll_vectors": vectors}, destination.with_suffix(".pt"))
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
