"""Generate a shared first-200-family EM2 trajectory through the production API."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path
import time

import torch

from gpurec.api.model import GeneReconModel
from gpurec.config import GpurecConfig
from gpurec.core.scheduling.batching import parse_families
from gpurec.fit.em_warmup import boxed_em_m_step

from hybrid_geometry import (
    calibrated_bfgs_seed,
    complete_information_z,
    transform_gradient,
    z_from_theta,
)


COMMON_START = (math.log2(0.01), math.log2(0.1), math.log2(0.01))


def _read_paths(list_file: str, limit: int) -> list[str]:
    rows = [line.strip() for line in open(list_file) if line.strip() and not line.startswith("#")]
    return rows if limit == 0 else rows[:limit]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _synchronize() -> None:
    torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--clade-budget", type=int, default=200740)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.limit != 200:
        raise ValueError("the shared hybrid artifact is defined on exactly the first 200 families")
    config = GpurecConfig.genewise_reference()
    solver = replace(config.solver, pi_iters=16, neumann_terms=16)
    paths = _read_paths(args.families, args.limit)
    if len(paths) != 200:
        raise ValueError(f"expected 200 family paths, found {len(paths)}")

    _synchronize()
    total_started = time.perf_counter()
    parse_started = time.perf_counter()
    parsed = parse_families(args.species, paths)
    parse_seconds = time.perf_counter() - parse_started
    metadata = parsed.families(list(range(len(paths))))
    family_clades = torch.tensor([int(row["C"]) for row in metadata], dtype=torch.int64)
    total_clades = int(family_clades.sum())

    _synchronize()
    build_started = time.perf_counter()
    model = GeneReconModel(
        args.species,
        paths,
        mode="genewise",
        device="cuda",
        dtype=torch.float32,
        solver_options=solver,
        config=config,
        clade_budget=args.clade_budget,
        parsed_families=parsed,
        family_indices=list(range(len(paths))),
    )
    model.receiver_weights.requires_grad_(False)
    _synchronize()
    build_seconds = time.perf_counter() - build_started

    lo = math.log2(config.rates.min_rate)
    hi = math.log2(config.rates.max_rate)
    theta_device = torch.tensor(COMMON_START, dtype=torch.float32, device="cuda").reshape(1, 3).repeat(200, 1)
    theta0 = theta_device.detach().double().cpu()
    counts_buffer = torch.empty((200, 4), dtype=torch.float64, device="cuda")
    nll_vectors: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    pass_seconds: list[float] = []
    ledger: list[dict[str, int | float | str]] = []

    theta_exact = theta0
    for step in range(2):
        _synchronize()
        pass_started = time.perf_counter()
        loss_vector, gradient, _ = model.genewise_loss_vector_and_grad(
            theta=theta_device,
            need_grad=True,
            event_counts_out=counts_buffer,
        )
        _synchronize()
        elapsed = time.perf_counter() - pass_started
        count_cpu = counts_buffer.detach().cpu().clone()
        gradient_cpu = gradient.detach().double().cpu()
        nll_cpu = loss_vector.detach().double().cpu()
        next_theta = boxed_em_m_step(count_cpu, lo, hi)

        nll_vectors.append(nll_cpu)
        gradients.append(gradient_cpu)
        counts.append(count_cpu)
        pass_seconds.append(elapsed)
        ledger.append({
            "phase": "em",
            "step": step,
            "families": len(paths),
            "clades": total_clades,
            "seconds": elapsed,
        })
        theta_exact = next_theta
        theta_device = next_theta.to(device="cuda", dtype=torch.float32).contiguous()

    theta1 = boxed_em_m_step(counts[0], lo, hi)
    theta2 = theta_exact
    theta1_evaluated = theta1.float().double()

    z0 = z_from_theta(theta0)
    z1 = z_from_theta(theta1)
    z2 = z_from_theta(theta2)
    gradient_z0 = transform_gradient(z0, gradients[0])
    gradient_z1 = transform_gradient(z1, gradients[1])
    step_z = z1 - z0
    change_gradient_z = gradient_z1 - gradient_z0
    information_z2 = complete_information_z(z2, counts[1])
    seed_z, seed_details = calibrated_bfgs_seed(information_z2, step_z, change_gradient_z)

    _synchronize()
    total_seconds = time.perf_counter() - total_started
    output = {
        "schema": "gpurec.em_hybrid_shared.v1",
        "description": "Production-API EM2 trajectory; no fit and no fitted-optimum inputs.",
        "paths": paths,
        "theta_native": {"theta0": theta0, "theta1": theta1, "theta2": theta2},
        "theta1_evaluated_fp32_as_fp64": theta1_evaluated,
        "gradient_theta": {"g0": gradients[0], "g1": gradients[1]},
        "event_counts": {"N0": counts[0], "N1": counts[1]},
        "nll_per_family_bits": {"nll0": nll_vectors[0], "nll1": nll_vectors[1]},
        "hierarchical": {
            "z0": z0,
            "z1": z1,
            "z2": z2,
            "g_z0": gradient_z0,
            "g_z1": gradient_z1,
            "step_z10": step_z,
            "gradient_change_z10": change_gradient_z,
            "information_z2_N1": information_z2,
            "scaled_bfgs_seed_z": seed_z,
            **seed_details,
        },
        "timing": {
            "parse_seconds": parse_seconds,
            "build_seconds": build_seconds,
            "pass_seconds": pass_seconds,
            "em_pass_seconds": sum(pass_seconds),
            "total_seconds": total_seconds,
        },
        "gradient_work": ledger,
        "gradient_calls": len(ledger),
        "gradient_clades": sum(int(row["clades"]) for row in ledger),
        "gradient_full_clade_equivalents": sum(int(row["clades"]) for row in ledger) / total_clades,
        "family_clades": family_clades,
        "metadata": {
            "n_families": len(paths),
            "clade_budget": args.clade_budget,
            "rate_box_log2": (lo, hi),
            "common_start_log2": COMMON_START,
            "theta_dtype": "torch.float32",
            "stored_dtype": "torch.float64",
            "accumulator_dtype": str(config.precision.accumulator_torch_dtype),
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "solver": asdict(solver),
            "precision": asdict(config.precision),
            "species": args.species,
            "species_sha256": _sha256(args.species),
            "families_list": args.families,
            "families_list_sha256": _sha256(args.families),
            "theta1_note": (
                "theta1 is the exact float64 M-step used by production secants; g1/nll1 were "
                "evaluated at its float32 cast, stored separately."
            ),
            "curvature_note": (
                "information_z2_N1 is constructed directly in z; scaled_bfgs_seed_z uses only "
                "the latest transformed z0->z1 pair and never reuses native B."
            ),
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    print(
        f"wrote {destination}: clades={total_clades:,}, nll0={float(nll_vectors[0].sum()):.6f}, "
        f"nll1={float(nll_vectors[1].sum()):.6f}, passes={pass_seconds}, "
        f"scale_valid={int(seed_details['scale_valid'].sum())}/200, "
        f"bfgs_valid={int(seed_details['bfgs_valid'].sum())}/200",
        flush=True,
    )
    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
