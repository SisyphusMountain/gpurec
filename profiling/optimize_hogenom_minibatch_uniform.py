"""Mini-batch gradient descent for global uniform HOGENOM rates.

This script tests stochastic optimization of one shared global ``[D, L, T]``
vector in uniform-transfer mode.  It uses fixed resident micro-batch models and
increases the effective batch size by accumulating gradients over more
micro-batches before each optimizer step.

The objective optimized by each stochastic step is the average NLL over the
selected families.  This has the same optimum as the summed full-dataset NLL
but keeps gradient magnitudes comparable across effective batch sizes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from gpurec import GeneReconModel


OPTIMIZED_ENV = {
    "GPUREC_FORWARD_LEAF_INDEX": "1",
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS": "64",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
    "GPUREC_FORWARD_DTS_OVERLAP_MODE": "off",
    "GPUREC_KERNELIZED_ACTIVE_MASK": "1",
}


@dataclass
class MicroBatch:
    start: int
    stop: int
    model: GeneReconModel

    @property
    def size(self) -> int:
        return self.stop - self.start


def _set_optimized_env() -> None:
    for key, value in OPTIMIZED_ENV.items():
        os.environ.setdefault(key, value)


def _family_id(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _gene_paths(dataset: Path) -> list[Path]:
    paths = sorted(dataset.glob("g_*.nwk"), key=_family_id)
    if not paths:
        raise FileNotFoundError(f"no g_*.nwk files in {dataset}")
    return paths


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_model(
    dataset: Path,
    gene_paths: list[Path],
    *,
    mode: str,
    fixed_pi: int,
    neumann_terms: int,
    device: torch.device,
    dtype: torch.dtype,
    preprocess_cache_dir: str,
    max_wave_size: int,
) -> GeneReconModel:
    return GeneReconModel.from_trees(
        species_tree=str(dataset / "sp.nwk"),
        gene_trees=[str(path) for path in gene_paths],
        mode=mode,
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        preprocess_cache_dir=preprocess_cache_dir,
        fixed_iters_Pi=fixed_pi,
        fixed_iters_E=None,
        max_iters_E=2000,
        tol_E=1e-10,
        neumann_terms=neumann_terms,
        use_pruning=False,
        max_wave_size=max_wave_size,
    )


def _build_micro_batches(
    dataset: Path,
    gene_paths: list[Path],
    *,
    micro_batch_size: int,
    fixed_pi: int,
    neumann_terms: int,
    device: torch.device,
    dtype: torch.dtype,
    preprocess_cache_dir: str,
    max_wave_size: int,
) -> list[MicroBatch]:
    batches: list[MicroBatch] = []
    for start in range(0, len(gene_paths), micro_batch_size):
        stop = min(start + micro_batch_size, len(gene_paths))
        model = _build_model(
            dataset,
            gene_paths[start:stop],
            mode="global",
            fixed_pi=fixed_pi,
            neumann_terms=neumann_terms,
            device=device,
            dtype=dtype,
            preprocess_cache_dir=preprocess_cache_dir,
            max_wave_size=max_wave_size,
        )
        batches.append(MicroBatch(start=start, stop=stop, model=model))
        print(f"[build] microbatch {len(batches):03d}: families {start}-{stop - 1}", flush=True)
    _sync(device)
    return batches


def _copy_theta(model: GeneReconModel, theta: torch.Tensor) -> None:
    with torch.no_grad():
        model.theta.copy_(theta.to(device=model.theta.device, dtype=model.theta.dtype))
        model.static.warm_E = None


@torch.no_grad()
def _full_nll(
    eval_model: GeneReconModel,
    theta: torch.Tensor,
) -> float:
    _copy_theta(eval_model, theta)
    value = float(eval_model.nll().detach().cpu())
    _sync(eval_model.theta.device)
    return value


def _per_family_validation(
    dataset: Path,
    gene_paths: list[Path],
    theta: torch.Tensor,
    *,
    fixed_pi: int,
    reference_pi: int,
    neumann_terms: int,
    device: torch.device,
    dtype: torch.dtype,
    preprocess_cache_dir: str,
    max_wave_size: int,
) -> pd.DataFrame:
    rows = []
    rates = torch.exp2(theta.detach().cpu()).tolist()
    for passes in (fixed_pi, reference_pi):
        model = _build_model(
            dataset,
            gene_paths,
            mode="genewise",
            fixed_pi=passes,
            neumann_terms=neumann_terms,
            device=device,
            dtype=dtype,
            preprocess_cache_dir=preprocess_cache_dir,
            max_wave_size=max_wave_size,
        )
        repeated_theta = theta.detach().to(device=device, dtype=dtype).expand(len(gene_paths), -1)
        with torch.no_grad():
            model.theta.copy_(repeated_theta)
            nll = model.nll_per_family().detach().cpu()
        _sync(device)
        rows.append(nll)

    fixed_nll, ref_nll = rows
    df = pd.DataFrame(
        {
            "family_id": [_family_id(path) for path in gene_paths],
            "family": [f"family_{_family_id(path):04d}" for path in gene_paths],
            "gene_file": [path.name for path in gene_paths],
            "D": rates[0],
            "L": rates[1],
            "T": rates[2],
            f"nll_fixed{fixed_pi}": fixed_nll.numpy(),
            f"nll_fixed{reference_pi}": ref_nll.numpy(),
        }
    )
    df["abs_diff"] = (fixed_nll - ref_nll).abs().numpy()
    return df


def _parse_schedule(text: str) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        batch_s, updates_s = item.split(":", 1)
        effective_batch = int(batch_s)
        updates = int(updates_s)
        if effective_batch < 1 or updates < 1:
            raise ValueError("schedule entries must be positive")
        schedule.append((effective_batch, updates))
    if not schedule:
        raise ValueError("empty schedule")
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("tests/data/hogenom_bench"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--preprocess-cache-dir", default="/tmp/gpurec_hogenom_minibatch_cache")
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--start-family", type=int, default=0)
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument(
        "--schedule",
        default="64:20,128:16,256:12,512:8,1055:4",
        help="comma-separated effective_batch:updates entries",
    )
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=2.0)
    parser.add_argument("--fixed-pi", type=int, default=20)
    parser.add_argument("--reference-pi", type=int, default=160)
    parser.add_argument("--neumann-terms", type=int, default=32)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-rates", default="0.05,0.05,0.05")
    args = parser.parse_args()

    _set_optimized_env()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir or (dataset / "gpurec" / "minibatch_uniform")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    gene_paths = _gene_paths(dataset)
    if args.start_family < 0:
        raise ValueError("--start-family must be non-negative")
    gene_paths = [path for path in gene_paths if _family_id(path) >= args.start_family]
    if args.max_families is not None:
        if args.max_families < 1:
            raise ValueError("--max-families must be positive when provided")
        gene_paths = gene_paths[: args.max_families]
    if not gene_paths:
        raise ValueError("empty family selection")
    schedule = _parse_schedule(args.schedule)
    if args.micro_batch_size < 1:
        raise ValueError("--micro-batch-size must be positive")

    init_rates = torch.tensor(
        [float(part) for part in args.init_rates.split(",")],
        device=device,
        dtype=dtype,
    )
    if tuple(init_rates.shape) != (3,):
        raise ValueError("--init-rates must contain three comma-separated values")
    theta = torch.log2(init_rates.clamp(args.min_rate, args.max_rate))
    theta_min = math.log2(args.min_rate)
    theta_max = math.log2(args.max_rate)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    micro_batches = _build_micro_batches(
        dataset,
        gene_paths,
        micro_batch_size=args.micro_batch_size,
        fixed_pi=args.fixed_pi,
        neumann_terms=args.neumann_terms,
        device=device,
        dtype=dtype,
        preprocess_cache_dir=args.preprocess_cache_dir,
        max_wave_size=args.max_wave_size,
    )
    eval_model = _build_model(
        dataset,
        gene_paths,
        mode="global",
        fixed_pi=args.fixed_pi,
        neumann_terms=args.neumann_terms,
        device=device,
        dtype=dtype,
        preprocess_cache_dir=args.preprocess_cache_dir,
        max_wave_size=args.max_wave_size,
    )

    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    step_index = 0
    full_start = time.perf_counter()
    history: list[dict[str, float | int | list[float]]] = []
    best_theta = theta.detach().clone()
    best_nll = _full_nll(eval_model, theta)
    print(f"[start] full fixed{args.fixed_pi} nll={best_nll:.6f}", flush=True)

    for stage_index, (target_effective_batch, updates) in enumerate(schedule, start=1):
        target_effective_batch = min(target_effective_batch, len(gene_paths))
        print(
            f"[stage {stage_index}] effective_batch={target_effective_batch} updates={updates}",
            flush=True,
        )
        for update in range(1, updates + 1):
            full_batch_indices = [
                idx
                for idx, mb in enumerate(micro_batches)
                if mb.size == args.micro_batch_size
            ]
            tail_indices = [
                idx
                for idx, mb in enumerate(micro_batches)
                if mb.size != args.micro_batch_size
            ]
            full_order = torch.randperm(
                len(full_batch_indices),
                generator=generator,
            ).tolist()
            order = [full_batch_indices[i] for i in full_order] + tail_indices
            selected: list[MicroBatch] = []
            selected_families = 0
            for idx in order:
                selected.append(micro_batches[idx])
                selected_families += micro_batches[idx].size
                if selected_families >= target_effective_batch:
                    break

            grad = torch.zeros_like(theta)
            batch_loss_sum = 0.0
            for mb in selected:
                _copy_theta(mb.model, theta)
                mb.model.zero_grad(set_to_none=True)
                loss = mb.model.nll()
                scaled_loss = loss / float(selected_families)
                scaled_loss.backward()
                _sync(device)
                if mb.model.theta.grad is None:
                    raise RuntimeError("missing mini-batch gradient")
                grad.add_(mb.model.theta.grad.detach().to(device=device, dtype=dtype))
                batch_loss_sum += float(loss.detach().cpu())

            step_index += 1
            m.mul_(args.beta1).add_(grad, alpha=1.0 - args.beta1)
            v.mul_(args.beta2).addcmul_(grad, grad, value=1.0 - args.beta2)
            m_hat = m / (1.0 - args.beta1**step_index)
            v_hat = v / (1.0 - args.beta2**step_index)
            theta = theta - args.lr * m_hat / (v_hat.sqrt() + args.eps)
            theta.clamp_(theta_min, theta_max)

            full_nll = _full_nll(eval_model, theta)
            if full_nll < best_nll:
                best_nll = full_nll
                best_theta = theta.detach().clone()
            rates = torch.exp2(theta.detach()).cpu().tolist()
            record = {
                "stage": stage_index,
                "update": update,
                "global_step": step_index,
                "target_effective_batch": target_effective_batch,
                "actual_effective_batch": selected_families,
                "micro_batches": len(selected),
                "batch_loss_sum": batch_loss_sum,
                "full_nll": full_nll,
                "best_full_nll": best_nll,
                "grad_norm": float(torch.linalg.vector_norm(grad).detach().cpu()),
                "D": rates[0],
                "L": rates[1],
                "T": rates[2],
                "elapsed_s": time.perf_counter() - full_start,
            }
            history.append(record)
            print(
                f"[update] stage={stage_index} update={update}/{updates} "
                f"eff={selected_families} full={full_nll:.6f} best={best_nll:.6f} "
                f"rates=({rates[0]:.5g},{rates[1]:.5g},{rates[2]:.5g}) "
                f"elapsed={record['elapsed_s']:.2f}s",
                flush=True,
            )

    theta = best_theta
    final_rates = torch.exp2(theta.detach()).cpu().tolist()
    validation = _per_family_validation(
        dataset,
        gene_paths,
        theta,
        fixed_pi=args.fixed_pi,
        reference_pi=args.reference_pi,
        neumann_terms=args.neumann_terms,
        device=device,
        dtype=dtype,
        preprocess_cache_dir=args.preprocess_cache_dir,
        max_wave_size=args.max_wave_size,
    )
    validation_path = output_dir / "fixed_pass_validation_minibatch_uniform.csv"
    validation.to_csv(validation_path, index=False, float_format="%.10g")
    history_path = output_dir / "history_minibatch_uniform.csv"
    pd.DataFrame(history).to_csv(history_path, index=False, float_format="%.10g")
    rates_path = output_dir / "model_rates_minibatch_uniform.csv"
    pd.DataFrame(
        [{"D": final_rates[0], "L": final_rates[1], "T": final_rates[2], "nll": best_nll}]
    ).to_csv(rates_path, index=False, float_format="%.10g")

    abs_diff = validation["abs_diff"]
    summary = {
        "dataset": str(dataset),
        "families": len(gene_paths),
        "micro_batch_size": args.micro_batch_size,
        "schedule": [
            {"effective_batch": batch, "updates": updates}
            for batch, updates in schedule
        ],
        "lr": args.lr,
        "fixed_pi": args.fixed_pi,
        "reference_pi": args.reference_pi,
        "neumann_terms": args.neumann_terms,
        "best_full_nll_fixed": best_nll,
        "validation_full_nll_fixed": float(validation[f"nll_fixed{args.fixed_pi}"].sum()),
        "validation_full_nll_reference": float(validation[f"nll_fixed{args.reference_pi}"].sum()),
        "max_abs_fixed_vs_reference": float(abs_diff.max()),
        "p99_abs_fixed_vs_reference": float(abs_diff.quantile(0.99)),
        "families_over_0_1": int((abs_diff > 0.1).sum()),
        "rates": {"D": final_rates[0], "L": final_rates[1], "T": final_rates[2]},
        "rates_path": str(rates_path),
        "history_path": str(history_path),
        "validation_path": str(validation_path),
        "elapsed_s": time.perf_counter() - full_start,
    }
    summary_path = output_dir / "summary_minibatch_uniform.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
