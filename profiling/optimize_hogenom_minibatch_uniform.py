"""Native mini-batch gradient descent for global uniform HOGENOM rates.

This script uses one resident :class:`UniformChunkedReconModel` and samples
prebuilt chunks for stochastic updates.  It is the efficient version of the
mini-batch experiment: E is computed once per update, selected chunks reuse the
resident wave layouts, and there are no per-mini-batch ``GeneReconModel``
objects.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from gpurec.api.uniform_chunked import UniformChunkedReconModel


def _family_id_from_path(path: str | Path) -> int:
    return int(Path(path).stem.split("_")[1])


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


def _select_chunks_for_effective_batch(
    model: UniformChunkedReconModel,
    *,
    target_families: int,
    generator: torch.Generator,
) -> tuple[list[int], int]:
    target = min(int(target_families), model.n_families)
    if target >= model.n_families:
        return list(range(model.n_chunks)), model.n_families

    counts = model.chunk_family_counts
    full_count = model.family_chunk_size if model.family_chunk_size > 0 else None
    full_chunks = [
        idx for idx, count in enumerate(counts)
        if full_count is None or count == full_count
    ]
    tail_chunks = [
        idx for idx, count in enumerate(counts)
        if full_count is not None and count != full_count
    ]
    perm = torch.randperm(len(full_chunks), generator=generator).tolist()
    order = [full_chunks[i] for i in perm] + tail_chunks
    selected: list[int] = []
    selected_families = 0
    for idx in order:
        selected.append(idx)
        selected_families += counts[idx]
        if selected_families >= target:
            break
    return selected, selected_families


def _manual_adam_update(
    theta: torch.nn.Parameter,
    grad: torch.Tensor,
    *,
    step_index: int,
    m: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    theta_min: float,
    theta_max: float,
) -> None:
    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    m_hat = m / (1.0 - beta1**step_index)
    v_hat = v / (1.0 - beta2**step_index)
    with torch.no_grad():
        theta.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)
        theta.clamp_(theta_min, theta_max)


@torch.no_grad()
def _copy_theta(dst: UniformChunkedReconModel, src_theta: torch.Tensor) -> None:
    dst.theta.copy_(src_theta.to(device=dst.theta.device, dtype=dst.theta.dtype))
    dst.clear_warm_start()


def _validate_fixed_passes(
    dataset: Path,
    theta: torch.Tensor,
    *,
    start: int,
    max_families: int | None,
    fixed_pi: int,
    reference_pi: int,
    family_chunk_size: int,
    max_wave_size: int,
    neumann_terms: int,
    device: str,
    dtype: torch.dtype,
    preprocess_cache_dir: str,
) -> pd.DataFrame:
    values: dict[int, torch.Tensor] = {}
    gene_trees: list[str] | None = None
    for passes in (fixed_pi, reference_pi):
        model = UniformChunkedReconModel.from_folder(
            dataset,
            start=start,
            max_families=max_families,
            device=device,
            dtype=dtype,
            theta_init_rates=(0.05, 0.05, 0.05),
            preprocess_cache_dir=preprocess_cache_dir,
            family_chunk_size=family_chunk_size,
            max_wave_size=max_wave_size,
            fixed_iters_Pi=passes,
            fixed_iters_E=None,
            max_iters_E=2000,
            tol_E=1e-10,
            neumann_terms=neumann_terms,
            use_pruning=False,
        )
        _copy_theta(model, theta)
        values[passes] = model.nll_per_family().detach().cpu()
        gene_trees = model.gene_trees

    if gene_trees is None:
        raise RuntimeError("validation did not run")
    family_ids = [_family_id_from_path(path) for path in gene_trees]
    rates = torch.exp2(theta.detach().cpu()).tolist()
    out = pd.DataFrame(
        {
            "family_id": family_ids,
            "family": [f"family_{family_id:04d}" for family_id in family_ids],
            "gene_file": [Path(path).name for path in gene_trees],
            "D": rates[0],
            "L": rates[1],
            "T": rates[2],
            f"nll_fixed{fixed_pi}": values[fixed_pi].numpy(),
            f"nll_fixed{reference_pi}": values[reference_pi].numpy(),
        }
    )
    out["abs_diff"] = (values[fixed_pi] - values[reference_pi]).abs().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("tests/data/hogenom_bench"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--preprocess-cache-dir", default="/tmp/gpurec_hogenom_native_minibatch_cache")
    parser.add_argument("--family-chunk-size", type=int, default=32)
    parser.add_argument("--start-family", type=int, default=0)
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument(
        "--schedule",
        default="64:8,128:8,256:6,512:4,1055:6",
        help="comma-separated effective_family_count:updates entries",
    )
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min-rate", type=float, default=1e-10)
    parser.add_argument("--max-rate", type=float, default=2.0)
    parser.add_argument("--fixed-pi", type=int, default=20)
    parser.add_argument("--reference-pi", type=int, default=160)
    parser.add_argument("--neumann-terms", type=int, default=8)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-rates", default="0.05,0.05,0.05")
    parser.add_argument("--full-eval-interval", type=int, default=4)
    args = parser.parse_args()

    if args.family_chunk_size < 1:
        raise ValueError("--family-chunk-size must be positive")
    if args.full_eval_interval < 1:
        raise ValueError("--full-eval-interval must be positive")
    dataset = args.dataset.resolve()
    output_dir = args.output_dir or (dataset / "gpurec" / "native_minibatch_uniform")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    schedule = _parse_schedule(args.schedule)

    init_rates = torch.tensor(
        [float(part) for part in args.init_rates.split(",")],
        device=args.device,
        dtype=dtype,
    )
    if tuple(init_rates.shape) != (3,):
        raise ValueError("--init-rates must contain three comma-separated values")
    if torch.any(init_rates <= 0):
        raise ValueError("--init-rates must be strictly positive")

    model = UniformChunkedReconModel.from_folder(
        dataset,
        start=args.start_family,
        max_families=args.max_families,
        device=args.device,
        dtype=dtype,
        theta_init_rates=tuple(float(x) for x in init_rates.detach().cpu().tolist()),
        preprocess_cache_dir=args.preprocess_cache_dir,
        family_chunk_size=args.family_chunk_size,
        max_wave_size=args.max_wave_size,
        fixed_iters_Pi=args.fixed_pi,
        fixed_iters_E=None,
        max_iters_E=2000,
        tol_E=1e-10,
        neumann_terms=args.neumann_terms,
        use_pruning=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    theta_min = math.log2(args.min_rate)
    theta_max = math.log2(args.max_rate)
    m = torch.zeros_like(model.theta)
    v = torch.zeros_like(model.theta)
    step_index = 0
    best_theta = model.theta.detach().clone()
    best_full_nll = float(model.nll().detach().cpu())
    initial_nll = best_full_nll
    full_start = time.perf_counter()
    history: list[dict[str, float | int | str]] = []

    print(
        f"[start] families={model.n_families} chunks={model.n_chunks} "
        f"chunk_size={args.family_chunk_size} full fixed{args.fixed_pi}={initial_nll:.6f}",
        flush=True,
    )

    for stage_index, (target_effective_batch, updates) in enumerate(schedule, start=1):
        print(
            f"[stage {stage_index}] target_effective_batch={target_effective_batch} updates={updates}",
            flush=True,
        )
        for update in range(1, updates + 1):
            chunk_indices, selected_families = _select_chunks_for_effective_batch(
                model,
                target_families=target_effective_batch,
                generator=generator,
            )
            loss, grad, stats = model.loss_and_grad(
                chunk_indices=chunk_indices,
                reduction="mean",
            )
            step_index += 1
            _manual_adam_update(
                model.theta,
                grad,
                step_index=step_index,
                m=m,
                v=v,
                lr=args.lr,
                beta1=args.beta1,
                beta2=args.beta2,
                eps=args.eps,
                theta_min=theta_min,
                theta_max=theta_max,
            )

            full_nll: float | None = None
            if step_index % args.full_eval_interval == 0:
                full_nll = float(model.nll().detach().cpu())
                if full_nll < best_full_nll:
                    best_full_nll = full_nll
                    best_theta = model.theta.detach().clone()

            rates = model.rates.detach().cpu().tolist()
            record = {
                "stage": stage_index,
                "update": update,
                "global_step": step_index,
                "target_effective_batch": min(target_effective_batch, model.n_families),
                "actual_effective_batch": selected_families,
                "selected_chunks": len(chunk_indices),
                "minibatch_mean_nll": float(loss.detach().cpu()),
                "full_nll": float("nan") if full_nll is None else full_nll,
                "best_full_nll": best_full_nll,
                "grad_norm": float(torch.linalg.vector_norm(grad).detach().cpu()),
                "D": rates[0],
                "L": rates[1],
                "T": rates[2],
                "elapsed_s": time.perf_counter() - full_start,
            }
            history.append(record)
            full_text = "skipped" if full_nll is None else f"{full_nll:.6f}"
            print(
                f"[update] stage={stage_index} update={update}/{updates} "
                f"eff={selected_families} chunks={len(chunk_indices)} "
                f"mb_mean={record['minibatch_mean_nll']:.6f} full={full_text} "
                f"best={best_full_nll:.6f} rates=({rates[0]:.5g},{rates[1]:.5g},{rates[2]:.5g}) "
                f"elapsed={record['elapsed_s']:.2f}s",
                flush=True,
            )

    _copy_theta(model, best_theta)
    final_rates = model.rates.detach().cpu().tolist()
    validation = _validate_fixed_passes(
        dataset,
        best_theta,
        start=args.start_family,
        max_families=args.max_families,
        fixed_pi=args.fixed_pi,
        reference_pi=args.reference_pi,
        family_chunk_size=args.family_chunk_size,
        max_wave_size=args.max_wave_size,
        neumann_terms=args.neumann_terms,
        device=args.device,
        dtype=dtype,
        preprocess_cache_dir=args.preprocess_cache_dir,
    )

    history_path = output_dir / "history_native_minibatch_uniform.csv"
    rates_path = output_dir / "model_rates_native_minibatch_uniform.csv"
    validation_path = output_dir / "fixed_pass_validation_native_minibatch_uniform.csv"
    summary_path = output_dir / "summary_native_minibatch_uniform.json"
    pd.DataFrame(history).to_csv(history_path, index=False, float_format="%.10g")
    pd.DataFrame(
        [{"D": final_rates[0], "L": final_rates[1], "T": final_rates[2], "nll": best_full_nll}]
    ).to_csv(rates_path, index=False, float_format="%.10g")
    validation.to_csv(validation_path, index=False, float_format="%.10g")

    abs_diff = validation["abs_diff"]
    summary = {
        "dataset": str(dataset),
        "families": model.n_families,
        "chunks": model.n_chunks,
        "family_chunk_size": args.family_chunk_size,
        "schedule": [
            {"effective_batch": batch, "updates": updates}
            for batch, updates in schedule
        ],
        "lr": args.lr,
        "fixed_pi": args.fixed_pi,
        "reference_pi": args.reference_pi,
        "neumann_terms": args.neumann_terms,
        "initial_full_nll_fixed": initial_nll,
        "best_full_nll_fixed": best_full_nll,
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
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
