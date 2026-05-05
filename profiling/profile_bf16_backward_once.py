#!/usr/bin/env python3
"""Capture one warmed global/uniform backward pass for Nsight tools."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gpurec import GeneReconModel


def _dtype(text: str) -> torch.dtype:
    value = text.lower()
    if value in ("fp32", "float32"):
        return torch.float32
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp64", "float64"):
        return torch.float64
    raise argparse.ArgumentTypeError("dtype must be fp32, bf16, or fp64")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/data/test_trees_1000")
    parser.add_argument("--max-families", type=int, default=100)
    parser.add_argument("--dtype", type=_dtype, default=torch.bfloat16)
    parser.add_argument("--cache-dir", default="/tmp/gpurec_profile_bf16_cache")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--fixed-iters-Pi", type=int, default=6)
    parser.add_argument("--neumann-terms", type=int, default=3)
    return parser.parse_args()


def _run_backward(model: GeneReconModel) -> tuple[float, float, float]:
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    loss = model()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, float(loss.detach().float().cpu())


def main() -> None:
    args = _parse_args()
    root = Path(args.dataset)
    genes = [str(path) for path in sorted(root.glob("g_*.nwk"))[: args.max_families]]
    if not genes:
        raise SystemExit(f"no gene trees found in {root}")

    model = GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=genes,
        mode="global",
        pibar_mode="uniform",
        device="cuda",
        dtype=args.dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        preprocess_cache_dir=args.cache_dir,
        fixed_iters_Pi=args.fixed_iters_Pi,
        max_wave_size=32768,
        neumann_terms=args.neumann_terms,
        use_pruning=True,
        pruning_threshold=1e-6,
    )

    torch.cuda.synchronize()
    for _ in range(args.warmup):
        _run_backward(model)

    torch.cuda.cudart().cudaProfilerStart()
    forward_s, backward_s, loss = _run_backward(model)
    torch.cuda.cudart().cudaProfilerStop()
    print(
        "profiled_backward",
        "dtype",
        str(args.dtype).replace("torch.", ""),
        "families",
        len(genes),
        "forward_s",
        f"{forward_s:.6f}",
        "backward_s",
        f"{backward_s:.6f}",
        "loss",
        f"{loss:.8f}",
        "grad",
        [float(x) for x in model.theta.grad.detach().float().cpu()],
        flush=True,
    )


if __name__ == "__main__":
    main()
