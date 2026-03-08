#!/usr/bin/env python3
"""
Profile src/reconciliation/likelihood.py using the same arguments as the reconcile CLI.

Usage example:
  python scripts/profile_likelihood.py \
    --species tests/data/test_mixed_200/sp.nwk \
    --gene tests/data/test_mixed_200/g.nwk \
    --delta 0.16103 --tau 0.156391 --lambda 1e-10 \
    --iters 3000 --dtype float64 --use-triton

Produces a Chrome trace JSON (default: likelihood_trace.json) you can open with
chrome://tracing or https://ui.perfetto.dev/.
"""

import sys
import argparse
from pathlib import Path
import torch

# Add project root to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reconciliation.reconcile import setup_fixed_points


def main():
    p = argparse.ArgumentParser(description="Profile likelihood (E/Pi fixed points) with PyTorch profiler")
    p.add_argument('--species', required=True, help='Species tree file (.nwk)')
    p.add_argument('--gene', required=True, help='Gene tree file (.nwk)')
    p.add_argument('--delta', type=float, required=True, help='Duplication rate')
    p.add_argument('--tau', type=float, required=True, help='Transfer rate')
    p.add_argument('--lambda', dest='lambda_param', type=float, required=True, help='Loss rate')
    p.add_argument('--iters', type=int, default=100, help='Max iterations for E and Pi fixed points')
    p.add_argument('--dtype', choices=['float32','float64','fp32','fp64'], default='float64')
    p.add_argument('--device', choices=['cpu','cuda'], default=None)
    p.add_argument('--use-triton', action='store_true', default=True, help='Use Triton LSE kernels')
    p.add_argument('--no-triton', dest='use_triton', action='store_false', help='Disable Triton kernels')
    p.add_argument('--trace', default=None, help='Output Chrome trace JSON path (omit to skip saving a heavy trace)')
    p.add_argument('--summary-only', action='store_true', help='Print aggregated profiler table only; do not export a trace')
    # Fine-grained toggles (each adds a lot of data if enabled)
    p.add_argument('--shapes', action='store_true', help='Record tensor shapes')
    p.add_argument('--memory', action='store_true', help='Profile memory allocations')
    p.add_argument('--stack', action='store_true', help='Capture Python stack traces for ops')
    p.add_argument('--flops', action='store_true', help='Estimate FLOPs (CPU only)')
    p.add_argument('--modules', action='store_true', help='Include module hierarchy')
    p.add_argument('--wallclock-only', action='store_true', help='Bypass profiler and report only wall-clock time')
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 if args.dtype in ('float32','fp32') else torch.float64

    if args.wallclock_only:
        # Simple wall-clock timing without profiler overhead
        import time
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = setup_fixed_points(
            args.species, args.gene,
            delta=args.delta, tau=args.tau, lambda_param=args.lambda_param,
            max_iters_E=args.iters, max_iters_Pi=args.iters,
            device=device, dtype=dtype, use_triton=args.use_triton,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        print(f"Wall-clock: {dt:.2f} ms")
        if 'log_likelihood' in result:
            print(f"Log-likelihood: {result['log_likelihood']:.6f}")
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    print(f"Profiling likelihood on device={device}, dtype={dtype}, iters={args.iters}")
    if args.trace and not args.summary_only:
        print(f"Trace -> {args.trace}")
    else:
        print("Trace export disabled (summary-only)")

    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.shapes,
        profile_memory=args.memory,
        with_stack=args.stack,
        with_flops=args.flops,
        with_modules=args.modules,
    ) as prof:
        # Run the full fixed-point setup once; internal code uses record_function and NVTX
        result = setup_fixed_points(
            args.species, args.gene,
            delta=args.delta, tau=args.tau, lambda_param=args.lambda_param,
            max_iters_E=args.iters, max_iters_Pi=args.iters,
            device=device, dtype=dtype, use_triton=args.use_triton,
        )
        # Ensure any pending CUDA work is finished before exporting
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Print summary table
    print("\nTop CUDA ops (or CPU if no CUDA):")
    print(prof.key_averages().table(sort_by='cuda_time_total' if device.type == 'cuda' else 'cpu_time_total', row_limit=30))

    # Export Chrome trace only if requested and not summary-only
    if args.trace and not args.summary_only:
        prof.export_chrome_trace(args.trace)
        print("\nTrace saved. Open in chrome://tracing or https://ui.perfetto.dev/")
    else:
        print("\nNo trace exported (use --trace to save one).")
    if 'log_likelihood' in result:
        print(f"Log-likelihood: {result['log_likelihood']:.6f}")


if __name__ == '__main__':
    main()
