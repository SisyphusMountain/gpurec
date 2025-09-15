#!/usr/bin/env python3
"""
Benchmark: matmul([1_000_000, 1_000], [1_000, 1_000]) in float32.

This script allocates A (1e6 x 1e3) and B (1e3 x 1e3) on CUDA (if available)
and measures the runtime of torch.matmul with warmup. It prints median latency
and effective GFLOP/s. Falls back to CPU is not supported for this size.

Run:
  python triton_examples/test_large_matmul_time.py
"""
import os
import sys
import math
import time
import torch

try:
    import triton
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False


def bytes_str(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f} {units[i]}"


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping")
        return 0

    device = torch.device("cuda")
    dtype = torch.float32
    M, K, N = 1_000_000, 1_000, 1_000

    # Memory estimate
    elt_bytes = torch.finfo(dtype).bits // 8
    bytes_A = M * K * elt_bytes
    bytes_B = K * N * elt_bytes
    bytes_C = M * N * elt_bytes
    total_bytes = bytes_A + bytes_B + bytes_C

    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {props.name} | total={bytes_str(total)} free={bytes_str(free)}")
    print(f"Planned allocation: A={bytes_str(bytes_A)}, B={bytes_str(bytes_B)}, C={bytes_str(bytes_C)} (total ~{bytes_str(total_bytes)})")

    if total_bytes > total * 0.85:
        print("Requested tensors exceed safe fraction of device memory; aborting to avoid OOM.")
        print("Tip: run on a larger GPU or reduce M via env M=... K=... N=...")
        # Allow override via env
        M_env = int(os.getenv("M", M))
        K_env = int(os.getenv("K", K))
        N_env = int(os.getenv("N", N))
        if (M_env, K_env, N_env) != (M, K, N):
            M, K, N = M_env, K_env, N_env
            print(f"Overriding with env sizes: M={M} K={K} N={N}")
        else:
            return 0

    # Allocate
    g = torch.Generator(device=device).manual_seed(0)
    A = torch.randn((M, K), device=device, dtype=dtype, generator=g)
    B = torch.randn((K, N), device=device, dtype=dtype, generator=g)

    # Warmup
    for _ in range(3):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Time
    def run():
        return torch.matmul(A, B)

    if HAVE_TRITON:
        q = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=q)
    else:
        iters = 5
        t0 = time.perf_counter()
        for _ in range(iters):
            C = run()
        torch.cuda.synchronize()
        ms = ((time.perf_counter() - t0) / iters) * 1e3
        min_ms = max_ms = ms

    # FLOPs: 2*M*N*K
    gflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e9
    print(f"matmul float32 [{M}x{K}] x [{K}x{N}] -> [{M}x{N}]\n  median: {ms:.3f} ms  ({gflops:.2f} GFLOP/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

