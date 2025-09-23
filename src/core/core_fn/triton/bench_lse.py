# bench_lse4.py
import argparse, math, contextlib
import torch

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timeit(fn, iters=100, warmup=10):
    # warmup
    for _ in range(warmup):
        fn()
    cuda_sync()
    # timed
    times = []
    for _ in range(iters):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
        else:
            import time
            t0 = time.perf_counter(); fn()
            times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    median = times[len(times)//2]
    p90    = times[math.floor(len(times)*0.9)]
    best   = min(times)
    return median, p90, best

def make_inputs(numel, dtype, device, requires_grad=False):
    x0 = torch.randn(numel, device=device, dtype=dtype, requires_grad=requires_grad)
    x1 = torch.randn_like(x0, requires_grad=requires_grad)
    x2 = torch.randn_like(x0, requires_grad=requires_grad)
    x3 = torch.randn_like(x0, requires_grad=requires_grad)
    return x0, x1, x2, x3

# ---------- Your op ----------
def bench_forward(lse4, numel, dtype, iters, warmup, device="cuda"):
    x0, x1, x2, x3 = make_inputs(numel, dtype, device, requires_grad=False)
    def run():
        y = lse4(x0, x1, x2, x3)
        # keep node alive
        _ = y.view(-1).sum()
    return timeit(run, iters, warmup)

def bench_backward_only(lse4, numel, dtype, iters, warmup, device="cuda"):
    x0, x1, x2, x3 = make_inputs(numel, dtype, device, requires_grad=True)
    y = lse4(x0, x1, x2, x3)
    go = torch.randn_like(y)
    # one warmup grad (outside timer), then reuse graph
    _ = torch.autograd.grad(y, (x0, x1, x2, x3), go, retain_graph=True)
    def run():
        _ = torch.autograd.grad(y, (x0, x1, x2, x3), go, retain_graph=True)
    return timeit(run, iters, warmup)

def bench_fwd_bwd(lse4, numel, dtype, iters, warmup, device="cuda"):
    def run():
        x0, x1, x2, x3 = make_inputs(numel, dtype, device, requires_grad=True)
        y = lse4(x0, x1, x2, x3)
        y.sum().backward()
    return timeit(run, iters, warmup)

# ---------- PyTorch baseline ----------
def baseline_forward(numel, dtype, iters, warmup, device="cuda"):
    x0, x1, x2, x3 = make_inputs(numel, dtype, device, requires_grad=False)
    def run():
        y = torch.logsumexp(torch.stack([x0, x1, x2, x3], dim=0), dim=0)
        _ = y.view(-1).sum()
    return timeit(run, iters, warmup)

def baseline_backward_only(numel, dtype, iters, warmup, device="cuda"):
    x0, x1, x2, x3 = make_inputs(numel, dtype, device, requires_grad=True)
    y = torch.logsumexp(torch.stack([x0, x1, x2, x3], dim=0), dim=0)
    go = torch.randn_like(y)
    # warmup grad, then reuse the same graph (retain_graph=True)
    _ = torch.autograd.grad(y, (x0, x1, x2, x3), go, retain_graph=True)
    def run():
        _ = torch.autograd.grad(y, (x0, x1, x2, x3), go, retain_graph=True)
    return timeit(run, iters, warmup)

def baseline_fwd_bwd(numel, dtype, iters, warmup, device="cuda"):
    def run():
        x0, x1, x2, x3 = make_inputs(numel, dtype, device, requires_grad=True)
        y = torch.logsumexp(torch.stack([x0, x1, x2, x3], dim=0), dim=0)
        y.sum().backward()
    return timeit(run, iters, warmup)

# ---------- Bytes moved rough estimate (optional bandwidth metric) ----------
def bytes_moved(numel, dtype, forward=True, saved_exps=True):
    itemsize = torch.tensor([], dtype=dtype).element_size()
    if forward:
        r = 4 * numel * itemsize
        w = 1 * numel * itemsize
        if saved_exps:  # your op writes 4 extra outputs
            w += 4 * numel * itemsize
        return r + w
    else:
        r = 5 * numel * itemsize  # e0..e3 + grad_out
        w = 4 * numel * itemsize  # g0..g3
        return r + w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numel", type=int, default=100_000_000)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float16","bfloat16","float32"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    # Import your wrapper (adjust path if needed)
    from fast_lse import lse4

    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {dtype} | numel: {args.numel:,}")
    print(f"iters={args.iters}, warmup={args.warmup}\n")

    f_med, f_p90, f_best = bench_forward(lse4, args.numel, dtype, args.iters, args.warmup, args.device)
    bw_f = bytes_moved(args.numel, dtype, forward=True, saved_exps=True) / (1024**3)
    print(f"[lse4] forward only      : median {f_med:.3f} ms | p90 {f_p90:.3f} | best {f_best:.3f}   (~{bw_f/f_med*1e3:.2f} GiB/s)")

    b_med, b_p90, b_best = bench_backward_only(lse4, args.numel, dtype, args.iters, args.warmup, args.device)
    bw_b = bytes_moved(args.numel, dtype, forward=False) / (1024**3)
    print(f"[lse4] backward only     : median {b_med:.3f} ms | p90 {b_p90:.3f} | best {b_best:.3f}   (~{bw_b/b_med*1e3:.2f} GiB/s)")

    fb_med, fb_p90, fb_best = bench_fwd_bwd(lse4, args.numel, dtype, args.iters, args.warmup, args.device)
    print(f"[lse4] forward + backward: median {fb_med:.3f} ms | p90 {fb_p90:.3f} | best {fb_best:.3f}")

    if args.baseline:
        blf_med, blf_p90, blf_best = baseline_forward(args.numel, dtype, args.iters, args.warmup, args.device)
        print(f"[torch.logsumexp] forward only      : median {blf_med:.3f} ms | p90 {blf_p90:.3f} | best {blf_best:.3f}")

        blb_med, blb_p90, blb_best = baseline_backward_only(args.numel, dtype, args.iters, args.warmup, args.device)
        print(f"[torch.logsumexp] backward only     : median {blb_med:.3f} ms | p90 {blb_p90:.3f} | best {blb_best:.3f}")

        blfb_med, blfb_p90, blfb_best = baseline_fwd_bwd(args.numel, dtype, args.iters, args.warmup, args.device)
        print(f"[torch.logsumexp] forward + backward: median {blfb_med:.3f} ms | p90 {blfb_p90:.3f} | best {blfb_best:.3f}")

if __name__ == "__main__":
    main()
