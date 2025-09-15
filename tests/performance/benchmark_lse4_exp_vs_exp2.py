import os
import torch
import triton
import triton.language as tl
import math
DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE  = torch.float32

# --------------------------
# Torch baseline
# --------------------------
def lse4_torch(x0, x1, x2, x3):
    return torch.logsumexp(torch.stack([x0, x1, x2, x3], dim=0), dim=0)

# --------------------------
# Triton kernels
#   - single kernel toggled by meta-parameter `use_base2`
#   - pairwise reduction for better ILP
# --------------------------
LOG2E = 1/math.log(2)  # 1/ln(2)
LN2   = math.log(2)  # ln(2)
LOG2E = tl.constexpr(LOG2E)
LN2   = tl.constexpr(LN2)

CONFIGS = [
    triton.Config({'BLOCK':  512}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=1),
]

@triton.autotune(configs=CONFIGS, key=['N', 'use_base2'])
@triton.jit
def lse4_pair_kernel(out_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr,
                     N,  # runtime length
                     use_base2: tl.constexpr,  # toggle math mode at compile time
                     BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # Alignment / coalescing hints
    tl.max_contiguous(offs, BLOCK)
    tl.multiple_of(offs, 128)

    mask = offs < N
    ninf = -float('inf')

    # Streaming loads; try ".cg" vs ".ca" on your GPU
    x0 = tl.load(x0_ptr + offs, mask=mask, other=ninf, cache_modifier=".cg")
    x1 = tl.load(x1_ptr + offs, mask=mask, other=ninf, cache_modifier=".cg")
    x2 = tl.load(x2_ptr + offs, mask=mask, other=ninf, cache_modifier=".cg")
    x3 = tl.load(x3_ptr + offs, mask=mask, other=ninf, cache_modifier=".cg")

    # max (balanced)
    m01 = tl.maximum(x0, x1)
    m23 = tl.maximum(x2, x3)
    m   = tl.maximum(m01, m23)
    m_safe = tl.where(m == ninf, 0.0, m)  # guard all -inf lanes

    # exponentials and sum (pairwise)
    if use_base2:
        e0 = tl.exp2((x0 - m_safe) * LOG2E)
        e1 = tl.exp2((x1 - m_safe) * LOG2E)
        e2 = tl.exp2((x2 - m_safe) * LOG2E)
        e3 = tl.exp2((x3 - m_safe) * LOG2E)
        s01 = e0 + e1
        s23 = e2 + e3
        s   = s01 + s23
        y = tl.where(s == 0, ninf, tl.log2(s) * LN2 + m)
    else:
        e0 = tl.exp(x0 - m_safe); e1 = tl.exp(x1 - m_safe)
        e2 = tl.exp(x2 - m_safe); e3 = tl.exp(x3 - m_safe)
        s01 = e0 + e1
        s23 = e2 + e3
        s   = s01 + s23
        y = tl.where(s == 0, ninf, tl.log(s) + m)

    tl.store(out_ptr + offs, y, mask=mask)

def lse4_triton_pair(x0, x1, x2, x3, *, use_base2: bool):
    out = torch.empty_like(x0)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    lse4_pair_kernel[grid](out, x0, x1, x2, x3, n, use_base2=use_base2)
    return out

# --------------------------
# Correctness self-test
# --------------------------
def _sprinkle_neginf(x: torch.Tensor, p=0.001):
    m = torch.rand_like(x) < p
    x[m] = -float('inf')
    return x

def self_test():
    torch.manual_seed(0)
    for n in [1, 7, 513, 65536]:
        x0 = _sprinkle_neginf(torch.randn(n, device=DEVICE, dtype=DTYPE))
        x1 = _sprinkle_neginf(torch.randn(n, device=DEVICE, dtype=DTYPE))
        x2 = _sprinkle_neginf(torch.randn(n, device=DEVICE, dtype=DTYPE))
        x3 = _sprinkle_neginf(torch.randn(n, device=DEVICE, dtype=DTYPE))
        yt = lse4_torch(x0, x1, x2, x3)
        ya = lse4_triton_pair(x0, x1, x2, x3, use_base2=False)
        yb = lse4_triton_pair(x0, x1, x2, x3, use_base2=True)
        # allow NaN==NaN on all -inf lanes
        assert torch.allclose(yt, ya, atol=1e-6, rtol=1e-6) or torch.isnan(yt - ya).all()
        assert torch.allclose(yt, yb, atol=1e-6, rtol=1e-6) or torch.isnan(yt - yb).all()
    print("[Self-test] OK")

# --------------------------
# Benchmark & save plot/CSV
# --------------------------
LINE_PROVIDERS = ["torch", "triton_exp", "triton_exp2"]
LINE_NAMES     = ["Torch LSE(4)", "Triton LSE(4) exp/log", "Triton LSE(4) exp2/log2"]
LINE_STYLES    = [('green','-'), ('blue','-'), ('orange','-')]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28)],  # tune as you like
        x_log=True,
        line_arg='provider',
        line_vals=LINE_PROVIDERS,
        line_names=LINE_NAMES,
        styles=LINE_STYLES,
        ylabel='Throughput (GB/s; 4 loads + 1 store)',
        plot_name='lse4_exp_vs_exp2',
        args={},
    )
)
def benchmark(size, provider):
    x0 = torch.randn(size, device=DEVICE, dtype=DTYPE)
    x1 = torch.randn(size, device=DEVICE, dtype=DTYPE)
    x2 = torch.randn(size, device=DEVICE, dtype=DTYPE)
    x3 = torch.randn(size, device=DEVICE, dtype=DTYPE)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: lse4_torch(x0, x1, x2, x3), quantiles=quantiles
        )
    elif provider == 'triton_exp':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: lse4_triton_pair(x0, x1, x2, x3, use_base2=False),
            quantiles=quantiles
        )
    else:  # 'triton_exp2'
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: lse4_triton_pair(x0, x1, x2, x3, use_base2=True),
            quantiles=quantiles
        )

    # Effective GB/s using 4 loads + 1 store
    g = lambda t_ms: (5 * x0.numel() * x0.element_size() * 1e-9) / (t_ms * 1e-3)
    return g(ms), g(max_ms), g(min_ms)

if __name__ == "__main__":
    self_test()
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    benchmark.run(print_data=True, show_plots=False, save_path=save_dir)
    print(f"Saved: {os.path.abspath(save_dir)}/lse4_exp_vs_exp2.png/.csv")
