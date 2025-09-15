import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE  = torch.float32

# ----------------------------------------------------------
# Common helpers
# ----------------------------------------------------------
def lse4_torch(x0, x1, x2, x3):
    return torch.logsumexp(torch.stack([x0, x1, x2, x3], dim=0), dim=0)

CONFIGS = [
    triton.Config({'BLOCK':  512}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
]

# ----------------------------------------------------------
# Triton kernel A: chain-sum (3 adds: (((a+b)+c)+d))
# ----------------------------------------------------------
@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def lse4_chain_kernel(OUT, X0, X1, X2, X3,
                      N,  # runtime size
                      BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # Hints
    tl.max_contiguous(offs, BLOCK)
    tl.multiple_of(offs, 128)

    mask = offs < N
    ninf = -float('inf')

    x0 = tl.load(X0 + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x1 = tl.load(X1 + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x2 = tl.load(X2 + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x3 = tl.load(X3 + offs, mask=mask, other=ninf, cache_modifier=".ca")

    m01 = tl.maximum(x0, x1)
    m23 = tl.maximum(x2, x3)
    m   = tl.maximum(m01, m23)
    m_safe = tl.where(m == ninf, 0.0, m)

    e0 = tl.exp(x0 - m_safe)
    e1 = tl.exp(x1 - m_safe)
    e2 = tl.exp(x2 - m_safe)
    e3 = tl.exp(x3 - m_safe)

    # Chain: (((e0 + e1) + e2) + e3)  -> 3 additions
    s = (e0 + e1)
    s = s + e2
    s = s + e3

    y = tl.where(s == 0, ninf, tl.log(s) + m)
    tl.store(OUT + offs, y, mask=mask)


# ----------------------------------------------------------
# Triton kernel B: pairwise tree (3 adds: (a+b) + (c+d))
# ----------------------------------------------------------
@triton.autotune(configs=CONFIGS, key=['N'])
@triton.jit
def lse4_pair_kernel(OUT, X0, X1, X2, X3,
                     N,  # runtime size
                     BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    # Hints
    tl.max_contiguous(offs, BLOCK)
    tl.multiple_of(offs, 128)

    mask = offs < N
    ninf = -float('inf')

    x0 = tl.load(X0 + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x1 = tl.load(X1 + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x2 = tl.load(X2 + offs, mask=mask, other=ninf, cache_modifier=".ca")
    x3 = tl.load(X3 + offs, mask=mask, other=ninf, cache_modifier=".ca")

    m01 = tl.maximum(x0, x1)
    m23 = tl.maximum(x2, x3)
    m   = tl.maximum(m01, m23)
    m_safe = tl.where(m == ninf, 0.0, m)

    e0 = tl.exp(x0 - m_safe)
    e1 = tl.exp(x1 - m_safe)
    e2 = tl.exp(x2 - m_safe)
    e3 = tl.exp(x3 - m_safe)

    # Pairwise tree: (e0 + e1) + (e2 + e3)  -> 3 additions
    s01 = e0 + e1
    s23 = e2 + e3
    s   = s01 + s23

    y = tl.where(s == 0, ninf, tl.log(s) + m)
    tl.store(OUT + offs, y, mask=mask)


# ----------------------------------------------------------
# Simple wrappers
# ----------------------------------------------------------
def lse4_triton_chain(x0, x1, x2, x3):
    out = torch.empty_like(x0)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    lse4_chain_kernel[grid](out, x0, x1, x2, x3, n)
    return out

def lse4_triton_pair(x0, x1, x2, x3):
    out = torch.empty_like(x0)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    lse4_pair_kernel[grid](out, x0, x1, x2, x3, n)
    return out

# ----------------------------------------------------------
# Quick correctness (including -inf lanes)
# ----------------------------------------------------------
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
        ya = lse4_triton_chain(x0, x1, x2, x3)
        yb = lse4_triton_pair(x0, x1, x2, x3)
        assert torch.allclose(yt, ya, atol=1e-6, rtol=1e-6) or torch.isnan(yt - ya).all()
        assert torch.allclose(yt, yb, atol=1e-6, rtol=1e-6) or torch.isnan(yt - yb).all()
    print("[Self-test] OK")

# ----------------------------------------------------------
# Perf plot: Torch vs Triton(chain) vs Triton(pairwise)
# ----------------------------------------------------------
LINE_PROVIDERS = ["torch", "triton_chain", "triton_pair"]
LINE_NAMES     = ["Torch (logsumexp of 4)", "Triton chain (3 adds)", "Triton pairwise (3 adds)"]
LINE_STYLES    = [('green', '-') , ('blue', '-') , ('orange', '-')]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28)],
        x_log=True,
        line_arg='provider',
        line_vals=LINE_PROVIDERS,
        line_names=LINE_NAMES,
        styles=LINE_STYLES,
        ylabel='Throughput (GB/s; 4 loads + 1 store)',
        plot_name='lse4-chain-vs-pairwise',
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
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: lse4_torch(x0, x1, x2, x3), quantiles=quantiles)
    elif provider == 'triton_chain':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: lse4_triton_chain(x0, x1, x2, x3), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: lse4_triton_pair(x0, x1, x2, x3), quantiles=quantiles)

    # Effective GB/s (4 inputs read + 1 output written)
    def gbps(t_ms):
        return 5 * x0.numel() * x0.element_size() * 1e-9 / (t_ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    self_test()
    benchmark.run(print_data=True, show_plots=True)
