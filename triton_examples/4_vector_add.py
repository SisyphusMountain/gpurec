import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# --------------------------
# 4-input add Triton kernel
# --------------------------
@triton.jit
def add4_kernel(x0_ptr, x1_ptr, x2_ptr, x3_ptr,
                out_ptr, n_elements,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Alignment / codegen hints
    tl.max_contiguous(offsets, BLOCK_SIZE)
    tl.multiple_of(offsets, 128)  # elements, not bytes

    mask = offsets < n_elements

    # Optional cache hint; remove if it regresses on your GPU
    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0, cache_modifier=".ca")
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0, cache_modifier=".ca")
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0, cache_modifier=".ca")
    x3 = tl.load(x3_ptr + offsets, mask=mask, other=0.0, cache_modifier=".ca")

    out = (x0 + x1) + (x2 + x3)
    tl.store(out_ptr + offsets, out, mask=mask)


def add_cfg(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor,
            *, BLOCK_SIZE: int, num_warps: int, num_stages: int):
    # guards
    assert x0.shape == x1.shape == x2.shape == x3.shape
    assert x0.device == x1.device == x2.device == x3.device == DEVICE
    assert x0.dtype == x1.dtype == x2.dtype == x3.dtype == torch.float32

    out = torch.empty_like(x0)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    # Launch with explicit tuning knobs
    add4_kernel[grid](
        x0, x1, x2, x3, out, n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# ---------------------------------
# Configs to sweep as separate lines
# ---------------------------------
CONFIGS = [
    {"BLOCK_SIZE": 512,  "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2},
    {"BLOCK_SIZE": 1024, "num_warps": 8, "num_stages": 2},
    {"BLOCK_SIZE": 2048, "num_warps": 8, "num_stages": 2},
]

def cfg_name(cfg):
    return f"triton_B{cfg['BLOCK_SIZE']}_W{cfg['num_warps']}_S{cfg['num_stages']}"

LINE_PROVIDERS = ["torch"] + [cfg_name(c) for c in CONFIGS]
LINE_NAMES     = ["Torch (4-in)"] + [f"Triton {cfg_name(c)[7:]}" for c in CONFIGS]  # prettier labels
LINE_STYLES    = [('green', '-')] + [('blue', '-')] * len(CONFIGS)  # customize if you like


# ----------------
# Perf report plot
# ----------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28)],  # 4 KB .. 256 MB elems-ish
        x_log=True,
        line_arg='provider',
        line_vals=LINE_PROVIDERS,
        line_names=LINE_NAMES,
        styles=LINE_STYLES,
        ylabel='GB/s',
        plot_name='vector-add-4in-multi-configs',
        args={},
    )
)
def benchmark(size, provider):
    # fresh tensors per size
    x0 = torch.rand(size, device=DEVICE, dtype=torch.float32)
    x1 = torch.rand(size, device=DEVICE, dtype=torch.float32)
    x2 = torch.rand(size, device=DEVICE, dtype=torch.float32)
    x3 = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: (x0 + x1 + x2 + x3), quantiles=quantiles
        )
    else:
        # find the matching cfg
        idx = LINE_PROVIDERS.index(provider) - 1  # -1 because 'torch' is first
        cfg = CONFIGS[idx]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add_cfg(x0, x1, x2, x3, **cfg), quantiles=quantiles
        )

    # 4 loads + 1 store = 5 * N * bytes
    def gbps(t_ms):
        return 5 * x0.numel() * x0.element_size() * 1e-9 / (t_ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Run it (set show_plots=True to render)
benchmark.run(print_data=True, show_plots=True)
