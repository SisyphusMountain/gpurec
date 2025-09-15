import os
import sys
import torch
import triton.testing as ttesting

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/reconciliation'))

from triton.lse import (
    lse4_torch, lse4_triton_chain, lse4_triton_pair,
    lse5_torch, lse5_triton_chain, lse5_triton_pair,
    lse7_torch, lse7_triton_chain, lse7_triton_pair,
    DEVICE, DTYPE
)

def _gbps(n_elems: int, bytes_per_elem: int, n_inputs: int, t_ms: float) -> float:
    # Effective throughput using (n_inputs loads + 1 store)
    return (n_inputs + 1) * n_elems * bytes_per_elem * 1e-9 / (t_ms * 1e-3)

def run_benchmarks(save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    # ----- 4 inputs -----
    LINE_PROVIDERS = ["torch", "triton_chain", "triton_pair"]
    LINE_NAMES     = ["Torch LSE(4)", "Triton chain (4)", "Triton pair (4)"]
    LINE_STYLES    = [('green', '-'), ('blue', '-'), ('orange', '-')]

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
            plot_name='lse4_chain_vs_pair',
            args={},
        )
    )
    def bench4(size, provider):
        x0 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x1 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x2 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x3 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = ttesting.do_bench(lambda: lse4_torch(x0, x1, x2, x3), quantiles=quantiles)
        elif provider == 'triton_chain':
            ms, min_ms, max_ms = ttesting.do_bench(lambda: lse4_triton_chain(x0, x1, x2, x3), quantiles=quantiles)
        else:
            ms, min_ms, max_ms = ttesting.do_bench(lambda: lse4_triton_pair (x0, x1, x2, x3), quantiles=quantiles)

        g = lambda t: _gbps(x0.numel(), x0.element_size(), 4, t)
        return g(ms), g(max_ms), g(min_ms)

    bench4.run(print_data=True, show_plots=False, save_path=save_dir)

    # ----- 5 inputs -----
    LINE_PROVIDERS = ["torch", "triton_chain", "triton_pair"]
    LINE_NAMES     = ["Torch LSE(5)", "Triton chain (5)", "Triton pair (5)"]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],
            x_vals=[2**i for i in range(12, 28)],
            x_log=True,
            line_arg='provider',
            line_vals=LINE_PROVIDERS,
            line_names=LINE_NAMES,
            styles=[('green','-'), ('blue','-'), ('orange','-')],
            ylabel='Throughput (GB/s; 5 loads + 1 store)',
            plot_name='lse5_chain_vs_pair',
            args={},
        )
    )
    def bench5(size, provider):
        x0 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x1 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x2 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x3 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x4 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = ttesting.do_bench(lambda: lse5_torch(x0, x1, x2, x3, x4), quantiles=quantiles)
        elif provider == 'triton_chain':
            ms, min_ms, max_ms = ttesting.do_bench(lambda: lse5_triton_chain(x0, x1, x2, x3, x4), quantiles=quantiles)
        else:
            ms, min_ms, max_ms = ttesting.do_bench(lambda: lse5_triton_pair (x0, x1, x2, x3, x4), quantiles=quantiles)

        g = lambda t: _gbps(x0.numel(), x0.element_size(), 5, t)
        return g(ms), g(max_ms), g(min_ms)

    bench5.run(print_data=True, show_plots=False, save_path=save_dir)

    # ----- 7 inputs -----
    LINE_PROVIDERS = ["torch", "triton_chain", "triton_pair"]
    LINE_NAMES     = ["Torch LSE(7)", "Triton chain (7)", "Triton pair (7)"]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],
            x_vals=[2**i for i in range(12, 28)],
            x_log=True,
            line_arg='provider',
            line_vals=LINE_PROVIDERS,
            line_names=LINE_NAMES,
            styles=[('green','-'), ('blue','-'), ('orange','-')],
            ylabel='Throughput (GB/s; 7 loads + 1 store)',
            plot_name='lse7_chain_vs_pair',
            args={},
        )
    )
    def bench7(size, provider):
        x0 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x1 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x2 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x3 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x4 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x5 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        x6 = torch.randn(size, device=DEVICE, dtype=DTYPE)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = ttesting.do_bench(
                lambda: lse7_torch(x0, x1, x2, x3, x4, x5, x6), quantiles=quantiles
            )
        elif provider == 'triton_chain':
            ms, min_ms, max_ms = ttesting.do_bench(
                lambda: lse7_triton_chain(x0, x1, x2, x3, x4, x5, x6), quantiles=quantiles
            )
        else:
            ms, min_ms, max_ms = ttesting.do_bench(
                lambda: lse7_triton_pair (x0, x1, x2, x3, x4, x5, x6), quantiles=quantiles
            )

        g = lambda t: _gbps(x0.numel(), x0.element_size(), 7, t)
        return g(ms), g(max_ms), g(min_ms)

    bench7.run(print_data=True, show_plots=False, save_path=save_dir)

    print(f"\nPlots and CSVs saved under: {os.path.abspath(save_dir)}")
    print("Files:")
    for fn in sorted(os.listdir(save_dir)):
        if fn.startswith("lse") and (fn.endswith(".png") or fn.endswith(".csv")):
            print("  -", fn)

if __name__ == "__main__":
    run_benchmarks(save_dir="results")