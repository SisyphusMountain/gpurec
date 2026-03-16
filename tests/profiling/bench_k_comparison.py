"""Benchmark k=16 vs k=32 vs k=64 sparse Pibar: timing and accuracy at S=20K.

Memory-efficient: runs each k value sequentially, cleaning up between runs.

Run with:
  python -m tests.profiling.bench_k_comparison
"""

import gc
import math
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.core.likelihood import compute_log_likelihood
from src.core.kernels.wave_step import wave_step_fused
from src.core.kernels.dts_fused import dts_fused

NEG_INF = float("-inf")

# Import from explore_hybrid_pibar
from tests.profiling.explore_hybrid_pibar import (
    _sparse_input_pibar_kernel,
    compute_pibar_sparse_input_triton_precomputed,
    compute_pibar_dense,
    _precompute_species_children,
    _compute_dts_cross,
    load_families_generic,
)


def run_full_convergence(wave_layout, sp_child1, sp_child2, DL_const, Ebar_vec,
                         E_vec, SL1_const, SL2_const, leaf_term, pD, pS, S,
                         device, dtype, pibar_fn, max_iters=50, tol=1e-3):
    """Run all waves to convergence. Returns (Pi, Pibar, total_iters)."""
    wave_metas = wave_layout['wave_metas']
    n_waves = len(wave_metas)
    C = int(wave_layout['ccp_helpers']['C'])
    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']

    _PI_INIT = torch.finfo(dtype).min
    Pi = torch.full((C, S), _PI_INIT, dtype=dtype, device=device)
    Pi[leaf_row_index, leaf_col_index] = 0.0
    Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

    total_iters = 0
    for wi in range(n_waves):
        meta = wave_metas[wi]
        ws, we, W = meta['start'], meta['end'], meta['W']
        if meta['has_splits']:
            dts_r = _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2,
                                       pD, pS, S, device, dtype)
        else:
            dts_r = None
        leaf_wt = leaf_term[ws:we]
        for li in range(max_iters):
            Pi_W = Pi[ws:we]
            Pibar_W = pibar_fn(Pi_W, wi)
            Pi_new = wave_step_fused(Pi_W, Pibar_W, DL_const, Ebar_vec, E_vec,
                                     SL1_const, SL2_const, sp_child1, sp_child2,
                                     leaf_wt, dts_r)
            total_iters += 1
            if li >= 3:
                sig = Pi_new > -100.0
                if not sig.any() or torch.abs(Pi_new - Pi_W)[sig].max().item() < tol:
                    Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W; break
            Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W

    torch.cuda.synchronize()
    del Pibar  # not needed for accuracy/logL
    return Pi, None, total_iters


def run_inline(wave_layout, sp_child1, sp_child2, DL_const, Ebar_vec,
               E_vec, SL1_const, SL2_const, leaf_term, pD, pS, S,
               device, dtype, tf_T, mt, k, precomp, max_iters=50, tol=1e-3,
               per_wave_sync=False):
    """Inline version — no closures, no function call overhead."""
    wave_metas = wave_layout['wave_metas']
    n_waves = len(wave_metas)
    C = int(wave_layout['ccp_helpers']['C'])
    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']

    _PI_INIT = torch.finfo(dtype).min
    Pi = torch.full((C, S), _PI_INIT, dtype=dtype, device=device)
    Pi[leaf_row_index, leaf_col_index] = 0.0
    Pibar = torch.full((C, S), NEG_INF, dtype=dtype, device=device)

    total_iters = 0
    wave_times = [] if per_wave_sync else None
    for wi in range(n_waves):
        meta = wave_metas[wi]
        ws, we, W = meta['start'], meta['end'], meta['W']
        if per_wave_sync:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if meta['has_splits']:
            dts_r = _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2,
                                       pD, pS, S, device, dtype)
        else:
            dts_r = None
        leaf_wt = leaf_term[ws:we]
        for li in range(max_iters):
            Pi_W = Pi[ws:we]
            Pibar_W = compute_pibar_sparse_input_triton_precomputed(
                Pi_W, tf_T, mt, k, precomp[wi])
            Pi_new = wave_step_fused(Pi_W, Pibar_W, DL_const, Ebar_vec, E_vec,
                                     SL1_const, SL2_const, sp_child1, sp_child2,
                                     leaf_wt, dts_r)
            total_iters += 1
            if li >= 3:
                sig = Pi_new > -100.0
                if not sig.any() or torch.abs(Pi_new - Pi_W)[sig].max().item() < tol:
                    Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W; break
            Pi[ws:we] = Pi_new; Pibar[ws:we] = Pibar_W
        if per_wave_sync:
            torch.cuda.synchronize()
            wave_times.append(time.perf_counter() - t0)

    torch.cuda.synchronize()
    del Pibar
    return Pi, total_iters, wave_times


def main():
    device = torch.device("cuda")
    dtype = torch.float32
    ds_name = "test_trees_10000"
    n_repeats = 3

    print(f"Loading {ds_name}...")
    wave_layout, sh, pS, pD, pL, tf, mv, Eo, root_ids, device, dtype = \
        load_families_generic(1, ds_name)

    S = int(sh["S"])
    C = int(wave_layout["ccp_helpers"]["C"])
    wave_metas = wave_layout['wave_metas']
    n_waves = len(wave_metas)
    mt = mv.squeeze(-1) if mv.ndim > 1 else mv
    tf_T = tf.T.contiguous()

    print(f"S={S}, C={C}, waves={n_waves}")

    sp_child1, sp_child2 = _precompute_species_children(sh, S, device)
    E_vec = Eo["E"]; Ebar_vec = Eo["E_bar"]
    DL_const = 1.0 + pD + E_vec
    SL1_const = pS + Eo["E_s2"]
    SL2_const = pS + Eo["E_s1"]

    leaf_row_index = wave_layout['leaf_row_index']
    leaf_col_index = wave_layout['leaf_col_index']
    _PI_INIT = torch.finfo(dtype).min
    leaf_term = torch.full((C, S), NEG_INF, device=device, dtype=dtype)
    leaf_term[leaf_row_index, leaf_col_index] = 0.0
    leaf_term.add_(pS)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True

    inv_perm = wave_layout['inv_perm']
    perm = wave_layout['perm']

    # ── Phase 1: Dense reference + extract all top-k indices ──
    print("\n=== Dense reference ===")

    # Delete tf (only tf_T is needed), free ~1.6 GB
    del tf

    def dense_pibar(Pi_W, wi):
        Pi_max = Pi_W.max(dim=1, keepdim=True).values
        return torch.log2(torch.exp2(Pi_W - Pi_max) @ tf_T) + Pi_max + mt

    Pi_dense, Pibar_dense, dense_iters = run_full_convergence(
        wave_layout, sp_child1, sp_child2, DL_const, Ebar_vec, E_vec,
        SL1_const, SL2_const, leaf_term, pD, pS, S, device, dtype,
        dense_pibar)

    Pi_dense_orig = Pi_dense[perm]
    dense_logL = compute_log_likelihood(Pi_dense_orig, Eo["E"], root_ids).item()
    print(f"Dense: logL={dense_logL:.4f}, {dense_iters} iters")

    # Extract top-k indices for ALL k values at once (max k = 64)
    k_values = [16, 32, 64]
    max_k = max(k_values)
    print(f"Extracting top-{max_k} indices from dense Pi...")
    all_topk = {}  # wi -> [W, max_k] indices
    for wi, meta in enumerate(wave_metas):
        ws, we = meta['start'], meta['end']
        _, idx = Pi_dense[ws:we].topk(max_k, dim=1)
        all_topk[wi] = idx.long()

    # Save dense Pi to CPU for accuracy comparison, free GPU copy
    Pi_dense_cpu = Pi_dense.cpu()
    del Pi_dense, Pibar_dense, Pi_dense_orig
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU freed. CUDA allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Phase 2: Sparse runs (sequential per k) ──
    results = {}
    for k in k_values:
        print(f"\n=== Sparse k={k} ===")

        # Slice precomputed indices to this k
        precomp = {wi: all_topk[wi][:, :k].contiguous() for wi in range(n_waves)}

        def make_sparse_pibar(k_val, precomp_dict):
            def sparse_pibar(Pi_W, wi):
                return compute_pibar_sparse_input_triton_precomputed(
                    Pi_W, tf_T, mt, k_val, precomp_dict[wi])
            return sparse_pibar

        pibar_fn = make_sparse_pibar(k, precomp)

        # Warmup + accuracy (compile Triton for this k)
        print(f"  Warmup k={k}...")
        Pi_s, _, s_iters = run_full_convergence(
            wave_layout, sp_child1, sp_child2, DL_const, Ebar_vec, E_vec,
            SL1_const, SL2_const, leaf_term, pD, pS, S, device, dtype,
            pibar_fn)

        Pi_s_orig = Pi_s[perm]
        sparse_logL = compute_log_likelihood(Pi_s_orig, Eo["E"], root_ids).item()

        # Accuracy vs dense (compare on CPU to save GPU mem)
        Pi_s_cpu = Pi_s.cpu()
        del Pi_s, Pi_s_orig  # free GPU immediately
        gc.collect()
        torch.cuda.empty_cache()

        sig = Pi_dense_cpu > -100.0
        if sig.any():
            pi_diff = torch.abs(Pi_s_cpu - Pi_dense_cpu)[sig]
            pi_max_err = pi_diff.max().item()
            pi_mean_err = pi_diff.mean().item()
        else:
            pi_max_err = pi_mean_err = 0.0
        logL_diff = abs(sparse_logL - dense_logL)
        del Pi_s_cpu

        print(f"  k={k}: logL={sparse_logL:.4f} (diff={logL_diff:.4f}), "
              f"{s_iters} iters")
        print(f"  Pi error: max={pi_max_err:.4f}, mean={pi_mean_err:.6f}")

        # Timed runs — inline, one at a time
        # First: wall-clock (no per-wave sync)
        times = []
        for rep in range(n_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            Pi_t, _, _ = run_inline(
                wave_layout, sp_child1, sp_child2, DL_const, Ebar_vec, E_vec,
                SL1_const, SL2_const, leaf_term, pD, pS, S, device, dtype,
                tf_T, mt, k, precomp)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            del Pi_t
            gc.collect()
            torch.cuda.empty_cache()
            times.append(t1 - t0)
            print(f"  Rep {rep+1} (wall): {times[-1]*1000:.1f} ms")

        # Then: per-wave sync (matches ncu_sparse_pibar.py approach)
        torch.cuda.synchronize()
        t0_total = time.perf_counter()
        Pi_t, _, wave_times = run_inline(
            wave_layout, sp_child1, sp_child2, DL_const, Ebar_vec, E_vec,
            SL1_const, SL2_const, leaf_term, pD, pS, S, device, dtype,
            tf_T, mt, k, precomp, per_wave_sync=True)
        t1_total = time.perf_counter()
        sum_wave = sum(wave_times) * 1000
        wall_sync = (t1_total - t0_total) * 1000
        print(f"  Per-wave sync: sum={sum_wave:.1f} ms, wall={wall_sync:.1f} ms")
        del Pi_t
        gc.collect()
        torch.cuda.empty_cache()

        avg_ms = sum(times) / len(times) * 1000
        min_ms = min(times) * 1000
        print(f"  k={k}: avg={avg_ms:.1f} ms, min={min_ms:.1f} ms")

        results[k] = {
            'logL': sparse_logL,
            'logL_diff': logL_diff,
            'pi_max_err': pi_max_err,
            'pi_mean_err': pi_mean_err,
            'iters': s_iters,
            'avg_ms': avg_ms,
            'min_ms': min_ms,
        }

        # Cleanup
        del precomp
        gc.collect()
        torch.cuda.empty_cache()

    # ── Dense timing (separate phase after sparse is done) ──
    print("\n=== Dense timing ===")
    # Free sparse-only data
    del all_topk
    gc.collect()
    torch.cuda.empty_cache()

    dense_times = []
    for rep in range(n_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        Pi_d, _, _ = run_full_convergence(
            wave_layout, sp_child1, sp_child2, DL_const, Ebar_vec, E_vec,
            SL1_const, SL2_const, leaf_term, pD, pS, S, device, dtype,
            dense_pibar)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        del Pi_d
        gc.collect()
        torch.cuda.empty_cache()
        dense_times.append(t1 - t0)
        print(f"  Rep {rep+1}: {dense_times[-1]*1000:.1f} ms")
    dense_avg = sum(dense_times) / len(dense_times) * 1000
    dense_min = min(dense_times) * 1000

    # ── Summary ──
    print("\n" + "="*70)
    print(f"{'Method':<15} {'logL':>12} {'logL diff':>10} {'Pi max err':>12} "
          f"{'Iters':>6} {'Avg ms':>10} {'Speedup':>8}")
    print("-"*70)
    print(f"{'Dense':<15} {dense_logL:>12.4f} {'—':>10} {'—':>12} "
          f"{dense_iters:>6} {dense_avg:>10.1f} {'1.00x':>8}")
    for k in [16, 32, 64]:
        r = results[k]
        speedup = dense_avg / r['avg_ms']
        print(f"{'Sparse k='+str(k):<15} {r['logL']:>12.4f} {r['logL_diff']:>10.4f} "
              f"{r['pi_max_err']:>12.4f} {r['iters']:>6} {r['avg_ms']:>10.1f} "
              f"{speedup:>7.2f}x")
    print("="*70)

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32


if __name__ == "__main__":
    main()
