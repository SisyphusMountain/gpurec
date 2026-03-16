"""Profile the fused uniform-Pibar kernel at large S.

Purpose:
    Measure where time is spent in compute_likelihood_batch(pibar_mode='uniform_approx')
    for large species trees (S~20K), after introducing the fused Triton kernel
    `wave_step_uniform_fused` which combines Pibar computation, DTS_L terms,
    logsumexp, and convergence checking into a single kernel launch per iteration.

Background:
    Before the fused kernel, the per-iteration self-loop had:
      1. _compute_Pibar_inline: 4 PyTorch ops (max, exp2, sum, log2+add) → [W,S] Pibar
      2. wave_step_fused: Triton kernel reading Pi + Pibar → [W,S] Pi_new
      3. Convergence check: allocate bool + diff tensors, GPU sync via .item()
    Total: ~5-6 kernel launches per iteration, ~2.2s for 232 iterations at S=20K.

    The fused kernel reduces this to 1 kernel launch per iteration:
      - Online max+sum (pass 1) + inline Pibar + DTS_L + convergence (pass 2)
      - Writes Pibar to global tensor in-place, no intermediate [W,S] allocation
      - Returns per-row max_diff for convergence (small [W] tensor)

Approach:
    1. High-level profiling: time each stage of compute_likelihood_batch.
    2. Inner-loop profiling: monkey-patch key functions to count calls and
       measure cumulative time (with per-call GPU sync for accuracy).
    3. End-to-end timing: multiple trials for stable measurement.

Usage:
    python -m tests.profiling.profile_fused_kernel [--n_families N] [--dataset DIR]
"""

import argparse
import glob
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@dataclass
class Timer:
    """Accumulates wall-clock time for named stages, with optional GPU sync."""
    times: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    @contextmanager
    def section(self, name: str, sync: bool = True):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if sync:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.times[name] = self.times.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self):
        self.times.clear()
        self.counts.clear()

    def report(self, title: str = "Timing report"):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        total = sum(self.times.values())
        for name in self.times:
            t = self.times[name]
            n = self.counts[name]
            pct = 100 * t / total if total > 0 else 0
            avg = t / n if n > 1 else t
            line = f"  {name:40s} {t:8.3f}s  ({pct:5.1f}%)"
            if n > 1:
                line += f"  [{n} calls, {avg*1000:.2f}ms avg]"
            print(line)
        print(f"  {'TOTAL':40s} {total:8.3f}s")
        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# High-level stage profiling
# ---------------------------------------------------------------------------

def profiled_likelihood(ds, pibar_mode='uniform_approx', timer=None):
    """Run compute_likelihood_batch with per-stage timing.

    Re-implements the steps of GeneDataset.compute_likelihood_batch
    with timing instrumentation around each major stage.
    """
    from src.core.extract_parameters import extract_parameters
    from src.core.likelihood import (
        E_fixed_point, Pi_wave_forward, compute_log_likelihood,
    )
    from src.core.batching import collate_gene_families, collate_wave, build_wave_layout
    from src.core.scheduling import compute_clade_waves

    device = ds.device
    dtype = torch.float32
    indices = list(range(len(ds.families)))
    T = timer or Timer()

    # A. Collate + extract parameters
    with T.section("A. collate + extract_params"):
        batch_items = [{
            'ccp': ds.families[i]['ccp_helpers'],
            'leaf_row_index': ds.families[i]['leaf_row_index'],
            'leaf_col_index': ds.families[i]['leaf_col_index'],
            'root_clade_id': int(ds.families[i]['root_clade_id']),
        } for i in indices]
        batched = collate_gene_families(batch_items, dtype=dtype, device=device)
        ccp_helpers = batched['ccp']
        leaf_row_index = batched['leaf_row_index']
        leaf_col_index = batched['leaf_col_index']
        root_clade_ids = batched['root_clade_ids']

        transfer_mat_unnorm = ds.tr_mat_unnormalized.to(device=device, dtype=dtype)
        theta0 = ds.families[0]['theta'].to(device=device, dtype=dtype)
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
            theta0, transfer_mat_unnorm, genewise=False, specieswise=False, pairwise=False)
        max_transfer_vec = (max_transfer_mat.squeeze(-1)
                            if max_transfer_mat.ndim == 2 else max_transfer_mat)

    # A2. Species helpers to GPU
    with T.section("A2. species_helpers to GPU"):
        _skip = {'ancestors_dense', 'Recipients_mat'} if pibar_mode == 'uniform_approx' else set()
        def _mv(t):
            return t.to(device=device, dtype=dtype) if t.dtype.is_floating_point else t.to(device)
        species_helpers = {
            k: (_mv(v) if torch.is_tensor(v) and k not in _skip else v)
            for k, v in ds.species_helpers.items()
        }

    # B. E step
    with T.section("B. E_step"):
        E_out = E_fixed_point(
            species_helpers=species_helpers, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=transfer_mat, max_transfer_mat=max_transfer_vec,
            max_iters=200, tolerance=1e-3, warm_start_E=None, dtype=dtype, device=device)
        E, E_s1, E_s2, Ebar = E_out['E'], E_out['E_s1'], E_out['E_s2'], E_out['E_bar']
        if pibar_mode == 'uniform_approx':
            del transfer_mat; del transfer_mat_unnorm; transfer_mat = None

    # C. Wave scheduling
    with T.section("C. wave scheduling"):
        families_waves, families_phases = [], []
        for idx in indices:
            fam = ds.families[idx]
            w, p = compute_clade_waves(fam['ccp_helpers'])
            families_waves.append(w); families_phases.append(p)
        offsets = [m['clade_offset'] for m in batched['family_meta']]
        cross_waves = collate_wave(families_waves, offsets)
        max_n_waves = max(len(p) for p in families_phases)
        cross_phases = [max(fp[k] for fp in families_phases if k < len(fp))
                        for k in range(max_n_waves)]

    # C2. build_wave_layout
    with T.section("C2. build_wave_layout"):
        wave_layout = build_wave_layout(
            waves=cross_waves, phases=cross_phases, ccp_helpers=ccp_helpers,
            leaf_row_index=leaf_row_index, leaf_col_index=leaf_col_index,
            root_clade_ids=root_clade_ids, device=device, dtype=dtype)

    # D. Pi wave forward
    with T.section("D. Pi_wave_forward"):
        Pi_out = Pi_wave_forward(
            wave_layout=wave_layout, species_helpers=species_helpers,
            E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=transfer_mat, max_transfer_mat=max_transfer_vec,
            device=device, dtype=dtype, local_iters=200, local_tolerance=1e-3,
            pibar_mode=pibar_mode)

    # E. Log-likelihood
    with T.section("E. compute_log_likelihood"):
        logL_vec = compute_log_likelihood(Pi_out['Pi'], E, root_clade_ids)
        logLs = [float(x) for x in logL_vec.detach().cpu().tolist()]

    return logLs, T


# ---------------------------------------------------------------------------
# Inner-loop profiling via monkey-patching
# ---------------------------------------------------------------------------

def profiled_inner_loop(ds, pibar_mode='uniform_approx'):
    """Profile Pi_wave_forward's inner loop by patching key functions.

    Patches:
      - _compute_dts_cross: counts calls, measures cumulative time
      - wave_step_uniform_fused: counts calls, separates kernel vs .item() time

    IMPORTANT: sync-per-call timing serializes GPU work, so measured kernel
    times represent true execution time but the total will be slower than
    un-instrumented runs due to lost pipelining.
    """
    import src.core.likelihood as L
    import src.core.kernels.wave_step as ws_mod

    # Save originals
    _orig_dts = L._compute_dts_cross
    _orig_fused = ws_mod.wave_step_uniform_fused

    # Counters
    stats = {
        'dts_calls': 0, 'dts_time': 0.0, 'dts_sizes': [],
        'fused_calls': 0, 'fused_kernel_time': 0.0, 'fused_item_time': 0.0,
    }

    def patched_dts(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig_dts(*args, **kwargs)
        torch.cuda.synchronize()
        stats['dts_time'] += time.perf_counter() - t0
        stats['dts_calls'] += 1
        meta = args[2]
        stats['dts_sizes'].append(len(meta['sl']))
        return result

    def patched_fused(Pi, Pibar, ws_val, W, S, mt, DL, Ebar, E, SL1, SL2,
                      sp1, sp2, lwt, DTS_reduced=None):
        import triton
        Pi_new = torch.empty((W, S), dtype=Pi.dtype, device=Pi.device)
        max_diff_buf = torch.empty(W, dtype=torch.float32, device=Pi.device)
        has_splits = DTS_reduced is not None
        BLOCK_S = min(256, triton.next_power_of_2(S))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ws_mod._wave_step_uniform_kernel[(W,)](
            Pi, ws_val, mt, DL, Ebar, E, SL1, SL2, sp1, sp2,
            lwt, DTS_reduced if has_splits else lwt, has_splits,
            Pi_new, Pibar, max_diff_buf,
            S, stride=S, BLOCK_S=BLOCK_S, num_warps=4)
        torch.cuda.synchronize()
        t_kernel = time.perf_counter() - t0

        t0 = time.perf_counter()
        max_diff = max_diff_buf.max().item()
        t_item = time.perf_counter() - t0

        stats['fused_kernel_time'] += t_kernel
        stats['fused_item_time'] += t_item
        stats['fused_calls'] += 1
        return Pi_new, max_diff

    # Patch
    L._compute_dts_cross = patched_dts
    L.wave_step_uniform_fused = patched_fused

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logLs = ds.compute_likelihood_batch(
            chunk_size=1,
            max_iters_Pi=200, tol_Pi=1e-3, pibar_mode=pibar_mode)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        # Restore originals
        L._compute_dts_cross = _orig_dts
        L.wave_step_uniform_fused = _orig_fused

    print(f"\n{'=' * 60}")
    print(f"  Inner-loop profiling (sync per call)")
    print(f"{'=' * 60}")
    print(f"  Total wall time:           {total:.3f}s")
    print(f"  DTS cross:                 {stats['dts_time']:.3f}s  [{stats['dts_calls']} calls]")
    if stats['dts_sizes']:
        top5 = sorted(stats['dts_sizes'], reverse=True)[:5]
        print(f"    Largest DTS sizes:       {top5}")
    print(f"  Fused kernel (GPU):        {stats['fused_kernel_time']:.3f}s  [{stats['fused_calls']} calls]")
    print(f"    avg per call:            {stats['fused_kernel_time']/max(1,stats['fused_calls'])*1000:.2f}ms")
    print(f"  Fused .item() sync:        {stats['fused_item_time']:.3f}s")
    accounted = stats['dts_time'] + stats['fused_kernel_time'] + stats['fused_item_time']
    print(f"  Accounted:                 {accounted:.3f}s")
    print(f"  Unaccounted:               {total - accounted:.3f}s")
    print(f"    (E-step, wave sched, build_layout, copies, leaf_wt, Python overhead)")
    print(f"{'=' * 60}")
    return logLs, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile fused uniform kernel")
    parser.add_argument("--n_families", type=int, default=1)
    parser.add_argument("--dataset", default="test_trees_10000",
                        help="Dataset dir under tests/data/")
    parser.add_argument("--pibar_mode", default="uniform_approx", choices=["uniform_approx", "dense", "uniform", "topk"])
    parser.add_argument("--skip_warmup", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.join("tests/data", args.dataset)
    sp_path = os.path.join(data_dir, "sp.nwk")
    gene_files = sorted(glob.glob(os.path.join(data_dir, "gene_*.nwk")))
    if not gene_files:
        gene_files = sorted(glob.glob(os.path.join(data_dir, "g_*.nwk")))
    gene_files = gene_files[:args.n_families]
    print(f"Dataset: {data_dir}")
    print(f"Families: {len(gene_files)}")
    print(f"pibar_mode: {args.pibar_mode}")

    from src.core.model import GeneDataset
    print("Loading GeneDataset...")
    t0 = time.time()
    ds = GeneDataset(sp_path, gene_files, genewise=False, specieswise=False, pairwise=False,
                     dtype=torch.float32, device=torch.device('cuda'))
    print(f"  Init: {time.time()-t0:.1f}s, S={ds.S}")

    # Warmup
    if not args.skip_warmup:
        print("Warmup (JIT compile)...")
        ds.compute_likelihood_batch(chunk_size=1,
                                     max_iters_Pi=200, tol_Pi=1e-3,
                                     pibar_mode=args.pibar_mode)
        torch.cuda.synchronize()

    # --- Test 1: High-level stage profiling ---
    print("\n" + "#" * 60)
    print("  TEST 1: High-level stage profiling")
    print("#" * 60)
    logLs, T = profiled_likelihood(ds, pibar_mode=args.pibar_mode)
    T.report("High-level stages")
    print(f"logLs: {logLs[:5]}")

    # --- Test 2: Inner-loop profiling ---
    print("\n" + "#" * 60)
    print("  TEST 2: Inner-loop profiling (sync per call)")
    print("#" * 60)
    logLs2, stats = profiled_inner_loop(ds, pibar_mode=args.pibar_mode)
    print(f"logLs: {logLs2[:5]}")

    # Consistency check
    max_diff = max(abs(a - b) for a, b in zip(logLs, logLs2))
    print(f"logL consistency check (Test1 vs Test2): max_diff = {max_diff:.6f}")

    # --- Test 3: Stable end-to-end timing ---
    print("\n" + "#" * 60)
    print("  TEST 3: End-to-end timing (5 trials)")
    print("#" * 60)
    for trial in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logLs = ds.compute_likelihood_batch(
            chunk_size=1,
            max_iters_Pi=200, tol_Pi=1e-3,
            pibar_mode=args.pibar_mode)
        torch.cuda.synchronize()
        print(f"  Trial {trial}: {time.perf_counter()-t0:.3f}s")

    # --- Summary ---
    S = ds.S
    C = sum(f['C'] for f in ds.families)
    print(f"\n{'=' * 60}")
    print(f"  Summary: S={S}, C={C}, {len(gene_files)} families")
    print(f"  Fused kernel: {stats['fused_calls']} calls, "
          f"{stats['fused_kernel_time']*1000:.0f}ms GPU time, "
          f"{stats['fused_kernel_time']/max(1,stats['fused_calls'])*1000:.2f}ms/call")
    print(f"  DTS cross: {stats['dts_calls']} calls, "
          f"{stats['dts_time']*1000:.0f}ms")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
