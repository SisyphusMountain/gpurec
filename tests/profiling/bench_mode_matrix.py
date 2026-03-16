"""Benchmark all (pibar_mode × param_mode) combinations for converged Pi.

Measures end-to-end time: extract_parameters + E_fixed_point + Pi_wave_forward.
Wave layout is built once and shared (not included in timing).
Runs for S~2000 (test_trees_1000) and S~20000 (test_trees_10000).
"""

import gc
import math
import time
from pathlib import Path

import torch

from src.core.preprocess_cpp import _load_extension
from src.core.extract_parameters import extract_parameters, extract_parameters_uniform
from src.core.likelihood import (
    E_fixed_point,
    Pi_wave_forward,
    compute_log_likelihood,
)
from src.core.scheduling import compute_clade_waves
from src.core.batching import collate_gene_families, collate_wave, build_wave_layout

_INV = 1.0 / math.log(2.0)
_ROOT = Path(__file__).resolve().parent.parent
D, L, T = 0.05, 0.05, 0.05

PIBAR_MODES = ['dense', 'uniform_approx', 'uniform', 'topk']
PARAM_MODES = ['global', 'specieswise', 'pairwise']

# pairwise not supported in Pi_wave_forward
VALID_COMBOS = {
    ('dense', 'global'), ('dense', 'specieswise'),
    ('uniform_approx', 'global'), ('uniform_approx', 'specieswise'),
    ('uniform', 'global'), ('uniform', 'specieswise'),
    ('topk', 'global'), ('topk', 'specieswise'),
}


def _load_data(ds_name, n_families=1, device=None, dtype=torch.float32):
    """Load and preprocess data. Keeps [S,S] matrices on CPU to save GPU memory."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ext = _load_extension()
    data_dir = _ROOT / "data" / ds_name
    sp_path = str(data_dir / "sp.nwk")

    gene_paths = sorted(data_dir.glob("g_*.nwk"))
    if not gene_paths:
        gene_paths = sorted(data_dir.glob("gene_*.nwk"))
    gene_paths = gene_paths[:n_families]

    batch_items = []
    sr = None
    for gp in gene_paths:
        raw = ext.preprocess(sp_path, [str(gp)])
        if sr is None:
            sr = raw['species']
        cr = raw['ccp']
        ch = {
            "split_leftrights_sorted": cr["split_leftrights_sorted"],
            "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=dtype) * _INV,
            "seg_parent_ids": cr["seg_parent_ids"],
            "ptr_ge2": cr["ptr_ge2"],
            "num_segs_ge2": int(cr["num_segs_ge2"]),
            "num_segs_eq1": int(cr["num_segs_eq1"]),
            "end_rows_ge2": int(cr["end_rows_ge2"]),
            "C": int(cr["C"]),
            "N_splits": int(cr["N_splits"]),
        }
        if "split_parents_sorted" in cr:
            ch["split_parents_sorted"] = cr["split_parents_sorted"]
        batch_items.append({
            "ccp": ch,
            "leaf_row_index": raw["leaf_row_index"].long(),
            "leaf_col_index": raw["leaf_col_index"].long(),
            "root_clade_id": int(cr["root_clade_id"]),
        })

    S = int(sr["S"])

    # Species helpers: only small tensors on GPU; [S,S] stays on CPU
    sh_base = {
        "S": S,
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
    }

    # CPU-only [S,S] data
    recipients_mat_cpu = sr["Recipients_mat"].to(dtype=dtype)
    ancestors_dense_cpu = sr["ancestors_dense"].to(dtype=dtype)

    # unnorm_row_max on GPU (only [S] vector)
    tm_unnorm_cpu = torch.log2(recipients_mat_cpu)
    unnorm_row_max = tm_unnorm_cpu.max(dim=-1).values.to(device=device, dtype=dtype)

    # Build wave layout once
    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    root_clade_ids = batched['root_clade_ids']
    families_waves, families_phases = [], []
    for bi in batch_items:
        w, p = compute_clade_waves(bi['ccp'])
        families_waves.append(w)
        families_phases.append(p)
    offsets = [m['clade_offset'] for m in batched['family_meta']]
    cross_waves = collate_wave(families_waves, offsets)
    max_n = max(len(p) for p in families_phases)
    cross_phases = [max(fp[k] if k < len(fp) else 1 for fp in families_phases) for k in range(max_n)]
    wave_layout = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=batched['ccp'],
        leaf_row_index=batched['leaf_row_index'],
        leaf_col_index=batched['leaf_col_index'],
        root_clade_ids=root_clade_ids,
        device=device, dtype=dtype,
    )

    return {
        'sh_base': sh_base,
        'S': S,
        'recipients_mat_cpu': recipients_mat_cpu,
        'ancestors_dense_cpu': ancestors_dense_cpu,
        'tm_unnorm_cpu': tm_unnorm_cpu,
        'unnorm_row_max': unnorm_row_max,
        'wave_layout': wave_layout,
        'root_clade_ids': root_clade_ids,
        'device': device,
        'dtype': dtype,
    }


def _run_forward(data, pibar_mode, param_mode):
    """Run full forward: extract_params → E_fixed_point → Pi_wave_forward → logL."""
    S = data['S']
    device = data['device']
    dtype = data['dtype']
    sh_base = data['sh_base']
    wave_layout = data['wave_layout']
    root_clade_ids = data['root_clade_ids']
    unnorm_row_max = data['unnorm_row_max']
    specieswise = (param_mode == 'specieswise')

    # --- Build species_helpers for this mode ---
    sh = dict(sh_base)

    # --- theta ---
    if specieswise:
        theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device)).unsqueeze(0).expand(S, -1).contiguous()
    else:
        theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))

    # --- extract parameters + mode-specific GPU tensors ---
    ancestors_T = None
    if pibar_mode == 'uniform_approx':
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters_uniform(
            theta, unnorm_row_max, specieswise=specieswise,
        )
    elif pibar_mode == 'uniform':
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters_uniform(
            theta, unnorm_row_max, specieswise=specieswise,
        )
        anc_dense = data['ancestors_dense_cpu'].to(device=device, dtype=dtype)
        sh['ancestors_dense'] = anc_dense
        ancestors_T = anc_dense.T.to_sparse_coo()
    else:  # dense or topk
        sh['Recipients_mat'] = data['recipients_mat_cpu'].to(device=device, dtype=dtype)
        tm_unnorm_gpu = data['tm_unnorm_cpu'].to(device=device, dtype=dtype)
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
            theta, tm_unnorm_gpu,
            genewise=False, specieswise=specieswise, pairwise=False,
        )
        if max_transfer_mat.ndim == 2:
            max_transfer_mat = max_transfer_mat.squeeze(-1)

    # --- E_fixed_point ---
    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=max_transfer_mat,
        max_iters=2000, tolerance=1e-8, warm_start_E=None,
        dtype=dtype, device=device, pibar_mode=pibar_mode,
        ancestors_T=ancestors_T,
    )

    # --- Pi_wave_forward ---
    Pi_out = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sh,
        E=E_out['E'], Ebar=E_out['E_bar'],
        E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=max_transfer_mat,
        device=device, dtype=dtype, pibar_mode=pibar_mode,
    )

    logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_ids)
    return logL


def _free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def bench_dataset(ds_name, n_families, n_warmup=2, n_trials=5):
    """Benchmark all valid mode combinations on a dataset."""
    print(f"\n{'='*70}")
    print(f"Dataset: {ds_name}, {n_families} family(ies)")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = _load_data(ds_name, n_families=n_families, device=device)
    S = data['S']
    print(f"S = {S}")

    results = {}

    for pibar_mode in PIBAR_MODES:
        for param_mode in PARAM_MODES:
            key = (pibar_mode, param_mode)
            if key not in VALID_COMBOS:
                results[key] = None
                continue

            _free_gpu()

            # Warmup
            ok = True
            for _ in range(n_warmup):
                try:
                    _run_forward(data, pibar_mode, param_mode)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                except Exception as e:
                    print(f"  {pibar_mode:20s} × {param_mode:12s}: FAILED ({e})")
                    results[key] = 'FAILED'
                    ok = False
                    break
            if not ok:
                _free_gpu()
                continue

            # Timed trials
            times = []
            logL_val = None
            for _ in range(n_trials):
                _free_gpu()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                logL = _run_forward(data, pibar_mode, param_mode)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                logL_val = logL.item()

            median_ms = sorted(times)[len(times) // 2] * 1000
            results[key] = (median_ms, logL_val)
            print(f"  {pibar_mode:20s} × {param_mode:12s}: {median_ms:8.1f} ms  (logL={logL_val:.4f})")

            _free_gpu()

    # Print markdown table
    print(f"\n--- Markdown table (S={S}) ---")
    header = "| pibar_mode \\ param_mode | " + " | ".join(pm for pm in PARAM_MODES) + " |"
    sep = "|---|" + "|".join(["---"] * len(PARAM_MODES)) + "|"
    print(header)
    print(sep)
    for pibar_mode in PIBAR_MODES:
        cells = []
        for param_mode in PARAM_MODES:
            key = (pibar_mode, param_mode)
            v = results.get(key)
            if v is None:
                cells.append("N/A")
            elif v == 'FAILED':
                cells.append("OOM")
            else:
                cells.append(f"{v[0]:.1f} ms")
        print(f"| **{pibar_mode}** | " + " | ".join(cells) + " |")

    return results, S


if __name__ == "__main__":
    r1, s1 = bench_dataset("test_trees_1000", n_families=1, n_warmup=3, n_trials=7)
    _free_gpu()
    r2, s2 = bench_dataset("test_trees_10000", n_families=1, n_warmup=2, n_trials=5)
