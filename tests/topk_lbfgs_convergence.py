#!/usr/bin/env python3
"""Test topk pibar_mode: forward accuracy vs dense, and L-BFGS-B convergence.

Uniform transfer rates (non-pairwise), test_trees_1000 (S=1999).
"""

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent  # tests/
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.preprocess_cpp import _load_extension
from src.core.extract_parameters import extract_parameters
from src.core.likelihood import (
    E_fixed_point,
    Pi_wave_forward,
    compute_log_likelihood,
)
from src.core.scheduling import compute_clade_waves
from src.core.batching import (
    collate_gene_families,
    collate_wave,
    build_wave_layout,
)
from src.optimization.theta_optimizer import optimize_theta_wave

_INV = 1.0 / math.log(2.0)
D, L, T = 0.05, 0.05, 0.05
TOL = 1e-3


def setup(ds_name, n_families, device, dtype):
    ext = _load_extension()
    data_dir = TEST_ROOT / "data" / ds_name
    assert data_dir.exists(), f"{ds_name} not found"
    sp_path = str(data_dir / "sp.nwk")
    gene_paths = sorted(data_dir.glob("g_*.nwk"))[:n_families]

    batch_items = []
    sr = None
    for gp in gene_paths:
        raw = ext.preprocess(sp_path, [str(gp)])
        if sr is None:
            sr = raw["species"]
        cr = raw["ccp"]
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

    sh = {
        "S": int(sr["S"]),
        "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device=device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device=device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    }

    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=device))
    tm_unnorm = torch.log2(sh["Recipients_mat"])
    log_pS, log_pD, log_pL, transfer_mat, mt_raw = extract_parameters(
        theta, tm_unnorm, genewise=False, specieswise=False, pairwise=False,
    )
    mt = mt_raw.squeeze(-1) if mt_raw.ndim == 2 else mt_raw

    E_out = E_fixed_point(
        species_helpers=sh, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt,
        max_iters=2000, tolerance=1e-8, warm_start_E=None,
        dtype=dtype, device=device,
    )

    # Build wave layout
    batched = collate_gene_families(batch_items, dtype=dtype, device=device)
    root_clade_ids = batched["root_clade_ids"]

    families_waves, families_phases = [], []
    for bi in batch_items:
        w, p = compute_clade_waves(bi["ccp"])
        families_waves.append(w)
        families_phases.append(p)

    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(families_waves, offsets)
    max_n = max(len(p) for p in families_phases)
    cross_phases = [
        max(fp[k] if k < len(fp) else 1 for fp in families_phases)
        for k in range(max_n)
    ]

    wave_layout = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=root_clade_ids,
        device=device, dtype=dtype,
    )

    return {
        "sh": sh, "E_out": E_out,
        "log_pS": log_pS, "log_pD": log_pD, "log_pL": log_pL,
        "transfer_mat": transfer_mat, "mt": mt,
        "wave_layout": wave_layout,
        "root_clade_ids": root_clade_ids,
        "theta": theta, "tm_unnorm": tm_unnorm,
        "device": device, "dtype": dtype,
    }


def run_forward(d, pibar_mode, topk_k=16):
    """Run Pi_wave_forward with given pibar_mode, return per-family logL."""
    Pi_out = Pi_wave_forward(
        wave_layout=d["wave_layout"], species_helpers=d["sh"],
        E=d["E_out"]["E"], Ebar=d["E_out"]["E_bar"],
        E_s1=d["E_out"]["E_s1"], E_s2=d["E_out"]["E_s2"],
        log_pS=d["log_pS"], log_pD=d["log_pD"], log_pL=d["log_pL"],
        transfer_mat=d["transfer_mat"], max_transfer_mat=d["mt"],
        device=d["device"], dtype=d["dtype"],
        pibar_mode=pibar_mode, topk_k=topk_k,
    )
    logL = compute_log_likelihood(Pi_out["Pi"], d["E_out"]["E"], d["root_clade_ids"])
    return logL, Pi_out


def main():
    device = torch.device("cuda")
    dtype = torch.float64
    ds_name = "test_trees_100"
    n_fam = 3

    print(f"=== Setup: {ds_name}, {n_fam} families, dtype={dtype} ===")
    d = setup(ds_name, n_fam, device, dtype)
    S = d["sh"]["S"]
    print(f"  S = {S}")

    # ---------------------------------------------------------------
    # Part 1: Forward accuracy — topk vs dense
    # ---------------------------------------------------------------
    print("\n=== Part 1: Forward accuracy (topk vs dense) ===")

    logL_dense, _ = run_forward(d, "dense")
    print(f"  Dense logL:  {[f'{x:.4f}' for x in logL_dense.tolist()]}")

    for k in [8, 16, 32, 64]:
        logL_topk, _ = run_forward(d, "topk", topk_k=k)
        diffs = (logL_topk - logL_dense).abs()
        rel_diffs = diffs / logL_dense.abs()
        print(f"  topk k={k:3d}: logL={[f'{x:.4f}' for x in logL_topk.tolist()]}  "
              f"max_abs_diff={diffs.max():.4e}  max_rel_diff={rel_diffs.max():.4e}")

    # Also compare to uniform_approx (the usual fast path)
    logL_ua, _ = run_forward(d, "uniform_approx")
    diffs_ua = (logL_ua - logL_dense).abs()
    print(f"  uniform_approx: max_abs_diff={diffs_ua.max():.4e}  "
          f"max_rel_diff={(diffs_ua / logL_dense.abs()).max():.4e}")

    # ---------------------------------------------------------------
    # Part 2: L-BFGS-B convergence with topk
    # ---------------------------------------------------------------
    print("\n=== Part 2: L-BFGS-B convergence ===")

    # Start from a slightly perturbed theta so there's something to optimize
    theta_init = d["theta"].clone()
    theta_init[0] += 0.5  # perturb D
    theta_init[2] -= 0.3  # perturb T

    tm_unnorm = d["tm_unnorm"].to(device=device, dtype=dtype)
    unnorm_row_max = tm_unnorm.max(dim=-1).values

    for pibar_mode in ["dense", "topk"]:
        print(f"\n--- pibar_mode={pibar_mode} ---")
        print(f"  theta_init = {theta_init.tolist()}")

        result = optimize_theta_wave(
            wave_layout=d["wave_layout"],
            species_helpers=d["sh"],
            root_clade_ids=d["root_clade_ids"],
            unnorm_row_max=unnorm_row_max,
            theta_init=theta_init.clone(),
            transfer_mat_unnormalized=tm_unnorm,
            steps=50,
            e_max_iters=2000,
            e_tol=1e-8,
            neumann_terms=4,
            use_pruning=False,
            cg_tol=1e-10,
            cg_maxiter=1000,
            specieswise=False,
            device=device,
            dtype=dtype,
            pibar_mode=pibar_mode,
            optimizer="lbfgs",
        )

        history = result["history"]
        scipy_result = result.get("scipy_result")

        print(f"  Final theta = {result['theta'].tolist()}")
        print(f"  Final rates = {result['rates'].tolist()}")
        print(f"  Final NLL   = {result['negative_log_likelihood']:.6f}")
        if scipy_result is not None:
            print(f"  scipy: success={scipy_result.success}, "
                  f"message={scipy_result.message}, nfev={scipy_result.nfev}")

        print(f"  History ({len(history)} evals):")
        for i, h in enumerate(history):
            print(f"    {i:3d}: NLL={h.negative_log_likelihood:12.6f}  "
                  f"|grad|={h.grad_infinity_norm:.4e}  "
                  f"rates=[{', '.join(f'{r:.6e}' for r in h.rates.tolist())}]")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
