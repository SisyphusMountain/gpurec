"""
Validate global / specieswise / genewise optimization on test_trees_100.

Runs L-BFGS to convergence in all three modes and compares final NLL
against AleRax reference values.

Usage:
    python experiments/validate_three_modes.py [--steps N] [--families N]

AleRax reference (ln-space, summed over 100 families):
    global      -9737.07
    genewise    -9573.40
    specieswise -8957.65
"""

import argparse
import math
import time
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "test_trees_100"
import math as _math

# AleRax reports in ln-space; we compute in log2.
# NLL_log2 = NLL_ln / ln(2)
_LN2 = _math.log(2)
ALERAX_NLL_LN = {
    "global":      9737.07,
    "genewise":    9573.40,
    "specieswise": 8957.65,
}
ALERAX_NLL = {k: v / _LN2 for k, v in ALERAX_NLL_LN.items()}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_wave_layout(ds, device, dtype):
    """Build cross-family wave layout for optimize_theta_wave."""
    from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
    from gpurec.core.scheduling import compute_clade_waves

    items = [
        {
            "ccp": fam["ccp_helpers"],
            "leaf_row_index": fam["leaf_row_index"],
            "leaf_col_index": fam["leaf_col_index"],
            "root_clade_id": int(fam["root_clade_id"]),
        }
        for fam in ds.families
    ]
    batched = collate_gene_families(items, dtype=dtype, device=device)

    families_waves, families_phases = [], []
    for fam in ds.families:
        w, p = compute_clade_waves(fam["ccp_helpers"])
        families_waves.append(w)
        families_phases.append(p)

    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(families_waves, offsets)

    max_n_waves = max(len(p) for p in families_phases)
    cross_phases = []
    for k in range(max_n_waves):
        phase_k = 1
        for fp in families_phases:
            if k < len(fp):
                phase_k = max(phase_k, fp[k])
        cross_phases.append(phase_k)

    family_clade_counts  = [m["C"]            for m in batched["family_meta"]]
    family_clade_offsets = [m["clade_offset"] for m in batched["family_meta"]]

    wave_layout = build_wave_layout(
        waves=cross_waves,
        phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device,
        dtype=dtype,
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
    )
    return wave_layout, batched["root_clade_ids"]


def _sp_helpers_gpu(ds, device, dtype):
    return {
        k: (
            v.to(device=device, dtype=dtype) if torch.is_tensor(v) and v.is_floating_point()
            else v.to(device=device) if torch.is_tensor(v)
            else v
        )
        for k, v in ds.species_helpers.items()
    }


def _report(mode, result, alerax_nll, elapsed):
    nll = result["negative_log_likelihood"]
    gap = nll - alerax_nll
    rates = result["rates"]
    n_steps = len(result["history"])
    print(f"\n{'─'*60}")
    print(f"  Mode: {mode}")
    print(f"  Steps: {n_steps}   Time: {elapsed:.1f}s")
    print(f"  NLL  our={nll:.3f}   AleRax={alerax_nll:.3f}   gap={gap:+.3f}")
    if rates.dim() == 1:
        # rates = 2^theta (already actual rates)
        print(f"  Rates  D={rates[0]:.5f}  L={rates[1]:.5f}  T={rates[2]:.5f}")
    elif rates.dim() == 2:  # [S,3] specieswise
        mean_r = rates.mean(0)
        print(f"  Rates (mean over species)  D={mean_r[0]:.5f}  L={mean_r[1]:.5f}  T={mean_r[2]:.5f}")


def _report_genewise(result, alerax_nll, elapsed):
    total_nll = float(result["nll"].sum())
    gap = total_nll - alerax_nll
    n_steps = len(result["history"])
    rates = result["rates"]  # [G, 3]
    mean_r = rates.mean(0)
    print(f"\n{'─'*60}")
    print(f"  Mode: genewise")
    print(f"  Steps: {n_steps}   Time: {elapsed:.1f}s")
    print(f"  NLL  our={total_nll:.3f}   AleRax={alerax_nll:.3f}   gap={gap:+.3f}")
    print(f"  Rates (mean over families)  D={mean_r[0]:.5f}  L={mean_r[1]:.5f}  T={mean_r[2]:.5f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Max L-BFGS steps")
    parser.add_argument("--families", type=int, default=0,
                        help="Use only first N families (0 = all 100)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    dtype  = torch.float32

    from gpurec.core.model import GeneDataset
    from gpurec.optimization.theta_optimizer import optimize_theta_wave, optimize_theta_genewise

    sp_path   = str(DATA_DIR / "sp.nwk")
    gene_paths = sorted(DATA_DIR.glob("g_*.nwk"))
    if args.families > 0:
        gene_paths = gene_paths[:args.families]
    gene_paths = [str(g) for g in gene_paths]
    G = len(gene_paths)
    print(f"Loaded {G} families from {DATA_DIR.name}")

    # ── Global ────────────────────────────────────────────────────────────────
    print("\n[1/3] Global optimization (3 shared parameters) ...")
    ds_global = GeneDataset(sp_path, gene_paths,
                            genewise=False, specieswise=False, pairwise=False,
                            dtype=dtype, device=device)
    S = ds_global.S

    wl, root_ids = _build_wave_layout(ds_global, device, dtype)
    sp_helpers = _sp_helpers_gpu(ds_global, device, dtype)
    theta_init_global = math.log2(0.02) * torch.ones(3, dtype=dtype, device=device)

    t0 = time.time()
    result_global = optimize_theta_wave(
        wave_layout=wl,
        species_helpers=sp_helpers,
        root_clade_ids=root_ids,
        unnorm_row_max=ds_global.unnorm_row_max.to(device=device, dtype=dtype),
        theta_init=theta_init_global,
        steps=args.steps,
        optimizer="lbfgs",
        specieswise=False,
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
    )
    _report("global", result_global, ALERAX_NLL["global"], time.time() - t0)

    # ── Specieswise ───────────────────────────────────────────────────────────
    print("\n[2/3] Specieswise optimization (3 parameters per species) ...")
    ds_sw = GeneDataset(sp_path, gene_paths,
                        genewise=False, specieswise=True, pairwise=False,
                        dtype=dtype, device=device)

    wl_sw, root_ids_sw = _build_wave_layout(ds_sw, device, dtype)
    sp_helpers_sw = _sp_helpers_gpu(ds_sw, device, dtype)
    # Initialize all species at global AleRax optimum
    global_rates = result_global["rates"].to(device=device, dtype=dtype)  # [3]
    theta_init_sw = global_rates.unsqueeze(0).expand(S, -1).clone()       # [S, 3]

    t0 = time.time()
    result_sw = optimize_theta_wave(
        wave_layout=wl_sw,
        species_helpers=sp_helpers_sw,
        root_clade_ids=root_ids_sw,
        unnorm_row_max=ds_sw.unnorm_row_max.to(device=device, dtype=dtype),
        theta_init=theta_init_sw,
        steps=args.steps,
        optimizer="lbfgs",
        specieswise=True,
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
    )
    _report("specieswise", result_sw, ALERAX_NLL["specieswise"], time.time() - t0)

    # ── Genewise ──────────────────────────────────────────────────────────────
    print("\n[3/3] Genewise optimization (3 parameters per gene family) ...")
    ds_gw = GeneDataset(sp_path, gene_paths,
                        genewise=True, specieswise=False, pairwise=False,
                        dtype=dtype, device=device)

    # Initialize all genes at global AleRax optimum
    theta_init_gw = global_rates.unsqueeze(0).expand(G, -1).clone()  # [G, 3]

    t0 = time.time()
    result_gw = optimize_theta_genewise(
        families=ds_gw.families,
        species_helpers=ds_gw.species_helpers,
        unnorm_row_max=ds_gw.unnorm_row_max,
        theta_init=theta_init_gw,
        max_steps=args.steps,
        lbfgs_m=10,
        grad_tol=1e-5,
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
    )
    _report_genewise(result_gw, ALERAX_NLL["genewise"], time.time() - t0)

    print(f"\n{'═'*60}")
    print("Summary (NLL gap = our − AleRax, negative = we beat AleRax):")
    print(f"  global:      {result_global['negative_log_likelihood'] - ALERAX_NLL['global']:+.3f}")
    print(f"  specieswise: {result_sw['negative_log_likelihood']     - ALERAX_NLL['specieswise']:+.3f}")
    print(f"  genewise:    {float(result_gw['nll'].sum())            - ALERAX_NLL['genewise']:+.3f}")


if __name__ == "__main__":
    main()
