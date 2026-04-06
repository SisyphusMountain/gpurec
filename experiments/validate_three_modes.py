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
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data" / "test_trees_100"

# AleRax reports in ln-space; we compute in log2.
# NLL_log2 = NLL_ln / ln(2)
_LN2 = math.log(2)
ALERAX_NLL_LN = {
    "global":      9737.07,
    "genewise":    9573.40,
    "specieswise": 8957.65,
}
ALERAX_NLL = {k: v / _LN2 for k, v in ALERAX_NLL_LN.items()}

# ──────────────────────────────────────────────────────────────────────────────
# AleRax reference parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_alerax_global_rates(data_dir):
    """Parse global AleRax rates from model_parameters (all rows identical)."""
    params_dir = data_dir / "output_global" / "model_parameters"
    for f in params_dir.iterdir():
        for line in f.read_text().splitlines():
            if line.startswith("node") or line.startswith("Node"):
                continue
            parts = line.split()
            if len(parts) == 4:
                # "node_id D L T"
                return {"D": float(parts[1]), "L": float(parts[2]), "T": float(parts[3])}
    return None


def _parse_alerax_genewise_rates(data_dir, G):
    """Parse per-family AleRax rates (alternating 'D L T' header + values)."""
    params_dir = data_dir / "output_genewise" / "model_parameters"
    rates = []
    for f in sorted(params_dir.iterdir()):
        for line in f.read_text().splitlines():
            if line.strip() == "D L T":
                continue
            parts = line.split()
            if len(parts) == 3:
                rates.append({"D": float(parts[0]), "L": float(parts[1]), "T": float(parts[2])})
    return rates[:G]


def _parse_alerax_per_fam_nll(data_dir, mode, G):
    """Parse per-family NLL from per_fam_likelihoods.txt (ln-space).

    Returns dict mapping family index → NLL in log2-space.
    """
    fpath = data_dir / f"output_{mode}" / "per_fam_likelihoods.txt"
    if not fpath.exists():
        return None
    result = {}
    for line in fpath.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            # "family_XXXX -123.456"  (ln-space, negative log-likelihood with minus sign)
            idx = int(parts[0].split("_")[1])
            if idx < G:
                result[idx] = -float(parts[1]) / _LN2  # convert to positive NLL in log2
    return result


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


def _write_wave_csv(path, history):
    """Write per-step CSV for global or specieswise optimizer history."""
    cumulative = 0.0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "nll", "grad_inf", "e_iters",
                         "cg_method", "cg_iters", "cg_residual", "cg_fallback",
                         "step_time_s", "cumulative_time_s"])
        for i, rec in enumerate(history, start=1):
            cumulative += rec.step_time_s
            sg = rec.solve_stats_G
            writer.writerow([
                i,
                f"{rec.negative_log_likelihood:.6f}",
                f"{rec.grad_infinity_norm:.6e}",
                rec.fp_info.iterations_E,
                sg.method,
                sg.iters,
                f"{sg.rel_residual:.2e}",
                int(sg.fallback_used),
                f"{rec.step_time_s:.4f}",
                f"{cumulative:.4f}",
            ])


def _write_genewise_csv(path, history):
    """Write per-step CSV for genewise optimizer history."""
    cumulative = 0.0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "nll_sum", "grad_inf", "n_active", "e_iters", "step_time_s", "cumulative_time_s"])
        for rec in history:
            cumulative += rec.get("step_time_s", 0.0)
            writer.writerow([
                rec["step"],
                f"{float(rec['nll'].sum()):.6f}",
                f"{rec['grad_inf']:.6e}",
                rec["n_active"],
                rec.get("e_iters", ""),
                f"{rec.get('step_time_s', 0.0):.4f}",
                f"{cumulative:.4f}",
            ])


def _rate_comparison(label, ours, alerax):
    """Format one line of rate comparison: our value, AleRax value, relative error."""
    rel = abs(ours - alerax) / max(abs(alerax), 1e-15)
    return f"  {label}  our={ours:.6f}  AleRax={alerax:.6f}  rel_err={rel:.2%}"


def _report(mode, result, alerax_nll, elapsed, csv_path=None, alerax_rates=None):
    nll = result["negative_log_likelihood"]
    gap = nll - alerax_nll
    rates = result["rates"]
    history = result["history"]
    n_steps = len(history)
    e_iters = [r.fp_info.iterations_E for r in history]
    times = [r.step_time_s for r in history]
    print(f"\n{'═'*60}")
    print(f"  REPORT: {mode}")
    print(f"{'═'*60}")
    print(f"  Steps: {n_steps}   Time: {elapsed:.1f}s")
    print(f"  NLL  our={nll:.3f}   AleRax={alerax_nll:.3f}   gap={gap:+.3f}")
    if rates.dim() == 1:
        print(f"  Rates  D={rates[0]:.6f}  L={rates[1]:.6f}  T={rates[2]:.6f}")
        if alerax_rates:
            print(_rate_comparison("D", float(rates[0]), alerax_rates["D"]))
            print(_rate_comparison("L", float(rates[1]), alerax_rates["L"]))
            print(_rate_comparison("T", float(rates[2]), alerax_rates["T"]))
    elif rates.dim() == 2:  # [S,3] specieswise
        mean_r = rates.mean(0)
        print(f"  Rates (mean over species)  D={mean_r[0]:.6f}  L={mean_r[1]:.6f}  T={mean_r[2]:.6f}")
    print(f"  E calls: {n_steps} (1 per step)   Pi calls: {n_steps} (1 batched per step)")
    print(f"  E iters/step  min={min(e_iters)}  max={max(e_iters)}  mean={sum(e_iters)/len(e_iters):.1f}")
    # CG/GMRES solver stats
    cg_iters = [r.solve_stats_G.iters for r in history]
    cg_methods = [r.solve_stats_G.method for r in history]
    fallbacks = sum(1 for r in history if r.solve_stats_G.fallback_used)
    print(f"  E adjoint     method={cg_methods[0]}  iters min={min(cg_iters)} max={max(cg_iters)} mean={sum(cg_iters)/len(cg_iters):.0f}"
          f"  fallbacks={fallbacks}/{n_steps}")
    print(f"  Step time     min={min(times):.2f}s  max={max(times):.2f}s  mean={sum(times)/len(times):.2f}s")
    if csv_path:
        _write_wave_csv(csv_path, history)
        print(f"  CSV written -> {csv_path}")
    print(f"{'═'*60}")


def _report_genewise(result, alerax_nll, elapsed, csv_path=None,
                     alerax_rates=None, alerax_per_fam_nll=None):
    total_nll = float(result["nll"].sum())
    gap = total_nll - alerax_nll
    history = result["history"]
    n_steps = len(history)
    G = len(result["rates"])
    rates = result["rates"]  # [G, 3]
    mean_r = rates.mean(0)
    e_iters = [r.get("e_iters", 0) for r in history if r.get("e_iters") is not None]
    times = [r.get("step_time_s", 0.0) for r in history]
    n_active = [r["n_active"] for r in history]
    print(f"\n{'═'*60}")
    print(f"  REPORT: genewise")
    print(f"{'═'*60}")
    print(f"  Steps: {n_steps}   Time: {elapsed:.1f}s  G={G} families")
    print(f"  NLL  our={total_nll:.3f}   AleRax={alerax_nll:.3f}   gap={gap:+.3f}")
    print(f"  Rates (mean over families)  D={mean_r[0]:.6f}  L={mean_r[1]:.6f}  T={mean_r[2]:.6f}")
    if alerax_rates:
        ar_D = sum(r["D"] for r in alerax_rates) / len(alerax_rates)
        ar_L = sum(r["L"] for r in alerax_rates) / len(alerax_rates)
        ar_T = sum(r["T"] for r in alerax_rates) / len(alerax_rates)
        print(f"  AleRax (mean over families)  D={ar_D:.6f}  L={ar_L:.6f}  T={ar_T:.6f}")
    # Per-family rate comparison
    if alerax_rates and len(alerax_rates) == G:
        d_errs, l_errs, t_errs = [], [], []
        for g in range(G):
            for idx, key, errs in [(0, "D", d_errs), (1, "L", l_errs), (2, "T", t_errs)]:
                ref = alerax_rates[g][key]
                our = float(rates[g, idx])
                errs.append(abs(our - ref) / max(abs(ref), 1e-15))
        print(f"  Per-family rate rel_err (mean/max):"
              f"  D={sum(d_errs)/G:.2%}/{max(d_errs):.2%}"
              f"  L={sum(l_errs)/G:.2%}/{max(l_errs):.2%}"
              f"  T={sum(t_errs)/G:.2%}/{max(t_errs):.2%}")
    # Per-family NLL comparison
    if alerax_per_fam_nll:
        our_nll = result["nll"]
        diffs = []
        for g in range(G):
            if g in alerax_per_fam_nll:
                diffs.append(float(our_nll[g]) - alerax_per_fam_nll[g])
        if diffs:
            print(f"  Per-family NLL gap (our-AleRax):"
                  f"  mean={sum(diffs)/len(diffs):+.3f}"
                  f"  min={min(diffs):+.3f}  max={max(diffs):+.3f}")
    print(f"  E calls: {n_steps} batched (1 per step, [G,S] tensor)")
    print(f"  Pi calls: {n_steps} x G_active (sequential per family per step)")
    if e_iters:
        print(f"  E iters/step  min={min(e_iters)}  max={max(e_iters)}  mean={sum(e_iters)/len(e_iters):.1f}")
    print(f"  Active genes  start={n_active[0]}  end={n_active[-1]}")
    print(f"  Step time     min={min(times):.2f}s  max={max(times):.2f}s  mean={sum(times)/len(times):.2f}s")
    if csv_path:
        _write_genewise_csv(csv_path, history)
        print(f"  CSV written -> {csv_path}")
    print(f"{'═'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Max optimizer steps")
    parser.add_argument("--families", type=int, default=0,
                        help="Use only first N families (0 = all 100)")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Directory to write CSV files (default: current dir)")
    parser.add_argument("--optimizer", type=str, default="lbfgs",
                        choices=["lbfgs", "adam", "sgd"],
                        help="Optimizer for global/specieswise modes (default: lbfgs)")
    parser.add_argument("--lr", type=float, default=0.2,
                        help="Learning rate for adam/sgd (default: 0.2, ignored for lbfgs)")
    parser.add_argument("--tol-theta", type=float, default=None,
                        help="Early stopping on theta change (default: 1e-3 for sgd, 0 for adam/lbfgs)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum for sgd (default: 0.9, ignored for adam/lbfgs)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64"],
                        help="Precision (default: float32)")
    args = parser.parse_args()

    # Create timestamped run subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.optimizer}_{args.dtype}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "command": " ".join(sys.argv),
        "timestamp": timestamp,
        **vars(args),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Run directory: {out_dir}")

    optimizer = args.optimizer
    lr = args.lr
    momentum = args.momentum
    if args.tol_theta is not None:
        tol_theta = args.tol_theta
    elif optimizer == 'sgd':
        tol_theta = 1e-3
    else:
        tol_theta = 0.0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    dtype  = torch.float64 if args.dtype == "float64" else torch.float32

    from gpurec.core.model import GeneDataset
    from gpurec.optimization.theta_optimizer import optimize_theta_wave, optimize_theta_genewise

    sp_path   = str(DATA_DIR / "sp.nwk")
    gene_paths = sorted(DATA_DIR.glob("g_*.nwk"))
    if args.families > 0:
        gene_paths = gene_paths[:args.families]
    gene_paths = [str(g) for g in gene_paths]
    G = len(gene_paths)
    print(f"Loaded {G} families from {DATA_DIR.name}")

    # Parse AleRax reference data
    alerax_global_rates = _parse_alerax_global_rates(DATA_DIR)
    alerax_genewise_rates = _parse_alerax_genewise_rates(DATA_DIR, G)
    alerax_gw_per_fam_nll = _parse_alerax_per_fam_nll(DATA_DIR, "genewise", G)

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
        lr=lr,
        tol_theta=tol_theta,
        optimizer=optimizer,
        momentum=momentum,
        specieswise=False,
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
        verbose=True,
    )
    _report("global", result_global, ALERAX_NLL["global"], time.time() - t0,
            csv_path=out_dir / "global_history.csv",
            alerax_rates=alerax_global_rates)

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
        lr=lr,
        tol_theta=tol_theta,
        optimizer=optimizer,
        momentum=momentum,
        specieswise=True,
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
        verbose=True,
    )
    _report("specieswise", result_sw, ALERAX_NLL["specieswise"], time.time() - t0,
            csv_path=out_dir / "specieswise_history.csv")

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
        verbose=True,
    )
    _report_genewise(result_gw, ALERAX_NLL["genewise"], time.time() - t0,
                     csv_path=out_dir / "genewise_history.csv",
                     alerax_rates=alerax_genewise_rates,
                     alerax_per_fam_nll=alerax_gw_per_fam_nll)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("Summary (NLL gap = our - AleRax, negative = we beat AleRax):")
    print(f"  global:      {result_global['negative_log_likelihood'] - ALERAX_NLL['global']:+.3f}")
    print(f"  specieswise: {result_sw['negative_log_likelihood']     - ALERAX_NLL['specieswise']:+.3f}")
    print(f"  genewise:    {float(result_gw['nll'].sum())            - ALERAX_NLL['genewise']:+.3f}")
    if alerax_global_rates:
        print(f"\nAleRax reference rates (global):")
        print(f"  D={alerax_global_rates['D']:.6f}  L={alerax_global_rates['L']:.6f}  T={alerax_global_rates['T']:.6f}")
    print(f"\nAleRax reference NLL (log2-space):")
    for mode, nll in ALERAX_NLL.items():
        print(f"  {mode:12s}  {nll:.3f}  (ln-space: {ALERAX_NLL_LN[mode]:.2f})")


if __name__ == "__main__":
    main()
