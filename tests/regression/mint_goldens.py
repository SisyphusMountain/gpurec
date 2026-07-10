"""Mint committed goldens: fit each mode, measure run-to-run spread, set tolerances = 3x spread.
Run ONCE on the target GPU (RTX 4090). Re-run to re-mint if rustree/gpurec change."""
from __future__ import annotations

import argparse, json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.fit.optimize import optimize, final_eval
from gpurec.bench.simulate import simulate_dataset, SIM_PARAMS

GOLDENS = Path(__file__).parent / "goldens"


def _git_rev(cwd):
    try:
        return subprocess.check_output(["git", "-C", str(cwd), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def fit_mode(mode, species_path, gene_paths):
    # e_adjoint_solver="neumann": at 500 leaves (S=999) the E-side adjoint solve's fp32 GMRES
    # (Arnoldi) orthogonalization residual floors around ~1e-6 and fails to converge mid-fit
    # (theta-dependent). The Neumann series wE = sum_k J^k q has no orthogonalization and reaches
    # ~1e-6 in <=10 terms -- productionized in Task A, FD-validated to match GMRES gradient. config
    # (optimize's) does NOT thread solver options, so it must be set at model-build time here.
    model = GeneReconModel(species_path, gene_paths, mode=mode, device="cuda", dtype=torch.float32,
                           solver_options=SolverOptions(e_adjoint_solver="neumann"))
    t0 = time.perf_counter()
    # group_index=model.rate_family_idx is only meaningful for the reduced-group rate
    # parameterization (theta as [G,3] group rows expanded to per-species/per-item rows via
    # gpurec.solver.penalties.group_expand); model.rate_family_idx is a per-batch [n_items]
    # family-membership index (all-zeros for global/specieswise, since theta there is not
    # grouped that way), which is shape-incompatible with group_expand's [G,K]->[S,K]
    # contract for every current mode. Passing it made optimize() silently mis-expand theta
    # (e.g. for "global" it collapsed the 3-vector [D,L,T] down to a repeat of theta[0]=D).
    # Omit it -- group_index stays at its documented default (None == identity, no grouping).
    theta_hat, _hist = optimize(
        model.batch_statics, model.theta.detach(), model.receiver_weights.detach(),
        verbose=False)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    nll_bits, _gn = final_eval(model.batch_statics, theta_hat, model.receiver_weights.detach())
    rates = (2.0 ** theta_hat.detach().float().cpu()).numpy()   # [.,3] order D,L,T
    return float(nll_bits), rates, float(wall_s)


def mint(mode, repeats=5):
    p = SIM_PARAMS[mode]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sp, genes = simulate_dataset(mode, Path(d) / mode, n_species=p["n_species"],
                                     n_families=p["n_families"], dtl=p["dtl"], seed=p["seed"])
        nlls, rate_runs, walls = [], [], []
        for _ in range(repeats):
            nll, rates, wall = fit_mode(mode, sp, genes)
            nlls.append(nll); rate_runs.append(rates); walls.append(wall)

    nlls = np.array(nlls); R = np.stack(rate_runs)          # [repeats, ., 3]
    nll_spread = float(nlls.std()) if repeats > 1 else 0.0
    rate_spread_rel = float(np.abs(R - R.mean(0)).max() / (np.abs(R).mean() + 1e-30)) if repeats > 1 else 0.0
    nll_rtol = max(3 * nll_spread / (abs(nlls.mean()) + 1e-30), 1e-3)
    rates_rtol = max(3 * rate_spread_rel, 1e-2)

    golden = {
        "mode": mode,
        "provenance": {
            "rustree_commit": _git_rev("/home/enzo/Documents/git/rustree"),
            "gpurec_commit": _git_rev(Path(__file__).resolve().parents[2]),
            "seed": p["seed"], "n_species": p["n_species"], "n_families": p["n_families"],
            "dtl": p["dtl"], "device": torch.cuda.get_device_name(0), "dtype": "float32",
            "repeats": repeats, "nll_spread": nll_spread, "rate_spread_rel": rate_spread_rel,
            "minted_utc": datetime.now(timezone.utc).isoformat(),
        },
        "nll": float(nlls.mean()),
        "rates": R.mean(0).tolist(),
        "recorded_wall_s": float(np.median(walls)),
        "tolerances": {"nll_rtol": nll_rtol, "rates_rtol": rates_rtol, "rates_atol": 1e-4},
    }
    GOLDENS.mkdir(exist_ok=True)
    (GOLDENS / f"{mode}.json").write_text(json.dumps(golden, indent=2))
    print(f"[{mode}] nll={golden['nll']:.6f} rtol={nll_rtol:.2e} rates_rtol={rates_rtol:.2e} "
          f"wall~{golden['recorded_wall_s']:.1f}s")
    return golden


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(SIM_PARAMS), default=None)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    for m in ([a.mode] if a.mode else list(SIM_PARAMS)):
        mint(m, repeats=a.repeats)
