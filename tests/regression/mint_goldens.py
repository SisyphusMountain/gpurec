"""Mint committed goldens: fit each mode, measure run-to-run spread, set tolerances = 3x spread.
Run ONCE on the target GPU (RTX 4090). Re-run to re-mint if rustree/gpurec change."""
from __future__ import annotations

import argparse, json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gpurec.fit.dtl_fit import fit_dtl
from gpurec.bench.simulate import (
    simulate_dataset, SIM_PARAMS, SPECIESWISE_GOLDEN_LAM, SPECIESWISE_GOLDEN_INIT_RATE,
)

GOLDENS = Path(__file__).parent / "goldens"


def _git_rev(cwd):
    if not cwd:
        return "unknown"
    try:
        return subprocess.check_output(["git", "-C", str(cwd), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def fit_mode(mode, species_path, gene_paths, *, verbose=False):
    """Fit one mode via its production recipe; returns (nll_bits, rates[.,3] D,L,T, wall_s).

    global/genewise are plug-and-play through fit_dtl. specieswise has no one-shot MLE, so it is fit
    by fit_specieswise at the committed SPECIESWISE_GOLDEN_LAM prior, with init AND prior mean
    theta_ref = constant log2(SPECIESWISE_GOLDEN_INIT_RATE) (see the design spec)."""
    if mode == "specieswise":
        # NB: no local `import torch` here -- torch is module-level (line 10). A function-local
        # import would make `torch` local to ALL of fit_mode and UnboundLocalError the
        # global/genewise branch's `torch.float32` below.
        import math
        from gpurec.api.model import GeneReconModel
        from gpurec.api.solver_options import SolverOptions
        from gpurec.fit.specieswise_fit import fit_specieswise
        model = GeneReconModel(species_path, gene_paths, mode="specieswise", device="cuda",
                               dtype=torch.float32, solver_options=SolverOptions())
        # init AND ridge center at a sensible rate, NOT the model's degenerate 1e-10 default (which
        # centers the prior on ~zero rates -> fit snaps to the prior in ~2 steps).
        S = model.theta.shape[0]
        theta0 = torch.full((S, 3), math.log2(SPECIESWISE_GOLDEN_INIT_RATE), device="cuda",
                            dtype=torch.float32)
        res = fit_specieswise(model.batch_statics, theta0, model.receiver_weights.detach(),
                              lam=SPECIESWISE_GOLDEN_LAM, theta_ref=theta0, verbose=verbose)
        rates = np.asarray(res["rates"])
        return float(res["nll_bits"]), rates, float(res["wall_s"])
    res = fit_dtl(species_path, gene_paths, mode, device="cuda", dtype=torch.float32, verbose=verbose)
    rates = np.asarray(res["rates"])   # cpu tensor -> ndarray
    return float(res["nll_bits"]), rates, float(res["wall_s"])


def mint(mode, repeats=5, verbose=False):
    p = SIM_PARAMS[mode]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sp, genes = simulate_dataset(mode, Path(d) / mode, n_species=p["n_species"],
                                     n_families=p["n_families"], dtl=p["dtl"], seed=p["seed"])
        nlls, rate_runs, walls = [], [], []
        for r in range(repeats):
            nll, rates, wall = fit_mode(mode, sp, genes, verbose=verbose)
            nlls.append(nll); rate_runs.append(rates); walls.append(wall)
            if verbose:
                print(f"[{mode}] repeat {r + 1}/{repeats}: nll={nll:.6f} wall={wall:.1f}s", flush=True)

    nlls = np.array(nlls); R = np.stack(rate_runs)          # [repeats, ., 3]
    nll_spread = float(nlls.std()) if repeats > 1 else 0.0
    rate_spread_rel = float(np.abs(R - R.mean(0)).max() / (np.abs(R).mean() + 1e-30)) if repeats > 1 else 0.0
    nll_rtol = max(3 * nll_spread / (abs(nlls.mean()) + 1e-30), 1e-3)
    rates_rtol = max(3 * rate_spread_rel, 1e-2)

    golden = {
        "mode": mode,
        "provenance": {
            "rustree_commit": _git_rev(os.environ.get("RUSTREE_ROOT")),
            "gpurec_commit": _git_rev(Path(__file__).resolve().parents[2]),
            "seed": p["seed"], "n_species": p["n_species"], "n_families": p["n_families"],
            "dtl": p["dtl"], "device": torch.cuda.get_device_name(0), "dtype": "float32",
            "repeats": repeats, "nll_spread": nll_spread, "rate_spread_rel": rate_spread_rel,
            "minted_utc": datetime.now(timezone.utc).isoformat(),
            "specieswise_golden_lam": SPECIESWISE_GOLDEN_LAM,
            "specieswise_golden_init_rate": SPECIESWISE_GOLDEN_INIT_RATE,
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
    ap.add_argument("--verbose", action="store_true", help="stream per-step optimize progress")
    a = ap.parse_args()
    for m in ([a.mode] if a.mode else list(SIM_PARAMS)):
        mint(m, repeats=a.repeats, verbose=a.verbose)
