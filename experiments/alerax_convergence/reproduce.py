#!/usr/bin/env python
"""Reproducible experiment: gpurec vs AleRax_fixed at fixed DTL rates, and AleRax's
*rate-dependent* fixed-point under-convergence.

Question
--------
Do gpurec and AleRax agree on the marginal reconciliation log-likelihood at
genuinely non-trivial rates (all of D, L, T distinct and large), and is AleRax's
shipped default a converged reference?

What it does
------------
For the ``test_mixed_200`` fixture (100 species, one family) it evaluates BOTH engines
at the *same fixed* rates (no rate optimisation on either side):

  * AleRax_fixed via ``--fix-rates --d D --l L --t T --fixed-point-iterations N``
  * gpurec       via ``theta = [log2 D, log2 L, log2 T]`` (its native rate order)

and reports:

  1. High rates D/L/T = 1.5/1.6/1.7 -- AleRax swept over N in {4, 16, 64} vs gpurec
     (converged + a self-convergence check). Shows AleRax's default N=4 under-converges
     by ~87 nats and lands on gpurec's value once N>=16.
  2. The fixture's own AleRax-fitted moderate rates (D~0.16, T~0.16) -- AleRax at N=4
     vs N=16 (already converged at 4) plus the stock-vs-fixed convention identity
     ``stock = AleRax_fixed + ln(2S-1)``.

It ends with PASS/FAIL assertions and writes ``results.json`` next to this file.

Run
---
    cd /home/enzo/Documents/git/gpurec/consolidate-release
    .venv/bin/python experiments/alerax_convergence/reproduce.py           # full (~4-5 min)
    .venv/bin/python experiments/alerax_convergence/reproduce.py --quick   # skip slow N=64

Requirements
------------
  * A CUDA GPU + the base env (torch + triton) -- gpurec's forward path.
  * An AleRax_fixed binary (has ``--fix-rates`` and ``--fixed-point-iterations``).
    Auto-located under agent-worktrees/, or set ``ALERAX_BIN=/path/to/alerax``.
    Build: ``agent-worktrees/*/AleRax_fixed/AleRax`` -> ``cmake -DCMAKE_BUILD_TYPE=Release && make``.
  * The fixture (ships the stock reference in ``output/``); override with ``FIXTURE=...``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LN2 = math.log(2.0)

DEFAULT_FIXTURE = Path(os.environ.get(
    "FIXTURE",
    "/home/enzo/Documents/git/gpurec/gergely_version/tests/data/test_mixed_200"))

HIGH_RATES = (1.5, 1.6, 1.7)          # D, L, T -- all distinct, all large
FIT_RATES = (0.16103, 1e-10, 0.156391)  # test_mixed_200's AleRax-fitted D, L, T


def find_alerax() -> str:
    """Locate an AleRax_fixed binary (env override, else agent-worktrees glob)."""
    env = os.environ.get("ALERAX_BIN")
    if env and Path(env).is_file():
        return env
    cands = sorted(glob.glob(
        "/home/enzo/Documents/git/gpurec/agent-worktrees/*/AleRax_fixed/AleRax/build/bin/alerax"))
    for c in cands:
        # sanity: must expose the fixed-point flag (i.e. be AleRax_fixed, not stock)
        h = subprocess.run([c, "--help"], capture_output=True, text=True)
        if "fixed-point-iterations" in (h.stdout + h.stderr):
            return c
    sys.exit("AleRax_fixed binary not found. Set ALERAX_BIN=/path/to/alerax. "
             "Build from agent-worktrees/*/AleRax_fixed/AleRax "
             "(cmake -DCMAKE_BUILD_TYPE=Release && make).")


def run_alerax(binpath: str, fixture: Path, rates, iters: int):
    """Evaluate AleRax_fixed at FIXED rates. Returns (logL_nats, n_species)."""
    d, l, t = rates
    work = tempfile.mkdtemp(prefix="alerax_conv_")
    try:
        for fn in ("families.txt", "sp.nwk", "g.nwk"):
            shutil.copy(fixture / fn, work)
        cmd = [binpath, "-f", "families.txt", "-s", "sp.nwk", "-p", "out",
               "--gene-tree-samples", "0", "--species-tree-search", "SKIP",
               "--fix-rates", "--d", f"{d:.12g}", "--l", f"{l:.12g}", "--t", f"{t:.12g}",
               "--fixed-point-iterations", str(iters)]
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        pf = Path(work) / "out" / "per_fam_likelihoods.txt"
        if not pf.is_file():
            raise RuntimeError(f"AleRax failed (iters={iters}):\n{proc.stdout}\n{proc.stderr}")
        logl = float(pf.read_text().split()[1])
        n_species = None
        for line in proc.stdout.splitlines():
            if "Number of species:" in line:
                n_species = int(line.split(":")[1].strip())
        return logl, n_species
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_gpurec(fixture: Path, rates, e_max_iter=2000, pi_iters=64, neumann_terms=64) -> float:
    """Evaluate gpurec at fixed rates -> logL_nats. theta order is [log2 D, log2 L, log2 T]."""
    import torch
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    d, l, t = rates
    theta = [math.log2(d), math.log2(l), math.log2(t)]
    model = GeneReconModel(
        str(fixture / "sp.nwk"), [str(fixture / "g.nwk")],
        mode="global", device="cuda",
        solver_options=SolverOptions(e_max_iter=e_max_iter, pi_iters=pi_iters,
                                     neumann_terms=neumann_terms))
    with torch.no_grad():
        model.theta.copy_(torch.tensor(theta, dtype=model.theta.dtype,
                                       device=model.theta.device))
        loss_bits = float(model())
    return -loss_bits * LN2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow N=64 high-rate AleRax run (N=16 already shows convergence)")
    ap.add_argument("--out", default=str(Path(__file__).with_name("results.json")))
    args = ap.parse_args()

    binp = find_alerax()
    fx = DEFAULT_FIXTURE
    if not (fx / "sp.nwk").is_file():
        sys.exit(f"fixture not found at {fx}; set FIXTURE=/path/to/test_mixed_200")

    high_iters = [4, 16] if args.quick else [4, 16, 64]
    print(f"AleRax_fixed : {binp}")
    print(f"fixture      : {fx}")
    print(f"high rates   : D/L/T = {HIGH_RATES}")
    print(f"fitted rates : D/L/T = {FIT_RATES}\n")

    # ---- (1) High rates: AleRax iteration sweep + gpurec ----
    print("[1/2] high rates -- AleRax iteration sweep + gpurec ...")
    ax_high, n_species = {}, None
    for it in high_iters:
        v, s = run_alerax(binp, fx, HIGH_RATES, it)
        ax_high[it] = v
        n_species = s or n_species
        print(f"      AleRax_fixed N={it:<3d} -> {v:.5f}")
    gp_high = run_gpurec(fx, HIGH_RATES)
    gp_high_heavy = run_gpurec(fx, HIGH_RATES, e_max_iter=6000, pi_iters=256, neumann_terms=256)
    gp_high_light = run_gpurec(fx, HIGH_RATES, e_max_iter=50, pi_iters=8, neumann_terms=8)
    print(f"      gpurec  converged      -> {gp_high:.5f}")
    print(f"      gpurec  heavy(256)     -> {gp_high_heavy:.5f}")
    print(f"      gpurec  light(8)       -> {gp_high_light:.5f}\n")

    # ---- (2) Fitted moderate rates: convergence + stock/fixed convention ----
    print("[2/2] fitted moderate rates -- AleRax N=4 vs N=16 + convention ...")
    ax_fit = {it: run_alerax(binp, fx, FIT_RATES, it)[0] for it in (4, 16)}
    gp_fit = run_gpurec(fx, FIT_RATES)
    stock = float((fx / "output" / "per_fam_likelihoods.txt").read_text().split()[1])
    ln_term = math.log(2 * n_species - 1)
    for it in (4, 16):
        print(f"      AleRax_fixed N={it:<3d} -> {ax_fit[it]:.5f}")
    print(f"      gpurec  converged      -> {gp_fit:.5f}")
    print(f"      stock (shipped, 4-iter)-> {stock:.5f}")
    print(f"      ln(2S-1) = ln({2*n_species-1}) = {ln_term:.5f}\n")

    results = {
        "fixture": str(fx), "alerax_bin": binp, "n_species": n_species,
        "high_rates_DLT": list(HIGH_RATES), "fit_rates_DLT": list(FIT_RATES),
        "high": {"alerax": ax_high, "gpurec_converged": gp_high,
                 "gpurec_heavy256": gp_high_heavy, "gpurec_light8": gp_high_light},
        "fit": {"alerax": ax_fit, "gpurec_converged": gp_fit,
                "stock_shipped": stock, "ln_2S_minus_1": ln_term},
    }
    Path(args.out).write_text(json.dumps(results, indent=2))

    # ---- self-checking assertions ----
    checks = [
        ("gpurec == AleRax_fixed(converged) at high rates (<=0.02 nats)",
         abs(gp_high - ax_high[16]) <= 0.02),
        ("gpurec self-converged at high rates (default == heavy256, <=1e-3)",
         abs(gp_high - gp_high_heavy) <= 1e-3),
        ("AleRax default N=4 under-converges at high rates (converged is >50 nats higher)",
         (gp_high - ax_high[4]) > 50.0),
        ("fitted moderate rates already converged at N=4 (N4 == N16, <=0.05)",
         abs(ax_fit[4] - ax_fit[16]) <= 0.05),
        ("gpurec == AleRax_fixed at fitted rates (<=0.05 nats)",
         abs(gp_fit - ax_fit[16]) <= 0.05),
        ("convention: stock == AleRax_fixed + ln(2S-1) (<=0.05 nats)",
         abs((stock - ax_fit[16]) - ln_term) <= 0.05),
    ]
    print("results ->", args.out, "\n")
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print()
    if ok:
        print("ALL CHECKS PASSED -- gpurec reproduces AleRax_fixed(converged) at low AND high "
              "rates; AleRax's default under-converges only at high rates.")
        return 0
    print("SOME CHECKS FAILED -- inspect the printed values and results.json.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
