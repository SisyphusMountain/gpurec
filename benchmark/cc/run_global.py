"""Timed global-mode (one shared D/L/T rate for all families) fit driver.

Deliberately calls `gpurec.fit.dtl_fit.fit_dtl(..., "global")` with NOTHING but the species tree,
the family list and the device, so whichever checkout is on PYTHONPATH runs its OWN default recipe.
That is the point of this driver: the old and the new code must be compared at their own defaults,
and passing a config here would force one checkout's settings onto the other.

Writes <out_dir>/<tag>.json with the fitted rates (D, L, T -- linear, relative to speciation), the
total NLL in bits and nats, the final projected-gradient size and the wall time, and
<out_dir>/<tag>.pt with the fitted 3-vector theta (log2 rates) and the family paths.

Usage:
  python benchmark/cc/run_global.py --species S.nwk --families LIST.txt --limit 0 \
      --out-dir DIR --tag NAME
`--limit 0` means all families in the list; `--limit N` keeps the first N.
"""
from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
import time

import torch

REPO_LEVEL_PRINT = builtins.print
_T0 = time.perf_counter()


def _timed_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    REPO_LEVEL_PRINT(f"[{time.perf_counter() - _T0:9.2f}s]", *args, **kwargs)


def _install_global_fit_shim() -> str:
    """Work around a bug that stops `fit_global` from running AT ALL, in both checkouts.

    `gpurec/fit/global_fit.py` builds its per-tier solver settings with
    `SolverOptions(pi_iters=..., neumann_terms=..., e_adjoint_solver="neumann")`, but
    `SolverOptions` has no `e_adjoint_solver` field, so every call raises
    `TypeError: SolverOptions.__init__() got an unexpected keyword argument 'e_adjoint_solver'`
    before any work happens. The line is byte-identical in the original checkout (817007e6) and at
    branch HEAD (a8598cee): the E-adjoint became Neumann-only and this caller was never updated, so
    the keyword names the only behaviour there is and dropping it changes nothing numerically.

    The shim replaces the module-level `_tier_solver_options` with the same function minus the dead
    keyword. It is installed identically for both checkouts, so the old-vs-new comparison is still
    old-code-versus-new-code and not old-workaround-versus-new-workaround. Returns a note recorded
    in the result JSON so no reader mistakes these numbers for stock behaviour.
    """
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit import global_fit

    if "e_adjoint_solver" in getattr(SolverOptions, "__dataclass_fields__", {}):
        return "none (SolverOptions accepts e_adjoint_solver)"

    def tier_solver_options(*, pi_iters: int, neumann_terms: int) -> SolverOptions:
        return SolverOptions(pi_iters=pi_iters, neumann_terms=neumann_terms)

    global_fit._tier_solver_options = tier_solver_options
    return "dropped fit_global's dead e_adjoint_solver kwarg (SolverOptions has no such field)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--families", required=True, help="listfile: one gene-tree path per line")
    parser.add_argument("--limit", required=True, type=int, help="0 = all families; N = first N")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    paths = [line.strip() for line in open(args.families)
             if line.strip() and not line.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]
    builtins.print = _timed_print
    print(f"[driver] global mode, {len(paths)} families, species={args.species}, "
          f"torch {torch.__version__}, gpu={torch.cuda.get_device_name(0)}")

    from gpurec.fit.dtl_fit import fit_dtl

    shim = _install_global_fit_shim()
    print(f"[driver] fit_global shim: {shim}")

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    res = fit_dtl(args.species, paths, "global", device="cuda", verbose=True)
    wall = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    rates = [float(x) for x in res["rates"].tolist()]
    summary = {
        "tag": args.tag, "mode": "global", "n_families": res["n_families"], "wall_s": wall,
        "nll_bits": res["nll_bits"], "nll_nats": res["nll_nats"],
        "theta_log2": [float(x) for x in res["theta"].tolist()],
        "rate_D": rates[0], "rate_L": rates[1], "rate_T": rates[2],
        "gnorm": res["gnorm"], "n_steps": res["n_steps"], "peak_gib": peak_gib,
        "fit_global_shim": shim,
    }
    print(f"[driver] DONE wall={wall:.1f}s nll_bits={res['nll_bits']:.3f} "
          f"nll_nats={res['nll_nats']:.3f} D={rates[0]:.6g} L={rates[1]:.6g} T={rates[2]:.6g} "
          f"steps={res['n_steps']} |Pg|={res['gnorm']:.3e} peak={peak_gib:.2f}GiB")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/{args.tag}.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    torch.save({"theta": res["theta"], "paths": paths}, f"{args.out_dir}/{args.tag}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
