"""Timed genewise fit driver for the Coleman 1007-leaf / 5124-family benchmark on the cluster.

Runs exactly the production recipe (`gpurec.fit.dtl_fit.fit_dtl`, mode="genewise", certify on) with
verbose progress, prefixes every printed line with elapsed seconds, and writes:
  <out_dir>/<tag>.json   headline numbers (NLL bits/nats, wall time, steps, builds, certificate)
  <out_dir>/<tag>.pt     fitted theta [F,3] (log2 rates) + the family paths, for likelihood comparison

Usage:
  python benchmark/cc/run_genewise.py --species S.nwk --families LIST.txt --limit 0 --out-dir DIR --tag NAME
`--limit 0` means all families in the list; `--limit N` keeps the first N (for smoke/profiling runs).
"""
from __future__ import annotations

import argparse
import builtins
import json
import sys
import time

import torch

REPO_LEVEL_PRINT = builtins.print
_T0 = time.perf_counter()


def _timed_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    REPO_LEVEL_PRINT(f"[{time.perf_counter() - _T0:9.2f}s]", *args, **kwargs)


GIB = 2 ** 30


class MemoryLedger:
    """Per-phase CUDA memory accounting for the boundaries ``fit_genewise`` already brackets.

    Installed with ``gpurec.fit.genewise_fit.set_memory_probe``; the recipe hands us the name of each
    boundary as it is crossed. On every call we read the three allocator counters, fold them into the
    row for that phase name, and then RESET the peak counter -- so the ``peak`` read at boundary *k*
    is the high-water mark of the interval since boundary *k-1*, i.e. of the phase that just ran,
    rather than a running maximum over the whole fit. The run's overall peak is kept separately in
    ``self.run_peak`` (the driver reports that instead of ``max_memory_allocated``, which the resets
    would otherwise truncate).
    """

    def __init__(self, snapshot_path, snapshot_above_gib):
        self.rows = {}
        self.order = []
        self.run_peak = 0
        self.snapshot_path = snapshot_path
        self.snapshot_above_gib = snapshot_above_gib
        self.snapshot_written = False

    def __call__(self, label):
        alloc = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak_reserved = torch.cuda.max_memory_reserved()
        self.run_peak = max(self.run_peak, peak)
        row = self.rows.get(label)
        if row is None:
            row = {"n": 0, "alloc_max": 0, "alloc_last": 0, "peak_max": 0,
                   "reserved_max": 0, "peak_reserved_max": 0}
            self.rows[label] = row
            self.order.append(label)
        row["n"] += 1
        row["alloc_last"] = alloc
        row["alloc_max"] = max(row["alloc_max"], alloc)
        row["peak_max"] = max(row["peak_max"], peak)
        row["reserved_max"] = max(row["reserved_max"], reserved)
        row["peak_reserved_max"] = max(row["peak_reserved_max"], peak_reserved)
        if (self.snapshot_path is not None and not self.snapshot_written
                and peak >= self.snapshot_above_gib * GIB):
            torch.cuda.memory._dump_snapshot(self.snapshot_path)
            self.snapshot_written = True
            print(f"[mem] snapshot written to {self.snapshot_path} at phase {label!r} "
                  f"(peak {peak / GIB:.2f} GiB)")
        torch.cuda.reset_peak_memory_stats()

    def table(self):
        head = (f"{'phase':<20}{'n':>6}{'alloc GiB':>12}{'peak GiB':>11}"
                f"{'reserved GiB':>14}{'peak resv GiB':>15}")
        lines = ["[mem] per-phase CUDA memory (max over every crossing of that boundary)", head,
                 "-" * len(head)]
        for label in self.order:
            r = self.rows[label]
            lines.append(f"{label:<20}{r['n']:>6}{r['alloc_max'] / GIB:>12.2f}"
                         f"{r['peak_max'] / GIB:>11.2f}{r['reserved_max'] / GIB:>14.2f}"
                         f"{r['peak_reserved_max'] / GIB:>15.2f}")
        lines.append(f"[mem] run peak allocated = {self.run_peak / GIB:.2f} GiB")
        return chr(10).join(lines)

    def as_json(self):
        return {label: dict(self.rows[label]) for label in self.order}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True, help="listfile: one .ale path per line")
    ap.add_argument("--limit", required=True, type=int, help="0 = all families; N = first N of the list")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init-rate", required=True,
                    help="start every family at this rate for D, L and T; 'none' = fit_dtl's default start")
    ap.add_argument("--forward-self-loop", required=True, choices=("linear", "log", "exact"),
                    help="forward self-loop kernel path (SolverOptions.forward_self_loop)")
    ap.add_argument("--adjoint-self-loop", required=True, choices=("series", "exact"),
                    help="wave-backward adjoint self-loop path (SolverOptions.adjoint_self_loop)")
    # Optional because it is a debugging aid, not a setting: leaving it out is exactly the
    # behaviour of a build without the facility. When given, a batch whose curvature probe or
    # gradient goes non-finite is reported and written there before the run dies, so the failure
    # can be replayed on that batch alone instead of re-running the whole fit.
    # Per-batch clade budget: 0 = automatic (derive from the card; the tuned 315,000 when it
    # fits, smaller when it does not). Required so a run's record always states which it used.
    ap.add_argument("--clade-budget", required=True, type=int,
                    help="clades per batch; 0 = derive from the device memory budget")
    ap.add_argument("--debug-dump-dir", required=False, default=None)
    # Memory instrumentation. Also a debugging aid rather than a setting: without --memory-table the
    # run is byte-for-byte the uninstrumented one (no probe is installed at all).
    ap.add_argument("--memory-table", action="store_true",
                    help="print per-phase allocated/peak/reserved CUDA memory at the recipe's own "
                         "phase boundaries")
    ap.add_argument("--memory-snapshot", required=False, default=None,
                    help="path for a torch.cuda.memory._dump_snapshot taken the first time a phase "
                         "peak exceeds --memory-snapshot-above-gib (needs --memory-table)")
    ap.add_argument("--memory-snapshot-above-gib", required=False, type=float, default=0.0)
    args = ap.parse_args()

    from gpurec.api import _failure_dump
    _failure_dump.use_directory(args.debug_dump_dir)

    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]
    builtins.print = _timed_print
    print(f"[driver] {len(paths)} families, species={args.species}, torch {torch.__version__}, "
          f"gpu={torch.cuda.get_device_name(0)}")

    from gpurec.config import GpurecConfig
    from gpurec.fit.dtl_fit import fit_dtl

    # The recipe's own reference config (identical to passing no config) with the kernel path set.
    cfg = GpurecConfig.genewise_reference()
    cfg.solver.forward_self_loop = args.forward_self_loop
    cfg.solver.adjoint_self_loop = args.adjoint_self_loop

    ledger = None
    if args.memory_table:
        from gpurec.fit import genewise_fit as _gf
        if args.memory_snapshot is not None:
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        ledger = MemoryLedger(args.memory_snapshot, args.memory_snapshot_above_gib)
        _gf.set_memory_probe(ledger)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    init_rate = None if args.init_rate == "none" else float(args.init_rate)
    res = fit_dtl(args.species, paths, "genewise", device="cuda", verbose=True, config=cfg,
                  init_rate=init_rate,
                  clade_budget=(None if args.clade_budget == 0 else args.clade_budget))
    wall = time.perf_counter() - t0
    # The ledger resets the peak counter at every phase boundary, so with it installed
    # ``max_memory_allocated`` only covers the last phase; the ledger's own running maximum is the
    # run peak. Without it, ``max_memory_allocated`` is the run peak as before.
    peak_b = torch.cuda.max_memory_allocated()
    if ledger is not None:
        ledger("driver_end")
        peak_b = ledger.run_peak
        print(ledger.table())
    peak_gib = peak_b / 2**30
    g = res["genewise_result"]
    summary = {
        "tag": args.tag, "forward_self_loop": args.forward_self_loop,
        "adjoint_self_loop": args.adjoint_self_loop,
        "init_rate": args.init_rate, "clade_budget": args.clade_budget,
        "n_families": res["n_families"], "wall_s": wall,
        "nll_bits": res["nll_bits"], "nll_nats": res["nll_nats"],
        "opt_seconds": g["opt_seconds"], "n_steps": g["n_steps"], "n_builds": g["n_builds"],
        # per-phase wall-clock split (see fit_genewise's result dict)
        "adam_seconds": g["adam_seconds"], "hessian_seconds": g["hessian_seconds"],
        "n_hessians": g["n_hessians"], "newton_grad_seconds": g["newton_grad_seconds"],
        "n_verify_builds": g["n_verify_builds"], "verify_seconds": g["verify_seconds"],
        "n_rebuilds": g["n_rebuilds"], "rebuild_seconds": g["rebuild_seconds"],
        "certify_seconds": g["certify_seconds"],
        "history": g["history"], "peak_gib": peak_gib,
        "memory_phases": (None if ledger is None else ledger.as_json()),
        "certificate": {k: g[k] for k in ("converged", "interior_pd", "bound_active", "unconverged",
                                           "premature_drops", "pg_max") if k in g},
    }
    print(f"[driver] DONE wall={wall:.1f}s nll_bits={res['nll_bits']:.3f} nll_nats={res['nll_nats']:.3f} "
          f"steps={g['n_steps']} builds={g['n_builds']} peak={peak_gib:.2f}GiB "
          f"adam={g['adam_seconds']:.0f}s hess={g['n_hessians']}x/{g['hessian_seconds']:.0f}s "
          f"grad={g['newton_grad_seconds']:.0f}s "
          f"verify={g['n_verify_builds']}x/{g['verify_seconds']:.0f}s "
          f"rebuild={g['n_rebuilds']}x/{g['rebuild_seconds']:.0f}s "
          f"cert={g['certify_seconds']:.0f}s {summary['certificate']}")
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/{args.tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    torch.save({"theta": res["theta"], "paths": paths}, f"{args.out_dir}/{args.tag}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
