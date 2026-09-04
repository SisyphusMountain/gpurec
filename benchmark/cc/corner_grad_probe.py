"""Gradient probe at one rate corner: who is wrong, the exact forward, the exact adjoint, or float32?

Builds, for one (D, L, T) log2-rate corner shared by all families:
  oracle   fp64  log@2048 sweeps + series@512 terms
  X64      fp64  exact/exact
  L32      fp32  log@2048 + series@512
  X32      fp32  exact/exact
  M32      fp32  exact forward + series@512 adjoint
and reports |grad| magnitudes and per-rate-component errors vs the oracle, plus the worst family.

Usage: corner_grad_probe.py --species S --families LIST --limit N --clade-budget B --corner D,L,T --exact-range-log2 R
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_exact_range_corners import _build, _forward_rows, _loss_and_grad  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int)
    ap.add_argument("--clade-budget", required=True, type=int)
    ap.add_argument("--corner", required=True, help="D,L,T log2 rates, e.g. =-19.9,-6,-19.9")
    ap.add_argument("--exact-range-log2", required=True, type=float)
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]
    d, l, t = (float(x) for x in args.corner.split(","))
    print(f"[probe] corner D={d} L={l} T={t}  families={len(paths)}", flush=True)

    specs = {
        "oracle": (torch.float64, "log", "series", 2048, 512),
        "log16k": (torch.float64, "log", "series", 16384, 2048),
        "X64": (torch.float64, "exact", "exact", 2048, 512),
        "L32": (torch.float32, "log", "series", 2048, 512),
        "X32": (torch.float32, "exact", "exact", 2048, 512),
        "M32": (torch.float32, "exact", "series", 2048, 512),
    }
    results = {}
    for name, (dtype, fwd, adj, pi, neu) in specs.items():
        m = _build(args.species, paths, args.clade_budget, pi, neu, fwd, adj, dtype, args.exact_range_log2)
        theta = torch.tensor([d, l, t], dtype=dtype, device="cuda").repeat(len(paths), 1).contiguous()
        rows, flagged = _forward_rows(m, theta)
        loss, grad = _loss_and_grad(m, theta)
        results[name] = (loss, grad, flagged)
        finite_rows = torch.cat([r[torch.isfinite(r)] for r in rows])
        print(f"[probe] {name:6s} max log2 Pi over all rows={float(finite_rows.max()):.4f} (>0 means Pi>1)  "
              f"n lanes with log2 Pi>0: {int((finite_rows > 0).sum())}", flush=True)
        print(f"[probe] {name:6s} flagged={flagged:7d} sum NLL={float(loss.sum()):.6f} bits  "
              f"max|grad|={float(grad.abs().max()):.4e}  per-rate max|grad| D/L/T="
              f"{[f'{float(v):.3e}' for v in grad.abs().amax(dim=0)]}", flush=True)
        del m, rows
        torch.cuda.empty_cache()

    ol, og, _ = results["oracle"]
    for name in ("log16k", "X64", "L32", "X32", "M32"):
        loss, grad, flagged = results[name]
        dg = (grad - og).abs()
        worst = int(dg.amax(dim=1).argmax())
        print(f"[probe] {name:6s} vs oracle: max|dNLL|={float((loss-ol).abs().max()):.3e} bits  "
              f"max|dgrad|={float(dg.max()):.3e}  per-rate D/L/T={[f'{float(v):.3e}' for v in dg.amax(dim=0)]}  "
              f"worst family #{worst}: grad={[f'{float(v):.4e}' for v in grad[worst]]} oracle={[f'{float(v):.4e}' for v in og[worst]]}  "
              f"rel(worst)={float(dg[worst].max() / (og[worst].abs().max() + 1e-300)):.3e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
