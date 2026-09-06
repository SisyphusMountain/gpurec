"""Save (or compare) the 200-family theta gradient so a speed change can be shown to be safe.

Writes a float64 copy of the gradient (and of the receiver-weight gradient, when the caller asks
for one) to a .pt file. Run once before a change and once after, then compare the two files with
--compare: the check is max|after - before| divided by max|before|, which must sit at the float32
rounding level (< 1e-5).

Usage:
  save_gradient_snapshot.py --species S --families LIST --limit 200 --clade-budget 100000 \
      --dtype float32 --out before.pt [--receiver-grad]
  save_gradient_snapshot.py --compare before.pt after.pt
"""
from __future__ import annotations

import argparse
import sys

import torch


def _compare(before_path: str, after_path: str) -> int:
    before = torch.load(before_path)
    after = torch.load(after_path)
    status = 0
    for key in sorted(set(before) | set(after)):
        if key not in before or key not in after:
            print(f"[grad] {key}: present in only one snapshot")
            status = 1
            continue
        b, a = before[key], after[key]
        if b.shape != a.shape:
            print(f"[grad] {key}: shape {tuple(b.shape)} -> {tuple(a.shape)}")
            status = 1
            continue
        scale = float(b.abs().max())
        diff = float((a - b).abs().max())
        rel = diff / scale if scale > 0.0 else diff
        ok = "OK" if rel < 1e-5 else "FAIL"
        print(
            f"[grad] {key}: max|before| {scale:.6e}  max|after-before| {diff:.6e}  "
            f"relative {rel:.3e}  {ok}"
        )
        if rel >= 1e-5:
            status = 1
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--species", default=None)
    ap.add_argument("--families", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clade-budget", type=int, default=None)
    ap.add_argument("--dtype", default=None, choices=("float32", "float64"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--receiver-grad", action="store_true")
    args = ap.parse_args()
    if args.compare is not None:
        return _compare(args.compare[0], args.compare[1])
    for name in ("species", "families", "limit", "clade_budget", "dtype", "out"):
        if getattr(args, name) is None:
            ap.error(f"--{name.replace('_', '-')} is required when not comparing")

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    dtype = getattr(torch, args.dtype)
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")][: args.limit]
    so = SolverOptions(**_BASE_SOLVER)
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=dtype,
                       solver_options=so, clade_budget=args.clade_budget)
    m.receiver_weights.requires_grad_(bool(args.receiver_grad))
    theta = torch.tensor([-6.0, -3.0, -6.0], dtype=dtype, device="cuda").repeat(len(paths), 1).contiguous()
    torch.cuda.reset_peak_memory_stats()
    loss_vec, grad_theta, grad_receiver = m.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    peak = torch.cuda.max_memory_allocated()
    snapshot = {
        "loss": loss_vec.double().cpu(),
        "grad_theta": grad_theta.double().cpu(),
    }
    if args.receiver_grad:
        snapshot["grad_receiver"] = grad_receiver.double().cpu()
    torch.save(snapshot, args.out)
    print(
        f"[grad] wrote {args.out}: NLL {float(loss_vec.double().sum()):.6f} "
        f"max|grad_theta| {float(grad_theta.abs().max()):.6e} "
        f"peak CUDA memory {peak / 2**20:.0f} MiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
