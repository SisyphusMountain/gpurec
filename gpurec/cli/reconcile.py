"""`gpurec reconcile` — marginal log-likelihood at fixed DTL rates."""
from __future__ import annotations

import math
import sys

from . import _common


def add_args(parser) -> None:
    _common.add_common_args(parser)
    parser.add_argument("--delta", type=float, default=1e-10, help="duplication rate D")
    parser.add_argument("--tau", type=float, default=1e-10, help="transfer rate T")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-10, help="loss rate L")
    parser.add_argument("--out", default=None, help="write per-family logL (AleRax per_fam format)")


def _set_rates(model, D, T, L):
    import torch
    # gpurec theta order is [log2 D, log2 L, log2 T] (theta[2] = transfer)
    triple = torch.tensor([math.log2(D), math.log2(L), math.log2(T)],
                          dtype=model.theta.dtype, device=model.theta.device)
    with torch.no_grad():
        if model.theta.dim() == 1:            # global (3,)
            model.theta.copy_(triple)
        else:                                  # (S,3) or (F,3): broadcast the global triple
            model.theta.copy_(triple.expand_as(model.theta))


def run_reconcile(args) -> int:
    import torch
    from gpurec.bench.alerax_io import norm_family_name

    model, genes = _common.build_model(args)
    _set_rates(model, args.delta, args.tau, args.lambda_)

    rows = []
    with torch.no_grad():
        if args.mode == "genewise":
            loss_vec = model.genewise_loss_vector()          # [F] NLL bits
            for path, loss_bits in zip(genes, loss_vec.tolist()):
                rows.append((norm_family_name(path), -_common.bits_to_nats(loss_bits)))
        else:
            loss_bits = float(model())                        # summed NLL bits
            fam = norm_family_name(genes[0]) if len(genes) == 1 else "all"
            rows.append((fam, -_common.bits_to_nats(loss_bits)))

    for fam, ll in rows:
        if not math.isfinite(ll):
            print(f"error: non-finite log-likelihood for {fam}", file=sys.stderr)
            return 1

    text = "".join(f"{fam} {ll:.6f}\n" for fam, ll in rows)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    sys.stdout.write(text)
    return 0
