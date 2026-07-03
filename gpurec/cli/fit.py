"""`gpurec fit` — optimize DTL rates in global / specieswise / genewise mode.

global / specieswise -> gpurec.optim.optimize + final_eval (writes AleRax-format rates).
genewise            -> gpurec.optim.genewise_fit.fit_genewise (writes a JSON sidecar).
"""
from __future__ import annotations

import json
import math
import time

from . import _common


def add_args(parser) -> None:
    _common.add_common_args(parser)
    parser.add_argument("--steps", type=int, default=300, help="optimizer steps (global/specieswise)")
    parser.add_argument("--init-rate", type=float, default=None, help="initial rate for D, T, L")
    parser.add_argument("--out", default=None,
                        help="write fitted rates (AleRax '# node D L T') + <out>.json sidecar")


def _write_outputs(out, node_rates, nll_bits, nll_nats, elapsed_s, mode, n_families):
    """node_rates: list[(node, D, L, T)] (gpurec order == AleRax cols); file uses 'node D L T'."""
    with open(out, "w") as fh:
        fh.write("# node D L T\n")
        for node, D, L, T in node_rates:
            fh.write(f"{node} {D:.10g} {L:.10g} {T:.10g}\n")
    sidecar = {"mode": mode, "nll_bits": nll_bits, "nll_nats": nll_nats,
               "elapsed_s": elapsed_s, "n_families": n_families}
    with open(f"{out}.json", "w") as fh:
        json.dump(sidecar, fh, indent=2)


def _write_genewise(out, res, nll_bits, nll_nats, elapsed_s):
    theta, rates = res.get("theta"), res.get("rates")
    payload = {"mode": "genewise", "nll_bits": nll_bits, "nll_nats": nll_nats,
               "elapsed_s": elapsed_s, "n_families": int(res.get("n_families", 0)),
               "theta_log2": theta.tolist() if hasattr(theta, "tolist") else theta,
               "rates": rates.tolist() if hasattr(rates, "tolist") else rates}
    with open(f"{out}.json", "w") as fh:
        json.dump(payload, fh, indent=2)


def _set_init_rate(model, rate):
    import torch
    lr = math.log2(rate)
    with torch.no_grad():
        model.theta.fill_(lr)


def _global_node_rates(theta_hat, mode, S):
    """Turn a fitted theta into list[(label, D, L, T)] rows (gpurec order == AleRax cols)."""
    t = theta_hat.detach().cpu()
    if t.dim() == 1:                         # global (3,)
        D, L, T = (2.0 ** t).tolist()
        return [("GLOBAL", D, L, T)]
    rows = []                                # specieswise (S,3)
    for i in range(t.shape[0]):
        D, L, T = (2.0 ** t[i]).tolist()
        rows.append((f"s{i}", D, L, T))
    return rows


def run_fit(args) -> int:
    t0 = time.perf_counter()

    if args.mode == "genewise":
        from gpurec.optim.genewise_fit import fit_genewise
        res = fit_genewise(args.species, args.gene, device=args.device,
                           dtype=_common.make_dtype(args.dtype), certify=True)
        nll_nats = float(res.get("loss_nats", float("nan")))
        nll_bits = nll_nats / math.log(2.0)
        n_fam = int(res.get("n_families", 0))
        elapsed = time.perf_counter() - t0
        print(f"fit(genewise): final NLL = {nll_nats:.6f} nats ({n_fam} families, {elapsed:.1f}s)")
        if args.out:
            _write_genewise(args.out, res, nll_bits, nll_nats, elapsed)
        return 0

    # global / specieswise
    from gpurec.optim.optimize import optimize, final_eval
    model, genes = _common.build_model(args)
    if args.init_rate is not None:
        _set_init_rate(model, args.init_rate)
    theta_hat, _hist = optimize(model.batch_statics, model.theta.detach(),
                                model.receiver_weights.detach(),
                                optimizer="adam", schedule="adaptive", max_steps=args.steps)
    loss_bits, gnorm = final_eval(model.batch_statics, theta_hat, model.receiver_weights.detach())
    loss_bits = float(loss_bits)
    nll_nats = _common.bits_to_nats(loss_bits)
    elapsed = time.perf_counter() - t0
    print(f"fit({args.mode}): final NLL = {nll_nats:.6f} nats "
          f"(grad_norm {float(gnorm):.2e}, {elapsed:.1f}s)")
    if args.out:
        S = model.theta.shape[0] if model.theta.dim() > 1 else 1
        _write_outputs(args.out, _global_node_rates(theta_hat, args.mode, S),
                       loss_bits, nll_nats, elapsed, args.mode, len(genes))
    return 0
