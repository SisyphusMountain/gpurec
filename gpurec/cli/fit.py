"""`gpurec fit` — optimize DTL rates in global / specieswise / genewise mode.

global   -> gpurec.fit.global_fit.fit_global (writes AleRax-format rates).
genewise -> gpurec.fit.genewise_fit.fit_genewise (writes a JSON sidecar).
specieswise -> not a one-shot CLI fit: `fit_dtl` raises `NotImplementedError` and this CLI
    surfaces that message cleanly (exit 1). Use gpurec.fit.specieswise_fit.fit_specieswise
    (single MAP prior fit) or gpurec.fit.map_cv.map_cv (cross-validates the prior) directly.
"""
from __future__ import annotations

import json
import math
import sys

from . import _common


def add_args(parser) -> None:
    _common.add_common_args(parser)
    parser.add_argument("--steps", type=int, default=300,
                        help="unused: global uses fit_global's own iteration budget; "
                             "specieswise is not a supported one-shot CLI fit")
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
    from gpurec.config import load_config
    from gpurec.fit.dtl_fit import fit_dtl

    # Mode->recipe dispatch lives in fit_dtl (the single canonical entry): genewise -> fit_genewise,
    # global -> fit_global, specieswise -> raises NotImplementedError with guidance to
    # fit_specieswise/map_cv (no well-posed one-shot fit; caught below and surfaced cleanly).
    # Pass explicit solver_options only when the user actually overrode one (--config or a
    # solver flag); otherwise leave it None so fit_dtl uses its
    # robust Neumann E-adjoint (converges to the fp32 floor with no orthogonalization residual floor).
    user_solver = (
        args.config is not None
        or args.pi_iters is not None
        or args.neumann_terms is not None
        or args.e_max_iter is not None
    )
    # Pass a config ONLY when the user actually supplied one, for the same reason as solver_options
    # above: the recipes treat a config's [solver] table as authoritative, so handing them a
    # default-constructed GpurecConfig would silently replace their tuned solver settings with the
    # library defaults.
    config = load_config(args.config) if args.config is not None else None
    try:
        res = fit_dtl(args.species, args.gene, args.mode, device=args.device,
                      dtype=(None if args.dtype is None else _common.make_dtype(args.dtype)),
                      config=config, max_steps=args.steps,
                      init_rate=args.init_rate,
                      solver_options=_common.make_solver_options(args) if user_solver else None)
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    nll_nats, nll_bits, elapsed = res["nll_nats"], res["nll_bits"], res["wall_s"]
    n_fam = res["n_families"]
    if not math.isfinite(nll_nats):
        print(f"error: non-finite NLL for fit({args.mode})", file=sys.stderr)
        return 1

    if args.mode == "genewise":
        print(f"fit(genewise): final NLL = {nll_nats:.6f} nats ({n_fam} families, {elapsed:.1f}s)")
        if args.out:
            _write_genewise(args.out, res["genewise_result"], nll_bits, nll_nats, elapsed)
        return 0

    gnorm = res.get("gnorm", float("nan"))
    print(f"fit({args.mode}): final NLL = {nll_nats:.6f} nats "
          f"(grad_norm {gnorm:.2e}, {elapsed:.1f}s)")
    if args.out:
        theta_hat = res["theta"]
        S = theta_hat.shape[0] if theta_hat.dim() > 1 else 1
        _write_outputs(args.out, _global_node_rates(theta_hat, args.mode, S),
                       nll_bits, nll_nats, elapsed, args.mode, n_fam)
    return 0
