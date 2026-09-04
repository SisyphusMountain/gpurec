"""Is the exact HVP bitwise reproducible, and if not, where does the variation come from?

Answers three questions that a single 20-repeat count cannot:

  * ``--trials N`` repeats the whole measurement, because the count itself is noisy -- one trial
    of the same configuration can land anywhere from 0/19 to 19/19.
  * ``--reuse-sv 1`` runs every repeat from ONE forward result, so the forward path cannot
    contribute; anything left is downstream of it.
  * ``--poison 1`` hands the caching allocator blocks pre-filled with a value the computation
    cannot produce, so a buffer read before it is written shows up as that value rather than as
    plausible-looking noise.

It also reports the per-entry spread in ulps and checksums every tensor the forward publishes for
the HVP, so "the forward varies" and "the HVP varies" can be told apart.

Usage:
  python benchmark/cc/test_hvp_reproducibility.py --dtype float64 --trials 5 --reuse-sv 1
"""
import argparse
import hashlib
import pathlib
import sys
import tempfile

import torch

WARM_START = [1]
SPECIES = "((A:1,B:1)AB:1,C:1)Root;"
GENE_DUP = "(((A_1:1,B_1:1)x:1,C_1:1)y:1,A_2:1)GeneRoot;"
TANGENT_SELF_ITERS = 256


def build(tmp, mode, dtype):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    sp = tmp / "s.nwk"
    sp.write_text(SPECIES)
    gene = tmp / "g.nwk"
    gene.write_text(GENE_DUP)
    options = SolverOptions(forward_self_loop=mode, use_hvp_warm_start=bool(WARM_START[0]))
    return GeneReconModel(str(sp), [str(gene)], mode="specieswise", device="cuda",
                          dtype=dtype, fraction_missing=None, solver_options=options)


def digest(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return repr(tensor)
    return hashlib.sha1(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--modes", default="log,exact")
    parser.add_argument("--warm-start", type=int, default=1)
    parser.add_argument("--fresh-model", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--reuse-sv", type=int, default=0,
                        help="run every repeat from ONE forward result, so the forward cannot vary")
    parser.add_argument("--poison", type=int, default=0,
                        help="fill freed allocator blocks with a sentinel before each HVP")
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    WARM_START[0] = args.warm_start

    from gpurec.solver.value_and_grad import forward_solve
    from gpurec.solver.hvp.exact import make_exact_hvp

    for mode in args.modes.split(","):
      counts = []
      for _trial in range(args.trials):
        with tempfile.TemporaryDirectory() as directory:
            tmp = pathlib.Path(directory)
            model = build(tmp, mode, dtype)
            theta0 = model.theta.detach().clone()
            torch.manual_seed(1)
            direction = torch.randn(theta0.numel(), dtype=dtype, device="cuda")
            direction = direction / direction.norm()
            static = model.batch_statics[0]
            weights = model.receiver_weights.detach()
            results, snapshots, shared_saved = [], [], None
            for _ in range(args.repeats):
                if args.fresh_model:
                    model = build(tmp, mode, dtype)
                    static = model.batch_statics[0]
                    weights = model.receiver_weights.detach()
                if args.reuse_sv and results:
                    saved = shared_saved
                else:
                    _loss, saved = forward_solve(static, theta0, weights)
                    shared_saved = saved
                snapshot = {k: digest(v) for k, v in saved.items() if torch.is_tensor(v)}
                state = saved.get("pi_state")
                if state is not None:
                    snapshot["pi_offset"] = digest(state.pi_offset)
                    snapshot["pibar_offset"] = digest(state.pibar_offset)
                snapshots.append(snapshot)
                if args.poison:
                    # Hand the caching allocator blocks full of a value nothing can produce, so
                    # any buffer a kernel reads without writing shows up immediately.
                    junk = [torch.full((1 << 20,), -1.0e300 if dtype == torch.float64 else -3.0e38,
                                       device="cuda", dtype=dtype) for _ in range(64)]
                    del junk
                hvp = make_exact_hvp(static, theta0, weights, saved,
                                     tangent_self_iters=TANGENT_SELF_ITERS)
                results.append(hvp(direction).reshape(-1).detach().clone())
            differ = sum(1 for r in results[1:] if not torch.equal(r, results[0]))
            stack = torch.stack(results)
            spread = (stack.amax(dim=0) - stack.amin(dim=0))
            magnitude = stack[0].abs()
            relative = (spread / magnitude.clamp(min=1e-300)).max()
            varying_entries = (spread > 0).nonzero().reshape(-1).tolist()
            print(f"[{mode}] HVP differs from repeat 0 in {differ}/{args.repeats - 1} repeats; "
                  f"max abs spread {float(spread.max()):.3e}, max relative {float(relative):.3e}")
            print(f"[{mode}]   entries that vary: {varying_entries} of {stack.shape[1]}; "
                  f"|Hu| at those = {[f'{float(magnitude[i]):.3e}' for i in varying_entries]}")
            varying = sorted(
                key for key in snapshots[0]
                if any(s[key] != snapshots[0][key] for s in snapshots[1:])
            )
            print(f"[{mode}]   forward tensors that vary across repeats: {varying or 'none'}")
            counts.append(differ)
      print(f"[{mode}] SUMMARY differing-repeat counts over {args.trials} independent trials: "
            f"{counts} (out of {args.repeats - 1} each)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
