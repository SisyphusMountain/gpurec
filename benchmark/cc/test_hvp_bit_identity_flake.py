"""Flake rate of the bit-identity HVP comparison, per forward_self_loop.

Reproduces exactly what tests/test_fraction_missing_hvp.py::
test_hvp_fraction_missing_zero_matches_none asserts -- two models that must compute the
same thing, whose HVPs are required to match bit for bit -- and repeats it, so the pass rate can be
compared across forward paths instead of inferred from one sample.
"""
import argparse
import pathlib
import sys
import tempfile

import torch

SPECIES = "((A:1,B:1)AB:1,C:1)Root;"
GENE_DUP = "(((A_1:1,B_1:1)x:1,C_1:1)y:1,A_2:1)GeneRoot;"
TANGENT_SELF_ITERS = 256


def build(tmp, fraction_missing, tag, mode, dtype):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    species = tmp / f"s_{tag}.nwk"
    species.write_text(SPECIES)
    gene = tmp / f"g_{tag}.nwk"
    gene.write_text(GENE_DUP)
    return GeneReconModel(str(species), [str(gene)], mode="specieswise", device="cuda",
                          dtype=dtype, fraction_missing=fraction_missing,
                          solver_options=SolverOptions(forward_self_loop=mode))


def hvp_of(model, theta, direction):
    from gpurec.solver.value_and_grad import forward_solve
    from gpurec.solver.hvp.exact import make_exact_hvp
    static = model.batch_statics[0]
    weights = model.receiver_weights.detach()
    _loss, saved = forward_solve(static, theta, weights)
    apply_hvp = make_exact_hvp(static, theta, weights, saved,
                               tangent_self_iters=TANGENT_SELF_ITERS)
    return apply_hvp(direction).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--modes", default="log,linear,exact")
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    for mode in args.modes.split(","):
        failures, worst_ulps = 0, 0.0
        with tempfile.TemporaryDirectory() as directory:
            tmp = pathlib.Path(directory)
            none_model = build(tmp, None, "none", mode, dtype)
            zero_model = build(tmp, 0.0, "zero", mode, dtype)
            assert none_model.leaf_fm_log is None and zero_model.leaf_fm_log is None
            theta0 = none_model.theta.detach().clone()
            torch.manual_seed(1)
            direction = torch.randn(theta0.numel(), dtype=dtype, device="cuda")
            direction = direction / direction.norm()
            for _ in range(args.repeats):
                a = hvp_of(none_model, theta0, direction)
                b = hvp_of(zero_model, zero_model.theta.detach().clone(), direction)
                if not torch.equal(a, b):
                    failures += 1
                    difference = (a - b).abs()
                    scale = torch.maximum(a.abs(), b.abs()).clamp(min=1e-300)
                    ulps = float((difference / scale).max()) / float(torch.finfo(dtype).eps)
                    worst_ulps = max(worst_ulps, ulps)
        print(f"[{mode}] bit-identity FAILS in {failures}/{args.repeats} repeats; "
              f"worst disagreement {worst_ulps:.1f} ulp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
