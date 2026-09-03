"""Specieswise MAP + cross-validation on archaea, in a form that runs UNCHANGED on two checkouts.

This is the A/B driver for "did the new exact tree solves change the specieswise fit?". The whole
point is that the SAME file is copied into both checkouts and run with each one's own ``PYTHONPATH``,
so every difference in the answer comes from the library, not from the driver.

What it runs:

  1. ``gpurec.fit.map_cv.map_cv`` -- k-fold-over-families cross-validation with a lambda homotopy.
     lambda is the precision of a CENTERED RIDGE penalty: it pulls every species' three log2 rates
     toward one common rate ``log2(init_rate)``. Large lambda first, each fit warm-started from the
     previous one. The score is the held-out negative log-likelihood summed over the families of the
     fold that was left out; lam* is the lambda with the smallest mean held-out score. map_cv then
     refits on all families at lam*.

     NOTE on which penalty: ``map_cv`` uses the centered ridge above. The other driver in the repo,
     ``experiments/sanderson_cv/run_cv.py``, uses the Sanderson tree-roughness penalty (each species
     pulled toward its PARENT). They are different objectives and their lambdas are not comparable,
     so this script uses one of them -- map_cv -- on both arms and never mixes the two.

  2. Optionally (``--joint 1``) a JOINT refit at lam*, where the per-species transfer receiver logits
     ``alpha`` are optimized alongside theta instead of being held at zero (uniform). That is the
     "joint receiver-weight" variant. It is written here rather than reused from
     ``experiments/sanderson_cv/run_cv_joint.py`` because that file is hard-wired to the hogenom
     dataset and imports two modules that still name the removed ``bicgstab_max_iter`` /
     ``bicgstab_tol`` / ``bicgstab_breakdown_tol`` solver options and therefore raise TypeError on
     import: ``run_cv`` (fixed in this branch) and ``converge_bounded_joint_archaea`` (not fixed).

Everything the comparison needs is written to ``--out`` as a .pt: the CV curve, lam*, the final
per-species rates, the final NLL, and wall times.

Usage (identical on both checkouts, only PYTHONPATH differs):
  python benchmark/cc/run_cv_archaea.py \
      --species .../archaea/species_reference/reference_species_tree.newick \
      --ale-dir .../archaea/ale_gene_tree_distributions/main_families_ge4seq \
      --n-families 1000 --k 3 --lambdas 100 10 1 --seed 0 \
      --adam-steps 10 --max-newton 8 --init-rate 0.1 --joint 1 --joint-iters 60 \
      --out $CC_RUNS/results/cv_archaea_new.pt
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time

import torch


def _family_paths(ale_dir, n_families):
    paths = sorted(glob.glob(os.path.join(ale_dir, "*.ale")))
    if not paths:
        raise FileNotFoundError(f"no .ale files under {ale_dir}")
    return paths[: int(n_families)]


def _library_identity():
    """What the run can say about the checkout it is actually importing.

    ``forward_self_loop`` / ``adjoint_self_loop`` exist only on the newer checkout; their absence is
    how the old one identifies itself, so the two output files are self-labelling.
    """
    import gpurec
    from gpurec.api.solver_options import SolverOptions

    options = SolverOptions()
    return {
        "gpurec_file": gpurec.__file__,
        "solver_option_fields": sorted(vars(options).keys()),
        "forward_self_loop": getattr(options, "forward_self_loop", None),
        "adjoint_self_loop": getattr(options, "adjoint_self_loop", None),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _joint_refit(species, paths, lam, theta_start, init_rate, iterations, seed):
    """Refit theta AND the per-species receiver logits alpha at a fixed lambda.

    The optimization variable is the single vector ``z = [theta flattened; alpha]``.
    ``make_value_and_grad(..., optimize_receiver=True)`` is what makes alpha a variable: it splits
    ``z``, feeds alpha into the solve as the receiver weights on every call, and returns the gradient
    of both blocks. The penalty is the same centered ridge map_cv used, applied to the theta block
    only -- alpha is unpenalized, exactly as in the repo's joint driver.

    alpha starts at small random values rather than exactly zero: at all-zero the receiver
    distribution is uniform, which is a stationary point of the softmax the optimizer cannot leave.
    """
    from scipy.optimize import minimize

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.map_cv import _CV_SO, heldout_nll
    from gpurec.solver.value_and_grad import make_value_and_grad

    options = SolverOptions(**_CV_SO)
    options.validate()
    model = GeneReconModel(str(species), [str(p) for p in paths], mode="specieswise",
                           device="cuda", solver_options=options)
    species_count = int(model.species_helpers["S"])
    theta_shape = (species_count, 3)
    theta_ref = torch.full(theta_shape, math.log2(init_rate), device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    alpha0 = (0.05 * torch.randn(species_count, generator=generator)).to("cuda", torch.float32)

    value_and_grad = make_value_and_grad(
        model.batch_statics, alpha0, theta_shape=theta_shape,
        optimize_receiver=True, prior=(float(lam), theta_ref),
    )
    theta_count = 3 * species_count
    calls = [0]

    def objective(z_numpy):
        calls[0] += 1
        z = torch.tensor(z_numpy, device="cuda", dtype=torch.float32)
        loss, gradient, _saved, _warm = value_and_grad(z)
        return float(loss), gradient.double().cpu().numpy()

    start = time.perf_counter()
    z0 = torch.cat([theta_start.reshape(-1).float().cuda(), alpha0]).double().cpu().numpy()
    result = minimize(objective, z0, jac=True, method="L-BFGS-B",
                      options={"maxiter": int(iterations), "maxfun": 2 * int(iterations),
                               "maxcor": 50, "ftol": 1e-12, "gtol": 1e-8})
    wall = time.perf_counter() - start
    z = torch.tensor(result.x, device="cuda", dtype=torch.float32)
    theta_hat = z[:theta_count].reshape(theta_shape)
    alpha_hat = z[theta_count:]
    alpha_hat = alpha_hat - alpha_hat.mean()  # the softmax ignores a constant shift; center to compare
    data_nll = heldout_nll(model.batch_statics, theta_hat, alpha_hat)
    out = {
        "theta": theta_hat.detach().cpu(),
        "alpha": alpha_hat.detach().cpu(),
        "penalized_loss": float(result.fun),
        "grad_norm": float(torch.tensor(result.jac).norm()),
        "data_nll_bits": data_nll,
        "iterations": int(result.nit),
        "solver_calls": calls[0],
        "wall_s": wall,
        "alpha_absmax": float(alpha_hat.abs().max()),
        "alpha_std": float(alpha_hat.std()),
    }
    del model
    torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--ale-dir", required=True)
    parser.add_argument("--n-families", required=True, type=int)
    parser.add_argument("--k", required=True, type=int, help="number of cross-validation folds")
    parser.add_argument("--lambdas", required=True, type=float, nargs="+",
                        help="ridge precisions to sweep, any order (map_cv sorts them descending)")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--adam-steps", required=True, type=int)
    parser.add_argument("--max-newton", required=True, type=int)
    parser.add_argument("--init-rate", required=True, type=float,
                        help="the common rate the ridge pulls every species toward")
    parser.add_argument("--joint", required=True, type=int, choices=(0, 1),
                        help="1 to also refit with the receiver logits as free variables")
    parser.add_argument("--joint-iters", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from gpurec.fit.map_cv import map_cv

    identity = _library_identity()
    print(f"[identity] {identity}", flush=True)

    paths = _family_paths(args.ale_dir, args.n_families)
    print(f"[setup] {len(paths)} families, k={args.k}, lambdas={args.lambdas}, seed={args.seed}",
          flush=True)

    start = time.perf_counter()
    result = map_cv(
        args.species, paths, k=args.k, lambdas=tuple(args.lambdas), mode="specieswise",
        init_rate=args.init_rate, seed=args.seed, solver_options=None, device="cuda",
        adam_steps=args.adam_steps, max_newton=args.max_newton, verbose=True,
    )
    cv_wall = time.perf_counter() - start
    print(f"[map_cv] {cv_wall:.1f}s  lam* = {result['lam_star']}  S = {result['S']}", flush=True)

    payload = {
        "identity": identity,
        "n_families": len(paths),
        "k": args.k,
        "lambdas": list(args.lambdas),
        "seed": args.seed,
        "init_rate": args.init_rate,
        "cv": result["cv"],
        "lam_star": result["lam_star"],
        "theta_final": result["theta_final"].cpu() if result["theta_final"] is not None else None,
        "S": result["S"],
        "cv_wall_s": cv_wall,
    }

    if args.joint and result["theta_final"] is not None:
        print("[joint] refitting theta together with the per-species receiver logits", flush=True)
        payload["joint"] = _joint_refit(
            args.species, paths, result["lam_star"], result["theta_final"], args.init_rate,
            args.joint_iters, args.seed,
        )
        joint = payload["joint"]
        print(
            f"[joint] {joint['wall_s']:.1f}s  penalized loss {joint['penalized_loss']:.4f}  "
            f"data NLL {joint['data_nll_bits']:.4f} bits  |g| {joint['grad_norm']:.3e}  "
            f"alpha max|.| {joint['alpha_absmax']:.4f}  alpha sd {joint['alpha_std']:.4f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)
    print(f"[out] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
