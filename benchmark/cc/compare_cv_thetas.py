"""Score two specieswise cross-validation fits under ONE common solver.

``run_cv_archaea.py`` writes a fitted per-species theta for each checkout. Those two runs each
scored themselves with their own library, so a difference in their reported NLL mixes two things:
where the fit LANDED, and how the number was computed. This script removes the second one -- it
loads both thetas and evaluates them both with whatever library it is itself imported from, on the
same families, so the remaining gap is purely about the fitted point.

It also reports the per-species rate difference (in the actual rates, not the log2 parameters), so
"the two fits agree" can be stated as a real number per rate rather than as one summary.

Usage:
  python benchmark/cc/compare_cv_thetas.py \
      --species .../reference_species_tree.newick \
      --ale-dir .../main_families_ge4seq --n-families 1000 \
      --left  $CC_RUNS/results/cv_archaea_old.pt \
      --right $CC_RUNS/results/cv_archaea_new.pt
"""
from __future__ import annotations

import argparse
import glob
import os

import torch

_RATE_NAMES = ("duplication", "transfer", "loss")


def _family_paths(ale_dir, n_families):
    paths = sorted(glob.glob(os.path.join(ale_dir, "*.ale")))
    return paths[: int(n_families)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--ale-dir", required=True)
    parser.add_argument("--n-families", required=True, type=int)
    parser.add_argument("--left", required=True, help="a run_cv_archaea.py .pt")
    parser.add_argument("--right", required=True, help="the other run_cv_archaea.py .pt")
    args = parser.parse_args()

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.map_cv import _CV_SO, heldout_nll

    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    print(f"[left ] {args.left}\n         {left['identity']}", flush=True)
    print(f"[right] {args.right}\n         {right['identity']}", flush=True)

    for label, payload in (("left", left), ("right", right)):
        curve = "  ".join(f"lam={lam:g}:{value:.4f}" for lam, value in sorted(payload["cv"].items()))
        print(
            f"[{label}] n={payload['n_families']} k={payload['k']} lam*={payload['lam_star']}  "
            f"CV curve: {curve}  cv_wall={payload['cv_wall_s']:.1f}s",
            flush=True,
        )

    options = SolverOptions(**_CV_SO)
    options.validate()
    paths = _family_paths(args.ale_dir, args.n_families)
    model = GeneReconModel(str(args.species), [str(p) for p in paths], mode="specieswise",
                           device="cuda", solver_options=options)
    receiver = model.receiver_weights.detach().clone()

    scores = {}
    for label, payload in (("left", left), ("right", right)):
        theta = payload["theta_final"].to("cuda", torch.float32)
        scores[label] = heldout_nll(model.batch_statics, theta, receiver)
        print(f"[rescore] {label} theta -> {scores[label]:.6f} bits over {len(paths)} families",
              flush=True)
    print(
        f"[rescore] right - left = {scores['right'] - scores['left']:+.6f} bits "
        f"({(scores['right'] - scores['left']) / max(1, len(paths)):+.6e} bits per family)",
        flush=True,
    )

    left_theta = left["theta_final"].double()
    right_theta = right["theta_final"].double()
    theta_gap = (right_theta - left_theta).abs()
    left_rates = 2.0 ** left_theta
    right_rates = 2.0 ** right_theta
    rate_gap = (right_rates - left_rates).abs()
    print(
        f"[theta] max |right - left| over all {theta_gap.numel()} entries = "
        f"{float(theta_gap.max()):.4e} log2 units; mean = {float(theta_gap.mean()):.4e}",
        flush=True,
    )
    for index, name in enumerate(_RATE_NAMES):
        print(
            f"[rates] {name:<12} left in [{float(left_rates[:, index].min()):.5f}, "
            f"{float(left_rates[:, index].max()):.5f}]  "
            f"right in [{float(right_rates[:, index].min()):.5f}, "
            f"{float(right_rates[:, index].max()):.5f}]  "
            f"max|difference| = {float(rate_gap[:, index].max()):.4e}  "
            f"mean|difference| = {float(rate_gap[:, index].mean()):.4e}",
            flush=True,
        )

    for label, payload in (("left", left), ("right", right)):
        joint = payload.get("joint")
        if joint is None:
            continue
        print(
            f"[joint {label}] data NLL {joint['data_nll_bits']:.4f} bits  "
            f"penalized loss {joint['penalized_loss']:.4f}  |g| {joint['grad_norm']:.3e}  "
            f"alpha max|.| {joint['alpha_absmax']:.4f}  alpha sd {joint['alpha_std']:.4f}  "
            f"{joint['iterations']} L-BFGS iterations in {joint['wall_s']:.1f}s",
            flush=True,
        )
    if left.get("joint") is not None and right.get("joint") is not None:
        alpha_gap = (right["joint"]["alpha"].double() - left["joint"]["alpha"].double()).abs()
        print(
            f"[joint] max |right alpha - left alpha| = {float(alpha_gap.max()):.4e}  "
            f"mean = {float(alpha_gap.mean()):.4e}  "
            f"data NLL right - left = "
            f"{right['joint']['data_nll_bits'] - left['joint']['data_nll_bits']:+.6f} bits",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
