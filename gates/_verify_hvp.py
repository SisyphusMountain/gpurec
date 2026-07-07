"""Phase-3 verification gate: analytic exact HVP == fp64 FD-of-gradient (+ symmetry).

Mirrors kernel-bench ``newton/verify.py::check_hvp_exact``. The analytic exact-Hessian HVP
(forward-over-reverse through the SO/tangent Triton kernels) must match a central finite
difference of the value-and-grad gradient. fp64 + a CONVERGED solver (high pi_iters / neumann /
tangent-self) so neither side is truncation-limited; tolerance ~5e-4 reflects residual solver
truncation (per the kbench gate's note).

    python -m gates._verify_hvp                 # tiny live-hogenom model (local, fp64)
    python -m gates._verify_hvp --capture       # the 666x80 capture (large fp64; use the A100)
"""

from __future__ import annotations

import glob
import math
import sys

import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.optim.hvp_exact import build_point_cache, make_exact_hvp
from gpurec.optim.newton_cg import _fd_hessian_hvp
from gpurec.optim.value_and_grad import forward_solve, make_value_and_grad

_ROOT = "/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_hogenom_core/hogenom"
_SP = (f"{_ROOT}/runs/MFP/true_start_ufboot1000/"
       "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/"
       "starting_species_tree.newick")
# converged solver: pi/neumann/tangent-self high enough that fwd & bwd are both converged
_SO = dict(e_max_iter=2000, e_tol=1e-10, pi_iters=128, neumann_terms=64,
           self_loop_solver="neumann", bicgstab_max_iter=500, bicgstab_tol=1e-10,
           bicgstab_breakdown_tol=1e-30, adjoint_pruning_threshold=1e-6,
           use_adjoint_pruning=True, pibar_side_threshold=0.0)


def _static_theta_rw_from_live(n_families, device, init_rate=0.1):
    so = SolverOptions(**_SO)
    so.validate()
    trees = sorted(glob.glob(f"{_ROOT}/families/*/gene_trees/ufboot1000.MFP.geneTree.newick"))[:n_families]
    m = GeneReconModel(_SP, trees, mode="specieswise", device=device, solver_options=so)
    if len(m.batch_statics) != 1:
        raise RuntimeError(f"expected 1 batch for {n_families} families, got {len(m.batch_statics)}")
    S = int(m.species_helpers["S"])
    theta = torch.full((S, 3), math.log2(init_rate), device=device, dtype=torch.float64)
    rw = torch.zeros(S, device=device, dtype=torch.float64)
    return m.batch_statics[0], theta, rw, S


def _static_theta_rw_from_capture(device):
    import os
    from gates._parity_kbench import _DEFAULT_CAP, gpurec_static_from_capture
    cap_path = os.environ.get("GPUREC_KB_CAPTURE", _DEFAULT_CAP)
    cap = torch.load(cap_path, map_location="cpu", weights_only=False)
    static = gpurec_static_from_capture(cap, device)
    for k, v in _SO.items():
        setattr(static.solver_options, k, v)
    static.solver_options.validate()
    theta = cap["inputs"]["theta"].to(device).double()
    rw = cap["inputs"]["col_weights"].to(device).double()
    return static, theta, rw, int(cap["meta"]["S"])


def run(n_families=8, device="cuda", use_capture=False, n_dirs=2, tangent_self_iters=128, seed=0):
    if use_capture:
        static, theta, rw, S = _static_theta_rw_from_capture(device)
        label = "666x80-capture"
    else:
        static, theta, rw, S = _static_theta_rw_from_live(n_families, device)
        label = f"{n_families}-family live"
    print(f"[hvp gate {label}] S={S} p={3*S} fp64 converged(pi={_SO['pi_iters']},"
          f"neu={_SO['neumann_terms']},tself={tangent_self_iters})")

    _loss, sv = forward_solve([static], theta, rw)
    gt, gc, cache = build_point_cache([static], theta, rw, sv)
    hvp = make_exact_hvp([static], theta, rw, sv, cache=cache, tangent_self_iters=tangent_self_iters)

    vg = make_value_and_grad([static], rw, theta_shape=(S, 3))
    x = theta.reshape(-1).contiguous()
    fd = _fd_hessian_hvp(vg, x, None, eps=1e-5)

    gen = torch.Generator(device=device).manual_seed(seed)
    p = 3 * S
    ok = True
    Hs = []
    for i in range(n_dirs):
        u = torch.randn(p, generator=gen, device=device, dtype=torch.float64)
        u /= u.norm()
        Ha = hvp(u).double()
        Hf = fd(u).double()
        Hs.append((u, Ha))
        finite = bool(torch.isfinite(Ha).all())
        abs_err = float((Ha - Hf).abs().max())
        rel = abs_err / max(float(Hf.abs().max()), 1e-30)
        good = finite and rel <= 5e-4
        ok &= good
        print(f"  dir {i}: finite={finite} |Ha|={float(Ha.norm()):.4f} |Hfd|={float(Hf.norm()):.4f} "
              f"max_abs={abs_err:.3e} max_rel={rel:.3e} {'PASS' if good else 'FAIL'}")
    if len(Hs) >= 2:
        (u, Hu), (w, Hw) = Hs[0], Hs[1]
        uHw, wHu = float(torch.dot(u, Hw)), float(torch.dot(w, Hu))
        scale = max((abs(float(torch.dot(u, Hu))) * abs(float(torch.dot(w, Hw)))) ** 0.5, 1e-30)
        sym = abs(uHw - wHu) / scale
        good = sym <= 5e-3
        ok &= good
        print(f"  symmetry: uHw={uHw:.6e} wHu={wHu:.6e} rel_asym={sym:.3e} {'PASS' if good else 'FAIL'}")
    print(f"[hvp gate] {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    cap = "--capture" in sys.argv
    nf = 8
    for a in sys.argv[1:]:
        if a.isdigit():
            nf = int(a)
    raise SystemExit(0 if run(n_families=nf, use_capture=cap) else 1)
