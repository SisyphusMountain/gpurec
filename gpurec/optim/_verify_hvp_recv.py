"""S2 gate: (theta, alpha) joint exact-HVP FD test infrastructure (kernels NOT yet changed).

This is the BASELINE gate that the receiver-weight HVP effort (plan steps S3..S8,
``docs/optim/receiver_weights_hvp_plan.md``) must turn green. It builds the joint-variable
``z = [theta.reshape(-1); alpha]`` (length ``4S``) HVP finite-difference infrastructure and
compares it, BLOCK BY BLOCK and under the gauge projector ``P_z``, against the CURRENT analytic
exact HVP (``make_exact_hvp``, theta-only).

What it does today (before any kernel change), on a NON-UNIFORM base alpha (so the receiver-weight
code paths are LIVE -- plan correction 2), fp64, converged solver (pi>=128, neumann>=64):

  * FD side: ``make_value_and_grad(optimize_receiver=True)`` + ``_fd_hessian_hvp`` perturb the FULL
    ``z`` in R^{4S}, so the FD HVP carries ALL four Hessian blocks (H_tt, H_ta, H_at, H_aa).
  * Analytic side: the current ``make_exact_hvp`` consumes ONLY ``u_theta`` (R^{3S}) and emits
    ``H_tt u_theta`` (R^{3S}); the alpha row + the alpha-coupling are MISSING. We embed it into
    R^{4S} with a zero alpha block (``H_at = 0``, ``H_aa = 0``) and pass only ``u[:3S]`` in
    (``H_ta = 0`` too): this is exactly "theta-only" today.

  * Everything is projected with ``P_z = blockdiag(I_{3S}, I - (1/S) 11^T)`` on BOTH sides AND in
    the denominator (plan correction 1). Per-block relative errors (pure-theta, pure-alpha, mixed)
    + null-space leakage are reported.

EXPECTED BASELINE (the discriminator):
  * pure-THETA block:  rel <= 5e-4  (PASS -- the theta-only HVP still works on a non-uniform base)
  * pure-ALPHA block:  rel = O(1)   (FAIL -- H_aa/H_at are missing today)
  * MIXED  block:      rel = O(1)   (FAIL -- H_ta/H_aa missing today)

``run()`` returns (theta_ok, alpha_ok, mixed_ok, leak_ana, leak_fd). ``ok_to_proceed`` for S2 is
``theta_ok and not alpha_ok`` -- the harness must DISCRIMINATE (theta passes, alpha fails).

As S3..S8 land, the SAME gate is re-run; the alpha/mixed blocks must turn green (rel <= 5e-4) and
the analytic null-space leakage must drop to the truncation floor.

    python -m gpurec.optim._verify_hvp_recv            # tiny live-hogenom model (local, fp64)
"""

from __future__ import annotations

import glob
import math

import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.core.parameters.extract_parameters import receiver_log_probs_from_weights
from gpurec.optim.hvp_exact import build_point_cache, make_exact_hvp
from gpurec.optim.newton_cg import _fd_hessian_hvp
from gpurec.optim.value_and_grad import forward_solve, make_value_and_grad

_ROOT = "/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_hogenom_core/hogenom"
_SP = (f"{_ROOT}/runs/MFP/true_start_ufboot1000/"
       "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/"
       "starting_species_tree.newick")
# converged fp64 solver (same as _verify_hvp / _verify_recv_grad): neither fwd nor bwd
# truncation-limited so the 5e-4 floor reflects residual solver truncation only.
_SO = dict(e_max_iter=2000, e_tol=1e-10, pi_iters=128, neumann_terms=64,
           self_loop_solver="neumann", bicgstab_max_iter=500, bicgstab_tol=1e-10,
           bicgstab_breakdown_tol=1e-30, adjoint_pruning_threshold=1e-6,
           use_adjoint_pruning=True, pibar_side_threshold=0.0)


# --------------------------------------------------------------------------- gauge helpers (P_z)
def proj_alpha(g_alpha: torch.Tensor) -> torch.Tensor:
    """P g_alpha = g_alpha - mean(g_alpha)  (mean-subtract; removes the 1_S null mode)."""
    return g_alpha - g_alpha.mean()


def proj_z(u: torch.Tensor, theta_numel: int) -> torch.Tensor:
    """P_z u = blockdiag(I_{theta}, I - (1/S) 11^T) u  (identity on theta, mean-subtract on alpha)."""
    return torch.cat([u[:theta_numel], proj_alpha(u[theta_numel:])])


def _alpha_leak(Hu: torch.Tensor, theta_numel: int) -> float:
    """Null-space leakage |(1/sqrt S) 1_S^T (Hu)_alpha| = sqrt(S) * |mean((Hu)_alpha)|.

    The gauge-invariance numerical certificate: H_aa 1_S = 0 and 1^T H_at = 0, so the all-ones
    component of the alpha row of H u must be at the truncation floor.
    """
    ga = Hu[theta_numel:]
    S = int(ga.numel())
    return float(abs(ga.sum()) / max(math.sqrt(S), 1.0))


def _valid_mass_min(static, alpha: torch.Tensor) -> float:
    """Min over species of valid_mass = 1 - ancestor_mass at receiver logits alpha.

    extract_parameters returns -inf where valid_mass <= 0; assert this is bounded away from 0 so
    the FD does not silently step into the -inf region. (Copied from _verify_recv_grad.)
    """
    rlp = receiver_log_probs_from_weights(alpha)
    sh = static.species_helpers
    sp_parent = sh["sp_parent"].to(device=alpha.device)
    depth = int(sh["max_ancestor_depth"])
    receiver_probs = torch.exp2(rlp)
    parent = sp_parent.to(dtype=torch.long)
    S = int(rlp.numel())
    cur = torch.arange(S, device=alpha.device, dtype=torch.long)
    ancestor_mass = torch.zeros((S,), device=alpha.device, dtype=rlp.dtype)
    zero = torch.zeros((), device=alpha.device, dtype=rlp.dtype)
    for _ in range(max(1, depth)):
        valid = (cur >= 0) & (cur < S)
        safe_cur = cur.clamp(0, max(S - 1, 0))
        ancestor_mass = ancestor_mass + torch.where(valid, receiver_probs.index_select(0, safe_cur), zero)
        next_cur = parent.index_select(0, safe_cur)
        cur = torch.where(valid, next_cur, torch.full_like(cur, -1))
    return float((1.0 - ancestor_mass).min())


# ------------------------------------------------------------------------------ fixture (S2: non-uniform base)
def _static_theta_alpha_from_live(n_families, device, seed=0, init_rate=0.1):
    """Build the 8-hogenom fixture (S=1331) at a NON-UNIFORM base alpha = 0.2*randn(S), seeded.

    Mirrors _verify_hvp's _static_theta_rw_from_live but replaces ``rw = zeros(S)`` (the degenerate
    UNIFORM center where the alpha paths are DEAD) with a seeded non-uniform base, and asserts
    not-uniform + valid_mass > 1e-3 (plan correction 2 / the S2 prerequisite).
    """
    so = SolverOptions(**_SO)
    so.validate()
    trees = sorted(glob.glob(
        f"{_ROOT}/families/*/gene_trees/ufboot1000.MFP.geneTree.newick"))[:n_families]
    m = GeneReconModel(_SP, trees, mode="specieswise", device=device, solver_options=so)
    if len(m.batch_statics) != 1:
        raise RuntimeError(f"expected 1 batch for {n_families} families, got {len(m.batch_statics)}")
    static = m.batch_statics[0]
    S = int(m.species_helpers["S"])
    theta = torch.full((S, 3), math.log2(init_rate), device=device, dtype=torch.float64)
    g = torch.Generator(device=device).manual_seed(seed)
    alpha = 0.2 * torch.randn(S, generator=g, device=device, dtype=torch.float64)
    alpha = alpha - alpha.mean()  # land on the gauge slice (cosmetic; P_z handles it regardless)
    assert not receiver_weights_are_uniform(alpha), \
        "base alpha is uniform -> alpha paths dead, gate certifies nothing (plan correction 2)"
    vm0 = _valid_mass_min(static, alpha)
    assert vm0 > 1e-3, f"valid_mass too small at base alpha: {vm0}"
    return static, theta, alpha, S, vm0


def _embed_theta_hvp(hvp_theta, theta_numel: int, p: int):
    """Wrap the CURRENT theta-only ``make_exact_hvp`` into a joint R^{4S}->R^{4S} operator.

    Today the analytic HVP only knows H_tt: it takes ``u_theta`` (R^{theta_numel}) and returns
    ``H_tt u_theta`` (R^{theta_numel}). We feed it ONLY ``u[:theta_numel]`` (so H_ta u_alpha is
    dropped -> implicitly 0) and emit a length-p vector with the alpha block zeroed (H_at=H_aa=0).
    This is the literal "alpha block missing" analytic operator the baseline must expose.
    """
    def hvp(u: torch.Tensor) -> torch.Tensor:
        u = u.double()
        Ht = hvp_theta(u[:theta_numel].contiguous()).double()  # R^{theta_numel}
        out = torch.zeros(p, device=u.device, dtype=torch.float64)
        out[:theta_numel] = Ht.reshape(-1)
        return out
    return hvp


# --------------------------------------------------------------------------------------------- gate
def run(n_families=8, device="cuda", seed=0, tangent_self_iters=128, eps=1e-5, n_each=3):
    static, theta, alpha, S, vm0 = _static_theta_alpha_from_live(n_families, device, seed=seed)
    theta_numel = 3 * S
    p = theta_numel + S  # 4S
    nonuniform = not receiver_weights_are_uniform(alpha)
    print(f"[recv-hvp gate {n_families}-family] S={S} theta_numel={theta_numel} p(4S)={p} "
          f"fp64 converged(pi={_SO['pi_iters']},neu={_SO['neumann_terms']},tself={tangent_self_iters}) "
          f"nonuniform={nonuniform} valid_mass_min={vm0:.4f}")

    # ---- analytic side (CURRENT theta-only make_exact_hvp), embedded into R^{4S}
    _loss, sv = forward_solve([static], theta, alpha)
    _gt, _gc, cache = build_point_cache([static], theta, alpha, sv)
    hvp_theta = make_exact_hvp([static], theta, alpha, sv, cache=cache,
                               tangent_self_iters=tangent_self_iters)
    Ha_op = _embed_theta_hvp(hvp_theta, theta_numel, p)

    # ---- FD side: joint z=[theta; alpha] in R^{4S} (carries all four blocks)
    vg = make_value_and_grad([static], alpha, theta_shape=(S, 3), optimize_receiver=True)
    z = torch.cat([theta.reshape(-1), alpha]).contiguous()
    Hf_op = _fd_hessian_hvp(vg, z, None, eps=eps)

    def block_dirs(kind, k):
        """Seeded unit directions: 'theta' (u_alpha=0), 'alpha' (u_theta=0), 'mixed'."""
        gen = torch.Generator(device=device).manual_seed(seed + {"theta": 100, "alpha": 200,
                                                                  "mixed": 300}[kind] + k)
        if kind == "theta":
            ut = torch.randn(theta_numel, generator=gen, device=device, dtype=torch.float64)
            u = torch.cat([ut, torch.zeros(S, device=device, dtype=torch.float64)])
        elif kind == "alpha":
            ua = torch.randn(S, generator=gen, device=device, dtype=torch.float64)
            u = torch.cat([torch.zeros(theta_numel, device=device, dtype=torch.float64), ua])
        else:
            u = torch.randn(p, generator=gen, device=device, dtype=torch.float64)
        return u / u.norm()

    def assert_valid_along(u):
        ua = u[theta_numel:]
        nu = float(ua.norm())
        if nu == 0.0:
            return
        ua = ua / nu
        for s in (+1.0, -1.0):
            vm = _valid_mass_min(static, alpha + s * eps * ua)
            assert vm > 1e-3, f"valid_mass collapses along alpha dir: {vm}"

    results = {}
    leaks_ana, leaks_fd = [], []
    # THETA-MILESTONE: the H_tt block on a pure-theta direction (u_alpha=0). The analytic op
    # emits ONLY the theta row (the alpha row H_at is the SEPARATE alpha-block run S4-S8, zeroed
    # by _embed_theta_hvp), so the FOUNDATION quantity proven here is |H_tt u_t - FD_theta_row| /
    # |FD_theta_row| on the theta directions -- the weighted forward-over-reverse THETA block.
    theta_block_rels = []
    ana_failed = False  # the CURRENT theta-only analytic HVP may NaN at a non-uniform base (see below)
    for kind in ("theta", "alpha", "mixed"):
        rels = []
        for k in range(n_each):
            u = block_dirs(kind, k)
            assert_valid_along(u)
            Hf = Hf_op(u)
            leaks_fd.append(_alpha_leak(Hf, theta_numel))
            try:
                Ha = Ha_op(u)
            except RuntimeError as e:
                # The theta-only make_exact_hvp's tangent E-adjoint BiCGSTAB diverges at a
                # non-uniform base (the tangent forward sweep linearizes the UNIFORM forward,
                # inconsistent with the weighted primal fixed point -> dPi/dE -> inf/NaN). This is
                # the S3+S7 prerequisite the gate exposes; report it instead of crashing.
                ana_failed = True
                print(f"  {kind:5s} dir {k}: analytic HVP RAISED ({str(e).splitlines()[0][:80]})")
                rels.append(float("inf"))
                if kind == "theta":
                    theta_block_rels.append(float("inf"))
                continue
            finite_a = bool(torch.isfinite(Ha).all())
            if not finite_a:
                ana_failed = True
            leaks_ana.append(_alpha_leak(Ha, theta_numel))
            PHa = proj_z(Ha, theta_numel)
            PHf = proj_z(Hf, theta_numel)
            num = float((PHa - PHf).norm())
            den = max(float(PHf.norm()), 1e-30)
            rel = num / den if finite_a else float("inf")
            rels.append(rel)
            # per-block sub-rel (which block dominates the mismatch)
            dt = float((PHa[:theta_numel] - PHf[:theta_numel]).norm())
            da = float((PHa[theta_numel:] - PHf[theta_numel:]).norm())
            # theta-block (H_tt) relative error on the pure-theta direction = the milestone quantity
            if kind == "theta":
                den_tt = max(float(PHf[:theta_numel].norm()), 1e-30)
                theta_block_rels.append((dt / den_tt) if finite_a else float("inf"))
            print(f"  {kind:5s} dir {k}: finite_a={finite_a} |PHa|={float(PHa.norm()):.4f} "
                  f"|PHf|={float(PHf.norm()):.4f} rel={rel:.3e}  "
                  f"(theta-part_err={dt:.3e} alpha-part_err={da:.3e})")
        results[kind] = max(rels)
    theta_block_rel = max(theta_block_rels) if theta_block_rels else float("inf")

    leak_ana = max(leaks_ana) if leaks_ana else float("nan")
    leak_fd = max(leaks_fd)
    print(f"  null-space leakage |1^T(Hu)_a|/sqrt(S):  analytic_max={leak_ana:.3e}  fd_max={leak_fd:.3e}")
    if ana_failed:
        print("  NOTE: the CURRENT theta-only make_exact_hvp could not produce a finite HVP at the "
              "non-uniform base (tangent E-adjoint diverged). This is the S3+S7 prerequisite the "
              "joint gate exposes; see the blocker in the report.")

    theta_ok = results["theta"] <= 5e-4
    alpha_ok = results["alpha"] <= 5e-4
    mixed_ok = results["mixed"] <= 5e-4
    # THETA-MILESTONE gate: the H_tt block on the pure-theta direction. This is the FOUNDATION
    # (S7+S3) result -- the weighted forward-over-reverse THETA block matches FD at the non-uniform
    # base. The full-vector theta-block `results["theta"]` ALSO carries the alpha ROW (H_at u_t),
    # which the analytic op zeros until the alpha-block run (S4-S8); so the milestone is gated on
    # theta_block_rel = |H_tt u_t - FD_theta_row|/|FD_theta_row|, NOT the full 4S vector.
    theta_block_ok = theta_block_rel <= 5e-4
    print(f"  THETA-MILESTONE  H_tt block (pure-theta dir) rel={theta_block_rel:.3e} "
          f"[{'PASS' if theta_block_ok else 'FAIL'}]  (<=5e-4: weighted fwd-over-reverse theta block)")
    print(f"  BLOCK rel (full 4S, incl. not-yet-built alpha row):  "
          f"theta={results['theta']:.3e} [{'PASS' if theta_ok else 'FAIL'}]  "
          f"alpha={results['alpha']:.3e} [{'PASS' if alpha_ok else 'FAIL'}]  "
          f"mixed={results['mixed']:.3e} [{'PASS' if mixed_ok else 'FAIL'}]")

    # The FD side (joint optimize_receiver=True) DOES carry the alpha block; confirm it is
    # non-trivial so the gate genuinely probes H_aa/H_at (not a zero direction).
    gen = torch.Generator(device=device).manual_seed(seed + 999)
    ua = torch.randn(S, generator=gen, device=device, dtype=torch.float64); ua /= ua.norm()
    uA = torch.cat([torch.zeros(theta_numel, device=device, dtype=torch.float64), ua])
    HfA = Hf_op(uA)
    fd_alpha_block_norm = float(proj_z(HfA, theta_numel)[theta_numel:].norm())
    print(f"  FD pure-alpha H_aa+H_at block norm (P-projected) = {fd_alpha_block_norm:.4e} "
          f"(must be >> 0: the missing block the effort must reproduce)")

    all_green = theta_ok and alpha_ok and mixed_ok
    # S2 ok_to_proceed: the harness DISCRIMINATES. The current analytic op has NO alpha block AND
    # (proven) cannot even evaluate at a non-uniform base -> the alpha/mixed gates FAIL while the
    # FD side carries a real alpha block. theta_block_state is reported separately.
    fd_carries_alpha = fd_alpha_block_norm > 1e-6
    baseline_discriminates = (not all_green) and fd_carries_alpha and (not alpha_ok)
    print(f"[recv-hvp gate] theta_milestone={theta_block_ok}  all_green={all_green}  "
          f"ana_failed={ana_failed}  fd_carries_alpha={fd_carries_alpha}  "
          f"discriminates={baseline_discriminates}")
    return dict(theta=results["theta"], alpha=results["alpha"], mixed=results["mixed"],
                theta_ok=theta_ok, alpha_ok=alpha_ok, mixed_ok=mixed_ok,
                theta_block_rel=theta_block_rel, theta_block_ok=theta_block_ok,
                leak_ana=leak_ana, leak_fd=leak_fd, all_green=all_green, ana_failed=ana_failed,
                fd_alpha_block_norm=fd_alpha_block_norm,
                baseline_discriminates=baseline_discriminates)


if __name__ == "__main__":
    r = run()
    # THETA-MILESTONE success == the H_tt block (pure-theta dir) matches FD at the non-uniform base
    # (S7+S3 done). The alpha/mixed blocks remain FAIL until the alpha-block run (S4-S8).
    raise SystemExit(0 if r["theta_block_ok"] else 1)
