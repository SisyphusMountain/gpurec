"""S1 FD gate: make_value_and_grad(optimize_receiver=True) joint z=[theta; alpha] gradient.

Verifies that wiring grad_receiver into the optimizer (z = [theta.reshape(-1); alpha], length
theta.numel()+S) yields a gradient g_z whose directional derivative matches a central FD of the
NLL, at a NON-UNIFORM base alpha (so the alpha code paths are LIVE, not the degenerate uniform
center), with a converged fp64 solver. Also checks the gauge identity 1^T g_alpha == 0.

    python -m gates._verify_recv_grad

Mirrors the fixture build of _verify_hvp / _verify_map.
"""

from __future__ import annotations

import glob
import math

import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.core.parameters.extract_parameters import (
    receiver_log_probs_from_weights,
    receiver_valid_log_normalizer,
)
from gpurec.solver.value_and_grad import make_value_and_grad
from gates._paths import HOGENOM_ROOT

_ROOT = str(HOGENOM_ROOT)
_SP = (f"{_ROOT}/runs/MFP/true_start_ufboot1000/"
       "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/"
       "starting_species_tree.newick")
# converged fp64 solver (same as _verify_hvp): neither fwd nor bwd truncation-limited
_SO = dict(e_max_iter=2000, e_tol=1e-10, pi_iters=128, neumann_terms=64,
           bicgstab_max_iter=500, bicgstab_tol=1e-10,
           bicgstab_breakdown_tol=1e-30, adjoint_pruning_threshold=1e-6,
           use_adjoint_pruning=True, pibar_side_threshold=0.0)


def _valid_mass_min(static, alpha):
    """Min over species of valid_mass = 1 - ancestor_mass at receiver logits alpha.

    extract_parameters returns -inf where valid_mass <= 0; assert this is bounded away from 0.
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


def run(n_families=8, device="cuda", seed=0):
    so = SolverOptions(**_SO)
    so.validate()
    trees = sorted(glob.glob(
        f"{_ROOT}/families/*/gene_trees/ufboot1000.MFP.geneTree.newick"))[:n_families]
    m = GeneReconModel(_SP, trees, mode="specieswise", device=device, solver_options=so)
    if len(m.batch_statics) != 1:
        raise RuntimeError(f"expected 1 batch, got {len(m.batch_statics)}")
    static = m.batch_statics[0]
    S = int(m.species_helpers["S"])
    theta_numel = 3 * S
    p = theta_numel + S

    torch.manual_seed(seed)
    theta = (torch.full((S, 3), math.log2(0.1), device=device, dtype=torch.float64)
             + 0.2 * torch.randn(S, 3, device=device, dtype=torch.float64))
    # NON-UNIFORM base alpha so the receiver-weight code paths are LIVE (uniform -> dead paths)
    gen0 = torch.Generator(device=device).manual_seed(0)
    alpha = 0.2 * torch.randn(S, generator=gen0, device=device, dtype=torch.float64)

    nonuniform = not receiver_weights_are_uniform(alpha)
    assert nonuniform, "base alpha is uniform -> alpha paths dead, gate certifies nothing"
    vm0 = _valid_mass_min(static, alpha)
    assert vm0 > 1e-3, f"valid_mass too small at base alpha: {vm0}"

    z0 = torch.cat([theta.reshape(-1), alpha]).contiguous()
    f = make_value_and_grad([static], alpha, theta_shape=(S, 3), optimize_receiver=True)
    L0, g_z, _, _ = f(z0)
    g_z = g_z.double()
    assert g_z.numel() == p, f"g_z len {g_z.numel()} != p {p}"
    print(f"[recv grad gate {n_families}-family] S={S} p={p} fp64 converged "
          f"(pi={_SO['pi_iters']},neu={_SO['neumann_terms']}) base_loss={L0:.4f} "
          f"valid_mass_min={vm0:.4f} nonuniform={nonuniform}")

    g_alpha = g_z[theta_numel:]
    g_theta = g_z[:theta_numel]
    # gauge: NLL invariant under alpha->alpha+c1 => 1^T g_alpha == 0
    one_dot = float(g_alpha.sum())
    gauge_rel = abs(one_dot) / max(float(g_alpha.norm()), 1e-30)
    print(f"  gauge: 1^T g_alpha = {one_dot:.3e}  |.|/|g_alpha| = {gauge_rel:.3e}")

    eps = 1e-6

    def fd_dir(v):
        Lp, _, _, _ = f(z0 + eps * v, want_grad=False)
        Lm, _, _, _ = f(z0 - eps * v, want_grad=False)
        return (Lp - Lm) / (2 * eps)

    ok = True
    rels = []
    # (a) random mixed directions in R^p
    for s in range(4):
        gen = torch.Generator(device=device).manual_seed(200 + s)
        v = torch.randn(p, generator=gen, device=device, dtype=torch.float64)
        v /= v.norm()
        # valid_mass guard at base +/- eps*v (alpha block of v)
        va = v[theta_numel:]
        vmp = _valid_mass_min(static, alpha + eps * va)
        vmm = _valid_mass_min(static, alpha - eps * va)
        assert min(vmp, vmm) > 1e-3, f"valid_mass collapses along dir {s}: {min(vmp, vmm)}"
        fd = fd_dir(v)
        ana = float(torch.dot(g_z, v))
        rel = abs(fd - ana) / max(1.0, abs(ana))
        rels.append(rel)
        ok &= rel < 1e-4
        print(f"  FD mixed dir {s}: fd={fd:.6f} ana={ana:.6f} rel={rel:.2e}")

    # (b) pure-alpha unit directions (the new block)
    arels = []
    for s in range(4):
        gen = torch.Generator(device=device).manual_seed(300 + s)
        va = torch.randn(S, generator=gen, device=device, dtype=torch.float64)
        va /= va.norm()
        v = torch.cat([torch.zeros(theta_numel, device=device, dtype=torch.float64), va])
        vmp = _valid_mass_min(static, alpha + eps * va)
        vmm = _valid_mass_min(static, alpha - eps * va)
        assert min(vmp, vmm) > 1e-3, f"valid_mass collapses pure-alpha dir {s}: {min(vmp, vmm)}"
        fd = fd_dir(v)
        ana = float(torch.dot(g_alpha, va))
        rel = abs(fd - ana) / max(1.0, abs(ana))
        arels.append(rel)
        ok &= rel < 1e-4
        print(f"  FD pure-alpha dir {s}: fd={fd:.6f} ana={ana:.6f} rel={rel:.2e}")

    ok &= gauge_rel < 1e-6

    # sanity: theta-only legacy contract still matches the joint theta block at uniform-base? No --
    # the alpha base changes the forward, so just confirm legacy API still runs and returns 3S grad.
    f_legacy = make_value_and_grad([static], alpha, theta_shape=(S, 3))
    _Ll, g_leg, _, _ = f_legacy(theta.reshape(-1))
    legacy_ok = g_leg.numel() == theta_numel
    ok &= legacy_ok

    print(f"  max rel (mixed)={max(rels):.2e}  max rel (pure-alpha)={max(arels):.2e}  "
          f"gauge_rel={gauge_rel:.2e}  legacy_len_ok={legacy_ok}")
    print(f"[recv grad gate] {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
