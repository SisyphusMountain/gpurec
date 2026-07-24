"""Exact second-order (HVP) fraction-missing correctness for the rate parameters ``theta``.

Fraction-missing is E-ONLY (it enters the extinction E at leaf species and NOTHING in
Pi/reconciliation), so the exact analytic HVP w.r.t. ``theta`` only needs the E-step tangent to
inject the fixed leaf boundary and the point-cache adjoint to be built with the same boundary.
fraction_missing is a CONSTANT input: no gradient or curvature is taken w.r.t. it.

FIXTURE NOTE. The task's exact gene tree ``((A_1,B_1),C_1)`` (matching the species topology) is
rate-DEGENERATE: its NLL is EXACTLY constant in ``theta`` (unique speciation-only reconciliation ->
true Hessian == 0), so it cannot exercise the second-order code at all (verified in
``test_specified_matching_gene_tree_is_rate_degenerate`` below). We keep the task's SPECIES tree and
use a gene tree with an in-species-A duplication (an extra ``A_2`` copy), which forces a
duplication + losses and yields a genuinely rate-dependent NLL whose Hessian fraction-missing
modifies (curvature -1.7e-2 without fm -> -4.0e-2 at fm=0.3).

Gate 1: analytic HVP ``H u`` (real ``make_exact_hvp`` entry point) vs central finite-difference of
the FIRST-ORDER gradient (``m().backward()`` -- the fraction-missing-correct gradient verified by
``test_fraction_missing_grad.py``) along a random unit direction ``u``. Relative L2 error < 1e-2.
Gate 2: with fraction_missing=0.0 the HVP is bit-identical to the no-fraction-missing HVP (the
``USE_FRACTION_MISSING`` constexpr compiles the new code out when ``leaf_fm_log is None``).
"""
import torch, pytest
pytestmark = pytest.mark.gpu

from gpurec.api.model import GeneReconModel
from gpurec.solver.value_and_grad import forward_solve
from gpurec.solver.hvp_exact import make_exact_hvp

_SPECIES = "((A:1,B:1)AB:1,C:1)Root;"
_GENE_DUP = "(((A_1:1,B_1:1)x:1,C_1:1)y:1,A_2:1)GeneRoot;"  # nonzero, fm-sensitive Hessian
_GENE_MATCH = "((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;"           # the task's exact (degenerate) gene tree
# Converged tangent self-loop: this tiny fixture's Pi fixed point converges in a handful of Jacobi
# steps, so this is well past convergence and matches the (converged) production forward/gradient.
_TANGENT_SELF_ITERS = 256


def _build(tmp_path, gene, fm, tag):
    sp = tmp_path / f"s_{tag}.nwk"; sp.write_text(_SPECIES)
    g = tmp_path / f"g_{tag}.nwk"; g.write_text(gene)
    return GeneReconModel(str(sp), [str(g)], mode="specieswise", device="cuda",
                          dtype=torch.float64, fraction_missing=fm)


def _autograd_grad(m, theta):
    """Production first-order gradient d(NLL)/dtheta via m().backward() -- the fraction-missing-correct
    gradient verified by test_fraction_missing_grad.py; the FD reference (independent of HVP internals)."""
    tt = theta.detach().clone().requires_grad_(True)
    m(tt).backward()
    return tt.grad.reshape(-1).detach().clone()


def _hvp(m, theta):
    st = m.batch_statics[0]
    cw = m.receiver_weights.detach()
    _loss, sv = forward_solve(st, theta, cw)
    return make_exact_hvp(st, theta, cw, sv, tangent_self_iters=_TANGENT_SELF_ITERS)


def test_hvp_matches_central_fd_of_grad_with_fraction_missing(tmp_path):
    m = _build(tmp_path, _GENE_DUP, 0.3, "dup")
    theta0 = m.theta.detach().clone()
    torch.manual_seed(0)
    u = torch.randn(theta0.numel(), dtype=torch.float64, device="cuda")
    u = u / u.norm()

    Hu = _hvp(m, theta0)(u).reshape(-1).double()

    eps = 1e-5
    flat = theta0.reshape(-1)
    gp = _autograd_grad(m, (flat + eps * u).reshape(theta0.shape))
    gm = _autograd_grad(m, (flat - eps * u).reshape(theta0.shape))
    fd = ((gp - gm) / (2 * eps)).double()

    # Guard: the fixture must actually have curvature, else the comparison is vacuous.
    assert float(fd.norm()) > 1e-3, f"fixture has ~zero true Hessian (||fd||={float(fd.norm()):.3e})"
    rel = float((Hu - fd).norm() / fd.norm())
    print(f"HVP-vs-FD relative L2 error = {rel:.3e}  (||Hu||={float(Hu.norm()):.4e}, "
          f"||fd||={float(fd.norm()):.4e})")
    assert rel < 1e-2, (
        f"analytic HVP disagrees with central FD of the fraction-missing gradient: "
        f"rel L2 = {rel:.3e}\nHu = {Hu}\nfd = {fd}"
    )


def test_hvp_fraction_missing_zero_bit_identical_to_none(tmp_path):
    m_none = _build(tmp_path, _GENE_DUP, None, "none")
    m_zero = _build(tmp_path, _GENE_DUP, 0.0, "zero")
    # fraction_missing=0.0 collapses to the no-fraction-missing fast path (build_fraction_missing_tensors
    # returns None when nothing is missing), so both take the USE_FRACTION_MISSING=False path.
    assert m_none.leaf_fm_log is None
    assert m_zero.leaf_fm_log is None

    theta0 = m_none.theta.detach().clone()
    torch.manual_seed(1)
    u = torch.randn(theta0.numel(), dtype=torch.float64, device="cuda")
    u = u / u.norm()

    Hu_none = _hvp(m_none, theta0)(u).reshape(-1)
    Hu_zero = _hvp(m_zero, m_zero.theta.detach().clone())(u).reshape(-1)
    assert torch.equal(Hu_none, Hu_zero), (
        "fraction_missing=0.0 HVP is not bit-identical to the no-fraction-missing HVP "
        f"(max abs diff {float((Hu_none - Hu_zero).abs().max()):.3e})"
    )


def test_specified_matching_gene_tree_is_rate_degenerate(tmp_path):
    """Documents the fixture choice: the task's exact (matching) gene tree has an EXACTLY
    theta-constant NLL, so BOTH the analytic HVP and the FD gradient are ~0 there -- it cannot
    validate curvature. This is why the gates above use the duplication gene tree instead."""
    m = _build(tmp_path, _GENE_MATCH, 0.3, "match")
    theta0 = m.theta.detach().clone()
    torch.manual_seed(0)
    u = torch.randn(theta0.numel(), dtype=torch.float64, device="cuda")
    u = u / u.norm()

    Hu = _hvp(m, theta0)(u).reshape(-1).double()
    eps = 1e-5
    flat = theta0.reshape(-1)
    fd = ((_autograd_grad(m, (flat + eps * u).reshape(theta0.shape))
           - _autograd_grad(m, (flat - eps * u).reshape(theta0.shape))) / (2 * eps)).double()
    print(f"[degenerate matching fixture] ||Hu||={float(Hu.norm()):.3e}  ||fd||={float(fd.norm()):.3e}")
    assert float(fd.norm()) < 1e-6, "matching gene tree unexpectedly has curvature"
    assert float(Hu.norm()) < 1e-6, "analytic HVP is nonzero where the true Hessian is zero"
