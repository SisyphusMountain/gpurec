"""The exact tree solves must equal the iterated ones when the weights are NOT uniform.

``forward_self_loop="exact"`` and ``adjoint_self_loop="exact"`` do not approximate the iteration --
they solve the same linear system on the species tree directly, so their answer is what the iteration
converges to. The scale runs that established this used UNIFORM transfer receiver weights (the
``receiver_weights`` vector all zero), which is what the genewise recipe uses, so the weighted code
path was thinly covered; a missing receiver-weight term in the exact TANGENT at internal species
nodes was found late by tests/test_origination_curvature.py.

These tests turn all three weightings on at once and check every output the fit consumes:

  * ``receiver_weights``    -- per-species logits for WHO receives a transfer. Non-zero values make
                              the receiver distribution non-uniform and switch the kernels onto their
                              weighted path.
  * ``origination_weights`` -- per-species logits for WHERE a family starts. They enter only the NLL
                              head, but differentiating that head drives the same adjoint solve.
  * ``fraction_missing``    -- per-leaf probability that a gene present in a species is not observed
                              there. It changes the leaf boundary of the extinction recurrence, i.e.
                              of the very system the exact solve eliminates.

Compared, in float64 on a small fixture: per-family NLL, the theta gradient, the receiver-weight
gradient, the origination gradient, the genewise 3x3 curvature blocks, and the full joint
(theta, receiver, origination) second-derivative matrix -- which can only be built with the weights
switched on at all, because it refuses to run at a uniform base.

Every solver setting is written out explicitly rather than inherited from ``SolverOptions``'
defaults: the defaults ARE one of the two arms, so a test that relied on them would stop comparing
anything the day they changed.
"""
from __future__ import annotations

import math

import pytest
import torch

from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.core.scheduling import batching

pytestmark = pytest.mark.gpu

# Everything except the two self-loop implementations is held fixed across the two arms, so a
# difference can only come from the self-loop.
#   * adjoint pruning OFF -- it drops small adjoint contributions and is a SEPARATE approximation;
#     left on, it would add its own difference on top of the one being measured.
#   * e_adjoint_tol None -- "use the dtype's own resolution" (about 1e-12 in float64). The genewise
#     recipe's 1e-7 is a float32 number and would cap this float64 comparison five orders early.
#   * e_tol 1e-15 is about 4.5x float64's eps (2.22e-16), i.e. as tight as the extinction fixed
#     point can actually get.
_SHARED_SOLVER = dict(
    e_max_iter=500,
    e_tol=1e-15,
    e_adjoint_max_iter=500,
    e_adjoint_tol=None,
    adjoint_pruning_threshold=1e-6,
    use_adjoint_pruning=False,
    pibar_side_threshold=0.0,
)

# Iteration counts for the "iterated" arm. The self-loop operator is a strong contraction (spectral
# radius about 0.04, so each term is ~25x smaller than the last); 200 forward iterations and 200
# Neumann terms are far past the point where a float64 sum can still move, which is what makes this
# arm the CONVERGED reference rather than a truncation of it.
_ITERATED_STEPS = 200

# Largest tolerated disagreement, relative to the largest entry of the reference. Both arms compute
# the same quantity by different arithmetic, so the gap is rounding, not method error; 1e-9 leaves
# room for the accumulation over the fixture's clades while still being ~7 orders below anything a
# real mistake in the weighted terms would produce (the tangent bug this file guards against showed
# up at 2.6e-8 on a symmetry check that had to hold to 1e-10).
_TOLERANCE = 1e-9

# The tangent used by the Hessian probes follows ``adjoint_self_loop``; when it iterates it needs an
# iteration count, and the exact tangent ignores this value entirely.
_TANGENT_ITERS = _ITERATED_STEPS

_SPECIES_NEWICK = "(((A:1,B:1)AB:1,(C:1,D:1)CD:1)ABCD:1,(E:1,F:1)EF:1)Root:1;\n"
_GENE_NEWICKS = (
    "(((A_1:1,B_1:1)g1:1,(C_1:1,D_1:1)g2:1)g3:1,(E_1:1,F_1:1)g4:1)GeneRoot:1;\n",
    "((A_1:1,C_1:1)h1:1,(E_1:1,F_1:1)h2:1)GeneRoot:1;\n",
    "((B_1:1,D_1:1)k1:1,(C_1:1,E_1:1)k2:1)GeneRoot:1;\n",
)
# Leaves given a non-zero fraction missing, and how much. Chosen by hand rather than at random so a
# failure is reproducible from the file alone: a mix of sizeable and small values, and two leaves
# (D, F) left fully observed so the "some species missing, some not" branch is exercised too.
#
# KNOWN LIMIT, unrelated to what this file tests: on real data these same values can make the
# gradient impossible to compute at all. The gradient needs a separate linear solve -- the
# extinction adjoint, ``_neumann_e_adjoint`` in gpurec/api/_implicit_grad.py -- whose Neumann series
# assumes the extinction step's Jacobian shrinks a vector. Measured on 100 Coleman families
# (benchmark/cc/diagnose_fraction_missing_adjoint.py): with no missing data it shrinks by 0.55 per
# term, at fraction_missing up to 0.05 by 0.55, up to 0.10 by 0.83, and at 0.20 and above it GROWS
# by about 1.45 per term until the term overflows -- so the solve raises and there is no gradient.
# Two of those 100 families do it on their own, both with a very small duplication and transfer rate
# and a much larger loss rate. This fixture's three families are far from that corner, so the values
# below are safe here.
_FRACTION_MISSING = {"A": 0.30, "B": 0.05, "C": 0.45, "E": 0.20}


def _require_gpu():
    pytest.importorskip("triton")
    try:
        batching._load_native_module()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        torch.empty(1, device="cuda")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CUDA runtime is unavailable: {exc}")


def _write_fixture(tmp_path):
    species = tmp_path / "species.nwk"
    species.write_text(_SPECIES_NEWICK, encoding="utf-8")
    genes = []
    for index, newick in enumerate(_GENE_NEWICKS):
        path = tmp_path / f"family{index}.nwk"
        path.write_text(newick, encoding="utf-8")
        genes.append(path)
    return species, genes


def _solver_options(arm):
    """``arm`` is "exact" or "iterated"; every other field comes from ``_SHARED_SOLVER``."""
    if arm == "exact":
        forward, adjoint = "exact", "exact"
    elif arm == "iterated":
        forward, adjoint = "log", "series"
    else:
        raise ValueError(f"unknown arm {arm!r}")
    options = SolverOptions(
        **{
            **_SHARED_SOLVER,
            "pi_iters": _ITERATED_STEPS,
            "neumann_terms": _ITERATED_STEPS,
            # 0.0 switches the Neumann early exit off, so the reference really runs every term.
            "neumann_term_tol": 0.0,
            "forward_self_loop": forward,
            "adjoint_self_loop": adjoint,
        }
    )
    options.validate()
    return options


def _build(tmp_path, arm, mode, fraction_missing):
    species, genes = _write_fixture(tmp_path)
    return GeneReconModel(
        str(species),
        [str(path) for path in genes],
        mode=mode,
        device="cuda",
        dtype=torch.float64,
        solver_options=_solver_options(arm),
        family_chunk_size=len(genes),
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=64,
        fraction_missing=fraction_missing,
    )


def _weights(species_count, receiver_scale, origination_scale):
    """Centered random logits, identical for both arms because the generator seed is fixed.

    Centered because both vectors enter through a softmax over the S species, which ignores a
    constant shift: removing that null direction leaves exactly the gradients an optimizer sees.
    """
    generator = torch.Generator(device="cpu").manual_seed(20260904)
    receiver = torch.randn(species_count, generator=generator, dtype=torch.float64)
    origination = torch.randn(species_count, generator=generator, dtype=torch.float64)
    receiver = receiver_scale * (receiver - receiver.mean())
    origination = origination_scale * (origination - origination.mean())
    return receiver.cuda(), origination.cuda()


def _theta(model, log2_rate):
    return torch.full(tuple(model.theta.shape), float(log2_rate), device="cuda",
                      dtype=torch.float64)


def _value_and_grads(model, theta, receiver, origination):
    """per-family NLL, d/dtheta, d/dreceiver_weights, d/dorigination_weights (genewise mode)."""
    from gpurec.api._execution import stream_genewise_loss_vector_grad

    loss, grad_theta, grad_receiver, grad_origination = stream_genewise_loss_vector_grad(
        model.batch_statics,
        theta,
        receiver,
        origination,
        need_grad=True,
        need_receiver_grad=True,
        update_warm_starts=False,
        need_origination_grad=True,
    )
    torch.cuda.synchronize()
    return {
        "nll": loss.detach().clone(),
        "grad_theta": grad_theta.detach().clone(),
        "grad_receiver": grad_receiver.detach().clone(),
        "grad_origination": grad_origination.detach().clone(),
    }


def _relative_gap(reference, candidate, label):
    """max |candidate - reference| divided by the reference's own largest entry (or 1 if it is 0).

    The measured number is printed as well as returned: a test that only asserts "below the
    tolerance" cannot tell a healthy 1e-14 from a 9e-10 that is one change away from failing, and
    ``pytest -s`` on this file is the cheapest way to see the real margin.
    """
    assert torch.isfinite(candidate).all(), f"{label}: the candidate produced a non-finite value"
    assert torch.isfinite(reference).all(), f"{label}: the reference produced a non-finite value"
    scale = float(reference.abs().amax())
    absolute = float((candidate - reference).abs().max())
    gap = absolute / (scale if scale > 0.0 else 1.0)
    print(f"    {label:<40} max|diff| = {absolute:.4e}  relative = {gap:.4e}  "
          f"(max|reference| = {scale:.4e})")
    return gap


# The three weight settings. "uniform" is the control: it is the configuration the scale runs already
# validated, so if it ever fails the problem is not about weighting at all.
_SETTINGS = [
    pytest.param(0.0, 0.0, None, id="uniform"),
    pytest.param(0.6, 0.5, None, id="weighted"),
    pytest.param(0.6, 0.5, _FRACTION_MISSING, id="weighted_and_missing"),
]


@pytest.mark.parametrize("receiver_scale,origination_scale,fraction_missing", _SETTINGS)
def test_genewise_value_and_gradients_match(tmp_path, receiver_scale, origination_scale,
                                            fraction_missing):
    """NLL and all three gradients agree between the exact solves and the converged iteration."""
    _require_gpu()
    results = {}
    for arm in ("iterated", "exact"):
        model = _build(tmp_path, arm, "genewise", fraction_missing)
        species_count = int(model.species_helpers["S"])
        receiver, origination = _weights(species_count, receiver_scale, origination_scale)
        theta = _theta(model, math.log2(0.1))
        results[arm] = _value_and_grads(model, theta, receiver, origination)
        del model
        torch.cuda.empty_cache()

    for name in ("nll", "grad_theta", "grad_receiver", "grad_origination"):
        gap = _relative_gap(results["iterated"][name], results["exact"][name], f"genewise {name}")
        assert gap < _TOLERANCE, (
            f"{name}: exact and converged-iterated differ by {gap:.3e} relative, "
            f"tolerance {_TOLERANCE:.0e}"
        )


@pytest.mark.parametrize("receiver_scale,origination_scale,fraction_missing", _SETTINGS)
def test_genewise_curvature_blocks_match(tmp_path, receiver_scale, origination_scale,
                                         fraction_missing):
    """The per-family 3x3 curvature the genewise fit's Newton step uses agrees between the arms.

    This is the check that covers the Hessian-probe TANGENT, which follows ``adjoint_self_loop`` --
    the exact tangent is selected by asking for the exact adjoint. ``_analytic_hessian`` reads the
    model's own ``receiver_weights`` buffer, so the random logits are written there rather than
    passed; it does not take origination weights at all, so both arms compute curvature under the
    uniform origination prior, which is what the production genewise fit does.
    """
    _require_gpu()
    from gpurec.fit.genewise_fit import _analytic_hessian

    results = {}
    for arm in ("iterated", "exact"):
        model = _build(tmp_path, arm, "genewise", fraction_missing)
        species_count = int(model.species_helpers["S"])
        receiver, _origination = _weights(species_count, receiver_scale, origination_scale)
        with torch.no_grad():
            model.receiver_weights.copy_(receiver)
        theta = _theta(model, math.log2(0.1))
        results[arm] = _analytic_hessian(model, theta, _TANGENT_ITERS, None, None).detach().clone()
        torch.cuda.synchronize()
        del model
        torch.cuda.empty_cache()

    gap = _relative_gap(results["iterated"], results["exact"], "genewise 3x3 curvature")
    assert gap < _TOLERANCE, (
        f"3x3 curvature blocks: exact and converged-iterated differ by {gap:.3e} relative, "
        f"tolerance {_TOLERANCE:.0e}"
    )


@pytest.mark.parametrize("receiver_scale,origination_scale,fraction_missing", _SETTINGS)
def test_specieswise_value_and_gradients_match(tmp_path, receiver_scale, origination_scale,
                                               fraction_missing):
    """The same check in specieswise mode, where theta is per-species instead of per-family.

    Specieswise is the mode the MAP + cross-validation recipe fits, and it reaches the solves through
    autograd on the summed loss rather than through the per-family streaming helper, so it exercises
    a different entry point into the same kernels.
    """
    _require_gpu()
    results = {}
    for arm in ("iterated", "exact"):
        model = _build(tmp_path, arm, "specieswise", fraction_missing)
        species_count = int(model.species_helpers["S"])
        receiver, origination = _weights(species_count, receiver_scale, origination_scale)
        theta = _theta(model, math.log2(0.1)).requires_grad_(True)
        receiver = receiver.clone().requires_grad_(True)
        origination = origination.clone().requires_grad_(True)
        loss = model(theta, receiver, origination)
        loss.backward()
        torch.cuda.synchronize()
        results[arm] = {
            "nll": loss.detach().clone().reshape(1),
            "grad_theta": theta.grad.detach().clone(),
            "grad_receiver": receiver.grad.detach().clone(),
            "grad_origination": origination.grad.detach().clone(),
        }
        del model
        torch.cuda.empty_cache()

    for name in ("nll", "grad_theta", "grad_receiver", "grad_origination"):
        gap = _relative_gap(results["iterated"][name], results["exact"][name], f"specieswise {name}")
        assert gap < _TOLERANCE, (
            f"specieswise {name}: exact and converged-iterated differ by {gap:.3e} relative, "
            f"tolerance {_TOLERANCE:.0e}"
        )


def test_joint_theta_receiver_origination_hessian_matches(tmp_path):
    """The full joint (theta, receiver, origination) second derivative agrees between the arms.

    This is the sharpest of the checks. ``build_joint_hvp`` refuses to run at uniform weights -- at
    all-zero logits the weighted branches of the tangent are dead code -- so it can only be exercised
    with the weights on, which is precisely the situation that was under-tested. The whole matrix is
    built one column at a time from unit directions, so the comparison covers every block, including
    the receiver-weight block whose exact-tangent term was the one found missing at internal species
    nodes.

    The matrix is (3S + 2S) by (3S + 2S): three rate parameters per species, plus one receiver logit
    and one origination logit per species.
    """
    _require_gpu()
    from gpurec.solver.curvature.origination import build_joint_hvp

    columns = {}
    for arm in ("iterated", "exact"):
        model = _build(tmp_path, arm, "specieswise", _FRACTION_MISSING)
        species_count = int(model.species_helpers["S"])
        receiver, origination = _weights(species_count, 0.6, 0.5)
        theta = _theta(model, math.log2(0.1))
        size = theta.numel() + 2 * species_count
        identity = torch.eye(size, dtype=torch.float64, device="cuda")
        hvp, _loss, _solve, _cache = build_joint_hvp(
            model.batch_statics[0], theta, receiver, origination,
            tangent_self_iters=_TANGENT_ITERS,
        )
        columns[arm] = torch.stack([hvp(identity[i]) for i in range(size)], dim=1)
        torch.cuda.synchronize()
        del model, hvp
        torch.cuda.empty_cache()

    gap = _relative_gap(columns["iterated"], columns["exact"],
                        "joint (theta, receiver, origination) Hessian")
    assert gap < _TOLERANCE, (
        f"joint (theta, receiver, origination) Hessian: exact and converged-iterated differ by "
        f"{gap:.3e} relative, tolerance {_TOLERANCE:.0e}"
    )


def test_weighted_setting_actually_changes_the_answer(tmp_path):
    """Guard against the whole file silently testing nothing.

    If the weights were dropped on the floor somewhere -- ignored by the model, or zeroed by a
    reshape -- every comparison above would still pass, because both arms would be computing the
    same uniform answer. This asserts the weighted and uniform settings genuinely disagree, so a
    passing equivalence test means the weighted path really ran.
    """
    _require_gpu()
    values = {}
    for label, receiver_scale, origination_scale, fraction_missing in (
        ("uniform", 0.0, 0.0, None),
        ("weighted", 0.6, 0.5, None),
        ("weighted_and_missing", 0.6, 0.5, _FRACTION_MISSING),
    ):
        model = _build(tmp_path, "exact", "genewise", fraction_missing)
        species_count = int(model.species_helpers["S"])
        receiver, origination = _weights(species_count, receiver_scale, origination_scale)
        theta = _theta(model, math.log2(0.1))
        values[label] = _value_and_grads(model, theta, receiver, origination)
        del model
        torch.cuda.empty_cache()

    for label in ("weighted", "weighted_and_missing"):
        gap = _relative_gap(values["uniform"]["nll"], values[label]["nll"], f"{label} NLL vs uniform")
        assert gap > 1e-4, (
            f"{label} gave essentially the same NLL as uniform (relative gap {gap:.3e}); "
            "the weights are not reaching the likelihood, so the equivalence tests above are vacuous"
        )
    receiver_gap = _relative_gap(
        values["uniform"]["grad_receiver"], values["weighted"]["grad_receiver"],
        "weighted receiver gradient vs uniform",
    )
    assert receiver_gap > 1e-4, (
        f"the receiver-weight gradient barely moved when the receiver logits were switched on "
        f"(relative gap {receiver_gap:.3e})"
    )
