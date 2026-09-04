"""Every fit/reconcile entry point must give the same answer under the exact self-loop solves
(the library defaults) as under the iterated ones.

Inside each wave, a clade row's likelihood depends on itself through "transfer out, then the
donor lineage is lost" and "duplicate, then one copy is lost". Solving that per-row equation is
the SELF-LOOP. ``SolverOptions.forward_self_loop`` and ``adjoint_self_loop`` now default to
``"exact"``, which eliminates the fixed point outright instead of iterating it; the iterated
paths (``"log"`` forward, ``"series"`` adjoint) remain selectable and are the reference.

The exact solve returns what the iteration converges to, so in float64 the two must agree to
round-off. In float32 they are two different roundings of the same quantity, so the tolerance
here is a frame around the float32 noise, not a claim of equality.

The genewise fit had these seams checked while it was being made fast; these entry points did
not, which is what this file covers: the global fit, the specieswise MAP fit, the MAP+CV
driver, the reconcile CLI (including its agreement with the AleRax reference files) and the
backtracking sampler.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data" / "alerax"

# The two self-loop configurations under test. "exact" is what SolverOptions() now gives you.
EXACT = dict(forward_self_loop="exact", adjoint_self_loop="exact")
ITERATED = dict(forward_self_loop="log", adjoint_self_loop="series")

# float64: the exact solve and the converged iteration are the same number, so only round-off
# separates them. Measured on these fixtures: 0.0 for reconcile, 1.4e-14 nats for fit_global.
TOL_FP64 = 1e-9
# float32: two roundings of that number. Measured worst case 2.8e-5 nats on a -6221 nat
# likelihood (relative 4.6e-9) and 1.2e-3 nats on a 12442 nat fit (relative 9.2e-8).
TOL_FP32 = 1e-5


def _close(a, b, rel):
    """Relative-to-magnitude comparison, with an absolute floor of ``rel`` for small values."""
    return abs(a - b) <= rel * max(1.0, abs(a))


@pytest.fixture(scope="module")
def multi_family(tmp_path_factory):
    """A small MULTI-family dataset: the fits need more than one family to be meaningful.

    ``test_trees_1`` and ``test_trees_2`` are shipped as separate single-family fixtures but
    share the SAME 8-tip species tree with different gene trees, so copying both of them three
    times gives six families with two distinct topologies. Returns ``(species_path, [gene...])``.
    """
    dest = tmp_path_factory.mktemp("multi_family")
    species = dest / "sp.nwk"
    shutil.copy(DATA / "test_trees_1" / "sp.nwk", species)
    genes = []
    for copy in range(3):
        for src in ("test_trees_1", "test_trees_2"):
            gene = dest / f"fam_{src}_{copy}.nwk"
            shutil.copy(DATA / src / "g.nwk", gene)
            genes.append(str(gene))
    return str(species), genes


# --------------------------------------------------------------------------------------
# reconcile: per-family log-likelihoods at fixed rates
# --------------------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("fixture", ["test_trees_1", "test_trees_200", "test_mixed_200"])
@pytest.mark.parametrize("dtype_name,tol", [("float64", TOL_FP64), ("float32", TOL_FP32)])
def test_reconcile_exact_matches_iterated(fixture, dtype_name, tol):
    """The reconcile path (fixed rates, forward only) must not depend on the self-loop mode."""
    import torch
    from gpurec.api.solver_options import SolverOptions
    from gpurec.bench.alerax_io import parse_alerax_parameters, global_rates
    from gpurec.bench.fidelity import reconcile_at_alerax_rates

    dtype = getattr(torch, dtype_name)
    rates = global_rates(parse_alerax_parameters(DATA / fixture / "ref" / "model_parameters.txt"))
    got = {}
    for tag, knobs in (("exact", EXACT), ("iterated", ITERATED)):
        out = reconcile_at_alerax_rates(
            str(DATA / fixture / "sp.nwk"), [str(DATA / fixture / "g.nwk")], rates,
            device="cuda", dtype=dtype, mode="global",
            solver_options=SolverOptions(**knobs))
        got[tag] = next(iter(out.values()))
    assert _close(got["exact"], got["iterated"], tol), (
        f"{fixture} {dtype_name}: exact {got['exact']!r} vs iterated {got['iterated']!r} "
        f"differ by {abs(got['exact'] - got['iterated']):.3e} nats")


@pytest.mark.gpu
@pytest.mark.parametrize("fixture", ["test_trees_1", "test_trees_200", "test_mixed_200"])
def test_reconcile_exact_matches_alerax_reference_fp64(fixture):
    """With the exact solves as defaults, float64 reconcile still reproduces AleRax.

    tests/test_fidelity_alerax.py already asserts this at the library default; this repeats it
    with the solver EXPLICITLY set to exact, so the fidelity guarantee is pinned to the solver
    rather than to whatever the default happens to be.
    """
    import torch
    from gpurec.api.solver_options import SolverOptions
    from gpurec.bench.alerax_io import (parse_alerax_likelihoods, parse_alerax_parameters,
                                        global_rates)
    from gpurec.bench.fidelity import reconcile_at_alerax_rates

    ref = DATA / fixture / "ref"
    rates = global_rates(parse_alerax_parameters(ref / "model_parameters.txt"))
    alerax = next(iter(parse_alerax_likelihoods(ref / "per_fam_likelihoods.txt").values()))
    out = reconcile_at_alerax_rates(
        str(DATA / fixture / "sp.nwk"), [str(DATA / fixture / "g.nwk")], rates,
        device="cuda", dtype=torch.float64, mode="global",
        solver_options=SolverOptions(**EXACT))
    got = next(iter(out.values()))
    # Observed <= 9.1e-13 nats on these fixtures; 1e-8 is the same gate
    # test_fidelity_alerax.py's float64 test uses.
    assert abs(got - alerax) <= 1e-8, (
        f"{fixture}: exact-solver float64 logL {got!r} vs AleRax {alerax!r} "
        f"differ by {abs(got - alerax):.3e} nats")


# --------------------------------------------------------------------------------------
# gpurec fit --mode global
# --------------------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name,tol", [("float64", TOL_FP64), ("float32", TOL_FP32)])
def test_fit_global_exact_matches_iterated(multi_family, dtype_name, tol):
    """fit_global's fitted rates and NLL must not depend on the self-loop mode.

    This also covers the tier collapse: under the exact forward, fit_global skips the accurate
    eval-tier rebuild because that tier would recompute an identical forward. If the collapse
    were wrong, the exact arm's NLL would drift from the iterated arm's.
    """
    import torch
    from gpurec.fit.global_fit import fit_global

    species, genes = multi_family
    dtype = getattr(torch, dtype_name)
    got = {}
    for tag, knobs in (("exact", EXACT), ("iterated", ITERATED)):
        got[tag] = fit_global(species, genes, device="cuda", dtype=dtype, init_rate=0.1,
                              solver_options=dict(knobs), verbose=False)
    a, b = got["exact"], got["iterated"]
    assert _close(a["nll_nats"], b["nll_nats"], tol), (
        f"{dtype_name}: NLL {a['nll_nats']!r} vs {b['nll_nats']!r} "
        f"(difference {abs(a['nll_nats'] - b['nll_nats']):.3e} nats)")
    assert math.isfinite(a["gnorm"]) and a["n_steps"] > 0


@pytest.mark.gpu
def test_fit_global_uses_the_analytic_hessian_not_finite_differences():
    """fit_global builds its 3x3 curvature by summing the per-family analytic blocks.

    theta is SHARED, so NLL(theta) = sum_f NLL_f(theta) and the curvature is the plain sum of
    the per-family 3x3 blocks -- families are independent, so there are no cross-family terms.
    This pins that identity: the summed analytic blocks must equal a finite-difference Hessian
    of the aggregate gradient, to within the finite difference's OWN truncation error, which is
    first order in the step. So halving the step must roughly halve the gap; if the analytic sum
    were a different matrix the gap would instead stall at that matrix's offset.
    """
    import torch
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _analytic_hessian

    species = str(DATA / "test_trees_1" / "sp.nwk")
    genes = [str(DATA / "test_trees_1" / "g.nwk"), str(DATA / "test_trees_2" / "g.nwk")]
    model = GeneReconModel(species, genes, mode="genewise", device="cuda",
                           dtype=torch.float64,
                           solver_options=SolverOptions(pi_iters=16, neumann_terms=16, **EXACT))
    n_fam = model.theta.shape[0]
    theta3 = torch.full((3,), -3.0, device="cuda", dtype=model.dtype)

    def aggregate_grad(vec):
        theta_all = vec.detach().reshape(1, 3).expand(n_fam, 3).contiguous()
        _loss, per_family, _ = model.genewise_loss_vector_and_grad(theta=theta_all, need_grad=True)
        return per_family.sum(0).reshape(3).to(model.dtype)

    theta_all = theta3.reshape(1, 3).expand(n_fam, 3).contiguous()
    analytic = _analytic_hessian(model, theta_all, 16, species, genes).sum(0)
    assert torch.isfinite(analytic).all()
    # symmetric, as a Hessian must be
    assert float((analytic - analytic.T).abs().max()) <= 1e-9 * float(analytic.abs().max())

    base = aggregate_grad(theta3)
    gaps = []
    for eps in (1e-2, 1e-4):
        fd = torch.zeros(3, 3, device="cuda", dtype=model.dtype)
        for j in range(3):
            bumped = theta3.clone()
            bumped[j] += eps
            fd[:, j] = (aggregate_grad(bumped) - base) / eps
        fd = 0.5 * (fd + fd.T)
        gaps.append(float((fd - analytic).abs().max()))
    # 100x smaller step must shrink the gap by roughly 100x (measured 1.79e-2 -> 1.79e-4).
    assert gaps[1] < gaps[0] / 30.0, (
        f"gap did not shrink with the finite-difference step ({gaps[0]:.3e} -> {gaps[1]:.3e}): "
        f"the analytic sum is not the matrix the finite difference converges to")


# --------------------------------------------------------------------------------------
# specieswise MAP fit and MAP+CV
# --------------------------------------------------------------------------------------

@pytest.mark.gpu
def test_fit_specieswise_exact_matches_iterated(multi_family):
    """The single specieswise MAP fit must not depend on the self-loop mode.

    Note fit_specieswise casts theta to float32 internally and reports its NLL through the
    fp64 ``final_eval``, so there is no separate float64 arm to run here.
    """
    import torch
    from gpurec import GeneReconModel, SolverOptions
    from gpurec.fit.specieswise_fit import fit_specieswise

    species, genes = multi_family
    got = {}
    for tag, knobs in (("exact", EXACT), ("iterated", ITERATED)):
        model = GeneReconModel(species, genes, mode="specieswise", device="cuda",
                               dtype=torch.float32,
                               solver_options=SolverOptions(pi_iters=64, neumann_terms=64, **knobs))
        n_species = int(model.species_helpers["S"])
        weights = model.receiver_weights.detach().clone()
        theta_ref = torch.full((n_species, 3), math.log2(0.1), device="cuda", dtype=torch.float32)
        got[tag] = fit_specieswise(model.batch_statics, theta_ref.clone(), weights, lam=10.0,
                                   theta_ref=theta_ref, adam_steps=5, max_newton=5, verbose=False)
        del model
        torch.cuda.empty_cache()
    a, b = got["exact"], got["iterated"]
    assert _close(a["nll_nats"], b["nll_nats"], TOL_FP32), (
        f"MAP NLL {a['nll_nats']!r} vs {b['nll_nats']!r} "
        f"(difference {abs(a['nll_nats'] - b['nll_nats']):.3e} nats)")
    assert float((a["theta"] - b["theta"]).abs().max()) <= 1e-3


@pytest.mark.gpu
def test_map_cv_parse_once_matches_re_parsing(multi_family):
    """map_cv rebuilds each fold/lambda model from ONE parse of the dataset.

    Before, every one of the 1 + 2k models re-read and re-parsed its gene-tree files. The
    subset rebuild must give the same models: this compares the batch-static tensors built both
    ways element by element (they must be bit-identical), then checks the CV curve itself.
    """
    import torch
    from gpurec import GeneReconModel
    from gpurec.core.scheduling.batching import parse_families
    from gpurec.fit import map_cv as map_cv_module

    species, genes = multi_family
    subset = [0, 2, 3, 5]
    parsed = parse_families(species, genes)
    from_parsed = map_cv_module._build(species, parsed, subset, genes, mode="specieswise",
                                       device="cuda", solver_options=None)
    from_files = GeneReconModel(species, [genes[i] for i in subset], mode="specieswise",
                                device="cuda", solver_options=None)

    def tensors_of(static):
        """Every tensor a batch static carries: the direct attributes plus the ones inside its
        ``species_helpers`` and ``wave_layout`` dicts (that is where most of them live)."""
        found = {}
        for name in dir(static):
            if name.startswith("_"):
                continue
            value = getattr(static, name, None)
            if torch.is_tensor(value):
                found[name] = value
            elif isinstance(value, dict):
                for key, inner in value.items():
                    if torch.is_tensor(inner):
                        found[f"{name}[{key}]"] = inner
        return found

    n_compared = 0
    for a_static, b_static in zip(from_parsed.batch_statics, from_files.batch_statics):
        a_tensors, b_tensors = tensors_of(a_static), tensors_of(b_static)
        assert set(a_tensors) == set(b_tensors)
        for name, a_val in a_tensors.items():
            b_val = b_tensors[name]
            n_compared += 1
            assert a_val.shape == b_val.shape and torch.equal(a_val, b_val), (
                f"batch static {name!r} differs between the parse-once and re-parse builds")
    assert n_compared >= 10, (
        f"only {n_compared} batch-static tensors were compared -- the check is too thin")
    assert len(from_parsed.families) == len(subset)


@pytest.mark.gpu
def test_map_cv_runs_end_to_end_under_both_self_loops(multi_family):
    """The whole k-fold + lambda-homotopy driver runs and picks the same lambda either way."""
    from gpurec.fit.map_cv import _CV_SO, map_cv

    species, genes = multi_family
    got = {}
    for tag, knobs in (("exact", EXACT), ("iterated", ITERATED)):
        solver = dict(_CV_SO)
        solver.update(knobs)
        got[tag] = map_cv(species, genes, k=3, lambdas=(0.0, 10.0), device="cuda",
                          adam_steps=3, max_newton=3, seed=0, verbose=False,
                          solver_options=solver)
    a, b = got["exact"], got["iterated"]
    assert a["lam_star"] == b["lam_star"]
    for lam in a["cv"]:
        assert math.isfinite(a["cv"][lam])
        assert _close(a["cv"][lam], b["cv"][lam], TOL_FP32), (
            f"held-out CV at lambda={lam}: {a['cv'][lam]!r} vs {b['cv'][lam]!r}")


# --------------------------------------------------------------------------------------
# backtracking / sample_reconciliations
# --------------------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_sample_reconciliations_identical_under_both_self_loops(tmp_path, dtype_name):
    """The sampler reads the forward's Pi/Pibar, which the exact solve returns converged.

    At a fixed seed the drawn reconciliation must therefore be the same one either way, and the
    root-species histogram over many seeds must match exactly.
    """
    import collections

    import torch
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.core.backtracking.input import sample_reconciliations

    species = tmp_path / "sp.nwk"
    species.write_text("((A:1,B:1)AB:1,C:1)Root;\n")
    gene = tmp_path / "g.nwk"
    # A duplicated, so duplication and transfer genuinely compete for the same clade and the
    # sampler has a real distribution to draw from rather than one forced reconciliation.
    gene.write_text("(((A_1:1,B_1:1)x:1,C_1:1)y:1,A_2:1)GeneRoot;\n")
    dtype = getattr(torch, dtype_name)

    draws = {}
    for tag, knobs in (("exact", EXACT), ("iterated", ITERATED)):
        model = GeneReconModel(str(species), [str(gene)], mode="global", device="cuda",
                               dtype=dtype,
                               solver_options=SolverOptions(pi_iters=64, neumann_terms=64, **knobs))
        triple = torch.tensor([math.log2(0.3)] * 3, dtype=model.theta.dtype,
                              device=model.theta.device)
        with torch.no_grad():
            model.theta.copy_(triple if model.theta.dim() == 1
                              else triple.expand_as(model.theta))
        histogram = collections.Counter()
        for seed in range(64):
            histogram[sample_reconciliations(model, family_index=0, seed=seed)[0][1]] += 1
        draws[tag] = (sample_reconciliations(model, family_index=0, seed=0), histogram)
        del model
        torch.cuda.empty_cache()

    (one_exact, hist_exact), (one_iterated, hist_iterated) = draws["exact"], draws["iterated"]
    assert one_exact == one_iterated, "the seed-0 reconciliation differs between the two solvers"
    assert hist_exact == hist_iterated, (
        f"root-species histograms differ: {dict(hist_exact)} vs {dict(hist_iterated)}")
    assert len(hist_exact) > 1, "fixture drew a single species -- the check is vacuous"


# --------------------------------------------------------------------------------------
# CLI plumbing (no GPU needed)
# --------------------------------------------------------------------------------------

def test_cli_self_loop_flags_reach_solver_options():
    """--forward-self-loop / --adjoint-self-loop must survive argument parsing into SolverOptions."""
    import argparse

    from gpurec.cli._common import add_common_args, make_solver_options

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    base = ["--species", "sp.nwk", "--gene", "g.nwk"]

    default = make_solver_options(parser.parse_args(base))
    assert (default.forward_self_loop, default.adjoint_self_loop) == ("exact", "exact")

    chosen = make_solver_options(parser.parse_args(
        base + ["--forward-self-loop", "log", "--adjoint-self-loop", "series",
                "--pi-iters", "32", "--neumann-terms", "8"]))
    assert chosen.forward_self_loop == "log"
    assert chosen.adjoint_self_loop == "series"
    assert (chosen.pi_iters, chosen.neumann_terms) == (32, 8)

    linear = make_solver_options(parser.parse_args(base + ["--forward-self-loop", "linear"]))
    assert linear.forward_self_loop == "linear"
    assert linear.adjoint_self_loop == "exact"   # untouched flag keeps the default


def test_cli_rejects_unknown_self_loop_mode():
    """A typo must fail at parse time rather than reaching the solver."""
    import argparse

    from gpurec.cli._common import add_common_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--species", "sp.nwk", "--gene", "g.nwk",
                           "--forward-self-loop", "neumann"])


def test_fit_cli_counts_self_loop_flags_as_a_solver_override():
    """`gpurec fit --forward-self-loop log` must not be silently dropped.

    cli/fit.py only forwards solver_options when it believes the user overrode something, so
    the new flags have to be part of that decision.
    """
    import inspect

    from gpurec.cli import fit as fit_cli

    source = inspect.getsource(fit_cli.run_fit)
    user_solver = source.split("user_solver = (", 1)[1].split(")", 1)[0]
    assert "forward_self_loop" in user_solver
    assert "adjoint_self_loop" in user_solver
