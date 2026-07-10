"""Regression test: the ridge-Newton polish (`newton_cg.newton_lanczos`) must not crash on
multi-batch datasets during its Armijo line search.

Root cause (diagnosed): `forward_solve` (`gpurec/solver/value_and_grad.py`) returns
`(loss, None)` for MULTIPLE batches (the exact-HVP saved-intermediates dict is a single-batch-only
artifact; multi-batch streams and frees the ~GB scratch). The line search in `newton_lanczos`
unconditionally did `warm_E = sv_t["E"]` on the accepted step, which crashes with
`TypeError: 'NoneType' object is not subscriptable` as soon as a fit runs with more than one
batch. The guard (`warm_E = sv_t["E"] if sv_t is not None else None`) fixes that access.

`global` (theta `(3,)`) uses `hvp_mode="fd"` and does NOT touch the exact HVP, so once the line
search is guarded the full default recipe (Adam -> ridge-Newton polish) completes end-to-end at
multi-batch scale.

`genewise` and `specieswise` still route through the SINGLE-BATCH-only exact HVP: `newton_polish`
-> `_exact_ridge_lambda` (optimize.py) -> `make_exact_hvp` -> `hvp_exact._single_static`, which
raises `NotImplementedError` for >1 batch. That needs the streaming multi-batch exact-HVP task and
is marked xfail below (NOT fixed here).
"""
import math

import pytest

rustree = pytest.importorskip("rustree")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from gpurec.api.model import GeneReconModel
from gpurec.api.solver_options import SolverOptions
from gpurec.bench.simulate import simulate_dataset
from gpurec.fit.optimize import optimize, final_eval


def _run_multibatch_full_recipe(mode, out_dir):
    # n_species=250 / n_families=500 with the default GeneReconModel family_chunk_size=300
    # guarantees > 1 batch, exercising the multi-batch `forward_solve` -> `sv_t is None` path that
    # crashed the line search.
    sp, genes = simulate_dataset(mode, out_dir, n_species=250, n_families=500, dtl=0.05, seed=1)

    model = GeneReconModel(
        sp, genes, mode=mode, device="cuda", dtype=torch.float32,
        solver_options=SolverOptions(e_adjoint_solver="neumann"),
    )
    assert len(model.batch_statics) > 1, "fixture is not genuinely multi-batch"

    theta_hat, hist = optimize(
        model.batch_statics, model.theta.detach(), model.receiver_weights.detach(),
        verbose=False,
    )  # DEFAULT recipe: Adam first-order stage -> ridge-Newton polish

    nll, gnorm = final_eval(model.batch_statics, theta_hat, model.receiver_weights.detach())

    assert math.isfinite(nll), f"{mode}: non-finite nll {nll}"
    assert math.isfinite(gnorm), f"{mode}: non-finite gnorm {gnorm}"
    rates = 2.0 ** theta_hat.detach()
    assert torch.isfinite(rates).all(), f"{mode}: non-finite rates {rates}"
    assert bool((rates > 0).all()), f"{mode}: non-positive rates {rates}"


@pytest.mark.gpu
@pytest.mark.slow
def test_newton_polish_multibatch_full_recipe_global(tmp_path):
    """global mode: hvp_mode='fd', no exact HVP -> the guarded line search completes the full
    Adam + ridge-Newton recipe at multi-batch scale."""
    _run_multibatch_full_recipe("global", tmp_path)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.xfail(
    reason="multi-batch exact HVP not ported: newton_polish -> _exact_ridge_lambda -> "
           "make_exact_hvp is single-batch-only (see hvp_exact._single_static); "
           "pending the streaming multi-batch exact-HVP task",
    strict=False,
)
def test_newton_polish_multibatch_full_recipe_genewise(tmp_path):
    """genewise mode: the ridge-lambda estimator invokes the single-batch-only exact HVP, which
    raises NotImplementedError for >1 batch. xfail until the streaming multi-batch exact HVP lands."""
    _run_multibatch_full_recipe("genewise", tmp_path)
