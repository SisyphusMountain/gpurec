"""Regression test: the ridge-Newton polish must run the FULL default recipe on MULTI-BATCH
datasets for both global and genewise modes.

Two multi-batch-unsafe accesses had to be guarded for this to work end-to-end:

1. `newton_cg.newton_lanczos` line search did `warm_E = sv_t["E"]` on the accepted step, but
   `forward_solve` (`gpurec/solver/value_and_grad.py`) returns `(loss, None)` for MULTIPLE batches
   (the exact-HVP saved-intermediates dict is a single-batch artifact; multi-batch streams and
   frees the ~GB scratch). That crashed with `TypeError: 'NoneType' object is not subscriptable`.
   Guarded: `warm_E = sv_t["E"] if sv_t is not None else None`.

2. For genewise (theta `(G,3)`, so not `is_global`) with the default `ridge=True`, `newton_polish`
   (`gpurec/fit/optimize.py`) called `_exact_ridge_lambda` -> `forward_solve` + `make_exact_hvp`,
   both SINGLE-BATCH-only (`make_exact_hvp` raises `NotImplementedError` for a batch list). Guarded
   the same way global already was: the exact-HVP ridge estimator is used only for a single-batch,
   non-global theta; otherwise `lam=0` and `newton_lanczos` self-damps its (multi-batch-safe)
   FD-Hessian descent.

Together these unblock global AND genewise multi-batch full recipes. Specieswise (the exact-HVP
`hvp_mode="exact"` path) is a separate, harder task and is intentionally NOT covered here.
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


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["global", "genewise"])
def test_newton_polish_multibatch_full_recipe(mode, tmp_path):
    # n_species=250 / n_families=500 with the default GeneReconModel family_chunk_size=300
    # guarantees > 1 batch, exercising the multi-batch `forward_solve` -> saved=None paths that
    # crashed the line search (sv_t) and the ridge-lambda estimator (make_exact_hvp).
    sp, genes = simulate_dataset(mode, tmp_path, n_species=250, n_families=500, dtl=0.05, seed=1)

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
