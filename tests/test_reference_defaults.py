import inspect

from gpurec.config import GpurecConfig
from gpurec.config.rates import RateBounds
from gpurec.fit.genewise_fit import fit_genewise, GENEWISE_REFERENCE, _BASE_SOLVER
from gpurec.fit.optimize import optimize, OPTIMIZE_REFERENCE
from gpurec.fit.map_cv import map_cv, MAP_CV_REFERENCE, _CV_SO


def _check(fn, ref):
    params = inspect.signature(fn).parameters
    for k, v in ref.items():
        assert k in params, f"{fn.__name__} has no param {k}"
        assert params[k].default == v, f"{fn.__name__}.{k}: default {params[k].default!r} != ref {v!r}"


def test_genewise_reference_matches_signature():
    _check(fit_genewise, GENEWISE_REFERENCE)


def test_optimize_reference_matches_signature():
    _check(optimize, OPTIMIZE_REFERENCE)


def test_map_cv_reference_matches_signature():
    _check(map_cv, MAP_CV_REFERENCE)


# --- GpurecConfig factories: single-sourced recipe presets (task-10) ---


def test_genewise_reference_factory_reproduces_recipe():
    cfg = GpurecConfig.genewise_reference()
    solver = cfg.solver
    assert solver.e_max_iter == 128
    assert solver.e_tol == 1e-8
    # The E-adjoint linear solve is a Neumann series (the old BiCGSTAB knobs
    # bicgstab_max_iter / bicgstab_tol / bicgstab_breakdown_tol are gone); the recipe pins its
    # iteration budget and its own tighter-than-dtype-auto residual target.
    assert solver.e_adjoint_max_iter == 128
    assert solver.e_adjoint_tol == 1e-7
    assert solver.adjoint_pruning_threshold == 1e-6
    assert solver.use_adjoint_pruning is True
    assert solver.pibar_side_threshold == 0.0
    # The recipe solves each clade row's self-loop exactly (tree elimination) in both directions.
    assert solver.forward_self_loop == "exact"
    assert solver.adjoint_self_loop == "exact"
    assert cfg.rates == RateBounds.genewise()


def test_map_cv_reference_factory_reproduces_recipe():
    solver = GpurecConfig.map_cv_reference().solver
    assert solver.pi_iters == 64
    assert solver.neumann_terms == 64
    assert solver.e_max_iter == 128
    assert solver.e_tol == 1e-8
    assert solver.e_adjoint_max_iter == 128
    # None = dtype-relative auto residual target (fp32 -> 1e-6, fp64 -> 1e-12), the robust default.
    assert solver.e_adjoint_tol is None
    assert solver.adjoint_pruning_threshold == 1e-6
    assert solver.use_adjoint_pruning is True
    assert solver.pibar_side_threshold == 0.0


def test_optimize_reference_factory_is_config_defaults():
    # optimize.py has no separate solver dict -- GeneReconModel falls back to the
    # default SolverOptions, so the "reference" config is just GpurecConfig() defaults.
    assert GpurecConfig.optimize_reference() == GpurecConfig()


def test_genewise_base_solver_matches_factory():
    """Drift guard: fit_genewise's derived ``_BASE_SOLVER`` == the factory's solver fields."""
    ref_solver = GpurecConfig.genewise_reference().solver
    for k, v in _BASE_SOLVER.items():
        assert getattr(ref_solver, k) == v


def test_map_cv_cv_so_matches_factory():
    """Drift guard: map_cv's derived ``_CV_SO`` == the factory's solver fields."""
    ref_solver = GpurecConfig.map_cv_reference().solver
    for k, v in _CV_SO.items():
        assert getattr(ref_solver, k) == v


def test_solver_options_use_hvp_warm_start_defaults_true():
    from gpurec import SolverOptions
    so = SolverOptions()
    assert so.use_hvp_warm_start is True
