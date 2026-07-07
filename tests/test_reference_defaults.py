import inspect
from gpurec.fit.genewise_fit import fit_genewise, GENEWISE_REFERENCE
from gpurec.fit.optimize import optimize, OPTIMIZE_REFERENCE
from gpurec.fit.map_cv import map_cv, MAP_CV_REFERENCE


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
