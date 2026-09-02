"""``RateBounds`` pins the global (1e-10, None) log2-rate box + the genewise (1e-6, 2.0) preset.

Before this module the global floor was copy-pasted as three identical ``min_rate=1e-10,
max_rate=None`` signature defaults across ``gpurec/optimization.py``
(``log2_rate_bounds``/``project_rate_gradient_``/``clamp_log_rate_``) plus ``gpurec/api/model.py``'s
theta init (``math.log2(1e-10)``), while ``fit_genewise``'s tighter box lived only as bare
``min_rate=1e-6, max_rate=2.0`` signature defaults (never surfaced in ``GENEWISE_REFERENCE``).
``RateBounds()`` is the global preset; ``RateBounds.genewise()`` is the genewise preset. No numeric
value changes anywhere -- see task-5 brief.
"""
import inspect
import math

import pytest
import torch

from gpurec.config.rates import RateBounds
from gpurec.optimization import clamp_log_rate_, log2_rate_bounds, project_rate_gradient_


def test_rate_bounds_defaults_pinned():
    b = RateBounds()
    assert b.min_rate == 1e-10
    assert b.max_rate is None
    assert b.init_rate == 1e-10
    assert b.bound_active_eps == 1e-6


def test_rate_bounds_genewise_preset():
    b = RateBounds.genewise()
    assert b.min_rate == 1e-6
    assert b.max_rate == 2.0
    # only min/max move for the genewise preset -- init_rate/bound_active_eps stay global
    assert b.init_rate == 1e-10
    assert b.bound_active_eps == 1e-6


def test_rate_bounds_importable_from_config_package():
    from gpurec.config import RateBounds as ReExported
    assert ReExported is RateBounds


def test_log2_rate_bounds_default_matches_global():
    lo, hi = log2_rate_bounds()
    assert lo == math.log2(1e-10)
    assert hi is None


def test_log2_rate_bounds_accepts_bounds_object():
    lo, hi = log2_rate_bounds(bounds=RateBounds.genewise())
    assert lo == math.log2(1e-6)
    assert hi == math.log2(2.0)


def test_log2_rate_bounds_kwargs_override_still_work():
    """The old min_rate/max_rate kwargs (positional or keyword) must keep working unchanged --
    several out-of-tree callers (experiments/sanderson_cv/*, gpurec/fit/genewise_fit.py) use them
    directly and must not break."""
    assert log2_rate_bounds(1e-6, 2.0) == (math.log2(1e-6), math.log2(2.0))
    assert log2_rate_bounds(min_rate=1e-6, max_rate=2.0) == (math.log2(1e-6), math.log2(2.0))


def test_log2_rate_bounds_kwarg_overrides_a_base_bounds_field():
    """An explicit min_rate/max_rate kwarg overrides just that field of `bounds`."""
    lo, hi = log2_rate_bounds(bounds=RateBounds.genewise(), max_rate=4.0)
    assert lo == math.log2(1e-6)
    assert hi == math.log2(4.0)


def test_project_rate_gradient_bounds_equivalent_to_kwargs():
    theta = torch.tensor([[math.log2(1e-6), math.log2(2.0), 0.0]], dtype=torch.float64)
    grad_kw = torch.tensor([[1.0, -1.0, 5.0]], dtype=torch.float64)
    grad_bounds = grad_kw.clone()
    project_rate_gradient_(theta, grad_kw, min_rate=1e-6, max_rate=2.0)
    project_rate_gradient_(theta, grad_bounds, bounds=RateBounds.genewise())
    assert torch.equal(grad_kw, grad_bounds)
    # at-lower-bound + grad>0 zeroed; at-upper-bound + grad<0 zeroed; interior untouched
    assert grad_kw.tolist() == [[0.0, 0.0, 5.0]]


def test_clamp_log_rate_bounds_equivalent_to_kwargs():
    theta_kw = torch.tensor([[math.log2(1e-10), math.log2(10.0), 0.0]], dtype=torch.float64)
    theta_bounds = theta_kw.clone()
    clamp_log_rate_(theta_kw, min_rate=1e-6, max_rate=2.0)
    clamp_log_rate_(theta_bounds, bounds=RateBounds.genewise())
    assert torch.equal(theta_kw, theta_bounds)
    assert theta_kw[0, 0].item() == pytest.approx(math.log2(1e-6))
    assert theta_kw[0, 1].item() == pytest.approx(math.log2(2.0))
    assert theta_kw[0, 2].item() == 0.0


def test_fit_genewise_signature_defaults_come_from_genewise_preset():
    """`fit_genewise`'s min_rate/max_rate signature defaults must equal ``RateBounds.genewise()``
    (the values move into the dataclass preset; ``GENEWISE_REFERENCE`` does not track them, see
    ``test_reference_defaults.py``)."""
    from gpurec.fit.genewise_fit import fit_genewise

    params = inspect.signature(fit_genewise).parameters
    preset = RateBounds.genewise()
    assert params["min_rate"].default == preset.min_rate
    assert params["max_rate"].default == preset.max_rate


@pytest.mark.gpu
def test_model_theta_init_uses_rate_bounds_init_rate():
    from gpurec import GeneReconModel

    d = "tests/data/alerax/test_trees_1"
    m = GeneReconModel(f"{d}/sp.nwk", [f"{d}/g.nwk"], mode="global", device="cuda")
    assert float(m.theta[0].detach()) == pytest.approx(math.log2(RateBounds().init_rate))
    assert float(m.theta[0].detach()) == pytest.approx(math.log2(1e-10))


@pytest.mark.gpu
def test_fit_genewise_still_uses_1e6_2p0_box():
    """End-to-end smoke test: the genewise fit recipe still floors/caps at (1e-6, 2.0) post-refactor,
    exercising the RateBounds-wired bound_active_eps certificate at genewise_fit.py:247,281."""
    from gpurec.fit.genewise_fit import fit_genewise

    d = "tests/data/alerax/test_trees_1"
    res = fit_genewise(f"{d}/sp.nwk", [f"{d}/g.nwk", f"{d}/g.nwk"], device="cuda",
                        adam_steps=1, pi_tiers=(16,), max_iter=2, check_every=1,
                        min_drop=1, hessian_refresh=5, verify_drop=False,
                        certify=True, certify_curvature=True)
    theta = res["theta"]
    lo, hi = math.log2(1e-6), math.log2(2.0)
    assert torch.isfinite(theta).all()
    assert float(theta.min()) >= lo - 1e-9
    assert float(theta.max()) <= hi + 1e-9
    assert "bound_active" in res
