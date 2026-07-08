"""``PenaltyOptions`` groups the existing regularizer config -- ``OriginationPenalty``, the
module-level ``tv_eps``, and the ridge homotopy defaults scattered across ``gpurec/fit/map_cv.py``
/ ``gpurec/fit/map_fit.py`` -- into one facade dataclass. Purely additive: no existing penalty
call site is rewired here (see task-7 brief). ``OriginationPenalty``/``DEFAULT_TV_EPS`` stay
importable and unchanged.
"""
from dataclasses import fields

from gpurec.solver.penalties import PenaltyOptions, OriginationPenalty, DEFAULT_TV_EPS


def test_penalty_options_defaults_pinned():
    opts = PenaltyOptions()
    assert isinstance(opts.origination, OriginationPenalty)
    assert opts.origination == OriginationPenalty()
    assert opts.tv_eps == DEFAULT_TV_EPS == 1e-3
    assert opts.lambdas == (0.0, 1.0, 10.0, 100.0, 1000.0)
    assert opts.lam_margin == 1.3
    assert opts.lam_floor == 1e-3


def test_penalty_options_origination_is_independent_instance():
    """Each PenaltyOptions() must build its own OriginationPenalty (default_factory), not share
    one mutable instance across constructions."""
    a = PenaltyOptions()
    b = PenaltyOptions()
    assert a.origination is not b.origination
    a.origination.l2 = 5.0
    assert b.origination.l2 == 0.0


def test_origination_penalty_still_importable_and_unchanged():
    """Back-compat: OriginationPenalty keeps its own field set/defaults, untouched by wrapping it
    in PenaltyOptions."""
    op = OriginationPenalty()
    names = {f.name for f in fields(op)}
    assert names == {
        "l2", "depth_lambda", "depth", "root_lambda", "root_index",
        "dirichlet_c", "barrier_kind", "dirichlet_pi", "floor",
    }
    assert op.l2 == 0.0
    assert op.depth_lambda == 0.0
    assert op.root_lambda == 0.0
    assert op.dirichlet_c == 0.0
    assert op.barrier_kind == "meanlog"
    assert op.any_active() is False
