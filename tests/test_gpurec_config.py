import dataclasses

import pytest

from gpurec.config import GpurecConfig
from gpurec.api.solver_options import SolverOptions
from gpurec.config.memory import MemoryOptions
from gpurec.config.newton import NewtonOptions
from gpurec.config.rates import RateBounds
from gpurec.solver.penalties import PenaltyOptions


def test_gpurec_config_builds_with_defaults():
    cfg = GpurecConfig()
    assert isinstance(cfg.solver, SolverOptions)
    assert isinstance(cfg.newton, NewtonOptions)
    assert isinstance(cfg.rates, RateBounds)
    assert isinstance(cfg.regularizer, PenaltyOptions)
    assert isinstance(cfg.memory, MemoryOptions)


def test_gpurec_config_default_equals_composed_defaults():
    cfg = GpurecConfig()
    assert cfg.solver == SolverOptions()
    assert cfg.newton == NewtonOptions()
    assert cfg.rates == RateBounds()
    assert cfg.regularizer == PenaltyOptions()
    assert cfg.memory == MemoryOptions()


def test_gpurec_config_validate_delegates():
    cfg = GpurecConfig()
    cfg.validate()  # must not raise

    bad = GpurecConfig(solver=SolverOptions(e_max_iter=0))
    with pytest.raises(ValueError):
        bad.validate()

    bad_newton = GpurecConfig(newton=NewtonOptions(max_newton=0))
    with pytest.raises(ValueError):
        bad_newton.validate()


def test_to_dict_is_nested_asdict():
    cfg = GpurecConfig()
    d = cfg.to_dict()
    assert d == dataclasses.asdict(cfg)
    assert isinstance(d["solver"], dict)
    assert d["solver"]["pi_iters"] == cfg.solver.pi_iters
    assert isinstance(d["regularizer"], dict)
    assert isinstance(d["regularizer"]["origination"], dict)


def test_from_dict_round_trips_default():
    cfg = GpurecConfig()
    assert GpurecConfig.from_dict(cfg.to_dict()) == cfg


def test_from_dict_round_trips_nondefault():
    cfg = GpurecConfig(solver=SolverOptions(pi_iters=128), newton=NewtonOptions(max_newton=7))
    assert GpurecConfig.from_dict(cfg.to_dict()) == cfg


def test_from_dict_unknown_top_level_key_raises():
    with pytest.raises(ValueError):
        GpurecConfig.from_dict({"not_a_real_field": 1})


def test_from_dict_unknown_nested_key_raises():
    with pytest.raises(ValueError):
        GpurecConfig.from_dict({"solver": {"not_a_real_field": 1}})


def test_from_dict_partial_dict_deep_merges_onto_defaults():
    cfg = GpurecConfig.from_dict({"solver": {"pi_iters": 128}})
    default = GpurecConfig()
    assert cfg.solver.pi_iters == 128
    # every other solver field stays at its default
    for f in dataclasses.fields(SolverOptions):
        if f.name == "pi_iters":
            continue
        assert getattr(cfg.solver, f.name) == getattr(default.solver, f.name)
    # every other top-level section is untouched
    assert cfg.newton == default.newton
    assert cfg.rates == default.rates
    assert cfg.regularizer == default.regularizer
    assert cfg.memory == default.memory


def test_from_dict_empty_dict_returns_defaults():
    assert GpurecConfig.from_dict({}) == GpurecConfig()


def test_gpurec_config_reexports_in_config_package():
    from gpurec.config import GpurecConfig as ReExported
    from gpurec.config import MemoryOptions as MemoryReExported
    from gpurec.config import NewtonOptions as NewtonReExported
    from gpurec.config import PenaltyOptions as PenaltyReExported
    from gpurec.config import RateBounds as RateReExported
    from gpurec.config import SolverOptions as SolverReExported

    assert ReExported is GpurecConfig
    assert SolverReExported is SolverOptions
    assert PenaltyReExported is PenaltyOptions
    assert NewtonReExported is NewtonOptions
    assert RateReExported is RateBounds
    assert MemoryReExported is MemoryOptions
