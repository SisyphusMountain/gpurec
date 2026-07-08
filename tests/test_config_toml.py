"""Tests for ``gpurec/config/defaults.toml`` and the ``from_toml``/``load_config`` loaders.

``test_defaults_toml_matches_dataclass_defaults`` is the PERMANENT guard that
``gpurec/config/defaults.toml`` never drifts from the ``GpurecConfig`` dataclass
defaults (the source of truth). If a dataclass default changes, this test fails
until ``defaults.toml`` is updated to match (or vice versa).
"""
import dataclasses

import pytest

from gpurec.config import DEFAULTS_TOML_PATH, GpurecConfig, load_config


def test_defaults_toml_matches_dataclass_defaults():
    assert GpurecConfig.from_toml(DEFAULTS_TOML_PATH) == GpurecConfig()


def test_defaults_toml_path_is_package_relative_and_exists():
    # Package-relative (Path(__file__).parent-derived), not a cwd-relative string.
    assert DEFAULTS_TOML_PATH.is_absolute()
    assert DEFAULTS_TOML_PATH.name == "defaults.toml"
    assert DEFAULTS_TOML_PATH.exists()


def test_user_toml_overrides_merge():
    cfg = GpurecConfig.from_dict({"solver": {"pi_iters": 128}})
    assert cfg.solver.pi_iters == 128 and cfg.newton.max_cg == 40


def test_from_toml_partial_file_merges_onto_defaults(tmp_path):
    p = tmp_path / "user.toml"
    p.write_text("[solver]\npi_iters = 128\n")
    cfg = GpurecConfig.from_toml(p)
    default = GpurecConfig()

    assert cfg.solver.pi_iters == 128
    for f in dataclasses.fields(default.solver):
        if f.name == "pi_iters":
            continue
        assert getattr(cfg.solver, f.name) == getattr(default.solver, f.name)
    assert cfg.newton == default.newton
    assert cfg.rates == default.rates
    assert cfg.regularizer == default.regularizer
    assert cfg.memory == default.memory


def test_from_toml_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[solver]\nnot_a_real_field = 1\n")
    with pytest.raises(ValueError):
        GpurecConfig.from_toml(p)


def test_load_config_none_returns_defaults():
    assert load_config() == GpurecConfig()
    assert load_config(None) == GpurecConfig()


def test_load_config_with_path_loads_and_merges(tmp_path):
    p = tmp_path / "user.toml"
    p.write_text("[rates]\nmin_rate = 1e-8\n")
    cfg = load_config(p)
    default = GpurecConfig()

    assert cfg.rates.min_rate == 1e-8
    assert cfg.rates.max_rate == default.rates.max_rate  # untouched None default
    assert cfg.solver == default.solver
    assert cfg.newton == default.newton
    assert cfg.regularizer == default.regularizer
    assert cfg.memory == default.memory


def test_none_only_fields_are_omitted_from_defaults_toml_and_stay_none():
    cfg = GpurecConfig.from_toml(DEFAULTS_TOML_PATH)
    assert cfg.solver.bicgstab_tol is None
    assert cfg.solver.bicgstab_breakdown_tol is None
    assert cfg.rates.max_rate is None
    assert cfg.regularizer.origination.depth is None
    assert cfg.regularizer.origination.root_index is None
    assert cfg.regularizer.origination.dirichlet_pi is None


def test_lambdas_tuple_default_coerced_from_toml_array():
    cfg = GpurecConfig.from_toml(DEFAULTS_TOML_PATH)
    assert isinstance(cfg.regularizer.lambdas, tuple)
    assert cfg.regularizer.lambdas == (0.0, 1.0, 10.0, 100.0, 1000.0)
    # Equality against the dataclass default only holds if list -> tuple
    # coercion actually happened (TOML arrays decode to Python lists).
    assert cfg.regularizer == GpurecConfig().regularizer


def test_tuple_field_override_from_dict_list_is_coerced_to_tuple():
    cfg = GpurecConfig.from_dict({"regularizer": {"lambdas": [1.0, 2.0]}})
    assert cfg.regularizer.lambdas == (1.0, 2.0)
    assert isinstance(cfg.regularizer.lambdas, tuple)


def test_tuple_field_override_from_toml_file_is_coerced_to_tuple(tmp_path):
    p = tmp_path / "user.toml"
    p.write_text("[regularizer]\nlambdas = [1.0, 2.0, 3.0]\n")
    cfg = GpurecConfig.from_toml(p)
    assert cfg.regularizer.lambdas == (1.0, 2.0, 3.0)
    assert isinstance(cfg.regularizer.lambdas, tuple)
