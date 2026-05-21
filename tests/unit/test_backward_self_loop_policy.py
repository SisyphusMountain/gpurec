import pytest
import torch

import gpurec.core.backward as backward


def _eligible_kwargs(**overrides):
    kwargs = {
        "dts_r": None,
        "nosplit_enabled": True,
        "split_enabled": True,
        "auto_wrapped": True,
        "dtype": torch.float32,
        "uniform_leaf_logp": torch.zeros(4, dtype=torch.float32),
        "S": 4,
        "compact_level_ptr": torch.tensor([0, 1], dtype=torch.int32),
        "compact_level_parents": torch.tensor([0], dtype=torch.int32),
        "compact_level_child1": torch.tensor([1], dtype=torch.int32),
        "compact_level_child2": torch.tensor([2], dtype=torch.int32),
    }
    kwargs.update(overrides)
    return kwargs


def test_cuda_self_loop_env_defaults_are_optional_for_both_backends(monkeypatch):
    monkeypatch.delenv("GPUREC_CUDA_SELF_LOOP_NOSPLIT", raising=False)
    monkeypatch.delenv("GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION", raising=False)
    monkeypatch.delenv("GPUREC_CUDA_SELF_LOOP_SPLIT", raising=False)

    assert backward._cuda_self_loop_options_from_env() == (
        "auto",
        True,
        False,
        "tree",
        "auto",
        True,
        False,
    )


@pytest.mark.parametrize(
    ("nosplit_token", "split_token", "expected"),
    (
        ("off", "off", ("off", False, False, "off", False, False)),
        ("enabled", "auto", ("enabled", True, False, "auto", True, False)),
        ("force", "required", ("force", True, True, "required", True, True)),
        ("1", "yes", ("1", True, True, "yes", True, True)),
    ),
)
def test_cuda_self_loop_env_mode_parsing_matches_runtime_helper(
    monkeypatch,
    nosplit_token,
    split_token,
    expected,
):
    monkeypatch.setenv("GPUREC_CUDA_SELF_LOOP_NOSPLIT", nosplit_token)
    monkeypatch.setenv("GPUREC_CUDA_SELF_LOOP_NOSPLIT_CORRECTION", "legacy")
    monkeypatch.setenv("GPUREC_CUDA_SELF_LOOP_SPLIT", split_token)

    (
        nosplit_mode,
        nosplit_enabled,
        nosplit_required,
        nosplit_correction,
        split_mode,
        split_enabled,
        split_required,
    ) = backward._cuda_self_loop_options_from_env()

    assert (
        nosplit_mode,
        nosplit_enabled,
        nosplit_required,
        split_mode,
        split_enabled,
        split_required,
    ) == expected
    assert nosplit_correction == "legacy"


def test_cuda_self_loop_wave_routes_nosplit_when_dts_cross_is_absent():
    wave_enabled, use_cuda = backward._cuda_self_loop_wave_backend(
        **_eligible_kwargs(dts_r=None, nosplit_enabled=True, split_enabled=False)
    )

    assert wave_enabled is True
    assert use_cuda is True


def test_cuda_self_loop_wave_routes_split_when_dts_cross_is_present():
    wave_enabled, use_cuda = backward._cuda_self_loop_wave_backend(
        **_eligible_kwargs(
            dts_r=torch.zeros(2, 4),
            nosplit_enabled=False,
            split_enabled=True,
        )
    )

    assert wave_enabled is True
    assert use_cuda is True


@pytest.mark.parametrize(
    "overrides",
    (
        {"auto_wrapped": False},
        {"dtype": torch.float64},
        {"uniform_leaf_logp": torch.zeros(3, dtype=torch.float32)},
        {"uniform_leaf_logp": None},
        {"compact_level_ptr": None},
        {"compact_level_parents": None},
        {"compact_level_child1": None},
        {"compact_level_child2": None},
    ),
)
def test_cuda_self_loop_required_mode_does_not_bypass_eligibility_gates(overrides):
    wave_enabled, use_cuda = backward._cuda_self_loop_wave_backend(
        **_eligible_kwargs(**overrides)
    )

    assert wave_enabled is True
    assert use_cuda is False


def test_cuda_self_loop_disabled_mode_keeps_wave_on_retained_fused_path():
    wave_enabled, use_cuda = backward._cuda_self_loop_wave_backend(
        **_eligible_kwargs(nosplit_enabled=False, split_enabled=False)
    )

    assert wave_enabled is False
    assert use_cuda is False


@pytest.mark.parametrize(
    ("dts_r", "expected"),
    (
        (None, (False, False, True)),
        (torch.zeros(1), (False, True, False)),
    ),
)
def test_optional_cuda_self_loop_failure_disables_only_failed_backend(
    dts_r,
    expected,
):
    assert (
        backward._cuda_self_loop_fallback_after_optional_failure(
            dts_r=dts_r,
            nosplit_enabled=True,
            split_enabled=True,
            nosplit_required=False,
            split_required=False,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("dts_r", "nosplit_required", "split_required"),
    (
        (None, True, False),
        (torch.zeros(1), False, True),
    ),
)
def test_required_cuda_self_loop_failure_re_raises_without_disabling_backends(
    dts_r,
    nosplit_required,
    split_required,
):
    assert backward._cuda_self_loop_fallback_after_optional_failure(
        dts_r=dts_r,
        nosplit_enabled=True,
        split_enabled=True,
        nosplit_required=nosplit_required,
        split_required=split_required,
    ) == (True, True, True)


def test_optional_cuda_self_loop_fallback_exception_surface_stays_narrow():
    assert backward._OPTIONAL_CUDA_SELF_LOOP_EXCEPTIONS == (
        ImportError,
        RuntimeError,
        ValueError,
    )
    assert Exception not in backward._OPTIONAL_CUDA_SELF_LOOP_EXCEPTIONS
