from __future__ import annotations

import pytest

from gpurec.core._helpers import _env_flag_enabled, _env_mode_enabled_required


@pytest.mark.parametrize(
    "token",
    ("", "0", "false", "no", "off", " FALSE ", " No "),
)
def test_env_flag_enabled_preserves_false_token_semantics(monkeypatch, token: str):
    monkeypatch.setenv("GPUREC_TEST_FLAG", token)

    assert not _env_flag_enabled("GPUREC_TEST_FLAG", "1")


@pytest.mark.parametrize("token", ("1", "true", "yes", "on", "auto", "enabled"))
def test_env_flag_enabled_treats_other_tokens_as_enabled(monkeypatch, token: str):
    monkeypatch.setenv("GPUREC_TEST_FLAG", token)

    assert _env_flag_enabled("GPUREC_TEST_FLAG", "0")


def test_env_flag_enabled_uses_default_for_unset_values(monkeypatch):
    monkeypatch.delenv("GPUREC_TEST_FLAG", raising=False)

    assert _env_flag_enabled("GPUREC_TEST_FLAG", "1")
    assert not _env_flag_enabled("GPUREC_TEST_FLAG", "0")


@pytest.mark.parametrize(
    ("token", "enabled", "required"),
    (
        ("auto", True, False),
        ("off", False, False),
        (" false ", False, False),
        ("force", True, True),
        ("required", True, True),
        ("1", True, True),
        ("yes", True, True),
    ),
)
def test_env_mode_enabled_required_preserves_cuda_toggle_semantics(
    monkeypatch,
    token: str,
    enabled: bool,
    required: bool,
):
    monkeypatch.setenv("GPUREC_TEST_MODE", token)

    mode, mode_enabled, mode_required = _env_mode_enabled_required(
        "GPUREC_TEST_MODE",
        "auto",
    )

    assert mode == token.strip().lower()
    assert mode_enabled is enabled
    assert mode_required is required


def test_env_mode_enabled_required_uses_auto_default(monkeypatch):
    monkeypatch.delenv("GPUREC_TEST_MODE", raising=False)

    assert _env_mode_enabled_required("GPUREC_TEST_MODE") == ("auto", True, False)
