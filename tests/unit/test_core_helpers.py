from __future__ import annotations

from types import SimpleNamespace

import pytest

import gpurec.core.batch_planning as batch_planning
import gpurec.core._helpers as helpers
from gpurec.core.kernels import wave_backward
from gpurec.core._helpers import (
    _env_flag_enabled,
    _env_mode_enabled_required,
    _nvtx_range,
)


def test_batch_planning_exports_supported_family_chunk_size_helper():
    namespace: dict[str, object] = {}
    exec("from gpurec.core.batch_planning import *", {}, namespace)

    assert "normalize_family_chunk_size" in batch_planning.__all__
    assert namespace["normalize_family_chunk_size"] is (
        batch_planning.normalize_family_chunk_size
    )


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
        ("enabled", True, False),
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


@pytest.mark.parametrize("token", ("", "0", "false", "no", "off", " FALSE "))
def test_cuda_pibar_from_ud_strict_false_tokens_do_not_require(
    monkeypatch,
    token: str,
):
    monkeypatch.setenv("GPUREC_CUDA_PIBAR_FROM_UD", "auto")
    monkeypatch.setenv("GPUREC_CUDA_PIBAR_FROM_UD_STRICT", token)

    assert wave_backward._cuda_pibar_from_ud_options() == ("auto", True, False)


@pytest.mark.parametrize("token", ("1", "true", "yes", "on", "required", "enabled"))
def test_cuda_pibar_from_ud_strict_truthy_tokens_require(monkeypatch, token: str):
    monkeypatch.setenv("GPUREC_CUDA_PIBAR_FROM_UD", "auto")
    monkeypatch.setenv("GPUREC_CUDA_PIBAR_FROM_UD_STRICT", token)

    assert wave_backward._cuda_pibar_from_ud_options() == ("auto", True, True)


def test_cuda_pibar_from_ud_disabled_mode_stays_not_required(monkeypatch):
    monkeypatch.setenv("GPUREC_CUDA_PIBAR_FROM_UD", "off")
    monkeypatch.setenv("GPUREC_CUDA_PIBAR_FROM_UD_STRICT", "1")

    assert wave_backward._cuda_pibar_from_ud_options() == ("off", False, False)


def test_nvtx_range_uses_context_manager_when_available(monkeypatch):
    events: list[object] = []

    class FakeRange:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")

    class FakeNvtx:
        def range(self, name: str):
            events.append(("range", name))
            return FakeRange()

        def range_push(self, name: str):
            raise AssertionError("range_push should not be used")

    monkeypatch.setattr(
        helpers.torch,
        "cuda",
        SimpleNamespace(nvtx=FakeNvtx()),
        raising=False,
    )

    with _nvtx_range("forward"):
        events.append("body")

    assert events == [("range", "forward"), "enter", "body", "exit"]


def test_nvtx_range_falls_back_when_context_enter_fails(monkeypatch):
    events: list[object] = []

    class FakeRange:
        def __enter__(self):
            events.append("enter")
            raise RuntimeError("enter unavailable")

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")

    class FakeNvtx:
        def range(self, name: str):
            events.append(("range", name))
            return FakeRange()

        def range_push(self, name: str):
            events.append(("push", name))

        def range_pop(self):
            events.append("pop")

    monkeypatch.setattr(
        helpers.torch,
        "cuda",
        SimpleNamespace(nvtx=FakeNvtx()),
        raising=False,
    )

    with _nvtx_range("enter"):
        events.append("body")

    assert events == [("range", "enter"), "enter", ("push", "enter"), "body", "pop"]


def test_nvtx_range_ignores_context_exit_failure(monkeypatch):
    events: list[object] = []

    class FakeRange:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append(("exit", exc_type))
            raise RuntimeError("exit unavailable")

    class FakeNvtx:
        def range(self, name: str):
            events.append(("range", name))
            return FakeRange()

    monkeypatch.setattr(
        helpers.torch,
        "cuda",
        SimpleNamespace(nvtx=FakeNvtx()),
        raising=False,
    )

    with _nvtx_range("exit"):
        events.append("body")

    assert events == [("range", "exit"), "enter", "body", ("exit", None)]


def test_nvtx_range_preserves_body_error_when_context_exit_fails(monkeypatch):
    events: list[object] = []

    class FakeRange:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append(("exit", exc_type))
            raise RuntimeError("exit unavailable")

    class FakeNvtx:
        def range(self, name: str):
            events.append(("range", name))
            return FakeRange()

    monkeypatch.setattr(
        helpers.torch,
        "cuda",
        SimpleNamespace(nvtx=FakeNvtx()),
        raising=False,
    )

    with pytest.raises(ValueError, match="body failed"):
        with _nvtx_range("body-error"):
            events.append("body")
            raise ValueError("body failed")

    assert events == [
        ("range", "body-error"),
        "enter",
        "body",
        ("exit", ValueError),
    ]


def test_nvtx_range_falls_back_to_push_pop(monkeypatch):
    events: list[object] = []

    class FakeNvtx:
        def range(self, name: str):
            events.append(("range", name))
            raise RuntimeError("range unavailable")

        def range_push(self, name: str):
            events.append(("push", name))

        def range_pop(self):
            events.append("pop")

    monkeypatch.setattr(
        helpers.torch,
        "cuda",
        SimpleNamespace(nvtx=FakeNvtx()),
        raising=False,
    )

    with _nvtx_range("backward"):
        events.append("body")

    assert events == [("range", "backward"), ("push", "backward"), "body", "pop"]


def test_nvtx_range_ignores_push_failures(monkeypatch):
    events: list[object] = []

    class FakeNvtx:
        def range_push(self, name: str):
            events.append(("push", name))
            raise RuntimeError("push unavailable")

        def range_pop(self):
            raise AssertionError("range_pop should not run after push failure")

    monkeypatch.setattr(
        helpers.torch,
        "cuda",
        SimpleNamespace(nvtx=FakeNvtx()),
        raising=False,
    )

    with _nvtx_range("optional"):
        events.append("body")

    assert events == [("push", "optional"), "body"]


def test_nvtx_range_ignores_pop_failures(monkeypatch):
    events: list[object] = []

    class FakeNvtx:
        def range_push(self, name: str):
            events.append(("push", name))

        def range_pop(self):
            events.append("pop")
            raise RuntimeError("pop unavailable")

    monkeypatch.setattr(
        helpers.torch,
        "cuda",
        SimpleNamespace(nvtx=FakeNvtx()),
        raising=False,
    )

    with _nvtx_range("cleanup"):
        events.append("body")

    assert events == [("push", "cleanup"), "body", "pop"]
