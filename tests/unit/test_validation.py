from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import gpurec.workflow.model_factory as workflow_model_factory
from gpurec.api._validation import (
    integer_value,
    nonnegative_int,
    positive_even_int,
    require_cuda_device,
    theta_init_base_from_rates,
)
from gpurec.workflow.config import RunConfig
from gpurec.workflow.model_factory import build_alerax_workflow_model


def test_require_cuda_device_rejects_malformed_cuda_spec() -> None:
    with pytest.raises(ValueError, match="invalid CUDA device"):
        require_cuda_device("cuda:not-a-device", owner="unit test")


def test_require_cuda_device_rejects_explicit_index_outside_available_range(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="only 1 CUDA device"):
        require_cuda_device("cuda:1", owner="unit test")


def test_require_cuda_device_accepts_available_explicit_index(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert require_cuda_device("cuda:0", owner="unit test") == torch.device("cuda:0")


def test_positive_even_int_accepts_positive_even_integer() -> None:
    assert positive_even_int("fixed_iters_Pi", 6) == 6


@pytest.mark.parametrize("value", [0, 3, 4.5, math.inf, True])
def test_positive_even_int_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="fixed_iters_Pi"):
        positive_even_int("fixed_iters_Pi", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -3, 4.0])
def test_integer_value_accepts_integral_values(value: object) -> None:
    assert integer_value("family_index", value) == int(value)


@pytest.mark.parametrize("value", [1.5, math.inf, math.nan, True])
def test_integer_value_rejects_non_integral_values(value: object) -> None:
    with pytest.raises(ValueError, match="family_index"):
        integer_value("family_index", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 3, 4.0])
def test_nonnegative_int_accepts_nonnegative_integral_values(value: object) -> None:
    assert nonnegative_int("family_chunk_candidates entries", value) == int(value)


@pytest.mark.parametrize("value", [-1, 1.5, math.inf, True])
def test_nonnegative_int_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="family_chunk_candidates entries"):
        nonnegative_int("family_chunk_candidates entries", value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "rates",
    [
        [True, 0.1, 0.1],
        torch.tensor([True, True, True]),
    ],
)
def test_theta_init_rates_reject_bool_values(rates: object) -> None:
    with pytest.raises(ValueError, match="theta_init_rates"):
        theta_init_base_from_rates(
            rates,  # type: ignore[arg-type]
            dtype=torch.float64,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "rates",
    [
        "0.1",
        object(),
        [[0.1], [0.1, 0.1]],
    ],
)
def test_theta_init_rates_wraps_non_numeric_conversion_errors(rates: object) -> None:
    with pytest.raises(ValueError, match="theta_init_rates"):
        theta_init_base_from_rates(
            rates,  # type: ignore[arg-type]
            dtype=torch.float64,
            device=torch.device("cpu"),
        )


def test_build_alerax_workflow_model_rejects_unavailable_cuda_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="global",
        device="cuda:2",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def fail_from_alerax_families(*args: object, **kwargs: object) -> object:
        raise AssertionError("model construction should not run")

    monkeypatch.setattr(
        workflow_model_factory.GeneReconModel,
        "from_alerax_families",
        staticmethod(fail_from_alerax_families),
    )

    with pytest.raises(ValueError, match="only 1 CUDA device"):
        build_alerax_workflow_model(config)
