from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gpurec.workflow.model_factory as workflow_model_factory
from gpurec.api import GeneReconModel
from gpurec.api._validation import (
    finite_float,
    integer_value,
    nonnegative_float,
    nonnegative_int,
    positive_float,
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


@pytest.mark.parametrize(
    ("validator", "name"),
    [
        (finite_float, "tol_E"),
        (nonnegative_float, "pi_max_diff_tol"),
        (positive_float, "min_rate"),
    ],
)
@pytest.mark.parametrize("value", [True, torch.tensor(True)])
def test_float_validators_reject_bool_values(
    validator,
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        validator(name, value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tol_E": True}, "tol_E"),
        ({"pi_max_diff_tol": True}, "pi_max_diff_tol"),
    ],
)
def test_gene_recon_model_rejects_bool_float_controls_before_device_check(
    kwargs: dict[str, object],
    message: str,
) -> None:
    dataset = SimpleNamespace(
        dtype=torch.float64,
        genewise=False,
        specieswise=False,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match=message):
        GeneReconModel(dataset=dataset, mode="global", **kwargs)  # type: ignore[arg-type]


def test_gene_recon_model_clamp_rejects_bool_min_rate_before_mutation() -> None:
    model = SimpleNamespace(theta=torch.nn.Parameter(torch.tensor([0.0])))

    with pytest.raises(ValueError, match="min_rate"):
        GeneReconModel.clamp_theta_(model, min_rate=True)  # type: ignore[arg-type]

    torch.testing.assert_close(model.theta.detach(), torch.tensor([0.0]))


def _fake_dataset_for_mode(
    mode: str,
    *,
    species_count: int = 2,
    family_count: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        dtype=torch.float64,
        genewise=mode == "genewise",
        specieswise=mode == "specieswise",
        device=torch.device("cpu"),
        S=species_count,
        families=[object() for _ in range(family_count)],
    )


@pytest.mark.parametrize(
    ("mode", "theta_init"),
    [
        ("global", torch.zeros(2)),
        ("global", torch.zeros(4)),
        ("global", torch.zeros(1, 3)),
        ("specieswise", torch.zeros(2)),
        ("specieswise", torch.zeros(2, 4)),
        ("specieswise", torch.zeros(3, 3)),
        ("genewise", torch.zeros(3)),
        ("genewise", torch.zeros(2, 4)),
        ("genewise", torch.zeros(3, 3)),
    ],
)
def test_gene_recon_model_rejects_invalid_theta_init_shape_before_device_check(
    mode: str,
    theta_init: torch.Tensor,
) -> None:
    dataset = _fake_dataset_for_mode(mode)

    with pytest.raises(ValueError, match="theta_init.*shape"):
        GeneReconModel(
            dataset=dataset,  # type: ignore[arg-type]
            mode=mode,
            theta_init=theta_init,
        )


@pytest.mark.parametrize(
    ("mode", "theta"),
    [
        ("global", torch.zeros(4)),
        ("specieswise", torch.zeros(2, 4)),
        ("genewise", torch.zeros(3, 3)),
    ],
)
def test_full_loss_for_theta_rejects_invalid_explicit_theta_shape_before_streaming(
    mode: str,
    theta: torch.Tensor,
) -> None:
    model = object.__new__(GeneReconModel)
    model._mode = mode
    model._dataset = _fake_dataset_for_mode(mode)

    def unexpected_stream(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("streaming should not run for invalid theta shape")

    model._stream_full_batches = unexpected_stream  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="theta.*shape"):
        model.full_loss_for_theta(theta)


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
