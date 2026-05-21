import pytest
import torch

from gpurec.core.extract_parameters import (
    as_family_param,
    as_family_species,
    extract_parameters_uniform,
)


DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _log2_softmax(values: torch.Tensor) -> torch.Tensor:
    max_value = values.max(dim=-1, keepdim=True).values
    normalizer = torch.log2(torch.exp2(values - max_value).sum(dim=-1, keepdim=True))
    return values - (normalizer + max_value)


def test_as_family_param_shape_table() -> None:
    scalar = torch.tensor(2.0, dtype=DTYPE)
    species = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    family = torch.tensor([4.0, 5.0], dtype=DTYPE)
    family_column = torch.tensor([[6.0], [7.0]], dtype=DTYPE)
    family_species = torch.arange(6.0, dtype=DTYPE).reshape(2, 3)

    torch.testing.assert_close(
        as_family_param(scalar, S=3, device=DEVICE, dtype=DTYPE),
        torch.tensor([[2.0]], dtype=DTYPE),
    )
    torch.testing.assert_close(
        as_family_param(species, S=3, device=DEVICE, dtype=DTYPE),
        species.reshape(1, 3),
    )
    torch.testing.assert_close(
        as_family_param(family, S=3, device=DEVICE, dtype=DTYPE),
        family.reshape(2, 1),
    )
    torch.testing.assert_close(
        as_family_param(family_column, S=3, device=DEVICE, dtype=DTYPE),
        family_column,
    )
    torch.testing.assert_close(
        as_family_param(family_species, S=3, device=DEVICE, dtype=DTYPE),
        family_species,
    )


def test_as_family_param_family_rows_disambiguates_bare_vector_when_g_equals_s() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)

    without_family_rows = as_family_param(values, S=3, device=DEVICE, dtype=DTYPE)
    with_family_rows = as_family_param(
        values,
        S=3,
        device=DEVICE,
        dtype=DTYPE,
        family_rows=3,
    )

    torch.testing.assert_close(without_family_rows, values.reshape(1, 3))
    torch.testing.assert_close(with_family_rows, values.reshape(3, 1))


def test_as_family_species_expands_family_rows_after_disambiguation() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)

    actual = as_family_species(
        values,
        S=3,
        device=DEVICE,
        dtype=DTYPE,
        family_rows=3,
    )

    torch.testing.assert_close(actual, values.reshape(3, 1).expand(3, 3))
    assert actual.is_contiguous()


def test_as_family_species_docstring_documents_family_rows_ambiguity() -> None:
    doc = " ".join((as_family_species.__doc__ or "").split())

    for token in (
        "family_rows",
        "G == S",
        "bare length-G",
        "[P, S]",
    ):
        assert token in doc


def test_as_family_param_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="bad_param.*shape"):
        as_family_param(
            torch.zeros((2, 2, 1), dtype=DTYPE),
            S=3,
            device=DEVICE,
            dtype=DTYPE,
            name="bad_param",
        )


def test_extract_parameters_uniform_global_matches_manual_log_softmax() -> None:
    theta = torch.tensor([0.2, -0.4, 0.8], dtype=DTYPE)
    unnorm_row_max = torch.tensor([0.1, -0.2], dtype=DTYPE)

    log_pS, log_pD, log_pL, max_transfer = extract_parameters_uniform(
        theta,
        unnorm_row_max,
        specieswise=False,
        genewise=False,
    )

    expected = _log2_softmax(torch.cat((theta.new_zeros((1,)), theta), dim=-1))
    torch.testing.assert_close(log_pS, expected[0])
    torch.testing.assert_close(log_pD, expected[1])
    torch.testing.assert_close(log_pL, expected[2])
    torch.testing.assert_close(max_transfer, expected[3] + unnorm_row_max)


def test_extract_parameters_uniform_specieswise_matches_manual_log_softmax() -> None:
    theta = torch.tensor([[0.2, -0.4, 0.8], [0.1, 0.3, -0.6]], dtype=DTYPE)
    unnorm_row_max = torch.tensor([0.1, -0.2], dtype=DTYPE)

    log_pS, log_pD, log_pL, max_transfer = extract_parameters_uniform(
        theta,
        unnorm_row_max,
        specieswise=True,
        genewise=False,
    )

    expected = _log2_softmax(
        torch.cat((theta.new_zeros((theta.shape[0], 1)), theta), dim=-1)
    )
    torch.testing.assert_close(log_pS, expected[:, 0])
    torch.testing.assert_close(log_pD, expected[:, 1])
    torch.testing.assert_close(log_pL, expected[:, 2])
    torch.testing.assert_close(max_transfer, expected[:, 3] + unnorm_row_max)


def test_extract_parameters_uniform_genewise_matches_manual_log_softmax() -> None:
    theta = torch.tensor([[0.2, -0.4, 0.8], [0.1, 0.3, -0.6]], dtype=DTYPE)
    unnorm_row_max = torch.tensor([0.1, -0.2, 0.4], dtype=DTYPE)

    log_pS, log_pD, log_pL, max_transfer = extract_parameters_uniform(
        theta,
        unnorm_row_max,
        specieswise=False,
        genewise=True,
    )

    expected = _log2_softmax(
        torch.cat((theta.new_zeros((theta.shape[0], 1)), theta), dim=-1)
    )
    torch.testing.assert_close(log_pS, expected[:, 0])
    torch.testing.assert_close(log_pD, expected[:, 1])
    torch.testing.assert_close(log_pL, expected[:, 2])
    torch.testing.assert_close(
        max_transfer,
        expected[:, 3].unsqueeze(-1) + unnorm_row_max,
    )
