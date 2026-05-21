from __future__ import annotations

import pytest
import torch

from gpurec.core.parameter_layout import ParameterLayout, RateMode


def test_rate_mode_normalizes_uniform_alias_to_global() -> None:
    assert RateMode.normalize("global") is RateMode.GLOBAL
    assert RateMode.normalize("uniform") is RateMode.GLOBAL
    assert RateMode.normalize("specieswise") is RateMode.SPECIESWISE
    assert RateMode.normalize("genewise") is RateMode.GENEWISE


def test_rate_mode_from_flags_matches_existing_mode_booleans() -> None:
    assert RateMode.from_flags(genewise=False, specieswise=False) is RateMode.GLOBAL
    assert RateMode.from_flags(genewise=False, specieswise=True) is RateMode.SPECIESWISE
    assert RateMode.from_flags(genewise=True, specieswise=False) is RateMode.GENEWISE


def test_rate_mode_rejects_combined_flags() -> None:
    with pytest.raises(ValueError, match="genewise\\+specieswise"):
        RateMode.from_flags(genewise=True, specieswise=True)


def test_global_layout_contract_has_no_theta_row_axis() -> None:
    layout = ParameterLayout.for_mode(
        "uniform",
        species_count=4,
        family_count=3,
        family_indices=[2, 0],
    )

    assert layout.mode is RateMode.GLOBAL
    assert layout.theta_shape == (3,)
    assert layout.row_axis is None
    assert layout.shared_across_families is True
    assert layout.family_indices == (2, 0)
    assert layout.species_indices == (0, 1, 2, 3)
    assert layout.theta_row_indices == ()
    assert layout.validate_theta_shape(torch.zeros(3)) == (3,)
    assert layout.as_metadata() == {
        "mode": "global",
        "theta_shape": [3],
        "row_axis": None,
        "shared_across_families": True,
        "family_indices": [2, 0],
        "species_indices": [0, 1, 2, 3],
        "theta_row_indices": [],
    }


def test_specieswise_layout_contract_exposes_species_rows() -> None:
    layout = ParameterLayout.for_mode(
        RateMode.SPECIESWISE,
        species_count=4,
        family_count=3,
        family_indices=[1],
    )

    assert layout.theta_shape == (4, 3)
    assert layout.row_axis == "species"
    assert layout.shared_across_families is True
    assert layout.family_indices == (1,)
    assert layout.species_indices == (0, 1, 2, 3)
    assert layout.theta_row_indices == (0, 1, 2, 3)
    assert layout.validate_theta_shape(torch.zeros(4, 3)) == (4, 3)


def test_genewise_layout_contract_exposes_active_family_rows() -> None:
    layout = ParameterLayout.for_mode(
        "genewise",
        species_count=4,
        family_count=5,
        family_indices=[4, 2],
    )

    assert layout.theta_shape == (5, 3)
    assert layout.row_axis == "family"
    assert layout.shared_across_families is False
    assert layout.family_indices == (4, 2)
    assert layout.species_indices == (0, 1, 2, 3)
    assert layout.theta_row_indices == (4, 2)
    assert layout.validate_theta_shape((5, 3)) == (5, 3)


def test_layout_infers_nonambiguous_shapes_without_explicit_mode() -> None:
    global_layout = ParameterLayout.from_shape(
        (3,),
        species_count=4,
        family_count=2,
    )
    species_layout = ParameterLayout.from_shape(
        torch.zeros(4, 3),
        species_count=4,
        family_count=2,
    )
    gene_layout = ParameterLayout.from_shape(
        torch.zeros(2, 3),
        species_count=4,
        family_count=2,
    )

    assert global_layout.mode is RateMode.GLOBAL
    assert species_layout.mode is RateMode.SPECIESWISE
    assert gene_layout.mode is RateMode.GENEWISE


def test_layout_requires_explicit_mode_when_species_and_family_counts_match() -> None:
    with pytest.raises(ValueError, match="ambiguous.*species_count.*family_count"):
        ParameterLayout.from_shape(torch.zeros(3, 3), species_count=3, family_count=3)

    species_layout = ParameterLayout.from_shape(
        torch.zeros(3, 3),
        species_count=3,
        family_count=3,
        mode="specieswise",
    )
    gene_layout = ParameterLayout.from_shape(
        torch.zeros(3, 3),
        species_count=3,
        family_count=3,
        mode="genewise",
    )

    assert species_layout.row_axis == "species"
    assert gene_layout.row_axis == "family"


def test_layout_rejects_shape_that_conflicts_with_explicit_mode() -> None:
    layout = ParameterLayout.for_mode(
        "specieswise",
        species_count=4,
        family_count=2,
    )

    with pytest.raises(ValueError, match="theta shape for specieswise mode"):
        layout.validate_theta_shape(torch.zeros(2, 3))


@pytest.mark.parametrize("family_indices", [[0, 0], [True], [-1], [3]])
def test_layout_rejects_invalid_family_index_metadata(
    family_indices: list[int],
) -> None:
    error_type = IndexError if family_indices in ([-1], [3]) else ValueError
    with pytest.raises(error_type):
        ParameterLayout.for_mode(
            "genewise",
            species_count=4,
            family_count=3,
            family_indices=family_indices,
        )
