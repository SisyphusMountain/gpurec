from __future__ import annotations

import pytest
import torch

from gpurec.core.kernels.wave_backward import (
    accumulate_split_dts_vjp,
    accumulate_split_pibar_vjp,
    compute_wave_adjoint,
)


def _wave_backward_kwargs() -> dict[str, object]:
    rows, species, wave_size = 2, 3, 1
    pi = torch.zeros((rows, species), dtype=torch.float32)
    return {
        "Pi_star": pi,
        "Pibar_star": pi.clone(),
        "ws": 0,
        "W": wave_size,
        "S": species,
        "dts_r": None,
        "rhs": torch.zeros((wave_size, species), dtype=torch.float32),
        "max_transfer_mat": torch.zeros(species, dtype=torch.float32),
        "DL_const": torch.zeros(species, dtype=torch.float32),
        "Ebar": torch.zeros(species, dtype=torch.float32),
        "E": torch.zeros(species, dtype=torch.float32),
        "SL1_const": torch.zeros(species, dtype=torch.float32),
        "SL2_const": torch.zeros(species, dtype=torch.float32),
        "sp_child1": torch.zeros(species, dtype=torch.long),
        "sp_child2": torch.zeros(species, dtype=torch.long),
        "leaf_term_wt": torch.zeros((wave_size, species), dtype=torch.float32),
        "sp_parent": torch.zeros(species, dtype=torch.long),
        "max_ancestor_depth": 1,
        "pibar_row_max": torch.zeros(rows, dtype=torch.float32),
    }


def _dts_backward_kwargs() -> dict[str, object]:
    rows, species, wave_size = 2, 3, 1
    pi = torch.zeros((rows, species), dtype=torch.float32)
    return {
        "Pi_star": pi,
        "Pibar_star": pi.clone(),
        "v_k": torch.zeros((wave_size, species), dtype=torch.float32),
        "ws": 0,
        "sl": torch.tensor([0], dtype=torch.long),
        "sr": torch.tensor([1], dtype=torch.long),
        "reduce_idx": torch.tensor([0], dtype=torch.long),
        "wlsp": torch.zeros(wave_size, dtype=torch.float32),
        "log_pD": torch.zeros(species, dtype=torch.float32),
        "log_pS": torch.zeros(species, dtype=torch.float32),
        "sp_child1": torch.zeros(species, dtype=torch.long),
        "sp_child2": torch.zeros(species, dtype=torch.long),
        "accumulated_rhs": torch.zeros((rows, species), dtype=torch.float32),
        "S": species,
    }


def _pibar_vjp_kwargs() -> dict[str, object]:
    rows, species = 2, 3
    return {
        "Pi_star": torch.zeros((rows, species), dtype=torch.float32),
        "pibar_ud": torch.zeros((rows, species), dtype=torch.float32),
        "pibar_A": torch.zeros(rows, dtype=torch.float32),
        "sl": torch.tensor([0], dtype=torch.long),
        "sr": torch.tensor([1], dtype=torch.long),
        "accumulated_rhs": torch.zeros((rows, species), dtype=torch.float32),
        "S": species,
        "pibar_row_max": torch.zeros(rows, dtype=torch.float32),
    }


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("sp_parent", "sp_parent is required for the retained backward fast path"),
        (
            "max_ancestor_depth",
            "max_ancestor_depth is required for the retained backward fast path",
        ),
        (
            "pibar_row_max",
            "pibar_row_max is required for the retained backward fast path",
        ),
    ],
)
def test_compute_wave_adjoint_requires_retained_metadata_before_launch(
    field: str,
    message: str,
) -> None:
    kwargs = _wave_backward_kwargs()
    kwargs[field] = None

    with pytest.raises(ValueError, match=message):
        compute_wave_adjoint(**kwargs)


def test_split_dts_vjp_output_requires_pibar_metadata_before_launch() -> None:
    kwargs = _dts_backward_kwargs()

    with pytest.raises(
        ValueError,
        match="max_transfer_mat and pibar_row_max are required when outputting Pibar u_d",
    ):
        accumulate_split_dts_vjp(**kwargs, output_pibar_ud=True)


def test_split_dts_vjp_rejects_invalid_pibar_max_transfer_shape_before_launch() -> None:
    kwargs = _dts_backward_kwargs()

    with pytest.raises(
        ValueError,
        match=r"max_transfer_mat must have shape \[S\] or \[G, S\]",
    ):
        accumulate_split_dts_vjp(
            **kwargs,
            output_pibar_ud=True,
            max_transfer_mat=torch.zeros((2, 2), dtype=torch.float32),
            pibar_row_max=torch.zeros(2, dtype=torch.float32),
        )


def test_split_dts_vjp_rejects_short_pibar_row_max_before_launch() -> None:
    kwargs = _dts_backward_kwargs()

    with pytest.raises(ValueError, match="one row-max value per Pi row"):
        accumulate_split_dts_vjp(
            **kwargs,
            output_pibar_ud=True,
            max_transfer_mat=torch.zeros(3, dtype=torch.float32),
            pibar_row_max=torch.zeros(1, dtype=torch.float32),
        )


def test_split_dts_vjp_active_split_sides_requires_staged_pibar_output() -> None:
    kwargs = _dts_backward_kwargs()

    with pytest.raises(ValueError, match="output_active_split_sides requires output_pibar_ud"):
        accumulate_split_dts_vjp(**kwargs, output_active_split_sides=True)


def test_pibar_vjp_requires_reduce_idx_with_active_parent_rows_before_launch() -> None:
    kwargs = _pibar_vjp_kwargs()

    with pytest.raises(ValueError, match="reduce_idx is required when active_parent_rows is provided"):
        accumulate_split_pibar_vjp(
            **kwargs,
            active_parent_rows=torch.ones(1, dtype=torch.bool),
            reduce_idx=None,
        )


def test_pibar_vjp_requires_pibar_row_max_before_launch() -> None:
    kwargs = _pibar_vjp_kwargs()
    kwargs["pibar_row_max"] = None

    with pytest.raises(ValueError, match="pibar_row_max is required for DTS-staged Pibar VJP"):
        accumulate_split_pibar_vjp(**kwargs)


def test_pibar_vjp_rejects_vector_side_threshold_before_launch() -> None:
    kwargs = _pibar_vjp_kwargs()

    with pytest.raises(ValueError, match="active_split_side_threshold tensor must contain one value"):
        accumulate_split_pibar_vjp(
            **kwargs,
            active_split_side_threshold=torch.zeros(2, dtype=torch.float32),
        )


def test_pibar_vjp_rejects_negative_side_threshold_before_launch() -> None:
    kwargs = _pibar_vjp_kwargs()

    with pytest.raises(ValueError, match="active_split_side_threshold must be non-negative"):
        accumulate_split_pibar_vjp(**kwargs, active_split_side_threshold=-0.1)
