from __future__ import annotations

import pytest
import torch

from gpurec.core.kernels.wave_backward import (
    dts_cross_backward_accum_fused,
    uniform_cross_pibar_vjp_tree_from_ud_fused,
    wave_backward_uniform_fused,
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
        "mt_squeezed": torch.zeros(species, dtype=torch.float32),
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
def test_wave_backward_uniform_fused_requires_retained_metadata_before_launch(
    field: str,
    message: str,
) -> None:
    kwargs = _wave_backward_kwargs()
    kwargs[field] = None

    with pytest.raises(ValueError, match=message):
        wave_backward_uniform_fused(**kwargs)


def test_dts_cross_backward_output_requires_pibar_metadata_before_launch() -> None:
    kwargs = _dts_backward_kwargs()

    with pytest.raises(
        ValueError,
        match="mt_squeezed and pibar_row_max are required when outputting Pibar u_d",
    ):
        dts_cross_backward_accum_fused(**kwargs, output_pibar_ud=True)


def test_dts_cross_backward_rejects_invalid_pibar_mt_shape_before_launch() -> None:
    kwargs = _dts_backward_kwargs()

    with pytest.raises(
        ValueError,
        match=r"mt_squeezed must have shape \[S\] or \[G, S\]",
    ):
        dts_cross_backward_accum_fused(
            **kwargs,
            output_pibar_ud=True,
            mt_squeezed=torch.zeros((2, 2), dtype=torch.float32),
            pibar_row_max=torch.zeros(2, dtype=torch.float32),
        )


def test_dts_cross_backward_rejects_short_pibar_row_max_before_launch() -> None:
    kwargs = _dts_backward_kwargs()

    with pytest.raises(ValueError, match="one row-max value per Pi row"):
        dts_cross_backward_accum_fused(
            **kwargs,
            output_pibar_ud=True,
            mt_squeezed=torch.zeros(3, dtype=torch.float32),
            pibar_row_max=torch.zeros(1, dtype=torch.float32),
        )


def test_dts_cross_backward_side_activity_requires_staged_pibar_output() -> None:
    kwargs = _dts_backward_kwargs()

    with pytest.raises(ValueError, match="output_pibar_side_active requires output_pibar_ud"):
        dts_cross_backward_accum_fused(**kwargs, output_pibar_side_active=True)


def test_uniform_pibar_vjp_requires_reduce_idx_with_active_mask_before_launch() -> None:
    kwargs = _pibar_vjp_kwargs()

    with pytest.raises(ValueError, match="reduce_idx is required when active_mask is provided"):
        uniform_cross_pibar_vjp_tree_from_ud_fused(
            **kwargs,
            active_mask=torch.ones(1, dtype=torch.bool),
            reduce_idx=None,
        )


def test_uniform_pibar_vjp_requires_pibar_row_max_before_launch() -> None:
    kwargs = _pibar_vjp_kwargs()
    kwargs["pibar_row_max"] = None

    with pytest.raises(ValueError, match="pibar_row_max is required for DTS-staged Pibar VJP"):
        uniform_cross_pibar_vjp_tree_from_ud_fused(**kwargs)


def test_uniform_pibar_vjp_rejects_vector_side_threshold_before_launch() -> None:
    kwargs = _pibar_vjp_kwargs()

    with pytest.raises(ValueError, match="side_active_threshold tensor must contain one value"):
        uniform_cross_pibar_vjp_tree_from_ud_fused(
            **kwargs,
            side_active_threshold=torch.zeros(2, dtype=torch.float32),
        )


def test_uniform_pibar_vjp_rejects_negative_side_threshold_before_launch() -> None:
    kwargs = _pibar_vjp_kwargs()

    with pytest.raises(ValueError, match="side_active_threshold must be non-negative"):
        uniform_cross_pibar_vjp_tree_from_ud_fused(**kwargs, side_active_threshold=-0.1)
