"""Reference preparation for backward consumers of centered Pi state.

The production centered forward stores an absolute log2 row ``x`` as

``x[row, species] = residual[row, species] + offset[row]``.

The retained first-order adjoint predates that representation and consumes
absolute Pi/Pibar matrices.  This module provides an explicit, deliberately
simple fp64 reconstruction boundary so that the retained adjoint can be used as
a correctness oracle while native offset-aware backward kernels are developed.

This is not the steady-state implementation: reconstructing two full fp64
matrices loses the centered representation's memory and bandwidth advantage.
Callers should keep the ``*_reference`` naming when wiring this path so it
cannot be mistaken for the eventual performance implementation.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class CenteredPiBackwardReferenceState(NamedTuple):
    """Absolute fp64 views required by the retained backward implementation."""

    pi_absolute: torch.Tensor
    pibar_absolute: torch.Tensor
    pibar_row_max_absolute: torch.Tensor


def _validate_centered_rows(
    residual: torch.Tensor,
    offset: torch.Tensor,
    *,
    residual_name: str,
    offset_name: str,
) -> None:
    if residual.ndim != 2:
        raise ValueError(f"{residual_name} must have shape [rows, species]")
    if not residual.is_floating_point():
        raise TypeError(f"{residual_name} must be floating point")
    if offset.ndim != 1 or int(offset.shape[0]) != int(residual.shape[0]):
        raise ValueError(f"{offset_name} must have shape [{int(residual.shape[0])}]")
    if offset.dtype != torch.float64:
        raise TypeError(f"{offset_name} must use torch.float64")
    if offset.device != residual.device:
        raise ValueError(f"{offset_name} must be on the same device as {residual_name}")


def reconstruct_centered_rows_reference(
    residual: torch.Tensor,
    offset: torch.Tensor,
    *,
    residual_name: str = "residual",
    offset_name: str = "offset",
) -> torch.Tensor:
    """Reconstruct an absolute log2 matrix in fp64.

    ``-inf`` lanes remain ``-inf``.  The canonical representation of an
    all-``-inf`` row is an all-``-inf`` residual with a finite (normally zero)
    offset; this function intentionally does not replace or clamp non-finite
    values because doing so would hide a broken centered-state producer.
    """

    _validate_centered_rows(
        residual,
        offset,
        residual_name=residual_name,
        offset_name=offset_name,
    )
    return residual.to(dtype=torch.float64) + offset.unsqueeze(1)


def prepare_centered_backward_reference(
    pi_residual: torch.Tensor,
    pibar_residual: torch.Tensor,
    pi_offset: torch.Tensor,
    pibar_offset: torch.Tensor,
    pibar_row_max_residual: torch.Tensor,
) -> CenteredPiBackwardReferenceState:
    """Return absolute fp64 primals and row normalizers for retained backward.

    The centered forward's ``pibar_row_max`` metadata is the (possibly
    receiver-weighted) row maximum of the *Pi residual*.  The retained
    backward expects the corresponding absolute Pi-row maximum, hence the
    ``+ pi_offset`` conversion.  ``pibar_offset`` must not be used for that
    metadata: Pibar has its own independently selected row gauge.
    """

    _validate_centered_rows(
        pi_residual,
        pi_offset,
        residual_name="pi_residual",
        offset_name="pi_offset",
    )
    _validate_centered_rows(
        pibar_residual,
        pibar_offset,
        residual_name="pibar_residual",
        offset_name="pibar_offset",
    )
    if pi_residual.shape != pibar_residual.shape:
        raise ValueError("pi_residual and pibar_residual must have the same shape")
    if pibar_row_max_residual.ndim != 1 or int(pibar_row_max_residual.shape[0]) != int(
        pi_residual.shape[0]
    ):
        raise ValueError(
            "pibar_row_max_residual must contain one value per Pi row"
        )
    if not pibar_row_max_residual.is_floating_point():
        raise TypeError("pibar_row_max_residual must be floating point")
    if pibar_row_max_residual.device != pi_residual.device:
        raise ValueError(
            "pibar_row_max_residual must be on the same device as pi_residual"
        )

    return CenteredPiBackwardReferenceState(
        pi_absolute=reconstruct_centered_rows_reference(
            pi_residual,
            pi_offset,
            residual_name="pi_residual",
            offset_name="pi_offset",
        ),
        pibar_absolute=reconstruct_centered_rows_reference(
            pibar_residual,
            pibar_offset,
            residual_name="pibar_residual",
            offset_name="pibar_offset",
        ),
        pibar_row_max_absolute=(
            pibar_row_max_residual.to(dtype=torch.float64) + pi_offset
        ),
    )
