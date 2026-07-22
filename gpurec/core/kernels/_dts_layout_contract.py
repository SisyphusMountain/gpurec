"""CPU-testable DTS parameter layout contract for retained backward kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
from typing import Any


class DtsBackwardLayoutCode(IntEnum):
    """How a biological rate varies across gene families and species.

    A DTS rate may be tied globally, vary only with the species-tree state,
    vary only between gene families, or vary along both axes.  The integer
    values are passed as compile-time addressing modes to Triton; they do not
    encode model parameters themselves.
    """

    SHARED_SCALAR = 0
    SHARED_SPECIES = 1
    FAMILY_SCALAR = 2
    FAMILY_SPECIES = 3


@dataclass(frozen=True)
class DtsBackwardParamLayout:
    """Classification of the two mathematical axes of a DTS rate tensor.

    ``code`` determines whether the family and species indices contribute to
    the tensor address. ``intent`` is its readable equivalent for diagnostics.
    ``ambiguous_1d_family_species`` records the special ``G == S`` case, where
    a bare vector cannot reveal which axis its entries describe.
    """

    code: DtsBackwardLayoutCode
    intent: str
    ambiguous_1d_family_species: bool = False


def dts_backward_param_layout(
    value_or_shape: Any,
    *,
    S: int,
    family_indexed: bool,
) -> DtsBackwardParamLayout:
    """Classify the retained DTS backward parameter or gradient layout.

    With ``family_indexed`` true, retained backward intentionally classifies any
    1-D non-scalar tensor as family scalar rows before considering ``[S]``.
    This preserves current kernel semantics and documents why bare ``[G]`` is
    ambiguous when ``G == S``.
    """

    species_count = _positive_count("S", S)
    shape = _shape_tuple(value_or_shape)

    if _numel(shape) == 1:
        return DtsBackwardParamLayout(
            DtsBackwardLayoutCode.SHARED_SCALAR,
            intent="shared_scalar",
        )
    if family_indexed and len(shape) == 1:
        return DtsBackwardParamLayout(
            DtsBackwardLayoutCode.FAMILY_SCALAR,
            intent="family_scalar",
            ambiguous_1d_family_species=shape[0] == species_count,
        )
    if len(shape) == 1 and shape[0] == species_count:
        return DtsBackwardParamLayout(
            DtsBackwardLayoutCode.SHARED_SPECIES,
            intent="shared_species",
        )
    if family_indexed:
        if len(shape) == 2 and shape[1] == 1:
            return DtsBackwardParamLayout(
                DtsBackwardLayoutCode.FAMILY_SCALAR,
                intent="family_scalar",
            )
        if len(shape) == 2 and shape[1] == species_count:
            return DtsBackwardParamLayout(
                DtsBackwardLayoutCode.FAMILY_SPECIES,
                intent="family_species",
            )
    raise ValueError(
        "DTS parameters must be scalar, [S], [G], [G, 1], or [G, S]; "
        f"got shape {shape} with S={species_count}"
    )


def _shape_tuple(value_or_shape: Any) -> tuple[int, ...]:
    raw_shape = getattr(value_or_shape, "shape", value_or_shape)
    if isinstance(raw_shape, (str, bytes)):
        raise ValueError("DTS parameter shape must be a sequence of dimensions")
    try:
        dims = tuple(raw_shape)
    except TypeError as exc:
        raise ValueError("DTS parameter shape must be a sequence of dimensions") from exc

    shape: list[int] = []
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, Integral):
            raise ValueError("DTS parameter shape dimensions must be integers")
        size = int(dim)
        if size < 0:
            raise ValueError("DTS parameter shape dimensions must be non-negative")
        shape.append(size)
    return tuple(shape)


def _positive_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


__all__ = [
    "DtsBackwardLayoutCode",
    "DtsBackwardParamLayout",
    "dts_backward_param_layout",
]
