"""Explicit theta layout contracts for reconciliation rate modes.

This module is intentionally inert: it describes and validates theta layout
intent without routing any kernels or changing model execution paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any


THETA_EVENT_COUNT = 3


class RateMode(str, Enum):
    """Canonical parameter-sharing modes for D/L/T theta tensors."""

    GLOBAL = "global"
    SPECIESWISE = "specieswise"
    GENEWISE = "genewise"

    @classmethod
    def normalize(cls, value: "RateMode | str") -> "RateMode":
        """Return a canonical rate mode, accepting ``uniform`` as global."""
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        if text == "uniform":
            text = "global"
        try:
            return cls(text)
        except ValueError as exc:
            valid = ("global", "uniform", "specieswise", "genewise")
            raise ValueError(f"unknown rate mode {value!r}; valid modes: {valid}") from exc

    @classmethod
    def from_flags(cls, *, genewise: bool, specieswise: bool) -> "RateMode":
        """Convert existing boolean mode flags into an explicit rate mode."""
        if not isinstance(genewise, bool) or not isinstance(specieswise, bool):
            raise ValueError("genewise and specieswise flags must be booleans")
        if genewise and specieswise:
            raise ValueError(
                "combined genewise+specieswise theta layouts are not part of "
                "this RateMode contract"
            )
        if genewise:
            return cls.GENEWISE
        if specieswise:
            return cls.SPECIESWISE
        return cls.GLOBAL


@dataclass(frozen=True)
class ParameterLayout:
    """Validated theta shape intent and indexing metadata for a rate mode.

    ``family_indices`` names the active family batch this metadata describes.
    It does not imply per-family parameters unless ``row_axis == "family"``.
    """

    mode: RateMode | str
    species_count: int
    family_count: int
    family_indices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        mode = RateMode.normalize(self.mode)
        species_count = _positive_count("species_count", self.species_count)
        family_count = _positive_count("family_count", self.family_count)
        family_indices = _normalize_family_indices(
            self.family_indices,
            family_count=family_count,
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "species_count", species_count)
        object.__setattr__(self, "family_count", family_count)
        object.__setattr__(self, "family_indices", family_indices)

    @classmethod
    def for_mode(
        cls,
        mode: RateMode | str,
        *,
        species_count: int,
        family_count: int,
        family_indices: Sequence[int] | None = None,
    ) -> "ParameterLayout":
        """Build layout metadata from explicit mode intent."""
        return cls(
            mode=mode,
            species_count=species_count,
            family_count=family_count,
            family_indices=family_indices,
        )

    @classmethod
    def from_shape(
        cls,
        theta_or_shape: Any,
        *,
        species_count: int,
        family_count: int,
        mode: RateMode | str | None = None,
        family_indices: Sequence[int] | None = None,
        name: str = "theta",
    ) -> "ParameterLayout":
        """Infer or validate layout intent from a theta shape.

        If ``mode`` is omitted and ``species_count == family_count``, a
        two-dimensional ``[N, 3]`` theta is rejected as ambiguous because it
        could mean either specieswise or genewise rows.
        """
        shape = _shape_tuple(name, theta_or_shape)
        if mode is not None:
            layout = cls.for_mode(
                mode,
                species_count=species_count,
                family_count=family_count,
                family_indices=family_indices,
            )
            layout.validate_theta_shape(shape, name=name)
            return layout

        species_count = _positive_count("species_count", species_count)
        family_count = _positive_count("family_count", family_count)
        inferred_mode = _infer_mode_from_shape(
            name,
            shape,
            species_count=species_count,
            family_count=family_count,
        )
        return cls.for_mode(
            inferred_mode,
            species_count=species_count,
            family_count=family_count,
            family_indices=family_indices,
        )

    @property
    def theta_shape(self) -> tuple[int, ...]:
        """Expected theta shape for this mode."""
        if self.mode is RateMode.GLOBAL:
            return (THETA_EVENT_COUNT,)
        if self.mode is RateMode.SPECIESWISE:
            return (self.species_count, THETA_EVENT_COUNT)
        return (self.family_count, THETA_EVENT_COUNT)

    @property
    def row_axis(self) -> str | None:
        """Theta row axis meaning: ``species``, ``family``, or ``None``."""
        if self.mode is RateMode.SPECIESWISE:
            return "species"
        if self.mode is RateMode.GENEWISE:
            return "family"
        return None

    @property
    def shared_across_families(self) -> bool:
        """Whether one active family can share theta rows with another."""
        return self.mode is not RateMode.GENEWISE

    @property
    def species_indices(self) -> tuple[int, ...]:
        """Species rows addressed by specieswise theta layouts."""
        return tuple(range(self.species_count))

    @property
    def theta_row_indices(self) -> tuple[int, ...]:
        """Rows of a two-dimensional theta addressed by this layout."""
        if self.mode is RateMode.GLOBAL:
            return ()
        if self.mode is RateMode.SPECIESWISE:
            return self.species_indices
        return tuple(self.family_indices)

    @property
    def e_shape(self) -> tuple[int, ...]:
        """Expected extinction-probability shape for this active layout."""
        if self.mode is RateMode.GENEWISE:
            return (len(self.family_indices), self.species_count)
        return (self.species_count,)

    def validate_theta_shape(
        self,
        theta_or_shape: Any,
        *,
        name: str = "theta",
    ) -> tuple[int, ...]:
        """Validate a theta tensor or shape against this explicit layout."""
        shape = _shape_tuple(name, theta_or_shape)
        expected = self.theta_shape
        if shape != expected:
            raise ValueError(
                f"{name} shape for {self.mode.value} mode must be {expected}, "
                f"got {shape}"
            )
        return shape

    def as_metadata(self) -> dict[str, Any]:
        """Return JSON-friendly layout and indexing metadata."""
        return {
            "mode": self.mode.value,
            "theta_shape": list(self.theta_shape),
            "row_axis": self.row_axis,
            "shared_across_families": self.shared_across_families,
            "e_shape": [int(dim) for dim in self.e_shape],
            "family_indices": [int(index) for index in self.family_indices],
            "species_indices": [int(index) for index in self.species_indices],
            "theta_row_indices": [int(index) for index in self.theta_row_indices],
        }


def _positive_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count


def _integer_index(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} entries must be integers")
    return int(value)


def _normalize_family_indices(
    family_indices: Sequence[int] | None,
    *,
    family_count: int,
) -> tuple[int, ...]:
    if family_indices is None:
        return tuple(range(family_count))
    if isinstance(family_indices, (str, bytes)):
        raise ValueError("family_indices must be a sequence of integers")
    try:
        values = tuple(
            _integer_index("family_indices", index)
            for index in family_indices
        )
    except TypeError as exc:
        raise ValueError("family_indices must be a sequence of integers") from exc
    if not values:
        raise ValueError("family_indices must not be empty")
    seen: set[int] = set()
    for index in values:
        if index < 0 or index >= family_count:
            raise IndexError(
                f"family index {index} out of range for {family_count} families"
            )
        if index in seen:
            raise ValueError(f"duplicate family index {index}")
        seen.add(index)
    return values


def _shape_tuple(name: str, theta_or_shape: Any) -> tuple[int, ...]:
    raw_shape = getattr(theta_or_shape, "shape", theta_or_shape)
    if isinstance(raw_shape, (str, bytes)):
        raise ValueError(f"{name} shape must be a sequence of dimensions")
    try:
        dims = tuple(raw_shape)
    except TypeError as exc:
        raise ValueError(f"{name} shape must be a sequence of dimensions") from exc
    shape: list[int] = []
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, Integral):
            raise ValueError(f"{name} shape dimensions must be integers")
        size = int(dim)
        if size < 0:
            raise ValueError(f"{name} shape dimensions must be non-negative")
        shape.append(size)
    return tuple(shape)


def _infer_mode_from_shape(
    name: str,
    shape: tuple[int, ...],
    *,
    species_count: int,
    family_count: int,
) -> RateMode:
    if shape == (THETA_EVENT_COUNT,):
        return RateMode.GLOBAL
    if len(shape) != 2 or shape[1] != THETA_EVENT_COUNT:
        raise ValueError(
            f"{name} shape must be (3,), ({species_count}, 3), "
            f"or ({family_count}, 3); got {shape}"
        )

    row_count = shape[0]
    matches_species = row_count == species_count
    matches_family = row_count == family_count
    if matches_species and matches_family:
        raise ValueError(
            f"{name} shape {shape} is ambiguous because species_count and "
            f"family_count are both {row_count}; pass mode='specieswise' or "
            "mode='genewise' to state theta row intent"
        )
    if matches_species:
        return RateMode.SPECIESWISE
    if matches_family:
        return RateMode.GENEWISE
    raise ValueError(
        f"{name} shape must be (3,), ({species_count}, 3), "
        f"or ({family_count}, 3); got {shape}"
    )


__all__ = ["ParameterLayout", "RateMode"]
