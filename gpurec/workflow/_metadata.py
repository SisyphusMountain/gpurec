from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any


MISSING = object()


def invalid_checkpoint_field(path: Path, key: str) -> RuntimeError:
    return RuntimeError(f"checkpoint {path} has invalid {key}")


def checkpoint_nonnegative_int(
    path: Path,
    key: str,
    value: Any,
    *,
    default: int | object = MISSING,
    allow_none: bool = False,
) -> int | None:
    if value is MISSING:
        if default is not MISSING:
            return int(default)
        raise invalid_checkpoint_field(path, key)
    if value is None:
        if allow_none:
            return None
        raise invalid_checkpoint_field(path, key)
    if isinstance(value, bool):
        raise invalid_checkpoint_field(path, key)
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        raw = float(value)
        if not math.isfinite(raw) or not raw.is_integer():
            raise invalid_checkpoint_field(path, key)
        number = int(raw)
    else:
        raise invalid_checkpoint_field(path, key)
    if number < 0:
        raise invalid_checkpoint_field(path, key)
    return number


def checkpoint_finite_float(
    path: Path,
    key: str,
    value: Any,
    *,
    allow_none: bool = False,
) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise invalid_checkpoint_field(path, key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise invalid_checkpoint_field(path, key)
    number = float(value)
    if not math.isfinite(number):
        raise invalid_checkpoint_field(path, key)
    return number


def checkpoint_string_list(path: Path, key: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise RuntimeError(f"checkpoint {path} has invalid {key} metadata")
    return list(value)


def model_family_names(model: Any) -> list[str]:
    if hasattr(model, "family_names"):
        return list(model.family_names)
    return []


def model_species_names(model: Any) -> list[str]:
    if hasattr(model, "species_names"):
        return list(model.species_names)
    return []
