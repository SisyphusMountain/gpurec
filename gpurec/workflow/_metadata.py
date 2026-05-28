"""Internal checkpoint metadata validation helpers.

The checkpoint, optimization, and resume paths share this module for typed
payload checks and model identity extraction. It is not a public workflow API;
use ``gpurec.workflow.checkpoint`` for supported checkpoint tooling.
"""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

from gpurec._validation import finite_float, nonnegative_int


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
            if not isinstance(default, int):
                raise invalid_checkpoint_field(path, key)
            return default
        raise invalid_checkpoint_field(path, key)
    if value is None:
        if allow_none:
            return None
        raise invalid_checkpoint_field(path, key)
    try:
        number = nonnegative_int(key, value)
    except ValueError as exc:
        raise invalid_checkpoint_field(path, key) from exc
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
    try:
        number = finite_float(key, value)
    except ValueError as exc:
        raise invalid_checkpoint_field(path, key) from exc
    return number


def checkpoint_string_list(path: Path, key: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise RuntimeError(f"checkpoint {path} has invalid {key} metadata")
    return list(value)


def checkpoint_progress(path: Path, payload: dict[str, Any]) -> tuple[int, int]:
    step = checkpoint_nonnegative_int(path, "step", payload.get("step", MISSING))
    if step is None:
        raise invalid_checkpoint_field(path, "step")
    next_step = checkpoint_nonnegative_int(
        path,
        "next_step",
        payload.get("next_step", MISSING),
    )
    if next_step is None:
        raise invalid_checkpoint_field(path, "next_step")
    if next_step not in {step, step + 1}:
        raise RuntimeError(f"checkpoint {path} has inconsistent progress metadata")
    return step, next_step


def checkpoint_status_dict(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if status is None:
        return {}
    if not isinstance(status, dict):
        raise RuntimeError(f"checkpoint {path} has invalid status metadata")
    return status


def model_family_names(model: Any) -> list[str]:
    if hasattr(model, "family_names"):
        return list(model.family_names)
    return []


def model_species_names(model: Any) -> list[str]:
    if hasattr(model, "species_names"):
        return list(model.species_names)
    return []
