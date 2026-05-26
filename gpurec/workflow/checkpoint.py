"""Workflow checkpoint helpers with an explicit lower-level support boundary.

The stable shortcut surface is ``gpurec.workflow``/top-level ``gpurec``.  This
submodule remains supported for advanced tooling that needs to inspect or
restore workflow checkpoints directly, but the payload schema is versioned and
callers should prefer ``optimize()``, ``sample()``, or ``RunConfig``/
``SamplingConfig`` unless they specifically need checkpoint metadata.

Version-1 checkpoints carry identity metadata for safe restore:
``family_names``, ``species_names``, and config identity fields
``species_tree``, ``families_file``, ``mode``, ``start``, and
``max_families``.  ``load_checkpoint()`` requires those fields to exist and
validates the name-list metadata. New checkpoints also carry ``route_metadata``
with the resolved objective, gradient route, rate parameterization, optimizer,
and solver route. ``validate_checkpoint_model_compatibility()`` compares
them with the active ``RunConfig`` and rebuilt model before
``restore_model_theta()`` copies parameters, first validating the stored config
with ``RunConfig.from_dict(...)``, normalizing only path identity fields during
comparison, requiring present route metadata to be complete for the current
route except audit-only default-route reporting fields, and allowing
mutable reporting fields such as the configured/effective step cap to differ
for resume.  ``load_checkpoint()`` is a lower-level payload reader and does not
reconstruct a full ``RunConfig``.
"""

from __future__ import annotations

import pickle
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from ._metadata import (
    checkpoint_progress,
    checkpoint_status_dict,
    checkpoint_string_list,
    model_family_names,
    model_species_names,
)
from .config import RunConfig, effective_route_metadata


__all__ = [
    "CHECKPOINT_VERSION",
    "load_checkpoint",
    "restore_model_theta",
    "save_checkpoint",
    "validate_checkpoint_model_compatibility",
]

CHECKPOINT_VERSION = 1
_REQUIRED_CHECKPOINT_KEYS = {"version", "config", "theta", "step", "next_step"}
_REQUIRED_CHECKPOINT_IDENTITY_KEYS = {"family_names", "species_names"}
_CHECKPOINT_CONFIG_IDENTITY_KEYS = (
    "species_tree",
    "families_file",
    "mode",
    "start",
    "max_families",
)
_ROUTE_METADATA_RESUME_COMPATIBILITY_EXEMPT_KEYS = frozenset(
    {
        "configured_steps",
        "mode_default_optimizer",
        "optimizer_step_cap",
        "optimizer_step_cap_reason",
        "production_default_optimizer_setting_mismatches",
        "production_default_route_mismatches",
        "uses_mode_default_optimizer",
        "uses_production_default_optimizer_settings",
        "uses_production_default_route",
    }
)


def _normalize_checkpoint_identity_value(key: str, value: Any) -> Any:
    if key in {"species_tree", "families_file"} and value is not None:
        return str(Path(value).expanduser().resolve())
    return value


def validate_checkpoint_model_compatibility(
    *,
    path: str | Path,
    config: RunConfig,
    model: Any,
    payload: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError(
            f"checkpoint {checkpoint_path} is incompatible with current run: "
            "invalid config metadata"
        )
    _require_config_identity_fields(checkpoint_path, checkpoint_config)
    _validate_checkpoint_run_config(checkpoint_path, checkpoint_config)
    current_config = config.to_dict()
    for key in _CHECKPOINT_CONFIG_IDENTITY_KEYS:
        checkpoint_value = _normalize_checkpoint_identity_value(
            key,
            checkpoint_config.get(key),
        )
        current_value = _normalize_checkpoint_identity_value(
            key,
            current_config.get(key),
        )
        if checkpoint_value != current_value:
            raise RuntimeError(
                f"checkpoint {checkpoint_path} is incompatible with current run: "
                f"config.{key} differs"
            )

    _require_route_metadata_compatible(
        checkpoint_path,
        payload.get("route_metadata"),
        effective_route_metadata(config),
    )

    checkpoint_family_names = checkpoint_string_list(
        checkpoint_path,
        "family_names",
        payload.get("family_names"),
    )
    current_family_names = model_family_names(model)
    if checkpoint_family_names != current_family_names:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} is incompatible with current run: "
            "family_names differ"
        )

    checkpoint_species_names = checkpoint_string_list(
        checkpoint_path,
        "species_names",
        payload.get("species_names"),
    )
    current_species_names = model_species_names(model)
    if checkpoint_species_names != current_species_names:
        raise RuntimeError(
            f"checkpoint {checkpoint_path} is incompatible with current run: "
            "species_names differ"
        )


def _require_route_metadata_compatible(
    path: Path,
    checkpoint_route: Any,
    current_route: dict[str, Any],
) -> None:
    if checkpoint_route is None:
        return
    if not isinstance(checkpoint_route, dict):
        raise RuntimeError(f"checkpoint {path} has invalid route_metadata")
    for key, current_value in current_route.items():
        if key in _ROUTE_METADATA_RESUME_COMPATIBILITY_EXEMPT_KEYS:
            continue
        if key in checkpoint_route and checkpoint_route[key] != current_value:
            raise RuntimeError(
                f"checkpoint {path} is incompatible with current run: "
                f"route_metadata.{key} differs"
            )
    missing = sorted(
        key
        for key in current_route
        if key not in _ROUTE_METADATA_RESUME_COMPATIBILITY_EXEMPT_KEYS
        and key not in checkpoint_route
    )
    if missing:
        raise RuntimeError(
            f"checkpoint {path} is incompatible with current run: "
            f"route_metadata missing key(s): {', '.join(missing)}"
        )


def _validate_checkpoint_run_config(path: Path, config: dict[str, Any]) -> None:
    try:
        RunConfig.from_dict(config)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"checkpoint {path} has invalid RunConfig metadata: {exc}"
        ) from exc


def save_checkpoint(
    path: str | Path,
    *,
    config: RunConfig,
    model: Any,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    status: dict[str, Any],
    row: dict[str, Any] | None = None,
    next_step: int | None = None,
    optimizer_phase: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "step": int(step),
        "next_step": int(step) + 1 if next_step is None else int(next_step),
        "config": config.to_dict(),
        "route_metadata": effective_route_metadata(config),
        "theta": model.theta.detach().cpu(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "optimizer_phase": optimizer_phase,
        "status": status,
        "last_row": row,
        "family_names": model_family_names(model),
        "species_names": model_species_names(model),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _safe_torch_load(
    path: Path,
    *,
    map_location: str | torch.device,
    artifact: str,
) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"could not safely load {artifact} {path}; regenerate the artifact "
            "or migrate it from a trusted source before retrying"
        ) from exc


def _validate_checkpoint_payload(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint {path} must contain a dictionary payload")
    missing = sorted(_REQUIRED_CHECKPOINT_KEYS - set(payload))
    if missing:
        raise RuntimeError(f"checkpoint {path} is missing key(s): {', '.join(missing)}")
    checkpoint_version = _checkpoint_version(path, payload["version"])
    if checkpoint_version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"checkpoint {path} has unsupported version {payload['version']!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    if not isinstance(payload["config"], dict):
        raise RuntimeError(f"checkpoint {path} has invalid config metadata")
    route_metadata = payload.get("route_metadata")
    if route_metadata is not None and not isinstance(route_metadata, dict):
        raise RuntimeError(f"checkpoint {path} has invalid route_metadata")
    missing_identity = sorted(_REQUIRED_CHECKPOINT_IDENTITY_KEYS - set(payload))
    if missing_identity:
        raise RuntimeError(
            f"checkpoint {path} is missing identity key(s): "
            f"{', '.join(missing_identity)}"
        )
    _require_config_identity_fields(path, payload["config"])
    checkpoint_progress(path, payload)
    theta = payload["theta"]
    if not torch.is_tensor(theta):
        raise RuntimeError(f"checkpoint {path} has invalid theta tensor")
    if not torch.is_floating_point(theta):
        raise RuntimeError(f"checkpoint {path} has invalid theta tensor dtype")
    if not bool(torch.isfinite(theta).all().item()):
        raise RuntimeError(f"checkpoint {path} has nonfinite theta tensor")
    optimizer_state = payload.get("optimizer_state")
    if optimizer_state is not None and not isinstance(optimizer_state, dict):
        raise RuntimeError(f"checkpoint {path} has invalid optimizer state")
    optimizer_phase = payload.get("optimizer_phase")
    if optimizer_phase is not None and not isinstance(optimizer_phase, str):
        raise RuntimeError(f"checkpoint {path} has invalid optimizer phase")
    checkpoint_status_dict(path, payload)
    checkpoint_string_list(path, "family_names", payload["family_names"])
    checkpoint_string_list(path, "species_names", payload["species_names"])
    return payload


def _require_config_identity_fields(path: Path, config: dict[str, Any]) -> None:
    missing = sorted(key for key in _CHECKPOINT_CONFIG_IDENTITY_KEYS if key not in config)
    if missing:
        raise RuntimeError(
            f"checkpoint {path} config is missing identity field(s): "
            f"{', '.join(missing)}"
        )


def _checkpoint_version(path: Path, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeError(
            f"checkpoint {path} has unsupported version {value!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    return int(value)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    payload = _safe_torch_load(path, map_location=map_location, artifact="checkpoint")
    return _validate_checkpoint_payload(payload, path)


def restore_model_theta(model: Any, payload: dict[str, Any]) -> None:
    theta = payload["theta"].to(device=model.theta.device, dtype=model.theta.dtype)
    if tuple(theta.shape) != tuple(model.theta.shape):
        raise RuntimeError(
            f"checkpoint theta shape {tuple(theta.shape)} does not match model "
            f"shape {tuple(model.theta.shape)}"
        )
    with torch.no_grad():
        model.theta.copy_(theta)
        model.theta.grad = None
    model.clear()
