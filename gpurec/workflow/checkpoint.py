from __future__ import annotations

import pickle
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig


CHECKPOINT_VERSION = 1
_REQUIRED_CHECKPOINT_KEYS = {"version", "config", "theta", "step", "next_step"}


def _family_names(model: Any) -> list[str]:
    if hasattr(model, "family_names"):
        return list(model.family_names)
    return []


def _species_names(model: Any) -> list[str]:
    if hasattr(model, "species_names"):
        return list(model.species_names)
    return []


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
    if isinstance(checkpoint_config, dict):
        current_config = config.to_dict()
        for key in ("species_tree", "families_file", "mode", "start", "max_families"):
            if key not in checkpoint_config:
                continue
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

    checkpoint_family_names = payload.get("family_names")
    if checkpoint_family_names is not None:
        model_family_names = _family_names(model)
        if list(checkpoint_family_names) != model_family_names:
            raise RuntimeError(
                f"checkpoint {checkpoint_path} is incompatible with current run: "
                "family_names differ"
            )

    checkpoint_species_names = payload.get("species_names")
    if checkpoint_species_names is not None:
        model_species_names = _species_names(model)
        if list(checkpoint_species_names) != model_species_names:
            raise RuntimeError(
                f"checkpoint {checkpoint_path} is incompatible with current run: "
                "species_names differ"
            )


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
        "theta": model.theta.detach().cpu(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "optimizer_phase": optimizer_phase,
        "status": status,
        "last_row": row,
        "family_names": _family_names(model),
        "species_names": _species_names(model),
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
    step = _checkpoint_int(path, "step", payload["step"])
    next_step = _checkpoint_int(path, "next_step", payload["next_step"])
    if next_step not in {step, step + 1}:
        raise RuntimeError(f"checkpoint {path} has inconsistent progress metadata")
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
    status = payload.get("status")
    if status is not None and not isinstance(status, dict):
        raise RuntimeError(f"checkpoint {path} has invalid status metadata")
    family_names = payload.get("family_names")
    if family_names is not None and not isinstance(family_names, list):
        raise RuntimeError(f"checkpoint {path} has invalid family metadata")
    species_names = payload.get("species_names")
    if species_names is not None and not isinstance(species_names, list):
        raise RuntimeError(f"checkpoint {path} has invalid species metadata")
    return payload


def _checkpoint_version(path: Path, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeError(
            f"checkpoint {path} has unsupported version {value!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    return int(value)


def _checkpoint_int(path: Path, key: str, value: Any) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"checkpoint {path} has invalid {key}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        raw = float(value)
        if not raw.is_integer():
            raise RuntimeError(f"checkpoint {path} has invalid {key}")
        number = int(raw)
    else:
        raise RuntimeError(f"checkpoint {path} has invalid {key}")
    if number < 0:
        raise RuntimeError(f"checkpoint {path} has invalid {key}")
    return number


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    payload = _safe_torch_load(path, map_location=map_location, artifact="checkpoint")
    return _validate_checkpoint_payload(payload, path)


def load_checkpoint_config(path: str | Path) -> RunConfig:
    payload = load_checkpoint(path, map_location="cpu")
    return RunConfig.from_dict(payload["config"])


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
