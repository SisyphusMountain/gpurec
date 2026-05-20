from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig


CHECKPOINT_VERSION = 1
_REQUIRED_CHECKPOINT_KEYS = {"version", "config", "theta"}


def _family_names(model: Any) -> list[str]:
    if hasattr(model, "family_names"):
        return list(model.family_names)
    return []


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
    except (pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
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
    try:
        checkpoint_version = int(payload["version"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"checkpoint {path} has unsupported version {payload['version']!r}; "
            f"expected {CHECKPOINT_VERSION}"
        ) from exc
    if checkpoint_version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"checkpoint {path} has unsupported version {payload['version']!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    if not isinstance(payload["config"], dict):
        raise RuntimeError(f"checkpoint {path} has invalid config metadata")
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
    return payload


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
