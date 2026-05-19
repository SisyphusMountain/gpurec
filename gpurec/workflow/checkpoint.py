from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import RunConfig


CHECKPOINT_VERSION = 1


def _family_names(model: Any) -> list[str]:
    if hasattr(model, "family_names"):
        return list(model.family_names)
    dataset = getattr(model, "_dataset", None)
    if dataset is not None and hasattr(dataset, "family_names"):
        return list(dataset.family_names)
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
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "step": int(step),
        "next_step": int(step) + 1,
        "config": config.to_dict(),
        "theta": model.theta.detach().cpu(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "status": status,
        "last_row": row,
        "family_names": _family_names(model),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


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
