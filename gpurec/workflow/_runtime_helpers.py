from __future__ import annotations

from typing import Any

import torch

from gpurec.api.autograd import (
    _clear_pi_adjoint_runtime_cache,
    _commit_pi_adjoint_pending_cache,
    _discard_pi_adjoint_pending_cache,
)
from gpurec.api.model import GeneReconModel


def _tensor_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _is_single_value_tensor(value: object) -> bool:
    return torch.is_tensor(value) and value.numel() == 1


def _is_finite_tensor(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and bool(torch.isfinite(tensor).all().item())


def _clear_cuda_allocator_cache_if_needed(model: GeneReconModel) -> None:
    theta = getattr(model, "theta", None)
    if bool(getattr(theta, "is_cuda", False)) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _drop_cached_static_states_if_needed(model: GeneReconModel) -> None:
    drop_cached_static_states = getattr(
        model,
        "drop_cached_static_states",
        None,
    )
    if callable(drop_cached_static_states):
        drop_cached_static_states()
    else:
        model.clear()
    _clear_cuda_allocator_cache_if_needed(model)


def _clear_cached_solver_runtime_state(model: GeneReconModel) -> None:
    """Clear mutable solver warm-start state without rebuilding static layouts."""
    statics = getattr(model, "cached_static_states", None)
    if statics is not None:
        for static in list(statics):
            if hasattr(static, "warm_E"):
                static.warm_E = None
            _clear_pi_adjoint_runtime_cache(static)
            if hasattr(static, "last_solver_stats"):
                static.last_solver_stats = None
    else:
        model.clear()
    _clear_cuda_allocator_cache_if_needed(model)


def _cached_static_states(model: GeneReconModel) -> list[Any]:
    statics = getattr(model, "cached_static_states", None)
    if statics is None:
        return []
    return list(statics)


def _commit_pi_adjoint_pending_caches(model: GeneReconModel) -> int:
    return sum(
        1
        for static in _cached_static_states(model)
        if _commit_pi_adjoint_pending_cache(static)
    )


def _discard_pi_adjoint_pending_caches(model: GeneReconModel) -> int:
    return sum(
        1
        for static in _cached_static_states(model)
        if _discard_pi_adjoint_pending_cache(static)
    )
