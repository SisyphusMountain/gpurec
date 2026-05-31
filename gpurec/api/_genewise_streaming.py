"""Genewise full-streaming helpers for :class:`gpurec.api.model.GeneReconModel`."""
from __future__ import annotations

from typing import Any

import torch

from ._tensor_validation import (
    _validate_genewise_gradient_matrix,
    _validate_genewise_loss_vector,
)
from ._uniform_evaluator import (
    evaluate_resident_static_state as _evaluate_static_state_impl,
)

_evaluate_static_state = _evaluate_static_state_impl


def full_genewise_nll_and_grad(
    model: Any,
    *,
    need_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Stream genewise per-family NLLs and optional independent gradients."""
    if model._mode != "genewise":
        raise ValueError(
            "full_genewise_nll_and_grad() is only valid in genewise mode"
        )

    values = torch.empty(
        (model.n_families,),
        device=model.theta.device,
        dtype=model.theta.dtype,
    )
    grad_total = torch.zeros_like(model.theta) if need_grad else None

    if not model._batched_resident:
        loss, grad = _evaluate_static_state(
            model._active_static(),
            model.theta,
            need_grad=need_grad,
            per_family=True,
        )
        loss = _validate_genewise_loss_vector(
            "genewise per-family NLL",
            loss,
            family_count=model.n_families,
        )
        values.copy_(loss.to(device=values.device, dtype=values.dtype))
        if need_grad:
            if grad is None or grad_total is None:
                raise RuntimeError("internal error: missing genewise gradient")
            grad = _validate_genewise_gradient_matrix(
                "genewise gradient",
                grad,
                expected_shape=tuple(int(dim) for dim in grad_total.shape),
            )
            grad_total.copy_(grad.to(device=grad_total.device, dtype=grad_total.dtype))
        return values, grad_total

    previous_batch = model.current_batch_index
    try:
        for batch_idx, metadata in enumerate(model.batch_metadata):
            model.select_batch(batch_idx)
            static = model._active_static()
            theta_batch = model._active_theta()
            batch_values, batch_grad = _evaluate_static_state(
                static,
                theta_batch,
                need_grad=need_grad,
                per_family=True,
            )
            batch_values = _validate_genewise_loss_vector(
                "genewise batch per-family NLL",
                batch_values,
                family_count=len(metadata.family_indices),
            )
            idx = torch.as_tensor(
                metadata.family_indices,
                dtype=torch.long,
                device=values.device,
            )
            values.index_copy_(
                0,
                idx,
                batch_values.to(device=values.device, dtype=values.dtype),
            )
            if need_grad:
                if batch_grad is None or grad_total is None:
                    raise RuntimeError("internal error: missing genewise batch gradient")
                batch_grad = _validate_genewise_gradient_matrix(
                    "genewise batch gradient",
                    batch_grad,
                    expected_shape=tuple(int(dim) for dim in theta_batch.shape),
                )
                grad_total.index_copy_(
                    0,
                    idx.to(device=grad_total.device),
                    batch_grad.to(device=grad_total.device, dtype=grad_total.dtype),
                )
    finally:
        model.select_batch(previous_batch)
    return values, grad_total


def full_nll_per_family(model: Any) -> torch.Tensor:
    """Return no-grad per-family NLL for every genewise family."""
    if model._mode != "genewise":
        raise ValueError(
            "full_nll_per_family() is only valid in genewise mode; use "
            "forward(reduce='per_family') under torch.no_grad() for "
            "shared-theta diagnostic values."
        )
    values, _grad = model.full_genewise_nll_and_grad(need_grad=False)
    return values
