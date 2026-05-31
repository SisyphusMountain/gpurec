"""Shared closure, loss, and gradient helpers for internal optimizers."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


LossClosure = Callable[[], Tensor]


def _scalar_loss_tensor(loss: object, owner: str, source: str) -> Tensor:
    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise ValueError(f"{owner} {source} must return a scalar Tensor")
    return loss.detach().reshape(())


def scalar_loss_tensor(loss: object, owner: str) -> Tensor:
    """Validate and detach a scalar closure loss tensor."""
    return _scalar_loss_tensor(loss, owner, "closure")


def loss_vector_tensor(loss: object, batch_size: int, owner: str) -> Tensor:
    """Validate and detach a row-wise loss vector."""
    if not torch.is_tensor(loss):
        raise TypeError(f"{owner} closure must return a Tensor")
    if loss.numel() != batch_size:
        raise ValueError(
            f"{owner} closure must return one loss per parameter row; "
            f"got shape {tuple(loss.shape)} for batch size {batch_size}"
        )
    return loss.detach().reshape(batch_size)


def flat_grad(
    param: Tensor,
    flat_like: Tensor,
    owner: str,
    *,
    row_batch_size: int | None = None,
) -> Tensor:
    """Return a detached dense flattened gradient matching optimizer layout."""
    grad = param.grad
    if grad is None:
        return torch.zeros_like(flat_like)
    if grad.is_sparse:
        grad = grad.to_dense()
    if torch.is_complex(grad):
        raise TypeError(f"{owner} only supports real-valued gradients")
    if row_batch_size is not None:
        return grad.detach().reshape(int(row_batch_size), -1)
    return grad.detach().reshape_as(flat_like)


def evaluate_scalar_with_grad(
    param: Tensor,
    flat_like: Tensor,
    closure: LossClosure,
    owner: str,
) -> tuple[Tensor, Tensor]:
    """Evaluate a scalar closure with gradients enabled."""
    with torch.enable_grad():
        loss = closure()
    return scalar_loss_tensor(loss, owner), flat_grad(param, flat_like, owner)


def evaluate_scalar_loss(
    closure: LossClosure,
    loss_closure: LossClosure | None,
    owner: str,
) -> Tensor:
    """Evaluate a scalar loss-only probe closure."""
    if loss_closure is None:
        with torch.enable_grad():
            loss = closure()
    else:
        with torch.no_grad():
            loss = loss_closure()
    return _scalar_loss_tensor(loss, owner, "loss closure")


def evaluate_vector_with_grad(
    param: Tensor,
    flat_like: Tensor,
    closure: LossClosure,
    batch_size: int,
    owner: str,
) -> tuple[Tensor, Tensor]:
    """Evaluate a row-wise closure with gradients enabled."""
    with torch.enable_grad():
        loss = closure()
    return (
        loss_vector_tensor(loss, batch_size, owner),
        flat_grad(param, flat_like, owner, row_batch_size=batch_size),
    )


def evaluate_vector_loss(
    closure: LossClosure,
    loss_closure: LossClosure | None,
    batch_size: int,
    owner: str,
) -> Tensor:
    """Evaluate a row-wise loss-only probe closure."""
    if loss_closure is None:
        with torch.enable_grad():
            loss = closure()
    else:
        with torch.no_grad():
            loss = loss_closure()
    return loss_vector_tensor(loss, batch_size, owner)


class ScalarClosureMixin:
    """Private scalar-optimizer closure methods."""

    _optimizer_name: str

    def _gather_flat_grad(self) -> Tensor:
        return flat_grad(self._param, self._flat_param(), self._optimizer_name)

    def _evaluate_with_grad(self, closure: LossClosure) -> tuple[Tensor, Tensor]:
        return evaluate_scalar_with_grad(
            self._param,
            self._flat_param(),
            closure,
            self._optimizer_name,
        )

    def _evaluate_loss(
        self,
        closure: LossClosure,
        loss_closure: LossClosure | None,
    ) -> Tensor:
        return evaluate_scalar_loss(closure, loss_closure, self._optimizer_name)


class VectorClosureMixin:
    """Private row-wise optimizer closure methods."""

    _optimizer_name: str

    def _gather_flat_grad(self) -> Tensor:
        return flat_grad(
            self._param,
            self._flat_param(),
            self._optimizer_name,
            row_batch_size=self._batch_size(),
        )

    def _loss_vector(self, loss: object) -> Tensor:
        return loss_vector_tensor(loss, self._batch_size(), self._optimizer_name)

    def _evaluate_with_grad(self, closure: LossClosure) -> tuple[Tensor, Tensor]:
        return evaluate_vector_with_grad(
            self._param,
            self._flat_param(),
            closure,
            self._batch_size(),
            self._optimizer_name,
        )

    def _evaluate_loss(
        self,
        closure: LossClosure,
        loss_closure: LossClosure | None,
    ) -> Tensor:
        return evaluate_vector_loss(
            closure,
            loss_closure,
            self._batch_size(),
            self._optimizer_name,
        )
