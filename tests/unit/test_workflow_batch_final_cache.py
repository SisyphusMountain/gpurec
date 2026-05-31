from __future__ import annotations

from types import SimpleNamespace

import torch

from gpurec.workflow._batch_final_cache import BatchFinalCache


def _model(n_families: int = 4) -> SimpleNamespace:
    theta = torch.zeros((n_families, 2), dtype=torch.float64, requires_grad=True)
    return SimpleNamespace(n_families=n_families, theta=theta)


def test_batch_final_cache_allocates_model_shaped_buffers() -> None:
    model = _model()

    cache = BatchFinalCache.create(model)

    assert cache.loss.shape == (model.n_families,)
    assert cache.loss.device == model.theta.device
    assert cache.loss.dtype == model.theta.dtype
    assert cache.grad.shape == model.theta.shape
    assert cache.grad.device == model.theta.device
    assert cache.grad.dtype == model.theta.dtype
    assert cache.ready.shape == (model.n_families,)
    assert cache.ready.device == model.theta.device
    assert cache.ready.dtype is torch.bool
    assert not bool(cache.ready.any().item())


def test_batch_final_cache_records_active_rows_and_returns_clones() -> None:
    model = _model()
    cache = BatchFinalCache.create(model)
    loss_vec = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=model.theta.dtype)
    grad = torch.arange(8, dtype=model.theta.dtype).reshape_as(model.theta)
    model.theta.grad = grad.clone()

    cache.cache(
        model=model,
        loss_vec=loss_vec,
        active_indices=torch.tensor([1, 3]),
    )

    assert cache.ready.tolist() == [False, True, False, True]
    assert torch.equal(
        cache.loss.index_select(0, torch.tensor([1, 3])),
        loss_vec[[1, 3]],
    )
    assert torch.equal(cache.grad.index_select(0, torch.tensor([1, 3])), grad[[1, 3]])
    assert cache.cached_final_result() is None

    loss_vec[1] = -1.0
    model.theta.grad[1, 0] = -1.0
    assert cache.loss[1].item() == 20.0
    assert cache.grad[1, 0].item() == 2.0

    remaining_loss = torch.tensor([11.0, 22.0, 33.0, 44.0], dtype=model.theta.dtype)
    remaining_grad = torch.arange(100, 108, dtype=model.theta.dtype).reshape_as(
        model.theta
    )
    model.theta.grad = remaining_grad.clone()
    cache.cache(
        model=model,
        loss_vec=remaining_loss,
        active_indices=torch.tensor([0, 2]),
    )

    cached_result = cache.cached_final_result()
    assert cached_result is not None
    final_loss, final_grad = cached_result
    assert torch.equal(
        final_loss,
        torch.tensor([11.0, 20.0, 33.0, 40.0], dtype=model.theta.dtype),
    )
    assert torch.equal(final_grad[0], remaining_grad[0])
    assert torch.equal(final_grad[1], grad[1])
    assert torch.equal(final_grad[2], remaining_grad[2])
    assert torch.equal(final_grad[3], grad[3])

    final_loss[0] = -5.0
    final_grad[0, 0] = -5.0
    assert cache.loss[0].item() == 11.0
    assert cache.grad[0, 0].item() == 100.0


def test_batch_final_cache_invalidates_only_selected_indices() -> None:
    model = _model()
    cache = BatchFinalCache.create(model)
    model.theta.grad = torch.ones_like(model.theta)
    cache.cache(
        model=model,
        loss_vec=torch.arange(4, dtype=model.theta.dtype),
        active_indices=torch.arange(4),
    )

    cache.invalidate([1, 3])

    assert cache.ready.tolist() == [True, False, True, False]
    assert cache.cached_final_result() is None
