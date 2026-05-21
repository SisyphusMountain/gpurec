from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gpurec.api import model as api_model
from gpurec.api.model import GeneReconModel
from gpurec.core.gradient_accumulator import GradientAccumulator
from gpurec.core.parameter_layout import ParameterLayout


@pytest.mark.parametrize(
    ("mode", "species_count", "family_count", "theta_shape"),
    [
        ("global", 2, 4, (3,)),
        ("specieswise", 2, 4, (2, 3)),
    ],
)
def test_shared_layouts_add_complete_gradients(
    mode: str,
    species_count: int,
    family_count: int,
    theta_shape: tuple[int, ...],
) -> None:
    layout = ParameterLayout.for_mode(
        mode,
        species_count=species_count,
        family_count=family_count,
    )
    theta = torch.zeros(theta_shape, dtype=torch.float64)
    first = torch.arange(theta.numel(), dtype=torch.float32).reshape(theta_shape)
    second = torch.full(theta_shape, 2.0, dtype=torch.float32)

    accumulator = GradientAccumulator.zeros_like(layout, theta)
    accumulator.add(first).add(second)

    assert accumulator.tensor.dtype == theta.dtype
    assert accumulator.tensor.device == theta.device
    torch.testing.assert_close(
        accumulator.result(),
        first.to(dtype=theta.dtype) + second.to(dtype=theta.dtype),
    )


def test_genewise_layout_accumulates_active_family_rows() -> None:
    layout = ParameterLayout.for_mode(
        "genewise",
        species_count=2,
        family_count=5,
        family_indices=[4, 1],
    )
    theta = torch.zeros(5, 3, dtype=torch.float32)
    contribution = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )

    accumulator = GradientAccumulator.zeros_like(layout, theta)
    accumulator.add(contribution)

    expected = torch.zeros_like(theta)
    expected[4] += contribution[0]
    expected[1] += contribution[1]
    torch.testing.assert_close(accumulator.result(), expected)


def test_genewise_layout_sums_duplicate_explicit_indices() -> None:
    layout = ParameterLayout.for_mode(
        "genewise",
        species_count=2,
        family_count=5,
    )
    theta = torch.zeros(5, 3, dtype=torch.float32)
    contribution = torch.tensor(
        [
            [1.0, 0.5, 0.25],
            [2.0, 1.5, 1.25],
            [4.0, 3.5, 3.25],
        ],
        dtype=torch.float32,
    )

    accumulator = GradientAccumulator.zeros_like(layout, theta)
    accumulator.add(contribution, family_indices=[2, 2, 4])

    expected = torch.zeros_like(theta)
    expected[2] += contribution[0] + contribution[1]
    expected[4] += contribution[2]
    torch.testing.assert_close(accumulator.result(), expected)


def test_accumulator_preserves_target_dtype_and_device() -> None:
    layout = ParameterLayout.for_mode(
        "genewise",
        species_count=3,
        family_count=4,
    )
    theta = torch.zeros(4, 3, dtype=torch.float64)
    contribution = torch.ones(2, 3, dtype=torch.float32)

    accumulator = GradientAccumulator.zeros_like(layout, theta)
    accumulator.add(contribution, family_indices=torch.tensor([0, 3]))

    assert accumulator.result().dtype == theta.dtype
    assert accumulator.result().device == theta.device
    expected = torch.zeros_like(theta)
    expected[0] += 1.0
    expected[3] += 1.0
    torch.testing.assert_close(accumulator.result(), expected)


def test_accumulator_rejects_invalid_public_shape() -> None:
    layout = ParameterLayout.for_mode(
        "global",
        species_count=2,
        family_count=3,
    )

    with pytest.raises(ValueError, match="theta shape"):
        GradientAccumulator.zeros_like(layout, torch.zeros(2))


def test_shared_accumulator_rejects_family_indices() -> None:
    layout = ParameterLayout.for_mode(
        "global",
        species_count=2,
        family_count=3,
    )
    accumulator = GradientAccumulator.zeros(layout)

    with pytest.raises(ValueError, match="only valid for genewise"):
        accumulator.add(torch.zeros(3), family_indices=[0])


@pytest.mark.parametrize(
    ("layout", "contribution", "family_indices"),
    [
        (
            ParameterLayout.for_mode("global", species_count=2, family_count=3),
            torch.zeros(2),
            None,
        ),
        (
            ParameterLayout.for_mode("specieswise", species_count=2, family_count=3),
            torch.zeros(3, 3),
            None,
        ),
        (
            ParameterLayout.for_mode("genewise", species_count=2, family_count=4),
            torch.zeros(3, 3),
            [0, 1],
        ),
    ],
)
def test_accumulator_rejects_contribution_shape_mismatch(
    layout: ParameterLayout,
    contribution: torch.Tensor,
    family_indices: list[int] | None,
) -> None:
    accumulator = GradientAccumulator.zeros(layout)

    with pytest.raises(ValueError, match="gradient contribution shape"):
        accumulator.add(contribution, family_indices=family_indices)


def test_accumulator_rejects_out_of_range_family_index() -> None:
    layout = ParameterLayout.for_mode(
        "genewise",
        species_count=2,
        family_count=4,
    )
    accumulator = GradientAccumulator.zeros(layout)

    with pytest.raises(IndexError, match="out of range"):
        accumulator.add(torch.zeros(1, 3), family_indices=[4])


def test_stream_full_batches_uses_accumulator_for_genewise_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(GeneReconModel)
    theta = torch.zeros(4, 3, dtype=torch.float64)

    object.__setattr__(model, "_batched_resident", True)
    object.__setattr__(model, "_mode", "genewise")
    object.__setattr__(
        model,
        "_dataset",
        SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float64,
            S=2,
            families=[object(), object(), object(), object()],
        ),
    )
    object.__setattr__(
        model,
        "_batch_specs",
        [
            SimpleNamespace(family_indices=(2, 0)),
            SimpleNamespace(family_indices=(1, 3)),
        ],
    )
    object.__setattr__(
        model,
        "_ensure_batch_static",
        lambda batch_idx: f"static-{batch_idx}",
    )

    def fake_evaluate_static_state(
        static: str,
        theta_batch: torch.Tensor,
        *,
        need_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert need_grad is True
        if static == "static-0":
            torch.testing.assert_close(theta_batch, theta[[2, 0]])
            return (
                torch.tensor(1.25, dtype=torch.float32),
                torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=torch.float32,
                ),
            )
        torch.testing.assert_close(theta_batch, theta[[1, 3]])
        return (
            torch.tensor(2.5, dtype=torch.float32),
            torch.tensor(
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                dtype=torch.float32,
            ),
        )

    monkeypatch.setattr(api_model, "_evaluate_static_state", fake_evaluate_static_state)

    loss, grad = GeneReconModel._stream_full_batches(model, theta, need_grad=True)

    assert grad is not None
    assert grad.dtype == theta.dtype
    torch.testing.assert_close(loss, torch.tensor(3.75, dtype=torch.float64))
    torch.testing.assert_close(
        grad,
        torch.tensor(
            [
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [1.0, 2.0, 3.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=torch.float64,
        ),
    )
