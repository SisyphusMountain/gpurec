from __future__ import annotations

import pytest
import torch

from gpurec.core.terms import gather_E_children


def test_gather_E_children_scatter_1d_child_values_into_parent_slots():
    E = torch.tensor([10.0, 20.0, 30.0])
    sp_P_idx = torch.tensor([0, 3])
    child_index = torch.tensor([1, 2])

    result = gather_E_children(E, sp_P_idx, child_index)

    expected = torch.tensor(
        [20.0, -float("inf"), -float("inf"), 30.0, -float("inf"), -float("inf")]
    )
    torch.testing.assert_close(result, expected)


def test_gather_E_children_scatter_2d_child_columns_into_parent_slots():
    E = torch.tensor([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
    sp_P_idx = torch.tensor([0, 3])
    child_index = torch.tensor([1, 2])

    result = gather_E_children(E, sp_P_idx, child_index)

    expected = torch.tensor(
        [
            [20.0, -float("inf"), -float("inf"), 30.0, -float("inf"), -float("inf")],
            [200.0, -float("inf"), -float("inf"), 300.0, -float("inf"), -float("inf")],
        ]
    )
    torch.testing.assert_close(result, expected)


def test_gather_E_children_rejects_unsupported_rank_with_shape():
    E = torch.zeros((1, 2, 3))
    sp_P_idx = torch.tensor([0, 3])
    child_index = torch.tensor([1, 2])

    with pytest.raises(ValueError, match=r"E must be 1D or 2D, got shape \(1, 2, 3\)"):
        gather_E_children(E, sp_P_idx, child_index)
