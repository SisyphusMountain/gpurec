from __future__ import annotations

import pytest
import torch

from gpurec.core.species import (
    species_parent_from_helpers,
    species_wave_topology,
    uniform_ancestors_t_from_topology,
)


def _species_helpers() -> dict[str, torch.Tensor | int]:
    return {
        "S": 5,
        "s_P_indexes": torch.tensor([0, 1, 5, 6]),
        "s_C12_indexes": torch.tensor([1, 3, 2, 4]),
    }


def test_species_parent_from_helpers_builds_parent_vector_with_requested_dtype():
    parent = species_parent_from_helpers(_species_helpers(), dtype=torch.int32)

    assert parent.dtype == torch.int32
    torch.testing.assert_close(parent, torch.tensor([-1, 0, 0, 1, 1], dtype=torch.int32))


def test_species_wave_topology_builds_child_parent_and_compact_level_tensors():
    topology = species_wave_topology(_species_helpers(), device=torch.device("cpu"))

    torch.testing.assert_close(topology["sp_child1_cpu"], torch.tensor([1, 3, 5, 5, 5]))
    torch.testing.assert_close(topology["sp_child2_cpu"], torch.tensor([2, 4, 5, 5, 5]))
    torch.testing.assert_close(topology["sp_parent_cpu"], torch.tensor([-1, 0, 0, 1, 1]))
    assert topology["max_ancestor_depth"] == 3
    torch.testing.assert_close(topology["compact_level_ptr"], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(topology["compact_level_parents"], torch.tensor([1, 0], dtype=torch.int32))
    torch.testing.assert_close(topology["compact_level_child1"], torch.tensor([3, 1], dtype=torch.int32))
    torch.testing.assert_close(topology["compact_level_child2"], torch.tensor([4, 2], dtype=torch.int32))


def test_species_wave_topology_reuses_complete_device_cache():
    helpers = _species_helpers()

    first = species_wave_topology(helpers, device="cpu")
    second = species_wave_topology(helpers, device=torch.device("cpu"))

    assert second is first


def test_species_helpers_reject_mismatched_or_cyclic_parent_metadata():
    mismatched = {
        "S": 2,
        "s_P_indexes": torch.tensor([0]),
        "s_C12_indexes": torch.tensor([1, 0]),
    }
    with pytest.raises(ValueError, match="must have the same length"):
        species_wave_topology(mismatched, device="cpu")

    cyclic = {
        "S": 2,
        "s_P_indexes": torch.tensor([0, 1]),
        "s_C12_indexes": torch.tensor([1, 0]),
    }
    with pytest.raises(RuntimeError, match="Cycle detected"):
        species_wave_topology(cyclic, device="cpu")
    with pytest.raises(RuntimeError, match="Cycle detected"):
        uniform_ancestors_t_from_topology(cyclic, device="cpu", dtype=torch.float32)


def test_uniform_ancestors_t_from_topology_builds_dense_reference_and_cache():
    helpers = _species_helpers()

    ancestors_t = uniform_ancestors_t_from_topology(
        helpers,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    cached = uniform_ancestors_t_from_topology(helpers, device="cpu", dtype=torch.float64)

    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    assert ancestors_t.is_sparse
    assert ancestors_t.dtype == torch.float64
    torch.testing.assert_close(ancestors_t.to_dense(), expected)
    assert cached is ancestors_t
