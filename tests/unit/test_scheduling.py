from __future__ import annotations

import pytest
import torch

from gpurec.core.scheduling import compute_clade_waves


@pytest.mark.parametrize(
    "helpers",
    (
        {},
        {"phased_waves": [[0]]},
        {"phased_phases": [1]},
    ),
)
def test_compute_clade_waves_requires_cpp_phased_metadata(helpers):
    with pytest.raises(RuntimeError, match="must include C\\+\\+ phased waves"):
        compute_clade_waves(helpers)


def test_compute_clade_waves_normalizes_tensor_waves_and_phases():
    helpers = {
        "phased_waves": [torch.tensor([0, 2]), [3]],
        "phased_phases": [torch.tensor(1), 3],
    }

    waves, phases = compute_clade_waves(helpers)

    assert waves == [[0, 2], [3]]
    assert phases == [1, 3]
    assert all(isinstance(phase, int) for phase in phases)


def test_compute_clade_waves_chunks_oversized_waves_and_repeats_phase():
    helpers = {
        "phased_waves": [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([9])],
        "phased_phases": [torch.tensor(2), torch.tensor(3)],
    }

    waves, phases = compute_clade_waves(helpers, max_wave_size=2)

    assert waves == [[0, 1], [2, 3], [4], [9]]
    assert phases == [2, 2, 2, 3]
