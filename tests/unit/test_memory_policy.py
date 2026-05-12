import torch

from gpurec.core.memory_policy import (
    estimate_chunk_payload_bytes,
    proposal0_wave_scratch_bytes,
    uniform_training_dense_state_bytes,
    uniform_training_payload_bytes,
)


def test_uniform_memory_policy_payload_formulas_are_explicit():
    C = 11
    S = 7
    W = 5
    itemsize = torch.empty((), dtype=torch.float32).element_size()

    assert uniform_training_dense_state_bytes(C, S, torch.float32) == 3 * C * S * itemsize
    assert proposal0_wave_scratch_bytes(W, S, torch.float32) == 10 * W * S * itemsize

    small_state = C * itemsize * 8
    expected = 3 * C * S * itemsize + 10 * W * S * itemsize + small_state
    assert (
        uniform_training_payload_bytes(
            C,
            S,
            torch.float32,
            max_wave_rows=W,
        )
        == expected
    )


def test_chunk_payload_uses_largest_resident_chunk():
    clades = [10, 20, 30, 40, 50]
    S = 3
    itemsize = torch.empty((), dtype=torch.float64).element_size()

    # Chunk size 2 gives clade chunks [30, 70, 50], so C=70.
    expected_C = 70
    expected_W = 16
    expected = (
        3 * expected_C * S * itemsize
        + 10 * expected_W * S * itemsize
        + expected_C * itemsize * 8
    )
    assert (
        estimate_chunk_payload_bytes(
            clades,
            S,
            torch.float64,
            family_chunk_size=2,
            max_wave_size=16,
        )
        == expected
    )
