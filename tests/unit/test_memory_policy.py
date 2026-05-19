import pytest
import torch

from gpurec.core.memory_policy import (
    choose_uniform_pipeline_policy,
    estimate_chunk_payload_bytes,
    proposal0_memory_gate,
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


def test_chunk_payload_uses_packed_plan_for_clade_first_fit():
    clades = [100, 1, 100, 1]
    S = 3
    itemsize = torch.empty((), dtype=torch.float32).element_size()

    expected_C = 200
    expected_W = 200
    expected = (
        3 * expected_C * S * itemsize
        + 10 * expected_W * S * itemsize
        + expected_C * itemsize * 8
    )
    assert (
        estimate_chunk_payload_bytes(
            clades,
            S,
            torch.float32,
            family_chunk_size=2,
            max_wave_size=1024,
            clade_budget=200,
            batch_packing="clade_first_fit",
        )
        == expected
    )


def test_policy_budget_uses_packed_clade_first_fit_plan(monkeypatch):
    monkeypatch.setattr(
        "gpurec.core.memory_policy.cuda_memory_budget_bytes",
        lambda device=None: 10_000,
    )

    with pytest.raises(RuntimeError, match="no retained uniform pipeline policy"):
        choose_uniform_pipeline_policy(
            [100, 1, 100, 1],
            1,
            torch.float32,
            family_chunk_candidates=(2,),
            max_wave_candidates=(1024,),
            clade_budget=200,
            batch_packing="clade_first_fit",
        )


@pytest.mark.parametrize(
    "call, match",
    [
        (
            lambda: proposal0_wave_scratch_bytes(-1, 3, torch.float32),
            "W",
        ),
        (
            lambda: proposal0_wave_scratch_bytes(True, 3, torch.float32),
            "W",
        ),
        (
            lambda: uniform_training_dense_state_bytes(3, 0, torch.float32),
            "S",
        ),
        (
            lambda: uniform_training_dense_state_bytes(3, 1.5, torch.float32),
            "S",
        ),
        (
            lambda: uniform_training_payload_bytes(
                3,
                2,
                torch.float32,
                max_wave_rows=0,
            ),
            "max_wave_rows",
        ),
        (
            lambda: estimate_chunk_payload_bytes(
                [1],
                2,
                torch.float32,
                family_chunk_size=1,
                max_wave_size=0,
            ),
            "max_wave_size",
        ),
        (
            lambda: estimate_chunk_payload_bytes(
                [1],
                2,
                torch.float32,
                family_chunk_size=1.5,
                max_wave_size=1,
            ),
            "family_chunk_size",
        ),
        (
            lambda: proposal0_memory_gate(
                1,
                2,
                torch.float32,
                already_live_bytes=-1,
            ),
            "already_live_bytes",
        ),
        (
            lambda: choose_uniform_pipeline_policy(
                [1],
                2,
                torch.float32,
                family_chunk_candidates=(-1,),
            ),
            "family_chunk_candidates",
        ),
        (
            lambda: choose_uniform_pipeline_policy(
                [1],
                2,
                torch.float32,
                family_chunk_candidates=(True,),
            ),
            "family_chunk_candidates",
        ),
        (
            lambda: choose_uniform_pipeline_policy(
                [1],
                2,
                torch.float32,
                max_wave_candidates=(1.5,),
            ),
            "max_wave_candidates",
        ),
    ],
)
def test_memory_policy_rejects_invalid_dimensions(call, match):
    with pytest.raises(ValueError, match=match):
        call()
