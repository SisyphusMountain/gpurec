from __future__ import annotations

import pytest
import torch

from gpurec.core.forward import (
    PiForwardRequest,
    PiOutputIntent,
    _carry_forward_converged_root_trace,
    pi_export_state_request,
    pi_output_intent,
    pi_root_row_loss_request,
    pi_training_state_request,
)


@pytest.mark.parametrize(
    (
        "return_original",
        "return_root_rows",
        "expected_name",
        "expected_original",
        "expected_root_rows",
        "expected_wave_ordered",
        "expected_saved_state",
    ),
    (
        (
            False,
            False,
            "training_or_wave_export_state",
            False,
            False,
            True,
            True,
        ),
        (
            True,
            False,
            "export_original_and_wave_rows",
            True,
            False,
            True,
            True,
        ),
        (
            False,
            True,
            "root_row_loss_only",
            False,
            True,
            False,
            False,
        ),
        (
            True,
            True,
            "legacy_original_root_rows",
            True,
            True,
            False,
            False,
        ),
    ),
)
def test_pi_output_intent_maps_legacy_booleans_to_internal_contract(
    return_original: bool,
    return_root_rows: bool,
    expected_name: str,
    expected_original: bool,
    expected_root_rows: bool,
    expected_wave_ordered: bool,
    expected_saved_state: bool,
):
    intent = pi_output_intent(
        return_original=return_original,
        return_root_rows=return_root_rows,
    )

    assert isinstance(intent, PiOutputIntent)
    assert intent.name == expected_name
    assert intent.emit_original_pi is expected_original
    assert intent.emit_root_rows is expected_root_rows
    assert intent.emit_wave_ordered_pi is expected_wave_ordered
    assert intent.retain_saved_state is expected_saved_state
    assert intent.can_skip_final_pibar is (not expected_saved_state)


def test_pi_forward_requests_name_internal_output_intent():
    training = pi_training_state_request()
    root_rows = pi_root_row_loss_request()
    export_original = pi_export_state_request(original_order=True)
    export_wave = pi_export_state_request(original_order=False)

    for request in (training, root_rows, export_original, export_wave):
        assert isinstance(request, PiForwardRequest)

    assert training.intent.name == "training_or_wave_export_state"
    assert root_rows.intent.name == "root_row_loss_only"
    assert export_original.intent.name == "export_original_and_wave_rows"
    assert export_wave.intent.name == "training_or_wave_export_state"


def test_converged_root_trace_carries_last_value_through_tail_rows():
    trace = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [-torch.inf, 30.0, 300.0],
            [-torch.inf, -torch.inf, -torch.inf],
        ]
    )
    roots_by_wave = [
        (torch.tensor([4]), torch.tensor([0])),
        (torch.tensor([8, 9]), torch.tensor([1, 2])),
    ]

    result = _carry_forward_converged_root_trace(
        trace,
        roots_by_wave,
        pi_wave_iterations=[2, 3],
        fixed_iters=4,
    )

    assert result is trace
    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [2.0, 30.0, 300.0],
                [2.0, 30.0, 300.0],
            ]
        ),
    )


def test_converged_root_trace_leaves_full_iteration_columns_unchanged():
    trace = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    roots_by_wave = [(torch.tensor([3, 4]), torch.tensor([0, 1]))]

    result = _carry_forward_converged_root_trace(
        trace.clone(),
        roots_by_wave,
        pi_wave_iterations=[4],
        fixed_iters=4,
    )

    torch.testing.assert_close(result, trace)
