from __future__ import annotations

import pytest

from gpurec.core.forward import (
    _pi_output_intent,
    pi_export_state_request,
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
    intent = _pi_output_intent(
        return_original=return_original,
        return_root_rows=return_root_rows,
    )

    assert intent.name == expected_name
    assert intent.emit_original_pi is expected_original
    assert intent.emit_root_rows is expected_root_rows
    assert intent.emit_wave_ordered_pi is expected_wave_ordered
    assert intent.retain_saved_state is expected_saved_state
    assert intent.can_skip_final_pibar is (not expected_saved_state)


def test_pi_forward_requests_name_internal_output_intent():
    assert (
        pi_training_state_request().intent.name
        == "training_or_wave_export_state"
    )
    assert pi_root_row_loss_request().intent.name == "root_row_loss_only"
    assert (
        pi_export_state_request(original_order=True).intent.name
        == "export_original_and_wave_rows"
    )
    assert (
        pi_export_state_request(original_order=False).intent.name
        == "training_or_wave_export_state"
    )
