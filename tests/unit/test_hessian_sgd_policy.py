from __future__ import annotations

from dataclasses import is_dataclass
from types import SimpleNamespace

from gpurec.workflow._hessian_sgd_policy import (
    HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,
    HessianSGDLineSearchDecision,
    hessian_sgd_active_clade_count,
    hessian_sgd_line_search_decision,
    hessian_sgd_should_carry_warmup_hessian,
    hessian_sgd_should_skip_full_after_warmup,
)


def test_hessian_sgd_line_search_decision_increments_without_activation():
    decision = hessian_sgd_line_search_decision(
        batchwise_hessian_sgd=True,
        phase="hessian-sgd",
        active_objective_scope=True,
        line_search_active=False,
        full_stage_plateau=False,
        accepted_fraction=0.5,
        loss_rejected_rows=1.0,
        current_low_accept_steps=0,
        solver_stage="full",
        stable_loss_steps=0,
        active_clade_count=10,
    )

    assert is_dataclass(HessianSGDLineSearchDecision)
    assert decision == HessianSGDLineSearchDecision(
        low_accept_steps=1,
        activate=False,
    )


def test_hessian_sgd_line_search_decision_activates_at_patience():
    decision = hessian_sgd_line_search_decision(
        batchwise_hessian_sgd=True,
        phase="hessian-sgd",
        active_objective_scope=True,
        line_search_active=False,
        full_stage_plateau=False,
        accepted_fraction=0.5,
        loss_rejected_rows=1.0,
        current_low_accept_steps=1,
        solver_stage="full",
        stable_loss_steps=0,
        active_clade_count=10,
    )

    assert decision == HessianSGDLineSearchDecision(
        low_accept_steps=2,
        activate=True,
    )


def test_hessian_sgd_line_search_decision_resets_without_low_acceptance():
    for accepted_fraction, loss_rejected_rows in ((None, 1.0), (0.5, 0.0), (0.9, 1.0)):
        decision = hessian_sgd_line_search_decision(
            batchwise_hessian_sgd=True,
            phase="hessian-sgd",
            active_objective_scope=True,
            line_search_active=False,
            full_stage_plateau=False,
            accepted_fraction=accepted_fraction,
            loss_rejected_rows=loss_rejected_rows,
            current_low_accept_steps=1,
            solver_stage="full",
            stable_loss_steps=0,
            active_clade_count=10,
        )

        assert decision == HessianSGDLineSearchDecision(
            low_accept_steps=0,
            activate=False,
        )


def test_hessian_sgd_line_search_decision_preserves_counter_when_ineligible():
    decision = hessian_sgd_line_search_decision(
        batchwise_hessian_sgd=False,
        phase="adam-fd-newton",
        active_objective_scope=False,
        line_search_active=True,
        full_stage_plateau=True,
        accepted_fraction=0.0,
        loss_rejected_rows=10.0,
        current_low_accept_steps=3,
        solver_stage="full",
        stable_loss_steps=0,
        active_clade_count=10,
    )

    assert decision == HessianSGDLineSearchDecision(
        low_accept_steps=3,
        activate=False,
    )


def test_hessian_sgd_line_search_decision_suppresses_huge_plateau_activation():
    decision = hessian_sgd_line_search_decision(
        batchwise_hessian_sgd=True,
        phase="hessian-sgd",
        active_objective_scope=True,
        line_search_active=False,
        full_stage_plateau=False,
        accepted_fraction=0.5,
        loss_rejected_rows=1.0,
        current_low_accept_steps=1,
        solver_stage="full",
        stable_loss_steps=1,
        active_clade_count=HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,
    )

    assert decision == HessianSGDLineSearchDecision(
        low_accept_steps=2,
        activate=False,
    )


def test_hessian_sgd_active_clade_count_accepts_missing_metadata():
    assert hessian_sgd_active_clade_count(None) == 0
    assert hessian_sgd_active_clade_count(SimpleNamespace(clade_count=None)) == 0
    assert hessian_sgd_active_clade_count(SimpleNamespace(clade_count=7)) == 7


def test_hessian_sgd_skip_full_after_warmup_policy():
    assert hessian_sgd_should_skip_full_after_warmup(
        batchwise_hessian_sgd=True,
        phase="hessian-sgd",
        line_search_active=False,
        active_clade_count=HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,
    )
    assert not hessian_sgd_should_skip_full_after_warmup(
        batchwise_hessian_sgd=True,
        phase="hessian-sgd",
        line_search_active=True,
        active_clade_count=HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,
    )
    assert not hessian_sgd_should_skip_full_after_warmup(
        batchwise_hessian_sgd=True,
        phase="adam-fd-newton",
        line_search_active=False,
        active_clade_count=HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,
    )
    assert not hessian_sgd_should_skip_full_after_warmup(
        batchwise_hessian_sgd=True,
        phase="hessian-sgd",
        line_search_active=False,
        active_clade_count=HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES - 1,
    )


def test_hessian_sgd_carry_warmup_hessian_policy_requires_state():
    assert hessian_sgd_should_carry_warmup_hessian(
        batchwise_hessian_sgd=True,
        phase="hessian-sgd",
        line_search_active=False,
        active_clade_count=HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,
        has_hessian_state=True,
    )
    assert not hessian_sgd_should_carry_warmup_hessian(
        batchwise_hessian_sgd=True,
        phase="hessian-sgd",
        line_search_active=False,
        active_clade_count=HESSIAN_SGD_NO_LINE_REFRESH_MIN_CLADES,
        has_hessian_state=False,
    )
