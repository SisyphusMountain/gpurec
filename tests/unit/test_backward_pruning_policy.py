import torch

from gpurec.core.backward_pruning_policy import (
    backward_pruning_policy,
    inactive_wave_accounting,
)


def _reference_active_mask(rhs: torch.Tensor, policy) -> torch.Tensor:
    row_absmax = rhs.abs().amax(dim=1)
    if policy.active_mask_strict_gt:
        return row_absmax > policy.active_mask_threshold
    return row_absmax >= policy.active_mask_threshold


def test_default_policy_avoids_cpu_wave_skips_but_keeps_device_masks(monkeypatch):
    monkeypatch.delenv("GPUREC_BACKWARD_NO_CPU_PRUNING", raising=False)
    monkeypatch.delenv("GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO", raising=False)

    policy = backward_pruning_policy(use_pruning=True, pruning_threshold=1e-6)

    assert policy.no_cpu_pruning is True
    assert policy.skip_inactive_pibar_zero is True
    assert policy.device_active_mask_enabled is True
    assert policy.cpu_wave_skip_enabled is False
    assert policy.active_mask_threshold == 1e-6
    assert policy.active_mask_strict_gt is False


def test_no_pruning_with_default_no_cpu_pruning_passes_no_active_mask(monkeypatch):
    monkeypatch.delenv("GPUREC_BACKWARD_NO_CPU_PRUNING", raising=False)

    policy = backward_pruning_policy(use_pruning=False, pruning_threshold=1e-6)

    assert policy.no_cpu_pruning is True
    assert policy.device_active_mask_enabled is False
    assert policy.cpu_wave_skip_enabled is False
    assert policy.active_mask_threshold == 0.0
    assert policy.active_mask_strict_gt is True


def test_cpu_pruning_branch_computes_exact_zero_mask_when_pruning_disabled(monkeypatch):
    monkeypatch.setenv("GPUREC_BACKWARD_NO_CPU_PRUNING", "0")
    policy = backward_pruning_policy(use_pruning=False, pruning_threshold=1e-6)
    rhs = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, -1.0e-12, 0.0],
            [1.0e-6, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert policy.device_active_mask_enabled is True
    assert policy.cpu_wave_skip_enabled is True
    assert policy.active_mask_threshold == 0.0
    assert policy.active_mask_strict_gt is True
    assert _reference_active_mask(rhs, policy).tolist() == [False, True, True]


def test_pruning_enabled_active_mask_includes_threshold_boundary(monkeypatch):
    monkeypatch.setenv("GPUREC_BACKWARD_NO_CPU_PRUNING", "0")
    policy = backward_pruning_policy(use_pruning=True, pruning_threshold=1e-6)
    rhs = torch.tensor(
        [
            [0.0, 9.0e-7],
            [-1.0e-6, 0.0],
            [0.0, 1.1e-6],
        ],
        dtype=torch.float64,
    )

    assert policy.active_mask_threshold == 1e-6
    assert policy.active_mask_strict_gt is False
    assert _reference_active_mask(rhs, policy).tolist() == [False, True, True]


def test_inactive_wave_accounting_matches_current_cpu_pruning_counters():
    assert inactive_wave_accounting(
        0,
        4,
        cpu_wave_skip_enabled=True,
    ) == (False, 1, 4)
    assert inactive_wave_accounting(
        2,
        4,
        cpu_wave_skip_enabled=True,
    ) == (True, 0, 2)
    assert inactive_wave_accounting(
        0,
        4,
        cpu_wave_skip_enabled=False,
    ) == (True, 0, 0)


def test_dts_skip_inactive_pibar_zero_uses_standard_env_truth_rules(monkeypatch):
    monkeypatch.setenv("GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO", "off")
    assert (
        backward_pruning_policy(use_pruning=True, pruning_threshold=1e-6)
        .skip_inactive_pibar_zero
        is False
    )

    monkeypatch.setenv("GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO", "yes")
    assert (
        backward_pruning_policy(use_pruning=True, pruning_threshold=1e-6)
        .skip_inactive_pibar_zero
        is True
    )
