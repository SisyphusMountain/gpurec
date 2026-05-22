import torch

from gpurec.core.backward_pruning_policy import (
    backward_pruning_policy,
)


def _reference_active_mask(rhs: torch.Tensor, policy) -> torch.Tensor:
    row_absmax = rhs.abs().amax(dim=1)
    if policy.active_mask_strict_gt:
        return row_absmax > policy.active_mask_threshold
    return row_absmax >= policy.active_mask_threshold


def test_default_policy_avoids_cpu_wave_skips_but_keeps_device_masks():
    policy = backward_pruning_policy(use_pruning=True, pruning_threshold=1e-6)

    assert policy.skip_inactive_pibar_zero is True
    assert policy.device_active_mask_enabled is True
    assert policy.cpu_wave_skip_enabled is False
    assert policy.active_mask_threshold == 1e-6
    assert policy.active_mask_strict_gt is False


def test_no_pruning_passes_no_active_mask():
    policy = backward_pruning_policy(use_pruning=False, pruning_threshold=1e-6)

    assert policy.device_active_mask_enabled is False
    assert policy.cpu_wave_skip_enabled is False
    assert policy.active_mask_threshold == 0.0
    assert policy.active_mask_strict_gt is True


def test_pruning_disabled_active_mask_would_use_exact_zero_boundary():
    policy = backward_pruning_policy(use_pruning=False, pruning_threshold=1e-6)
    rhs = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, -1.0e-12, 0.0],
            [1.0e-6, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert policy.device_active_mask_enabled is False
    assert policy.cpu_wave_skip_enabled is False
    assert policy.active_mask_threshold == 0.0
    assert policy.active_mask_strict_gt is True
    assert _reference_active_mask(rhs, policy).tolist() == [False, True, True]


def test_pruning_enabled_active_mask_includes_threshold_boundary():
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


def test_skip_inactive_pibar_zero_is_fixed_enabled():
    assert (
        backward_pruning_policy(use_pruning=True, pruning_threshold=1e-6)
        .skip_inactive_pibar_zero
        is True
    )
