"""Private policy helpers for retained backward active-mask pruning."""

from __future__ import annotations

from dataclasses import dataclass

from ._helpers import _env_flag_enabled


@dataclass(frozen=True)
class BackwardPruningPolicy:
    """Resolved branch policy for ``Pi_wave_backward`` active-mask pruning.

    ``no_cpu_pruning`` preserves the current runtime branch name.  When true,
    backward avoids host-side ``active_mask.any()`` readbacks; it still passes
    device active masks into kernels when ``use_pruning`` is true.
    """

    use_pruning: bool
    pruning_threshold: float
    no_cpu_pruning: bool
    skip_inactive_pibar_zero: bool

    @property
    def active_mask_threshold(self) -> float:
        """Threshold passed to the active-mask kernel for this run."""
        return self.pruning_threshold if self.use_pruning else 0.0

    @property
    def active_mask_strict_gt(self) -> bool:
        """Whether mask rows use ``absmax > threshold`` instead of ``>=``."""
        return not self.use_pruning

    @property
    def device_active_mask_enabled(self) -> bool:
        """Whether this policy computes and passes a device active mask."""
        return self.use_pruning or not self.no_cpu_pruning

    @property
    def cpu_wave_skip_enabled(self) -> bool:
        """Whether backward performs host-side all-inactive wave skips."""
        return not self.no_cpu_pruning


def backward_pruning_policy(
    *,
    use_pruning: bool,
    pruning_threshold: float,
) -> BackwardPruningPolicy:
    """Resolve backward pruning environment flags with current defaults."""
    return BackwardPruningPolicy(
        use_pruning=bool(use_pruning),
        pruning_threshold=float(pruning_threshold),
        no_cpu_pruning=_env_flag_enabled("GPUREC_BACKWARD_NO_CPU_PRUNING", "1"),
        skip_inactive_pibar_zero=_env_flag_enabled(
            "GPUREC_DTS_SKIP_INACTIVE_PIBAR_ZERO",
            "1",
        ),
    )


def inactive_wave_accounting(
    active_count: int,
    wave_size: int,
    *,
    cpu_wave_skip_enabled: bool,
) -> tuple[bool, int, int]:
    """Return ``(process_wave, waves_skipped, clades_skipped)`` for a wave."""
    if not cpu_wave_skip_enabled:
        return True, 0, 0

    active_count = int(active_count)
    wave_size = int(wave_size)
    if active_count == 0:
        return False, 1, wave_size
    return True, 0, wave_size - active_count
