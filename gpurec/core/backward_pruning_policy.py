"""Private policy helpers for retained backward active-wave-row pruning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackwardPruningPolicy:
    """Resolved branch policy for ``Pi_wave_backward`` active-wave-row pruning.

    Backward avoids host-side active-wave-row readbacks in the retained production
    path.  It still passes device active wave-rows into kernels when
    ``use_pruning`` is true.
    """

    use_pruning: bool
    pruning_threshold: float
    skip_inactive_pibar_zero: bool

    @property
    def active_wave_row_threshold(self) -> float:
        """Threshold passed to the active-wave-row kernel for this run."""
        return self.pruning_threshold if self.use_pruning else 0.0

    @property
    def active_wave_row_strict_gt(self) -> bool:
        """Whether mask rows use ``absmax > threshold`` instead of ``>=``."""
        return not self.use_pruning

    @property
    def device_active_wave_rows_enabled(self) -> bool:
        """Whether this policy computes and passes a device active wave-row."""
        return self.use_pruning

    @property
    def cpu_wave_skip_enabled(self) -> bool:
        """Whether backward performs host-side all-inactive wave skips."""
        return False


def backward_pruning_policy(
    *,
    use_pruning: bool,
    pruning_threshold: float,
) -> BackwardPruningPolicy:
    """Resolve the fixed retained backward pruning policy."""
    return BackwardPruningPolicy(
        use_pruning=bool(use_pruning),
        pruning_threshold=float(pruning_threshold),
        skip_inactive_pibar_zero=True,
    )
