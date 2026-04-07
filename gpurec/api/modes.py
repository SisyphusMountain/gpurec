"""Mode taxonomy for the notebook-friendly API.

The three modes (`global`, `specieswise`, `genewise`) map to the existing
`(genewise, specieswise, pairwise)` boolean grid used by `GeneDataset` and
`extract_parameters`. Pairwise is intentionally excluded from the enum;
`GeneReconModel` fails fast if a pairwise dataset is passed in.
"""
from __future__ import annotations

import math

import torch


# mode -> (genewise, specieswise, pairwise)
_MODE_MAP: dict[str, tuple[bool, bool, bool]] = {
    "global":      (False, False, False),  # theta [3]
    "specieswise": (False, True,  False),  # theta [S, 3]
    "genewise":    (True,  False, False),  # theta [G, 3]
}


def _mode_to_flags(mode: str) -> tuple[bool, bool, bool]:
    if mode not in _MODE_MAP:
        raise ValueError(
            f"Unknown mode {mode!r}. Valid: {sorted(_MODE_MAP)}"
        )
    return _MODE_MAP[mode]


def _default_theta_init(dataset, mode: str) -> torch.Tensor:
    """log2(1e-10) initialization, matching gpurec/core/model.py:43-53."""
    base = math.log2(1e-10)
    if mode == "global":
        shape: tuple[int, ...] = (3,)
    elif mode == "specieswise":
        shape = (int(dataset.S), 3)
    elif mode == "genewise":
        shape = (len(dataset.families), 3)
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return torch.full(shape, base, dtype=dataset.dtype, device=dataset.device)
