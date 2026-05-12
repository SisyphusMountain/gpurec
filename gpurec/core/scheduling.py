"""Clade wave scheduling for the retained phased Pi computation path."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_clade_waves(
    ccp_helpers: Dict[str, Any],
    *,
    max_wave_size: int | None = None,
) -> Tuple[List[List[int]], List[int]]:
    """Return phased wave assignments from preprocessed ccp_helpers.

    Parameters
    ----------
    ccp_helpers:
        Dict returned by ``preprocess`` (the ``"ccp"`` sub-dict).
    max_wave_size:
        Optional cap on wave size. If provided and the C++ phased waves
        need to be split, they are chunked accordingly.

    Returns
    -------
    waves : list[list[int]]
        ``waves[k]`` = list of clade IDs in wave k.
    phases : list[int]
        ``phases[k]`` = phase label (1=leaf, 2=internal, 3=root) for wave k.
    """
    if 'phased_waves' not in ccp_helpers or 'phased_phases' not in ccp_helpers:
        raise RuntimeError("preprocessed helpers must include C++ phased waves")
    raw_waves = ccp_helpers['phased_waves']
    raw_phases = ccp_helpers['phased_phases']
    waves = []
    phases = []
    for w, ph in zip(raw_waves, raw_phases):
        wlist = w.tolist() if hasattr(w, 'tolist') else list(w)
        if max_wave_size is not None and len(wlist) > max_wave_size:
            for start in range(0, len(wlist), max_wave_size):
                waves.append(wlist[start:start + max_wave_size])
                phases.append(int(ph))
        else:
            waves.append(wlist)
            phases.append(int(ph))
    return waves, phases
