"""Clade wave scheduling for parallel Pi computation.

Uses the C++ phased scheduler (3 phases: leaves / internal / root) with
λ-priority greedy batching and packet-aware sibling grouping.

Falls back to a Python BFS implementation if the C++ phased data is not
available in the preprocessed helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .legacy import _compute_clade_waves_bfs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_clade_waves(
    ccp_helpers: Dict[str, Any],
    *,
    max_wave_size: int | None = None,
) -> Tuple[List[List[int]], List[int]]:
    """Return phased wave assignments from preprocessed ccp_helpers.

    If the C++ phased waves are available (``ccp_helpers['phased_waves']``),
    use them directly.  Otherwise fall back to the Python BFS scheduler.

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
    if 'phased_waves' in ccp_helpers:
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

    # Fallback: Python BFS (no phase info)
    return _compute_clade_waves_bfs(ccp_helpers)


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------


def wave_stats(
    waves: List[List[int]],
    ccp_helpers: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return per-wave statistics for verification and tuning."""
    split_counts: List[int] = ccp_helpers["split_counts"].tolist()
    stats = []
    for k, wave_clades in enumerate(waves):
        counts = [split_counts[c] for c in wave_clades]
        stats.append({
            "wave": k,
            "n_clades": len(wave_clades),
            "n_splits": sum(counts),
            "max_splits_per_clade": max(counts) if counts else 0,
        })
    return stats
