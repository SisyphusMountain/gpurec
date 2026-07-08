"""Memory-policy / infra knobs for CUDA memory-budget heuristics and cache-empty gates.

These fields were, before this module existed, scattered as individual signature-default
literals across ``gpurec/core/memory_policy.py`` (``cuda_memory_budget_bytes``'s
``default_fraction``/``default_reserve_gib``, ``proposal0_wave_scratch_bytes``'s
``scratch_tensors``), ``gpurec/solver/value_and_grad.py`` (``free_cuda_cache_if_tight``'s
``min_free_gib``, ``make_value_and_grad``'s ``grad_avg_K``), ``gpurec/solver/curvature.py``
(the HVP-rebuild ``free_cuda_cache_if_tight(min_free_gib=8.0)`` call), and
``gpurec/solver/hvp_exact.py`` (``free_cache_every``'s ``32`` env-var fallback).
``MemoryOptions`` is the single source of truth for those values; the affected sites now read
from it. These knobs affect memory/performance decisions only, never numerics -- no gradient or
likelihood value depends on them. The two ``min_free_gib_*`` fields are DISTINCT, intentionally
different gates (driver-cache-empty vs HVP-rebuild), not a numeric inconsistency to collapse.

``fraction``/``reserve_gib``/``free_cache_every`` remain overridable at call time by the
pre-existing env vars (``GPUREC_MEMORY_POLICY_FRACTION``, ``GPUREC_MEMORY_POLICY_RESERVE_GIB``,
``NEWTON_FREE_CACHE_EVERY``); this dataclass supplies the DEFAULT the env var falls back to, not
a replacement for the env-var lookup.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryOptions:
    """Memory/infra knobs for CUDA budget heuristics and cache-empty gates (no numeric effect)."""
    fraction: float = 0.85          # usable frac of total GPU mem (env GPUREC_MEMORY_POLICY_FRACTION)
    reserve_gib: float = 1.0        # headroom GiB (env GPUREC_MEMORY_POLICY_RESERVE_GIB)
    scratch_tensors: int = 10       # [W,S]-tensor multiplier in wave mem estimate
    min_free_gib_driver: float = 4.0  # value_and_grad cache-empty gate
    min_free_gib_hvp: float = 8.0     # curvature HVP-rebuild gate
    free_cache_every: int = 32        # empty CUDA cache every-K-waves (env NEWTON_FREE_CACHE_EVERY)
    grad_avg_k: int = 1               # backward passes averaged
