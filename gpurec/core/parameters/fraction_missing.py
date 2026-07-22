# gpurec/core/parameters/fraction_missing.py
"""Host-side construction of the per-species fraction-missing leaf boundary.

``fraction_missing_s = 1 - p_obs_s`` is the probability that a gene present in
species ``s`` is unobserved at that leaf. It enters the E-step and Pi recurrence
as a leaf boundary (see docs/superpowers/plans/2026-07-22-fraction-missing.md).
"""
from __future__ import annotations

import torch

NEG_INF = float("-inf")


def species_leaf_mask(species_helpers: dict) -> torch.Tensor:
    """[S] bool: True at species-tree leaves.

    A species node is a leaf iff its first child is the Rust sentinel ``S``
    (internal nodes get real child indices ``< S``).
    """
    S = int(species_helpers["S"])
    sp_child1 = species_helpers["sp_child1"]
    return sp_child1.to(torch.long) == S


def build_fraction_missing_tensors(
    species_helpers: dict,
    *,
    fraction_missing,
    species_name_to_index: dict[str, int] | None = None,
    device,
    dtype,
) -> torch.Tensor | None:
    """Map ``fraction_missing`` to ``leaf_fm_log`` ([S], log2, -inf off-leaf/observed).

    ``fraction_missing`` may be:
      * ``None``                       -> every gene observed (returns ``None``);
      * ``float``                      -> same value at every species-tree leaf;
      * ``dict {species_name: frac}``  -> per-leaf by name (absent names = observed);
      * length-S sequence/tensor       -> per-species (internal entries forced off).

    Returns ``None`` when the result is empty (all observed) so callers keep the
    no-overhead default path. Values must lie in ``[0, 1)``.
    """
    if fraction_missing is None:
        return None

    S = int(species_helpers["S"])
    leaf_mask = species_leaf_mask(species_helpers).to(device)
    fm = torch.zeros(S, dtype=dtype, device=device)

    if isinstance(fraction_missing, dict):
        if species_name_to_index is None:
            raise ValueError(
                "fraction_missing as a {name: frac} dict requires species_name_to_index "
                "(load it via gpurec.core.scheduling.batching.species_name_to_index)."
            )
        for name, val in fraction_missing.items():
            if name not in species_name_to_index:
                raise KeyError(f"fraction_missing species name {name!r} not in species tree")
            fm[int(species_name_to_index[name])] = float(val)
    elif isinstance(fraction_missing, (int, float)):
        fm[leaf_mask] = float(fraction_missing)
    else:
        t = torch.as_tensor(fraction_missing, dtype=dtype, device=device).reshape(-1)
        if t.numel() != S:
            raise ValueError(
                f"fraction_missing tensor has length {t.numel()}, expected S={S} "
                "(per-species) or pass a dict keyed by species name."
            )
        fm = t.clone()

    # Internal species never carry a missing term.
    fm = torch.where(leaf_mask, fm, torch.zeros((), dtype=dtype, device=device))

    fmax = float(fm.max())
    if fmax >= 1.0 or float(fm.min()) < 0.0:
        raise ValueError("fraction_missing values must lie in [0, 1).")
    if fmax <= 0.0:
        return None  # nothing missing -> keep the fast default path

    positive = fm > 0
    # Clamp-free log2: take log2 only on the strictly-positive branch; the
    # masked_fill(1.0) makes the off-branch log2(1)=0 (finite, never NaN), then
    # torch.where selects -inf there.
    leaf_fm_log = torch.where(
        positive,
        torch.log2(fm.masked_fill(~positive, 1.0)),
        torch.full_like(fm, NEG_INF),
    )
    return leaf_fm_log
