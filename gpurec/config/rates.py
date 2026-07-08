"""Box bounds + init for log2 DTL event rates (relative to speciation = 1).

``RateBounds`` is the single source of truth for the min/max rate floor+cap shared by the rate
optimizer (``gpurec/optimization.py``), the initial ``theta`` ``Parameter`` (``gpurec/api/model.py``),
and the genewise fit recipe's bound-active certificate (``gpurec/fit/genewise_fit.py``). Before this
module, the GLOBAL floor (``1e-10``, no cap) was copy-pasted as three identical signature defaults in
``optimization.py`` plus ``model.py``'s theta init, while ``fit_genewise``'s tighter box
(``1e-6``/``2.0``) lived only as bare signature defaults -- not surfaced anywhere as a named config
object. ``RateBounds()`` pins the global values; ``RateBounds.genewise()`` is the genewise preset.
No numeric value changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RateBounds:
    """Box bounds + init for log2 DTL event rates (relative to speciation=1)."""
    min_rate: float = 1e-10          # global floor (optimization.py / model init)
    max_rate: Optional[float] = None # no cap by default
    init_rate: float = 1e-10         # theta Parameter init
    bound_active_eps: float = 1e-6   # |theta - bound| < eps => 'active' (genewise cert)

    @classmethod
    def genewise(cls) -> "RateBounds":
        """The ``fit_genewise`` preset: floor ``1e-6``, cap ``2.0`` (unchanged from their prior bare
        signature defaults); ``init_rate``/``bound_active_eps`` keep the global ``RateBounds()``
        values."""
        return cls(min_rate=1e-6, max_rate=2.0)
