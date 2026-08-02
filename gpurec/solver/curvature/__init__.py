"""Per-parameter-block joint-curvature Newton drivers (receiver, origination, genewise),
built on the shared gauge-projected core in ``gauge.py``."""
from gpurec.solver.curvature.gauge import (
    certify_min,
    gauge_operator,
    newton_min,
    resolve_newton,
)

__all__ = ["resolve_newton", "gauge_operator", "certify_min", "newton_min"]
