"""Stable optimization toolkit: value/grad, CG/Lanczos, curvature + exact HVP, tangents."""
from gpurec.solver.value_and_grad import (
    FORWARD_SAVED_NAMES,
    forward_solve,
    free_cuda_cache_if_tight,
    make_value_and_grad,
)

__all__ = [
    "FORWARD_SAVED_NAMES", "forward_solve", "free_cuda_cache_if_tight", "make_value_and_grad",
]
