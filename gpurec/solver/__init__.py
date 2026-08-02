"""Stable optimization toolkit: value/grad, CG/Lanczos, curvature + exact HVP, tangents, penalties."""
from gpurec.solver.penalties import (
    OriginationPenalty,
    group_expand,
    group_reduce,
    origination_penalty_and_grad,
    tv_prior_and_grad,
)
from gpurec.solver.value_and_grad import (
    FORWARD_SAVED_NAMES,
    forward_solve,
    free_cuda_cache_if_tight,
    make_value_and_grad,
)

__all__ = [
    "FORWARD_SAVED_NAMES", "forward_solve", "free_cuda_cache_if_tight", "make_value_and_grad",
    "OriginationPenalty", "origination_penalty_and_grad", "tv_prior_and_grad",
    "group_expand", "group_reduce",
]
