"""Likelihood optimization / MAP / cross-validation layer for gpurec.

Ported from the kernel-bench ``newton/`` research package onto gpurec's batched
streaming evaluators (``gpurec.api._execution``), so the optimization, MAP, and
cross-validation work runs on the full hogenom dataset rather than the frozen
80-family capture. See ``docs/optim/`` for the findings that motivated it.

Terminology rename applied at the kernel-bench -> gpurec port boundary:
``item -> family``, ``col -> receiver``, ``state -> species``, ``node -> sp``,
``solve_e_pi -> solve_resident_e_pi``, ``col_weights -> receiver_weights``.

Phases:
  1. ``value_and_grad`` / ``cg`` / ``optimize`` / ``baselines`` -- the no-HVP core.
  2. ``map_cv`` -- k-fold-over-families cross-validation with lambda-homotopy.
  3. ``hvp_exact`` / ``forward_tangent`` / ``ggn`` / ``newton_cg`` / ``map_fit`` -- exact
     second-order (needs the SO/tangent Triton kernels in ``gpurec.core.kernels``).
"""

from gpurec.optim.value_and_grad import (
    FORWARD_SAVED_NAMES,
    forward_solve,
    free_cuda_cache_if_tight,
    make_value_and_grad,
)

__all__ = [
    "FORWARD_SAVED_NAMES",
    "forward_solve",
    "free_cuda_cache_if_tight",
    "make_value_and_grad",
]
