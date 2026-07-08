import torch


def dtype_rel_tol_default(dtype) -> float:
    """Dtype-matched relative-residual target for dtype-aware linear/fixed-point solves.

    The target must sit ABOVE the finite-precision stagnation floor (a few * the
    unit roundoff ``eps``) yet stay BELOW the ~2e-4 downstream gradient atomic-noise
    floor, so the solve is as tight as the working precision *reliably* allows
    without wasting iterations. These are exactly the values used elsewhere in
    the codebase for the same purpose (e.g. the E-adjoint BiCGSTAB/GMRES solve
    and ``gpurec.solver.forward_tangent._default_tol``):

      * fp32: ``1e-6`` (~8.4x fp32 eps 1.19e-7).
      * fp64: ``1e-12`` (~4.5e3x fp64 eps 2.2e-16; tight but trivially reachable).
    """
    return 1e-12 if dtype == torch.float64 else 1e-6


def dtype_rel_tol_floor(dtype) -> float:
    """Tightest relative residual an iteration can reach in a given dtype.

    A small multiple of the unit roundoff. Any requested target below this is
    physically unreachable, so callers clamp up to it (with a warning) rather
    than spin to ``max_iter`` and raise on a solve that is, in fact, fully
    converged.
    """
    return 4.0 * float(torch.finfo(dtype).eps)
