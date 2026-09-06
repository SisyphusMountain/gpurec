from dataclasses import dataclass
from typing import Optional

import torch

# The self-loop early-exit tolerance below (``neumann_term_tol``) is written in units
# of FLOAT32 precision, because float32 is the production model
# dtype. Applied unchanged in float64 it would stop an iteration about nine orders
# of magnitude before float64's own resolution and silently cap its accuracy -- which
# is exactly what tests/test_origination_curvature.py caught: a Hessian that must be
# symmetric to 1e-10 came out at 2.6e-8. ``dtype_scaled_self_loop_tol`` rescales them
# to whatever dtype a solve actually runs in.
_SELF_LOOP_TOL_REFERENCE_DTYPE = torch.float32


def dtype_scaled_self_loop_tol(tol, dtype) -> float:
    """Rescale a float32-referenced self-loop early-exit tolerance to ``dtype``.

    The configured number says "stop once the remaining change is this small
    relative to the row's own value, measured in float32 precision". Carrying that
    meaning to another dtype means scaling by the ratio of machine epsilons, so:

      * float32  -> factor exactly 1.0, i.e. bit-for-bit the configured value;
      * float64  -> factor 1.86e-9 (2.22e-16 / 1.19e-7), so 1e-7 becomes 1.9e-16,
        just under one float64 ulp -- the exit still fires, but only once the term
        genuinely can no longer move the sum at float64 resolution.

    ``tol`` of 0.0 (early exit disabled) stays 0.0.
    """
    tol = float(tol)
    if tol < 0.0:
        raise ValueError("self-loop early-exit tolerance must be non-negative")
    reference_eps = float(torch.finfo(_SELF_LOOP_TOL_REFERENCE_DTYPE).eps)
    return tol * (float(torch.finfo(dtype).eps) / reference_eps)


@dataclass
class SolverOptions:
    """Runtime controls for the exact reconciliation solver and its safeguards."""

    e_max_iter: int = 128
    e_tol: float = 1e-8
    pi_iters: int = 64
    neumann_terms: int = 64
    # Early-exit threshold for the wave self-loop Neumann series, in units of float32
    # precision (see dtype_scaled_self_loop_tol above: a float64 solve rescales it by
    # eps64/eps32 = 1.86e-9 so it never caps float64). Each row block stops as soon as
    # its largest remaining term is at or below
    # neumann_term_tol * (that block's largest |adjoint|) -- i.e. once the term can
    # no longer move the accumulated adjoint at float32 resolution. The test is
    # purely relative to each row's own adjoint: an absolute floor makes it far too
    # loose for the many rows whose adjoint is well below 1, which are exactly the
    # near-converged families the Newton loop is testing. The self-loop operator is
    # a strong contraction (spectral radius ~0.04, so each term is ~25x smaller than
    # the last), so this fires after ~6 terms and the rest of the 16/64-term budget
    # did nothing. 1e-7 is just below the fp32 unit roundoff 1.19e-7, so the dropped
    # tail is smaller than the rounding already present in the sum.
    # Set to 0.0 to disable the exit and always run the full neumann_terms.
    neumann_term_tol: float = 1e-7
    # Max iterations for the linear E-adjoint solve: a Neumann series
    # (:func:`gpurec.api._implicit_grad._neumann_e_adjoint`), valid because the
    # E-step self-map Jacobian is a contraction (the forward E fixed point
    # converges). No orthogonalization, so no fp32 Arnoldi-style residual floor.
    e_adjoint_max_iter: int = 128
    # E-adjoint solve relative-residual target. ``None`` = dtype-relative auto (the
    # robust default): the target is derived from ``torch.finfo(working_dtype).eps``
    # inside the solver (fp32 -> 1e-6, fp64 -> 1e-12). A hardcoded fp32-eps value
    # such as the old 1e-7 sits *below* the achievable fp32 floor and made the solve
    # raise on an essentially-converged iterate; ``None`` avoids that. Pass an
    # explicit float only to solve tighter than the dtype default (e.g. 1e-10 in fp64).
    e_adjoint_tol: Optional[float] = None
    adjoint_pruning_threshold: float = 1e-6
    use_adjoint_pruning: bool = True
    pibar_side_threshold: float = 0.0
    # Convergence tolerance for the E-step tangent fixed point in the forward-mode
    # JVP (`gpurec.solver.hvp.forward_tangent`). Distinct from `e_tol` (the primal
    # E-step fixed point).
    e_tangent_tol: float = 1e-9
    # Rows whose lanes span more than this many binary orders below their maximum are solved in
    # log space as a numerical fallback. The exact path carries one scale per clade row in scaled linear space,
    # so a species lane this far under the row maximum is zero in float32 (whose exponent floor is
    # 126 binary orders, denormals aside) where the log path still carries it. Such a row is
    # flagged by the exact forward and handed to the log-space sweeps, and its adjoint and tangent
    # to the Neumann series and the tangent sweeps, so the exact defaults keep the log path's
    # range. The default 100 leaves margin under 126 for what the solve itself adds on top of the
    # entry range. In float64 the kernels scale it by the ratio of exponent ranges, 1022/126 (see
    # ``gpurec.core.kernels.pi_forward.exact_range_for_dtype``), so one number covers both.
    exact_range_log2: float = 100.0
    def validate(self) -> None:
        if int(self.e_max_iter) < 1:
            raise ValueError("e_max_iter must be at least 1")
        if float(self.e_tol) <= 0.0:
            raise ValueError("e_tol must be positive")
        if int(self.pi_iters) < 2 or int(self.pi_iters) % 2 != 0:
            raise ValueError("pi_iters must be an even integer at least 2")
        if int(self.neumann_terms) < 0:
            raise ValueError("neumann_terms must be non-negative")
        if float(self.neumann_term_tol) < 0.0:
            raise ValueError("neumann_term_tol must be non-negative (0.0 disables the early exit)")
        if int(self.e_adjoint_max_iter) < 1:
            raise ValueError("e_adjoint_max_iter must be at least 1")
        if self.e_adjoint_tol is not None and float(self.e_adjoint_tol) <= 0.0:
            raise ValueError("e_adjoint_tol must be positive or None (dtype-auto)")
        if float(self.adjoint_pruning_threshold) < 0.0:
            raise ValueError("adjoint_pruning_threshold must be non-negative")
        if float(self.pibar_side_threshold) < 0.0:
            raise ValueError("pibar_side_threshold must be non-negative")
        if float(self.exact_range_log2) <= 0.0:
            raise ValueError("exact_range_log2 must be positive")
