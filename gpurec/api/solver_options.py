from dataclasses import dataclass
from typing import Optional


@dataclass
class SolverOptions:
    """Runtime-tunable solver controls for fixed-point and series solves."""

    e_max_iter: int = 128
    e_tol: float = 1e-8
    pi_iters: int = 64
    neumann_terms: int = 64
    # Max iterations for the linear E-adjoint / GGN solve: a Neumann series
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
    # Warm-starts gpurec.solver.hvp.exact's analytic-HVP construction: (1) reuses the existing
    # static.warm_v dict for build_point_cache's own backward pass (no new memory), and (2) caches
    # the tangent-adjoint sweep's own v_k per probe_id in static.warm_v_tangent (new memory, ~probe
    # count x a single warm_v's footprint). Independent of GPUREC_WARM_ADJOINT -- config-only, no
    # env var. Default True: part (1) is pure upside when repeatedly calling make_exact_hvp on the
    # same static; part (2) only activates when a caller explicitly passes probe_id= to hvp(), so
    # existing CG/Lanczos-based callers (which never do) are unaffected either way. Set False to
    # save the warm_v_tangent memory.
    use_hvp_warm_start: bool = True
    pibar_side_threshold: float = 0.0
    # Convergence tolerance for the E-step tangent fixed point in the forward-mode
    # JVP (`gpurec.solver.hvp.forward_tangent`). Distinct from `e_tol` (the primal
    # E-step fixed point).
    e_tangent_tol: float = 1e-9

    def validate(self) -> None:
        if int(self.e_max_iter) < 1:
            raise ValueError("e_max_iter must be at least 1")
        if float(self.e_tol) <= 0.0:
            raise ValueError("e_tol must be positive")
        if int(self.pi_iters) < 2 or int(self.pi_iters) % 2 != 0:
            raise ValueError("pi_iters must be an even integer at least 2")
        if int(self.neumann_terms) < 0:
            raise ValueError("neumann_terms must be non-negative")
        if int(self.e_adjoint_max_iter) < 1:
            raise ValueError("e_adjoint_max_iter must be at least 1")
        if self.e_adjoint_tol is not None and float(self.e_adjoint_tol) <= 0.0:
            raise ValueError("e_adjoint_tol must be positive or None (dtype-auto)")
        if float(self.adjoint_pruning_threshold) < 0.0:
            raise ValueError("adjoint_pruning_threshold must be non-negative")
        if float(self.pibar_side_threshold) < 0.0:
            raise ValueError("pibar_side_threshold must be non-negative")
