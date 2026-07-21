from dataclasses import dataclass
from typing import Optional


@dataclass
class SolverOptions:
    """Runtime-tunable solver controls for fixed-point and series solves."""

    e_max_iter: int = 128
    e_tol: float = 1e-8
    pi_iters: int = 64
    neumann_terms: int = 64
    # Max iterations for the linear E-adjoint / GGN solve. That solve now uses
    # GMRES (:func:`gpurec.api._implicit_grad._gmres`), which is breakdown-free and
    # converges in O(10) steps on the well-conditioned E-adjoint, so it early-stops
    # far below this cap; the cap only bounds the Krylov basis. (Field name kept for
    # backward compatibility with saved configs.)
    bicgstab_max_iter: int = 128
    # E-adjoint solve relative-residual target. ``None`` = dtype-relative auto (the
    # robust default): the target is derived from ``torch.finfo(working_dtype).eps``
    # inside the solver (fp32 -> 1e-6, fp64 -> 1e-12). A hardcoded fp32-eps value
    # such as the old 1e-7 sits *below* the achievable fp32 floor and made the solve
    # raise on an essentially-converged iterate; ``None`` avoids that. Pass an
    # explicit float only to solve tighter than the dtype default (e.g. 1e-10 in fp64).
    bicgstab_tol: Optional[float] = None
    # BiCGSTAB-only breakdown guard; ignored by the GMRES E-adjoint solve. Retained
    # so existing configs that set it still load. Only used if ``_bicgstab`` is
    # called directly (e.g. its unit tests).
    bicgstab_breakdown_tol: Optional[float] = None
    # Solver for the E-adjoint backward linear solve ``(I - J) wE = q`` (the final linear
    # solve in ``_e_adjoint_and_theta_vjp``). ``"gmres"`` (default, backward compatible): matrix-free
    # GMRES, breakdown-free but has a theta-dependent fp32 residual floor from Arnoldi
    # orthogonalization (~1e-6 to 5.5e-6) that can make it fail to converge mid-optimization at
    # large species counts. ``"neumann"``: Neumann-series solve (no orthogonalization -> no fp32
    # floor), valid because the E-step self-map Jacobian ``J`` is a contraction (the forward E
    # fixed point converges) -- validated to reach ~1e-6 relative residual in <=10 terms across a
    # full-scale fit's backward solves.
    e_adjoint_solver: str = "gmres"
    adjoint_pruning_threshold: float = 1e-6
    use_adjoint_pruning: bool = True
    # Warm-starts gpurec.solver.hvp_exact's analytic-HVP construction: (1) reuses the existing
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
    # JVP (`gpurec.solver.forward_tangent`). Distinct from `e_tol` (the primal
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
        if int(self.bicgstab_max_iter) < 1:
            raise ValueError("bicgstab_max_iter must be at least 1")
        if self.bicgstab_tol is not None and float(self.bicgstab_tol) <= 0.0:
            raise ValueError("bicgstab_tol must be positive or None (dtype-auto)")
        if self.bicgstab_breakdown_tol is not None and float(self.bicgstab_breakdown_tol) <= 0.0:
            raise ValueError("bicgstab_breakdown_tol must be positive or None (dtype-auto)")
        e_adjoint_solver = str(self.e_adjoint_solver).strip().lower()
        if e_adjoint_solver not in ("gmres", "neumann"):
            raise ValueError("e_adjoint_solver must be one of: gmres, neumann")
        self.e_adjoint_solver = e_adjoint_solver
        if float(self.adjoint_pruning_threshold) < 0.0:
            raise ValueError("adjoint_pruning_threshold must be non-negative")
        if float(self.pibar_side_threshold) < 0.0:
            raise ValueError("pibar_side_threshold must be non-negative")
