from dataclasses import dataclass
from typing import Optional


@dataclass
class SolverOptions:
    """Runtime-tunable solver controls for fixed-point and series solves."""

    e_max_iter: int = 2000
    e_tol: float = 1e-8
    pi_iters: int = 64
    neumann_terms: int = 64
    self_loop_solver: str = "neumann"
    bicgstab_max_iter: int = 500
    # BiCGSTAB E-adjoint solve tolerances. ``None`` = dtype-relative auto (the
    # robust default): the relative-residual target and breakdown guard are
    # derived from ``torch.finfo(working_dtype).eps`` inside ``_bicgstab`` (fp32
    # -> 1e-6 target, fp64 -> 1e-12). A hardcoded fp32-eps value such as the old
    # 1e-7 sits *below* the achievable fp32 floor and made the solve raise on an
    # essentially-converged iterate; ``None`` avoids that. Pass an explicit float
    # only to solve tighter than the dtype default (e.g. 1e-10 in fp64).
    bicgstab_tol: Optional[float] = None
    bicgstab_breakdown_tol: Optional[float] = None
    adjoint_pruning_threshold: float = 1e-6
    use_adjoint_pruning: bool = True
    pibar_side_threshold: float = 0.0

    def validate(self) -> None:
        if int(self.e_max_iter) < 1:
            raise ValueError("e_max_iter must be at least 1")
        if float(self.e_tol) <= 0.0:
            raise ValueError("e_tol must be positive")
        if int(self.pi_iters) < 2 or int(self.pi_iters) % 2 != 0:
            raise ValueError("pi_iters must be an even integer at least 2")
        if int(self.neumann_terms) < 0:
            raise ValueError("neumann_terms must be non-negative")
        self_loop_solver = str(self.self_loop_solver).strip().lower()
        if self_loop_solver not in ("neumann", "gmres"):
            raise ValueError("self_loop_solver must be one of: neumann, gmres")
        self.self_loop_solver = self_loop_solver
        if int(self.bicgstab_max_iter) < 1:
            raise ValueError("bicgstab_max_iter must be at least 1")
        if self.bicgstab_tol is not None and float(self.bicgstab_tol) <= 0.0:
            raise ValueError("bicgstab_tol must be positive or None (dtype-auto)")
        if self.bicgstab_breakdown_tol is not None and float(self.bicgstab_breakdown_tol) <= 0.0:
            raise ValueError("bicgstab_breakdown_tol must be positive or None (dtype-auto)")
        if float(self.adjoint_pruning_threshold) < 0.0:
            raise ValueError("adjoint_pruning_threshold must be non-negative")
        if float(self.pibar_side_threshold) < 0.0:
            raise ValueError("pibar_side_threshold must be non-negative")
