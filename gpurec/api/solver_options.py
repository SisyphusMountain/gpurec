from dataclasses import dataclass


@dataclass
class SolverOptions:
    """Runtime-tunable solver controls for fixed-point and series solves."""

    e_init: float = -1.0
    e_max_iter: int = 2000
    e_tol: float = 1e-8
    pi_iters: int = 6
    neumann_terms: int = 3
    bicgstab_max_iter: int = 500
    bicgstab_tol: float = 1e-7
    bicgstab_breakdown_tol: float = 1e-30
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
        if int(self.bicgstab_max_iter) < 1:
            raise ValueError("bicgstab_max_iter must be at least 1")
        if float(self.bicgstab_tol) <= 0.0:
            raise ValueError("bicgstab_tol must be positive")
        if float(self.bicgstab_breakdown_tol) <= 0.0:
            raise ValueError("bicgstab_breakdown_tol must be positive")
        if float(self.adjoint_pruning_threshold) < 0.0:
            raise ValueError("adjoint_pruning_threshold must be non-negative")
        if float(self.pibar_side_threshold) < 0.0:
            raise ValueError("pibar_side_threshold must be non-negative")
