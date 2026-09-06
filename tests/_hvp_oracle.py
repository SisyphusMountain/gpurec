"""Finite-difference test oracle for the analytic Hessian-vector product."""

import torch

from gpurec.solver.value_and_grad import free_cuda_cache_if_tight


def fd_hessian_hvp(vg, theta_vec, warm_E, *, eps=1e-5):
    """Return a central-difference gradient oracle along normalized directions."""
    base = theta_vec.double()

    def hvp(v):
        v = v.double()
        norm = float(torch.linalg.vector_norm(v))
        if norm == 0.0:
            return torch.zeros_like(v)
        direction = v / norm
        free_cuda_cache_if_tight()
        output = vg((base + eps * direction).to(theta_vec.dtype), warm_E=warm_E)
        grad_plus = output[1].double()
        del output
        output = vg((base - eps * direction).to(theta_vec.dtype), warm_E=warm_E)
        grad_minus = output[1].double()
        del output
        return norm * (grad_plus - grad_minus) / (2 * eps)

    return hvp
