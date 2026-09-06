"""Independent CPU check of the fixed-native-face hierarchical pullback Hessian."""
from __future__ import annotations

import itertools
import json
import math

import torch


def theta_of(z):
    u, v, w = z.unbind(-1)
    softplus2 = lambda x: torch.logaddexp(torch.zeros_like(x), x * math.log(2)) / math.log(2)
    d = u + softplus2(w) - softplus2(v)
    return torch.stack((d, w, d + v), dim=-1)


def main():
    rng = torch.Generator().manual_seed(60209)
    errors = dict(projector=0.0, tangent=0.0, gradient=0.0, hessian=0.0)
    cases = 0
    for _ in range(32):
        z = torch.randn(3, dtype=torch.float64, generator=rng) * 4
        theta0 = theta_of(z)
        gradient = torch.randn(3, dtype=torch.float64, generator=rng) * 10
        matrix = torch.randn(3, 3, dtype=torch.float64, generator=rng)
        hessian = matrix.T @ matrix + torch.eye(3, dtype=torch.float64)
        jac = torch.autograd.functional.jacobian(theta_of, z)
        maps = torch.stack([torch.autograd.functional.hessian(
            lambda x, i=i: theta_of(x)[i], z) for i in range(3)])
        h_phi = jac.T @ hessian @ jac + torch.einsum("i,ijk->jk", gradient, maps)
        g_phi = jac.T @ gradient
        for bits in itertools.product((False, True), repeat=3):
            fixed = torch.tensor(bits)
            free = ~fixed
            normals = jac * fixed[:, None]
            gram = normals @ normals.T + torch.diag(free.double())
            projector = torch.eye(3, dtype=torch.float64) - normals.T @ torch.linalg.solve(gram, normals)
            errors["projector"] = max(errors["projector"], float((projector @ projector - projector).abs().max()))
            errors["tangent"] = max(errors["tangent"], float((normals @ projector).abs().max()))
            eigenvalues, eigenvectors = torch.linalg.eigh(projector)
            tangent = eigenvectors[:, eigenvalues > 0.5]
            cases += 1
            if tangent.shape[1] == 0:
                continue

            def objective(t):
                theta = theta_of(z + tangent @ t)
                theta = torch.where(fixed, theta0, theta)
                displacement = theta - theta0
                return gradient @ displacement + 0.5 * displacement @ hessian @ displacement

            zero = torch.zeros(tangent.shape[1], dtype=torch.float64)
            reference_g = torch.autograd.functional.jacobian(objective, zero)
            reference_h = torch.autograd.functional.hessian(objective, zero)
            corrected = h_phi - torch.einsum("i,ijk->jk", gradient * fixed, maps)
            predicted_h = tangent.T @ corrected @ tangent
            errors["gradient"] = max(errors["gradient"], float((reference_g - tangent.T @ g_phi).abs().max()))
            errors["hessian"] = max(errors["hessian"], float((reference_h - predicted_h).abs().max()))
    assert max(errors.values()) < 2e-11, errors
    print(json.dumps(dict(cases=cases, max_absolute_errors=errors), indent=2))


if __name__ == "__main__":
    main()
