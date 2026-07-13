import pytest
import torch

from gpurec.core.kernels.e_step import e_step_triton_autograd
from gpurec.core.kernels.e_step_so import e_step_backward_so


def _require_cuda() -> torch.device:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")


def _extinction_inputs(device: torch.device):
    dtype = torch.float32
    extinction = torch.tensor(
        [[-0.7, -1.1, -0.4]], device=device, dtype=dtype
    )
    speciation_log_probability = torch.tensor(
        [[-1.2, -1.4, -1.6]], device=device, dtype=dtype
    )
    duplication_log_probability = torch.tensor(
        [[-2.0, -2.2, -2.4]], device=device, dtype=dtype
    )
    loss_log_probability = torch.tensor(
        [[-1.8, -1.5, -1.3]], device=device, dtype=dtype
    )
    max_transfer = torch.tensor(
        [[-0.4, -0.2, -0.3]], device=device, dtype=dtype
    )
    receiver_log_probs = torch.tensor(
        [-0.9, -2.1, -1.4], device=device, dtype=dtype
    )
    species_parent = torch.tensor([-1, 0, 0], device=device, dtype=torch.int32)
    species_child1 = torch.tensor([1, -1, -1], device=device, dtype=torch.int32)
    species_child2 = torch.tensor([2, -1, -1], device=device, dtype=torch.int32)
    return (
        extinction,
        speciation_log_probability,
        duplication_log_probability,
        loss_log_probability,
        max_transfer,
        receiver_log_probs,
        species_parent,
        species_child1,
        species_child2,
    )


@pytest.mark.gpu
def test_uniform_extinction_vjp_has_no_receiver_weight_gradient() -> None:
    device = _require_cuda()
    (
        extinction,
        speciation_log_probability,
        duplication_log_probability,
        loss_log_probability,
        max_transfer,
        receiver_log_probs,
        species_parent,
        species_child1,
        species_child2,
    ) = _extinction_inputs(device)
    extinction.requires_grad_(True)
    receiver_log_probs.requires_grad_(True)

    outputs = e_step_triton_autograd(
        extinction,
        speciation_log_probability,
        duplication_log_probability,
        loss_log_probability,
        max_transfer,
        receiver_log_probs,
        species_parent,
        species_child1,
        species_child2,
        2,
        use_receiver_weights=False,
    )
    (receiver_gradient,) = torch.autograd.grad(
        sum(output.sum() for output in outputs), receiver_log_probs
    )
    torch.testing.assert_close(
        receiver_gradient,
        torch.zeros_like(receiver_gradient),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.gpu
def test_uniform_extinction_vjp_directional_derivative_has_no_receiver_gradient() -> None:
    device = _require_cuda()
    (
        extinction,
        speciation_log_probability,
        duplication_log_probability,
        loss_log_probability,
        max_transfer,
        receiver_log_probs,
        species_parent,
        species_child1,
        species_child2,
    ) = _extinction_inputs(device)
    extinction_new, extinction_child1, extinction_child2, extinction_complement = (
        output.detach()
        for output in e_step_triton_autograd(
            extinction,
            speciation_log_probability,
            duplication_log_probability,
            loss_log_probability,
            max_transfer,
            receiver_log_probs,
            species_parent,
            species_child1,
            species_child2,
            2,
            use_receiver_weights=False,
        )
    )
    torch.manual_seed(7)
    random_like = lambda value: torch.randn_like(value) * 0.1
    result = e_step_backward_so(
        extinction,
        extinction_new,
        extinction_child1,
        extinction_child2,
        extinction_complement,
        speciation_log_probability,
        duplication_log_probability,
        loss_log_probability,
        receiver_log_probs,
        species_parent,
        species_child1,
        species_child2,
        2,
        random_like(extinction_new),
        random_like(extinction_complement),
        random_like(extinction),
        random_like(extinction_new),
        random_like(extinction_child1),
        random_like(extinction_child2),
        random_like(extinction_complement),
        random_like(speciation_log_probability),
        random_like(duplication_log_probability),
        random_like(loss_log_probability),
        random_like(receiver_log_probs),
        use_receiver_weights=False,
    )
    receiver_gradient_directional_derivative = result[-1]
    torch.testing.assert_close(
        receiver_gradient_directional_derivative,
        torch.zeros_like(receiver_gradient_directional_derivative),
        rtol=0.0,
        atol=0.0,
    )
