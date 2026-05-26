import torch

from gpurec.optimization import ProjectedLBFGS


def test_projected_lbfgs_restores_parameter_when_line_search_rejects_all_probes():
    theta = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = ProjectedLBFGS(
        [theta],
        lr=10.0,
        max_ls=3,
        lower_bound=-100.0,
        upper_bound=100.0,
    )

    def closure():
        optimizer.zero_grad()
        loss = theta.square().sum()
        loss.backward()
        return loss

    def rejecting_loss_closure():
        return theta.new_tensor(1.0e9)

    loss = optimizer.step(closure, loss_closure=rejecting_loss_closure)

    state = optimizer.state[theta]
    assert torch.equal(theta.detach(), torch.tensor([1.0]))
    assert loss.item() == 1.0
    assert state["last_accepted"] is False
    assert state["last_alpha"] == 0.0
    assert state["last_step_inf"] == 0.0
    assert state["last_loss_evals"] == 3
