import math
from pathlib import Path

import pytest
import torch

from gpurec import BatchedLBFGS, GeneReconModel, SolverOptions
from gpurec.core.inference.solver import nll_from_root_rows, nll_vector_from_root_rows
from gpurec.core.scheduling import batching


def _require_native_cuda() -> None:
    try:
        batching._load_native_module()
    except Exception as exc:
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _write_tiny_genewise_problem(tmp_path: Path) -> tuple[Path, list[Path]]:
    species_tree = tmp_path / "species.nwk"
    species_tree.write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
    gene_trees = []
    for idx, newick in enumerate(
        [
            "(A_1:1,(B_1:1,C_1:1)X:1)R:1;\n",
            "((A_2:1,B_2:1)X:1,C_2:1)R:1;\n",
        ]
    ):
        path = tmp_path / f"family_{idx}.nwk"
        path.write_text(newick, encoding="utf-8")
        gene_trees.append(path)
    return species_tree, gene_trees


def _tiny_genewise_model(tmp_path: Path) -> GeneReconModel:
    _require_native_cuda()
    species_tree, gene_trees = _write_tiny_genewise_problem(tmp_path)
    model = GeneReconModel(
        species_tree,
        gene_trees,
        mode="genewise",
        device="cuda",
        family_chunk_size=2,
        clade_budget=None,
        max_wave_size=8,
        solver_options=SolverOptions(e_max_iter=100, e_tol=1e-8, pi_iters=4, neumann_terms=4),
    )
    model.to(dtype=torch.float64)
    model.receiver_weights.requires_grad_(False)
    with torch.no_grad():
        model.theta.fill_(math.log2(0.05))
        model.clear_warm_starts()
    return model


def test_batched_lbfgs_converges_independent_quadratics():
    theta = torch.nn.Parameter(
        torch.tensor(
            [[4.0, -3.0], [-2.0, 5.0], [0.5, -4.0]],
            dtype=torch.float64,
        )
    )
    target = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [-2.0, 1.0]],
        dtype=torch.float64,
    )
    scale = torch.tensor([1e-3, 1.0, 1e3], dtype=torch.float64)
    optimizer = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=3,
        history_size=5,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )

    def loss_vec() -> torch.Tensor:
        return scale * ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    start = loss_vec().detach().clone()
    for _ in range(4):
        final = optimizer.step(closure, loss_closure=loss_vec)

    torch.testing.assert_close(theta.detach(), target, rtol=0.0, atol=1e-8)
    assert torch.all(final < start * 1e-12)


def test_batched_lbfgs_uses_rowwise_line_search():
    theta = torch.nn.Parameter(torch.tensor([[2.0], [2.0]], dtype=torch.float64))
    target = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    scale = torch.tensor([1e-3, 1e6], dtype=torch.float64)
    optimizer = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=2,
        history_size=3,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )

    def loss_vec() -> torch.Tensor:
        return scale * ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    start = loss_vec().detach().clone()
    final = optimizer.step(closure, loss_closure=loss_vec)

    assert torch.all(final < start)
    assert abs(float(theta.detach()[0, 0]) - 1.0) < 1e-6
    assert abs(float(theta.detach()[1, 0]) + 1.0) < 1e-6


def test_batched_lbfgs_projects_to_bounds():
    theta = torch.nn.Parameter(torch.tensor([[0.0], [0.0]], dtype=torch.float64))
    target = torch.tensor([[-10.0], [2.0]], dtype=torch.float64)
    optimizer = BatchedLBFGS(
        [theta],
        lr=1.0,
        max_iter=3,
        history_size=3,
        lower_bound=-1.0,
    )

    def loss_vec() -> torch.Tensor:
        return ((theta - target) ** 2).sum(dim=1)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_vec()
        loss.sum().backward()
        return loss

    for _ in range(3):
        optimizer.step(closure, loss_closure=loss_vec)

    assert torch.all(theta.detach() >= -1.0)
    assert math.isclose(float(theta.detach()[0, 0]), -1.0, abs_tol=1e-12)
    assert math.isclose(float(theta.detach()[1, 0]), 2.0, abs_tol=1e-8)


def test_nll_vector_sums_to_scalar_loss():
    root_rows = torch.tensor(
        [[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]],
        dtype=torch.float64,
    )
    E = torch.tensor(
        [[-4.0, -5.0, -6.0], [-3.0, -4.0, -5.0]],
        dtype=torch.float64,
    )

    loss_vec = nll_vector_from_root_rows(root_rows, E)
    scalar = nll_from_root_rows(root_rows, E)

    torch.testing.assert_close(scalar, loss_vec.sum())
    assert loss_vec.shape == (2,)


def test_genewise_loss_vector_default_does_not_mutate_warm_starts(tmp_path: Path):
    model = _tiny_genewise_model(tmp_path)

    model.genewise_loss_vector()
    assert all(static.warm_E is None for static in model.batch_statics)

    model.genewise_loss_vector(update_warm_starts=True)
    assert all(static.warm_E is not None for static in model.batch_statics)


def test_batched_lbfgs_moves_tiny_genewise_model(tmp_path: Path):
    model = _tiny_genewise_model(tmp_path)
    start_theta = model.theta.detach().clone()
    start_loss = float(model().detach().cpu())
    model.clear_warm_starts()

    optimizer = BatchedLBFGS(
        [model.theta],
        lr=0.25,
        max_iter=1,
        history_size=10,
        max_ls=8,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure() -> torch.Tensor:
        loss_vec, grad_theta, _grad_receiver = model.genewise_loss_vector_and_grad(need_grad=True)
        assert grad_theta is not None
        model.theta.grad = grad_theta
        return loss_vec

    def loss_closure() -> torch.Tensor:
        return model.genewise_loss_vector()

    final_vec = optimizer.step(closure, loss_closure=loss_closure)
    final_loss = float(model().detach().cpu())
    opt_state = optimizer.state[model.theta]

    assert bool(opt_state["last_accepted"].all().item())
    assert final_loss < start_loss
    assert float((model.theta.detach() - start_theta).abs().max().cpu()) > 1e-3
    assert float(final_vec.sum().detach().cpu()) == pytest.approx(final_loss, rel=0.0, abs=1e-8)
