from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(scope="module")
def one_gene_tree(data_dir_1000: Path):
    genes = sorted(data_dir_1000.glob("g_*.nwk"))[:1]
    if not genes:
        pytest.skip(f"need 1 gene tree in {data_dir_1000}")
    return str(data_dir_1000 / "sp.nwk"), [str(genes[0])]


@pytest.mark.slow
def test_adaptive_iterations_match_fixed_when_tolerances_force_max(
    one_gene_tree,
    tmp_path,
):
    sp, genes = one_gene_tree
    kwargs = dict(
        species_tree=sp,
        gene_trees=genes,
        mode="specieswise",
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_E=4,
        fixed_iters_Pi=4,
        neumann_terms=4,
    )
    fixed = GeneReconModel.from_trees(**kwargs)
    adaptive = GeneReconModel.from_trees(
        **kwargs,
        adaptive_iters=True,
        convergence_check_interval=4,
        e_logsumexp_tol=0.0,
        pi_max_diff_tol=0.0,
        gradient_change_tol=0.0,
        gradient_change_rtol=0.0,
    )

    loss_fixed = fixed()
    loss_adaptive = adaptive()
    loss_fixed.backward()
    loss_adaptive.backward()

    torch.testing.assert_close(loss_adaptive.detach(), loss_fixed.detach(), rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(adaptive.theta.grad, fixed.theta.grad, rtol=1e-5, atol=1e-4)
    assert fixed.static.last_solver_stats["Neumann_terms"] == 4
    assert adaptive.static.last_solver_stats["Neumann_terms"] == 4
    assert fixed.static.last_solver_stats["Pi_wave_iterations"]
    assert adaptive.static.last_solver_stats["Pi_wave_iterations"]
    assert fixed.static.last_solver_stats["E_adjoint_iterations"] >= 0
    assert adaptive.static.last_solver_stats["E_adjoint_iterations"] >= 0


@pytest.mark.slow
def test_adaptive_iterations_stop_before_caps_with_loose_tolerances(
    one_gene_tree,
):
    sp, genes = one_gene_tree
    fixed_iters = 8
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode="specieswise",
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_E=fixed_iters,
        fixed_iters_Pi=fixed_iters,
        neumann_terms=4,
        adaptive_iters=True,
        convergence_check_interval=2,
        e_logsumexp_tol=1e30,
        pi_max_diff_tol=1e30,
    )

    loss = model()
    loss.backward()

    assert torch.isfinite(loss)
    assert model.theta.grad is not None
    assert torch.isfinite(model.theta.grad).all()
    stats = model.static.last_solver_stats
    assert 0 < stats["E_iterations"] < fixed_iters
    assert 0 < stats["Pi_max_iterations"] < fixed_iters
    assert stats["Pi_wave_iterations"]
    assert max(stats["Pi_wave_iterations"]) < fixed_iters
    assert stats["Pi_converged_waves"] == stats["Pi_wave_count"]
    assert stats["Neumann_terms"] == 4
