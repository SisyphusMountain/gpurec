from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel
from gpurec.optimization import BatchedLBFGS


_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data" / "test_trees_3"


@pytest.fixture
def trees():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not DATA_DIR.exists():
        pytest.skip("test_trees_3 dataset not present")
    return str(DATA_DIR / "sp.nwk"), [str(DATA_DIR / "g.nwk")]


@pytest.mark.parametrize("mode", ["global", "specieswise", "genewise"])
def test_gene_recon_model_forward_backward_modes(trees, mode):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode=mode,
        pibar_mode="uniform",
        device="cuda",
        dtype=torch.float32,
        fixed_iters_Pi=2,
        neumann_terms=2,
    )

    loss = model()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.theta.grad is not None
    assert model.theta.grad.shape == model.theta.shape
    assert torch.isfinite(model.theta.grad).all()


def test_pytorch_adam_updates_global_model(trees):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode="global",
        pibar_mode="uniform",
        device="cuda",
        dtype=torch.float32,
        fixed_iters_Pi=2,
        neumann_terms=2,
    )
    opt = torch.optim.Adam([model.theta], lr=0.01)

    before = model.theta.detach().clone()
    opt.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    opt.step()
    model.clamp_theta_(min_rate=1e-10, max_rate=2.0)

    assert not torch.equal(model.theta.detach(), before)
    assert torch.isfinite(model.theta).all()


def test_batched_lbfgs_genewise_runs_one_polish_step(trees):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode="genewise",
        pibar_mode="uniform",
        device="cuda",
        dtype=torch.float32,
        fixed_iters_Pi=2,
        neumann_terms=2,
    )
    opt = BatchedLBFGS(
        [model.theta],
        lr=0.5,
        max_iter=1,
        line_search_fn="armijo",
        max_ls=2,
    )

    def closure():
        opt.zero_grad(set_to_none=True)
        losses = model.nll_per_family()
        losses.sum().backward()
        return losses.detach()

    before = model.theta.detach().clone()
    losses = opt.step(closure)
    model.clamp_theta_(min_rate=1e-10, max_rate=2.0)

    assert losses.shape == (1,)
    assert torch.isfinite(losses).all()
    assert not torch.equal(model.theta.detach(), before)
