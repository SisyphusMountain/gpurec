from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel
from gpurec.optimization import BatchedLBFGS


_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data" / "test_trees_1000"


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture
def trees():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not DATA_DIR.exists():
        pytest.skip("test_trees_1000 dataset not present")
    genes = sorted(DATA_DIR.glob("g_*.nwk"))[:1]
    if not genes:
        pytest.skip("test_trees_1000 gene trees not present")
    return str(DATA_DIR / "sp.nwk"), [str(genes[0])]


@pytest.mark.parametrize("mode", ["global", "specieswise", "genewise"])
def test_gene_recon_model_forward_backward_modes(trees, mode):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode=mode,
        device="cuda",
        dtype=torch.float32,
        fixed_iters_Pi=2,
        neumann_terms=2,
    )

    pi = model.pi_matrix()
    assert pi.ndim == 2
    assert pi.shape[1] == model.n_species
    assert torch.isfinite(pi).any()

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


def test_uniform_origination_probs_match_default_global_model(tmp_path, trees):
    sp, genes = trees
    kwargs = dict(
        species_tree=sp,
        gene_trees=genes,
        mode="global",
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_E=2,
        fixed_iters_Pi=2,
        neumann_terms=1,
        preprocess_cache_dir=tmp_path / "preprocess",
    )
    default = GeneReconModel.from_trees(**kwargs)
    uniform = torch.ones(default.n_species, device="cuda", dtype=torch.float32)
    weighted = GeneReconModel.from_trees(
        **kwargs,
        origination_probs=uniform,
    )

    default_loss = default()
    weighted_loss = weighted()
    default_loss.backward()
    weighted_loss.backward()

    torch.testing.assert_close(weighted_loss.detach(), default_loss.detach(), rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(weighted.theta.grad, default.theta.grad, rtol=1e-5, atol=1e-4)


def test_batched_lbfgs_genewise_runs_one_polish_step(trees):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode="genewise",
        device="cuda",
        dtype=torch.float32,
        fixed_iters_Pi=2,
        neumann_terms=2,
    )
    opt = BatchedLBFGS(
        [model.theta],
        lr=0.5,
        max_iter=1,
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


@pytest.fixture
def two_trees():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not DATA_DIR.exists():
        pytest.skip("test_trees_1000 dataset not present")
    genes = sorted(DATA_DIR.glob("g_*.nwk"))[:2]
    if len(genes) < 2:
        pytest.skip("need at least two gene trees")
    return str(DATA_DIR / "sp.nwk"), [str(g) for g in genes]


@pytest.mark.parametrize("mode", ["global", "specieswise", "genewise"])
def test_memory_safe_resident_batches_match_resident_and_slice(tmp_path, two_trees, mode):
    sp, genes = two_trees
    kwargs = dict(
        mode=mode,
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_E=2,
        fixed_iters_Pi=2,
        neumann_terms=1,
        preprocess_cache_dir=tmp_path / "preprocess",
    )
    resident = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        **kwargs,
    )
    batched = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        family_chunk_size=1,
        lazy_preprocess=True,
        prefetch_batches=0,
        **kwargs,
    )
    sliced = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes[:1],
        **kwargs,
    )

    assert len(batched.batch_metadata) == 2
    assert batched.current_batch_metadata.family_indices == (0,)
    assert batched.current_batch_metadata.family_names == (
        batched.family_names[0],
    )
    if mode == "genewise":
        assert batched.current_batch_metadata.parameter_mapping["batch_theta_rows"] == (0,)

    active_loss = batched()
    sliced_loss = sliced()
    active_loss.backward()
    sliced_loss.backward()
    torch.testing.assert_close(active_loss.detach(), sliced_loss.detach(), rtol=1e-5, atol=1e-4)
    if mode == "genewise":
        torch.testing.assert_close(
            batched.theta.grad[0],
            sliced.theta.grad[0],
            rtol=1e-4,
            atol=1e-3,
        )
        torch.testing.assert_close(
            batched.theta.grad[1],
            torch.zeros_like(batched.theta.grad[1]),
            rtol=0.0,
            atol=0.0,
        )
    else:
        torch.testing.assert_close(
            batched.theta.grad,
            sliced.theta.grad,
            rtol=1e-4,
            atol=1e-3,
        )

    batched.theta.grad = None
    resident.theta.grad = None
    batched.clear()

    resident_loss = resident()
    full_loss = batched.full_loss()
    resident_loss.backward()
    full_loss.backward()
    torch.testing.assert_close(full_loss.detach(), resident_loss.detach(), rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(
        batched.theta.grad,
        resident.theta.grad,
        rtol=1e-4,
        atol=1e-3,
    )

    next_meta = batched.next()
    assert next_meta.family_indices == (1,)
