"""Integration tests for the chunked global/uniform PyTorch API."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data" / "test_trees_1000"
HOGENOM_DIR = _ROOT / "data" / "hogenom_bench"

from gpurec import GeneReconModel, UniformChunkedReconModel


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not DATA_DIR.exists(), reason=f"dataset not present: {DATA_DIR}"),
]


def _genes(n: int) -> list[str]:
    if not DATA_DIR.exists():
        pytest.skip(f"dataset not present: {DATA_DIR}")
    genes = sorted(DATA_DIR.glob("g_*.nwk"))[:n]
    if len(genes) < n:
        pytest.skip(f"need {n} gene trees in {DATA_DIR}")
    return [str(p) for p in genes]


@pytest.mark.slow
def test_chunked_uniform_matches_resident_global_model(tmp_path):
    species_tree = str(DATA_DIR / "sp.nwk")
    genes = _genes(4)
    kwargs = dict(
        species_tree=species_tree,
        gene_trees=genes,
        device="cuda",
        dtype=torch.float64,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_Pi=6,
        neumann_terms=3,
        use_pruning=True,
        pruning_threshold=1e-6,
        preprocess_cache_dir=str(tmp_path),
    )

    resident = GeneReconModel.from_trees(
        mode="global",
        max_wave_size=32768,
        **kwargs,
    )
    chunked = UniformChunkedReconModel(
        family_chunk_size=2,
        max_wave_size=32768,
        profile=False,
        **kwargs,
    )

    resident_loss = resident()
    chunked_loss = chunked()
    resident_loss.backward()
    chunked_loss.backward()

    torch.testing.assert_close(chunked_loss, resident_loss, rtol=1e-10, atol=1e-8)
    torch.testing.assert_close(
        chunked.theta.grad,
        resident.theta.grad,
        rtol=1e-7,
        atol=1e-6,
    )


def test_chunked_uniform_from_folder_and_adam_step(tmp_path):
    model = UniformChunkedReconModel.from_folder(
        DATA_DIR,
        max_families=3,
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        preprocess_cache_dir=str(tmp_path),
        family_chunk_size=2,
        max_wave_size=32768,
        fixed_iters_E=4,
        fixed_iters_Pi=6,
    )
    before = float(model.nll().item())
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = model()
        loss.backward()
        opt.step()
        model.clamp_theta_()
    after = float(model.nll().item())

    assert len(model._state.built_chunks) == 2
    assert model._state.fixed_iters_E == 4
    assert torch.isfinite(model.theta).all()
    assert after < before


def test_chunked_uniform_chunk_subset_nll_and_gradient(tmp_path):
    model = UniformChunkedReconModel.from_folder(
        DATA_DIR,
        max_families=4,
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        preprocess_cache_dir=str(tmp_path),
        family_chunk_size=2,
        max_wave_size=32768,
        fixed_iters_Pi=6,
        neumann_terms=4,
        use_pruning=False,
    )

    assert len(model._state.built_chunks) == 2
    full_vec = model.nll_per_family()
    chunk0 = model.nll_per_family(chunk_indices=[0])
    chunk1 = model.nll_per_family(chunk_indices=[1])
    torch.testing.assert_close(
        torch.cat([chunk0, chunk1]),
        full_vec,
        rtol=1e-6,
        atol=1e-4,
    )

    model.zero_grad(set_to_none=True)
    full_loss = model()
    full_loss.backward()
    full_grad = model.theta.grad.detach().clone()

    subset_loss, subset_grad, stats = model.loss_and_grad(chunk_indices=[0, 1])
    torch.testing.assert_close(subset_loss, full_loss.detach(), rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(subset_grad, full_grad, rtol=1e-5, atol=1e-4)
    assert stats["selected_families"] == 4
    assert stats["selected_chunks"] == [0, 1]

    mean_loss, mean_grad, mean_stats = model.loss_and_grad(
        chunk_indices=[0],
        reduction="mean",
    )
    selected = mean_stats["selected_families"]
    sum_loss, sum_grad, _ = model.loss_and_grad(chunk_indices=[0], reduction="sum")
    torch.testing.assert_close(mean_loss * selected, sum_loss, rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(mean_grad * selected, sum_grad, rtol=1e-5, atol=1e-4)


def test_chunked_uniform_accepts_hogenom_unrooted_binary_newick(tmp_path):
    if not HOGENOM_DIR.exists():
        pytest.skip(f"dataset not present: {HOGENOM_DIR}")

    model = UniformChunkedReconModel.from_folder(
        HOGENOM_DIR,
        max_families=3,
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        preprocess_cache_dir=str(tmp_path),
        family_chunk_size=3,
        max_wave_size=32768,
        fixed_iters_Pi=6,
    )

    assert len(model._state.dataset.families) == 3
    assert sum(c.spec.clades for c in model._state.built_chunks) > 0
    assert sum(c.waves for c in model._state.built_chunks) > 0
