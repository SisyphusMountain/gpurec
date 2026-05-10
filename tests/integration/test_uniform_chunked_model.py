"""Integration tests for the chunked global/uniform PyTorch API."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, UniformChunkedReconModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data" / "test_trees_20"
HOGENOM_DIR = _ROOT / "data" / "hogenom_bench"


def _genes(n: int) -> list[str]:
    if not DATA_DIR.exists():
        pytest.skip(f"dataset not present: {DATA_DIR}")
    genes = sorted(DATA_DIR.glob("g_*.nwk"))[:n]
    if len(genes) < n:
        pytest.skip(f"need {n} gene trees in {DATA_DIR}")
    return [str(p) for p in genes]


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
        pibar_mode="uniform",
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

    assert model.batch_summary()["chunks"] == 2
    assert model.batch_summary()["fixed_iters_E"] == 4
    assert torch.isfinite(model.theta).all()
    assert after < before


def test_chunked_uniform_to_float64_casts_static_state(tmp_path):
    model = UniformChunkedReconModel.from_folder(
        DATA_DIR,
        max_families=3,
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        preprocess_cache_dir=str(tmp_path),
        family_chunk_size=2,
        max_wave_size=32768,
        fixed_iters_Pi=6,
    )
    _ = model.nll()
    assert model._state.warm_E is not None

    model.to(torch.float64)

    assert model.theta.dtype == torch.float64
    assert model._state.dtype == torch.float64
    assert model._state.unnorm_row_max.dtype == torch.float64
    assert model._state.warm_E is None
    assert model.chunks[0].wave_layout["leaf_col_index"].dtype == torch.long

    loss = model()
    loss.backward()
    assert loss.dtype == torch.float64
    assert model.theta.grad.dtype == torch.float64
    assert torch.isfinite(model.theta.grad).all()


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

    summary = model.batch_summary()
    assert summary["families"] == 3
    assert summary["total_clades"] > 0
    assert summary["total_waves"] > 0
