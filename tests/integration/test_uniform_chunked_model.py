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
]


def _require_data_dir() -> None:
    if not DATA_DIR.exists():
        pytest.skip(f"dataset not present: {DATA_DIR}")


def _genes(n: int) -> list[str]:
    _require_data_dir()
    genes = sorted(DATA_DIR.glob("g_*.nwk"))[:n]
    if len(genes) < n:
        pytest.skip(f"need {n} gene trees in {DATA_DIR}")
    return [str(p) for p in genes]


@pytest.mark.slow
def test_chunked_uniform_matches_resident_global_model(tmp_path):
    _require_data_dir()
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
    _require_data_dir()
    model = UniformChunkedReconModel.from_folder(
        DATA_DIR,
        max_families=3,
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
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

    assert model.chunk_count == 2
    assert model.fixed_iters_E == 4
    assert torch.isfinite(model.theta).all()
    assert after < before


def test_chunked_uniform_rust_layout_is_deterministic():
    _require_data_dir()

    def build() -> UniformChunkedReconModel:
        return UniformChunkedReconModel.from_folder(
            DATA_DIR,
            max_families=4,
            device="cuda",
            dtype=torch.float32,
            theta_init_rates=(0.05, 0.05, 0.05),
            family_chunk_size=2,
            max_wave_size=128,
            fixed_iters_Pi=6,
        )

    first = build()
    second = build()

    assert [
        (
            chunk.family_indices,
            chunk.clade_count,
            chunk.split_count,
            chunk.wave_count,
            chunk.max_wave_split_rows,
        )
        for chunk in first.chunk_metadata
    ] == [
        (
            chunk.family_indices,
            chunk.clade_count,
            chunk.split_count,
            chunk.wave_count,
            chunk.max_wave_split_rows,
        )
        for chunk in second.chunk_metadata
    ]
    torch.testing.assert_close(
        first.nll_per_family(),
        second.nll_per_family(),
        rtol=1e-6,
        atol=1e-5,
    )


def test_chunked_uniform_chunk_subset_nll_and_gradient(tmp_path):
    _require_data_dir()
    model = UniformChunkedReconModel.from_folder(
        DATA_DIR,
        max_families=5,
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        family_chunk_size=2,
        max_wave_size=32768,
        fixed_iters_E=4,
        fixed_iters_Pi=6,
        neumann_terms=4,
        use_pruning=False,
        warm_start_E=False,
    )

    assert model.chunk_count == 3
    all_chunks = list(range(model.chunk_count))
    selected_chunks = [2, 0]

    full_vec = model.nll_per_family()
    explicit_full_vec = model.nll_per_family(chunk_indices=all_chunks)
    torch.testing.assert_close(
        explicit_full_vec,
        full_vec,
        rtol=1e-6,
        atol=1e-4,
    )
    selected_vec = model.nll_per_family(
        chunk_indices=torch.tensor(selected_chunks, device=model.theta.device)
    )
    expected_selected_vec = torch.cat(
        [
            full_vec[list(model.chunk_metadata[idx].family_indices)]
            for idx in selected_chunks
        ]
    )
    torch.testing.assert_close(
        selected_vec,
        expected_selected_vec,
        rtol=1e-6,
        atol=1e-4,
    )
    selected_nll = model.nll(chunk_indices=selected_chunks)
    torch.testing.assert_close(selected_nll, selected_vec.sum(), rtol=1e-6, atol=1e-4)

    model.zero_grad(set_to_none=True)
    full_loss = model()
    full_loss.backward()
    full_grad = model.theta.grad.detach().clone()

    subset_loss, subset_grad, stats = model.loss_and_grad(chunk_indices=all_chunks)
    torch.testing.assert_close(subset_loss, full_loss.detach(), rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(subset_grad, full_grad, rtol=1e-5, atol=1e-4)
    assert stats["selected_families"] == model.n_families
    assert stats["total_families"] == model.n_families
    assert stats["selected_chunks"] == all_chunks
    assert stats["reduction"] == "sum"
    assert stats["scale"] == 1.0

    sum_loss, sum_grad, sum_stats = model.loss_and_grad(
        chunk_indices=selected_chunks,
        reduction="sum",
    )
    torch.testing.assert_close(sum_loss, selected_nll, rtol=1e-6, atol=1e-4)
    assert sum_stats["selected_chunks"] == selected_chunks
    assert sum_stats["selected_families"] == selected_vec.numel()
    assert sum_stats["total_families"] == model.n_families
    assert sum_stats["reduction"] == "sum"
    assert sum_stats["scale"] == 1.0
    assert [row["idx"] for row in sum_stats["chunk_rows"]] == selected_chunks
    assert [row["families"] for row in sum_stats["chunk_rows"]] == [
        model.chunk_metadata[idx].family_count
        for idx in selected_chunks
    ]
    assert sum_stats["reduced_loss"] == pytest.approx(
        float(sum_loss.detach().cpu()),
        rel=1e-6,
        abs=1e-4,
    )
    assert sum_stats["reduced_grad_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(sum_grad).detach().cpu()),
        rel=1e-6,
        abs=1e-4,
    )

    mean_loss, mean_grad, mean_stats = model.loss_and_grad(
        chunk_indices=selected_chunks,
        reduction="mean",
    )
    selected = mean_stats["selected_families"]
    torch.testing.assert_close(mean_loss * selected, sum_loss, rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(mean_grad * selected, sum_grad, rtol=1e-5, atol=1e-4)
    assert mean_stats["selected_chunks"] == selected_chunks
    assert mean_stats["reduction"] == "mean"
    assert mean_stats["scale"] == pytest.approx(1.0 / selected)

    estimate_loss, estimate_grad, estimate_stats = model.loss_and_grad(
        chunk_indices=selected_chunks,
        reduction="full_sum_estimate",
    )
    scale = model.n_families / selected
    torch.testing.assert_close(estimate_loss, sum_loss * scale, rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(estimate_grad, sum_grad * scale, rtol=1e-5, atol=1e-4)
    assert estimate_stats["selected_chunks"] == selected_chunks
    assert estimate_stats["reduction"] == "full_sum_estimate"
    assert estimate_stats["scale"] == pytest.approx(scale)
    assert estimate_stats["reduced_loss"] == pytest.approx(
        float(estimate_loss.detach().cpu()),
        rel=1e-6,
        abs=1e-4,
    )
    assert estimate_stats["reduced_grad_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(estimate_grad).detach().cpu()),
        rel=1e-6,
        abs=1e-4,
    )


@pytest.mark.slow
def test_chunked_uniform_accepts_hogenom_unrooted_binary_newick(tmp_path):
    if not HOGENOM_DIR.exists():
        pytest.skip(f"dataset not present: {HOGENOM_DIR}")

    model = UniformChunkedReconModel.from_folder(
        HOGENOM_DIR,
        max_families=3,
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        family_chunk_size=3,
        max_wave_size=32768,
        fixed_iters_Pi=6,
    )

    metadata = model.chunk_metadata
    assert model.n_families == 3
    assert sum(chunk.clade_count for chunk in metadata) > 0
    assert sum(chunk.wave_count for chunk in metadata) > 0

    per_family = model.nll_per_family()
    loss = model()
    loss.backward()
    backward_grad = model.theta.grad.detach().clone()
    direct_loss, direct_grad, stats = model.loss_and_grad()

    assert per_family.shape == (model.n_families,)
    assert torch.isfinite(per_family).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(backward_grad).all()
    assert torch.isfinite(direct_grad).all()
    torch.testing.assert_close(per_family.sum(), loss.detach(), rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(direct_loss, loss.detach(), rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(direct_grad, backward_grad, rtol=1e-5, atol=1e-4)
    assert stats["selected_families"] == model.n_families
    assert stats["total_families"] == model.n_families
    assert stats["reduction"] == "sum"
