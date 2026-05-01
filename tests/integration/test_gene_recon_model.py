"""End-to-end integration tests for the new ``GeneReconModel`` API.

Covers:

* Convergence: NLL decreases monotonically with Adam.
* Equivalence: same trajectory as ``optimize_theta_wave`` when both run Adam
  with the same lr / theta_init.
* L-BFGS: standard PyTorch closure pattern works.
* Device / dtype roundtrip via ``.to``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel
from gpurec.optimization.theta_optimizer import optimize_theta_wave


_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data" / "test_trees_20"
N_FAMILIES = 5


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def trees():
    if not DATA_DIR.exists():
        pytest.skip("test_trees_20 dataset not present")
    sp = str(DATA_DIR / "sp.nwk")
    genes = sorted(DATA_DIR.glob("g_*.nwk"))[:N_FAMILIES]
    if len(genes) < N_FAMILIES:
        pytest.skip(f"Need {N_FAMILIES} gene families")
    return sp, [str(p) for p in genes]


# ──────────────────────────────────────────────────────────────────────
# 1. Adam loop converges (NLL decreases)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["global", "specieswise", "genewise"])
def test_adam_decreases_nll(trees, mode):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp, gene_trees=genes, mode=mode,
        device=_device(), dtype=torch.float64,
        theta_init_rates=(0.05, 0.05, 0.05),
    )
    nll_initial = float(model().item())

    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    history = [nll_initial]
    for _ in range(20):
        opt.zero_grad()
        loss = model()
        loss.backward()
        opt.step()
        model.clamp_theta_()
        history.append(float(model().item()))

    nll_final = history[-1]
    assert nll_final < nll_initial, f"NLL did not decrease: {history}"
    # Improvement should be substantial for 20 steps from a poor init
    assert nll_initial - nll_final > 5.0, (
        f"Improvement {nll_initial - nll_final} too small over 20 steps"
    )


# ──────────────────────────────────────────────────────────────────────
# 2. Equivalence with optimize_theta_wave (Adam optimizer)
# ──────────────────────────────────────────────────────────────────────

def test_adam_matches_optimize_theta_wave(trees):
    """Both code paths run torch.optim.Adam over the same gradient pipeline,
    so 20 steps should yield identical theta trajectories and identical
    per-step NLL histories. We compare the full history element-by-element.
    """
    import math

    sp, genes = trees
    device = _device()
    dtype = torch.float64
    lr = 0.1
    n_steps = 20

    # Path A: new API (record per-step NLL BEFORE the optimizer step, to
    # match optimize_theta_wave's history convention). Opt back into adaptive
    # Pi convergence so this path is bitwise-identical to optimize_theta_wave.
    model = GeneReconModel.from_trees(
        species_tree=sp, gene_trees=genes, mode="global",
        device=device, dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        max_iters_Pi=50,
        tol_Pi=1e-3,
        fixed_iters_Pi=None,
    )
    opt_new = torch.optim.Adam(model.parameters(), lr=lr)
    history_new: list[float] = []
    for _ in range(n_steps):
        opt_new.zero_grad()
        loss = model()
        history_new.append(float(loss.item()))
        loss.backward()
        opt_new.step()
        model.clamp_theta_()
    theta_new = model.theta.detach().cpu().clone()

    # Path B: existing optimize_theta_wave
    static = model._static
    theta_init_old = torch.full(
        (3,), math.log2(0.05), dtype=dtype, device=device
    )
    result_old = optimize_theta_wave(
        wave_layout=static.wave_layout,
        species_helpers=static.species_helpers,
        root_clade_ids=static.root_clade_ids,
        unnorm_row_max=static.unnorm_row_max,
        theta_init=theta_init_old,
        steps=n_steps,
        lr=lr,
        tol_theta=0.0,                 # disable early stopping
        optimizer="adam",
        specieswise=False,
        device=device,
        dtype=dtype,
        pibar_mode="uniform",
        verbose=False,
    )
    theta_old = result_old["theta"].detach().cpu()
    history_old = [r.negative_log_likelihood for r in result_old["history"]]

    # Final theta agreement (computed after the last opt.step() in both)
    assert torch.allclose(theta_new, theta_old, rtol=1e-10, atol=1e-10), (
        f"theta diverged: new={theta_new}, old={theta_old}, "
        f"max_abs_diff={(theta_new - theta_old).abs().max().item():.3e}"
    )

    # Per-step NLL trajectory agreement
    assert len(history_new) == len(history_old) == n_steps
    for i, (a, b) in enumerate(zip(history_new, history_old)):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (
            f"NLL diverged at step {i}: new={a}, old={b}, diff={a - b:.3e}"
        )


def test_preprocess_cache_matches_single_path(trees, tmp_path, monkeypatch):
    sp, genes = trees
    genes = genes[:3]
    kwargs = dict(
        species_tree=sp,
        gene_trees=genes,
        mode="global",
        pibar_mode="uniform",
        device=_device(),
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
    )

    monkeypatch.setenv("GPUREC_PREPROCESS_MODE", "single")
    single = GeneReconModel.from_trees(**kwargs)
    nll_single = float(single().item())

    monkeypatch.delenv("GPUREC_PREPROCESS_MODE", raising=False)
    cached_populate = GeneReconModel.from_trees(
        **kwargs,
        preprocess_cache_dir=tmp_path,
    )
    nll_populate = float(cached_populate().item())

    cached_hit = GeneReconModel.from_trees(
        **kwargs,
        preprocess_cache_dir=tmp_path,
    )
    nll_hit = float(cached_hit().item())

    assert nll_populate == nll_single
    assert nll_hit == nll_single
    assert list(tmp_path.glob("family-*.pt"))
    assert list(tmp_path.glob("species-*.pt"))


# ──────────────────────────────────────────────────────────────────────
# 3. L-BFGS closure pattern
# ──────────────────────────────────────────────────────────────────────

def test_lbfgs_closure_runs(trees):
    """Standard PyTorch L-BFGS closure pattern should work end-to-end."""
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp, gene_trees=genes, mode="global",
        device=_device(), dtype=torch.float64,
        theta_init_rates=(0.05, 0.05, 0.05),
    )
    nll_initial = float(model().item())

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=10,
        history_size=5,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = model()
        loss.backward()
        return loss

    opt.step(closure)

    nll_final = float(model().item())
    assert nll_final < nll_initial, (
        f"L-BFGS did not improve NLL: {nll_initial} -> {nll_final}"
    )


# ──────────────────────────────────────────────────────────────────────
# 4. dtype / device roundtrip
# ──────────────────────────────────────────────────────────────────────

def test_dtype_roundtrip(trees):
    """``model.to(torch.float64)`` should move all internal tensors and
    leave Long index tensors untouched."""
    sp, genes = trees
    if not torch.cuda.is_available():
        pytest.skip("Need CUDA for the float32 path")
    model = GeneReconModel.from_trees(
        species_tree=sp, gene_trees=genes, mode="global",
        device="cuda", dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
    )
    assert model.theta.dtype == torch.float32

    model.to(torch.float64)
    assert model.theta.dtype == torch.float64
    assert model._static.unnorm_row_max.dtype == torch.float64

    # Long index tensor inside wave_layout must remain Long
    leaf_col = model._static.wave_layout["leaf_col_index"]
    assert leaf_col.dtype == torch.long, (
        f"Long index tensor was cast to {leaf_col.dtype}"
    )

    # Forward pass should still work
    nll = model()
    assert torch.isfinite(nll), "NLL is non-finite after dtype switch"
    nll.backward()
    assert torch.isfinite(model.theta.grad).all()


def test_warm_e_resets_on_to(trees):
    """``warm_E`` should be cleared on ``.to`` to avoid stale-device tensors."""
    sp, genes = trees
    if not torch.cuda.is_available():
        pytest.skip("Need CUDA for warm_E path")
    model = GeneReconModel.from_trees(
        species_tree=sp, gene_trees=genes, mode="global",
        device="cuda", dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
    )
    # Populate warm_E
    _ = model()
    assert model._static.warm_E is not None
    # Move dtype: warm_E should be cleared
    model.to(torch.float64)
    assert model._static.warm_E is None


# ──────────────────────────────────────────────────────────────────────
# 5. Properties
# ──────────────────────────────────────────────────────────────────────

def test_properties(trees):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp, gene_trees=genes, mode="genewise",
        device=_device(), dtype=torch.float64,
        theta_init_rates=(0.05, 0.05, 0.05),
    )
    assert model.mode == "genewise"
    assert model.n_families == N_FAMILIES
    assert model.n_species > 0
    rates = model.rates
    assert rates.shape == (N_FAMILIES, 3)
    # 2^log2(0.05) == 0.05
    assert torch.allclose(rates, torch.full_like(rates, 0.05), rtol=1e-12)


def test_log_likelihood_helper(trees):
    """``log_likelihood()`` returns the negation of ``forward()`` as a float."""
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp, gene_trees=genes, mode="global",
        device=_device(), dtype=torch.float64,
        theta_init_rates=(0.05, 0.05, 0.05),
    )
    nll = float(model().item())
    logL = model.log_likelihood()
    assert isinstance(logL, float)
    assert logL == pytest.approx(-nll, rel=1e-12, abs=1e-12)
