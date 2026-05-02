"""Correctness tests for the autograd bridge in ``gpurec.api``.

Two complementary checks:

1. **NLL agreement** — ``model()`` returns exactly the same NLL as
   ``GeneDataset.compute_likelihood_batch`` for every (mode, pibar_mode) combo.
2. **Gradient FD agreement** — ``model.theta.grad`` (from ``loss.backward()``)
   matches central finite differences of ``model.forward()``.

Together these prove the autograd bridge is wired correctly: forward delegates
to the existing pipeline, backward delegates to the existing implicit gradient,
and the chain rule with ``grad_output`` is consistent.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel
from gpurec.api.autograd import _GeneReconFunction
from gpurec.core.model import GeneDataset


_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data" / "test_trees_20"
N_FAMILIES = 3


# (mode, pibar_mode) — same coverage as test_fd_all_modes.ALL_MODES, dropping
# the genewise+dense combo (not yet supported by the dense backward path,
# matching the existing test exclusions).
MODE_COMBOS = [
    ("global",      "uniform"),
    ("global",      "dense"),
    ("specieswise", "uniform"),
    ("specieswise", "dense"),
    ("genewise",    "uniform"),
]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def gene_paths() -> list[str]:
    if not DATA_DIR.exists():
        pytest.skip("test_trees_20 dataset not present")
    paths = sorted(DATA_DIR.glob("g_*.nwk"))[:N_FAMILIES]
    if len(paths) < N_FAMILIES:
        pytest.skip(f"Need {N_FAMILIES} gene families")
    return [str(p) for p in paths]


@pytest.fixture(scope="module")
def species_path() -> str:
    if not DATA_DIR.exists():
        pytest.skip("test_trees_20 dataset not present")
    return str(DATA_DIR / "sp.nwk")


def _build_model(species_path, gene_paths, mode, pibar_mode, dtype=torch.float64):
    return GeneReconModel.from_trees(
        species_tree=species_path,
        gene_trees=gene_paths,
        mode=mode,
        pibar_mode=pibar_mode,
        device=_device(),
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
    )


# ──────────────────────────────────────────────────────────────────────
# 1. NLL agreement
# ──────────────────────────────────────────────────────────────────────

def _sync_dataset_theta(model: GeneReconModel) -> None:
    """Copy ``model.theta`` into ``model._dataset.families[i]['theta']`` so the
    legacy ``GeneDataset.compute_likelihood_batch`` evaluates at the same point.
    GeneDataset stores per-family theta independently of model.theta."""
    theta = model.theta.detach()
    for i, fam in enumerate(model._dataset.families):
        if model.mode == "genewise":
            fam["theta"] = theta[i].clone()
        else:
            fam["theta"] = theta.clone()


@pytest.mark.parametrize("mode,pibar_mode", MODE_COMBOS)
def test_model_nll_matches_compute_likelihood_batch(
    species_path, gene_paths, mode, pibar_mode
):
    """``model()`` must return the same NLL as
    ``GeneDataset.compute_likelihood_batch`` for the same data and theta.
    """
    model = _build_model(species_path, gene_paths, mode, pibar_mode)

    # New API
    new_nll = float(model().item())

    # Existing API: sync theta into the dataset, then call the legacy path.
    _sync_dataset_theta(model)
    nll_per_fam = model._dataset.compute_likelihood_batch(
        pibar_mode=pibar_mode,
        max_iters_E=2000,
        tol_E=1e-8,
        max_iters_Pi=2000,
        tol_Pi=1e-6,
    )
    old_nll = float(sum(nll_per_fam))

    assert math.isclose(new_nll, old_nll, rel_tol=1e-9, abs_tol=1e-9), (
        f"new={new_nll}, old={old_nll}, delta={new_nll - old_nll}"
    )


# ──────────────────────────────────────────────────────────────────────
# 2. Gradient FD agreement (central differences)
# ──────────────────────────────────────────────────────────────────────

def _fd_grad(model: GeneReconModel, eps: float, indices: list[tuple]):
    """Central FD on a sparse subset of theta entries."""
    theta = model.theta
    grads = {}
    with torch.no_grad():
        for idx in indices:
            orig = theta[idx].clone()
            theta[idx] = orig + eps
            f_p = float(model().item())
            theta[idx] = orig - eps
            f_m = float(model().item())
            theta[idx] = orig
            # Force a fresh forward to clear any warm_E side effects
            _ = model()
            grads[idx] = (f_p - f_m) / (2 * eps)
    return grads


def _select_indices(theta: torch.Tensor, max_n: int = 6) -> list[tuple]:
    """Pick a deterministic, well-spread subset of indices to FD-check."""
    flat_idx = torch.linspace(0, theta.numel() - 1, steps=max_n).long().tolist()
    return [
        tuple(int(c) for c in torch.unravel_index(torch.tensor(i), theta.shape))
        for i in flat_idx
    ]


@pytest.mark.parametrize("mode,pibar_mode", MODE_COMBOS)
def test_autograd_matches_fd(species_path, gene_paths, mode, pibar_mode):
    """``loss.backward()`` gradient matches central FD on a sample of entries."""
    model = _build_model(species_path, gene_paths, mode, pibar_mode)

    # Analytic gradient via autograd bridge
    model.zero_grad()
    nll = model()
    nll.backward()
    grad_analytic = model.theta.grad.detach().clone()

    # Central FD
    eps = 1e-4
    idxs = _select_indices(model.theta, max_n=4)
    grad_fd = _fd_grad(model, eps, idxs)

    # Compare
    for idx in idxs:
        a = float(grad_analytic[idx].item())
        f = grad_fd[idx]
        # Tolerance: log2-space gradients near rates ~0.05; |grad| typically
        # in the 1-10 range. Use atol/rtol that match test_fd_all_modes.
        assert math.isclose(a, f, rel_tol=2e-3, abs_tol=2e-3), (
            f"mode={mode} pibar={pibar_mode} idx={idx}: "
            f"analytic={a:.6e}, fd={f:.6e}, diff={a - f:.3e}"
        )


# ──────────────────────────────────────────────────────────────────────
# 3. grad_output scaling
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,pibar_mode", [("global", "uniform")])
def test_grad_scales_with_loss_multiplier(species_path, gene_paths, mode, pibar_mode):
    """``(2 * loss).backward()`` should give 2x the gradient of ``loss.backward()``.

    Verifies that the autograd Function correctly multiplies its analytic
    gradient by ``grad_output``.
    """
    model = _build_model(species_path, gene_paths, mode, pibar_mode)

    model.zero_grad()
    model().backward()
    g1 = model.theta.grad.detach().clone()

    model.zero_grad()
    (2.0 * model()).backward()
    g2 = model.theta.grad.detach().clone()

    assert torch.allclose(g2, 2.0 * g1, rtol=1e-12, atol=1e-12), (
        f"g1={g1}, g2={g2}, ratio={(g2 / g1).mean().item()}"
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Per-family reduce (genewise only)
# ──────────────────────────────────────────────────────────────────────

def test_per_family_sums_to_total(species_path, gene_paths):
    """In genewise mode: ``nll_per_family().sum() == nll()`` and the gradient
    of ``per_fam.sum()`` matches the gradient of ``nll()`` exactly."""
    model = _build_model(species_path, gene_paths, "genewise", "uniform")

    model.zero_grad()
    per_fam = model.nll_per_family()
    total_via_per_fam = per_fam.sum()
    total_via_per_fam.backward()
    grad_A = model.theta.grad.detach().clone()

    model.zero_grad()
    total_direct = model()
    total_direct.backward()
    grad_B = model.theta.grad.detach().clone()

    assert math.isclose(
        float(total_via_per_fam.item()),
        float(total_direct.item()),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert torch.allclose(grad_A, grad_B, rtol=1e-12, atol=1e-12)


def test_gradcheck_global_uniform_small():
    """Autograd bridge backward matches torch's finite-difference gradcheck."""
    data_dir = _ROOT / "data" / "test_trees_3"
    if not data_dir.exists():
        pytest.skip("test_trees_3 dataset not present")

    model = GeneReconModel.from_trees(
        species_tree=str(data_dir / "sp.nwk"),
        gene_trees=[str(data_dir / "g.nwk")],
        mode="global",
        pibar_mode="uniform",
        device=_device(),
        dtype=torch.float64,
        theta_init_rates=(0.05, 0.05, 0.05),
        max_iters_E=2000,
        tol_E=1e-10,
        max_iters_Pi=2000,
        tol_Pi=1e-9,
        fixed_iters_Pi=6,
        neumann_terms=5,
        use_pruning=False,
    )
    theta = model.theta.detach().clone().requires_grad_(True)

    def fn(theta_in):
        model.static.warm_E = None
        return _GeneReconFunction.apply(theta_in, model.static, "sum")

    assert torch.autograd.gradcheck(
        fn,
        (theta,),
        eps=1e-4,
        atol=2e-3,
        rtol=2e-3,
        nondet_tol=1e-8,
        fast_mode=False,
    )


def test_per_family_rejects_non_genewise(species_path, gene_paths):
    """``nll_per_family()`` is invalid outside genewise mode."""
    model = _build_model(species_path, gene_paths, "global", "uniform")
    with pytest.raises(ValueError, match="genewise"):
        model.nll_per_family()


# ──────────────────────────────────────────────────────────────────────
# 5. Pairwise fail-fast
# ──────────────────────────────────────────────────────────────────────

def test_pairwise_dataset_raises(species_path, gene_paths):
    """A pairwise GeneDataset must be rejected at construction time."""
    ds = GeneDataset(
        species_tree_path=species_path,
        gene_tree_paths=gene_paths,
        genewise=False,
        specieswise=False,
        pairwise=True,
        dtype=torch.float64,
        device=_device(),
    )
    with pytest.raises(NotImplementedError, match="pairwise"):
        GeneReconModel(dataset=ds, mode="global")
