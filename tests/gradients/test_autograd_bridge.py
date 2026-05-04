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


def _loss_and_grad(model: GeneReconModel) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    model._static.warm_E = None
    loss = model()
    loss.backward()
    return float(loss.item()), model.theta.grad.detach().clone()


def test_uniform_static_state_runs_without_dense_species_matrices(
    species_path, gene_paths
):
    """Uniform mode must not need dense transfer matrices after construction.

    This simulates Proposal 8's lazy preprocessing boundary by deleting the
    dense recipients matrix from the runtime dataset after ``ReconStaticState``
    has been built. Forward and backward should match an unmodified uniform
    model because uniform extraction only needs ``unnorm_row_max`` and
    ``ancestors_T``.
    """
    baseline = _build_model(species_path, gene_paths, "global", "uniform")
    lazy = _build_model(species_path, gene_paths, "global", "uniform")
    lazy.theta.data.copy_(baseline.theta.detach())

    assert lazy._static.transfer_mat_unnormalized is None
    assert lazy._static.ancestors_T is not None

    lazy._static.species_helpers.pop("Recipients_mat", None)
    lazy._dataset.tr_mat_unnormalized = None
    lazy._dataset.species_helpers = {
        k: v for k, v in lazy._dataset.species_helpers.items()
        if k != "Recipients_mat"
    }
    for fam in lazy._dataset.families:
        fam["transfer_mat_unnormalized"] = None

    baseline_nll, baseline_grad = _loss_and_grad(baseline)
    lazy_nll, lazy_grad = _loss_and_grad(lazy)

    assert lazy_nll == pytest.approx(baseline_nll, rel=1e-12, abs=1e-12)
    assert torch.allclose(lazy_grad, baseline_grad, rtol=1e-10, atol=1e-10)


def test_uniform_model_construction_omits_dense_species_matrix(
    species_path, gene_paths
):
    model = _build_model(species_path, gene_paths, "global", "uniform")

    assert "Recipients_mat" not in model._static.species_helpers
    assert "ancestors_dense" not in model._static.species_helpers
    assert model._dataset.tr_mat_unnormalized is None
    assert "Recipients_mat" not in model._dataset.species_helpers
    assert "ancestors_dense" not in model._dataset.species_helpers
    assert model._static.transfer_mat_unnormalized is None
    assert model._static.ancestors_T is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fallback check")
@pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta state:UserWarning")
def test_uniform_torch_spmm_fallback_matches_default_without_dense_species_matrix(
    species_path, gene_paths, monkeypatch
):
    baseline = _build_model(species_path, gene_paths, "global", "uniform")
    fallback = _build_model(species_path, gene_paths, "global", "uniform")
    fallback.theta.data.copy_(baseline.theta.detach())

    baseline_loss = baseline().detach()
    baseline._static.warm_E = None
    fallback._static.warm_E = None
    monkeypatch.setenv("GPUREC_UNIFORM_IMPL", "torch_spmm")
    fallback_loss = fallback().detach()

    assert "ancestors_dense" not in fallback._static.species_helpers
    assert torch.allclose(fallback_loss, baseline_loss, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("pibar_mode", ["dense", "topk"])
def test_dense_and_topk_static_state_retain_dense_transfer_matrix(
    species_path, gene_paths, pibar_mode
):
    model = _build_model(species_path, gene_paths, "global", pibar_mode)

    assert "Recipients_mat" in model._static.species_helpers
    assert model._static.transfer_mat_unnormalized is not None
    assert model._static.ancestors_T is None

    loss = model()
    assert torch.isfinite(loss)

    model._static.warm_E = None
    model._static.transfer_mat_unnormalized = None
    with pytest.raises((AttributeError, TypeError)):
        model()


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


def test_no_grad_genewise_uniform_uses_equivalent_inference_path(
    species_path, gene_paths
):
    """No-grad genewise NLL should match the differentiable saved-tensor path.

    The no-grad path is allowed to avoid returning/storing full ``Pi``/``Pibar``
    and compute likelihood from root rows only, but it must preserve the scalar
    and per-family NLL values.
    """
    model = _build_model(species_path, gene_paths, "genewise", "uniform")

    model._static.warm_E = None
    differentiable_total = model()

    model._static.warm_E = None
    with torch.no_grad():
        inference_total = model()
        inference_per_family = model.nll_per_family()

    assert not inference_total.requires_grad
    assert not inference_per_family.requires_grad
    assert math.isclose(
        float(inference_total.item()),
        float(differentiable_total.detach().item()),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert math.isclose(
        float(inference_per_family.sum().item()),
        float(inference_total.item()),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_genewise_uniform_backward_matches_individual_uniform_trees(
    species_path, gene_paths
):
    """Batched genewise gradients match running each gene tree independently.

    This pins down the genewise autograd path as a true cross-family batching of
    independent uniform-mode problems.  The weighted per-family backward also
    checks that ``grad_output`` from ``nll_per_family()`` is applied row-wise to
    the corresponding gene's parameters.
    """
    dtype = torch.float64
    device = _device()
    rates = torch.tensor(
        [
            [0.050, 0.040, 0.030],
            [0.060, 0.025, 0.045],
            [0.030, 0.055, 0.035],
        ],
        dtype=dtype,
        device=device,
    )
    theta = torch.log2(rates)
    solver_kwargs = dict(
        fixed_iters_Pi=6,
        max_iters_E=128,
        tol_E=0.0,
        neumann_terms=3,
        use_pruning=False,
        cg_tol=1e-10,
        cg_maxiter=1000,
    )

    batched = GeneReconModel.from_trees(
        species_tree=species_path,
        gene_trees=gene_paths,
        mode="genewise",
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
        **solver_kwargs,
    )
    with torch.no_grad():
        batched.theta.copy_(theta)

    weights = torch.tensor([0.5, 1.25, 2.0], dtype=dtype, device=device)
    batched.zero_grad(set_to_none=True)
    batched.static.warm_E = None
    per_family_nll = batched.nll_per_family()
    (weights * per_family_nll).sum().backward()
    batched_grad = batched.theta.grad.detach().clone()

    single_nlls = []
    single_grads = []
    for i, gene_path in enumerate(gene_paths):
        single = GeneReconModel.from_trees(
            species_tree=species_path,
            gene_trees=[gene_path],
            mode="global",
            pibar_mode="uniform",
            device=device,
            dtype=dtype,
            **solver_kwargs,
        )
        with torch.no_grad():
            single.theta.copy_(theta[i])
        single.zero_grad(set_to_none=True)
        single.static.warm_E = None
        nll = single()
        nll.backward()
        single_nlls.append(nll.detach())
        single_grads.append(single.theta.grad.detach().clone())

    single_nll = torch.stack(single_nlls)
    single_grad = torch.stack(single_grads)

    torch.testing.assert_close(
        per_family_nll.detach(),
        single_nll,
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        batched_grad,
        weights.view(-1, 1) * single_grad,
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused path required")
def test_global_uniform_fused_high_neumann_matches_tight_generic_gmres(
    monkeypatch,
):
    """High-iteration fused uniform backward matches a tight generic solve.

    The production generic path intentionally uses a short GMRES solve, and the
    production fused path often uses a short Neumann series.  This test tightens
    both sides so it checks implementation correctness rather than default
    approximation settings.
    """
    data_dir = _ROOT / "data" / "test_trees_1000"
    if not data_dir.exists():
        pytest.skip("test_trees_1000 dataset not present")
    gene_path = data_dir / "g_0002.nwk"
    if not gene_path.exists():
        pytest.skip("test_trees_1000/g_0002.nwk not present")

    from gpurec.core import backward as backward_core

    orig_gmres = backward_core._gmres_self_loop_solve

    def tight_gmres(
        rhs,
        ingredients,
        sp_child1,
        sp_child2,
        S,
        W,
        pibar_mode,
        transfer_mat_T,
        ancestors_T,
        max_iters=30,
        tol=1e-5,
    ):
        return orig_gmres(
            rhs,
            ingredients,
            sp_child1,
            sp_child2,
            S,
            W,
            pibar_mode,
            transfer_mat_T,
            ancestors_T,
            max_iters=40,
            tol=1e-14,
        )

    monkeypatch.setattr(backward_core, "_gmres_self_loop_solve", tight_gmres)

    solver_kwargs = dict(
        fixed_iters_Pi=6,
        max_iters_E=128,
        tol_E=0.0,
        neumann_terms=20,
        use_pruning=False,
        cg_tol=1e-10,
        cg_maxiter=1000,
    )

    def run_backward(*, fused: bool):
        monkeypatch.setenv(
            "GPUREC_FUSED_UNIFORM_BACKWARD",
            "1" if fused else "0",
        )
        model = GeneReconModel.from_trees(
            species_tree=str(data_dir / "sp.nwk"),
            gene_trees=[str(gene_path)],
            mode="global",
            pibar_mode="uniform",
            device=torch.device("cuda"),
            dtype=torch.float64,
            theta_init_rates=(0.05, 0.05, 0.05),
            **solver_kwargs,
        )
        model.zero_grad(set_to_none=True)
        model.static.warm_E = None
        nll = model()
        nll.backward()
        torch.cuda.synchronize()
        return nll.detach(), model.theta.grad.detach().clone()

    generic_nll, generic_grad = run_backward(fused=False)
    fused_nll, fused_grad = run_backward(fused=True)

    torch.testing.assert_close(fused_nll, generic_nll, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        fused_grad,
        generic_grad,
        rtol=1e-10,
        atol=1e-8,
    )


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
