from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel
from gpurec.core.likelihood import compute_nll_root_rows
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


def _root_row_loss(model, state, pi):
    root_rows = torch.as_tensor(
        model.current_batch_metadata.root_clade_rows,
        device=pi.device,
        dtype=torch.long,
    )
    return compute_nll_root_rows(
        pi.index_select(0, root_rows),
        state.e,
        state.origination_probs,
        origination_probs_prepared=True,
    ).sum()


def _balanced_newick(names: list[str], *, root_name: str) -> str:
    def subtree(items: list[str]) -> str:
        if len(items) == 1:
            return f"{items[0]}:1"
        mid = len(items) // 2
        return f"({subtree(items[:mid])},{subtree(items[mid:])}):1"

    root = subtree(names)
    if root.endswith(":1"):
        root = root[:-2]
    return f"{root}{root_name};\n"


def _write_large_matched_tree_pair(tmp_path: Path) -> tuple[str, list[str]]:
    names = [f"s{i:03d}" for i in range(129)]
    species_tree = tmp_path / "sp.nwk"
    gene_tree = tmp_path / "g.nwk"
    species_tree.write_text(_balanced_newick(names, root_name="Root"), encoding="utf-8")
    gene_tree.write_text(_balanced_newick(names, root_name="Gene"), encoding="utf-8")
    return str(species_tree), [str(gene_tree)]


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


def test_gene_recon_model_pi_adjoint_warmstart_records_residual_stats(tmp_path):
    sp, genes = _write_large_matched_tree_pair(tmp_path)
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode="genewise",
        device="cuda",
        dtype=torch.float32,
        fixed_iters_E=2,
        fixed_iters_Pi=2,
        neumann_terms=2,
        pi_adjoint_warmstart=True,
    )

    loss = model()
    loss.backward()

    stats = model.static.last_solver_stats
    assert stats["Pi_adjoint_warmstart_enabled"] is True
    assert stats["Pi_adjoint_residual_wave_count"] > 0
    assert stats["Pi_adjoint_residual_absmax"] >= 0.0
    assert stats["Pi_adjoint_residual_relmax"] >= 0.0
    assert torch.isfinite(torch.tensor(stats["Pi_adjoint_residual_absmax"]))
    assert torch.isfinite(torch.tensor(stats["Pi_adjoint_residual_relmax"]))


@pytest.mark.parametrize("mode", ["global", "specieswise", "genewise"])
def test_resident_evaluation_paths_remain_consistent(trees, mode):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode=mode,
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_E=2,
        fixed_iters_Pi=2,
        neumann_terms=1,
    )

    assert len(model.batch_metadata) == 1
    assert model.current_batch_metadata.family_indices == (0,)

    forward_loss = model()
    forward_loss.backward()
    forward_grad = model.theta.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    full_loss = model.full_loss()
    explicit_theta_loss = model.full_loss_for_theta(model.theta.detach())
    with torch.no_grad():
        no_grad_loss = model()
        state = model.reconciliation_state()
        state_wave = model.reconciliation_state(original_order=False)
        pi = model.pi_matrix()
        wave_pi = model.pi_matrix(original_order=False)
    state_loss = _root_row_loss(model, state, state.pi)
    pi_loss = _root_row_loss(model, state, pi)
    perm = model.cached_static_states[0].wave_layout["perm"]

    post_probe_loss = model()
    post_probe_loss.backward()

    assert torch.isfinite(forward_loss)
    assert no_grad_loss.requires_grad is False
    for tensor in (
        state.e,
        state.pi,
        state.log_p_s,
        state.log_p_d,
        state.log_p_l,
        state.max_transfer,
    ):
        assert tensor.device == model.theta.device
        assert tensor.dtype == model.theta.dtype
    torch.testing.assert_close(full_loss.detach(), forward_loss.detach(), rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(
        explicit_theta_loss.detach(),
        forward_loss.detach(),
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(
        no_grad_loss,
        forward_loss.detach(),
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(state.pi, pi, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(
        state.pi,
        state_wave.pi.index_select(0, perm),
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(state_wave.pi, wave_pi, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(state_loss, forward_loss.detach(), rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(pi_loss, forward_loss.detach(), rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(
        post_probe_loss.detach(),
        forward_loss.detach(),
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(model.theta.grad, forward_grad, rtol=1e-4, atol=1e-3)


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
    with torch.no_grad():
        batched_no_grad = batched.full_loss_for_theta(batched.theta.detach())
        resident_no_grad = resident.full_loss_for_theta(resident.theta.detach())
    torch.testing.assert_close(
        batched_no_grad,
        resident_no_grad,
        rtol=1e-5,
        atol=1e-4,
    )

    next_meta = batched.next()
    assert next_meta.family_indices == (1,)
