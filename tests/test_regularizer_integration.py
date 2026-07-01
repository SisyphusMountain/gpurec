"""Integration tests for the TV prior / origination penalty / grouped-theta wiring
in ``gpurec.optim.value_and_grad.make_value_and_grad``.

This file holds the CPU-checkable portion (no Triton/CUDA forward solve required).
The end-to-end no-op-equals-baseline GPU validation is added in Task 7.
"""
import torch
import pytest


def test_group_index_theta_shape_contract():
    from gpurec.optim.penalties import group_expand, group_reduce
    gidx = torch.tensor([0, 0, 1])           # 3 species, 2 categories
    theta_g = torch.randn(2, 3, dtype=torch.float64)
    theta_s = group_expand(theta_g, gidx)
    assert theta_s.shape == (3, 3)
    grad_s = torch.ones(3, 3, dtype=torch.float64)
    grad_g = group_reduce(grad_s, gidx, 2)
    assert grad_g.shape == (2, 3)
    assert torch.allclose(grad_g[0], torch.full((3,), 2.0, dtype=torch.float64))
    assert torch.allclose(grad_g[1], torch.full((3,), 1.0, dtype=torch.float64))


def test_tv_with_smooth_polish_is_rejected():
    from gpurec.optim.optimize import optimize
    with pytest.raises(ValueError, match="non-smooth"):
        optimize(object(), torch.zeros(3, 3), torch.zeros(3),
                 tv_penalty=(1.0, torch.tensor([-1, 0, 0]), 1e-3), polish_mode="ridge")


# ---------------------------------------------------------------------------------------------
# GPU end-to-end tests (Task 7)
#
# NOTE on the `examples/tiny/` fixture: that directory (species.nwk / gene.nwk / families.txt /
# gene.map) uses a families.txt-manifest + gene.map leaf-renaming format that no loader in this
# codebase currently consumes (grep for "families.txt"/"gene.map"/"FAMILIES]" across gpurec/
# turns up nothing but this docstring and README.md). The real, working construction path --
# used throughout tests/test_public_api_integration.py -- is
# ``GeneReconModel(species_tree_path, [gene_tree_path, ...], ...)`` where gene-tree leaves are
# named ``<species>_<id>`` directly (no separate mapping file; see CLAUDE.md "Leaf->species =
# prefix before first `_`"). So these tests vendor the same tiny in-memory fixture pattern already
# exercised (and CUDA-gated) by that file's ``_write_tiny_example``/`_require_cuda_triton``/
# ``_require_preprocess_native`` helpers, using a slightly larger 4-leaf/7-species tree (reused
# verbatim from that file's ``test_zero_receiver_logits_match_uniform_receiver_formula``) so the
# species tree has actual parent-child edges for the TV prior to act on.
# ---------------------------------------------------------------------------------------------
cuda = pytest.mark.gpu


def _write_tiny_species_gene(tmp_path):
    species_tree = tmp_path / "species.nwk"
    gene_tree = tmp_path / "family_0.nwk"
    species_tree.write_text("((A:1,B:1)AB:1,(C:1,D:1)CD:1)Root:1;\n", encoding="utf-8")
    gene_tree.write_text("((A_1:1,B_1:1)GAB:1,(C_1:1,D_1:1)GCD:1)GeneRoot:1;\n", encoding="utf-8")
    return species_tree, [gene_tree]


def _require_preprocess_native():
    from gpurec.core.scheduling import batching
    try:
        batching._load_native_module()
    except Exception as exc:
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")


def _require_cuda_triton():
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        torch.empty(1, device="cuda")
    except Exception as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return torch.device("cuda")


def _solver_options():
    from gpurec import SolverOptions
    return SolverOptions(e_max_iter=4, e_tol=1e-4, pi_iters=2, neumann_terms=0)


def _build_specieswise_model(tmp_path, device):
    from gpurec import GeneReconModel
    species_tree, gene_trees = _write_tiny_species_gene(tmp_path)
    return GeneReconModel(
        species_tree,
        gene_trees,
        mode="specieswise",
        device=device,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
        solver_options=_solver_options(),
    )


@cuda
def test_disabled_regularizers_equal_baseline(tmp_path):
    """``make_value_and_grad`` at the new kwargs' implicit defaults must match a second,
    independently-built closure that passes the same defaults EXPLICITLY
    (``tv_penalty=None, origination_penalty=None, group_index=None``) -- the no-op-at-default
    contract for Task 7's end-to-end wiring.

    Robustness note: two separate Triton forward/backward evaluations of the *identical* closure
    at the *identical* theta can differ by a small amount from atomic-reduction nondeterminism
    (see test_optim_golden.py's GRAD_RTOL=2e-3 / LOSS_RTOL=1e-3 for the scale of this on a much
    bigger fixture). A strict ``atol=0`` compare would be flaky here for that reason. Instead we
    MEASURE the nondeterminism floor directly, by calling one closure (`f_default`) twice at the
    same theta, and require the "explicit-None" closure's output to be within a small multiple of
    that measured floor (plus a tiny absolute backstop in case the floor happens to measure exactly
    zero on a given run). This is the option (a) robustness strategy from the task brief. The
    authoritative bit-exact disabled-default gate is ``test_optim_golden.py``; this is
    supplementary e2e coverage that exercises the real ``GeneReconModel`` / ``make_value_and_grad``
    wiring end to end on CUDA.
    """
    _require_preprocess_native()
    device = _require_cuda_triton()
    from gpurec.optim.value_and_grad import make_value_and_grad

    model = _build_specieswise_model(tmp_path, device)
    theta = model.theta.detach().reshape(-1).clone()
    receiver_weights = model.receiver_weights.detach()

    f_default = make_value_and_grad(model.batch_statics, receiver_weights, theta_shape=model.theta.shape)
    f_explicit = make_value_and_grad(
        model.batch_statics, receiver_weights, theta_shape=model.theta.shape,
        tv_penalty=None, origination_penalty=None, group_index=None,
    )

    # Nondeterminism floor: same closure, same theta, called twice.
    loss_a, grad_a, _, _ = f_default(theta)
    loss_b, grad_b, _, _ = f_default(theta)
    floor_loss = abs(loss_a - loss_b)
    floor_grad = (grad_a - grad_b).norm().item()

    loss_c, grad_c, _, _ = f_explicit(theta)

    tol_loss = max(5.0 * floor_loss, 1e-6 * max(1.0, abs(loss_a)))
    tol_grad = max(5.0 * floor_grad, 1e-6 * max(1.0, grad_a.norm().item()))

    assert abs(loss_a - loss_c) <= tol_loss, (loss_a, loss_c, floor_loss, tol_loss)
    assert (grad_a - grad_c).norm().item() <= tol_grad, (floor_grad, tol_grad)


@cuda
def test_tv_penalty_changes_gradient(tmp_path):
    """A nonzero TV penalty must change ``d(loss)/d(theta)`` vs. the same closure without it.

    ``model.theta`` is initialized to a UNIFORM constant across every (species, rate) entry
    (``math.log2(1e-10)`` everywhere, see ``GeneReconModel.__init__``), which makes every
    parent-child TV difference exactly zero -- and hence the TV penalty AND its gradient exactly
    zero at that specific point (``rho'(0) = 0/eps = 0``). So this test evaluates at a
    deliberately non-uniform, still fully deterministic and fixed theta (the model's init plus a
    small deterministic per-(species, rate) ramp), for which the TV term is a large deterministic
    additive gradient -- robust to the same atomic-reduction noise floor measured/discussed in
    ``test_disabled_regularizers_equal_baseline`` above (the assertion below additionally checks
    the observed difference is much larger than that floor, not just "not exactly equal").
    """
    _require_preprocess_native()
    device = _require_cuda_triton()
    from gpurec.optim.value_and_grad import make_value_and_grad

    model = _build_specieswise_model(tmp_path, device)
    sp_parent = model.species_helpers["sp_parent"]
    receiver_weights = model.receiver_weights.detach()

    S, K = model.theta.shape
    ramp = torch.arange(S * K, device=device, dtype=model.theta.dtype).reshape(S, K) * 0.01
    theta = (model.theta.detach() + ramp).reshape(-1).clone()

    f0 = make_value_and_grad(model.batch_statics, receiver_weights, theta_shape=model.theta.shape)
    f1 = make_value_and_grad(
        model.batch_statics, receiver_weights, theta_shape=model.theta.shape,
        tv_penalty=(1.0, sp_parent, 1e-3),
    )
    _, g0, _, _ = f0(theta)
    _, g1, _, _ = f1(theta)

    # Nondeterminism floor for this same fixture/theta, measured the same way as the no-op test.
    _, g0_repeat, _, _ = f0(theta)
    floor_grad = (g0 - g0_repeat).norm().item()

    diff = (g0 - g1).norm().item()
    assert not torch.allclose(g0, g1)
    assert diff > max(10.0 * floor_grad, 1e-4), (diff, floor_grad)
