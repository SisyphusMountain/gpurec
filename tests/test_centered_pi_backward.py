from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.config import GpurecConfig, PrecisionOptions
from gpurec.core.inference.solver import (
    receiver_weights_are_uniform,
    solve_resident_e_pi,
)
from gpurec.core.parameters.extract_parameters import (
    origination_log_probs_from_weights,
)
from gpurec.core.scheduling import batching


def _require_native_cuda() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        batching._load_native_module()
    except Exception as exc:
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")


def _write_split_fanout_example(tmp_path: Path) -> tuple[Path, Path]:
    species_tree = tmp_path / "species.nwk"
    gene_tree = tmp_path / "family_0.ale"
    species_tree.write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
    # Parent clade 4 has two observed decompositions.  The resulting layout
    # exercises both the one-split DTS path and the grouped fanout path.
    gene_tree.write_text(
        """#constructor_string
A_1,B_1,C_1
#observations
120
#Bip_counts
4 90
5 75
6 45
#Bip_bls
1 1
2 1
3 1
4 1
5 1
6 1
#Dip_counts
4 2 3 50
4 3 2 40
5 1 3 75
6 1 2 45
#last_leafset_id
6
#leaf-id
A_1 1
B_1 2
C_1 3
#set-id
1 : 3
2 : 1
3 : 2
4 : 1 2
5 : 2 3
6 : 1 3
#END
""",
        encoding="utf-8",
    )
    return species_tree, gene_tree


def _options(dtype: torch.dtype) -> SolverOptions:
    return SolverOptions(
        e_max_iter=32,
        e_tol=1e-12 if dtype == torch.float64 else 1e-7,
        pi_iters=8,
        neumann_terms=16,
        use_adjoint_pruning=False,
    )


def _model(
    species_tree: Path,
    genes: list[Path],
    dtype: torch.dtype = torch.float32,
    accumulator_dtype: torch.dtype = torch.float64,
) -> GeneReconModel:
    dtype_name = "float64" if dtype == torch.float64 else "float32"
    accumulator_name = (
        "float64" if accumulator_dtype == torch.float64 else "float32"
    )
    return GeneReconModel(
        species_tree,
        genes,
        device="cuda",
        dtype=dtype,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=8,
        solver_options=_options(dtype),
        config=GpurecConfig(
            precision=PrecisionOptions(
                model_dtype=dtype_name,
                accumulator_dtype=accumulator_name,
            )
        ),
    )


def _inputs(model: GeneReconModel, weighted: bool):
    theta = torch.tensor(
        [-1.7, -2.3, -1.1], device=model.theta.device, dtype=model.theta.dtype
    )
    receiver = torch.zeros_like(model.receiver_weights)
    origination = torch.zeros_like(model.origination_weights)
    if weighted:
        receiver = torch.linspace(-1.0, 0.9, receiver.numel(), device=receiver.device)
        origination = torch.linspace(0.8, -0.7, origination.numel(), device=origination.device)
    return theta, receiver, origination


def _implicit_kwargs(model, solved, receiver_weights, origination_weights):
    (
        E,
        E_s1,
        E_s2,
        Ebar,
        _root_rows,
        _pi,
        _pibar,
        _pibar_row_max,
        log_pS,
        log_pD,
        log_pL,
        max_transfer,
        receiver_log_probs,
    ) = solved
    static = model.batch_statics[0]
    if torch.all(origination_weights == origination_weights.reshape(-1)[0]):
        origination_log_probs = None
        origination_probs = None
    else:
        origination_log_probs = origination_log_probs_from_weights(origination_weights)
        origination_probs = torch.exp2(origination_log_probs)
    return dict(
        E_star=E,
        E_s1=E_s1,
        E_s2=E_s2,
        Ebar=Ebar,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        max_transfer_mat=max_transfer,
        receiver_log_probs=receiver_log_probs,
        use_receiver_weights=not receiver_weights_are_uniform(receiver_weights),
        # This test compares BOTH returned gradients (theta and receiver weights) against fp64.
        need_receiver_grad=True,
        theta=model.theta,
        receiver_weights=receiver_weights,
        family_idx=static.rate_family_idx,
        specieswise=static.specieswise,
        genewise=static.genewise,
        neumann_terms=model.solver_options.neumann_terms,
        neumann_term_tol=model.solver_options.neumann_term_tol,
        adjoint_self_loop=model.solver_options.adjoint_self_loop,
        # The E-adjoint linear solve is a Neumann series -- its budget and relative-residual target.
        # (The old BiCGSTAB fields and the e_adjoint_solver selector no longer exist.)
        e_adjoint_max_iter=model.solver_options.e_adjoint_max_iter,
        e_adjoint_tol=model.solver_options.e_adjoint_tol,
        adjoint_pruning_threshold=model.solver_options.adjoint_pruning_threshold,
        use_adjoint_pruning=False,
        pibar_side_threshold=model.solver_options.pibar_side_threshold,
        origination_log_probs=origination_log_probs,
        origination_probs=origination_probs,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("pruned", [False, True])
@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_native_backward_matches_fp64(
    tmp_path: Path,
    weighted: bool,
    pruned: bool,
    accumulator_dtype: torch.dtype,
) -> None:
    _require_native_cuda()
    species_tree, gene = _write_split_fanout_example(tmp_path)
    model = _model(
        species_tree, [gene], accumulator_dtype=accumulator_dtype
    )
    reference_model = _model(species_tree, [gene], torch.float64)
    theta, receiver, origination = _inputs(model, weighted)
    theta64, receiver64, origination64 = (
        theta.double(), receiver.double(), origination.double()
    )
    with torch.no_grad():
        model.theta.copy_(theta)
        reference_model.theta.copy_(theta64)

    split_metas = [meta for meta in model.wave_layout["wave_metas"] if "sl" in meta]
    assert any(int(meta["n_eq1"]) > 0 for meta in split_metas)
    assert any(
        meta.get("ge2_parent_ids") is not None
        and int(meta["ge2_parent_ids"].numel()) > 0
        for meta in split_metas
    )

    solved = solve_resident_e_pi(
        model.batch_statics[0], model.theta, receiver, warm_start_E=None
    )
    reference_solved = solve_resident_e_pi(
        reference_model.batch_statics[0],
        reference_model.theta,
        receiver64,
        warm_start_E=None,
    )
    pi, pibar, pibar_row_max = solved[5], solved[6], solved[7]
    state = model.batch_statics[0].pi_forward_state
    reference_state = reference_model.batch_statics[0].pi_forward_state
    assert state is not None and reference_state is not None
    kwargs = _implicit_kwargs(model, solved, receiver, origination)
    reference_kwargs = _implicit_kwargs(
        reference_model, reference_solved, receiver64, origination64
    )
    kwargs["use_adjoint_pruning"] = pruned
    kwargs["adjoint_pruning_threshold"] = 2.0 if pruned else 1e-6
    reference_kwargs["use_adjoint_pruning"] = pruned
    reference_kwargs["adjoint_pruning_threshold"] = 2.0 if pruned else 1e-6
    native = implicit_grad_loglik_vjp_wave(
        model.wave_layout,
        model.species_helpers,
        Pi_star_wave=pi,
        Pibar_star_wave=pibar,
        uniform_pibar_row_max=pibar_row_max,
        pi_offset=state.pi_offset,
        pibar_offset=state.pibar_offset,
        **kwargs,
    )
    reference = implicit_grad_loglik_vjp_wave(
        reference_model.wave_layout,
        reference_model.species_helpers,
        Pi_star_wave=reference_solved[5],
        Pibar_star_wave=reference_solved[6],
        uniform_pibar_row_max=reference_solved[7],
        pi_offset=reference_state.pi_offset,
        pibar_offset=reference_state.pibar_offset,
        **reference_kwargs,
    )

    for native_grad, reference_grad in zip(native, reference):
        assert torch.isfinite(native_grad).all().item()
        torch.testing.assert_close(
            native_grad.double(), reference_grad, rtol=5e-5, atol=2e-5
        )


def _value_and_grads(model, theta, receiver, origination):
    theta = theta.detach().clone().requires_grad_(True)
    receiver = receiver.detach().clone().requires_grad_(True)
    origination = origination.detach().clone().requires_grad_(True)
    loss = model(
        theta=theta,
        receiver_weights=receiver,
        origination_weights=origination,
    )
    grads = torch.autograd.grad(loss, (theta, receiver, origination))
    return loss.detach(), tuple(grad.detach() for grad in grads)


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [False, True])
def test_native_backward_matches_single_streamed_and_fp64(
    tmp_path: Path,
    weighted: bool,
) -> None:
    _require_native_cuda()
    species_tree, gene = _write_split_fanout_example(tmp_path)
    single = _model(species_tree, [gene])
    streamed = _model(species_tree, [gene, gene])
    reference = _model(species_tree, [gene, gene], torch.float64)
    theta, receiver, origination = _inputs(single, weighted)

    single_loss, single_grads = _value_and_grads(
        single, theta, receiver, origination
    )
    streamed_loss, streamed_grads = _value_and_grads(
        streamed, theta, receiver, origination
    )
    reference_loss, reference_grads = _value_and_grads(
        reference, theta.double(), receiver.double(), origination.double()
    )

    assert single_loss.dtype == torch.float64
    assert streamed_loss.dtype == torch.float64
    torch.testing.assert_close(
        streamed_loss, 2.0 * single_loss, rtol=0.0, atol=1e-5
    )
    torch.testing.assert_close(
        streamed_loss, reference_loss, rtol=0.0, atol=2e-5
    )
    for single_grad, streamed_grad, reference_grad in zip(
        single_grads, streamed_grads, reference_grads
    ):
        torch.testing.assert_close(
            streamed_grad, 2.0 * single_grad, rtol=3e-5, atol=3e-6
        )
        torch.testing.assert_close(
            streamed_grad.double(), reference_grad, rtol=5e-5, atol=2e-5
        )


@pytest.mark.gpu
def test_centered_autograd_saves_offsets_from_each_forward(tmp_path: Path) -> None:
    """A later solve on the same static must not replace an earlier ctx's gauges."""
    _require_native_cuda()
    species_tree, gene = _write_split_fanout_example(tmp_path)
    shared = _model(species_tree, [gene])
    receiver = torch.linspace(
        -0.8, 0.7, shared.receiver_weights.numel(), device="cuda"
    )
    origination = torch.linspace(
        0.6, -0.5, shared.origination_weights.numel(), device="cuda"
    )
    theta_a = torch.tensor([-1.7, -2.3, -1.1], device="cuda", requires_grad=True)
    theta_b = torch.tensor([0.4, -4.2, -2.6], device="cuda", requires_grad=True)

    # Both contexts share one mutable batch static. Forward B replaces the
    # static's current offset sidecar before backward A executes.
    loss_a = shared(theta=theta_a, receiver_weights=receiver, origination_weights=origination)
    loss_b = shared(theta=theta_b, receiver_weights=receiver, origination_weights=origination)
    grad_a, grad_b = torch.autograd.grad(loss_a + loss_b, (theta_a, theta_b))

    references = []
    for theta in (theta_a.detach(), theta_b.detach()):
        model = _model(species_tree, [gene])
        theta_ref = theta.clone().requires_grad_(True)
        loss_ref = model(
            theta=theta_ref,
            receiver_weights=receiver,
            origination_weights=origination,
        )
        (gradient_ref,) = torch.autograd.grad(loss_ref, theta_ref)
        references.append(gradient_ref)

    torch.testing.assert_close(grad_a, references[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(grad_b, references[1], rtol=2e-5, atol=2e-6)
