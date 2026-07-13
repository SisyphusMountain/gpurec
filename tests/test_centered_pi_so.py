from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.api._execution import evaluate_static_loss_grad
from gpurec.core.kernels.dts_so import dts_backward_so
from gpurec.core.kernels.wave_so import wave_backward_so
from gpurec.solver.hvp_exact import make_exact_hvp
from gpurec.solver.value_and_grad import forward_solve


def _require_cuda() -> torch.device:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda")


def _write_split_fanout_example(tmp_path: Path) -> tuple[Path, Path]:
    species_tree = tmp_path / "species.nwk"
    gene_tree = tmp_path / "family_0.ale"
    species_tree.write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
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


@pytest.mark.gpu
@pytest.mark.parametrize(("has_splits", "weighted"), [(False, True), (True, False)])
def test_centered_wave_so_matches_reconstructed_absolute(
    has_splits: bool,
    weighted: bool,
) -> None:
    device = _require_cuda()
    dtype = torch.float32
    S, C, ws, W = 3, 2, 1, 1
    pi = torch.tensor(
        [[-0.7, -1.4, -0.2], [-0.25, -1.1, -1.8]], device=device, dtype=dtype
    )
    pibar = torch.tensor(
        [[-1.0, -0.4, -1.7], [-0.6, -1.5, -0.9]], device=device, dtype=dtype
    )
    pi_offset = torch.tensor([2.0, 4.0], device=device, dtype=torch.float64)
    pibar_offset = torch.tensor([2.75, 1.5], device=device, dtype=torch.float64)
    pi_absolute = (pi.double() + pi_offset[:, None]).to(dtype)
    pibar_absolute = (pibar.double() + pibar_offset[:, None]).to(dtype)

    dPi = torch.tensor(
        [[0.1, -0.2, 0.3], [-0.4, 0.25, 0.15]], device=device, dtype=dtype
    )
    dPibar = torch.tensor(
        [[-0.3, 0.2, 0.1], [0.35, -0.1, 0.05]], device=device, dtype=dtype
    )
    v = torch.tensor([[0.7, -0.25, 0.4]], device=device, dtype=dtype)
    col = torch.tensor([-1.8, -1.2, -2.1], device=device, dtype=dtype)
    dcol = torch.tensor([0.12, -0.08, 0.03], device=device, dtype=dtype)
    row_max_residual = (pi + col if weighted else pi).amax(dim=1)
    row_max_absolute = row_max_residual + pi_offset.to(dtype)

    mc = torch.tensor([[0.2, 0.1, 0.3]], device=device, dtype=dtype)
    dl = torch.tensor([[-0.8, -1.0, -0.6]], device=device, dtype=dtype)
    ebar = torch.tensor([[-1.2, -0.7, -1.1]], device=device, dtype=dtype)
    e = torch.tensor([[-0.5, -0.9, -0.4]], device=device, dtype=dtype)
    sl1 = torch.tensor([[-1.4, -1.1, -0.8]], device=device, dtype=dtype)
    sl2 = torch.tensor([[-1.3, -0.75, -1.05]], device=device, dtype=dtype)
    dDL = torch.tensor([[0.03, -0.07, 0.11]], device=device, dtype=dtype)
    dEbar = torch.tensor([[-0.02, 0.08, -0.04]], device=device, dtype=dtype)
    dE = torch.tensor([[0.05, -0.03, 0.09]], device=device, dtype=dtype)
    dSL1 = torch.tensor([[0.06, 0.02, -0.05]], device=device, dtype=dtype)
    dSL2 = torch.tensor([[-0.01, 0.04, 0.07]], device=device, dtype=dtype)
    child1 = torch.tensor([1, S, S], device=device, dtype=torch.long)
    child2 = torch.tensor([2, S, S], device=device, dtype=torch.long)
    parent = torch.tensor([-1, 0, 0], device=device, dtype=torch.long)
    leaf_state = torch.tensor([1, 2], device=device, dtype=torch.long)
    leaf_logp = torch.tensor([[-1.5, -0.9, -1.25]], device=device, dtype=dtype)
    dleaf = torch.tensor([[0.02, -0.06, 0.04]], device=device, dtype=dtype)
    item_idx = torch.zeros((C,), device=device, dtype=torch.long)
    d_rhs = torch.tensor(
        [[0.0, 0.0, 0.0], [0.04, -0.02, 0.01]], device=device, dtype=dtype
    )

    dts_offset = torch.tensor([5.5], device=device, dtype=torch.float64)
    dts = torch.tensor([[-0.45, -1.3, -0.7]], device=device, dtype=dtype)
    dts_absolute = (dts.double() + dts_offset[:, None]).to(dtype)
    d_dts = torch.tensor([[0.09, -0.05, 0.02]], device=device, dtype=dtype)

    common = dict(
        v=v,
        ws=ws,
        W=W,
        S=S,
        mc=mc,
        DL=dl,
        dDL=dDL,
        Ebar=ebar,
        dEbar=dEbar,
        E=e,
        dE=dE,
        SL1=sl1,
        dSL1=dSL1,
        SL2=sl2,
        dSL2=dSL2,
        col_log_probs=col,
        node_child1=child1,
        node_child2=child2,
        node_parent=parent,
        max_ancestor_depth=2,
        dts_r=dts_absolute if has_splits else None,
        d_dts=d_dts if has_splits else None,
        leaf_state_idx=leaf_state,
        leaf_logp=leaf_logp,
        dleaf_logp=dleaf,
        item_idx=item_idx,
        has_leaf_term=not has_splits,
        use_col_weights=weighted,
        d_rhs=d_rhs,
        dcol=dcol,
    )
    absolute = wave_backward_so(
        pi_absolute,
        dPi,
        pibar_absolute,
        dPibar,
        pibar_row_max=row_max_absolute,
        pi_offset=torch.zeros_like(pi_offset),
        pibar_offset=torch.zeros_like(pibar_offset),
        dts_offset=(torch.zeros_like(dts_offset) if has_splits else None),
        **common,
    )
    common["dts_r"] = dts if has_splits else None
    centered = wave_backward_so(
        pi,
        dPi,
        pibar,
        dPibar,
        pibar_row_max=row_max_residual,
        pi_offset=pi_offset,
        pibar_offset=pibar_offset,
        dts_offset=dts_offset if has_splits else None,
        **common,
    )

    for centered_value, absolute_value in zip(centered, absolute):
        torch.testing.assert_close(centered_value, absolute_value, rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("fanout", [1, 2])
def test_centered_dts_so_matches_reconstructed_absolute(fanout: int) -> None:
    device = _require_cuda()
    dtype = torch.float32
    S, C, ws = 3, 3, 2
    pi = torch.tensor(
        [
            [-0.2, -1.0, -1.7],
            [-0.8, -0.3, -1.2],
            [-0.45, -1.4, -0.65],
        ],
        device=device,
        dtype=dtype,
    )
    pibar = torch.tensor(
        [
            [-0.7, -1.3, -0.4],
            [-1.1, -0.5, -0.9],
            [-0.6, -1.0, -0.8],
        ],
        device=device,
        dtype=dtype,
    )
    pi_offset = torch.tensor([2.0, -1.0, 4.0], device=device, dtype=torch.float64)
    pibar_offset = torch.tensor([0.5, 1.5, 3.25], device=device, dtype=torch.float64)
    pi_absolute = (pi.double() + pi_offset[:, None]).to(dtype)
    pibar_absolute = (pibar.double() + pibar_offset[:, None]).to(dtype)
    row_max_residual = pi.amax(dim=1)
    row_max_absolute = row_max_residual + pi_offset.to(dtype)

    dPi = torch.tensor(
        [[0.2, -0.1, 0.05], [-0.3, 0.15, 0.08], [0.12, -0.04, 0.2]],
        device=device,
        dtype=dtype,
    )
    dPibar = torch.tensor(
        [[-0.05, 0.12, -0.08], [0.09, -0.11, 0.07], [0.03, 0.02, -0.06]],
        device=device,
        dtype=dtype,
    )
    v = torch.tensor([[0.4, -0.2, 0.65]], device=device, dtype=dtype)
    sl = torch.tensor([0, 1][:fanout], device=device, dtype=torch.long)
    sr = torch.tensor([1, 0][:fanout], device=device, dtype=torch.long)
    meta = {
        "sl": sl,
        "sr": sr,
        "reduce_idx": torch.zeros((fanout,), device=device, dtype=torch.long),
        "log_split_probs": torch.tensor(
            [-0.4, -1.1][:fanout], device=device, dtype=torch.float64
        ),
        "start": ws,
    }
    child1 = torch.tensor([1, S, S], device=device, dtype=torch.long)
    child2 = torch.tensor([2, S, S], device=device, dtype=torch.long)
    parent = torch.tensor([-1, 0, 0], device=device, dtype=torch.long)
    item_idx = torch.zeros((C,), device=device, dtype=torch.long)
    log_pD = torch.tensor([[-1.7, -1.9, -2.1]], device=device, dtype=dtype)
    log_pS = torch.tensor([[-1.1, -1.3, -1.5]], device=device, dtype=dtype)
    dlog_pD = torch.tensor([[0.03, -0.06, 0.08]], device=device, dtype=dtype)
    dlog_pS = torch.tensor([[-0.02, 0.07, 0.04]], device=device, dtype=dtype)
    mc = torch.tensor([[0.2, 0.1, 0.3]], device=device, dtype=dtype)
    dmc = torch.tensor([[0.05, -0.04, 0.02]], device=device, dtype=dtype)
    col = torch.tensor([-1.4, -1.7, -1.2], device=device, dtype=dtype)
    dcol = torch.tensor([0.04, -0.03, 0.01], device=device, dtype=dtype)
    levels = dict(
        compact_level_ptr=torch.tensor([0, 1], device=device, dtype=torch.long),
        compact_level_parents=torch.tensor([0], device=device, dtype=torch.long),
        compact_level_child1=torch.tensor([1], device=device, dtype=torch.long),
        compact_level_child2=torch.tensor([2], device=device, dtype=torch.long),
    )

    def run(Pi, Pibar, row_max, pi_gauge, pibar_gauge):
        d_rhs = torch.zeros((C, S), device=device, dtype=dtype)
        d_grad_pD = torch.zeros((1, S), device=device, dtype=dtype)
        d_grad_pS = torch.zeros((1, S), device=device, dtype=dtype)
        d_grad_mt = torch.zeros((1, S), device=device, dtype=dtype)
        d_grad_col = torch.zeros((S,), device=device, dtype=dtype)
        dts_backward_so(
            Pi,
            dPi,
            Pibar,
            dPibar,
            v,
            ws,
            meta,
            S,
            log_pD,
            log_pS,
            dlog_pD,
            dlog_pS,
            mc,
            dmc,
            col,
            child1,
            child2,
            parent,
            2,
            row_max,
            item_idx,
            d_rhs,
            d_grad_pD,
            d_grad_pS,
            d_grad_mt,
            d_grad_col,
            use_col_weights=True,
            dcol=dcol,
            pi_offset=pi_gauge,
            pibar_offset=pibar_gauge,
            **levels,
        )
        return d_rhs, d_grad_pD, d_grad_pS, d_grad_mt, d_grad_col

    absolute = run(
        pi_absolute,
        pibar_absolute,
        row_max_absolute,
        torch.zeros_like(pi_offset),
        torch.zeros_like(pibar_offset),
    )
    centered = run(pi, pibar, row_max_residual, pi_offset, pibar_offset)
    for centered_value, absolute_value in zip(centered, absolute):
        torch.testing.assert_close(centered_value, absolute_value, rtol=3e-5, atol=3e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [False, True])
def test_exact_hvp_matches_fp64_and_finite_difference(
    weighted: bool,
) -> None:
    _require_cuda()
    data = Path("tests/data/alerax/test_trees_200")
    species_tree, gene_tree = data / "sp.nwk", data / "g.nwk"

    def build(dtype: torch.dtype, e_tol: float) -> GeneReconModel:
        return GeneReconModel(
            species_tree,
            [gene_tree],
            mode="genewise",
            device="cuda",
            dtype=dtype,
            family_chunk_size=1,
            clade_budget=None,
            batch_packing="sequential",
            max_wave_size=8,
            solver_options=SolverOptions(
                e_max_iter=512,
                e_tol=e_tol,
                pi_iters=64,
                neumann_terms=64,
                use_adjoint_pruning=False,
            ),
        )

    centered_model = build(torch.float32, 1e-8)
    reference_model = build(torch.float64, 1e-12)
    theta = torch.full((1, 3), -3.321928, device="cuda", dtype=torch.float32)
    receiver = torch.zeros_like(centered_model.receiver_weights)
    if weighted:
        receiver = torch.linspace(-1.0, 0.9, receiver.numel(), device=receiver.device)
    direction = torch.tensor([[0.4, -0.7, 0.2]], device="cuda", dtype=torch.float32)

    centered_static = centered_model.batch_statics[0]
    reference_static = reference_model.batch_statics[0]
    _, reference_saved = forward_solve(
        [reference_static], theta.double(), receiver.double()
    )
    _, centered_saved = forward_solve([centered_static], theta, receiver)
    assert centered_saved["pi_state"] is not None
    split_metas = [meta for meta in centered_model.wave_layout["wave_metas"] if "sl" in meta]
    assert any(int(meta["n_eq1"]) > 0 for meta in split_metas)
    assert any(
        meta.get("ge2_parent_ids") is not None
        and int(meta["ge2_parent_ids"].numel()) > 0
        for meta in split_metas
    )

    reference_hvp = make_exact_hvp(
        [reference_static],
        theta.double(),
        receiver.double(),
        reference_saved,
        tangent_self_iters=64,
    )
    centered_hvp = make_exact_hvp(
        [centered_static], theta, receiver, centered_saved, tangent_self_iters=64
    )
    reference_value = reference_hvp(direction.double())
    centered_value = centered_hvp(direction).double()
    assert torch.isfinite(centered_value).all().item()
    torch.testing.assert_close(centered_value, reference_value, rtol=2e-3, atol=2e-3)

    origination = torch.zeros_like(centered_model.origination_weights)

    def centered_gradient(theta_value: torch.Tensor) -> torch.Tensor:
        centered_static.warm_E = None
        _, gradient, _, _ = evaluate_static_loss_grad(
            centered_static,
            theta_value,
            receiver,
            origination,
            need_grad=True,
        )
        return gradient.reshape(-1)

    eps = 2e-2
    fd_value = (
        centered_gradient(theta + eps * direction)
        - centered_gradient(theta - eps * direction)
    ).double() / (2.0 * eps)
    assert torch.isfinite(fd_value).all().item()
    error = float((centered_value - fd_value).abs().max())
    scale = max(float(fd_value.abs().max()), 1e-12)
    assert error / scale < 1e-2, f"centered HVP finite-difference relative error {error / scale:.3e}"
