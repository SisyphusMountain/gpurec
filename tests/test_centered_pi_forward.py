from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.api._execution import evaluate_static_convergence
from gpurec.core.inference.solver import solve_forward_residual, solve_resident_e_pi
from gpurec.core.kernels.centered_pi_forward import (
    compute_dts_forward_centered,
    compute_leaf_initial_wave_step_centered,
    compute_wave_step_centered,
)
from gpurec.core.kernels.dts_fused import compute_dts_forward
from gpurec.core.kernels.wave_step import compute_leaf_initial_wave_step
from gpurec.core.scheduling import batching


def _require_cuda() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _require_native_cuda() -> None:
    _require_cuda()
    try:
        batching._load_native_module()
    except Exception as exc:
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")


def _write_tiny_ale_example(tmp_path: Path) -> tuple[Path, list[Path]]:
    species_tree = tmp_path / "species.nwk"
    gene_tree = tmp_path / "family_0.ale"
    species_tree.write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
    gene_tree.write_text(
        """#constructor_string
A_1,B_1,C_1
#observations
120
#Bip_counts
4 120
5 120
6 120
#Bip_bls
1 1
2 1
3 1
4 1
5 1
6 1
#Dip_counts
4 2 3 120
5 1 3 120
6 1 2 120
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
    return species_tree, [gene_tree]


@pytest.mark.gpu
@pytest.mark.parametrize("use_receiver_weights", [False, True])
def test_centered_leaf_initialization_matches_nonzero_observation_seed(
    use_receiver_weights: bool,
) -> None:
    _require_cuda()
    device = torch.device("cuda")
    dtype = torch.float32
    S = 5

    # Root 0 has children 1/2; internal node 1 has leaf children 3/4.
    sp_child1 = torch.tensor([1, 3, S, S, S], device=device)
    sp_child2 = torch.tensor([2, 4, S, S, S], device=device)
    sp_subtree_start = torch.tensor([0, 1, 4, 2, 3], device=device)
    sp_subtree_end = torch.tensor([5, 4, 5, 3, 4], device=device)
    leaf_species = torch.tensor([3], device=device)
    family_idx = torch.tensor([0], device=device)

    max_transfer = torch.tensor(
        [[-0.2, -0.4, -0.6, -0.8, -1.0]], device=device, dtype=dtype
    )
    dl_const = torch.tensor(
        [[-1.3, -1.1, -0.9, -0.7, -0.5]], device=device, dtype=dtype
    )
    ebar = torch.tensor(
        [[-2.0, -1.8, -1.6, -1.4, -1.2]], device=device, dtype=dtype
    )
    e = torch.tensor(
        [[-0.9, -1.0, -1.1, -1.2, -1.3]], device=device, dtype=dtype
    )
    sl1 = torch.tensor(
        [[-1.7, -1.5, -1.3, -1.1, -0.9]], device=device, dtype=dtype
    )
    sl2 = torch.tensor(
        [[-1.6, -1.4, -1.2, -1.0, -0.8]], device=device, dtype=dtype
    )
    receiver_log_probs = torch.tensor(
        [-3.0, -2.5, -2.0, -1.5, -1.0], device=device, dtype=dtype
    )
    # The selected leaf's observation seed is deliberately far from log2(1)=0.
    leaf_logp = torch.tensor(
        [[-0.3, -1.1, -2.2, -4.75, -0.8]], device=device, dtype=dtype
    )

    absolute = torch.empty((1, S), device=device, dtype=dtype)
    centered = torch.empty_like(absolute)
    offset = torch.empty((1,), device=device, dtype=torch.float64)
    args = (
        0,
        1,
        S,
        max_transfer,
        dl_const,
        ebar,
        e,
        sl1,
        sl2,
        receiver_log_probs,
        sp_child1,
        sp_child2,
        sp_subtree_start,
        sp_subtree_end,
        leaf_species,
        leaf_logp,
        family_idx,
    )
    compute_leaf_initial_wave_step(
        absolute, *args, use_receiver_weights=use_receiver_weights
    )
    compute_leaf_initial_wave_step_centered(
        centered, offset, *args, use_receiver_weights=use_receiver_weights
    )

    reconstructed = centered.double() + offset[:, None]
    torch.testing.assert_close(
        reconstructed, absolute.double(), rtol=0.0, atol=5e-7
    )
    torch.testing.assert_close(offset, absolute.amax(dim=1).double(), rtol=0.0, atol=5e-7)


@pytest.mark.gpu
def test_centered_wave_residual_aligns_offset_frames_and_ignores_inactive_rows() -> None:
    _require_cuda()
    device = torch.device("cuda")
    dtype = torch.float32
    S = 3
    W = 2
    neg_inf = float("-inf")

    pi_in = torch.tensor(
        [[-0.25, -1.5, -2.0], [neg_inf, neg_inf, neg_inf]],
        device=device,
        dtype=dtype,
    )
    pi_in_offset = torch.tensor([100.0, -40.0], device=device, dtype=torch.float64)
    pi_out = torch.empty_like(pi_in)
    pi_out_offset = torch.empty_like(pi_in_offset)
    pibar = torch.empty_like(pi_in)
    pibar_offset = torch.empty_like(pi_in_offset)
    pibar_row_max = torch.empty((W,), device=device, dtype=dtype)
    pi_residual = torch.full((W,), float("nan"), device=device, dtype=dtype)

    # The split offsets deliberately move each output into a different frame,
    # but their residual rows are impossible and do not change the recurrence.
    dts = torch.full_like(pi_in, neg_inf)
    dts_offset = torch.tensor([105.0, -30.0], device=device, dtype=torch.float64)
    max_transfer = torch.zeros((W, S), device=device, dtype=dtype)
    dl_const = torch.tensor(
        [[0.25, -0.5, 1.0], [neg_inf, neg_inf, neg_inf]],
        device=device,
        dtype=dtype,
    )
    impossible = torch.full((W, S), neg_inf, device=device, dtype=dtype)
    receiver_log_probs = torch.zeros((S,), device=device, dtype=dtype)
    children = torch.full((S,), S, device=device, dtype=torch.long)
    parents = torch.full((S,), -1, device=device, dtype=torch.long)
    leaf_species = torch.zeros((W,), device=device, dtype=torch.long)
    leaf_logp = torch.zeros((W, S), device=device, dtype=dtype)
    family_idx = torch.arange(W, device=device, dtype=torch.long)

    compute_wave_step_centered(
        pi_in,
        pi_in_offset,
        pi_out,
        pi_out_offset,
        pibar,
        pibar_offset,
        0,
        W,
        S,
        max_transfer,
        dl_const,
        impossible,
        impossible,
        impossible,
        impossible,
        receiver_log_probs,
        children,
        children,
        parents,
        1,
        dts,
        dts_offset,
        leaf_species_idx=leaf_species,
        leaf_logp=leaf_logp,
        family_idx=family_idx,
        pibar_row_max=pibar_row_max,
        store_final_pibar=True,
        has_leaf_term=False,
        use_receiver_weights=False,
        pi_residual_out=pi_residual,
    )

    expected_absolute = pi_in[0].double() + pi_in_offset[0] + dl_const[0].double()
    reconstructed = pi_out[0].double() + pi_out_offset[0]
    torch.testing.assert_close(reconstructed, expected_absolute, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        pi_out_offset,
        torch.tensor([105.0, 0.0], device=device, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isneginf(pi_out[1]).all().item()
    # A residual-only comparison would report the five-unit gauge shift. The
    # frame-aligned absolute update is max(|DL|)=1, while the inactive row is 0.
    torch.testing.assert_close(
        pi_residual,
        torch.tensor([1.0, 0.0], device=device, dtype=dtype),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.gpu
def test_centered_final_pibar_all_impossible_row_uses_zero_offset() -> None:
    """The heuristic final-Pibar gauge must preserve canonical empty rows."""
    _require_cuda()
    device = torch.device("cuda")
    S = 1
    pi_in = torch.zeros((1, S), device=device)
    pi_in_offset = torch.tensor([17.0], device=device, dtype=torch.float64)
    pi_out = torch.empty_like(pi_in)
    pi_out_offset = torch.empty_like(pi_in_offset)
    pibar = torch.empty_like(pi_in)
    pibar_offset = torch.full_like(pi_in_offset, float("nan"))
    rates = torch.zeros((1, S), device=device)
    impossible = torch.full_like(rates, float("-inf"))
    children = torch.full((S,), S, device=device, dtype=torch.long)

    compute_wave_step_centered(
        pi_in,
        pi_in_offset,
        pi_out,
        pi_out_offset,
        pibar,
        pibar_offset,
        0,
        1,
        S,
        rates,
        rates,
        impossible,
        impossible,
        impossible,
        impossible,
        rates[0],
        children,
        children,
        torch.full((S,), -1, device=device, dtype=torch.long),
        1,
        leaf_species_idx=torch.zeros((1,), device=device, dtype=torch.long),
        leaf_logp=rates,
        family_idx=torch.zeros((1,), device=device, dtype=torch.long),
        pibar_row_max=torch.empty((1,), device=device),
        store_final_pibar=True,
        has_leaf_term=False,
        use_receiver_weights=False,
    )

    assert torch.isneginf(pibar).all().item()
    assert torch.equal(pibar_offset, torch.zeros_like(pibar_offset))


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dts_values", "receiver_values", "weighted", "expected_residual", "expected_offset"),
    [
        ([-100.0, -101.0, -102.0], [0.0, 0.0, 0.0], False, [0.0, -1.0, -2.0], -300.0),
        # The raw residual maximum is zero even though receiver weighting makes
        # the weighted row maximum -100. Centering must use the former.
        ([0.0, -100.0, -101.0], [-100.0, 0.0, -10.0], True, [0.0, -100.0, -101.0], -200.0),
    ],
)
def test_centered_split_input_uses_raw_max_for_virtual_dts_frame(
    dts_values,
    receiver_values,
    weighted,
    expected_residual,
    expected_offset,
) -> None:
    """The first split iteration must not let a shallow DTS frame undo centering."""
    _require_cuda()
    device = torch.device("cuda")
    dtype = torch.float32
    S = 3
    dts = torch.tensor([dts_values], device=device, dtype=dtype)
    dts_offset = torch.tensor([-200.0], device=device, dtype=torch.float64)
    dts_center_offset = torch.empty_like(dts_offset)
    absolute_before = dts.double() + dts_offset[:, None]
    pi_out = torch.empty_like(dts)
    pi_out_offset = torch.empty_like(dts_offset)
    pibar = torch.empty_like(dts)
    pibar_offset = torch.empty_like(dts_offset)
    impossible = torch.full_like(dts, float("-inf"))
    children = torch.full((S,), S, device=device, dtype=torch.long)

    compute_wave_step_centered(
        dts,
        dts_offset,
        pi_out,
        pi_out_offset,
        pibar,
        pibar_offset,
        0,
        1,
        S,
        torch.zeros_like(dts),
        impossible,
        impossible,
        impossible,
        impossible,
        impossible,
        torch.tensor(receiver_values, device=device, dtype=dtype),
        children,
        children,
        torch.full((S,), -1, device=device, dtype=torch.long),
        1,
        dts,
        dts_offset,
        dts_center_offset,
        leaf_species_idx=torch.zeros((1,), device=device, dtype=torch.long),
        leaf_logp=impossible,
        family_idx=torch.zeros((1,), device=device, dtype=torch.long),
        pibar_row_max=torch.empty((1,), device=device, dtype=dtype),
        store_final_pibar=True,
        has_leaf_term=False,
        input_ws=0,
        use_receiver_weights=weighted,
    )

    torch.testing.assert_close(
        dts, torch.tensor([dts_values], device=device)
    )
    torch.testing.assert_close(
        dts_offset,
        torch.tensor([-200.0], device=device, dtype=torch.float64),
    )
    torch.testing.assert_close(
        dts.double() + dts_offset[:, None], absolute_before
    )
    torch.testing.assert_close(
        dts_center_offset,
        torch.tensor([expected_offset], device=device, dtype=torch.float64),
    )
    torch.testing.assert_close(
        pi_out_offset,
        torch.tensor([expected_offset], device=device, dtype=torch.float64),
    )
    torch.testing.assert_close(
        pi_out, torch.tensor([expected_residual], device=device)
    )
    torch.testing.assert_close(
        pi_out.double() + pi_out_offset[:, None],
        absolute_before,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("fanout", [1, 3])
def test_centered_dts_accepts_float64_split_statics_and_matches_float64_reference(
    fanout: int,
) -> None:
    _require_cuda()
    device = torch.device("cuda")
    S = 5
    pi = torch.tensor(
        [
            [-0.2, -1.1, -2.4, -0.7, -3.0],
            [-1.7, -0.3, -1.2, -2.2, -0.9],
            [-0.6, -2.0, -0.4, -1.5, -1.0],
            [-1.3, -0.8, -2.1, -0.5, -1.7],
        ],
        device=device,
        dtype=torch.float32,
    )
    pibar = torch.tensor(
        [
            [-0.9, -1.8, -0.1, -2.0, -1.2],
            [-1.1, -0.5, -2.3, -0.8, -1.4],
            [-2.2, -0.7, -1.0, -1.6, -0.2],
            [-0.4, -1.9, -0.6, -1.1, -2.5],
        ],
        device=device,
        dtype=torch.float32,
    )
    pi_offset = torch.tensor(
        [1200.125, -50.25, 800.5, 2.0], device=device, dtype=torch.float64
    )
    pibar_offset = torch.tensor(
        [1199.75, -49.875, 801.25, 1.5], device=device, dtype=torch.float64
    )
    pairs = [(0, 1), (2, 3), (1, 2)][:fanout]
    lefts = torch.tensor([left for left, _ in pairs], device=device)
    rights = torch.tensor([right for _, right in pairs], device=device)
    reduce_idx = torch.zeros((fanout,), device=device, dtype=torch.long)
    sp_child1 = torch.tensor([1, 3, S, S, S], device=device)
    sp_child2 = torch.tensor([2, 4, S, S, S], device=device)
    family_idx = torch.tensor([0], device=device)
    log_p_d = torch.tensor(
        [[-2.1, -2.3, -2.5, -2.7, -2.9]], device=device, dtype=torch.float32
    )
    log_p_s = torch.tensor(
        [[-1.4, -1.6, -1.8, -2.0, -2.2]], device=device, dtype=torch.float32
    )
    # Batch preprocessing intentionally owns these statics in float64.
    log_split_probs = torch.tensor(
        [-0.123456789, -1.234567891, -2.345678912][:fanout],
        device=device,
        dtype=torch.float64,
    )

    layout = {}
    if fanout > 1:
        layout = {
            "n_eq1": 0,
            "eq1_reduce_idx": reduce_idx[:0],
            "ge2_ptr": torch.tensor([0, fanout], device=device, dtype=torch.long),
            "ge2_parent_ids": torch.tensor([0], device=device, dtype=torch.long),
            "ge2_max_fanout": fanout,
        }

    centered_out, out_offset = compute_dts_forward_centered(
        pi,
        pi_offset,
        pibar,
        pibar_offset,
        lefts,
        rights,
        sp_child1,
        sp_child2,
        1,
        reduce_idx,
        log_p_d,
        log_p_s,
        family_idx,
        log_split_probs=log_split_probs,
        **layout,
    )

    pi_absolute = pi.double() + pi_offset[:, None]
    pibar_absolute = pibar.double() + pibar_offset[:, None]
    reference = compute_dts_forward(
        pi_absolute,
        pibar_absolute,
        lefts,
        rights,
        sp_child1,
        sp_child2,
        1,
        reduce_idx,
        log_p_d.double(),
        log_p_s.double(),
        family_idx,
        log_split_probs=log_split_probs,
        **layout,
    )
    reconstructed = centered_out.double() + out_offset[:, None]
    torch.testing.assert_close(reconstructed, reference, rtol=0.0, atol=1e-5)
    assert out_offset.dtype == torch.float64


@pytest.mark.gpu
def test_centered_dts_empty_and_inactive_rows_use_zero_offset() -> None:
    _require_cuda()
    device = torch.device("cuda")
    S = 3
    pi = torch.zeros((2, S), device=device)
    pibar = torch.zeros_like(pi)
    pi_offset = torch.tensor([17.0, -4.0], device=device, dtype=torch.float64)
    pibar_offset = torch.tensor([16.0, -3.0], device=device, dtype=torch.float64)
    lefts = torch.tensor([0], device=device, dtype=torch.long)
    rights = torch.tensor([1], device=device, dtype=torch.long)
    reduce_idx = torch.tensor([0], device=device, dtype=torch.long)
    children = torch.full((S,), S, device=device, dtype=torch.long)
    rates = torch.full((1, S), -2.0, device=device)
    family_idx = torch.tensor([0], device=device, dtype=torch.long)

    empty, empty_offset = compute_dts_forward_centered(
        pi,
        pi_offset,
        pibar,
        pibar_offset,
        lefts[:0],
        rights[:0],
        children,
        children,
        1,
        reduce_idx[:0],
        rates,
        rates,
        family_idx,
    )
    assert torch.isneginf(empty).all().item()
    assert torch.equal(empty_offset, torch.zeros_like(empty_offset))

    inactive, inactive_offset = compute_dts_forward_centered(
        pi,
        pi_offset,
        pibar,
        pibar_offset,
        lefts,
        rights,
        children,
        children,
        1,
        reduce_idx,
        rates,
        rates,
        family_idx,
        active_parent_rows=torch.zeros((1,), device=device, dtype=torch.bool),
    )
    assert torch.isneginf(inactive).all().item()
    assert torch.equal(inactive_offset, torch.zeros_like(inactive_offset))


@pytest.mark.gpu
@pytest.mark.parametrize("fanout", [1, 2])
@pytest.mark.parametrize("pruned", [False, True])
def test_centered_dts_impossible_eq1_and_ge2_rows_canonicalize_offset(
    fanout: int,
    pruned: bool,
) -> None:
    """An all-impossible row is ``(-inf residual, zero offset)`` in every DTS path."""
    _require_cuda()
    device = torch.device("cuda")
    S = 3
    pi = torch.full((2, S), float("-inf"), device=device, dtype=torch.float32)
    pibar = torch.full_like(pi, float("-inf"))
    # Deliberately noncanonical child gauges make this a producer test rather
    # than relying on the upstream invariant to hide a missing output reset.
    pi_offset = torch.tensor([17.0, -4.0], device=device, dtype=torch.float64)
    pibar_offset = torch.tensor([16.0, -3.0], device=device, dtype=torch.float64)
    lefts = torch.tensor([0, 1][:fanout], device=device, dtype=torch.long)
    rights = torch.tensor([1, 0][:fanout], device=device, dtype=torch.long)
    reduce_idx = torch.zeros(fanout, device=device, dtype=torch.long)
    child1 = torch.tensor([1, S, S], device=device, dtype=torch.long)
    child2 = torch.tensor([2, S, S], device=device, dtype=torch.long)
    family_idx = torch.tensor([0], device=device, dtype=torch.long)
    log_p_d = torch.full((1, S), -2.0, device=device)
    log_p_s = torch.full((1, S), -1.5, device=device)
    layout = {}
    if fanout > 1:
        layout = {
            "n_eq1": 0,
            "eq1_reduce_idx": reduce_idx[:0],
            "ge2_ptr": torch.tensor([0, fanout], device=device, dtype=torch.long),
            "ge2_parent_ids": torch.tensor([0], device=device, dtype=torch.long),
            "ge2_max_fanout": fanout,
        }

    residual, offset = compute_dts_forward_centered(
        pi,
        pi_offset,
        pibar,
        pibar_offset,
        lefts,
        rights,
        child1,
        child2,
        1,
        reduce_idx,
        log_p_d,
        log_p_s,
        family_idx,
        active_parent_rows=(
            torch.tensor([False], device=device) if pruned else None
        ),
        **layout,
    )

    assert torch.isneginf(residual).all().item()
    assert torch.equal(offset, torch.zeros_like(offset))


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [False, True])
def test_centered_pi_forward_matches_loss_for_receiver_and_origination_cases(
    tmp_path: Path,
    weighted: bool,
) -> None:
    _require_native_cuda()
    species_tree, gene_trees = _write_tiny_ale_example(tmp_path)
    options = SolverOptions(
        e_max_iter=8,
        e_tol=1e-5,
        pi_iters=4,
        pi_representation="absolute",
        neumann_terms=0,
    )
    model = GeneReconModel(
        species_tree,
        gene_trees,
        device="cuda",
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=8,
        solver_options=options,
    )
    receiver_weights = torch.zeros_like(model.receiver_weights)
    origination_weights = torch.zeros_like(model.origination_weights)
    if weighted:
        receiver_weights = torch.linspace(
            -1.25, 1.75, receiver_weights.numel(), device=receiver_weights.device
        )
        origination_weights = torch.linspace(
            0.9, -1.1, origination_weights.numel(), device=origination_weights.device
        )

    split_metas = [meta for meta in model.wave_layout["wave_metas"] if "sl" in meta]
    assert any(int(meta["n_eq1"]) > 0 for meta in split_metas)
    assert any(
        meta.get("ge2_parent_ids") is not None
        and int(meta["ge2_parent_ids"].numel()) > 0
        for meta in split_metas
    )

    absolute_solve = solve_resident_e_pi(
        model.batch_statics[0],
        model.theta.detach(),
        receiver_weights,
        warm_start_E=None,
    )
    absolute_forward_residual = solve_forward_residual(
        model.batch_statics[0],
        model.theta.detach(),
        receiver_weights,
        pi_iters=4,
        warm_start_E=None,
    )
    with torch.no_grad():
        base_loss = model(
            receiver_weights=receiver_weights,
            origination_weights=origination_weights,
        )
    absolute_theta = model.theta.detach().clone().requires_grad_(True)
    absolute_receiver = receiver_weights.detach().clone().requires_grad_(True)
    absolute_origination = origination_weights.detach().clone().requires_grad_(True)
    absolute_grad_loss = model(
        theta=absolute_theta,
        receiver_weights=absolute_receiver,
        origination_weights=absolute_origination,
    )
    absolute_grads = torch.autograd.grad(
        absolute_grad_loss,
        (absolute_theta, absolute_receiver, absolute_origination),
    )
    model.clear_warm_starts()
    model.configure_solver(pi_representation="centered")
    centered_solve = solve_resident_e_pi(
        model.batch_statics[0],
        model.theta.detach(),
        receiver_weights,
        warm_start_E=None,
    )
    centered_forward_residual = solve_forward_residual(
        model.batch_statics[0],
        model.theta.detach(),
        receiver_weights,
        pi_iters=4,
        warm_start_E=None,
    )
    centered_convergence = evaluate_static_convergence(
        model.batch_statics[0],
        model.theta.detach(),
        receiver_weights,
        pi_iters_high=4,
        neumann_terms=2,
    )
    with torch.no_grad():
        centered_loss = model(
            receiver_weights=receiver_weights,
            origination_weights=origination_weights,
        )
    centered_theta = model.theta.detach().clone().requires_grad_(True)
    centered_receiver = receiver_weights.detach().clone().requires_grad_(True)
    centered_origination = origination_weights.detach().clone().requires_grad_(True)
    centered_grad_loss = model(
        theta=centered_theta,
        receiver_weights=centered_receiver,
        origination_weights=centered_origination,
    )
    centered_grads = torch.autograd.grad(
        centered_grad_loss,
        (centered_theta, centered_receiver, centered_origination),
    )
    centered_state = getattr(model.batch_statics[0], "centered_pi_forward_state", None)

    assert centered_state is not None
    assert centered_state.pi_offset.dtype == torch.float64
    assert centered_state.pibar_offset.dtype == torch.float64
    assert torch.isfinite(centered_loss).item()
    torch.testing.assert_close(centered_loss, base_loss.double(), rtol=0.0, atol=5e-6)
    torch.testing.assert_close(
        centered_forward_residual,
        absolute_forward_residual,
        rtol=0.0,
        atol=1e-5,
    )
    torch.testing.assert_close(
        centered_convergence[0],
        centered_forward_residual,
        rtol=0.0,
        atol=0.0,
    )
    assert all(torch.isfinite(value).all().item() for value in centered_convergence)
    torch.testing.assert_close(
        centered_grad_loss,
        absolute_grad_loss.double(),
        rtol=0.0,
        atol=5e-6,
    )
    for centered_grad, absolute_grad in zip(centered_grads, absolute_grads):
        assert torch.isfinite(centered_grad).all().item()
        torch.testing.assert_close(
            centered_grad,
            absolute_grad,
            rtol=2e-5,
            atol=2e-5,
        )

    absolute_root, absolute_pi, absolute_pibar, absolute_row_max = (
        absolute_solve[4],
        absolute_solve[5],
        absolute_solve[6],
        absolute_solve[7],
    )
    centered_root, centered_pi, centered_pibar, centered_row_max = (
        centered_solve[4],
        centered_solve[5],
        centered_solve[6],
        centered_solve[7],
    )
    pi_reconstructed = centered_pi.double() + centered_state.pi_offset[:, None]
    pibar_reconstructed = centered_pibar.double() + centered_state.pibar_offset[:, None]
    # Centering intentionally changes fp32 rounding; this toy fixture is a
    # mathematical-parity check, while the HOGENOM capture harness supplies the
    # large-magnitude fp64 accuracy ordering.
    torch.testing.assert_close(centered_root, absolute_root.double(), rtol=0.0, atol=1.5e-5)
    torch.testing.assert_close(pi_reconstructed, absolute_pi.double(), rtol=0.0, atol=1e-5)
    torch.testing.assert_close(pibar_reconstructed, absolute_pibar.double(), rtol=0.0, atol=1e-5)
    # The saved row max is in the same residual frame as Pi, so adding Pi's
    # offset reconstructs the absolute metadata consumed by the normal path.
    torch.testing.assert_close(
        centered_row_max.double() + centered_state.pi_offset,
        absolute_row_max.double(),
        rtol=0.0,
        atol=1e-5,
    )



@pytest.mark.gpu
def test_centered_pi_forward_streams_batches_without_losing_fp64_loss(tmp_path: Path) -> None:
    _require_native_cuda()
    species_tree, gene_trees = _write_tiny_ale_example(tmp_path)
    options = SolverOptions(
        e_max_iter=8,
        e_tol=1e-5,
        pi_iters=4,
        pi_representation="centered",
        neumann_terms=0,
    )
    single = GeneReconModel(
        species_tree,
        gene_trees,
        device="cuda",
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=8,
        solver_options=options,
    )
    single_loss = single().detach()

    streamed = GeneReconModel(
        species_tree,
        [gene_trees[0], gene_trees[0]],
        device="cuda",
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=8,
        solver_options=options,
    )
    with torch.no_grad():
        streamed_loss = streamed()
    assert streamed_loss.dtype == torch.float64
    torch.testing.assert_close(streamed_loss, 2.0 * single_loss, rtol=0.0, atol=1e-5)
