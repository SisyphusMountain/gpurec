import math
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.api._execution import evaluate_static_convergence
from gpurec.core.inference.forward import pi_wave_forward
from gpurec.core.inference.solver import solve_forward_residual, solve_resident_e_pi
from gpurec.core.kernels.pi_forward import (
    compute_dts_forward,
    compute_leaf_initial_wave_step,
    compute_wave_step,
)
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
def test_pi_wave_forward_rejects_accumulator_narrower_than_residual() -> None:
    _require_cuda()
    value = torch.zeros((1, 1), device="cuda", dtype=torch.float64)
    with pytest.raises(TypeError, match="must not be narrower"):
        pi_wave_forward(
            {},
            {},
            value,
            value,
            value,
            value,
            value,
            value,
            value,
            value.reshape(-1),
            family_idx=torch.zeros(1, device="cuda", dtype=torch.long),
            accumulator_dtype=torch.float32,
            self_loop_mode="linear",
            linear_tol=1e-6,
            exact_range_log2=100.0,
            linear_iterations_out=None,
            exact_guard_trips_out=None,
        )


@pytest.mark.gpu
def test_dts_forward_rejects_mixed_offset_dtypes() -> None:
    _require_cuda()
    pi = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    index = torch.empty(0, device="cuda", dtype=torch.long)
    with pytest.raises(TypeError, match="match accumulator dtype"):
        compute_dts_forward(
            pi,
            torch.zeros(1, device="cuda", dtype=torch.float32),
            pi.clone(),
            torch.zeros(1, device="cuda", dtype=torch.float64),
            index,
            index,
            torch.ones(1, device="cuda", dtype=torch.long),
            torch.ones(1, device="cuda", dtype=torch.long),
            1,
            index,
            pi,
            pi,
            torch.zeros(1, device="cuda", dtype=torch.long),
        )


def test_dts_forward_rejects_split_probabilities_outside_model_dtype() -> None:
    pi = torch.zeros((1, 1), dtype=torch.float32)
    index = torch.zeros(1, dtype=torch.long)
    with pytest.raises(TypeError, match="log_split_probs must match residual dtype"):
        compute_dts_forward(
            pi,
            torch.zeros(1, dtype=torch.float64),
            pi.clone(),
            torch.zeros(1, dtype=torch.float64),
            index,
            index,
            torch.ones(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            1,
            index,
            pi,
            pi,
            index,
            log_split_probs=torch.zeros(1, dtype=torch.float64),
        )


@pytest.mark.gpu
@pytest.mark.parametrize("use_receiver_weights", [False, True])
@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_centered_leaf_initialization_matches_nonzero_observation_seed(
    use_receiver_weights: bool,
    accumulator_dtype: torch.dtype,
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

    residual = torch.empty((1, S), device=device, dtype=dtype)
    offset = torch.empty((1,), device=device, dtype=accumulator_dtype)
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
        residual, offset, *args, use_receiver_weights=use_receiver_weights
    )

    # The same canonical kernel in fp64 is the precision oracle. Inputs remain
    # in the same gauges; only the residual arithmetic changes dtype.
    reference_residual = torch.empty((1, S), device=device, dtype=torch.float64)
    reference_offset = torch.empty((1,), device=device, dtype=torch.float64)
    args64 = tuple(
        value.double() if torch.is_tensor(value) and value.is_floating_point() else value
        for value in args
    )
    compute_leaf_initial_wave_step(
        reference_residual,
        reference_offset,
        *args64,
        use_receiver_weights=use_receiver_weights,
    )
    reconstructed = residual.to(accumulator_dtype) + offset[:, None]
    reference = reference_residual + reference_offset[:, None]
    torch.testing.assert_close(
        reconstructed.double(), reference, rtol=0.0, atol=5e-7
    )
    torch.testing.assert_close(offset, reconstructed.amax(dim=1), rtol=0.0, atol=5e-7)


@pytest.mark.gpu
@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_centered_wave_residual_aligns_offset_frames_and_ignores_inactive_rows(
    accumulator_dtype: torch.dtype,
) -> None:
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
    pi_in_offset = torch.tensor(
        [100.0, -40.0], device=device, dtype=accumulator_dtype
    )
    pi_out = torch.empty_like(pi_in)
    pi_out_offset = torch.empty_like(pi_in_offset)
    pibar = torch.empty_like(pi_in)
    pibar_offset = torch.empty_like(pi_in_offset)
    pibar_row_max = torch.empty((W,), device=device, dtype=dtype)
    pi_residual = torch.full((W,), float("nan"), device=device, dtype=dtype)

    # The split offsets deliberately move each output into a different frame,
    # but their residual rows are impossible and do not change the recurrence.
    dts = torch.full_like(pi_in, neg_inf)
    dts_offset = torch.tensor(
        [105.0, -30.0], device=device, dtype=accumulator_dtype
    )
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

    compute_wave_step(
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
        row_mask=None,
    )

    expected_absolute = pi_in[0].double() + pi_in_offset[0] + dl_const[0].double()
    reconstructed = pi_out[0].double() + pi_out_offset[0]
    torch.testing.assert_close(reconstructed, expected_absolute, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        pi_out_offset,
        torch.tensor([105.0, 0.0], device=device, dtype=accumulator_dtype),
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

    compute_wave_step(
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
        row_mask=None,
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

    compute_wave_step(
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
        row_mask=None,
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
@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_centered_dts_uses_model_dtype_split_statics_and_matches_float64_reference(
    fanout: int,
    accumulator_dtype: torch.dtype,
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
        [1200.125, -50.25, 800.5, 2.0], device=device, dtype=accumulator_dtype
    )
    pibar_offset = torch.tensor(
        [1199.75, -49.875, 801.25, 1.5], device=device, dtype=accumulator_dtype
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
    split_values = [-0.123456789, -1.234567891, -2.345678912][:fanout]
    log_split_probs = torch.tensor(
        split_values,
        device=device,
        dtype=pi.dtype,
    )
    reference_log_split_probs = torch.tensor(
        split_values, device=device, dtype=torch.float64
    )

    layout = {}
    if fanout > 1:
        layout = {
            "n_single_split_parents": 0,
            "single_split_parent_rows": reduce_idx[:0],
            "multiple_split_group_ptr": torch.tensor(
                [0, fanout], device=device, dtype=torch.long
            ),
            "multiple_split_parent_rows": torch.tensor(
                [0], device=device, dtype=torch.long
            ),
            "max_splits_per_multiple_parent": fanout,
        }

    centered_out, out_offset = compute_dts_forward(
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

    reference_residual, reference_offset = compute_dts_forward(
        pi.double(),
        pi_offset.double(),
        pibar.double(),
        pibar_offset.double(),
        lefts,
        rights,
        sp_child1,
        sp_child2,
        1,
        reduce_idx,
        log_p_d.double(),
        log_p_s.double(),
        family_idx,
        log_split_probs=reference_log_split_probs,
        **layout,
    )
    reconstructed = centered_out.double() + out_offset[:, None]
    reference = reference_residual + reference_offset[:, None]
    atol = 3e-4 if accumulator_dtype == torch.float32 else 1e-5
    torch.testing.assert_close(reconstructed, reference, rtol=0.0, atol=atol)
    assert out_offset.dtype == accumulator_dtype


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

    empty, empty_offset = compute_dts_forward(
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

    inactive, inactive_offset = compute_dts_forward(
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
            "n_single_split_parents": 0,
            "single_split_parent_rows": reduce_idx[:0],
            "multiple_split_group_ptr": torch.tensor(
                [0, fanout], device=device, dtype=torch.long
            ),
            "multiple_split_parent_rows": torch.tensor(
                [0], device=device, dtype=torch.long
            ),
            "max_splits_per_multiple_parent": fanout,
        }

    residual, offset = compute_dts_forward(
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
def test_pi_forward_matches_fp64_for_receiver_and_origination_cases(
    tmp_path: Path,
    weighted: bool,
) -> None:
    _require_native_cuda()
    species_tree, gene_trees = _write_tiny_ale_example(tmp_path)
    def build(dtype: torch.dtype, e_tol: float) -> GeneReconModel:
        return GeneReconModel(
            species_tree,
            gene_trees,
            device="cuda",
            dtype=dtype,
            family_chunk_size=1,
            clade_budget=None,
            batch_packing="sequential",
            max_wave_size=8,
            solver_options=SolverOptions(
                e_max_iter=32,
                e_tol=e_tol,
                pi_iters=8,
                neumann_terms=16,
            ),
        )

    model32 = build(torch.float32, 1e-7)
    model64 = build(torch.float64, 1e-12)
    with torch.no_grad():
        model64.theta.copy_(model32.theta.double())
    receiver32 = torch.zeros_like(model32.receiver_weights)
    origination32 = torch.zeros_like(model32.origination_weights)
    if weighted:
        receiver32 = torch.linspace(
            -1.25, 1.75, receiver32.numel(), device=receiver32.device
        )
        origination32 = torch.linspace(
            0.9, -1.1, origination32.numel(), device=origination32.device
        )
    receiver64 = receiver32.double()
    origination64 = origination32.double()

    split_metas = [meta for meta in model32.wave_layout["wave_metas"] if "sl" in meta]
    assert any(int(meta["n_eq1"]) > 0 for meta in split_metas)
    assert any(
        meta.get("ge2_parent_ids") is not None
        and int(meta["ge2_parent_ids"].numel()) > 0
        for meta in split_metas
    )

    solved32 = solve_resident_e_pi(
        model32.batch_statics[0],
        model32.theta.detach(),
        receiver32,
        warm_start_E=None,
    )
    # The batch static intentionally stores only the most recent solve's sidecar.
    # Retain the state paired with this result before later diagnostic/model solves
    # replace that reference.
    state32 = model32.batch_statics[0].pi_forward_state
    assert state32 is not None
    residual32 = solve_forward_residual(
        model32.batch_statics[0],
        model32.theta.detach(),
        receiver32,
        pi_iters=8,
        warm_start_E=None,
    )
    solved64 = solve_resident_e_pi(
        model64.batch_statics[0],
        model64.theta.detach(),
        receiver64,
        warm_start_E=None,
    )
    state64 = model64.batch_statics[0].pi_forward_state
    assert state64 is not None
    residual64 = solve_forward_residual(
        model64.batch_statics[0],
        model64.theta.detach(),
        receiver64,
        pi_iters=8,
        warm_start_E=None,
    )
    convergence = evaluate_static_convergence(
        model32.batch_statics[0],
        model32.theta.detach(),
        receiver32,
        pi_iters_high=8,
        neumann_terms=16,
    )
    def loss_and_grads(model, receiver, origination):
        theta = model.theta.detach().clone().requires_grad_(True)
        receiver = receiver.detach().clone().requires_grad_(True)
        origination = origination.detach().clone().requires_grad_(True)
        loss = model(
            theta=theta,
            receiver_weights=receiver,
            origination_weights=origination,
        )
        return loss, torch.autograd.grad(loss, (theta, receiver, origination))

    loss32, grads32 = loss_and_grads(model32, receiver32, origination32)
    loss64, grads64 = loss_and_grads(model64, receiver64, origination64)
    assert state32.pi_offset.dtype == state64.pi_offset.dtype == torch.float64
    assert torch.isfinite(loss32).item()
    torch.testing.assert_close(loss32, loss64, rtol=0.0, atol=2e-5)
    torch.testing.assert_close(residual32, residual64, rtol=0.0, atol=2e-5)
    torch.testing.assert_close(convergence[0], residual32, rtol=0.0, atol=0.0)
    assert all(torch.isfinite(value).all().item() for value in convergence)
    for grad32, grad64 in zip(grads32, grads64):
        assert torch.isfinite(grad32).all().item()
        torch.testing.assert_close(grad32.double(), grad64, rtol=5e-5, atol=2e-5)

    root32, pi32, pibar32, row_max32 = solved32[4:8]
    root64, pi64, pibar64, row_max64 = solved64[4:8]
    torch.testing.assert_close(root32, root64, rtol=0.0, atol=2e-5)
    torch.testing.assert_close(
        state32.reconstruct_pi(pi32), state64.reconstruct_pi(pi64), rtol=0.0, atol=2e-5
    )
    pi_absolute32 = state32.reconstruct_pi(pi32)
    pibar_absolute32 = state32.reconstruct_pibar(pibar32)
    pibar_absolute64 = state64.reconstruct_pibar(pibar64)
    finite32 = torch.isfinite(pibar_absolute32)
    finite64 = torch.isfinite(pibar_absolute64)
    # Pibar excludes the current species' ancestor path by subtracting its mass
    # from the row sum.  A complement below one fp32 ulp of the row maximum can
    # therefore round to zero in fp32 even though its log value is representable.
    # That is the ordinary hot-reduction precision floor, not a framing error.
    assert not bool((finite32 & ~finite64).any().item())
    common_finite = finite32 & finite64
    torch.testing.assert_close(
        pibar_absolute32[common_finite],
        pibar_absolute64[common_finite],
        rtol=0.0,
        atol=2e-5,
    )
    fp32_log_ulp = math.log2(torch.finfo(torch.float32).eps)
    lost_sub_ulp = ~finite32 & finite64
    resolution_floor = pi_absolute32.amax(dim=1, keepdim=True) + fp32_log_ulp
    assert bool(
        (pibar_absolute64[lost_sub_ulp] <= resolution_floor.expand_as(pibar_absolute64)[lost_sub_ulp])
        .all()
        .item()
    )
    torch.testing.assert_close(
        state32.reconstruct_pibar_row_max(row_max32),
        state64.reconstruct_pibar_row_max(row_max64),
        rtol=0.0,
        atol=2e-5,
    )



@pytest.mark.gpu
def test_pi_forward_streams_batches_without_losing_fp64_loss(tmp_path: Path) -> None:
    _require_native_cuda()
    species_tree, gene_trees = _write_tiny_ale_example(tmp_path)
    options = SolverOptions(
        e_max_iter=8,
        e_tol=1e-5,
        pi_iters=4,
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


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [False, True])
def test_exact_tree_self_loop_matches_converged_log_self_loop(
    tmp_path: Path,
    weighted: bool,
) -> None:
    """``forward_self_loop="exact"`` solves the fixed point the log path iterates towards.

    The exact solve eliminates the self-loop on the species tree instead of sweeping it, so the
    only honest reference is the log-space path run far past convergence (``pi_iters=256`` here,
    against the exact solve's own 8, which it ignores). Everything is float64 so the comparison
    is about the two algorithms, not about float32 rounding.

    Compared only WITHIN ``window`` log2 units of each row's maximum, for a reason that is a
    property of the representation and not of this kernel. The exact solve, like the fused linear
    one, works on a row rescaled so its largest entry is 1; a lane whose true value is further
    below that maximum than the dtype's roundoff (about 53 log2 units in float64) is built
    entirely out of terms that round to nothing, so it comes out as an exact zero and is
    published as -inf. The log-space path has no such floor. ``window`` stays well inside it.
    """
    _require_native_cuda()
    species_tree, gene_trees = _write_tiny_ale_example(tmp_path)

    def build(mode: str, pi_iters: int) -> GeneReconModel:
        return GeneReconModel(
            species_tree,
            gene_trees,
            device="cuda",
            dtype=torch.float64,
            family_chunk_size=1,
            clade_budget=None,
            batch_packing="sequential",
            max_wave_size=8,
            solver_options=SolverOptions(
                e_max_iter=64,
                e_tol=1e-13,
                pi_iters=pi_iters,
                neumann_terms=16,
                forward_self_loop=mode,
            ),
        )

    reference = build("log", 256)
    candidate = build("exact", 8)
    with torch.no_grad():
        candidate.theta.copy_(reference.theta)
    receiver = torch.zeros_like(reference.receiver_weights)
    if weighted:
        receiver = torch.linspace(
            -1.25, 1.75, receiver.numel(), device=receiver.device, dtype=receiver.dtype
        )

    def solved_rows(model: GeneReconModel):
        static = model.batch_statics[0]
        result = solve_resident_e_pi(
            static, model.theta.detach(), receiver, warm_start_E=None
        )
        state = static.pi_forward_state
        pi = result[5].to(state.pi_offset.dtype) + state.pi_offset.unsqueeze(1)
        pibar = result[6].to(state.pibar_offset.dtype) + state.pibar_offset.unsqueeze(1)
        return pi, pibar, result[4]

    # What limits agreement inside the window is not a lane's depth below its row maximum but the
    # size of the row's whole transfer mass T, which the transfer term multiplies: a lane picks up
    # about one roundoff of ``gamma[s] * T`` however small the lane itself is. Measured on this
    # example at window = 30, the worst entry differs by 4.6e-6 log2 units (a relative 6.7e-8 on a
    # value near 69). ``tolerance`` sits an order of magnitude above that and four orders BELOW
    # the scale an actual algorithmic regression shows up at -- the two bugs this test was written
    # after both moved entries by more than 10 log2 units.
    window = 30.0
    tolerance = 1e-4

    def in_window(rows: torch.Tensor) -> torch.Tensor:
        """Reference entries no more than ``window`` log2 units below their own row's maximum."""
        finite = torch.isfinite(rows)
        row_max = torch.where(finite, rows, torch.full_like(rows, -float("inf")))
        row_max = row_max.amax(dim=1, keepdim=True)
        return finite & torch.isfinite(row_max) & (rows >= row_max - window)

    reference_pi, reference_pibar, reference_root = solved_rows(reference)
    exact_pi, exact_pibar, exact_root = solved_rows(candidate)

    for name, reference_rows, exact_rows in (
        ("Pi", reference_pi, exact_pi),
        ("Pibar", reference_pibar, exact_pibar),
    ):
        selected = in_window(reference_rows)
        assert bool(selected.any()), f"no {name} entries inside the {window} log2 window"
        # Nothing the reference can still resolve at this depth may vanish on the exact path.
        assert bool(
            torch.isfinite(exact_rows[selected]).all()
        ), f"exact {name} lost an entry the reference resolves"
        torch.testing.assert_close(
            exact_rows[selected], reference_rows[selected], rtol=0.0, atol=tolerance
        )
    torch.testing.assert_close(exact_root, reference_root, rtol=0.0, atol=tolerance)
