from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.core.kernels.pi_forward import compute_dts_forward
from gpurec.core.kernels.dts_tangent import compute_dts_tangent
from gpurec.core.scheduling import batching
from gpurec.solver.forward_tangent import jvp_root_scores
from gpurec.solver.value_and_grad import forward_solve


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
@pytest.mark.parametrize("fanout", [1, 3])
@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_centered_dts_tangent_matches_absolute_fp64(
    fanout: int,
    accumulator_dtype: torch.dtype,
) -> None:
    """Pin child-combination frame corrections for both eq1 and ge2 DTS."""
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
    # Large, deliberately unrelated gauges make a missing correction obvious.
    pi_offset = torch.tensor(
        [1200.125, -50.25, 800.5, 2.0], device=device, dtype=accumulator_dtype
    )
    pibar_offset = torch.tensor(
        [1199.75, -49.875, 801.25, 1.5], device=device, dtype=accumulator_dtype
    )
    dpi = torch.tensor(
        [
            [0.3, -0.2, 0.4, -0.1, 0.5],
            [-0.4, 0.1, 0.2, -0.3, 0.6],
            [0.7, -0.5, 0.1, 0.2, -0.2],
            [-0.1, 0.4, -0.6, 0.3, 0.2],
        ],
        device=device,
        dtype=torch.float32,
    )
    dpibar = torch.flip(dpi, dims=(1,)) * 0.7
    pairs = [(0, 1), (2, 3), (1, 2)][:fanout]
    lefts = torch.tensor([left for left, _ in pairs], device=device)
    rights = torch.tensor([right for _, right in pairs], device=device)
    reduce_idx = torch.zeros((fanout,), device=device, dtype=torch.long)
    sp_child1 = torch.tensor([1, 3, S, S, S], device=device)
    sp_child2 = torch.tensor([2, 4, S, S, S], device=device)
    family_idx = torch.tensor([0], device=device)
    log_p_d = torch.tensor(
        [[-2.1, -2.3, -2.5, -2.7, -2.9]], device=device
    )
    log_p_s = torch.tensor(
        [[-1.4, -1.6, -1.8, -2.0, -2.2]], device=device
    )
    dlog_p_d = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4, 0.5]], device=device
    )
    dlog_p_s = torch.tensor(
        [[-0.3, 0.2, -0.1, 0.4, -0.5]], device=device
    )
    log_split_probs = torch.tensor(
        [-0.123456789, -1.234567891, -2.345678912][:fanout],
        device=device,
        dtype=torch.float64,
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

    dts, dts_offset = compute_dts_forward(
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
    got = compute_dts_tangent(
        pi,
        pibar,
        dpi,
        dpibar,
        lefts,
        rights,
        sp_child1,
        sp_child2,
        1,
        reduce_idx,
        log_p_d,
        log_p_s,
        dlog_p_d,
        dlog_p_s,
        dts,
        family_idx,
        log_split_probs=log_split_probs,
        pi_offset=pi_offset,
        pibar_offset=pibar_offset,
        gene_split_offset=dts_offset,
    )

    dts_reference, dts_reference_offset = compute_dts_forward(
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
        log_split_probs=log_split_probs,
        **layout,
    )
    reference = compute_dts_tangent(
        pi.double(),
        pibar.double(),
        dpi.double(),
        dpibar.double(),
        lefts,
        rights,
        sp_child1,
        sp_child2,
        1,
        reduce_idx,
        log_p_d.double(),
        log_p_s.double(),
        dlog_p_d.double(),
        dlog_p_s.double(),
        dts_reference,
        family_idx,
        log_split_probs=log_split_probs,
        pi_offset=pi_offset.double(),
        pibar_offset=pibar_offset.double(),
        gene_split_offset=dts_reference_offset,
    )
    torch.testing.assert_close(got.double(), reference, rtol=2e-6, atol=3e-7)


@pytest.mark.gpu
def test_centered_root_jvp_matches_fp64_uniform_and_weighted(tmp_path: Path) -> None:
    """Exercise leaf, eq1, and ge2 waves through the complete centered JVP."""
    _require_native_cuda()
    species_tree, gene_trees = _write_tiny_ale_example(tmp_path)
    common = dict(
        e_max_iter=256,
        pi_iters=16,
        neumann_terms=0,
    )
    centered_model = GeneReconModel(
        species_tree,
        gene_trees,
        device="cuda",
        dtype=torch.float32,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=8,
        solver_options=SolverOptions(**common, e_tol=1e-7),
    )
    reference_model = GeneReconModel(
        species_tree,
        gene_trees,
        device="cuda",
        dtype=torch.float64,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=8,
        solver_options=SolverOptions(**common, e_tol=1e-12),
    )
    split_metas = [
        meta for meta in centered_model.wave_layout["wave_metas"] if "sl" in meta
    ]
    assert any(int(meta["n_eq1"]) > 0 for meta in split_metas)
    assert any(
        meta.get("ge2_parent_ids") is not None
        and int(meta["ge2_parent_ids"].numel()) > 0
        for meta in split_metas
    )

    theta32 = centered_model.theta.detach().clone()
    theta64 = theta32.double()
    torch.manual_seed(12)
    v32 = torch.randn_like(theta32) * 0.2
    v64 = v32.double()
    S = int(centered_model.receiver_weights.numel())

    for weighted in (False, True):
        if weighted:
            alpha32 = torch.linspace(-0.7, 0.9, S, device="cuda")
            u_alpha32 = torch.randn(S, device="cuda") * 0.15
        else:
            alpha32 = torch.zeros(S, device="cuda")
            u_alpha32 = None
        alpha64 = alpha32.double()
        u_alpha64 = None if u_alpha32 is None else u_alpha32.double()

        _, centered_saved = forward_solve(
            centered_model.batch_statics, theta32, alpha32
        )
        assert centered_saved["pi_state"] is not None
        centered_jvp = jvp_root_scores(
            centered_model.batch_statics[0],
            theta32,
            v32,
            centered_saved,
            self_iters=16,
            e_tol=1e-7,
            alpha=alpha32 if weighted else None,
            u_alpha=u_alpha32,
        )

        _, reference_saved = forward_solve(
            reference_model.batch_statics, theta64, alpha64
        )
        assert reference_saved["pi_state"] is not None
        reference_jvp = jvp_root_scores(
            reference_model.batch_statics[0],
            theta64,
            v64,
            reference_saved,
            self_iters=16,
            e_tol=1e-12,
            alpha=alpha64 if weighted else None,
            u_alpha=u_alpha64,
        )
        torch.testing.assert_close(
            centered_jvp.double(), reference_jvp, rtol=3e-5, atol=3e-6
        )
