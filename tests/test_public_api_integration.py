from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions, sample_reconciliations
from gpurec.core.backtracking import input as backtracking_input
from gpurec.core.inference.logspace import logsumexp2
from gpurec.core.parameters.extract_parameters import (
    extract_parameters_uniform,
    extract_parameters_weighted_receivers,
)
from gpurec.core.scheduling import batching
from gpurec.core.scheduling.batching import build_wave_layout, build_wave_layout_from_plan, preprocess_dataset


def _write_tiny_example(tmp_path: Path) -> tuple[Path, list[Path]]:
    species_tree = tmp_path / "species.nwk"
    gene_tree = tmp_path / "family_0.nwk"
    species_tree.write_text("(A:1,B:1)Root:1;\n", encoding="utf-8")
    gene_tree.write_text("(A_1:1,B_1:1)GeneRoot:1;\n", encoding="utf-8")
    return species_tree, [gene_tree]


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


def _require_preprocess_native() -> None:
    try:
        batching._load_native_module()
    except Exception as exc:
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")


def _require_backtrack_native() -> None:
    try:
        backtracking_input._load_native_module()
    except Exception as exc:
        pytest.skip(f"gpurec-backtrack extension is unavailable: {exc}")


def _require_cuda_triton() -> torch.device:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        torch.empty(1, device="cuda")
    except Exception as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return torch.device("cuda")


def _solver_options() -> SolverOptions:
    return SolverOptions(e_max_iter=4, e_tol=1e-4, pi_iters=2, neumann_terms=0)


def test_tiny_example_construction(tmp_path: Path) -> None:
    species_tree, gene_trees = _write_tiny_example(tmp_path)

    assert species_tree.read_text(encoding="utf-8").strip() == "(A:1,B:1)Root:1;"
    assert gene_trees[0].read_text(encoding="utf-8").strip() == "(A_1:1,B_1:1)GeneRoot:1;"


def test_preprocess_dataset_on_tiny_example(tmp_path: Path) -> None:
    _require_preprocess_native()
    species_tree, gene_trees = _write_tiny_example(tmp_path)

    raw = preprocess_dataset(
        species_tree,
        gene_trees,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="depth_first_fit",
        max_wave_size=4,
    )

    species = raw["species"]
    family = raw["families"][0]
    assert species["S"] == 3
    assert species["sp_child1"].dtype == torch.int32
    assert species["sp_parent"].dtype == torch.int32
    assert species["compact_level_ptr"].dtype == torch.int64
    assert family["C"] == 3
    assert family["root_clade_id"] == 0
    assert family["leaf_row_index"] == [1, 2]
    assert family["leaf_col_index"] == [0, 1]
    assert family["split_parents_sorted"] == [0]
    assert family["split_leftrights_sorted"] == [1, 2]
    assert raw["batches"] == [[0]]
    assert len(raw["batch_wave_layouts"]) == 1


def test_preprocess_dataset_on_tiny_ale_example(tmp_path: Path) -> None:
    _require_preprocess_native()
    species_tree, gene_trees = _write_tiny_ale_example(tmp_path)

    raw = preprocess_dataset(
        species_tree,
        gene_trees,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=8,
    )

    family = raw["families"][0]
    assert family["C"] == 7
    assert family["N_splits"] == 6
    assert family["root_clade_id"] == 0
    assert family["leaf_row_index"] == [1, 2, 3]
    assert family["leaf_col_index"] == [3, 0, 1]
    assert family["split_counts"][0] == 3

    root_logp = [
        logp
        for parent, logp in zip(
            family["split_parents_sorted"], family["log_split_probs_sorted"]
        )
        if parent == 0
    ]
    assert len(root_logp) == 3
    assert root_logp == pytest.approx([torch.log2(torch.tensor(1 / 3)).item()] * 3)


def test_rust_plan_and_python_wave_layout_match_on_tiny_example(tmp_path: Path) -> None:
    _require_preprocess_native()
    species_tree, gene_trees = _write_tiny_example(tmp_path)
    raw = preprocess_dataset(
        species_tree,
        gene_trees,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
    )

    from_plan = build_wave_layout_from_plan(raw["batch_wave_layouts"][0], device="cpu")
    from_python = build_wave_layout(raw["families"], device="cpu", max_wave_size=4)

    for key in ("perm", "leaf_species_index", "root_clade_ids", "family_idx"):
        assert torch.equal(from_plan[key], from_python[key])
    assert len(from_plan["wave_metas"]) == len(from_python["wave_metas"])
    for planned, fallback in zip(from_plan["wave_metas"], from_python["wave_metas"]):
        fallback_has_splits = bool(fallback.get("has_splits", "sl" in fallback))
        assert planned["start"] == fallback["start"]
        assert planned["W"] == fallback["W"]
        assert planned["end"] == fallback.get("end", fallback["start"] + fallback["W"])
        assert planned["has_splits"] == fallback_has_splits
        fallback_phase = fallback.get("phase", 2 if fallback_has_splits else 1)
        assert (planned["phase"] == 1) == (fallback_phase == 1)
        if planned["has_splits"]:
            for key in ("sl", "sr", "reduce_idx", "log_split_probs"):
                assert torch.equal(planned[key], fallback[key])


@pytest.mark.parametrize(
    ("mode", "theta_shape"),
    [
        ("global", (3,)),
        ("specieswise", (3, 3)),
        ("genewise", (1, 3)),
    ],
)
def test_gene_recon_model_constructs_for_public_modes(
    tmp_path: Path,
    mode: str,
    theta_shape: tuple[int, ...],
) -> None:
    _require_preprocess_native()
    species_tree, gene_trees = _write_tiny_example(tmp_path)

    model = GeneReconModel(
        species_tree,
        gene_trees,
        mode=mode,
        device="cpu",
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
        solver_options=_solver_options(),
    )

    assert model.mode == mode
    assert tuple(model.theta.shape) == theta_shape
    assert tuple(model.receiver_weights.shape) == (3,)
    assert torch.equal(model.receiver_weights.detach(), torch.zeros_like(model.receiver_weights))
    assert len(model.families) == 1
    assert model.family_batches == [[0]]
    assert model.species_helpers["sp_child1"].device.type == "cpu"
    assert model.wave_layout["perm"].device.type == "cpu"
    assert model.configure_solver(e_max_iter=5).e_max_iter == 5
    model.clear_warm_starts()


def test_gene_recon_model_forward_and_backward_on_tiny_example(tmp_path: Path) -> None:
    _require_preprocess_native()
    device = _require_cuda_triton()
    species_tree, gene_trees = _write_tiny_example(tmp_path)
    model = GeneReconModel(
        species_tree,
        gene_trees,
        device=device,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
        solver_options=_solver_options(),
    )

    loss = model()
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()

    loss.backward()
    assert model.theta.grad is not None
    assert model.theta.grad.shape == model.theta.shape
    assert torch.isfinite(model.theta.grad).all().item()
    assert model.receiver_weights.grad is not None
    assert model.receiver_weights.grad.shape == model.receiver_weights.shape
    assert torch.isfinite(model.receiver_weights.grad).all().item()


def test_gene_recon_model_backward_accepts_gmres_solver_options(tmp_path: Path) -> None:
    _require_preprocess_native()
    device = _require_cuda_triton()
    species_tree, gene_trees = _write_tiny_example(tmp_path)
    model = GeneReconModel(
        species_tree,
        gene_trees,
        device=device,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
        solver_options=SolverOptions(
            e_max_iter=4,
            e_tol=1e-4,
            pi_iters=2,
            neumann_terms=1,
            self_loop_solver="gmres",
        ),
    )

    loss = model()
    assert torch.isfinite(loss).item()

    loss.backward()
    assert model.theta.grad is not None
    assert torch.isfinite(model.theta.grad).all().item()
    assert model.receiver_weights.grad is not None
    assert torch.isfinite(model.receiver_weights.grad).all().item()


def test_zero_receiver_logits_match_uniform_receiver_formula(tmp_path: Path) -> None:
    _require_preprocess_native()
    species_tree = tmp_path / "species.nwk"
    gene_tree = tmp_path / "family_0.nwk"
    species_tree.write_text("((A:1,B:1)AB:1,(C:1,D:1)CD:1)Root:1;\n", encoding="utf-8")
    gene_tree.write_text("((A_1:1,B_1:1)GAB:1,(C_1:1,D_1:1)GCD:1)GeneRoot:1;\n", encoding="utf-8")
    model = GeneReconModel(
        species_tree,
        [gene_tree],
        device="cpu",
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
        solver_options=_solver_options(),
    )
    theta = model.theta.detach()
    S = int(model.species_helpers["S"])
    E = torch.linspace(-0.7, -0.1, S)

    *_, uniform_max_transfer = extract_parameters_uniform(
        theta,
        model.species_helpers["unnorm_row_max"],
    )
    *_, weighted_max_transfer, receiver_log_probs = extract_parameters_weighted_receivers(
        theta,
        model.receiver_weights.detach(),
        model.species_helpers,
    )

    for donor in range(S):
        ancestors = set()
        cur = donor
        while cur >= 0:
            ancestors.add(cur)
            cur = int(model.species_helpers["sp_parent"][cur])
        valid = torch.tensor([idx not in ancestors for idx in range(S)])
        old_pibar = uniform_max_transfer[donor] + logsumexp2(E[valid], dim=0)
        new_pibar = weighted_max_transfer[donor] + logsumexp2(receiver_log_probs[valid] + E[valid], dim=0)
        assert torch.allclose(new_pibar, old_pibar, atol=1e-6)


def test_non_uniform_receiver_logits_change_likelihood(tmp_path: Path) -> None:
    _require_preprocess_native()
    device = _require_cuda_triton()
    species_tree, gene_trees = _write_tiny_example(tmp_path)
    model = GeneReconModel(
        species_tree,
        gene_trees,
        device=device,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
        solver_options=_solver_options(),
    )
    with torch.no_grad():
        model.theta.copy_(torch.tensor([0.0, 0.0, 5.0], device=device))

    uniform_loss = model().detach()
    model.clear_warm_starts()
    biased = torch.tensor([4.0, -4.0, 0.0], device=device)
    biased_loss = model(receiver_weights=biased).detach()

    assert torch.isfinite(biased_loss).item()
    assert not torch.allclose(uniform_loss, biased_loss, atol=1e-6, rtol=1e-6)


def test_receiver_gradients_accumulate_across_batches(tmp_path: Path) -> None:
    _require_preprocess_native()
    device = _require_cuda_triton()
    species_tree, one_tree = _write_tiny_example(tmp_path)
    gene_trees = [one_tree[0], one_tree[0]]
    model = GeneReconModel(
        species_tree,
        gene_trees,
        device=device,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
        solver_options=_solver_options(),
    )
    assert len(model.batch_statics) == 2
    single = GeneReconModel(
        species_tree,
        [one_tree[0]],
        device=device,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
        solver_options=_solver_options(),
    )
    with torch.no_grad():
        single.theta.copy_(model.theta)
        single.receiver_weights.copy_(model.receiver_weights)

    model().backward()
    single().backward()

    assert model.receiver_weights.grad is not None
    assert single.receiver_weights.grad is not None
    assert torch.isfinite(model.receiver_weights.grad).all().item()
    assert torch.allclose(model.receiver_weights.grad, 2.0 * single.receiver_weights.grad, atol=1e-6, rtol=1e-5)


def test_sample_reconciliations_on_tiny_example(tmp_path: Path) -> None:
    _require_preprocess_native()
    _require_backtrack_native()
    device = _require_cuda_triton()
    species_tree, gene_trees = _write_tiny_example(tmp_path)
    model = GeneReconModel(
        species_tree,
        gene_trees,
        device=device,
        family_chunk_size=1,
        clade_budget=None,
        batch_packing="sequential",
        max_wave_size=4,
        solver_options=_solver_options(),
    )

    sample = sample_reconciliations(model, family_index=0, seed=1)

    assert isinstance(sample, list)
    assert sample[0][0] == "speciation"
    assert [event[0] for event in sample].count("leaf") == 2
    assert all(len(event) == 3 for event in sample)
