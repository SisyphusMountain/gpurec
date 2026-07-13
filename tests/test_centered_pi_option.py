from types import SimpleNamespace

import pytest
import torch

from gpurec.api import _execution
from gpurec.api.solver_options import SolverOptions
from gpurec.cli import _common
from gpurec.cli.main import build_parser
from gpurec.core.backtracking import input as backtracking_input
from gpurec.core.inference import forward, solver
from gpurec.fit import dtl_fit, global_fit
from gpurec.solver import value_and_grad


def _centered_static(*, genewise: bool = False):
    return SimpleNamespace(
        solver_options=SolverOptions(pi_representation="centered"),
        genewise=genewise,
    )


def test_pi_representation_default_and_validation():
    assert SolverOptions().pi_representation == "absolute"

    centered = SolverOptions(pi_representation="  CENTERED ")
    centered.validate()
    assert centered.pi_representation == "centered"

    with pytest.raises(ValueError, match="pi_representation"):
        SolverOptions(pi_representation="relative").validate()


def test_cli_pi_representation_overrides_toml(tmp_path):
    config = tmp_path / "gpurec.toml"
    config.write_text('[solver]\npi_representation = "absolute"\n', encoding="utf-8")
    args = build_parser().parse_args(
        [
            "reconcile",
            "--species",
            "species.nwk",
            "--gene",
            "gene.nwk",
            "--config",
            str(config),
            "--pi-representation",
            "centered",
        ]
    )
    assert _common.make_solver_options(args).pi_representation == "centered"


def test_fit_dtl_routes_centered_representation_to_genewise_recipe(monkeypatch):
    options = SolverOptions(pi_representation="centered")
    captured = {}

    def fake_fit_genewise(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "loss_bits": 1.0,
            "theta": torch.zeros((1, 3)),
            "rates": torch.ones((1, 3)),
            "n_families": 1,
        }

    monkeypatch.setattr(dtl_fit, "fit_genewise", fake_fit_genewise)
    dtl_fit.fit_dtl(
        "species.nwk",
        ["gene.nwk"],
        "genewise",
        solver_options=options,
    )

    assert captured["solver_options"] is options


def test_global_tiers_preserve_centered_representation():
    options = SolverOptions(pi_representation="centered")

    fit_tier = global_fit._tier_solver_options(
        options, pi_iters=16, neumann_terms=16
    )
    eval_tier = global_fit._tier_solver_options(
        options, pi_iters=64, neumann_terms=64
    )

    assert fit_tier.pi_representation == "centered"
    assert eval_tier.pi_representation == "centered"
    assert (fit_tier.pi_iters, eval_tier.pi_iters) == (16, 64)


def test_pi_wave_forward_dispatches_centered_explicitly(monkeypatch):
    sentinel = object()
    captured = {}

    def fake_centered(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(forward, "pi_wave_forward_centered", fake_centered)
    result = forward.pi_wave_forward(
        wave_layout="layout",
        species_helpers="species",
        e="e",
        e_bar="e_bar",
        e_s1="e_s1",
        e_s2="e_s2",
        log_p_s="log_p_s",
        log_p_d="log_p_d",
        max_transfer_mat="max_transfer",
        receiver_log_probs="receiver",
        family_idx="family",
        pi_representation="centered",
    )

    assert result is sentinel
    assert captured["wave_layout"] == "layout"
    assert captured["family_idx"] == "family"


def test_solver_routes_pi_representation(monkeypatch):
    options = SolverOptions(pi_representation="centered")
    static = SimpleNamespace(
        solver_options=options,
        species_helpers={
            "S": 2,
            "unnorm_row_max": torch.zeros(2),
            "sp_parent": torch.zeros(2, dtype=torch.long),
            "sp_child1": torch.zeros(2, dtype=torch.long),
            "sp_child2": torch.zeros(2, dtype=torch.long),
            "max_ancestor_depth": 0,
        },
        specieswise=False,
        genewise=False,
        wave_layout={"root_clade_ids": torch.tensor([0])},
        rate_family_idx=torch.zeros(1, dtype=torch.long),
    )
    theta = torch.zeros(3)
    receiver_weights = torch.zeros(2)
    scalar = torch.zeros(1)
    species = torch.zeros(2)

    monkeypatch.setattr(
        solver,
        "extract_parameters_uniform",
        lambda *_args, **_kwargs: (scalar, scalar, scalar, species),
    )
    monkeypatch.setattr(
        solver,
        "e_fixed_point_triton",
        lambda *_args, **_kwargs: (species, species, species, species),
    )
    captured = {}

    def fake_pi_wave_forward(**kwargs):
        captured.update(kwargs)
        matrix = torch.zeros((1, 2))
        return matrix, matrix, matrix, torch.zeros(1)

    monkeypatch.setattr(solver, "pi_wave_forward", fake_pi_wave_forward)
    solver.solve_resident_e_pi(static, theta, receiver_weights)

    assert captured["pi_representation"] == "centered"


def test_centered_forward_solve_snapshots_matching_sidecar(monkeypatch):
    static = _centered_static()
    static.warm_E = None
    sidecar = SimpleNamespace(
        pi_offset=torch.tensor([11.0], dtype=torch.float64),
        pibar_offset=torch.tensor([12.0], dtype=torch.float64),
    )
    rows = torch.zeros((1, 2))

    def fake_solve(*_args, **_kwargs):
        static.centered_pi_forward_state = sidecar
        return (rows,) * len(value_and_grad.FORWARD_SAVED_NAMES)

    monkeypatch.setattr(value_and_grad, "solve_resident_e_pi", fake_solve)
    monkeypatch.setattr(
        value_and_grad,
        "nll_from_root_rows",
        lambda *_args, **_kwargs: torch.tensor(3.0),
    )
    loss, saved = value_and_grad.forward_solve(
        [static], torch.zeros(3), torch.zeros(2)
    )

    assert loss.item() == 3.0
    assert saved["centered_pi_state"] is sidecar
    # A later solve may replace the mutable static field; the saved forward
    # must retain the offsets that belong to its own residual matrices.
    static.centered_pi_forward_state = SimpleNamespace(
        pi_offset=torch.tensor([99.0], dtype=torch.float64),
        pibar_offset=torch.tensor([100.0], dtype=torch.float64),
    )
    assert saved["centered_pi_state"] is sidecar


def test_centered_backtracking_reconstructs_selected_family(monkeypatch):
    pi_residual = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    pibar_residual = -pi_residual
    pi_offset = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
    pibar_offset = -pi_offset
    static = _centered_static()
    static.family_indices = [0, 1]
    static.wave_layout = {"perm": torch.tensor([2, 0, 1])}
    static.centered_pi_forward_state = SimpleNamespace(
        pi_offset=pi_offset,
        pibar_offset=pibar_offset,
    )
    family = {
        "C": 2,
        "root_clade_id": 1,
        "leaf_row_index": [0],
        "leaf_col_index": [0],
        "split_parents_sorted": [1],
        "split_leftrights_sorted": [0, 0],
        "log_split_probs_sorted": [0.0],
    }
    model = SimpleNamespace(
        families=[{"C": 1}, family],
        batch_statics=[static],
        theta=torch.zeros(3),
        receiver_weights=torch.zeros(2),
        species_helpers={
            "S": 2,
            "sp_child1": torch.tensor([-1, -1]),
            "sp_child2": torch.tensor([-1, -1]),
            "sp_subtree_start": torch.tensor([0, 1]),
            "sp_subtree_end": torch.tensor([1, 2]),
        },
        _theta_for_static=lambda _static, theta: theta,
    )
    species_state = torch.zeros((2, 2))

    def fake_solve(*_args, **_kwargs):
        return (
            species_state,
            species_state,
            species_state,
            species_state,
            torch.zeros((2, 2)),
            pi_residual,
            pibar_residual,
            torch.zeros(3),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )

    captured = {}

    class FakeNative:
        @staticmethod
        def sample_reconciliations_torch(*args):
            captured["args"] = args
            return {"sample": "ok"}

    monkeypatch.setattr(backtracking_input, "solve_resident_e_pi", fake_solve)
    monkeypatch.setattr(backtracking_input, "_load_native_module", lambda: FakeNative())

    assert backtracking_input.sample_reconciliations(model, family_index=1) == {
        "sample": "ok"
    }
    native_args = captured["args"]
    assert native_args[5].tolist() == [[110.0, 111.0], [220.0, 221.0]]
    assert native_args[6].tolist() == [[-110.0, -111.0], [-220.0, -221.0]]


def test_streamed_centered_scalar_loss_preserves_fp64(monkeypatch):
    statics = [_centered_static(), _centered_static()]
    monkeypatch.setattr(
        _execution,
        "evaluate_static_loss_grad",
        lambda *_args, **_kwargs: (torch.tensor(0.25, dtype=torch.float64), None, None, None),
    )

    loss, *_ = _execution.stream_batches(
        statics,
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        genewise=False,
        need_grad=False,
    )

    assert loss.dtype == torch.float64
    assert loss.item() == 0.5


def test_streamed_absolute_scalar_loss_uses_fp64_head_dtype(monkeypatch):
    static = SimpleNamespace(solver_options=SolverOptions())
    monkeypatch.setattr(
        _execution,
        "evaluate_static_loss_grad",
        lambda *_args, **_kwargs: (torch.tensor(0.25, dtype=torch.float64), None, None, None),
    )

    loss, *_ = _execution.stream_batches(
        [static],
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        genewise=False,
        need_grad=False,
    )

    assert loss.dtype == torch.float64


def test_streamed_centered_genewise_loss_preserves_fp64(monkeypatch):
    statics = []
    for index in range(2):
        static = _centered_static(genewise=True)
        static.family_index_tensor = torch.tensor([index])
        statics.append(static)
    monkeypatch.setattr(
        _execution,
        "evaluate_static_loss_vector_grad",
        lambda *_args, **_kwargs: (torch.tensor([0.25], dtype=torch.float64), None, None, None),
    )

    loss, *_ = _execution.stream_genewise_loss_vector_grad(
        statics,
        torch.zeros((2, 3), dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float32),
        need_grad=False,
    )

    assert loss.dtype == torch.float64
    torch.testing.assert_close(loss, torch.tensor([0.25, 0.25], dtype=torch.float64))
