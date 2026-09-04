import dataclasses
from types import SimpleNamespace
import inspect

import pytest
import torch

from gpurec.api import _execution
from gpurec.api.solver_options import SolverOptions
from gpurec.cli import _common
from gpurec.cli.main import build_parser
from gpurec.core.backtracking import input as backtracking_input
from gpurec.core.inference import solver
from gpurec.fit import dtl_fit, global_fit
from gpurec.solver import value_and_grad


def _static(
    *,
    genewise: bool = False,
    accumulator_dtype: torch.dtype = torch.float64,
):
    return SimpleNamespace(
        solver_options=SolverOptions(),
        genewise=genewise,
        accumulator_dtype=accumulator_dtype,
    )


def test_pi_representation_option_is_removed(tmp_path):
    assert not hasattr(SolverOptions(), "pi_representation")
    with pytest.raises(TypeError, match="pi_representation"):
        SolverOptions(pi_representation="centered")

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "reconcile",
                "--species",
                "species.nwk",
                "--gene",
                "gene.nwk",
                "--pi-representation",
                "centered",
            ]
        )

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
        ]
    )
    with pytest.raises(ValueError, match="pi_representation"):
        _common.make_solver_options(args)


def test_fit_dtl_routes_solver_options_to_genewise_recipe(monkeypatch):
    options = SolverOptions(pi_iters=16)
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


def test_global_tiers_have_no_representation_option():
    # ``_tier_solver_options`` now takes the caller's already-resolved base solver settings and
    # overrides only the two tier counts, so the base has to be passed in. It is a dict of every
    # SolverOptions field, exactly as fit_global builds it.
    base = {f.name: getattr(SolverOptions(), f.name) for f in dataclasses.fields(SolverOptions)}
    fit_tier = global_fit._tier_solver_options(base, pi_iters=16, neumann_terms=16)
    eval_tier = global_fit._tier_solver_options(base, pi_iters=64, neumann_terms=64)

    assert not hasattr(fit_tier, "pi_representation")
    assert not hasattr(eval_tier, "pi_representation")
    assert (fit_tier.pi_iters, eval_tier.pi_iters) == (16, 64)
    assert (fit_tier.neumann_terms, eval_tier.neumann_terms) == (16, 64)
    # Everything except the two tier counts is carried through from the base unchanged.
    assert fit_tier.e_adjoint_max_iter == SolverOptions().e_adjoint_max_iter
    assert fit_tier.forward_self_loop == SolverOptions().forward_self_loop


def test_pi_wave_forward_has_no_representation_parameter():
    assert "pi_representation" not in inspect.signature(solver.pi_wave_forward).parameters


def test_solver_stores_required_pi_state(monkeypatch):
    options = SolverOptions()
    static = SimpleNamespace(
        solver_options=options,
        species_helpers={
            "S": 2,
            "unnorm_row_max": torch.zeros(2),
            "sp_parent": torch.zeros(2, dtype=torch.long),
            "sp_child1": torch.zeros(2, dtype=torch.long),
            "sp_child2": torch.zeros(2, dtype=torch.long),
            "max_ancestor_depth": 0,
            # Two leaves, no internal node: every height is 0 and the compact level table is empty.
            "sp_height": torch.zeros(2, dtype=torch.int32),
            "compact_level_ptr": torch.zeros(1, dtype=torch.long),
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

    sidecar = SimpleNamespace(
        pi_offset=torch.zeros(1, dtype=torch.float64),
        pibar_offset=torch.zeros(1, dtype=torch.float64),
    )

    def fake_pi_wave_forward(**kwargs):
        captured.update(kwargs)
        matrix = torch.zeros((1, 2))
        return matrix, matrix, matrix, torch.zeros(1), sidecar

    monkeypatch.setattr(solver, "pi_wave_forward", fake_pi_wave_forward)
    solver.solve_resident_e_pi(static, theta, receiver_weights)

    assert "pi_representation" not in captured
    assert static.pi_forward_state is sidecar


def test_forward_solve_snapshots_matching_sidecar(monkeypatch):
    static = _static()
    static.warm_E = None
    sidecar = SimpleNamespace(
        pi_offset=torch.tensor([11.0], dtype=torch.float64),
        pibar_offset=torch.tensor([12.0], dtype=torch.float64),
    )
    rows = torch.zeros((1, 2))

    def fake_solve(*_args, **_kwargs):
        static.pi_forward_state = sidecar
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
    assert saved["pi_state"] is sidecar
    # A later solve may replace the mutable static field; the saved forward
    # must retain the offsets that belong to its own residual matrices.
    static.pi_forward_state = SimpleNamespace(
        pi_offset=torch.tensor([99.0], dtype=torch.float64),
        pibar_offset=torch.tensor([100.0], dtype=torch.float64),
    )
    assert saved["pi_state"] is sidecar


def test_backtracking_reconstructs_selected_family(monkeypatch):
    pi_residual = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    pibar_residual = -pi_residual
    pi_offset = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
    pibar_offset = -pi_offset
    static = _static()
    static.family_indices = [0, 1]
    static.wave_layout = {"perm": torch.tensor([2, 0, 1])}
    static.pi_forward_state = SimpleNamespace(
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


@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_streamed_scalar_loss_uses_configured_accumulator(
    monkeypatch,
    accumulator_dtype,
):
    statics = [
        _static(accumulator_dtype=accumulator_dtype),
        _static(accumulator_dtype=accumulator_dtype),
    ]
    source_dtype = (
        torch.float64 if accumulator_dtype == torch.float32 else torch.float32
    )
    monkeypatch.setattr(
        _execution,
        "evaluate_static_loss_grad",
        lambda *_args, **_kwargs: (
            torch.tensor(0.25, dtype=source_dtype),
            None,
            None,
            None,
        ),
    )

    loss, *_ = _execution.stream_batches(
        statics,
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        genewise=False,
        need_grad=False,
        need_receiver_grad=False,
    )

    assert loss.dtype == accumulator_dtype
    assert loss.item() == 0.5


def test_streamed_scalar_loss_without_policy_derives_theta_dtype(monkeypatch):
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
        need_receiver_grad=False,
    )

    assert loss.dtype == torch.float32


@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_streamed_genewise_loss_uses_configured_accumulator(
    monkeypatch,
    accumulator_dtype,
):
    statics = []
    for index in range(2):
        static = _static(
            genewise=True,
            accumulator_dtype=accumulator_dtype,
        )
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
        need_receiver_grad=False,
    )

    assert loss.dtype == accumulator_dtype
    torch.testing.assert_close(
        loss,
        torch.tensor([0.25, 0.25], dtype=accumulator_dtype),
    )


def test_streaming_rejects_mixed_accumulator_dtypes():
    with pytest.raises(ValueError, match="share one accumulator dtype"):
        _execution.stream_batches(
            [
                _static(accumulator_dtype=torch.float32),
                _static(accumulator_dtype=torch.float64),
            ],
            torch.zeros(3),
            torch.zeros(2),
            torch.zeros(2),
            genewise=False,
            need_grad=False,
            need_receiver_grad=False,
        )
