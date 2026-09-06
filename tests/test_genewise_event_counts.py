"""Focused tests for the opt-in genewise expected-event-count output."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gpurec.api._execution as execution
import gpurec.api.model as model_module
from gpurec.api.model import GeneReconModel
from gpurec.core.scheduling import batching


def _stub_model(dtype: torch.dtype = torch.float32) -> GeneReconModel:
    model = GeneReconModel.__new__(GeneReconModel)
    torch.nn.Module.__init__(model)
    model.genewise = True
    model.theta = torch.nn.Parameter(torch.zeros(3, 3, dtype=dtype))
    model.receiver_weights = torch.nn.Parameter(torch.zeros(2, dtype=dtype), requires_grad=False)
    model.origination_weights = torch.nn.Parameter(torch.zeros(2, dtype=dtype), requires_grad=False)
    model.batch_statics = []
    return model


def test_public_event_count_buffer_validation_and_return_arity(monkeypatch):
    model = _stub_model()
    seen = {}

    def fake_stream(*args, event_counts_out=None, **kwargs):
        seen["buffer"] = event_counts_out
        return torch.zeros(3), torch.zeros(3, 3), None, None

    monkeypatch.setattr(model_module, "stream_genewise_loss_vector_grad", fake_stream)
    counts = torch.empty(3, 4, dtype=torch.float64)
    result = model.genewise_loss_vector_and_grad(event_counts_out=counts)
    assert len(result) == 3
    assert seen["buffer"] is counts

    with pytest.raises(ValueError, match="requires need_grad=True"):
        model.genewise_loss_vector_and_grad(need_grad=False, event_counts_out=counts)
    with pytest.raises(ValueError, match=r"shape \(3, 4\)"):
        model.genewise_loss_vector_and_grad(event_counts_out=torch.empty(3, 3))
    with pytest.raises(TypeError, match="must not be narrower"):
        double_model = _stub_model(torch.float64)
        double_model.genewise_loss_vector_and_grad(event_counts_out=torch.empty(3, 4, dtype=torch.float32))
    with pytest.raises(TypeError, match="torch.float32 or torch.float64"):
        model.genewise_loss_vector_and_grad(event_counts_out=torch.empty(3, 4, dtype=torch.int64))


def test_stream_scatters_batch_local_event_counts(monkeypatch):
    statics = [
        SimpleNamespace(genewise=True, family_index_tensor=torch.tensor([2, 0])),
        SimpleNamespace(genewise=True, family_index_tensor=torch.tensor([1])),
    ]

    def fake_evaluate(
        static, theta, receiver_weights, origination_weights, *, need_grad,
        need_receiver_grad, update_warm_start, need_origination_grad,
        event_counts_out,
    ):
        assert event_counts_out is not None
        event_counts_out.copy_(theta[:, :1] + torch.arange(4, dtype=theta.dtype))
        return theta[:, 0].clone(), torch.ones_like(theta), None, None

    monkeypatch.setattr(execution, "evaluate_static_loss_vector_grad", fake_evaluate)
    theta = torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    counts = torch.full((3, 4), float("nan"))
    loss, gradient, receiver_gradient, origination_gradient = execution.stream_genewise_loss_vector_grad(
        statics,
        theta,
        torch.zeros(2),
        torch.zeros(2),
        need_grad=True,
        need_receiver_grad=False,
        event_counts_out=counts,
    )
    torch.testing.assert_close(loss, theta[:, 0])
    torch.testing.assert_close(gradient, torch.ones_like(theta))
    torch.testing.assert_close(counts, theta[:, :1] + torch.arange(4, dtype=theta.dtype))
    assert receiver_gradient is None
    assert origination_gradient is None


def _require_native_cuda() -> None:
    try:
        batching._load_native_module()
    except Exception as exc:
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _tiny_model(tmp_path: Path) -> GeneReconModel:
    _require_native_cuda()
    species = tmp_path / "species.nwk"
    species.write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
    families = []
    for index, newick in enumerate((
        "(A_1:1,(B_1:1,C_1:1)X:1)R:1;\n",
        "((A_2:1,B_2:1)X:1,C_2:1)R:1;\n",
    )):
        family = tmp_path / f"family_{index}.nwk"
        family.write_text(newick, encoding="utf-8")
        families.append(family)
    model = GeneReconModel(
        species, families, mode="genewise", device="cuda", dtype=torch.float32,
        family_chunk_size=2, clade_budget=None, max_wave_size=8,
    )
    model.receiver_weights.requires_grad_(False)
    with torch.no_grad():
        model.theta[:] = torch.tensor([
            [math.log2(0.02), math.log2(0.1), math.log2(0.04)],
            [math.log2(0.05), math.log2(0.2), math.log2(0.03)],
        ], device="cuda")
    return model


@pytest.mark.parametrize("trainable_receiver", [False, True])
def test_event_counts_preserve_gradient_and_fold_back_to_it(tmp_path: Path, trainable_receiver: bool):
    model = _tiny_model(tmp_path)
    model.receiver_weights.requires_grad_(trainable_receiver)
    default_result = model.genewise_loss_vector_and_grad()
    counts = torch.empty(2, 4, device="cuda", dtype=torch.float64)
    counted_result = model.genewise_loss_vector_and_grad(event_counts_out=counts)

    assert len(default_result) == len(counted_result) == 3
    torch.testing.assert_close(counted_result[0], default_result[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(counted_result[1], default_result[1], rtol=2e-5, atol=2e-4)
    if trainable_receiver:
        assert counted_result[2] is not None and default_result[2] is not None
        torch.testing.assert_close(counted_result[2], default_result[2], rtol=2e-5, atol=2e-4)
    else:
        assert counted_result[2] is None and default_result[2] is None
    assert bool(torch.isfinite(counts).all())
    assert bool((counts >= 0.0).all())

    theta = model.theta.detach().double()
    rates = torch.exp2(theta)
    event_probabilities = rates / (1.0 + rates.sum(dim=1, keepdim=True))
    folded = event_probabilities * counts.sum(dim=1, keepdim=True) - counts[:, 1:]
    torch.testing.assert_close(folded, counted_result[1].double(), rtol=2e-4, atol=3e-4)
