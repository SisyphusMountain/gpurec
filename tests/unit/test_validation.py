from __future__ import annotations

from pathlib import Path

import pytest
import torch

import gpurec.workflow.model_factory as workflow_model_factory
from gpurec.api._validation import require_cuda_device
from gpurec.workflow.config import RunConfig
from gpurec.workflow.model_factory import build_alerax_workflow_model


def test_require_cuda_device_rejects_malformed_cuda_spec() -> None:
    with pytest.raises(ValueError, match="invalid CUDA device"):
        require_cuda_device("cuda:not-a-device", owner="unit test")


def test_require_cuda_device_rejects_explicit_index_outside_available_range(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="only 1 CUDA device"):
        require_cuda_device("cuda:1", owner="unit test")


def test_require_cuda_device_accepts_available_explicit_index(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert require_cuda_device("cuda:0", owner="unit test") == torch.device("cuda:0")


def test_build_alerax_workflow_model_rejects_unavailable_cuda_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = RunConfig(
        species_tree=tmp_path / "sp.nwk",
        families_file=tmp_path / "families.txt",
        out_dir=tmp_path / "out",
        mode="global",
        device="cuda:2",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def fail_from_alerax_families(*args: object, **kwargs: object) -> object:
        raise AssertionError("model construction should not run")

    monkeypatch.setattr(
        workflow_model_factory.GeneReconModel,
        "from_alerax_families",
        staticmethod(fail_from_alerax_families),
    )

    with pytest.raises(ValueError, match="only 1 CUDA device"):
        build_alerax_workflow_model(config)
