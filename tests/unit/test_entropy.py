from __future__ import annotations

import math

import pytest
import torch

import gpurec
from gpurec.entropy import (
    compute_reconciliation_entropy,
    reconciliation_entropy_from_payload,
)


NEG = -1.0e300


def _matrix(values: list[list[float]]) -> dict[str, object]:
    rows = len(values)
    cols = len(values[0]) if rows else 0
    return {
        "rows": rows,
        "cols": cols,
        "data": [item for row in values for item in row],
    }


def _base_payload(
    *,
    pi: list[list[float]],
    pibar: list[list[float]] | None = None,
    e: list[float] | None = None,
    ebar: list[float] | None = None,
    log_p_s: list[float] | None = None,
    log_p_d: list[float] | None = None,
    log_p_l: list[float] | None = None,
    max_transfer: list[float] | None = None,
    leaf_species: list[int | None] | None = None,
    splits: list[dict[str, float | int]] | None = None,
) -> dict[str, object]:
    species = len(pi[0])
    clades = len(pi)
    return {
        "species_newick": "s0;",
        "species_names_postorder": ["s0"],
        "root_clade": 0,
        "leaf_species": leaf_species if leaf_species is not None else [0] * clades,
        "clade_leaf_labels": [""] * clades,
        "splits": [] if splits is None else splits,
        "pi": _matrix(pi),
        "pibar": _matrix(pibar if pibar is not None else [[NEG] * species for _ in range(clades)]),
        "e": e if e is not None else [NEG] * species,
        "ebar": ebar if ebar is not None else [NEG] * species,
        "log_p_s": log_p_s if log_p_s is not None else [0.0] * species,
        "log_p_d": log_p_d if log_p_d is not None else [NEG] * species,
        "log_p_l": log_p_l if log_p_l is not None else [NEG] * species,
        "max_transfer": max_transfer if max_transfer is not None else [NEG] * species,
        "origination_probs": [1.0] + [0.0] * (species - 1),
        "seed": None,
        "max_events": None,
    }


def test_leaf_only_entropy_is_zero():
    payload = _base_payload(pi=[[0.0]], leaf_species=[0])

    result = reconciliation_entropy_from_payload(
        payload,
        species_child1=[-1],
        species_child2=[-1],
        species_parent=[-1],
        mode="both",
    )

    assert result["collapsed_bits"] == pytest.approx(0.0)
    assert result["expanded_bits"] == pytest.approx(0.0)
    assert result["collapsed_converged"] is True
    assert result["expanded_pi_converged"] is True


def test_split_choice_entropy_is_one_bit():
    payload = _base_payload(
        pi=[[0.0], [0.0], [0.0]],
        leaf_species=[None, 0, 0],
        log_p_s=[0.0],
        log_p_d=[0.0],
        splits=[
            {"parent": 0, "left": 1, "right": 2, "log_prob": -1.0},
            {"parent": 0, "left": 1, "right": 2, "log_prob": -1.0},
        ],
    )

    result = reconciliation_entropy_from_payload(
        payload,
        species_child1=[-1],
        species_child2=[-1],
        species_parent=[-1],
        mode="collapsed",
    )

    assert result["collapsed_bits"] == pytest.approx(1.0)


def test_expanded_mode_adds_hidden_duplication_loss_orientation_entropy():
    pi_root = math.log2(2.0 / 3.0)
    payload = _base_payload(
        pi=[[pi_root]],
        e=[-1.0],
        log_p_s=[-1.0],
        log_p_d=[-2.0],
        log_p_l=[-1.0],
        leaf_species=[0],
    )

    result = reconciliation_entropy_from_payload(
        payload,
        species_child1=[-1],
        species_child2=[-1],
        species_parent=[-1],
        mode="both",
        tol=1e-12,
    )

    binary_h = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert result["collapsed_bits"] == pytest.approx(binary_h / 0.75)
    assert result["expanded_bits"] > result["collapsed_bits"]


def test_top_level_entropy_exports_are_lazy():
    assert "compute_reconciliation_entropy" in gpurec.__all__
    assert "reconciliation_entropy_from_payload" in gpurec.__all__
    assert gpurec._LAZY_EXPORTS["compute_reconciliation_entropy"] == "gpurec.entropy"
    assert gpurec.compute_reconciliation_entropy is compute_reconciliation_entropy


def test_compute_reconciliation_entropy_uses_exported_backtracking_payload(monkeypatch):
    import gpurec.entropy as entropy

    payload = _base_payload(pi=[[0.0]], leaf_species=[0])
    model = object()
    calls: list[tuple[object, int]] = []

    def fake_export(model_arg: object, *, family_index: int):
        calls.append((model_arg, family_index))
        return payload

    monkeypatch.setattr(entropy, "export_backtracking_input", fake_export)
    monkeypatch.setattr(
        entropy,
        "_species_tensors_from_model",
        lambda *_args, **_kwargs: (
            torch.tensor([-1]),
            torch.tensor([-1]),
            torch.tensor([-1]),
        ),
    )

    result = compute_reconciliation_entropy(model, family_index=3, mode="collapsed")

    assert calls == [(model, 3)]
    assert result["collapsed_bits"] == pytest.approx(0.0)
