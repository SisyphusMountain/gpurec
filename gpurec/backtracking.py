"""Rust stochastic backtracking bridge for gpurec models."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from gpurec.api.model import FamilyInput, GeneReconModel, ReconciliationState


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKTRACK_MANIFEST = _REPO_ROOT / "crates" / "gpurec-backtrack" / "Cargo.toml"
_BACKTRACK_BINARY_ENV = "GPUREC_BACKTRACK_BIN"
EVENT_KEYS = ("S", "SL", "D", "DL", "T", "TL", "L", "Leaf")


def _optional_int_limit(
    name: str,
    value: int | None,
    *,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if number < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be positive")
    return number


def _validate_backtracking_limits(
    *,
    seed: int | None,
    max_events: int | None,
) -> tuple[int | None, int | None]:
    return (
        _optional_int_limit("seed", seed, minimum=0),
        _optional_int_limit("max_events", max_events, minimum=1),
    )


def _tensor_list(value: Any, *, dtype: torch.dtype | None = None) -> list:
    tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().cpu().tolist()


def _validate_finite_json_numbers(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _validate_finite_json_numbers(item, path=f"{path}[{idx}]")
        return
    if isinstance(value, tuple):
        for idx, item in enumerate(value):
            _validate_finite_json_numbers(item, path=f"{path}[{idx}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = (
                f"{path}.{key}" if isinstance(key, str) else f"{path}[{key!r}]"
            )
            _validate_finite_json_numbers(item, path=child_path)


def _activate_family_batch(model: GeneReconModel, family_index: int) -> tuple[int, int]:
    """Select the resident batch containing ``family_index``.

    ``GeneReconModel`` can stream memory-safe resident batches.  In that mode
    the active Pi matrix is batch-local, so backtracking must both activate the
    family batch and slice Pi with the family offset inside that batch.
    """
    active = model.activate_family(family_index)
    return active.clade_offset, active.local_family_index


def _species_vector(
    param: torch.Tensor,
    *,
    family_index: int,
    S: int,
    familywise_1d: bool = False,
) -> torch.Tensor:
    param = torch.as_tensor(param).detach()
    if param.ndim == 0:
        return param.reshape(1).expand(S)
    if param.ndim == 1:
        if familywise_1d:
            return param[family_index].reshape(1).expand(S)
        if int(param.shape[0]) == S:
            return param
        return param[family_index].reshape(1).expand(S)
    if param.ndim == 2:
        if int(param.shape[1]) == S:
            return param[family_index]
        if int(param.shape[1]) == 1:
            return param[family_index, 0].reshape(1).expand(S)
    raise ValueError(f"cannot convert parameter with shape {tuple(param.shape)} to [S]")


def _family_origination_probs(
    probs: torch.Tensor | None,
    *,
    family_index: int,
    S: int,
) -> list[float] | None:
    if probs is None:
        return None
    probs = probs.detach()
    if probs.ndim == 1:
        if int(probs.shape[0]) != S:
            raise ValueError(f"origination_probs length {int(probs.shape[0])} != S={S}")
        return _tensor_list(probs, dtype=torch.float64)
    if probs.ndim == 2:
        if int(probs.shape[1]) != S:
            raise ValueError(f"origination_probs shape {tuple(probs.shape)} incompatible with S={S}")
        return _tensor_list(probs[family_index], dtype=torch.float64)
    raise ValueError(f"unsupported origination_probs shape {tuple(probs.shape)}")


def _backtrack_command(
    *,
    cargo_manifest: str | Path,
    backtrack_binary: str | Path | None,
) -> list[str]:
    if backtrack_binary is None:
        backtrack_binary = os.environ.get(_BACKTRACK_BINARY_ENV)
    if backtrack_binary is not None:
        return [str(Path(backtrack_binary))]

    manifest = Path(cargo_manifest)
    if not manifest.exists():
        raise RuntimeError(
            "gpurec stochastic backtracking needs a Rust backtracking binary. "
            f"Set {_BACKTRACK_BINARY_ENV} or pass backtrack_binary; default "
            f"source manifest not found at {manifest}"
        )
    return [
        "cargo",
        "run",
        "--locked",
        "--quiet",
        "--manifest-path",
        str(manifest),
        "--",
    ]


def _evaluate_backtracking_state(model: GeneReconModel) -> ReconciliationState:
    return model.reconciliation_state(original_order=True)


def _run_backtracking_payload(
    payload: dict[str, Any],
    *,
    cargo_manifest: str | Path,
    backtrack_binary: str | Path | None,
    build_args: Callable[[Path, Path], list[str]],
    read_output: Callable[[Path], Any],
) -> Any:
    with tempfile.TemporaryDirectory(prefix="gpurec-backtrack-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.json"
        _validate_finite_json_numbers(payload, path="backtracking payload")
        input_path.write_text(
            json.dumps(payload, allow_nan=False),
            encoding="utf-8",
        )
        command = _backtrack_command(
            cargo_manifest=cargo_manifest,
            backtrack_binary=backtrack_binary,
        )
        full_command = command + build_args(input_path, tmp_path)
        result = subprocess.run(
            full_command,
            check=False,
            capture_output=True,
            cwd=str(_REPO_ROOT),
            text=True,
        )
        if result.returncode != 0:
            command_text = " ".join(shlex.quote(part) for part in full_command)
            details = [
                "gpurec backtracking command failed",
                f"exit code {result.returncode}",
                f"command: {command_text}",
            ]
            if result.stderr:
                details.append(f"stderr: {result.stderr.strip()}")
            if result.stdout:
                details.append(f"stdout: {result.stdout.strip()}")
            raise RuntimeError("; ".join(details))
        return read_output(tmp_path)


def _read_required_output(path: Path, *, description: str) -> str:
    if not path.exists():
        raise RuntimeError(
            "gpurec backtracking command succeeded but did not write "
            f"{description} at {path}"
        )
    return path.read_text(encoding="utf-8")


def export_backtracking_input(
    model: GeneReconModel,
    *,
    family_index: int = 0,
    seed: int | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable state consumed by the Rust sampler.

    The exported probabilities are base-2 logs in gpurec's local clade order
    for the selected family.
    """

    seed, max_events = _validate_backtracking_limits(seed=seed, max_events=max_events)
    family = model.family_input(family_index)
    offset, parameter_family_index = _activate_family_batch(model, family_index)
    state = _evaluate_backtracking_state(model)
    C = family.clade_count
    S = model.n_species
    ccp = family.ccp_helpers
    familywise_parameters = getattr(model, "mode", None) == "genewise"

    pi = state.pi[offset : offset + C].detach().to(dtype=torch.float64).cpu().contiguous()
    e = _species_vector(
        state.e,
        family_index=parameter_family_index,
        S=S,
    ).to(dtype=torch.float64)
    log_p_s = _species_vector(
        state.log_p_s,
        family_index=parameter_family_index,
        S=S,
        familywise_1d=familywise_parameters,
    ).to(dtype=torch.float64)
    log_p_d = _species_vector(
        state.log_p_d,
        family_index=parameter_family_index,
        S=S,
        familywise_1d=familywise_parameters,
    ).to(dtype=torch.float64)
    max_transfer = _species_vector(
        state.max_transfer,
        family_index=parameter_family_index,
        S=S,
        familywise_1d=familywise_parameters,
    ).to(dtype=torch.float64)

    N = int(ccp["N_splits"])
    parents = torch.as_tensor(ccp["split_parents_sorted"], dtype=torch.long).cpu()
    leftrights = torch.as_tensor(ccp["split_leftrights_sorted"], dtype=torch.long).cpu()
    lefts = leftrights[:N]
    rights = leftrights[N:]
    log_probs = torch.as_tensor(ccp["log_split_probs_sorted"], dtype=torch.float64).cpu()
    splits = [
        {
            "parent": int(parent),
            "left": int(left),
            "right": int(right),
            "log_prob": float(log_prob),
        }
        for parent, left, right, log_prob in zip(parents, lefts, rights, log_probs)
    ]

    leaf_species: list[int | None] = [None] * C
    leaf_rows = torch.as_tensor(family.leaf_row_index, dtype=torch.long).cpu()
    leaf_cols = torch.as_tensor(family.leaf_col_index, dtype=torch.long).cpu()
    for row, col in zip(leaf_rows.tolist(), leaf_cols.tolist()):
        leaf_species[int(row)] = int(col)

    labels = list(family.clade_leaf_labels)
    if len(labels) != C:
        labels = [""] * C

    species_names = model.species_names
    species_newick = model.species_tree_path.read_text()
    origination_probs = _family_origination_probs(
        state.origination_probs,
        family_index=parameter_family_index,
        S=S,
    )

    payload = {
        "species_newick": species_newick,
        "species_names_postorder": species_names,
        "root_clade": family.root_clade_id,
        "leaf_species": leaf_species,
        "clade_leaf_labels": labels,
        "splits": splits,
        "pi": {"rows": C, "cols": S, "data": pi.reshape(-1).tolist()},
        "e": _tensor_list(e, dtype=torch.float64),
        "log_p_s": _tensor_list(log_p_s, dtype=torch.float64),
        "log_p_d": _tensor_list(log_p_d, dtype=torch.float64),
        "max_transfer": _tensor_list(max_transfer, dtype=torch.float64),
        "origination_probs": origination_probs,
        "seed": seed,
        "max_events": max_events,
    }
    _validate_finite_json_numbers(payload, path="backtracking payload")
    return payload


def sample_recphyloxml(
    model: GeneReconModel,
    *,
    family_index: int = 0,
    seed: int | None = None,
    max_events: int | None = None,
    cargo_manifest: str | Path = _BACKTRACK_MANIFEST,
    backtrack_binary: str | Path | None = None,
) -> str:
    """Run the Rust sampler and return one RecPhyloXML document."""

    payload = export_backtracking_input(
        model,
        family_index=family_index,
        seed=seed,
        max_events=max_events,
    )

    def build_args(input_path: Path, tmp_path: Path) -> list[str]:
        return [str(input_path), str(tmp_path / "sample.xml")]

    def read_output(tmp_path: Path) -> str:
        return _read_required_output(
            tmp_path / "sample.xml",
            description="single-sample RecPhyloXML output",
        )

    return _run_backtracking_payload(
        payload,
        cargo_manifest=cargo_manifest,
        backtrack_binary=backtrack_binary,
        build_args=build_args,
        read_output=read_output,
    )


def sample_recphyloxmls(
    model: GeneReconModel,
    *,
    family_index: int = 0,
    num_samples: int,
    seed: int = 0,
    max_events: int | None = None,
    cargo_manifest: str | Path = _BACKTRACK_MANIFEST,
    backtrack_binary: str | Path | None = None,
) -> list[str]:
    """Run the Rust sampler once and return multiple RecPhyloXML documents."""

    num_samples = int(
        _optional_int_limit("num_samples", num_samples, minimum=1)
    )
    payload = export_backtracking_input(
        model,
        family_index=family_index,
        seed=seed,
        max_events=max_events,
    )

    def build_args(input_path: Path, tmp_path: Path) -> list[str]:
        return [
            "--samples",
            str(num_samples),
            "--seed",
            str(seed),
            "--output-dir",
            str(tmp_path / "samples"),
            str(input_path),
        ]

    def read_output(tmp_path: Path) -> list[str]:
        output_dir = tmp_path / "samples"
        if not output_dir.is_dir():
            raise RuntimeError(
                "gpurec backtracking command succeeded but did not create "
                f"multi-sample output directory {output_dir}"
            )
        missing = [
            f"sample_{sample_idx}.xml"
            for sample_idx in range(num_samples)
            if not (output_dir / f"sample_{sample_idx}.xml").exists()
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                "gpurec backtracking command succeeded but did not write "
                f"{len(missing)} of {num_samples} expected RecPhyloXML outputs: "
                f"{joined}"
            )
        return [
            (output_dir / f"sample_{sample_idx}.xml").read_text(encoding="utf-8")
            for sample_idx in range(num_samples)
        ]

    return _run_backtracking_payload(
        payload,
        cargo_manifest=cargo_manifest,
        backtrack_binary=backtrack_binary,
        build_args=build_args,
        read_output=read_output,
    )


def recphyloxml_event_counts(xml: str, *, alerax_style: bool = True) -> dict[str, int]:
    """Count events in a RecPhyloXML document.

    When ``alerax_style`` is true, a speciation/duplication/transfer node with
    an immediate loss child is counted as ``SL``/``DL``/``TL`` and that direct
    loss child is not also counted as ``L``. This matches AleRax's
    ``*_eventCounts_*.txt`` convention more closely than raw XML tag counts.
    """

    root = ET.fromstring(xml)
    counts = {key: 0 for key in EVENT_KEYS}

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def direct_children(clade) -> list:
        return [child for child in clade if local_name(child.tag) == "clade"]

    def event_name(clade) -> str | None:
        events = next((child for child in clade if local_name(child.tag) == "eventsRec"), None)
        if events is None:
            return None
        for event in events:
            name = local_name(event.tag)
            if name in {"speciation", "duplication", "branchingOut", "loss", "leaf"}:
                return name
        return None

    def visit(clade, *, suppress_loss: bool = False) -> None:
        event = event_name(clade)
        children = direct_children(clade)
        child_events = [event_name(child) for child in children]
        has_loss_child = "loss" in child_events

        loss_children_are_composite = False
        if event == "speciation":
            if alerax_style and has_loss_child:
                counts["SL"] += 1
                loss_children_are_composite = True
            else:
                counts["S"] += 1
        elif event == "duplication":
            if alerax_style and has_loss_child:
                counts["DL"] += 1
                loss_children_are_composite = True
            else:
                counts["D"] += 1
        elif event == "branchingOut":
            if alerax_style and has_loss_child:
                counts["TL"] += 1
                loss_children_are_composite = True
            else:
                counts["T"] += 1
        elif event == "loss":
            if not suppress_loss:
                counts["L"] += 1
        elif event == "leaf":
            counts["Leaf"] += 1

        for child in children:
            visit(
                child,
                suppress_loss=loss_children_are_composite and event_name(child) == "loss",
            )

    for rec_gene_tree in root.iter():
        if local_name(rec_gene_tree.tag) != "recGeneTree":
            continue
        phylogeny = next(
            (child for child in rec_gene_tree if local_name(child.tag) == "phylogeny"),
            None,
        )
        if phylogeny is None:
            continue
        clade = next((child for child in phylogeny if local_name(child.tag) == "clade"), None)
        if clade is not None:
            visit(clade)
    return counts
