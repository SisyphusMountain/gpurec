"""Rust stochastic backtracking bridge for gpurec models."""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from gpurec.api._validation import integer_value
from gpurec.api.model import FamilyInput, GeneReconModel, ReconciliationState
from gpurec.recphyloxml import (
    EVENT_KEYS,
    direct_children,
    iter_rec_gene_tree_clades,
    primary_event_name,
)

__all__ = [
    "EVENT_KEYS",
    "ensure_backtracking_available",
    "export_backtracking_input",
    "recphyloxml_event_counts",
    "sample_recphyloxml",
    "sample_recphyloxmls",
]


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKTRACK_MANIFEST = _REPO_ROOT / "crates" / "gpurec-backtrack" / "Cargo.toml"
_BACKTRACK_BINARY_ENV = "GPUREC_BACKTRACK_BIN"
_BACKTRACK_BINARY_GUIDANCE = (
    f"Set {_BACKTRACK_BINARY_ENV}, pass backtrack_binary in Python, or use "
    "--backtrack-binary in the CLI"
)
_BACKTRACK_HELP_TIMEOUT_SECONDS = 30
_BACKTRACK_HELP_MARKERS = (
    "usage: gpurec-backtrack",
    "--samples",
    "input.json",
)
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class _BacktrackInvocation:
    command: list[str]
    cwd: str | None


def _optional_int_limit(
    name: str,
    value: int | None,
    *,
    minimum: int,
    maximum: int | None = None,
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
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _validate_backtracking_limits(
    *,
    seed: int | None,
    max_events: int | None,
) -> tuple[int | None, int | None]:
    return (
        _optional_int_limit("seed", seed, minimum=0, maximum=_UINT64_MAX),
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
    source = "backtrack_binary/--backtrack-binary"
    if backtrack_binary is None:
        backtrack_binary = os.environ.get(_BACKTRACK_BINARY_ENV)
        source = _BACKTRACK_BINARY_ENV
    if backtrack_binary is not None:
        text = os.fspath(backtrack_binary)
        path = Path(text).expanduser()
        has_separator = any(sep and sep in text for sep in (os.sep, os.altsep))
        if (
            not isinstance(backtrack_binary, str)
            or path.is_absolute()
            or path.parent != Path(".")
            or has_separator
        ):
            resolved = path.resolve()
            if not resolved.is_file():
                raise RuntimeError(
                    f"gpurec backtracking binary from {source} does not exist "
                    f"or is not a file: {resolved}"
                )
            if not os.access(resolved, os.X_OK):
                raise RuntimeError(
                    f"gpurec backtracking binary from {source} is not executable: "
                    f"{resolved}"
                )
            return [str(resolved)]
        if shutil.which(text) is None:
            raise RuntimeError(
                f"gpurec backtracking binary {text!r} from {source} was not found "
                "on PATH"
            )
        return [text]

    manifest = Path(cargo_manifest)
    if not manifest.exists():
        raise RuntimeError(
            "gpurec stochastic backtracking needs a Rust backtracking binary. "
            f"{_BACKTRACK_BINARY_GUIDANCE}; default source manifest not found "
            f"at {manifest}"
        )
    if shutil.which("cargo") is None:
        raise RuntimeError(
            "gpurec stochastic backtracking fallback requires cargo on PATH. "
            f"Install Rust/Cargo, or {_BACKTRACK_BINARY_GUIDANCE}."
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


def _is_cargo_fallback_command(command: list[str]) -> bool:
    return (
        len(command) >= 7
        and command[0] == "cargo"
        and command[1] == "run"
        and "--manifest-path" in command
        and command[-1] == "--"
    )


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _backtrack_invocation(
    *,
    cargo_manifest: str | Path,
    backtrack_binary: str | Path | None,
) -> _BacktrackInvocation:
    command = _backtrack_command(
        cargo_manifest=cargo_manifest,
        backtrack_binary=backtrack_binary,
    )
    cwd = str(_REPO_ROOT) if _is_cargo_fallback_command(command) else None
    return _BacktrackInvocation(command=command, cwd=cwd)


def _validate_backtracking_help(invocation: _BacktrackInvocation) -> None:
    command = invocation.command + ["--help"]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            cwd=invocation.cwd,
            text=True,
            timeout=_BACKTRACK_HELP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "gpurec backtracking command timed out while validating --help: "
            f"{_command_text(command)}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "gpurec backtracking command could not be executed while validating "
            f"--help: {_command_text(command)}"
        ) from exc

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        details = [
            "gpurec backtracking command failed while validating --help",
            f"exit code {result.returncode}",
            f"command: {_command_text(command)}",
        ]
        if result.stderr:
            details.append(f"stderr: {result.stderr.strip()}")
        if result.stdout:
            details.append(f"stdout: {result.stdout.strip()}")
        raise RuntimeError("; ".join(details))

    missing = [marker for marker in _BACKTRACK_HELP_MARKERS if marker not in output]
    if missing:
        raise RuntimeError(
            "gpurec backtracking command did not report the expected "
            f"gpurec-backtrack help text: {_command_text(command)}"
        )


def ensure_backtracking_available(
    backtrack_binary: str | Path | None = None,
    *,
    cargo_manifest: str | Path = _BACKTRACK_MANIFEST,
) -> None:
    """Validate that the Rust backtracking command resolves and runs."""

    invocation = _backtrack_invocation(
        cargo_manifest=cargo_manifest,
        backtrack_binary=backtrack_binary,
    )
    _validate_backtracking_help(invocation)


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
        invocation = _backtrack_invocation(
            cargo_manifest=cargo_manifest,
            backtrack_binary=backtrack_binary,
        )
        full_command = invocation.command + build_args(input_path, tmp_path)
        result = subprocess.run(
            full_command,
            check=False,
            capture_output=True,
            cwd=invocation.cwd,
            text=True,
        )
        if result.returncode != 0:
            details = [
                "gpurec backtracking command failed",
                f"exit code {result.returncode}",
                f"command: {_command_text(full_command)}",
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
    family_index = integer_value("family_index", family_index)
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
    species_newick = model.species_tree_path.read_text(encoding="utf-8")
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
    base_seed = _optional_int_limit("seed", seed, minimum=0, maximum=_UINT64_MAX)
    if base_seed is None:
        base_seed = 0
    max_seed = base_seed + num_samples - 1
    if max_seed > _UINT64_MAX:
        raise ValueError(
            "seed range exceeds u64 maximum: "
            f"seed={base_seed}, num_samples={num_samples}, max_seed={max_seed}"
        )
    payload = export_backtracking_input(
        model,
        family_index=family_index,
        seed=base_seed,
        max_events=max_events,
    )

    def build_args(input_path: Path, tmp_path: Path) -> list[str]:
        return [
            "--samples",
            str(num_samples),
            "--seed",
            str(base_seed),
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

    counts = {key: 0 for key in EVENT_KEYS}

    def visit(clade, *, suppress_loss: bool = False) -> None:
        event = primary_event_name(clade)
        children = direct_children(clade, "clade")
        child_events = [primary_event_name(child) for child in children]
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
                suppress_loss=(
                    loss_children_are_composite
                    and primary_event_name(child) == "loss"
                ),
            )

    for clade in iter_rec_gene_tree_clades(xml):
        visit(clade)
    return counts
