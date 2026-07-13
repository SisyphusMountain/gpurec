#!/usr/bin/env python3
"""Locate fp32 forward-error sources on a resident HOGENOM subset.

The diagnostic runs the same input through absolute fp64, absolute fp32, and
centered fp32 solves.  It observes existing Python launch boundaries without
changing production kernels, copying mutable outputs before their ping-pong
buffers are reused.  Centered residuals are reconstructed in fp64 before they
are matched to absolute tensors.

The default output is a compact JSON summary under the ignored ``output/``
directory.  Pass ``--tensor-output`` to additionally retain the captured CPU
tensors in a ``torch.save`` file; a one-family, 64-Pi-iteration capture can be
several hundred MiB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from gpurec import GeneReconModel, SolverOptions
import gpurec.core.inference.forward as forward_module
import gpurec.core.inference.solver as solver_module
import gpurec.core.kernels.e_step as e_step_module
from gpurec.core.inference.logspace import log2_survival, logsumexp2, survival_from_E
from gpurec.core.parameters.extract_parameters import (
    as_family_species,
    origination_log_probs_from_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_HOGENOM_ROOT_ENV = os.environ.get("GPUREC_HOGENOM_ROOT")
DEFAULT_HOGENOM_ROOT = Path(_HOGENOM_ROOT_ENV) if _HOGENOM_ROOT_ENV else None
DEFAULT_FAMILIES_FILE = REPO_ROOT / "experiments/sanderson_cv/families_1055.txt"
DEFAULT_OUTPUT = REPO_ROOT / "output/forward_precision_diagnostic.json"


def _csv_floats(value: str) -> list[float]:
    result = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hogenom-root", type=Path, default=DEFAULT_HOGENOM_ROOT)
    parser.add_argument("--families-file", type=Path, default=DEFAULT_FAMILIES_FILE)
    parser.add_argument("--families", type=int, default=1)
    parser.add_argument("--mode", choices=("global", "specieswise", "genewise"), default="global")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument(
        "--theta",
        type=_csv_floats,
        default=_csv_floats("-1.75,-0.4,0.65"),
        help="three DTL logits, broadcast over species/families when needed",
    )
    parser.add_argument("--e-max-iter", type=int, default=128)
    parser.add_argument("--e-tol", type=float, default=1e-8)
    parser.add_argument("--pi-iters", type=int, default=64)
    parser.add_argument("--family-chunk-size", type=int, default=300)
    parser.add_argument("--clade-budget", type=int, default=315_000)
    parser.add_argument("--max-wave-size", type=int, default=8192)
    parser.add_argument("--top-stages", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tensor-output",
        type=Path,
        help="optional torch.save path for all captured CPU tensors",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.hogenom_root is None:
        parser.error("set GPUREC_HOGENOM_ROOT or pass --hogenom-root")
    if args.families < 1:
        parser.error("--families must be at least 1")
    if len(args.theta) != 3:
        parser.error("--theta must contain exactly three DTL logits")
    if args.e_max_iter < 1 or args.e_tol <= 0.0:
        parser.error("E controls require --e-max-iter >= 1 and --e-tol > 0")
    if args.pi_iters < 2 or args.pi_iters % 2:
        parser.error("--pi-iters must be an even integer at least 2")
    if args.top_stages < 1:
        parser.error("--top-stages must be positive")
    return args


def _git(command: list[str]) -> str:
    completed = subprocess.run(
        ["git", *command], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip()


def environment_record() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "git_revision": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
        },
    }


def dataset_paths(args: argparse.Namespace) -> tuple[Path, list[Path], list[str]]:
    species_tree = args.hogenom_root / (
        "runs/MFP/true_start_ufboot1000/"
        "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/"
        "species_trees/starting_species_tree.newick"
    )
    family_names = [
        line.strip()
        for line in args.families_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.families]
    gene_trees = [
        args.hogenom_root / "families" / name / "gene_trees/ufboot1000.MFP.geneTree.newick"
        for name in family_names
    ]
    missing = [path for path in (species_tree, *gene_trees) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing HOGENOM inputs:\n" + "\n".join(map(str, missing[:10])))
    return species_tree, gene_trees, family_names


def build_model(
    args: argparse.Namespace,
    species_tree: Path,
    gene_trees: list[Path],
    *,
    dtype: torch.dtype,
    representation: str,
) -> GeneReconModel:
    model = GeneReconModel(
        species_tree,
        gene_trees,
        mode=args.mode,
        device="cuda",
        dtype=dtype,
        family_chunk_size=args.family_chunk_size,
        clade_budget=args.clade_budget,
        max_wave_size=args.max_wave_size,
        solver_options=SolverOptions(
            e_max_iter=args.e_max_iter,
            e_tol=args.e_tol,
            pi_iters=args.pi_iters,
            pi_representation=representation,
        ),
    )
    if len(model.batch_statics) != 1:
        raise ValueError(
            "forward precision capture requires one resident batch; increase the clade budget "
            "or reduce --families"
        )
    with torch.no_grad():
        theta_row = torch.tensor(args.theta, dtype=dtype, device="cuda")
        model.theta.copy_(theta_row.expand_as(model.theta))
        model.receiver_weights.zero_()
        model.origination_weights.zero_()
        if args.weighted:
            model.receiver_weights.copy_(
                torch.linspace(
                    -1.25, 1.75, model.receiver_weights.numel(), dtype=dtype, device="cuda"
                ).reshape_as(model.receiver_weights)
            )
            model.origination_weights.copy_(
                torch.linspace(
                    0.9, -1.1, model.origination_weights.numel(), dtype=dtype, device="cuda"
                ).reshape_as(model.origination_weights)
            )
    return model


def _cpu_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().to(device="cpu", copy=True)


class CaptureRecorder:
    """Own captured tensors and wave-local iteration bookkeeping for one solve."""

    def __init__(self, static, *, variant: str, centered: bool):
        self.variant = variant
        self.centered = centered
        self.tensors: dict[str, torch.Tensor] = {}
        self.tensor_metadata: dict[str, dict[str, Any]] = {}
        self.e_iteration = 0
        self.wave_by_start = {
            int(meta["start"]): (index, int(meta["W"]))
            for index, meta in enumerate(static.wave_layout["wave_metas"])
        }
        self.next_wave_iteration: dict[int, int] = {}

    def record(self, key: str, value: torch.Tensor, *, frame: str = "native") -> None:
        if key in self.tensors:
            raise RuntimeError(f"duplicate capture key: {key}")
        captured = _cpu_copy(value)
        self.tensors[key] = captured
        self.tensor_metadata[key] = {
            "shape": list(captured.shape),
            "dtype": str(captured.dtype).removeprefix("torch."),
            "frame": frame,
        }

    def record_centered(
        self,
        key: str,
        residual: torch.Tensor,
        offset: torch.Tensor,
    ) -> None:
        reconstructed = residual.double() + offset.double().reshape(-1, 1)
        self.record(key, reconstructed, frame="centered_reconstructed")

    def wave(self, ws: int) -> tuple[int, int]:
        try:
            return self.wave_by_start[int(ws)]
        except KeyError as exc:
            raise RuntimeError(f"unknown wave start {ws}") from exc

    @staticmethod
    def wave_key(wave_index: int, suffix: str) -> str:
        return f"wave/{wave_index:03d}/{suffix}"


@contextmanager
def install_capture(recorder: CaptureRecorder) -> Iterator[None]:
    """Temporarily observe launch wrappers used by the solver's imported functions."""

    originals = {
        "e_launch": e_step_module._launch_e_step_forward_2d,
        "extract_uniform": solver_module.extract_parameters_uniform,
        "extract_weighted": solver_module.extract_parameters_weighted_receivers,
        "dts": forward_module.compute_dts_forward,
        "leaf": forward_module.compute_leaf_initial_wave_step,
        "wave": forward_module.compute_wave_step,
        "dts_centered": forward_module.compute_dts_forward_centered,
        "leaf_centered": forward_module.compute_leaf_initial_wave_step_centered,
        "wave_centered": forward_module.compute_wave_step_centered,
    }

    def e_launch(*args, **kwargs):
        result = originals["e_launch"](*args, **kwargs)
        max_diff = kwargs.get("max_diff_out")
        if max_diff is None:
            prefix = "e/final_aux"
        else:
            prefix = f"e/iteration/{recorder.e_iteration:03d}"
            recorder.e_iteration += 1
        recorder.record(f"{prefix}/E_in", args[0])
        for name, tensor in zip(("E_new", "E_s1", "E_s2", "Ebar"), result):
            recorder.record(f"{prefix}/{name}", tensor)
        if max_diff is not None:
            recorder.record(f"{prefix}/max_diff", max_diff)
        return result

    def _record_parameters(result) -> None:
        names = ("log_pS", "log_pD", "log_pL", "max_transfer", "receiver_log_probs")
        for name, tensor in zip(names, result):
            recorder.record(f"parameter/{name}", tensor)

    def extract_uniform(*args, **kwargs):
        result = originals["extract_uniform"](*args, **kwargs)
        _record_parameters(result)
        return result

    def extract_weighted(*args, **kwargs):
        result = originals["extract_weighted"](*args, **kwargs)
        _record_parameters(result)
        return result

    def dts(*args, **kwargs):
        result = originals["dts"](*args, **kwargs)
        ws = int(kwargs.get("family_offset", 0))
        wave_index, _ = recorder.wave(ws)
        recorder.next_wave_iteration[ws] = 1
        recorder.record(recorder.wave_key(wave_index, "dts"), result)
        log_split_probs = kwargs.get("log_split_probs")
        if log_split_probs is not None:
            recorder.record(recorder.wave_key(wave_index, "log_split_probs_raw"), log_split_probs)
            recorder.record(
                recorder.wave_key(wave_index, "log_split_probs_kernel"),
                log_split_probs.reshape(-1).to(dtype=args[0].dtype),
            )
        return result

    def dts_centered(*args, **kwargs):
        result = originals["dts_centered"](*args, **kwargs)
        ws = int(kwargs.get("family_offset", 0))
        wave_index, _ = recorder.wave(ws)
        recorder.next_wave_iteration[ws] = 1
        recorder.record_centered(recorder.wave_key(wave_index, "dts"), result[0], result[1])
        log_split_probs = kwargs.get("log_split_probs")
        if log_split_probs is not None:
            recorder.record(recorder.wave_key(wave_index, "log_split_probs_raw"), log_split_probs)
            recorder.record(
                recorder.wave_key(wave_index, "log_split_probs_kernel"),
                log_split_probs.reshape(-1).to(dtype=args[0].dtype),
            )
        return result

    def leaf(*args, **kwargs):
        result = originals["leaf"](*args, **kwargs)
        ws, width = int(args[1]), int(args[2])
        wave_index, expected_width = recorder.wave(ws)
        if width != expected_width:
            raise RuntimeError("leaf wrapper width does not match wave metadata")
        recorder.next_wave_iteration[ws] = 1
        recorder.record(
            recorder.wave_key(wave_index, "iteration/000/pi"), args[0].narrow(0, ws, width)
        )
        return result

    def leaf_centered(*args, **kwargs):
        result = originals["leaf_centered"](*args, **kwargs)
        ws, width = int(args[2]), int(args[3])
        wave_index, expected_width = recorder.wave(ws)
        if width != expected_width:
            raise RuntimeError("centered leaf wrapper width does not match wave metadata")
        recorder.next_wave_iteration[ws] = 1
        recorder.record_centered(
            recorder.wave_key(wave_index, "iteration/000/pi"),
            args[0].narrow(0, ws, width),
            args[1].narrow(0, ws, width),
        )
        return result

    def wave(*args, **kwargs):
        result = originals["wave"](*args, **kwargs)
        ws, width = int(args[3]), int(args[4])
        wave_index, expected_width = recorder.wave(ws)
        if width != expected_width:
            raise RuntimeError("wave wrapper width does not match wave metadata")
        local_iter = recorder.next_wave_iteration.get(ws, 1)
        recorder.next_wave_iteration[ws] = local_iter + 1
        prefix = recorder.wave_key(wave_index, f"iteration/{local_iter:03d}")
        recorder.record(f"{prefix}/pi", args[1].narrow(0, ws, width))
        if kwargs.get("input_ws") is not None:
            recorder.record(f"{prefix}/dts_consumed", args[17])
        if kwargs.get("store_final_pibar", False):
            recorder.record(f"{prefix}/pibar", args[2].narrow(0, ws, width))
            row_max = kwargs["pibar_row_max"]
            recorder.record(f"{prefix}/pibar_row_max", row_max.narrow(0, ws, width))
        return result

    def wave_centered(*args, **kwargs):
        result = originals["wave_centered"](*args, **kwargs)
        ws, width = int(args[6]), int(args[7])
        wave_index, expected_width = recorder.wave(ws)
        if width != expected_width:
            raise RuntimeError("centered wave wrapper width does not match wave metadata")
        local_iter = recorder.next_wave_iteration.get(ws, 1)
        recorder.next_wave_iteration[ws] = local_iter + 1
        prefix = recorder.wave_key(wave_index, f"iteration/{local_iter:03d}")
        recorder.record_centered(
            f"{prefix}/pi",
            args[2].narrow(0, ws, width),
            args[3].narrow(0, ws, width),
        )
        if kwargs.get("input_ws") is not None:
            dts_key = f"{prefix}/dts_consumed"
            recorder.record_centered(dts_key, args[20], args[21])
            recorder.record(
                f"{dts_key}_residual",
                args[20],
                frame="centered_dts_storage_residual",
            )
            recorder.record(
                f"{dts_key}_virtual_center_offset",
                args[22],
                frame="centered_dts_virtual_consumer_offset",
            )
        if kwargs.get("store_final_pibar", False):
            recorder.record_centered(
                f"{prefix}/pibar",
                args[4].narrow(0, ws, width),
                args[5].narrow(0, ws, width),
            )
            row_max = kwargs["pibar_row_max"].narrow(0, ws, width).double()
            pi_offset = args[3].narrow(0, ws, width).double()
            recorder.record(
                f"{prefix}/pibar_row_max",
                row_max + pi_offset,
                frame="centered_reconstructed",
            )
        return result

    e_step_module._launch_e_step_forward_2d = e_launch
    solver_module.extract_parameters_uniform = extract_uniform
    solver_module.extract_parameters_weighted_receivers = extract_weighted
    forward_module.compute_dts_forward = dts
    forward_module.compute_leaf_initial_wave_step = leaf
    forward_module.compute_wave_step = wave
    forward_module.compute_dts_forward_centered = dts_centered
    forward_module.compute_leaf_initial_wave_step_centered = leaf_centered
    forward_module.compute_wave_step_centered = wave_centered
    try:
        yield
    finally:
        e_step_module._launch_e_step_forward_2d = originals["e_launch"]
        solver_module.extract_parameters_uniform = originals["extract_uniform"]
        solver_module.extract_parameters_weighted_receivers = originals["extract_weighted"]
        forward_module.compute_dts_forward = originals["dts"]
        forward_module.compute_leaf_initial_wave_step = originals["leaf"]
        forward_module.compute_wave_step = originals["wave"]
        forward_module.compute_dts_forward_centered = originals["dts_centered"]
        forward_module.compute_leaf_initial_wave_step_centered = originals["leaf_centered"]
        forward_module.compute_wave_step_centered = originals["wave_centered"]


def head_components(
    root_rows: torch.Tensor,
    E: torch.Tensor,
    origination_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the exact NLL head split into inspectable reduction components."""

    uniform = solver_module.origination_weights_are_uniform(origination_weights)
    if uniform:
        origination_log_probs = None
        origination_probs = None
        root_input = root_rows
    else:
        origination_log_probs = origination_log_probs_from_weights(
            origination_weights.to(device=root_rows.device, dtype=origination_weights.dtype)
        )
        origination_probs = torch.exp2(origination_log_probs)
        root_input = root_rows + origination_log_probs

    root_row_max = root_input.max(dim=-1, keepdim=True).values
    root_row_max_safe = torch.where(
        torch.isneginf(root_row_max), torch.zeros_like(root_row_max), root_row_max
    )
    root_exp_terms = torch.exp2(root_input - root_row_max_safe)
    root_sum = root_exp_terms.sum(dim=-1)
    root_logsumexp = logsumexp2(root_input, dim=-1)
    log_numerator = (
        root_logsumexp - math.log2(root_rows.shape[-1])
        if uniform
        else root_logsumexp
    )
    survival_terms = -torch.expm1(E * math.log(2.0))
    survival = survival_from_E(E, origination_probs)
    log_survival = log2_survival(E, origination_probs)
    nll_vector = -(log_numerator - log_survival)
    result = {
        "root_input": root_input,
        "root_row_max": root_row_max,
        "root_exp_terms": root_exp_terms,
        "root_sum": root_sum,
        "root_logsumexp": root_logsumexp,
        "log_numerator": log_numerator,
        "survival_terms": survival_terms,
        "survival": survival,
        "log_survival": log_survival,
        "nll_vector": nll_vector,
        "loss": nll_vector.sum(),
    }
    if origination_log_probs is not None and origination_probs is not None:
        result["origination_log_probs"] = origination_log_probs
        result["origination_probs"] = origination_probs
    return result


def record_terminal(
    recorder: CaptureRecorder,
    static,
    solved: tuple[torch.Tensor, ...],
    origination_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    (
        E,
        E_s1,
        E_s2,
        Ebar,
        root_rows,
        pi,
        pibar,
        pibar_row_max,
        log_pS,
        log_pD,
        log_pL,
        max_transfer,
        receiver_log_probs,
    ) = solved
    centered_state = getattr(static, "centered_pi_forward_state", None)
    if centered_state is None:
        pi_absolute = pi
        pibar_absolute = pibar
        pibar_row_max_absolute = pibar_row_max
    else:
        pi_absolute = centered_state.reconstruct_pi(pi)
        pibar_absolute = centered_state.reconstruct_pibar(pibar)
        pibar_row_max_absolute = centered_state.reconstruct_pibar_row_max(pibar_row_max)

    terminal = {
        "E": E,
        "E_s1": E_s1,
        "E_s2": E_s2,
        "Ebar": Ebar,
        "root_rows": root_rows,
        "pi": pi_absolute,
        "pibar": pibar_absolute,
        "pibar_row_max": pibar_row_max_absolute,
        "log_pS": log_pS,
        "log_pD": log_pD,
        "log_pL": log_pL,
        "max_transfer": max_transfer,
        "receiver_log_probs": receiver_log_probs,
    }
    for name, tensor in terminal.items():
        frame = "centered_reconstructed" if centered_state is not None and name in {
            "root_rows", "pi", "pibar", "pibar_row_max"
        } else "native"
        recorder.record(f"terminal/{name}", tensor, frame=frame)

    family_rows = int(E.shape[0])
    species = int(E.shape[1])
    log_pD_family = as_family_species(log_pD, species, family_rows)
    log_pS_family = as_family_species(log_pS, species, family_rows)
    recorder.record("parameter/dl_const", 1.0 + log_pD_family + E)
    recorder.record("parameter/sl1_const", log_pS_family + E_s2)
    recorder.record("parameter/sl2_const", log_pS_family + E_s1)
    # The uniform path constructs this after parameter extraction, so ensure it
    # is present under the same key as the weighted extraction result.
    if "parameter/receiver_log_probs" not in recorder.tensors:
        recorder.record("parameter/receiver_log_probs", receiver_log_probs)

    # Centered root reconstruction is fp64 even for an fp32 solve. Cast every
    # input back to the configured state dtype for a genuine working-dtype head
    # control rather than an accidental mixed-precision result.
    working_head = head_components(
        root_rows.to(dtype=E.dtype),
        E,
        origination_weights.to(device=E.device, dtype=E.dtype),
    )
    for name, tensor in working_head.items():
        recorder.record(f"head/{name}", tensor)
    if solver_module.origination_weights_are_uniform(origination_weights):
        production_loss = solver_module.nll_from_root_rows(root_rows, E)
    else:
        # Production promotes raw nonuniform origination weights before the
        # fp64 likelihood-head softmax; keep this distinct from the explicit
        # working-dtype head captured above.
        o_lp = origination_log_probs_from_weights(origination_weights.double())
        production_loss = solver_module.nll_from_root_rows(
            root_rows,
            E,
            origination_log_probs=o_lp,
            origination_probs=torch.exp2(o_lp),
        )
    return {name: _cpu_copy(tensor) for name, tensor in terminal.items()} | {
        "origination_weights": _cpu_copy(origination_weights),
        "working_dtype_head_loss": _cpu_copy(working_head["loss"]),
        "production_loss": _cpu_copy(production_loss),
    }


@dataclass
class VariantRun:
    name: str
    representation: str
    dtype: torch.dtype
    recorder: CaptureRecorder
    terminal: dict[str, torch.Tensor]
    elapsed_s: float
    species: int
    clades: int
    waves: int


def run_variant(
    args: argparse.Namespace,
    species_tree: Path,
    gene_trees: list[Path],
    *,
    name: str,
    dtype: torch.dtype,
    representation: str,
) -> VariantRun:
    model = build_model(
        args, species_tree, gene_trees, dtype=dtype, representation=representation
    )
    static = model.batch_statics[0]
    static.warm_E = None
    recorder = CaptureRecorder(static, variant=name, centered=representation == "centered")
    theta = model._theta_for_static(static, model.theta)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), install_capture(recorder):
        solved = solver_module.solve_resident_e_pi(
            static, theta, model.receiver_weights, warm_start_E=None
        )
        terminal = record_terminal(recorder, static, solved, model.origination_weights)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    run = VariantRun(
        name=name,
        representation=representation,
        dtype=dtype,
        recorder=recorder,
        terminal=terminal,
        elapsed_s=elapsed_s,
        species=int(static.species_helpers["S"]),
        clades=int(static.wave_layout["leaf_species_index"].numel()),
        waves=len(static.wave_layout["wave_metas"]),
    )
    del model, solved
    torch.cuda.empty_cache()
    return run


def _finite_quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q).item()) if values.numel() else 0.0


def tensor_error_summary(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual64 = actual.detach().double()
    reference64 = reference.detach().double()
    result: dict[str, Any] = {
        "actual_shape": list(actual64.shape),
        "reference_shape": list(reference64.shape),
    }
    if actual64.shape != reference64.shape:
        result["shape_mismatch"] = True
        return result

    # E0 uses the working dtype's finite minimum as its negative-infinity
    # sentinel.  Comparing fp32/fp64 sentinel magnitudes would manufacture an
    # error of ~1e308 even though their logical state is identical.
    actual_sentinel = (
        actual == torch.finfo(actual.dtype).min
        if actual.dtype.is_floating_point
        else torch.zeros_like(actual, dtype=torch.bool)
    )
    reference_sentinel = (
        reference == torch.finfo(reference.dtype).min
        if reference.dtype.is_floating_point
        else torch.zeros_like(reference, dtype=torch.bool)
    )
    actual_finite = torch.isfinite(actual64) & ~actual_sentinel
    reference_finite = torch.isfinite(reference64) & ~reference_sentinel
    actual_neginf = torch.isneginf(actual64)
    reference_neginf = torch.isneginf(reference64)
    common = actual_finite & reference_finite
    result.update(
        finite_count_actual=int(actual_finite.sum().item()),
        finite_count_reference=int(reference_finite.sum().item()),
        finite_mask_mismatch_count=int((actual_finite != reference_finite).sum().item()),
        minimum_sentinel_mismatch_count=int(
            (actual_sentinel != reference_sentinel).sum().item()
        ),
        neginf_mask_mismatch_count=int((actual_neginf != reference_neginf).sum().item()),
        nan_count_actual=int(torch.isnan(actual64).sum().item()),
        nan_count_reference=int(torch.isnan(reference64).sum().item()),
    )
    if not bool(common.any()):
        result["common_finite_count"] = 0
        return result

    delta = actual64[common] - reference64[common]
    absolute = delta.abs()
    reference_values = reference64[common]
    ref_norm = torch.linalg.vector_norm(reference_values).clamp_min(1e-300)
    worst_flat = int(absolute.argmax().item())
    result.update(
        common_finite_count=int(common.sum().item()),
        abs_max=float(absolute.max().item()),
        abs_mean=float(absolute.mean().item()),
        abs_rms=float(torch.sqrt(torch.mean(delta.square())).item()),
        abs_p50=_finite_quantile(absolute, 0.50),
        abs_p95=_finite_quantile(absolute, 0.95),
        abs_p99=_finite_quantile(absolute, 0.99),
        abs_l2=float(torch.linalg.vector_norm(delta).item()),
        rel_l2=float((torch.linalg.vector_norm(delta) / ref_norm).item()),
        worst_common_flat_index=worst_flat,
        worst_actual=float(actual64[common][worst_flat].item()),
        worst_reference=float(reference_values[worst_flat].item()),
    )
    rounded_reference = reference_values.float().double()
    result["abs_max_vs_reference_rounded_fp32"] = float(
        (actual64[common] - rounded_reference).abs().max().item()
    )

    if actual64.ndim == 2:
        delta_full = torch.where(common, actual64 - reference64, torch.nan)
        row_counts = common.sum(dim=1)
        valid_rows = row_counts > 0
        if bool(valid_rows.any()):
            row_shift = torch.nansum(delta_full, dim=1) / row_counts.clamp_min(1)
            centered_delta = delta_full - row_shift[:, None]
            result["row_shift_abs_max"] = float(row_shift[valid_rows].abs().max().item())
            result["within_row_abs_max"] = float(
                torch.nan_to_num(centered_delta.abs(), nan=0.0).max().item()
            )
    return result


def compare_traces(
    actual: CaptureRecorder,
    reference: CaptureRecorder,
    *,
    top_stages: int,
) -> dict[str, Any]:
    shared = sorted(set(actual.tensors) & set(reference.tensors))
    missing = sorted(set(reference.tensors) - set(actual.tensors))
    extra = sorted(set(actual.tensors) - set(reference.tensors))
    stages = {}
    for key in shared:
        if key == "e/iteration/000/max_diff":
            stages[key] = {
                "comparison_skipped": (
                    "the first update is measured from each dtype's finite-minimum E0 sentinel"
                )
            }
        else:
            stages[key] = tensor_error_summary(actual.tensors[key], reference.tensors[key])
    ranked = sorted(
        (
            {"key": key, "abs_max": float(summary["abs_max"]), "rel_l2": float(summary["rel_l2"])}
            for key, summary in stages.items()
            if "abs_max" in summary
        ),
        key=lambda row: row["abs_max"],
        reverse=True,
    )[:top_stages]
    return {
        "reference": reference.variant,
        "actual": actual.variant,
        "shared_tensor_count": len(shared),
        "missing_keys": missing,
        "extra_keys": extra,
        "top_stages_by_abs_max": ranked,
        "stages": stages,
    }


def _head_loss(
    root_rows: torch.Tensor,
    E: torch.Tensor,
    origination_weights: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> float:
    components = head_components(
        root_rows.to(dtype=dtype, device="cuda"),
        E.to(dtype=dtype, device="cuda"),
        origination_weights.to(dtype=dtype, device="cuda"),
    )
    return float(components["loss"].detach().cpu())


def head_hybrid_controls(reference: VariantRun, candidates: list[VariantRun]) -> dict[str, Any]:
    ref_root = reference.terminal["root_rows"]
    ref_E = reference.terminal["E"]
    ref_weights = reference.terminal["origination_weights"]
    reference_loss = _head_loss(ref_root, ref_E, ref_weights, dtype=torch.float64)
    result: dict[str, Any] = {
        "absolute_fp64_solver_fp64_head": reference_loss,
        "absolute_fp64_solver_fp32_head": _head_loss(
            ref_root, ref_E, ref_weights, dtype=torch.float32
        ),
    }
    for candidate in candidates:
        root = candidate.terminal["root_rows"]
        extinction = candidate.terminal["E"]
        weights = candidate.terminal["origination_weights"]
        prefix = candidate.name
        result[f"{prefix}_working_dtype_head"] = float(
            candidate.terminal["working_dtype_head_loss"].item()
        )
        result[f"{prefix}_production_head"] = float(
            candidate.terminal["production_loss"].item()
        )
        result[f"{prefix}_solver_fp64_head"] = _head_loss(
            root, extinction, weights, dtype=torch.float64
        )
        result[f"{prefix}_root_with_reference_E_fp64_head"] = _head_loss(
            root, ref_E, ref_weights, dtype=torch.float64
        )
        result[f"reference_root_with_{prefix}_E_fp64_head"] = _head_loss(
            ref_root, extinction, ref_weights, dtype=torch.float64
        )
    result["errors_vs_absolute_fp64_solver_fp64_head"] = {
        name: abs(float(value) - reference_loss)
        for name, value in result.items()
        if isinstance(value, float) and name != "absolute_fp64_solver_fp64_head"
    }
    return result


def variant_record(run: VariantRun) -> dict[str, Any]:
    return {
        "name": run.name,
        "representation": run.representation,
        "dtype": str(run.dtype).removeprefix("torch."),
        "elapsed_s_including_synchronous_capture": run.elapsed_s,
        "species": run.species,
        "clades": run.clades,
        "waves": run.waves,
        "e_iterations": run.recorder.e_iteration,
        "captured_tensor_count": len(run.recorder.tensors),
        "working_dtype_head_loss": float(
            run.terminal["working_dtype_head_loss"].item()
        ),
        "production_loss": float(run.terminal["production_loss"].item()),
        "tensor_metadata": run.recorder.tensor_metadata,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("forward precision capture requires CUDA")
    species_tree, gene_trees, family_names = dataset_paths(args)

    specifications = (
        ("absolute_fp64", torch.float64, "absolute"),
        ("absolute_fp32", torch.float32, "absolute"),
        ("centered_fp32", torch.float32, "centered"),
    )
    runs: list[VariantRun] = []
    for name, dtype, representation in specifications:
        print(json.dumps({"event": "variant_start", "name": name}), flush=True)
        run = run_variant(
            args,
            species_tree,
            gene_trees,
            name=name,
            dtype=dtype,
            representation=representation,
        )
        runs.append(run)
        print(
            json.dumps(
                {
                    "event": "variant_done",
                    "name": name,
                    "loss": float(run.terminal["production_loss"].item()),
                    "captures": len(run.recorder.tensors),
                }
            ),
            flush=True,
        )

    reference = runs[0]
    comparisons = {
        run.name: compare_traces(run.recorder, reference.recorder, top_stages=args.top_stages)
        for run in runs[1:]
    }
    output = {
        "benchmark": "forward_precision_diagnostic",
        "environment": environment_record(),
        "dataset": {
            "hogenom_root": str(args.hogenom_root),
            "species_tree": str(species_tree),
            "families_file": str(args.families_file),
            "family_names": family_names,
        },
        "method": {
            "mode": args.mode,
            "weighted_receiver_and_origination": bool(args.weighted),
            "theta": args.theta,
            "e_max_iter": args.e_max_iter,
            "e_tol": args.e_tol,
            "pi_iters": args.pi_iters,
            "capture": "synchronous CPU copies immediately after existing Python launch wrappers",
            "centered_matching": "fp64 reconstruction of residual + row offset",
            "timing_warning": "capture timings are diagnostic and not performance measurements",
        },
        "variants": [variant_record(run) for run in runs],
        "comparisons": comparisons,
        "head_hybrid_controls": head_hybrid_controls(reference, runs[1:]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    if args.tensor_output is not None:
        args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "gpurec_forward_precision_capture_v1",
                "summary_output": str(args.output),
                "variants": {run.name: run.recorder.tensors for run in runs},
            },
            args.tensor_output,
        )
    print(
        json.dumps(
            {
                "event": "done",
                "output": str(args.output),
                "tensor_output": None if args.tensor_output is None else str(args.tensor_output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
