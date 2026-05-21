"""Chunked PyTorch interface for global/uniform reconciliation.

This module exposes the optimized uniform forward/backward pipeline as a
``torch.nn.Module`` that can be used with ordinary PyTorch optimizers.  Unlike
``GeneReconModel``, it does not build one resident wave layout for all gene
families.  It streams families through fixed resident chunks, accumulates the
shared global adjoints, and returns a scalar NLL whose backward pass supplies
the precomputed gradient with respect to the three global DTL rates.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Sequence

import torch

from gpurec.core.backward import Pi_wave_backward
from gpurec.core.batching import family_schedule_summary
from gpurec.core.batch_planning import (
    normalize_batch_packing,
    normalize_clade_budget,
    plan_family_batches,
)
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import (
    E_fixed_point,
    compute_nll,
    compute_nll_root_rows,
    prepare_origination_probs,
)
from gpurec.core.memory_policy import UniformPipelinePolicy, choose_uniform_pipeline_policy
from gpurec.core.model import (
    GeneDataset,
    normalize_family_selection,
    normalize_family_tree_paths,
    parse_alerax_family_file,
)
from gpurec.optimization.implicit_grad import _e_adjoint_and_theta_vjp

from ._family_layout import (
    build_family_wave_layout,
    family_wave_inputs,
    origination_probs_for_family_indices,
)
from ._validation import (
    bool_value,
    integer_value,
    nonnegative_int,
    nonnegative_float,
    optional_positive_int,
    positive_even_int,
    positive_float,
    positive_int,
    require_cuda_device,
    require_default_objective,
    theta_init_base_from_rates,
)


UNIFORM_OPTIMIZED_DEFAULT_FLAGS = {
    "GPUREC_SELF_LOOP_2D_BLOCK_W": "1",
}


@dataclass(frozen=True)
class _UniformChunkSpec:
    indices: list[int]
    clades: int
    splits: int


@dataclass(frozen=True)
class _UniformBuiltChunk:
    spec: _UniformChunkSpec
    wave_layout: dict[str, Any]
    waves: int
    max_wave: int
    split_rows: int
    max_wave_split_rows: int


@dataclass(frozen=True)
class UniformChunkMetadata:
    """Public metadata for one resident uniform chunk."""

    chunk_index: int
    family_indices: tuple[int, ...]
    family_names: tuple[str, ...]
    gene_tree_paths: tuple[tuple[str, ...], ...]
    family_count: int
    clade_count: int
    split_count: int
    wave_count: int
    max_wave_size: int
    split_rows: int
    max_wave_split_rows: int


@dataclass
class UniformChunkedState:
    dataset: GeneDataset
    species_helpers: dict[str, Any]
    ancestors_T: torch.Tensor | None
    unnorm_row_max: torch.Tensor
    built_chunks: list[_UniformBuiltChunk]
    device: torch.device
    dtype: torch.dtype
    origination_probs: torch.Tensor | None = None
    fixed_iters_Pi: int = 6
    fixed_iters_E: int | None = None
    max_iters_E: int = 2000
    tol_E: float = 1e-8
    neumann_terms: int = 3
    use_pruning: bool = True
    pruning_threshold: float = 1e-6
    warm_start_E: bool = True
    profile: bool = False
    warm_E: torch.Tensor | None = None


def _set_default_flags() -> None:
    for key, value in UNIFORM_OPTIMIZED_DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)


def _as_auto_int(name: str, value: int | float | str | None) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "auto", "default"):
            return "auto"
        if text in ("0", "none", "null"):
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, 'auto', or none") from exc
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, 'auto', or none")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{name} must be an integer, 'auto', or none")
        return int(number)
    raise ValueError(f"{name} must be an integer, 'auto', or none")


def _auto_nonnegative_int(
    name: str,
    value: int | float | str | None,
) -> int | str | None:
    normalized = _as_auto_int(name, value)
    if isinstance(normalized, int) and normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _auto_positive_int(
    name: str,
    value: int | float | str | None,
) -> int | str | None:
    normalized = _as_auto_int(name, value)
    if isinstance(normalized, int) and normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _normalize_nonnegative_int_sequence(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    try:
        return tuple(
            nonnegative_int(f"{name} entries", value) for value in values
        )
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc


def _normalize_positive_int_sequence(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    try:
        return tuple(positive_int(f"{name} entries", value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc


def _normalize_uniform_solver_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate public solver kwargs before CUDA setup or AleRax parsing."""
    normalized = dict(kwargs)
    if "dtype" in normalized:
        normalized["dtype"] = _validate_uniform_dtype(normalized["dtype"])
    if normalized.get("fixed_iters_E") is not None:
        normalized["fixed_iters_E"] = positive_int(
            "fixed_iters_E",
            normalized["fixed_iters_E"],
        )
    if "fixed_iters_Pi" in normalized:
        normalized["fixed_iters_Pi"] = positive_even_int(
            "fixed_iters_Pi",
            normalized["fixed_iters_Pi"],
        )
    if "max_iters_E" in normalized:
        normalized["max_iters_E"] = positive_int(
            "max_iters_E",
            normalized["max_iters_E"],
        )
    if "neumann_terms" in normalized:
        normalized["neumann_terms"] = positive_int(
            "neumann_terms",
            normalized["neumann_terms"],
        )
    if "tol_E" in normalized:
        normalized["tol_E"] = nonnegative_float("tol_E", normalized["tol_E"])
    if "pruning_threshold" in normalized:
        normalized["pruning_threshold"] = nonnegative_float(
            "pruning_threshold",
            normalized["pruning_threshold"],
        )
    for name in (
        "refresh_preprocess_cache",
        "use_pruning",
        "warm_start_E",
        "profile",
        "set_optimized_env",
    ):
        if name in normalized:
            normalized[name] = bool_value(name, normalized[name])
    if "family_chunk_size" in normalized:
        normalized["family_chunk_size"] = _auto_nonnegative_int(
            "family_chunk_size",
            normalized["family_chunk_size"],
        )
    if "max_wave_size" in normalized:
        normalized["max_wave_size"] = _auto_positive_int(
            "max_wave_size",
            normalized["max_wave_size"],
        )
    if "max_root_wave_size" in normalized:
        normalized["max_root_wave_size"] = optional_positive_int(
            "max_root_wave_size",
            normalized["max_root_wave_size"],
        )
    if "clade_budget" in normalized:
        normalized["clade_budget"] = normalize_clade_budget(
            normalized["clade_budget"]
        )
    if "batch_packing" in normalized:
        normalized["batch_packing"] = normalize_batch_packing(
            normalized["batch_packing"]
        )
    if "family_chunk_candidates" in normalized:
        normalized["family_chunk_candidates"] = _normalize_nonnegative_int_sequence(
            "family_chunk_candidates",
            normalized["family_chunk_candidates"],
        )
    if "max_wave_candidates" in normalized:
        normalized["max_wave_candidates"] = _normalize_positive_int_sequence(
            "max_wave_candidates",
            normalized["max_wave_candidates"],
        )
    return normalized


def _validate_uniform_dtype(dtype: Any) -> torch.dtype:
    if dtype not in (torch.float32, torch.float64, torch.bfloat16):
        raise ValueError(f"dtype must be fp32, fp64, or bf16, got {dtype}")
    return dtype


def _selected_gene_paths(
    folder: Path,
    *,
    gene_glob: str,
    start: int,
    max_families: int | None,
) -> list[str]:
    start, max_families = normalize_family_selection(start, max_families)
    paths = sorted(folder.glob(gene_glob))
    if not paths and gene_glob == "g_*.nwk":
        single = folder / "g.nwk"
        if single.exists():
            paths = [single]
    if not paths:
        raise FileNotFoundError(f"no gene trees matching {gene_glob!r} in {folder}")
    stop = None if max_families is None else start + max_families
    selected = paths[start:stop]
    if not selected:
        raise ValueError(
            f"empty family selection: start={start}, max_families={max_families}, "
            f"available={len(paths)}"
        )
    return [str(p) for p in selected]


def _make_chunks(
    indices: Sequence[int],
    clade_counts: Sequence[int],
    split_counts: Sequence[int],
    *,
    family_chunk_size: int,
    clade_budget: int | None,
    batch_packing: str = "sequential",
    leaf_counts: Sequence[int] | None = None,
    nonleaf_counts: Sequence[int] | None = None,
    schedule_depths: Sequence[int] | None = None,
    max_wave_size: int | None = None,
) -> list[_UniformChunkSpec]:
    plans = plan_family_batches(
        indices=indices,
        clade_counts=clade_counts,
        split_counts=split_counts,
        family_chunk_size=family_chunk_size,
        clade_budget=clade_budget,
        batch_packing=batch_packing,
        leaf_counts=leaf_counts,
        nonleaf_counts=nonleaf_counts,
        schedule_depths=schedule_depths,
        max_wave_size=max_wave_size,
    )
    return [
        _UniformChunkSpec(plan.indices, plan.clades, plan.splits)
        for plan in plans
    ]


def _build_chunk(
    dataset: GeneDataset,
    spec: _UniformChunkSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_wave_size: int | None,
    max_root_wave_size: int | None,
) -> _UniformBuiltChunk:
    family_layout = build_family_wave_layout(
        family_wave_inputs(dataset, spec.indices),
        device=device,
        dtype=dtype,
        max_wave_size=max_wave_size,
        max_root_wave_size=max_root_wave_size,
    )
    wave_layout = family_layout.wave_layout

    metas = wave_layout["wave_metas"]
    max_wave = max((int(m["W"]) for m in metas), default=0)
    split_rows = sum(int(m["sl"].numel()) for m in metas if m.get("has_splits", False))
    max_wave_split_rows = max(
        (
            int(m["sl"].numel()) if m.get("has_splits", False) else 0
            for m in metas
        ),
        default=0,
    )
    return _UniformBuiltChunk(
        spec=spec,
        wave_layout=wave_layout,
        waves=len(metas),
        max_wave=max_wave,
        split_rows=split_rows,
        max_wave_split_rows=max_wave_split_rows,
    )


def _accumulate_pi_backward(
    acc: dict[str, Any] | None,
    pi_bwd: dict[str, Any],
) -> dict[str, Any]:
    tensor_keys = (
        "grad_E",
        "grad_Ebar",
        "grad_E_s1",
        "grad_E_s2",
        "grad_log_pD",
        "grad_log_pS",
        "grad_max_transfer_mat",
    )
    scalar_keys = (
        "n_waves_total",
        "n_waves_skipped",
        "n_waves_processed",
        "n_clades_total",
        "n_clades_skipped",
        "n_clades_active",
    )
    if acc is None:
        acc = {key: pi_bwd[key].detach().clone() for key in tensor_keys}
        for key in scalar_keys:
            acc[key] = int(pi_bwd.get(key, 0))
        return acc
    for key in tensor_keys:
        acc[key].add_(pi_bwd[key])
    for key in scalar_keys:
        acc[key] = int(acc.get(key, 0)) + int(pi_bwd.get(key, 0))
    return acc


def _time_cuda_ms(enabled: bool, fn):
    if not enabled:
        return 0.0, fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


def _root_count_tensor(
    state: UniformChunkedState,
    count: int | None = None,
) -> torch.Tensor:
    return torch.zeros(
        (len(state.dataset.families) if count is None else int(count),),
        device=state.device,
        dtype=torch.long,
    )


def _selected_chunks(
    state: UniformChunkedState,
    chunk_indices: Sequence[int] | torch.Tensor | None,
) -> list[tuple[int, _UniformBuiltChunk]]:
    if chunk_indices is None:
        return list(enumerate(state.built_chunks))
    if torch.is_tensor(chunk_indices):
        values = chunk_indices.detach().cpu().reshape(-1).tolist()
    else:
        try:
            values = list(chunk_indices)
        except TypeError as exc:
            raise ValueError("chunk_indices must be a sequence of integers") from exc
    indices = [
        integer_value("chunk_indices entries", value)
        for value in values
    ]
    if not indices:
        raise ValueError("chunk_indices must not be empty")
    n_chunks = len(state.built_chunks)
    selected: list[tuple[int, _UniformBuiltChunk]] = []
    seen: set[int] = set()
    for idx in indices:
        if idx < 0 or idx >= n_chunks:
            raise IndexError(f"chunk index {idx} out of range for {n_chunks} chunks")
        if idx in seen:
            raise ValueError(f"duplicate chunk index {idx}")
        seen.add(idx)
        selected.append((idx, state.built_chunks[idx]))
    return selected


def _evaluate_chunked_uniform(
    state: UniformChunkedState,
    theta: torch.Tensor,
    *,
    need_grad: bool,
    per_family: bool = False,
    chunk_indices: Sequence[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    if per_family and need_grad:
        raise ValueError("per-family output is only supported for no-grad evaluation")

    selected_chunks = _selected_chunks(state, chunk_indices)
    selected_family_indices = [
        family_idx
        for _chunk_idx, chunk in selected_chunks
        for family_idx in chunk.spec.indices
    ]
    selected_origination_probs = origination_probs_for_family_indices(
        state.origination_probs,
        selected_family_indices,
    )
    selected_family_count = sum(len(chunk.spec.indices) for _idx, chunk in selected_chunks)
    theta_eval = theta.detach().to(device=state.device, dtype=state.dtype)
    log_pS, log_pD, log_pL, max_transfer_vec = extract_parameters_uniform(
        theta_eval,
        state.unnorm_row_max,
        specieswise=False,
        genewise=False,
    )
    profile = bool(state.profile and state.device.type == "cuda")
    e_max_iters = (
        state.fixed_iters_E
        if state.fixed_iters_E is not None
        else state.max_iters_E
    )
    e_tolerance = -1.0 if state.fixed_iters_E is not None else state.tol_E
    e_ms, e_out = _time_cuda_ms(
        profile,
        lambda: E_fixed_point(
            species_helpers=state.species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            max_transfer_mat=max_transfer_vec,
            max_iters=e_max_iters,
            tolerance=e_tolerance,
            warm_start_E=state.warm_E if state.warm_start_E else None,
            dtype=state.dtype,
            device=state.device,
            ancestors_T=state.ancestors_T,
        ),
    )
    state.warm_E = e_out["E"].detach() if state.warm_start_E else None

    total_loss = torch.zeros((), device=state.device, dtype=state.dtype)
    per_family_parts: list[torch.Tensor] = []
    pi_bwd_acc: dict[str, Any] | None = None
    forward_ms = float(e_ms)
    pi_forward_ms = 0.0
    backward_ms = 0.0
    pi_backward_ms = 0.0
    chunk_stats: list[dict[str, Any]] = []

    for chunk_idx, built in selected_chunks:
        chunk_origination_probs = origination_probs_for_family_indices(
            state.origination_probs,
            built.spec.indices,
        )

        def run_forward():
            pi_out = Pi_wave_forward(
                wave_layout=built.wave_layout,
                species_helpers=state.species_helpers,
                E=e_out["E"],
                Ebar=e_out["E_bar"],
                E_s1=e_out["E_s1"],
                E_s2=e_out["E_s2"],
                log_pS=log_pS,
                log_pD=log_pD,
                max_transfer_mat=max_transfer_vec,
                device=state.device,
                dtype=state.dtype,
                fixed_iters=state.fixed_iters_Pi,
                return_original=False,
                return_root_rows=not need_grad,
            )
            if need_grad:
                loss_vec = compute_nll(
                    pi_out["Pi_wave_ordered"],
                    e_out["E"],
                    built.wave_layout["root_clade_ids"],
                    chunk_origination_probs,
                    origination_probs_prepared=True,
                )
            else:
                loss_vec = compute_nll_root_rows(
                    pi_out["Pi_root_rows"],
                    e_out["E"],
                    chunk_origination_probs,
                    origination_probs_prepared=True,
                )
            return pi_out, loss_vec

        fwd_ms, (pi_out, loss_vec) = _time_cuda_ms(profile, run_forward)
        pi_forward_ms += fwd_ms
        forward_ms += fwd_ms
        total_loss = total_loss + loss_vec.sum()
        if per_family:
            per_family_parts.append(loss_vec.detach())

        bwd_ms = 0.0
        if need_grad:
            def run_backward():
                return Pi_wave_backward(
                    wave_layout=built.wave_layout,
                    Pi_star_wave=pi_out["Pi_wave_ordered"],
                    Pibar_star_wave=pi_out["Pibar_wave_ordered"],
                    E=e_out["E"],
                    Ebar=e_out["E_bar"],
                    E_s1=e_out["E_s1"],
                    E_s2=e_out["E_s2"],
                    log_pS=log_pS,
                    log_pD=log_pD,
                    log_pL=log_pL,
                    max_transfer_mat=max_transfer_vec,
                    species_helpers=state.species_helpers,
                    root_clade_ids_perm=built.wave_layout["root_clade_ids"],
                    device=state.device,
                    dtype=state.dtype,
                    neumann_terms=state.neumann_terms,
                    use_pruning=state.use_pruning,
                    pruning_threshold=state.pruning_threshold,
                    uniform_pibar_row_max=pi_out.get("uniform_pibar_row_max"),
                    origination_probs=chunk_origination_probs,
                    origination_probs_prepared=True,
                )

            bwd_ms, pi_bwd = _time_cuda_ms(profile, run_backward)
            pi_backward_ms += bwd_ms
            backward_ms += bwd_ms
            pi_bwd_acc = _accumulate_pi_backward(pi_bwd_acc, pi_bwd)
            del pi_bwd

        chunk_stats.append(
            {
                "idx": chunk_idx,
                "family_start": int(built.spec.indices[0]),
                "family_stop": int(built.spec.indices[-1]) + 1,
                "families": len(built.spec.indices),
                "clades": built.spec.clades,
                "splits": built.spec.splits,
                "waves": built.waves,
                "max_wave": built.max_wave,
                "split_rows": built.split_rows,
                "max_wave_split_rows": built.max_wave_split_rows,
                "forward_ms": fwd_ms,
                "pi_backward_ms": bwd_ms,
            }
        )
        del pi_out, loss_vec

    grad_theta: torch.Tensor | None = None
    e_adjoint_ms = 0.0
    if need_grad:
        if pi_bwd_acc is None:
            raise RuntimeError("internal error: no Pi backward result was accumulated")

        def run_e_adjoint():
            return _e_adjoint_and_theta_vjp(
                pi_bwd_acc,
                e_out["E"],
                e_out["E_bar"],
                e_out["E_s1"],
                e_out["E_s2"],
                log_pS,
                log_pD,
                log_pL,
                max_transfer_vec,
                state.species_helpers,
                _root_count_tensor(state, selected_family_count),
                theta_eval,
                state.unnorm_row_max,
                False,
                state.device,
                state.dtype,
                genewise=False,
                ancestors_T=state.ancestors_T,
                origination_probs=selected_origination_probs,
                origination_probs_prepared=True,
            )

        e_adjoint_ms, (grad_theta, _stats) = _time_cuda_ms(profile, run_e_adjoint)
        backward_ms += e_adjoint_ms

    if state.device.type == "cuda":
        torch.cuda.synchronize(state.device)
        peak_alloc_gib = torch.cuda.max_memory_allocated(state.device) / (1024 ** 3)
        peak_reserved_gib = torch.cuda.max_memory_reserved(state.device) / (1024 ** 3)
    else:
        peak_alloc_gib = float("nan")
        peak_reserved_gib = float("nan")

    out_loss = (
        torch.cat(per_family_parts, dim=0)
        if per_family
        else total_loss.detach()
    )
    stats = {
        "loss": float(total_loss.detach().cpu()),
        "selected_chunks": [int(idx) for idx, _chunk in selected_chunks],
        "selected_families": int(selected_family_count),
        "total_families": len(state.dataset.families),
        "forward_ms": forward_ms,
        "e_ms": float(e_ms),
        "pi_forward_ms": pi_forward_ms,
        "backward_ms": backward_ms,
        "pi_backward_ms": pi_backward_ms,
        "e_adjoint_ms": e_adjoint_ms,
        "total_ms": forward_ms + backward_ms,
        "peak_alloc_gib": peak_alloc_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "chunk_rows": chunk_stats,
        "grad_norm": (
            float(torch.linalg.vector_norm(grad_theta).detach().cpu())
            if grad_theta is not None
            else None
        ),
    }
    return out_loss, grad_theta, stats


class _UniformChunkedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, state: UniformChunkedState):
        require_default_objective("UniformChunkedReconModel")
        with torch.no_grad():
            loss, grad_theta, stats = _evaluate_chunked_uniform(
                state,
                theta,
                need_grad=True,
            )
        if grad_theta is None:
            raise RuntimeError("internal error: missing chunked uniform gradient")
        ctx.save_for_backward(grad_theta.detach().to(device=theta.device, dtype=theta.dtype))
        return loss.to(device=theta.device, dtype=theta.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (grad_theta,) = ctx.saved_tensors
        return grad_theta * grad_output.to(device=grad_theta.device, dtype=grad_theta.dtype), None


class UniformChunkedReconModel(torch.nn.Module):
    """Global/uniform DTL model backed by the optimized chunked pipeline.

    The module has one trainable parameter, ``theta`` with shape ``[3]`` in
    log2-rate space.  ``forward()`` returns the summed negative log-likelihood
    in bits.  The forward call computes both the objective and the analytical
    gradient internally; the subsequent ``loss.backward()`` call simply returns
    that cached gradient to PyTorch.

    Tree inputs use the retained preprocessing parser's simple Newick subset:
    one rooted binary species tree, unquoted labels, ignored numeric branch
    lengths, and gene-tree files that may contain multiple semicolon-delimited
    records for CCP amalgamation.

    ``torch.float32`` and ``torch.float64`` are the supported production dtypes.
    ``torch.bfloat16`` is accepted only on this direct API as an experimental
    CUDA memory-saving path for forward/NLL probes; workflow configuration and
    CLI runs intentionally expose only fp32/fp64, and the retained Pi
    backward/gradient path does not support bf16.
    """

    def __init__(
        self,
        *,
        species_tree: str | os.PathLike[str],
        gene_trees: Sequence[str | os.PathLike[str] | Sequence[str | os.PathLike[str]]],
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        theta_init_rates: tuple[float, float, float] = (0.05, 0.05, 0.05),
        family_names: Sequence[str] | None = None,
        leaf_species_maps: Sequence[dict[str, str]] | None = None,
        preprocess_cache_dir: str | os.PathLike[str] | None = None,
        refresh_preprocess_cache: bool = False,
        family_chunk_size: int | str = "auto",
        max_wave_size: int | str | None = "auto",
        max_root_wave_size: int | None = None,
        clade_budget: int | None = None,
        batch_packing: str = "sequential",
        family_chunk_candidates: Sequence[int] = (25, 50, 10, 75, 100),
        max_wave_candidates: Sequence[int] = (8192, 16384, 4096, 32768),
        fixed_iters_Pi: int = 6,
        fixed_iters_E: int | None = None,
        max_iters_E: int = 2000,
        tol_E: float = 1e-8,
        neumann_terms: int = 3,
        use_pruning: bool = True,
        pruning_threshold: float = 1e-6,
        warm_start_E: bool = True,
        profile: bool = False,
        set_optimized_env: bool = True,
        origination_probs: torch.Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        require_default_objective("UniformChunkedReconModel")
        dtype = _validate_uniform_dtype(dtype)
        theta_init = theta_init_base_from_rates(
            theta_init_rates,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        if theta_init is None:
            raise ValueError("theta_init_rates must be provided")
        if fixed_iters_E is not None:
            fixed_iters_E = positive_int("fixed_iters_E", fixed_iters_E)
        fixed_iters_Pi = positive_even_int("fixed_iters_Pi", fixed_iters_Pi)
        max_iters_E = positive_int("max_iters_E", max_iters_E)
        neumann_terms = positive_int("neumann_terms", neumann_terms)
        tol_E = nonnegative_float("tol_E", tol_E)
        pruning_threshold = nonnegative_float("pruning_threshold", pruning_threshold)
        refresh_preprocess_cache = bool_value(
            "refresh_preprocess_cache",
            refresh_preprocess_cache,
        )
        use_pruning = bool_value("use_pruning", use_pruning)
        warm_start_E = bool_value("warm_start_E", warm_start_E)
        profile = bool_value("profile", profile)
        set_optimized_env = bool_value("set_optimized_env", set_optimized_env)
        chunk_value = _auto_nonnegative_int("family_chunk_size", family_chunk_size)
        wave_value = _auto_positive_int("max_wave_size", max_wave_size)
        max_root_wave_size = optional_positive_int(
            "max_root_wave_size",
            max_root_wave_size,
        )
        clade_budget = normalize_clade_budget(clade_budget)
        normalized_packing = normalize_batch_packing(batch_packing)
        family_chunk_candidates = _normalize_nonnegative_int_sequence(
            "family_chunk_candidates",
            family_chunk_candidates,
        )
        max_wave_candidates = _normalize_positive_int_sequence(
            "max_wave_candidates",
            max_wave_candidates,
        )
        gene_paths = normalize_family_tree_paths(gene_trees)

        if set_optimized_env:
            _set_default_flags()
        device = require_cuda_device(device, owner="UniformChunkedReconModel")
        theta_init = theta_init.to(device=device)

        dataset = GeneDataset(
            species_tree_path=str(species_tree),
            gene_tree_paths=gene_paths,
            genewise=False,
            specieswise=False,
            dtype=dtype,
            device=device,
            preprocess_cache_dir=preprocess_cache_dir,
            refresh_preprocess_cache=refresh_preprocess_cache,
            family_names=family_names,
            leaf_species_maps=leaf_species_maps,
        )
        species_helpers, ancestors_T = dataset._species_helpers_for_mode(
            device=device,
            dtype=dtype,
        )
        unnorm_row_max = dataset.unnorm_row_max.to(device=device, dtype=dtype)
        prepared_origination_probs = prepare_origination_probs(
            origination_probs,
            S=int(dataset.S),
            device=device,
            dtype=dtype,
            family_count=len(dataset.families) if origination_probs is not None else None,
        )

        clade_counts = [int(f["C"]) for f in dataset.families]
        split_counts = [int(f["N_splits"]) for f in dataset.families]
        leaf_counts: list[int] | None = None
        nonleaf_counts: list[int] | None = None
        schedule_depths: list[int] | None = None
        if normalized_packing == "depth_first_fit":
            summaries = [
                family_schedule_summary(fam["ccp_helpers"])
                for fam in dataset.families
            ]
            leaf_counts = [int(summary["leaf_count"]) for summary in summaries]
            nonleaf_counts = [int(summary["nonleaf_count"]) for summary in summaries]
            schedule_depths = [int(summary["max_level"]) for summary in summaries]
        memory_policy: UniformPipelinePolicy | None = None
        if chunk_value == "auto" or wave_value == "auto":
            chunk_candidates = (
                family_chunk_candidates
                if chunk_value == "auto"
                else (int(chunk_value),)
            )
            wave_candidates = (
                max_wave_candidates
                if wave_value == "auto"
                else (sum(clade_counts) if wave_value is None else int(wave_value),)
            )
            memory_policy = choose_uniform_pipeline_policy(
                clade_counts,
                int(dataset.S),
                dtype,
                device=device,
                family_chunk_candidates=chunk_candidates,
                max_wave_candidates=wave_candidates,
                clade_budget=clade_budget,
                batch_packing=normalized_packing,
                leaf_counts=leaf_counts,
                nonleaf_counts=nonleaf_counts,
                schedule_depths=schedule_depths,
            )
            if chunk_value == "auto":
                chunk_value = memory_policy.family_chunk_size
            if wave_value == "auto":
                wave_value = memory_policy.max_wave_size

        family_chunk_n = 0 if chunk_value is None else int(chunk_value)
        max_wave_n = None if wave_value is None else int(wave_value)
        specs = _make_chunks(
            list(range(len(dataset.families))),
            clade_counts,
            split_counts,
            family_chunk_size=family_chunk_n,
            clade_budget=clade_budget,
            batch_packing=normalized_packing,
            leaf_counts=leaf_counts,
            nonleaf_counts=nonleaf_counts,
            schedule_depths=schedule_depths,
            max_wave_size=max_wave_n,
        )
        built_chunks = [
            _build_chunk(
                dataset,
                spec,
                device=device,
                dtype=dtype,
                max_wave_size=max_wave_n,
                max_root_wave_size=max_root_wave_size,
            )
            for spec in specs
        ]
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        self.theta = torch.nn.Parameter(theta_init)
        self.register_buffer("origination_probs", prepared_origination_probs)
        self._state = UniformChunkedState(
            dataset=dataset,
            species_helpers=species_helpers,
            ancestors_T=ancestors_T,
            unnorm_row_max=unnorm_row_max,
            built_chunks=built_chunks,
            device=device,
            dtype=dtype,
            origination_probs=self.origination_probs,
            fixed_iters_Pi=fixed_iters_Pi,
            fixed_iters_E=fixed_iters_E,
            max_iters_E=max_iters_E,
            tol_E=tol_E,
            neumann_terms=neumann_terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            warm_start_E=warm_start_E,
            profile=profile,
        )
        self.family_chunk_size = family_chunk_n
        self.max_wave_size = max_wave_n
        self.max_root_wave_size = max_root_wave_size
        self.clade_budget = clade_budget
        self.batch_packing = normalized_packing
        self.memory_policy = memory_policy
        self.gene_trees = dataset.gene_tree_paths
        self.family_names = dataset.family_names
        self.species_tree = str(species_tree)

    @property
    def n_families(self) -> int:
        return len(self._state.dataset.families)

    @property
    def family_count(self) -> int:
        return self.n_families

    @property
    def chunk_count(self) -> int:
        return len(self._state.built_chunks)

    @property
    def fixed_iters_Pi(self) -> int:
        return self._state.fixed_iters_Pi

    @property
    def fixed_iters_E(self) -> int | None:
        return self._state.fixed_iters_E

    @property
    def chunk_metadata(self) -> tuple[UniformChunkMetadata, ...]:
        return tuple(
            UniformChunkMetadata(
                chunk_index=chunk_idx,
                family_indices=tuple(built.spec.indices),
                family_names=tuple(
                    self.family_names[family_idx]
                    for family_idx in built.spec.indices
                ),
                gene_tree_paths=tuple(
                    tuple(self.gene_trees[family_idx])
                    for family_idx in built.spec.indices
                ),
                family_count=len(built.spec.indices),
                clade_count=int(built.spec.clades),
                split_count=int(built.spec.splits),
                wave_count=int(built.waves),
                max_wave_size=int(built.max_wave),
                split_rows=int(built.split_rows),
                max_wave_split_rows=int(built.max_wave_split_rows),
            )
            for chunk_idx, built in enumerate(self._state.built_chunks)
        )

    @classmethod
    def from_trees(
        cls,
        species_tree: str | os.PathLike[str],
        gene_trees: Sequence[str | os.PathLike[str] | Sequence[str | os.PathLike[str]]],
        *,
        mode: str = "global",
        **kwargs: Any,
    ) -> "UniformChunkedReconModel":
        """Build the chunked global/uniform model from explicit tree paths.

        This mirrors :meth:`gpurec.GeneReconModel.from_trees` for constructor
        homogeneity.  The chunked implementation is intentionally global-only;
        use :class:`gpurec.GeneReconModel` for specieswise or genewise rates.
        Tree files use the supported simple Newick subset documented by
        :meth:`gpurec.GeneReconModel.from_trees`.
        """
        normalized = str(mode).strip().lower()
        if normalized not in {"global", "uniform"}:
            raise ValueError(
                "UniformChunkedReconModel.from_trees only supports mode='global' "
                f"or mode='uniform', got {mode!r}"
            )
        require_default_objective("UniformChunkedReconModel")
        kwargs = _normalize_uniform_solver_kwargs(kwargs)
        return cls(species_tree=species_tree, gene_trees=gene_trees, **kwargs)

    @classmethod
    def from_folder(
        cls,
        folder: str | os.PathLike[str],
        *,
        species_tree_name: str = "sp.nwk",
        gene_glob: str = "g_*.nwk",
        start: int = 0,
        max_families: int | None = None,
        **kwargs: Any,
    ) -> "UniformChunkedReconModel":
        """Build a model from a folder containing ``sp.nwk`` and gene trees.

        The folder contents use the supported simple Newick subset documented
        by :meth:`gpurec.GeneReconModel.from_trees`.
        """
        require_default_objective("UniformChunkedReconModel")
        start, max_families = normalize_family_selection(start, max_families)
        kwargs = _normalize_uniform_solver_kwargs(kwargs)
        root = Path(folder)
        species_tree = root / species_tree_name
        if not species_tree.exists():
            raise FileNotFoundError(f"missing species tree: {species_tree}")
        genes = _selected_gene_paths(
            root,
            gene_glob=gene_glob,
            start=start,
            max_families=max_families,
        )
        return cls(species_tree=species_tree, gene_trees=genes, **kwargs)

    @classmethod
    def from_alerax_families(
        cls,
        species_tree: str | os.PathLike[str],
        families_file: str | os.PathLike[str],
        *,
        mode: str = "global",
        start: int = 0,
        max_families: int | None = None,
        **kwargs: Any,
    ) -> "UniformChunkedReconModel":
        """Build the uniform model from an AleRax family/CCP list.

        Referenced tree files use the supported simple Newick subset documented
        by :meth:`gpurec.GeneReconModel.from_trees`.
        """
        normalized = str(mode).strip().lower()
        if normalized not in {"global", "uniform"}:
            raise ValueError(
                "UniformChunkedReconModel.from_alerax_families only supports "
                f"mode='global' or mode='uniform', got {mode!r}"
            )
        require_default_objective("UniformChunkedReconModel")
        start, max_families = normalize_family_selection(start, max_families)
        kwargs = _normalize_uniform_solver_kwargs(kwargs)
        theta_init_base_from_rates(
            kwargs.get("theta_init_rates", (0.05, 0.05, 0.05)),
            dtype=kwargs.get("dtype", torch.float32),
            device=torch.device("cpu"),
        )
        require_cuda_device(
            kwargs.get("device", "cuda"),
            owner="UniformChunkedReconModel",
        )
        family_names, tree_paths, leaf_maps = parse_alerax_family_file(
            families_file,
            start=start,
            max_families=max_families,
        )
        return cls(
            species_tree=species_tree,
            gene_trees=tree_paths,
            family_names=family_names,
            leaf_species_maps=leaf_maps,
            **kwargs,
        )

    def forward(self) -> torch.Tensor:
        if not torch.is_grad_enabled() or not self.theta.requires_grad:
            return self.nll()
        return _UniformChunkedFunction.apply(self.theta, self._state)

    @torch.no_grad()
    def nll(
        self,
        chunk_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss, _grad, stats = _evaluate_chunked_uniform(
            self._state,
            self.theta,
            need_grad=False,
            chunk_indices=chunk_indices,
        )
        return loss

    @torch.no_grad()
    def nll_per_family(
        self,
        chunk_indices: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss, _grad, stats = _evaluate_chunked_uniform(
            self._state,
            self.theta,
            need_grad=False,
            per_family=True,
            chunk_indices=chunk_indices,
        )
        return loss

    @torch.no_grad()
    def loss_and_grad(
        self,
        *,
        chunk_indices: Sequence[int] | torch.Tensor | None = None,
        reduction: str = "sum",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Evaluate selected chunks and return ``(loss, grad, stats)``.

        This bypasses PyTorch autograd and exposes the custom chunked gradient
        directly, which is useful for stochastic optimizers that sample chunks.

        ``reduction`` controls the returned loss and gradient scale:

        - ``"sum"``: selected-family NLL sum.
        - ``"mean"``: selected-family mean NLL.
        - ``"full_sum_estimate"``: selected sum scaled by
          ``total_families / selected_families``.
        """
        if reduction not in ("sum", "mean", "full_sum_estimate"):
            raise ValueError(
                "reduction must be 'sum', 'mean', or 'full_sum_estimate', "
                f"got {reduction!r}"
            )
        loss, grad, stats = _evaluate_chunked_uniform(
            self._state,
            self.theta,
            need_grad=True,
            chunk_indices=chunk_indices,
        )
        if grad is None:
            raise RuntimeError("internal error: missing chunked uniform gradient")
        selected_families = int(stats["selected_families"])
        total_families = int(stats["total_families"])
        if reduction == "mean":
            scale = 1.0 / float(selected_families)
        elif reduction == "full_sum_estimate":
            scale = float(total_families) / float(selected_families)
        else:
            scale = 1.0
        if scale != 1.0:
            scale_t = torch.as_tensor(scale, device=loss.device, dtype=loss.dtype)
            loss = loss * scale_t
            grad = grad * scale_t.to(device=grad.device, dtype=grad.dtype)
        stats = dict(stats)
        stats["reduction"] = reduction
        stats["scale"] = float(scale)
        stats["reduced_loss"] = float(loss.detach().cpu())
        stats["reduced_grad_norm"] = float(torch.linalg.vector_norm(grad).detach().cpu())
        return (
            loss.to(device=self.theta.device, dtype=self.theta.dtype),
            grad.to(device=self.theta.device, dtype=self.theta.dtype),
            stats,
        )

    def clamp_theta_(
        self,
        min_rate: float = 1e-10,
        max_rate: float | None = None,
    ) -> None:
        min_rate = positive_float("min_rate", min_rate)
        if max_rate is not None:
            max_rate = positive_float("max_rate", max_rate)
        if max_rate is not None and max_rate < min_rate:
            raise ValueError("max_rate must be greater than or equal to min_rate")
        with torch.no_grad():
            self.theta.clamp_(
                min=math.log2(min_rate),
                max=None if max_rate is None else math.log2(max_rate),
            )


__all__ = [
    "UniformChunkMetadata",
    "UniformChunkedReconModel",
]
