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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from gpurec.core.backward import Pi_wave_backward
from gpurec.core.batching import (
    build_wave_layout,
    collate_gene_families,
    collate_wave,
    split_phase_waves,
)
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import (
    E_fixed_point,
    compute_log_likelihood,
    compute_log_likelihood_root_rows,
)
from gpurec.core.memory_policy import UniformPipelinePolicy, choose_uniform_pipeline_policy
from gpurec.core.model import GeneDataset
from gpurec.core.scheduling import compute_clade_waves
from gpurec.optimization.implicit_grad import _e_adjoint_and_theta_vjp


UNIFORM_OPTIMIZED_DEFAULT_FLAGS = {
    "GPUREC_FORWARD_LEAF_INDEX": "1",
    "GPUREC_UNIFORM_PINGPONG": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_TILE_SPLITS": "64",
    "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
    "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
    "GPUREC_FORWARD_DTS_OVERLAP_MODE": "off",
    "GPUREC_KERNELIZED_ACTIVE_MASK": "1",
    "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
    "GPUREC_FUSED_DTS_BACKWARD_ACCUM": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
    "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL": "tree",
    "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
    "GPUREC_BACKWARD_LEAF_INDEX": "1",
    "GPUREC_FUSED_WAVE_PARAM_ACCUM": "1",
    "GPUREC_DTS_PIBAR_UD_FUSION": "1",
    "GPUREC_DTS_PIBAR_UD_SKIP_ZERO_SIDES": "1",
    "GPUREC_DTS_PIBAR_UD_COMPACT_LEVELS": "1",
    "GPUREC_DTS_GRAD_MT_TWO_STAGE": "1",
    "GPUREC_BACKWARD_PARENT_REDUCED_DTS": "tiled",
    "GPUREC_BACKWARD_PARENT_REDUCED_DTS_TILE_SPLITS": "64",
    "GPUREC_SELF_LOOP_2D_TRITON": "auto",
    "GPUREC_SELF_LOOP_2D_BLOCK_W": "1",
}


@dataclass(frozen=True)
class UniformChunkSpec:
    indices: list[int]
    clades: int
    splits: int


@dataclass(frozen=True)
class UniformBuiltChunk:
    spec: UniformChunkSpec
    wave_layout: dict[str, Any]
    waves: int
    max_wave: int
    split_rows: int
    max_wave_split_rows: int


@dataclass
class UniformChunkedState:
    dataset: GeneDataset
    species_helpers: dict[str, Any]
    ancestors_T: torch.Tensor | None
    unnorm_row_max: torch.Tensor
    built_chunks: list[UniformBuiltChunk]
    device: torch.device
    dtype: torch.dtype
    fixed_iters_Pi: int | None = 6
    fixed_iters_E: int | None = None
    max_iters_E: int = 2000
    tol_E: float = 1e-8
    max_iters_Pi: int = 2000
    tol_Pi: float = 1e-6
    neumann_terms: int = 3
    use_pruning: bool = True
    pruning_threshold: float = 1e-6
    cg_tol: float = 1e-8
    cg_maxiter: int = 500
    gmres_restart: int = 40
    warm_start_E: bool = True
    profile: bool = False
    warm_E: torch.Tensor | None = None
    last_stats: dict[str, Any] = field(default_factory=dict)


def _apply_tensor_tree(obj: Any, fn) -> Any:
    if torch.is_tensor(obj):
        if obj.is_floating_point() or obj.is_complex():
            return fn(obj)
        moved = fn(obj)
        return moved if moved.dtype == obj.dtype else moved.to(dtype=obj.dtype)
    if isinstance(obj, dict):
        return {k: _apply_tensor_tree(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply_tensor_tree(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_apply_tensor_tree(v, fn) for v in obj)
    return obj


def _apply_to_built_chunk(chunk: UniformBuiltChunk, fn) -> UniformBuiltChunk:
    return UniformBuiltChunk(
        spec=chunk.spec,
        wave_layout=_apply_tensor_tree(chunk.wave_layout, fn),
        waves=chunk.waves,
        max_wave=chunk.max_wave,
        split_rows=chunk.split_rows,
        max_wave_split_rows=chunk.max_wave_split_rows,
    )


def _apply_to_chunked_state(state: UniformChunkedState, fn) -> UniformChunkedState:
    """Move/cast tensor fields while preserving integer index dtypes."""
    new_state = UniformChunkedState(
        dataset=state.dataset,
        species_helpers=_apply_tensor_tree(state.species_helpers, fn),
        ancestors_T=_apply_tensor_tree(state.ancestors_T, fn),
        unnorm_row_max=_apply_tensor_tree(state.unnorm_row_max, fn),
        built_chunks=[
            _apply_to_built_chunk(chunk, fn)
            for chunk in state.built_chunks
        ],
        device=state.device,
        dtype=state.dtype,
        fixed_iters_Pi=state.fixed_iters_Pi,
        fixed_iters_E=state.fixed_iters_E,
        max_iters_E=state.max_iters_E,
        tol_E=state.tol_E,
        max_iters_Pi=state.max_iters_Pi,
        tol_Pi=state.tol_Pi,
        neumann_terms=state.neumann_terms,
        use_pruning=state.use_pruning,
        pruning_threshold=state.pruning_threshold,
        cg_tol=state.cg_tol,
        cg_maxiter=state.cg_maxiter,
        gmres_restart=state.gmres_restart,
        warm_start_E=state.warm_start_E,
        profile=state.profile,
        warm_E=None,
        last_stats={},
    )
    new_state.device = new_state.unnorm_row_max.device
    new_state.dtype = new_state.unnorm_row_max.dtype
    return new_state


def _set_default_flags() -> None:
    for key, value in UNIFORM_OPTIMIZED_DEFAULT_FLAGS.items():
        os.environ.setdefault(key, value)


def _as_auto_int(value: int | str | None) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "auto", "default"):
            return "auto"
        if text in ("0", "none", "null"):
            return None
        return int(text)
    return int(value)


def _selected_gene_paths(
    folder: Path,
    *,
    gene_glob: str,
    start: int,
    max_families: int | None,
) -> list[str]:
    paths = sorted(folder.glob(gene_glob))
    if not paths and gene_glob == "g_*.nwk":
        single = folder / "g.nwk"
        if single.exists():
            paths = [single]
    if not paths:
        raise FileNotFoundError(f"no gene trees matching {gene_glob!r} in {folder}")
    if start < 0:
        raise ValueError("start must be non-negative")
    stop = None if max_families is None else start + int(max_families)
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
) -> list[UniformChunkSpec]:
    chunks: list[UniformChunkSpec] = []
    current: list[int] = []
    current_clades = 0
    current_splits = 0

    def flush() -> None:
        nonlocal current, current_clades, current_splits
        if current:
            chunks.append(UniformChunkSpec(list(current), current_clades, current_splits))
            current = []
            current_clades = 0
            current_splits = 0

    for idx in indices:
        n_clades = int(clade_counts[idx])
        n_splits = int(split_counts[idx])
        family_cap_hit = family_chunk_size > 0 and len(current) >= family_chunk_size
        clade_cap_hit = (
            clade_budget is not None
            and current
            and current_clades + n_clades > clade_budget
        )
        if family_cap_hit or clade_cap_hit:
            flush()
        current.append(int(idx))
        current_clades += n_clades
        current_splits += n_splits
    flush()
    return chunks


def _build_chunk(
    dataset: GeneDataset,
    spec: UniformChunkSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_wave_size: int | None,
    max_root_wave_size: int | None,
) -> UniformBuiltChunk:
    items = []
    fam_waves = []
    fam_phases = []
    for idx in spec.indices:
        fam = dataset.families[idx]
        items.append(
            {
                "ccp": fam["ccp_helpers"],
                "leaf_row_index": fam["leaf_row_index"],
                "leaf_col_index": fam["leaf_col_index"],
                "root_clade_id": int(fam["root_clade_id"]),
            }
        )
        waves_i, phases_i = compute_clade_waves(fam["ccp_helpers"])
        fam_waves.append(waves_i)
        fam_phases.append(phases_i)

    batched = collate_gene_families(items, dtype=dtype, device=device)
    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(fam_waves, offsets)
    max_n_waves = max(len(p) for p in fam_phases)
    cross_phases: list[int] = []
    for k in range(max_n_waves):
        phase_k = 1
        for phases_i in fam_phases:
            if k < len(phases_i):
                phase_k = max(phase_k, int(phases_i[k]))
        cross_phases.append(phase_k)

    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=None,
        max_wave_size=max_wave_size,
    )
    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=3,
        max_wave_size=max_root_wave_size,
    )

    family_clade_counts = [m["C"] for m in batched["family_meta"]]
    family_clade_offsets = [m["clade_offset"] for m in batched["family_meta"]]
    wave_layout = build_wave_layout(
        waves=cross_waves,
        phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device,
        dtype=dtype,
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
    )

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
    return UniformBuiltChunk(
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


def _root_count_tensor(state: UniformChunkedState) -> torch.Tensor:
    return torch.zeros(
        (len(state.dataset.families),),
        device=state.device,
        dtype=torch.long,
    )


def _evaluate_chunked_uniform(
    state: UniformChunkedState,
    theta: torch.Tensor,
    *,
    need_grad: bool,
    per_family: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    if per_family and need_grad:
        raise ValueError("per-family output is only supported for no-grad evaluation")

    theta_eval = theta.detach().to(device=state.device, dtype=state.dtype)
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = extract_parameters_uniform(
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
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            max_iters=e_max_iters,
            tolerance=e_tolerance,
            warm_start_E=state.warm_E if state.warm_start_E else None,
            dtype=state.dtype,
            device=state.device,
            pibar_mode="uniform",
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

    for chunk_idx, built in enumerate(state.built_chunks):
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
                log_pL=log_pL,
                transfer_mat=transfer_mat,
                max_transfer_mat=max_transfer_vec,
                device=state.device,
                dtype=state.dtype,
                local_iters=state.max_iters_Pi,
                local_tolerance=state.tol_Pi,
                fixed_iters=state.fixed_iters_Pi,
                pibar_mode="uniform",
                return_original=False,
                need_pibar=need_grad,
                return_root_rows=not need_grad,
            )
            if need_grad:
                loss_vec = compute_log_likelihood(
                    pi_out["Pi_wave_ordered"],
                    e_out["E"],
                    built.wave_layout["root_clade_ids"],
                )
            else:
                loss_vec = compute_log_likelihood_root_rows(
                    pi_out["Pi_root_rows"],
                    e_out["E"],
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
                    pibar_mode="uniform",
                    transfer_mat=transfer_mat,
                    ancestors_T=state.ancestors_T,
                    uniform_pibar_row_max=pi_out.get("uniform_pibar_row_max"),
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
                _root_count_tensor(state),
                theta_eval,
                state.unnorm_row_max,
                False,
                state.device,
                state.dtype,
                genewise=False,
                cg_tol=state.cg_tol,
                cg_maxiter=state.cg_maxiter,
                gmres_restart=state.gmres_restart,
                pibar_mode="uniform",
                transfer_mat=transfer_mat,
                transfer_mat_unnormalized=None,
                ancestors_T=state.ancestors_T,
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
        if os.environ.get("GPUREC_ALERAX_COMPAT", "0") != "0":
            raise NotImplementedError(
                "GPUREC_ALERAX_COMPAT changes the forward objective to "
                "AleRax's fixed-pass evaluator. UniformChunkedReconModel's "
                "custom backward differentiates GPUREC's default fixed-point "
                "objective, so gradient-based optimization is disabled in "
                "this mode."
            )
        with torch.no_grad():
            loss, grad_theta, stats = _evaluate_chunked_uniform(
                state,
                theta,
                need_grad=True,
            )
        if grad_theta is None:
            raise RuntimeError("internal error: missing chunked uniform gradient")
        state.last_stats = stats
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
    """

    def __init__(
        self,
        *,
        species_tree: str | os.PathLike[str],
        gene_trees: Sequence[str | os.PathLike[str]],
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        theta_init_rates: tuple[float, float, float] = (0.05, 0.05, 0.05),
        preprocess_cache_dir: str | os.PathLike[str] | None = None,
        refresh_preprocess_cache: bool = False,
        family_chunk_size: int | str = "auto",
        max_wave_size: int | str | None = "auto",
        max_root_wave_size: int | None = None,
        clade_budget: int | None = None,
        family_chunk_candidates: Sequence[int] = (25, 50, 10, 75, 100),
        max_wave_candidates: Sequence[int] = (8192, 16384, 4096, 32768),
        fixed_iters_Pi: int | None = 6,
        fixed_iters_E: int | None = None,
        max_iters_E: int = 2000,
        tol_E: float = 1e-8,
        max_iters_Pi: int = 2000,
        tol_Pi: float = 1e-6,
        neumann_terms: int = 3,
        use_pruning: bool = True,
        pruning_threshold: float = 1e-6,
        cg_tol: float = 1e-8,
        cg_maxiter: int = 500,
        gmres_restart: int = 40,
        warm_start_E: bool = True,
        profile: bool = False,
        set_optimized_env: bool = True,
    ) -> None:
        super().__init__()
        if set_optimized_env:
            _set_default_flags()
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("UniformChunkedReconModel currently requires a CUDA device")
        if dtype not in (torch.float32, torch.float64, torch.bfloat16):
            raise ValueError(f"dtype must be fp32, fp64, or bf16, got {dtype}")
        if fixed_iters_E is not None:
            fixed_iters_E = int(fixed_iters_E)
            if fixed_iters_E < 1:
                raise ValueError("fixed_iters_E must be >= 1 when provided")

        gene_paths = [str(p) for p in gene_trees]
        if not gene_paths:
            raise ValueError("gene_trees must not be empty")

        dataset = GeneDataset(
            species_tree_path=str(species_tree),
            gene_tree_paths=gene_paths,
            genewise=False,
            specieswise=False,
            pairwise=False,
            dtype=dtype,
            device=device,
            preprocess_cache_dir=preprocess_cache_dir,
            refresh_preprocess_cache=refresh_preprocess_cache,
            retain_dense_species_matrices=False,
        )
        species_helpers, ancestors_T = dataset._species_helpers_for_mode(
            pibar_mode="uniform",
            device=device,
            dtype=dtype,
        )
        unnorm_row_max = dataset.unnorm_row_max.to(device=device, dtype=dtype)

        clade_counts = [int(f["C"]) for f in dataset.families]
        split_counts = [int(f["N_splits"]) for f in dataset.families]
        chunk_value = _as_auto_int(family_chunk_size)
        wave_value = _as_auto_int(max_wave_size)
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
            )
            if chunk_value == "auto":
                chunk_value = memory_policy.family_chunk_size
            if wave_value == "auto":
                wave_value = memory_policy.max_wave_size
            os.environ["GPUREC_SELF_LOOP_2D_TRITON"] = (
                "1" if memory_policy.proposal0 else "0"
            )

        family_chunk_n = 0 if chunk_value is None else int(chunk_value)
        max_wave_n = None if wave_value is None else int(wave_value)
        specs = _make_chunks(
            list(range(len(dataset.families))),
            clade_counts,
            split_counts,
            family_chunk_size=family_chunk_n,
            clade_budget=clade_budget,
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

        rates = torch.tensor(theta_init_rates, device=device, dtype=dtype)
        if torch.any(rates <= 0):
            raise ValueError("theta_init_rates must be strictly positive")
        self.theta = torch.nn.Parameter(torch.log2(rates))
        self._state = UniformChunkedState(
            dataset=dataset,
            species_helpers=species_helpers,
            ancestors_T=ancestors_T,
            unnorm_row_max=unnorm_row_max,
            built_chunks=built_chunks,
            device=device,
            dtype=dtype,
            fixed_iters_Pi=fixed_iters_Pi,
            fixed_iters_E=fixed_iters_E,
            max_iters_E=max_iters_E,
            tol_E=tol_E,
            max_iters_Pi=max_iters_Pi,
            tol_Pi=tol_Pi,
            neumann_terms=neumann_terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
            gmres_restart=gmres_restart,
            warm_start_E=warm_start_E,
            profile=profile,
        )
        self.family_chunk_size = family_chunk_n
        self.max_wave_size = max_wave_n
        self.max_root_wave_size = max_root_wave_size
        self.clade_budget = clade_budget
        self.memory_policy = memory_policy
        self.gene_trees = gene_paths
        self.species_tree = str(species_tree)

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
        """Build a model from a folder containing ``sp.nwk`` and gene trees."""
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

    def forward(self) -> torch.Tensor:
        if not torch.is_grad_enabled() or not self.theta.requires_grad:
            return self.nll()
        return _UniformChunkedFunction.apply(self.theta, self._state)

    @torch.no_grad()
    def nll(self) -> torch.Tensor:
        loss, _grad, stats = _evaluate_chunked_uniform(
            self._state,
            self.theta,
            need_grad=False,
        )
        self._state.last_stats = stats
        return loss

    @torch.no_grad()
    def nll_per_family(self) -> torch.Tensor:
        loss, _grad, stats = _evaluate_chunked_uniform(
            self._state,
            self.theta,
            need_grad=False,
            per_family=True,
        )
        self._state.last_stats = stats
        return loss

    @torch.no_grad()
    def log_likelihood(self) -> float:
        return float(-self.nll().item())

    def clamp_theta_(self, min_rate: float = 1e-10, max_rate: float | None = None) -> None:
        if min_rate <= 0:
            raise ValueError("min_rate must be strictly positive")
        if max_rate is not None and max_rate < min_rate:
            raise ValueError("max_rate must be greater than or equal to min_rate")
        with torch.no_grad():
            self.theta.clamp_(
                min=math.log2(min_rate),
                max=None if max_rate is None else math.log2(max_rate),
            )

    def clear_warm_start(self) -> None:
        self._state.warm_E = None

    @property
    def rates(self) -> torch.Tensor:
        return torch.exp2(self.theta.detach())

    @property
    def n_families(self) -> int:
        return len(self._state.dataset.families)

    @property
    def n_species(self) -> int:
        return int(self._state.dataset.S)

    @property
    def chunks(self) -> list[UniformBuiltChunk]:
        return list(self._state.built_chunks)

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._state.last_stats)

    def batch_summary(self) -> dict[str, Any]:
        chunks = self._state.built_chunks
        total_clades = sum(c.spec.clades for c in chunks)
        total_splits = sum(c.spec.splits for c in chunks)
        return {
            "families": self.n_families,
            "species": self.n_species,
            "chunks": len(chunks),
            "family_chunk_size": self.family_chunk_size,
            "max_wave_size": self.max_wave_size,
            "max_root_wave_size": self.max_root_wave_size,
            "clade_budget": self.clade_budget,
            "fixed_iters_E": self._state.fixed_iters_E,
            "fixed_iters_Pi": self._state.fixed_iters_Pi,
            "total_clades": total_clades,
            "total_splits": total_splits,
            "max_chunk_clades": max((c.spec.clades for c in chunks), default=0),
            "max_chunk_splits": max((c.spec.splits for c in chunks), default=0),
            "max_wave": max((c.max_wave for c in chunks), default=0),
            "total_waves": sum(c.waves for c in chunks),
            "dtype": str(self._state.dtype).replace("torch.", ""),
            "device": str(self._state.device),
            "memory_policy": self.memory_policy,
        }

    def _apply(self, fn):
        super()._apply(fn)
        self._state = _apply_to_chunked_state(self._state, fn)
        # The original auto policy estimate was dtype-specific.  The already
        # built chunking remains valid, but the estimate should not be reused.
        self.memory_policy = None
        return self


__all__ = ["UniformChunkedReconModel", "UniformBuiltChunk", "UniformChunkSpec"]
