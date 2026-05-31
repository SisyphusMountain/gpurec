"""Private evaluation helpers for :mod:`gpurec.api.uniform_chunked`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from gpurec.core.backward import Pi_wave_backward
from gpurec.core.extract_parameters import extract_parameters_uniform
from gpurec.core.forward import pi_root_row_loss_request, pi_training_state_request
from gpurec.core.gradient_accumulator import StructuredGradientAccumulator
from gpurec.core.likelihood import E_fixed_point
from gpurec.optimization.implicit_grad import _e_adjoint_and_theta_vjp

from ._uniform_evaluator import compute_pi_output_root_nll
from ._validation import integer_value


_PI_BACKWARD_TENSOR_KEYS = (
    "grad_E",
    "grad_Ebar",
    "grad_E_s1",
    "grad_E_s2",
    "grad_log_pD",
    "grad_log_pS",
    "grad_max_transfer_mat",
)
_PI_BACKWARD_COUNTER_KEYS = (
    "n_waves_total",
    "n_waves_skipped",
    "n_waves_processed",
    "n_clades_total",
    "n_clades_skipped",
    "n_clades_active",
)


@dataclass(frozen=True)
class _UniformChunkedEvaluation:
    loss: torch.Tensor
    grad_theta: torch.Tensor | None
    stats: dict[str, Any]
    per_family_nll: torch.Tensor | None = None


@dataclass(frozen=True)
class _UniformChunkedReadOnlyEvaluation:
    loss: torch.Tensor
    stats: dict[str, Any]
    per_family_nll: torch.Tensor | None = None


@dataclass(frozen=True)
class _UniformChunkStatsRow:
    idx: int
    family_start: int
    family_stop: int
    families: int
    clades: int
    splits: int
    waves: int
    max_wave: int
    split_rows: int
    max_wave_split_rows: int
    forward_ms: float
    pi_backward_ms: float

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "family_start": self.family_start,
            "family_stop": self.family_stop,
            "families": self.families,
            "clades": self.clades,
            "splits": self.splits,
            "waves": self.waves,
            "max_wave": self.max_wave,
            "split_rows": self.split_rows,
            "max_wave_split_rows": self.max_wave_split_rows,
            "forward_ms": self.forward_ms,
            "pi_backward_ms": self.pi_backward_ms,
        }


def _new_pi_backward_accumulator() -> StructuredGradientAccumulator:
    return StructuredGradientAccumulator(
        tensor_keys=_PI_BACKWARD_TENSOR_KEYS,
        counter_keys=_PI_BACKWARD_COUNTER_KEYS,
    )


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
    state: Any,
    count: int | None = None,
) -> torch.Tensor:
    return torch.zeros(
        (len(state.dataset.families) if count is None else int(count),),
        device=state.device,
        dtype=torch.long,
    )


def _selected_chunks(
    state: Any,
    chunk_indices: Sequence[int] | torch.Tensor | None,
) -> list[tuple[int, Any]]:
    if chunk_indices is None:
        return list(enumerate(state.built_chunks))
    if torch.is_tensor(chunk_indices):
        values = chunk_indices.detach().cpu().reshape(-1).tolist()
    else:
        try:
            values = list(chunk_indices)
        except TypeError as exc:
            raise ValueError("chunk_indices must be a sequence of integers") from exc
    indices = [integer_value("chunk_indices entries", value) for value in values]
    if not indices:
        raise ValueError("chunk_indices must not be empty")
    n_chunks = len(state.built_chunks)
    selected: list[tuple[int, Any]] = []
    seen: set[int] = set()
    for idx in indices:
        if idx < 0 or idx >= n_chunks:
            raise IndexError(f"chunk index {idx} out of range for {n_chunks} chunks")
        if idx in seen:
            raise ValueError(f"duplicate chunk index {idx}")
        seen.add(idx)
        selected.append((idx, state.built_chunks[idx]))
    return selected


def _e_adjoint_stats_fields(stats: Any) -> dict[str, Any]:
    """Return direct ``loss_and_grad()`` telemetry for the E-adjoint solve."""
    return {
        "e_adjoint_method": str(getattr(stats, "method", "")),
        "e_adjoint_iterations": int(getattr(stats, "iters", 0)),
        "e_adjoint_rel_res": float(getattr(stats, "rel_res", float("nan"))),
        "e_adjoint_success": bool(getattr(stats, "success", True)),
    }


def _chunk_stats_row(
    *,
    chunk_idx: int,
    built: Any,
    forward_ms: float,
    pi_backward_ms: float,
) -> dict[str, Any]:
    indices = built.spec.indices
    row = _UniformChunkStatsRow(
        idx=int(chunk_idx),
        family_start=int(indices[0]),
        family_stop=int(indices[-1]) + 1,
        families=len(indices),
        clades=int(built.spec.clades),
        splits=int(built.spec.splits),
        waves=int(built.waves),
        max_wave=int(built.max_wave),
        split_rows=int(built.split_rows),
        max_wave_split_rows=int(built.max_wave_split_rows),
        forward_ms=float(forward_ms),
        pi_backward_ms=float(pi_backward_ms),
    )
    return row.as_public_dict()


def _require_chunked_gradient_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "UniformChunkedReconModel gradient evaluation requires float32 or "
            "float64; the retained Pi_wave_backward path does not support "
            f"{dtype}. Use nll() under no_grad for bf16 forward probes."
        )


def _evaluate_chunked_uniform_result(
    state: Any,
    theta: torch.Tensor,
    *,
    need_grad: bool,
    collect_per_family: bool = False,
    chunk_indices: Sequence[int] | torch.Tensor | None = None,
) -> _UniformChunkedEvaluation:
    if collect_per_family and need_grad:
        raise ValueError("per-family output is only supported for no-grad evaluation")
    if need_grad:
        _require_chunked_gradient_dtype(state.dtype)

    selected_chunks = _selected_chunks(state, chunk_indices)
    selected_family_indices = [
        family_idx
        for _chunk_idx, chunk in selected_chunks
        for family_idx in chunk.spec.indices
    ]
    selected_origination_prior = state.origination_prior.select_families(
        selected_family_indices,
    )
    selected_origination_probs = selected_origination_prior.probs
    selected_family_count = sum(
        len(chunk.spec.indices) for _idx, chunk in selected_chunks
    )
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
            e_shape=(int(state.species_helpers["S"]),),
        ),
    )
    state.warm_E = e_out["E"].detach() if state.warm_start_E else None

    total_loss = torch.zeros((), device=state.device, dtype=state.dtype)
    per_family_parts: list[torch.Tensor] = []
    pi_bwd_accumulator = _new_pi_backward_accumulator() if need_grad else None
    forward_ms = float(e_ms)
    pi_forward_ms = 0.0
    backward_ms = 0.0
    pi_backward_ms = 0.0
    chunk_stats: list[dict[str, Any]] = []
    pi_request = pi_training_state_request() if need_grad else pi_root_row_loss_request()

    for chunk_idx, built in selected_chunks:
        chunk_origination_probs = state.origination_prior.select_families(
            built.spec.indices,
        ).probs

        def run_forward():
            pi_out = pi_request.run(
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
            )
            if need_grad:
                loss_vec = compute_pi_output_root_nll(
                    pi_out,
                    e_out["E"],
                    chunk_origination_probs,
                    root_clade_ids=built.wave_layout["root_clade_ids"],
                    origination_probs_prepared=True,
                )
            else:
                loss_vec = compute_pi_output_root_nll(
                    pi_out,
                    e_out["E"],
                    chunk_origination_probs,
                    origination_probs_prepared=True,
                )
            return pi_out, loss_vec

        fwd_ms, (pi_out, loss_vec) = _time_cuda_ms(profile, run_forward)
        pi_forward_ms += fwd_ms
        forward_ms += fwd_ms
        total_loss = total_loss + loss_vec.sum()
        if collect_per_family:
            per_family_parts.append(loss_vec.detach())

        bwd_ms = 0.0
        if need_grad:

            def _run_backward(
                pi_output: dict[str, torch.Tensor | None],
            ) -> torch.Tensor:
                return Pi_wave_backward(
                    wave_layout=built.wave_layout,
                    Pi_star_wave=pi_output["Pi_wave_ordered"],
                    Pibar_star_wave=pi_output["Pibar_wave_ordered"],
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
                    uniform_pibar_row_max=pi_output.get("uniform_pibar_row_max"),
                    origination_probs=chunk_origination_probs,
                    origination_probs_prepared=True,
                )

            bwd_ms, pi_bwd = _time_cuda_ms(
                profile,
                lambda pi_output=pi_out: _run_backward(pi_output),
            )
            pi_backward_ms += bwd_ms
            backward_ms += bwd_ms
            if pi_bwd_accumulator is None:
                raise RuntimeError("internal error: missing Pi backward accumulator")
            pi_bwd_accumulator.add(pi_bwd)
            del pi_bwd

        chunk_stats.append(
            _chunk_stats_row(
                chunk_idx=chunk_idx,
                built=built,
                forward_ms=fwd_ms,
                pi_backward_ms=bwd_ms,
            )
        )
        del pi_out, loss_vec

    grad_theta: torch.Tensor | None = None
    e_adjoint_ms = 0.0
    e_adjoint_stats: dict[str, Any] = {}
    if need_grad:
        if pi_bwd_accumulator is None or pi_bwd_accumulator.is_empty:
            raise RuntimeError("internal error: no Pi backward result was accumulated")
        pi_bwd_acc = pi_bwd_accumulator.result()

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

        e_adjoint_ms, (grad_theta, e_stats) = _time_cuda_ms(profile, run_e_adjoint)
        e_adjoint_stats = _e_adjoint_stats_fields(e_stats)
        backward_ms += e_adjoint_ms

    if state.device.type == "cuda":
        torch.cuda.synchronize(state.device)
        peak_alloc_gib = torch.cuda.max_memory_allocated(state.device) / (1024**3)
        peak_reserved_gib = torch.cuda.max_memory_reserved(state.device) / (1024**3)
    else:
        peak_alloc_gib = float("nan")
        peak_reserved_gib = float("nan")

    per_family_nll = torch.cat(per_family_parts, dim=0) if collect_per_family else None
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
        **e_adjoint_stats,
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
    return _UniformChunkedEvaluation(
        loss=total_loss.detach(),
        grad_theta=grad_theta,
        stats=stats,
        per_family_nll=per_family_nll,
    )


def _evaluate_chunked_uniform(
    state: Any,
    theta: torch.Tensor,
    *,
    need_grad: bool,
    per_family: bool = False,
    chunk_indices: Sequence[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    result = _evaluate_chunked_uniform_result(
        state,
        theta,
        need_grad=need_grad,
        collect_per_family=per_family,
        chunk_indices=chunk_indices,
    )
    if per_family:
        if result.per_family_nll is None:
            raise RuntimeError("internal error: missing chunked per-family NLL")
        return result.per_family_nll, result.grad_theta, result.stats
    return result.loss, result.grad_theta, result.stats


def _evaluate_chunked_uniform_read_only(
    state: Any,
    theta: torch.Tensor,
    *,
    collect_per_family: bool = False,
    chunk_indices: Sequence[int] | torch.Tensor | None = None,
) -> _UniformChunkedReadOnlyEvaluation:
    with torch.no_grad():
        result = _evaluate_chunked_uniform_result(
            state,
            theta,
            need_grad=False,
            collect_per_family=collect_per_family,
            chunk_indices=chunk_indices,
        )
    return _UniformChunkedReadOnlyEvaluation(
        loss=result.loss,
        stats=result.stats,
        per_family_nll=result.per_family_nll,
    )
