"""Streaming implementations extracted from :meth:`GeneReconModel._stream_full_batches`."""
from __future__ import annotations

from typing import Any

import torch

from gpurec.core.gradient_accumulator import GradientAccumulator
from gpurec.core.parameter_layout import ParameterLayout
from gpurec.core.forward import prepare_shared_pi_forward_constants
from gpurec.core.likelihood import compute_origination_denominator
from ._uniform_evaluator import evaluate_resident_static_state, evaluate_resident_no_grad_with_solved_e
from .autograd import solve_resident_e
from ._tensor_validation import _validate_gradient_shape, _validate_scalar_loss


def stream_single_static(
    model: Any,
    theta: torch.Tensor,
    *,
    need_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    loss, grad = evaluate_resident_static_state(
        model._active_static(),
        theta,
        need_grad=need_grad,
    )
    loss = _validate_scalar_loss("full-batch NLL", loss)
    if need_grad:
        if grad is None:
            raise RuntimeError("internal error: missing full-batch gradient")
        grad = _validate_gradient_shape(
            "full-batch gradient",
            grad,
            expected_shape=tuple(int(dim) for dim in theta.shape),
        )
    return loss, grad


def stream_shared_theta_loss_only(
    model: Any,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, None]:
    first_static = model._ensure_batch_static(0)
    e_solve = solve_resident_e(first_static, theta)
    origination_denominator = (
        compute_origination_denominator(
            e_solve.e_out["E"],
            model._origination_prior.probs,
            origination_probs_prepared=True,
        )
        if model._origination_prior.is_shared
        else None
    )
    prepared_shared_constants = None
    if all(
        field in e_solve.e_out
        for field in ("E_bar", "E_s1", "E_s2")
    ):
        prepared_shared_constants = prepare_shared_pi_forward_constants(
            E=e_solve.e_out["E"],
            Ebar=e_solve.e_out["E_bar"],
            E_s1=e_solve.e_out["E_s1"],
            E_s2=e_solve.e_out["E_s2"],
            log_pS=e_solve.log_p_s,
            log_pD=e_solve.log_p_d,
            max_transfer_mat=e_solve.max_transfer,
            S=int(model._dataset.S),
        )
    scratch_shape = (
        max(meta.clade_count for meta in model.batch_metadata),
        int(model._dataset.S),
    )
    scratch_tensors = (
        torch.empty(
            scratch_shape,
            device=model._dataset.device,
            dtype=model._dataset.dtype,
        ),
        torch.empty(
            scratch_shape,
            device=model._dataset.device,
            dtype=model._dataset.dtype,
        ),
    )
    total_loss = torch.zeros(
        (),
        device=model._dataset.device,
        dtype=model._dataset.dtype,
    )
    stream_count = (
        min(
            getattr(model, "shared_loss_batch_streams", 1),
            len(model._batch_specs),
        )
        if model._dataset.device.type == "cuda"
        else 1
    )
    if stream_count <= 1:
        for batch_idx in range(len(model._batch_specs)):
            static = model._ensure_batch_static(batch_idx)
            loss_i = evaluate_resident_no_grad_with_solved_e(
                static,
                e_solve,
                scratch_tensors=scratch_tensors,
                origination_denominator=origination_denominator,
                prepared_shared_constants=prepared_shared_constants,
            )
            loss_i = _validate_scalar_loss("full-batch NLL", loss_i)
            total_loss = total_loss + loss_i.to(
                device=total_loss.device,
                dtype=total_loss.dtype,
            )
    else:
        stream_scratch_tensors = [scratch_tensors]
        for _ in range(1, stream_count):
            stream_scratch_tensors.append(
                (
                    torch.empty(
                        scratch_shape,
                        device=model._dataset.device,
                        dtype=model._dataset.dtype,
                    ),
                    torch.empty(
                        scratch_shape,
                        device=model._dataset.device,
                        dtype=model._dataset.dtype,
                    ),
                )
            )
        current_stream = torch.cuda.current_stream(model._dataset.device)
        streams = [
            torch.cuda.Stream(device=model._dataset.device)
            for _ in range(stream_count)
        ]
        for stream in streams:
            stream.wait_stream(current_stream)
        batch_losses: list[torch.Tensor | None] = [None for _ in model._batch_specs]
        for batch_idx in range(len(model._batch_specs)):
            static = model._ensure_batch_static(batch_idx)
            stream_idx = batch_idx % stream_count
            stream = streams[stream_idx]
            with torch.cuda.stream(stream):
                batch_losses[batch_idx] = (
                    evaluate_resident_no_grad_with_solved_e(
                        static,
                        e_solve,
                        scratch_tensors=stream_scratch_tensors[stream_idx],
                        origination_denominator=origination_denominator,
                        prepared_shared_constants=prepared_shared_constants,
                    )
                )
        for stream in streams:
            current_stream.wait_stream(stream)
        for loss_i in batch_losses:
            if loss_i is None:
                raise RuntimeError("internal error: missing batch loss")
            loss_i = _validate_scalar_loss("full-batch NLL", loss_i)
            total_loss = total_loss + loss_i.to(
                device=total_loss.device,
                dtype=total_loss.dtype,
            )
    return total_loss.detach(), None


def stream_batched_loss_and_grad(
    model: Any,
    theta: torch.Tensor,
    *,
    need_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    total_loss = torch.zeros((), device=model._dataset.device, dtype=model._dataset.dtype)
    grad_accumulator = (
        GradientAccumulator.zeros_like(
            ParameterLayout.for_mode(
                model._mode,
                species_count=int(model._dataset.S),
                family_count=len(model._dataset.families),
            ),
            theta,
        )
        if need_grad
        else None
    )
    for batch_idx in range(len(model._batch_specs)):
        static = model._ensure_batch_static(batch_idx)
        theta_batch = model._theta_for_batch_index(batch_idx, theta)
        loss_i, grad_i = evaluate_resident_static_state(
            static,
            theta_batch,
            need_grad=need_grad,
        )
        loss_i = _validate_scalar_loss("batch NLL", loss_i)
        total_loss = total_loss + loss_i.to(
            device=total_loss.device,
            dtype=total_loss.dtype,
        )
        if need_grad:
            if grad_i is None or grad_accumulator is None:
                raise RuntimeError("internal error: missing batch gradient")
            grad_accumulator.add(
                grad_i,
                family_indices=(
                    model._batch_specs[batch_idx].family_indices
                    if model._mode == "genewise"
                    else None
                ),
            )
    return (
        total_loss.detach(),
        None if grad_accumulator is None else grad_accumulator.result().detach(),
    )


def stream_full_batches(
    model: Any,
    theta: torch.Tensor,
    *,
    need_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not model._batched_resident:
        return stream_single_static(model, theta, need_grad=need_grad)
    if not need_grad and model._mode != "genewise":
        return stream_shared_theta_loss_only(model, theta)
    return stream_batched_loss_and_grad(model, theta, need_grad=need_grad)
