import math

import torch

from gpurec.core.inference.logspace import safe_log2

_LN2 = 0.6931471805599453
_SUPPORTED_ACCUMULATOR_DTYPES = (torch.float32, torch.float64)


def resolve_accumulator_dtype(
    accumulator_dtype: torch.dtype | None,
    *,
    fallback: torch.dtype,
) -> torch.dtype:
    """Resolve and validate the dtype used by small numerical reductions.

    Callers that own runtime configuration pass ``accumulator_dtype``
    explicitly. Standalone helpers fall back to their input dtype instead of
    silently selecting a wider, hardcoded precision.
    """

    dtype = fallback if accumulator_dtype is None else accumulator_dtype
    if dtype not in _SUPPORTED_ACCUMULATOR_DTYPES:
        raise TypeError("accumulator_dtype must be torch.float32 or torch.float64")
    return dtype


def _log_softmax2(
    logits: torch.Tensor,
    *,
    inputs_are_log2: bool = True,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Base-2 log-softmax evaluated in the selected accumulator dtype."""

    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=logits.dtype,
    )
    compute = logits.to(dtype=accumulator_dtype)
    if inputs_are_log2:
        compute = compute * _LN2
    return (torch.log_softmax(compute, dim=-1) / _LN2).to(logits.dtype)


def as_family_param(t, family_rows, S=None):
    if t.ndim == 0:
        return t.reshape(1, 1).expand(int(family_rows), 1).contiguous()
    if t.ndim == 1:
        if S is not None and int(family_rows) == 1 and int(t.numel()) == int(S):
            return t.reshape(1, int(S)).contiguous()
        return t.reshape(int(family_rows), 1).contiguous()
    return t.contiguous()


def as_family_species(t, S, family_rows):
    param = as_family_param(t, family_rows, S)
    if int(param.shape[1]) == int(S):
        return param.contiguous()
    return param.expand(int(family_rows), int(S)).contiguous()


def extract_parameters_uniform(
    theta,
    unnorm_row_max,
    *,
    specieswise=False,
    genewise=False,
    accumulator_dtype: torch.dtype | None = None,
):
    # The preprocessing static uses the configured accumulator dtype. Cast it
    # to the model dtype at this dense-compute boundary.
    unnorm_row_max = unnorm_row_max.to(device=theta.device, dtype=theta.dtype)
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    logits = torch.cat((zeros, theta), dim=-1)
    result = _log_softmax2(logits, accumulator_dtype=accumulator_dtype)
    log_pT = result[..., 3]
    if specieswise and not genewise:
        max_transfer = log_pT + unnorm_row_max
    else:
        max_transfer = log_pT.unsqueeze(-1) + unnorm_row_max
    return result[..., 0], result[..., 1], result[..., 2], max_transfer


def receiver_log_probs_from_weights(
    receiver_weights: torch.Tensor,
    *,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return _log_softmax2(
        receiver_weights,
        inputs_are_log2=False,
        accumulator_dtype=accumulator_dtype,
    )


def origination_log_probs_from_weights(
    origination_weights: torch.Tensor,
    *,
    accumulator_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Base-2 log of the per-branch origination distribution (softmax over the S species nodes).

    Mirrors ``receiver_log_probs_from_weights``: ``origination_weights`` are unconstrained logits,
    gauge-fixed by the softmax (invariant under a constant shift). The default all-equal weights
    give ``-log2(S)`` everywhere, i.e. the uniform origination prior the model assumes by default.
    """
    return _log_softmax2(
        origination_weights,
        inputs_are_log2=False,
        accumulator_dtype=accumulator_dtype,
    )


def receiver_valid_log_normalizer(
    receiver_log_probs: torch.Tensor,
    sp_parent: torch.Tensor,
    max_ancestor_depth: int,
) -> torch.Tensor:
    # For each donor ``s`` the valid transfer recipients are all species that are NOT ``s`` itself
    # nor a strict ancestor of ``s``. The normalizer is ``-log2(valid_mass)`` where ``valid_mass`` is
    # the receiver probability on those valid recipients. Computing it as ``1 - ancestor_mass``
    # catastrophically cancels once the excluded mass rounds to 1 (deep donor, or receiver weight
    # concentrated on the ancestors) -- the old code then hid the collapse behind a ``clamp_min(tiny)``
    # before ``log2``. Instead accumulate the valid mass DIRECTLY as a sum over non-excluded recipients:
    # a sum of non-negative terms cannot cancel, so no ``1 - .`` and no clamp are needed.
    S = int(receiver_log_probs.numel())
    receiver_probs = torch.exp2(receiver_log_probs)
    parent = sp_parent.to(device=receiver_log_probs.device, dtype=torch.long)
    idx = torch.arange(S, device=receiver_log_probs.device, dtype=torch.long)
    cur = idx.clone()
    excluded = torch.zeros((S, S), dtype=torch.bool, device=receiver_log_probs.device)  # excluded[s, r]

    for _ in range(max(1, int(max_ancestor_depth))):
        valid = (cur >= 0) & (cur < S)
        safe_cur = cur.clamp(0, max(S - 1, 0))  # index-only clamp (bounds safety); the ``valid`` mask carries the logic
        # Mark ``s`` itself and every ancestor visited. Written as a read-modify-write over the
        # FULL row index rather than ``excluded[idx[valid], safe_cur[valid]] = True``: selecting
        # with a boolean mask has to count the True entries on the host, so each such select ran
        # ``nonzero`` and stalled the host until the GPU drained -- 68 device-to-host round trips
        # per call here, measured at 198 ms of GPU idle per 200-family gradient. An invalid row
        # ORs in ``False`` at its clamped column, which leaves that entry exactly as it was, so
        # the marked set is identical to the masked write's.
        excluded[idx, safe_cur] = excluded[idx, safe_cur] | valid
        cur = torch.where(valid, parent.index_select(0, safe_cur), torch.full_like(cur, -1))

    valid_mass = (receiver_probs.unsqueeze(0) * (~excluded)).sum(dim=-1)  # sum of non-excluded receiver probs
    return torch.where(
        valid_mass > 0.0,
        -safe_log2(valid_mass),
        receiver_log_probs.new_full((S,), float("-inf")),
    )


def extract_parameters_weighted_receivers(
    theta,
    receiver_weights,
    species_helpers,
    *,
    specieswise=False,
    genewise=False,
    uniform_fast=False,
    accumulator_dtype: torch.dtype | None = None,
):
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    logits = torch.cat((zeros, theta), dim=-1)
    result = _log_softmax2(logits, accumulator_dtype=accumulator_dtype)
    log_pT = result[..., 3]
    receiver_log_probs = receiver_log_probs_from_weights(
        receiver_weights.to(device=theta.device, dtype=theta.dtype),
        accumulator_dtype=accumulator_dtype,
    )
    receiver_norm = receiver_valid_log_normalizer(
        receiver_log_probs,
        species_helpers["sp_parent"],
        int(species_helpers["max_ancestor_depth"]),
    )
    if specieswise and not genewise:
        max_transfer = log_pT + receiver_norm
    else:
        max_transfer = log_pT.unsqueeze(-1) + receiver_norm
    if uniform_fast:
        max_transfer = max_transfer - math.log2(int(species_helpers["S"]))
    return result[..., 0], result[..., 1], result[..., 2], max_transfer, receiver_log_probs
