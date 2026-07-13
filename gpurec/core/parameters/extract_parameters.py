import math

import torch

from gpurec.core.inference.logspace import safe_log2

_LN2 = 0.6931471805599453


def _log_softmax2(
    logits: torch.Tensor, *, inputs_are_log2: bool = True
) -> torch.Tensor:
    """Base-2 log-softmax with a cheap fp64 evaluation for fp32 logits."""
    compute = logits.double() if logits.dtype == torch.float32 else logits
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


def extract_parameters_uniform(theta, unnorm_row_max, *, specieswise=False, genewise=False):
    # unnorm_row_max is a full-precision float64 batch static (batching.py); cast to the compute
    # dtype so the transfer term runs at the model's precision (f64 keeps it; f32 downcasts to the
    # old value). Mirrors the receiver_weights cast in extract_parameters_weighted_receivers.
    unnorm_row_max = unnorm_row_max.to(device=theta.device, dtype=theta.dtype)
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    logits = torch.cat((zeros, theta), dim=-1)
    result = _log_softmax2(logits)
    log_pT = result[..., 3]
    if specieswise and not genewise:
        max_transfer = log_pT + unnorm_row_max
    else:
        max_transfer = log_pT.unsqueeze(-1) + unnorm_row_max
    return result[..., 0], result[..., 1], result[..., 2], max_transfer


def receiver_log_probs_from_weights(receiver_weights: torch.Tensor) -> torch.Tensor:
    return _log_softmax2(receiver_weights, inputs_are_log2=False)


def origination_log_probs_from_weights(origination_weights: torch.Tensor) -> torch.Tensor:
    """Base-2 log of the per-branch origination distribution (softmax over the S species nodes).

    Mirrors ``receiver_log_probs_from_weights``: ``origination_weights`` are unconstrained logits,
    gauge-fixed by the softmax (invariant under a constant shift). The default all-equal weights
    give ``-log2(S)`` everywhere, i.e. the uniform origination prior the model assumes by default.
    """
    return _log_softmax2(origination_weights, inputs_are_log2=False)


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
        excluded[idx[valid], safe_cur[valid]] = True  # mark ``s`` itself and every ancestor visited
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
):
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    logits = torch.cat((zeros, theta), dim=-1)
    result = _log_softmax2(logits)
    log_pT = result[..., 3]
    receiver_log_probs = receiver_log_probs_from_weights(receiver_weights.to(device=theta.device, dtype=theta.dtype))
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
