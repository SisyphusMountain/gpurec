import math

import torch

_LN2 = 0.6931471805599453


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
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    logits = torch.cat((zeros, theta), dim=-1)
    result = torch.log_softmax(logits * _LN2, dim=-1) / _LN2
    log_pT = result[..., 3]
    if specieswise and not genewise:
        max_transfer = log_pT + unnorm_row_max
    else:
        max_transfer = log_pT.unsqueeze(-1) + unnorm_row_max
    return result[..., 0], result[..., 1], result[..., 2], max_transfer


def receiver_log_probs_from_weights(receiver_weights: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(receiver_weights, dim=-1) / _LN2


def receiver_valid_log_normalizer(
    receiver_log_probs: torch.Tensor,
    sp_parent: torch.Tensor,
    max_ancestor_depth: int,
) -> torch.Tensor:
    S = int(receiver_log_probs.numel())
    receiver_probs = torch.exp2(receiver_log_probs)
    parent = sp_parent.to(device=receiver_log_probs.device, dtype=torch.long)
    cur = torch.arange(S, device=receiver_log_probs.device, dtype=torch.long)
    ancestor_mass = torch.zeros((S,), device=receiver_log_probs.device, dtype=receiver_log_probs.dtype)
    zero = torch.zeros((), device=receiver_log_probs.device, dtype=receiver_log_probs.dtype)

    for _ in range(max(1, int(max_ancestor_depth))):
        valid = (cur >= 0) & (cur < S)
        safe_cur = cur.clamp(0, max(S - 1, 0))
        ancestor_mass = ancestor_mass + torch.where(valid, receiver_probs.index_select(0, safe_cur), zero)
        next_cur = parent.index_select(0, safe_cur)
        cur = torch.where(valid, next_cur, torch.full_like(cur, -1))

    valid_mass = 1.0 - ancestor_mass
    return torch.where(
        valid_mass > 0.0,
        -torch.log2(valid_mass.clamp_min(torch.finfo(receiver_log_probs.dtype).tiny)),
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
    result = torch.log_softmax(logits * _LN2, dim=-1) / _LN2
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
