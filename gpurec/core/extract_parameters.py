import torch

from .log2_utils import log2_softmax


def as_family_param(t, *, S, device, dtype, family_rows=None, name="parameter"):
    """Normalize scalar/species/family parameters to [P, 1] or [P, S]."""
    if not torch.is_tensor(t):
        raise TypeError(f"{name} must be a tensor")
    if t.device != device or t.dtype != dtype:
        t = t.to(device=device, dtype=dtype)
    if t.ndim == 0:
        return t.reshape(1, 1)
    if t.ndim == 1:
        n = int(t.shape[0])
        if family_rows is not None and n == int(family_rows):
            return t.reshape(n, 1)
        if n == int(S):
            return t.reshape(1, int(S))
        return t.reshape(n, 1)
    if t.ndim == 2 and int(t.shape[1]) in (1, int(S)):
        return t
    raise ValueError(f"{name} must have shape [], [S], [P], [P, 1], or [P, S]")


def as_family_species(t, *, S, device, dtype, family_rows=None, name="parameter"):
    param = as_family_param(
        t, S=S, device=device, dtype=dtype, family_rows=family_rows, name=name
    )
    return param.contiguous() if int(param.shape[1]) == int(S) else param.expand(-1, int(S)).contiguous()


def extract_parameters_uniform(theta, unnorm_row_max, specieswise, genewise=False):
    """Extract DTL log-probabilities for the retained uniform-transfer path.

    Returns ``(log_pS, log_pD, log_pL, max_transfer_mat)``. Uniform transfer
    only needs each row's max transfer log-probability, never a dense
    ``[S, S]`` transfer matrix.
    """
    if genewise:
        if specieswise:
            zeros = theta.new_zeros((*theta.shape[:2], 1))
            result = log2_softmax(torch.cat((zeros, theta), dim=-1), dim=-1)
            log_pT = result[..., 3]
            max_transfer_mat = log_pT + unnorm_row_max
        else:
            zeros = theta.new_zeros((theta.shape[0], 1))
            result = log2_softmax(torch.cat((zeros, theta), dim=-1), dim=-1)
            log_pT = result[:, 3]
            max_transfer_mat = log_pT.unsqueeze(-1) + unnorm_row_max
        return result[..., 0], result[..., 1], result[..., 2], max_transfer_mat

    if specieswise:
        zeros = theta.new_zeros((theta.shape[0], 1))
        result = log2_softmax(torch.cat((zeros, theta), dim=-1), dim=-1)
        max_transfer_mat = result[..., 3] + unnorm_row_max
    else:
        zeros = theta.new_zeros((1,))
        result = log2_softmax(torch.cat((zeros, theta), dim=-1), dim=-1)
        max_transfer_mat = result[3] + unnorm_row_max

    return result[..., 0], result[..., 1], result[..., 2], max_transfer_mat
