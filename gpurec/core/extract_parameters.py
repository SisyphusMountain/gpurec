import torch

from .log2_utils import log2_softmax


def extract_parameters_uniform(theta, unnorm_row_max, specieswise, genewise=False):
    """Extract DTL log-probabilities for the retained uniform-transfer path.

    Returns ``(log_pS, log_pD, log_pL, None, max_transfer_mat)``. The ``None``
    placeholder keeps call sites simple: uniform transfer only needs each row's
    max transfer log-probability, never a dense ``[S, S]`` transfer matrix.
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
        return result[..., 0], result[..., 1], result[..., 2], None, max_transfer_mat

    if specieswise:
        zeros = theta.new_zeros((theta.shape[0], 1))
        result = log2_softmax(torch.cat((zeros, theta), dim=-1), dim=-1)
        max_transfer_mat = result[..., 3] + unnorm_row_max
    else:
        zeros = theta.new_zeros((1,))
        result = log2_softmax(torch.cat((zeros, theta), dim=-1), dim=-1)
        max_transfer_mat = result[3] + unnorm_row_max

    return result[..., 0], result[..., 1], result[..., 2], None, max_transfer_mat
