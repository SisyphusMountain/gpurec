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
