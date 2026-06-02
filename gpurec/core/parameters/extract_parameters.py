import torch

_LN2 = 0.6931471805599453


def as_family_param(t, family_rows):
    if t.ndim == 0:
        return t.reshape(1, 1).expand(int(family_rows), 1).contiguous()
    return t.reshape(int(family_rows), 1).contiguous()


def as_family_species(t, S, family_rows):
    if t.ndim == 0:
        return t.reshape(1, 1).expand(int(family_rows), int(S)).contiguous()
    if t.ndim == 1:
        if int(family_rows) == 1 and int(t.numel()) == int(S):
            return t.reshape(1, int(S)).contiguous()
        return t.reshape(int(family_rows), 1).expand(int(family_rows), int(S)).contiguous()
    return t.contiguous()


def extract_parameters_uniform(theta, unnorm_row_max):
    zeros = theta.new_zeros((*theta.shape[:-1], 1))
    logits = torch.cat((zeros, theta), dim=-1)
    result = torch.log_softmax(logits * _LN2, dim=-1) / _LN2
    log_pT = result[..., 3]
    return result[..., 0], result[..., 1], result[..., 2], log_pT.unsqueeze(-1) + unnorm_row_max
