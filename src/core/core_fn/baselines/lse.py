import torch


def lse4_torch(x0, x1, x2, x3):
    return torch.logsumexp(torch.stack([x0, x1, x2, x3], dim=0), dim=0)

def lse5_torch(x0, x1, x2, x3, x4):
    return torch.logsumexp(torch.stack([x0, x1, x2, x3, x4], dim=0), dim=0)

def lse7_torch(x0, x1, x2, x3, x4, x5, x6):
    return torch.logsumexp(torch.stack([x0, x1, x2, x3, x4, x5, x6], dim=0), dim=0)
