"""Term helpers for reconciliation likelihood."""
import torch

NEG_INF = float("-inf")

def gather_E_children(E, sp_P_idx, child_index):
    """Gather E values at species children into a 2*S layout.

    Supports E of shape [S] or [N_genes, S]. Returns tensors of shape [2*S]
    or [N_genes, 2*S] respectively, where entries not corresponding to
    parent-child slots are set to -inf.
    """
    if E.ndim == 1:
        S = E.shape[0]
        out = torch.full((2 * S,), NEG_INF, device=E.device, dtype=E.dtype)
        values = E.index_select(0, child_index)
        out.index_copy_(0, sp_P_idx, values)
        return out
    elif E.ndim == 2:
        N, S = E.shape
        out = torch.full((N, 2 * S), NEG_INF, device=E.device, dtype=E.dtype)
        # Select child columns and scatter them into parent slots for all genes
        values = E.index_select(1, child_index)  # [N, n_children]
        out.index_copy_(1, sp_P_idx, values)
        return out
    else:
        raise ValueError(f"E must be 1D or 2D, got shape {tuple(E.shape)}")
