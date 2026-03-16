import torch
from .log2_utils import logsumexp2

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

def gather_Pi_children(Pi, sp_P_idx, child_index):
    C, S = Pi.shape
    # TODO: fix this. filling and initializing is slow
    Pi_children = torch.full((C, 2*S), float("-inf"), device=Pi.device, dtype=Pi.dtype)  # [2C, S]
    values = torch.index_select(Pi, 1, child_index) # [C, N_internal_nodes]
    Pi_children.index_copy_(1, sp_P_idx, values)
    return Pi_children

def compute_DTS(log_pD, log_pS, Pi_s12, Pi, Pibar, log_split_probs, split_leftrights_sorted, N_splits, S):
    # If too heavy in memory, don't create the DTS array and just do logaddexp 4 times.
    Pi_leftright = torch.index_select(Pi, 0, split_leftrights_sorted) # [2 N_splits, S]

    Pi_left, Pi_right = torch.chunk(Pi_leftright, 2, dim=0)  # Each [N_splits, S]
    Pibar_leftrights = torch.index_select(Pibar, 0, split_leftrights_sorted) # [2 N_splits, S]
    Pibar_left, Pibar_right = torch.chunk(Pibar_leftrights, 2, dim=0)  # Each [N_splits, S]

    DTS = torch.empty((5, N_splits, S), device=Pi.device, dtype=Pi.dtype)
    # D term
    DTS[0] = log_pD + Pi_left + Pi_right
    # T terms
    DTS[1] = Pi_left + Pibar_right
    DTS[2] = Pi_right + Pibar_left

    Pi_s12_leftright = torch.index_select(Pi_s12, 0, split_leftrights_sorted) # [2 N_splits, 2 S]
    Pi_s1_left = Pi_s12_leftright[:N_splits, :S]
    Pi_s2_left = Pi_s12_leftright[:N_splits, S:]
    Pi_s1_right = Pi_s12_leftright[N_splits:, :S]
    Pi_s2_right = Pi_s12_leftright[N_splits:, S:]
    
    # S terms
    DTS[3] = log_pS + Pi_s1_left + Pi_s2_right
    DTS[4] = log_pS + Pi_s1_right + Pi_s2_left
    DTS_term = log_split_probs + logsumexp2(DTS, dim=0)
    return DTS_term

def compute_DTS_L(log_pD, log_pS, Pi, Pibar, Pi_s12, E, Ebar, E_s1, E_s2, clade_species_map, log_2):
    # If too heavy in memory, don't create the DTS_L array and just do logaddexp 5 times
    DTS_L = torch.empty((6, Pi.shape[0], Pi.shape[1]), dtype=log_pD.dtype, device=log_pD.device)
    # DL
    DTS_L[0] = ((log_2 + log_pD + E).unsqueeze(0) + Pi)
    # TL
    DTS_L[1] = (Pi + Ebar.unsqueeze(0))
    DTS_L[2] = (Pibar + E.unsqueeze(0))
    # SL
    Pi_s1, Pi_s2 = torch.chunk(Pi_s12, 2, dim=1)  # Each [C, S]
    DTS_L[3] = ((log_pS + E_s2).unsqueeze(0) + Pi_s1)
    DTS_L[4] = ((log_pS + E_s1).unsqueeze(0) + Pi_s2)
    # leaf
    DTS_L[5] = (log_pS + clade_species_map)
    DTS_L_term = logsumexp2(DTS_L, dim=0)
    return DTS_L_term

