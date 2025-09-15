"""
Extinction probability computation for phylogenetic reconciliation.

This module implements the fixed-point iteration for computing extinction
probabilities on species tree branches, in both linear and log space.
"""

import torch
from typing import Dict, Tuple, Optional

def E_step_log(log_E: torch.Tensor,
               s_C1_indices: torch.Tensor,
               s_C2_indices: torch.Tensor,
               internal_mask: torch.Tensor,
               Recipients_mat: torch.Tensor,
               log_pS: torch.Tensor,
               log_pD: torch.Tensor,
               log_pT: torch.Tensor,
               log_pL: torch.Tensor,
               return_components: bool = False) -> Tuple[torch.Tensor, ...]:
    """
    Single iteration of extinction probability computation in log space.
    
    Log-space version for numerical stability with very small probabilities.
    
    Args:
        log_E: Current log extinction probabilities [S]
        s_C1_indices: Indices of left children for internal nodes
        s_C2_indices: Indices of right children for internal nodes  
        internal_mask: Boolean mask for internal species nodes [S]
        Recipients_mat: Transfer recipient probability matrix [S, S]
        log_pS, log_pD, log_pT, log_pL: Log event probabilities
        return_components: If True, return intermediate values
        
    Returns:
        Tuple of (new_log_E, log_E_s1, log_E_s2, log_E_bar)
        If return_components is False, only returns new_log_E
    """
    S = log_E.shape[0]
    device = log_E.device
    dtype = log_E.dtype
    
    # Gather extinction probabilities for children
    log_E_s1 = torch.full_like(log_E, float('-inf'))
    log_E_s2 = torch.full_like(log_E, float('-inf'))
    
    if internal_mask.any():
        log_E_s1_values = torch.index_select(log_E, 0, s_C1_indices)
        log_E_s2_values = torch.index_select(log_E, 0, s_C2_indices)
        
        log_E_s1[internal_mask] = log_E_s1_values
        log_E_s2[internal_mask] = log_E_s2_values
    
    # Compute event contributions in log space
    log_speciation = log_pS + log_E_s1 + log_E_s2
    log_duplication = log_pD + 2 * log_E
    
    # Transfer computation with numerical stability
    max_log_E = torch.max(log_E)
    # Use exp-normalize trick for matrix multiplication
    E_normalized = torch.exp(log_E - max_log_E)
    E_bar_normalized = torch.mv(Recipients_mat, E_normalized)
    log_E_bar = torch.log(E_bar_normalized) + max_log_E
    log_transfer = log_pT + log_E + log_E_bar
    
    # Combine using log-sum-exp
    contributions = torch.stack([
        log_speciation,
        log_duplication,
        log_transfer,
        log_pL.expand_as(log_speciation)
    ], dim=0)
    
    new_log_E = torch.logsumexp(contributions, dim=0)
    if return_components:
        return new_log_E, log_E_s1, log_E_s2, log_E_bar
    else:
        return new_log_E

