#!/usr/bin/env python3
"""
Utility functions for obtaining converged log_E and log_Pi data for testing.
This module provides clean interfaces for getting fully converged reconciliation
data that can be used across multiple test files.
"""

import torch
import math
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reconciliation.likelihood import E_step, Pi_step
from src.core.ccp import build_ccp_from_single_tree, build_ccp_helpers, build_clade_species_mapping
from src.core.tree_helpers import build_species_helpers


def get_converged_reconciliation_data(test_case="test_trees_1", dtype=torch.float64, device=None):
    """
    Get fully converged log_E and log_Pi data for testing.
    
    Args:
        test_case: Test case directory name (e.g., "test_trees_1")
        dtype: PyTorch dtype for computations
        device: PyTorch device (defaults to CPU)
        
    Returns:
        dict: Contains all converged data and helper structures
    """
    if device is None:
        device = torch.device('cpu')
        
    repo_root = Path(__file__).parent.parent.parent
    test_data_dir = repo_root / "tests" / "data" / test_case
    
    species_tree_path = test_data_dir / "sp.nwk"
    gene_tree_path = test_data_dir / "g.nwk"
    
    # Parameters (small values for quick convergence)
    delta = 1e-10
    tau = 1e-10 
    lambda_param = 1e-10
    
    print(f"🔧 Computing converged reconciliation data for {test_case}...")
    
    # Build structures
    ccp = build_ccp_from_single_tree(str(gene_tree_path))
    species_helpers = build_species_helpers(str(species_tree_path), device, dtype)
    
    # Build clade-species mapping
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    log_clade_species_map = torch.log(clade_species_map + 1e-45)
    log_clade_species_map[clade_species_map == 0] = float('-inf')
    
    # Build CCP helpers
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    split_probs = ccp_helpers['split_probs']
    zero_split_mask = (split_probs == 0)
    log_split_probs = torch.where(
        zero_split_mask,
        split_probs.new_full(split_probs.shape, float('-inf')),
        torch.log(split_probs)
    )
    
    log_ccp_helpers = {
        'ccp_leaves_mask': ccp_helpers['ccp_leaves_mask'],
        'C': ccp_helpers['C'],
        'ccp': ccp,
        'split_parents': ccp_helpers['split_parents'],
        'split_lefts': ccp_helpers['split_lefts'],
        'split_rights': ccp_helpers['split_rights'],
        'log_split_probs': log_split_probs,
    }
    
    # Event probabilities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S, p_D, p_T, p_L = 1.0/rates_sum, delta/rates_sum, tau/rates_sum, lambda_param/rates_sum
    
    log_pS = torch.log(torch.tensor(p_S, dtype=dtype, device=device))
    log_pD = torch.log(torch.tensor(p_D, dtype=dtype, device=device))
    log_pT = torch.log(torch.tensor(p_T, dtype=dtype, device=device))
    log_pL = torch.log(torch.tensor(p_L, dtype=dtype, device=device))
    
    S, C = species_helpers["S"], len(ccp.clades)
    
    # Converge log_E
    log_E = torch.full((S,), -math.log(2), dtype=dtype, device=device)
    for _ in range(50):
        log_E_old = log_E.clone()
        log_E_new, _, _, _ = E_step(
            log_E, species_helpers['s_C1_indexes'], species_helpers['s_C2_indexes'],
            species_helpers['sp_internal_mask'], species_helpers['Recipients_mat'],
            log_pS, log_pD, log_pT, log_pL
        )
        if torch.max(torch.abs(log_E_new - log_E_old)) < 1e-12:
            break
        log_E = log_E_new
    
    # Get derived quantities
    _, log_E_s1, log_E_s2, log_Ebar = E_step(
        log_E, species_helpers['s_C1_indexes'], species_helpers['s_C2_indexes'],
        species_helpers['sp_internal_mask'], species_helpers['Recipients_mat'],
        log_pS, log_pD, log_pT, log_pL
    )
    
    # Converge log_Pi  
    log_Pi = torch.full((C, S), -math.log(2), dtype=dtype, device=device)
    
    # Initialize leaves
    for c in range(C):
        clade = ccp.id_to_clade[c]
        if clade.is_leaf():
            mapped_species = torch.nonzero(torch.exp(log_clade_species_map[c]) > 1e-10, as_tuple=False).flatten()
            for s_idx in mapped_species:
                log_Pi[c, s_idx] = log_clade_species_map[c, s_idx]
    
    # Converge Pi
    for _ in range(100):
        log_Pi_old = log_Pi.clone()
        log_Pi = Pi_step(
            log_Pi, log_ccp_helpers, species_helpers, log_clade_species_map,
            log_E, log_Ebar, log_E_s1, log_E_s2, log_pS, log_pD, log_pT
        )
        if torch.max(torch.abs(log_Pi - log_Pi_old)) < 1e-12:
            break
    
    # Verify convergence quality
    finite_count = torch.isfinite(log_Pi).sum().item()
    total_count = log_Pi.numel()
    
    print(f"   Converged log_E: [{torch.min(log_E):.6f}, {torch.max(log_E):.6f}]")
    print(f"   Converged log_Pi: {finite_count}/{total_count} finite ({100*finite_count/total_count:.1f}%)")
    print(f"   log_Pi range: [{torch.min(log_Pi):.6f}, {torch.max(log_Pi):.6f}]")
    
    if finite_count != total_count:
        print(f"❌ WARNING: log_Pi is not 100% finite after convergence!")
        print(f"   -inf count: {torch.isinf(log_Pi & (log_Pi < 0)).sum().item()}")
        print(f"   +inf count: {torch.isinf(log_Pi & (log_Pi > 0)).sum().item()}")
        print(f"   NaN count: {torch.isnan(log_Pi).sum().item()}")
    
    return {
        'log_E': log_E,
        'log_E_s1': log_E_s1,
        'log_E_s2': log_E_s2,
        'log_Ebar': log_Ebar,
        'log_Pi': log_Pi,
        'log_ccp_helpers': log_ccp_helpers,
        'species_helpers': species_helpers,
        'log_clade_species_map': log_clade_species_map,
        'log_pS': log_pS,
        'log_pD': log_pD,
        'log_pT': log_pT,
        'log_pL': log_pL,
        'S': S,
        'C': C,
        'ccp': ccp
    }


def get_scatter_test_data(converged_data):
    """
    Extract ScatterLogSumExp test data from converged reconciliation data.
    
    Args:
        converged_data: Output from get_converged_reconciliation_data()
        
    Returns:
        dict: Data structures for ScatterLogSumExp testing
    """
    return {
        'split_parents': converged_data['log_ccp_helpers']['split_parents'],
        'ccp_leaves_mask': converged_data['log_ccp_helpers']['ccp_leaves_mask'],
        'C': converged_data['C'],
        'S': converged_data['S']
    }