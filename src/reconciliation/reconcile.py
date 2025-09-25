"""
Main reconciliation function for CCP reconciliation.

This module implements the complete reconciliation workflow including
your fixes for numerical stability.
"""

import torch
import math
from typing import Dict, Optional

from ..core.ccp import build_ccp_from_single_tree, get_root_clade_id, build_ccp_helpers, build_clade_species_mapping
from ..core.tree_helpers import build_species_helpers
from ..core.likelihood import E_fixed_point, Pi_fixed_point, compute_log_likelihood


def setup_fixed_points(species_tree_path: str, 
                gene_tree_path: str, 
                delta: float = 1e-10, 
                tau: float = 0.05, 
                lambda_param: float = 1e-10,
                max_iters_E: int = 100, 
                max_iters_Pi: int = 100,
                tol_E: float = 1e-9,
                tol_Pi: float = 1e-9,
                device: Optional[torch.device] = None, 
                dtype: torch.dtype = torch.float64, 
                debug: bool = False,
                use_theta: bool = False,
                theta: Optional[torch.Tensor] = None,
                use_triton: bool = True,
                compare_triton: bool = False) -> Dict:
    """
    Log-space CCP reconciliation with numerical stability fixes.
    
    Args:
        species_tree_path: Path to species tree file (.nwk)
        gene_tree_path: Path to gene tree file (.nwk)
        delta: Duplication rate parameter (ignored if use_theta=True)
        tau: Transfer rate parameter (ignored if use_theta=True)
        lambda_param: Loss rate parameter (ignored if use_theta=True)
        max_iters_E: Maximum E iterations for convergence
        max_iters_Pi: Maximum Pi iterations for convergence
        tol_E: Tolerance for E convergence
        tol_Pi: Tolerance for Pi convergence
        device: PyTorch device (auto-detected if None)
        dtype: PyTorch data type
        debug: Enable debug output
        use_theta: If True, use theta parameterization instead of individual rates
        theta: 3-element tensor [log_delta, log_tau, log_lambda] (required if use_theta=True)
        use_triton: If True, use Triton LSE kernels when available (default: True)
        
    Returns:
        Dictionary with:
            - log_likelihood: Final log-likelihood value
            - Pi: Likelihood matrix [C, S]
            - ccp: CCP container
            - E: Extinction probabilities [S]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if debug:
        print(f"Parameters: δ={delta}, τ={tau}, λ={lambda_param}")
    
    # Build CCP from gene tree
    ccp = build_ccp_from_single_tree(gene_tree_path, debug=debug)
    
    # Build species helpers
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    
    # Build clade-species mapping
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    log_clade_species_map = torch.log(clade_species_map + 1e-45)
    log_clade_species_map[clade_species_map == 0] = float('-inf')
    
    # Build CCP helpers for GPU parallelization
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)

    # Determine parameter source
    if use_theta:
        if theta is None:
            raise ValueError("theta parameter required when use_theta=True")
        param_tensor = theta
        # Extract values for debug display
        if debug:
            exp_theta = torch.exp(theta)
            delta_val, tau_val, lambda_val = exp_theta[0].item(), exp_theta[1].item(), exp_theta[2].item()
            print(f"Theta parameters: δ={delta_val:.6f}, τ={tau_val:.6f}, λ={lambda_val:.6f}")
    else:
        # Create theta tensor from individual parameters
        param_tensor = torch.tensor([math.log(max(delta, 1e-10)), 
                                   math.log(max(tau, 1e-10)), 
                                   math.log(max(lambda_param, 1e-10))], 
                                  dtype=dtype, device=device)
        if debug:
            print(f"Individual parameters: δ={delta}, τ={tau}, λ={lambda_param}")
    
    # Initialize matrices
    result_E = E_fixed_point(species_helpers=species_helpers,
                          theta=param_tensor,
                          max_iters=max_iters_E,
                          tolerance=tol_E,
                          return_components=True,
                          warm_start_E=None,
                          use_triton=use_triton,
                          compare_triton=compare_triton)
    E = result_E['E']
    E_s1 = result_E['E_s1']
    E_s2 = result_E['E_s2']
    Ebar = result_E['E_bar']
    
    # === LIKELIHOOD COMPUTATION ===
    result_Pi = Pi_fixed_point(ccp_helpers=ccp_helpers,
                  species_helpers=species_helpers,
                  clade_species_map=log_clade_species_map,
                  E=E,
                  Ebar=Ebar,
                  E_s1=E_s1,
                  E_s2=E_s2,
                  theta=param_tensor,
                  max_iters=max_iters_Pi,
                  tolerance=tol_Pi,
                  warm_start_Pi=None,
                  use_triton=use_triton,
                  compare_triton=compare_triton)
    Pi = result_Pi['Pi']
    
    # === COMPUTE LOG-LIKELIHOOD ===
    root_clade_id = get_root_clade_id(ccp, debug=debug)
    
    
    # Compute log-likelihood: log(sum(Pi[root, :]))
    log_likelihood = compute_log_likelihood(Pi, root_clade_id)
    if debug:
        print(f"📊 Log-likelihood: {log_likelihood:.6f}")
    
    
    if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
        if debug:
            print("⚠️  WARNING: Numerical instability detected!")
    else:
        if debug:
            print("✅ No numerical instability detected")
    
    return {
        'log_likelihood': float(log_likelihood),
        'Pi': Pi,
        'ccp': ccp,
        'E': E,
        'Ebar': Ebar,
        'E_s1': E_s1,
        'E_s2': E_s2,
        'species_helpers': species_helpers,
        'clade_species_map': log_clade_species_map,
        'ccp_helpers': ccp_helpers,
        'theta': param_tensor,
        'root_clade_id': root_clade_id,
    }
