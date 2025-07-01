#!/usr/bin/env python3
"""
Test script to compare CCP implementation with AleRax results.
"""

import subprocess
import os
import re
import torch
from matmul_ale_ccp_corrected import main_ccp
import argparse


def extract_alerax_likelihood(output_dir):
    """Extract log-likelihood from AleRax output."""
    # Look for the reconciliation likelihood in the output
    results_file = os.path.join(output_dir, "results.txt")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            content = f.read()
            # Look for pattern like "Reconciliation likelihood: -XXX.XXX"
            match = re.search(r"Reconciliation log-likelihood:\s*(-?\d+\.?\d*)", content)
            if match:
                return float(match.group(1))
    
    # Alternative: check log file
    log_files = [f for f in os.listdir(output_dir) if f.endswith('.log')]
    for log_file in log_files:
        with open(os.path.join(output_dir, log_file), 'r') as f:
            content = f.read()
            # Look for final likelihood
            matches = re.findall(r"LL=(-?\d+\.?\d*)", content)
            if matches:
                return float(matches[-1])
    
    return None


def extract_alerax_rates(output_dir):
    """Extract inferred D, T, L rates from AleRax output."""
    params_file = os.path.join(output_dir, "model_parameters", "model_parameters.txt")
    if not os.path.exists(params_file):
        return None, None, None
    
    with open(params_file, 'r') as f:
        lines = f.readlines()
        
    # Skip header and look for rates
    for line in lines[1:]:  # Skip header
        parts = line.strip().split()
        if len(parts) >= 4:
            # Format: branch_id D T L
            d_rate = float(parts[1])
            t_rate = float(parts[2])
            l_rate = float(parts[3])
            return d_rate, t_rate, l_rate
    
    return None, None, None


def run_alerax(species_tree, gene_tree, output_dir, delta=None, tau=None, lambda_param=None):
    """Run AleRax with specified parameters."""
    # First, create .ale file using ALEobserve
    gene_base = os.path.basename(gene_tree).replace('.nwk', '')
    ale_file = f"{gene_base}.ale"
    
    if not os.path.exists(ale_file):
        print(f"Creating ALE file with ALEobserve...")
        cmd = ["ALEobserve", gene_tree]
        subprocess.run(cmd, check=True)
    
    # Create families file
    families_file = "families.txt"
    with open(families_file, 'w') as f:
        f.write(f"{ale_file}\n")
    
    # Run AleRax
    cmd = [
        "alerax",
        "-f", families_file,
        "-s", species_tree,
        "-p", output_dir,
        "--gene-tree-samples", "0",
        "--species-tree-search", "SKIP"
    ]
    
    # If rates are specified, add them
    if delta is not None and tau is not None and lambda_param is not None:
        cmd.extend([
            "--per-family-rates", "false",
            "--delta", str(delta),
            "--tau", str(tau),
            "--lambda", str(lambda_param),
            "--optimize-rates", "false"
        ])
    
    print(f"Running AleRax: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"AleRax failed with error:\n{result.stderr}")
        return None
    
    # Extract likelihood
    likelihood = extract_alerax_likelihood(output_dir)
    return likelihood


def compare_implementations(species_tree, gene_tree, delta, tau, lambda_param):
    """Compare our implementation with AleRax."""
    print("="*80)
    print(f"Testing with:")
    print(f"  Species tree: {species_tree}")
    print(f"  Gene tree: {gene_tree}")
    print(f"  Parameters: delta={delta}, tau={tau}, lambda={lambda_param}")
    print("="*80)
    
    # Run our implementation
    print("\n--- Running our CCP implementation ---")
    result = main_ccp(species_tree, gene_tree, delta, tau, lambda_param, iters=100)
    our_ll = result['log_likelihood']
    print(f"\nOur log-likelihood: {our_ll:.6f}")
    
    # Run AleRax
    print("\n--- Running AleRax ---")
    output_dir = "alerax_output"
    alerax_ll = run_alerax(species_tree, gene_tree, output_dir, delta, tau, lambda_param)
    
    if alerax_ll is not None:
        print(f"\nAleRax log-likelihood: {alerax_ll:.6f}")
        diff = abs(our_ll - alerax_ll)
        print(f"\nDifference: {diff:.6f}")
        
        if diff < 0.1:
            print("✓ Results match closely!")
        else:
            print("✗ Results differ significantly")
            
            # Try to understand the difference
            print("\nDebugging information:")
            print(f"  Our Pi root sum: {result['Pi'][result['ccp'].clade_to_id[max(result['ccp'].clades, key=lambda c: c.size)]].sum():.6e}")
            print(f"  Our E mean: {result['E'].mean():.6f}")
    else:
        print("Failed to extract AleRax likelihood")
    
    return our_ll, alerax_ll


def main():
    parser = argparse.ArgumentParser(description="Compare CCP implementation with AleRax")
    parser.add_argument("--test", type=int, default=1, help="Test case number (1 or 2)")
    args = parser.parse_args()
    
    # Test cases
    if args.test == 1:
        species_tree = "test_trees_1/sp.nwk"
        gene_tree = "test_trees_1/g.nwk"
    else:
        species_tree = "test_trees_2/sp.nwk"
        gene_tree = "test_trees_2/g.nwk"
    
    # Test with different parameter settings
    test_params = [
        (0.1, 0.1, 0.1),  # Equal rates
        (1.0, 0.1, 0.1),  # High duplication
        (0.1, 1.0, 0.1),  # High transfer
        (0.1, 0.1, 1.0),  # High loss
    ]
    
    for delta, tau, lambda_param in test_params:
        compare_implementations(species_tree, gene_tree, delta, tau, lambda_param)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()