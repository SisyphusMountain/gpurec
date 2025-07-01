#!/usr/bin/env python3
"""
Detailed benchmarking script for the GPU-parallelized CCP reconciliation algorithm.
"""

import sys
sys.path.append('.')

from matmul_ale_ccp import *
import time
import torch

def benchmark_ccp_algorithm(species_tree_path, gene_tree_path, delta=1e-10, tau=0.05, lambda_param=1e-10, 
                           iters=5, device=None, dtype=torch.float64):
    """Benchmark each component of the CCP algorithm with detailed timing."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 GPU-Parallelized CCP Reconciliation Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Iterations: {iters}")
    print()
    
    # Overall timer
    total_start = time.time()
    
    # ==== PHASE 1: CCP CONSTRUCTION ====
    print("📊 PHASE 1: CCP Construction")
    ccp_start = time.time()
    
    ccp = build_ccp_from_single_tree(gene_tree_path)
    
    ccp_time = time.time() - ccp_start
    print(f"⏱️  CCP Construction: {ccp_time:.4f}s")
    print(f"   └─ {len(ccp.clades)} clades, {len(ccp.splits)} split groups")
    
    # Count total splits
    total_splits = sum(len(splits) for splits in ccp.splits.values())
    print(f"   └─ {total_splits} total splits")
    
    # ==== PHASE 2: SPECIES TREE HELPERS ====
    print("\n📊 PHASE 2: Species Tree Setup")
    species_start = time.time()
    
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    
    species_time = time.time() - species_start
    print(f"⏱️  Species Setup: {species_time:.4f}s")
    print(f"   └─ {species_helpers['S']} species nodes")
    
    # ==== PHASE 3: CLADE-SPECIES MAPPING ====
    print("\n📊 PHASE 3: Clade-Species Mapping")
    mapping_start = time.time()
    
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    
    mapping_time = time.time() - mapping_start
    print(f"⏱️  Mapping Construction: {mapping_time:.4f}s")
    print(f"   └─ {clade_species_map.shape[0]} × {clade_species_map.shape[1]} matrix")
    
    # ==== PHASE 4: CCP HELPERS (PARALLEL STRUCTURES) ====
    print("\n📊 PHASE 4: GPU Parallel Structures")
    ccp_helpers_start = time.time()
    
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    ccp_helpers_time = time.time() - ccp_helpers_start
    print(f"⏱️  Parallel Structures: {ccp_helpers_time:.4f}s")
    print(f"   └─ {len(ccp_helpers['split_parents'])} vectorized splits")
    
    # Calculate memory usage
    memory_mb = (ccp_helpers['split_parents'].numel() * 8 +  # int64
                 ccp_helpers['split_lefts'].numel() * 8 + 
                 ccp_helpers['split_rights'].numel() * 8 +
                 ccp_helpers['split_probs'].numel() * 8) / 1024**2  # float64
    print(f"   └─ ~{memory_mb:.2f} MB split data")
    
    # ==== PHASE 5: PARAMETER SETUP ====
    print("\n📊 PHASE 5: Parameter Setup")
    param_start = time.time()
    
    # Compute event probabilities
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    # Initialize matrices
    S = species_helpers["S"]
    C = len(ccp.clades)
    E = torch.zeros(S, dtype=dtype, device=device)
    Pi = torch.zeros((C, S), dtype=dtype, device=device)
    
    param_time = time.time() - param_start
    print(f"⏱️  Parameter Setup: {param_time:.4f}s")
    print(f"   └─ Pi matrix: {C} × {S} = {C*S:,} elements")
    print(f"   └─ Memory: ~{C*S*8/1024**2:.2f} MB")
    
    # ==== PHASE 6: EXTINCTION PROBABILITY COMPUTATION ====
    print("\n📊 PHASE 6: Extinction Probabilities")
    extinction_start = time.time()
    
    for iter_e in range(iters):
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    
    extinction_time = time.time() - extinction_start
    print(f"⏱️  Extinction Computation: {extinction_time:.4f}s")
    print(f"   └─ {iters} iterations, {extinction_time/iters*1000:.2f}ms per iteration")
    
    # ==== PHASE 7: LIKELIHOOD COMPUTATION (MAIN ALGORITHM) ====
    print("\n📊 PHASE 7: Likelihood Computation (Main Algorithm)")
    
    # Benchmark single iteration in detail
    print("   🔬 Single Iteration Breakdown:")
    
    # Time the Pi update function
    pi_update_start = time.time()
    Pi_new = Pi_update_ccp_parallel(Pi, ccp_helpers, species_helpers, clade_species_map, 
                                   E, Ebar, p_S, p_D, p_T)
    single_iteration_time = time.time() - pi_update_start
    print(f"      └─ Pi Update (1 iter): {single_iteration_time:.4f}s")
    
    # Now run full iterations
    likelihood_start = time.time()
    for iter_pi in range(iters):
        iter_start = time.time()
        Pi_new = Pi_update_ccp_parallel(Pi, ccp_helpers, species_helpers, clade_species_map, 
                                       E, Ebar, p_S, p_D, p_T)
        iter_time = time.time() - iter_start
        
        if iter_pi < 3:  # Show first few iterations
            print(f"      └─ Iteration {iter_pi+1}: {iter_time:.4f}s")
        
        # Check convergence
        if iter_pi > 0:
            diff = torch.abs(Pi_new - Pi).max()
            if diff < 1e-10:
                print(f"      └─ Converged at iteration {iter_pi+1}")
                break
        Pi = Pi_new
    
    likelihood_time = time.time() - likelihood_start
    print(f"⏱️  Total Likelihood: {likelihood_time:.4f}s")
    print(f"   └─ {iters} iterations, avg {likelihood_time/iters*1000:.2f}ms per iteration")
    
    # ==== PHASE 8: FINAL CALCULATIONS ====
    print("\n📊 PHASE 8: Final Calculations")
    final_start = time.time()
    
    # Calculate log-likelihood
    root_clade_id = get_root_clade_id(ccp)
    root_pi_sum = Pi[root_clade_id, :].sum()
    log_likelihood = torch.log(root_pi_sum)
    
    final_time = time.time() - final_start
    print(f"⏱️  Final Calculations: {final_time:.4f}s")
    print(f"   └─ Log-likelihood: {log_likelihood:.4f}")
    
    # ==== TOTAL TIMING SUMMARY ====
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("📈 PERFORMANCE SUMMARY")
    print("="*60)
    
    components = [
        ("CCP Construction", ccp_time),
        ("Species Setup", species_time), 
        ("Mapping Construction", mapping_time),
        ("Parallel Structures", ccp_helpers_time),
        ("Parameter Setup", param_time),
        ("Extinction Computation", extinction_time),
        ("Likelihood Computation", likelihood_time),
        ("Final Calculations", final_time)
    ]
    
    print(f"{'Component':<25} {'Time (s)':<10} {'% of Total':<12}")
    print("-" * 50)
    
    for name, comp_time in components:
        percentage = (comp_time / total_time) * 100
        print(f"{name:<25} {comp_time:<10.4f} {percentage:<12.1f}%")
    
    print("-" * 50)
    print(f"{'TOTAL':<25} {total_time:<10.4f} {'100.0':<12}%")
    
    # ==== THROUGHPUT ANALYSIS ====
    print("\n📊 THROUGHPUT ANALYSIS")
    print("-" * 30)
    
    # Calculate key metrics
    genes = len([c for c in ccp.clades if c.is_leaf()])
    species = species_helpers["S"]
    matrix_elements = C * S
    splits_processed = len(ccp_helpers['split_parents']) * iters
    
    print(f"Genes processed: {genes:,}")
    print(f"Species processed: {species:,}")
    print(f"Matrix elements: {matrix_elements:,}")
    print(f"Splits processed: {splits_processed:,}")
    print(f"Processing rate: {matrix_elements/total_time:,.0f} elements/sec")
    print(f"Split rate: {splits_processed/likelihood_time:,.0f} splits/sec")
    
    return {
        'total_time': total_time,
        'component_times': dict(components),
        'log_likelihood': float(log_likelihood),
        'genes': genes,
        'species': species,
        'matrix_elements': matrix_elements
    }

def main():
    print("🎯 Benchmarking GPU-Parallelized CCP Algorithm")
    print("Testing on massive phylogenetic trees...\n")
    
    # Test on the huge trees
    species_path = "big_test/sp.nwk"
    gene_path = "big_test/g.nwk"
    
    # Run benchmark
    results = benchmark_ccp_algorithm(
        species_path, gene_path,
        delta=1e-10, tau=0.05, lambda_param=1e-10,
        iters=5
    )
    
    print(f"\n🏆 Benchmark Complete!")
    print(f"   Total time: {results['total_time']:.2f}s")
    print(f"   Processing rate: {results['matrix_elements']/results['total_time']:,.0f} elements/sec")

if __name__ == "__main__":
    main()