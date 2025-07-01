#!/usr/bin/env python3
"""
Detailed benchmarking script for the log-space CCP implementation.
Times individual functions and components for performance analysis.
"""

import sys
sys.path.append('.')

import time
import torch
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree, build_species_helpers, 
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, E_step, Pi_update_ccp_log
)

def benchmark_log_ccp_detailed(species_tree_path, gene_tree_path, delta=1e-10, tau=0.05, lambda_param=1e-10, 
                              iters=5, device=None, dtype=torch.float64):
    """Detailed benchmarking of log-space CCP with component timing."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔬 Detailed Log-Space CCP Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Parameters: δ={delta}, τ={tau}, λ={lambda_param}")
    print(f"Iterations: {iters}")
    print()
    
    total_start = time.time()
    
    # ===== PHASE 1: CCP CONSTRUCTION =====
    print("📊 PHASE 1: CCP Construction")
    ccp_start = time.time()
    ccp = build_ccp_from_single_tree(gene_tree_path)
    ccp_time = time.time() - ccp_start
    print(f"   ⏱️  {ccp_time:.4f}s")
    print(f"   └─ {len(ccp.clades)} clades, {len(ccp.splits)} split groups")
    
    # ===== PHASE 2: SPECIES TREE SETUP =====
    print("\n🌳 PHASE 2: Species Tree Setup")
    species_start = time.time()
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    species_time = time.time() - species_start
    print(f"   ⏱️  {species_time:.4f}s")
    print(f"   └─ {species_helpers['S']} species nodes")
    
    # ===== PHASE 3: CLADE-SPECIES MAPPING =====
    print("\n🗺️  PHASE 3: Clade-Species Mapping")
    mapping_start = time.time()
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    mapping_time = time.time() - mapping_start
    print(f"   ⏱️  {mapping_time:.4f}s")
    print(f"   └─ {clade_species_map.shape[0]} × {clade_species_map.shape[1]} matrix")
    
    # ===== PHASE 4: CCP HELPERS =====
    print("\n⚡ PHASE 4: CCP Helpers (GPU Parallel Structures)")
    helpers_start = time.time()
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    helpers_time = time.time() - helpers_start
    print(f"   ⏱️  {helpers_time:.4f}s")
    print(f"   └─ {len(ccp_helpers['split_parents'])} vectorized splits")
    
    # ===== PHASE 5: PARAMETER SETUP =====
    print("\n🧮 PHASE 5: Parameter Setup")
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
    log_Pi = torch.full((C, S), float('-inf'), dtype=dtype, device=device)
    
    # Set leaf probabilities
    for c in range(C):
        clade = ccp.id_to_clade[c]
        if clade.is_leaf():
            mapped_species = torch.nonzero(clade_species_map[c] > 0, as_tuple=False).flatten()
            if len(mapped_species) > 0:
                log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=dtype))
                log_Pi[c, mapped_species] = log_prob
    
    param_time = time.time() - param_start
    print(f"   ⏱️  {param_time:.4f}s")
    print(f"   └─ log_Pi matrix: {C} × {S} = {C*S:,} elements")
    
    # ===== PHASE 6: EXTINCTION PROBABILITIES =====
    print("\n💀 PHASE 6: Extinction Probabilities")
    extinction_start = time.time()
    
    extinction_times = []
    for iter_e in range(iters):
        iter_start = time.time()
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
        iter_time = time.time() - iter_start
        extinction_times.append(iter_time)
    
    extinction_time = time.time() - extinction_start
    print(f"   ⏱️  {extinction_time:.4f}s total")
    print(f"   └─ {extinction_time/iters*1000:.2f}ms per iteration")
    print(f"   └─ Range: {min(extinction_times)*1000:.2f}-{max(extinction_times)*1000:.2f}ms")
    
    # ===== PHASE 7: LIKELIHOOD COMPUTATION (DETAILED) =====
    print("\n🧮 PHASE 7: Likelihood Computation (Pi Updates)")
    
    # Proper GPU timing methodology
    warmup_iters = 5
    timing_iters = 20
    
    print(f"   🔥 Warmup phase ({warmup_iters} iterations)...")
    
    # Warmup to ensure GPU is ready and any compilation is done
    for iter_warmup in range(warmup_iters):
        log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                       E, Ebar, p_S, p_D, p_T)
        log_Pi = log_Pi_new
        if iter_warmup == 0:
            print(f"      └─ Warmup iteration {iter_warmup+1} completed (potential compilation)")
        elif iter_warmup == warmup_iters - 1:
            print(f"      └─ Warmup iteration {iter_warmup+1} completed (GPU ready)")
    
    # Ensure GPU is synchronized before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print(f"   ⏱️  Timing phase ({timing_iters} iterations)...")
    
    # Time the full batch of iterations
    timing_start = time.time()
    
    for iter_pi in range(timing_iters):
        log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                       E, Ebar, p_S, p_D, p_T)
        log_Pi = log_Pi_new
        
        # Show progress for some iterations
        if iter_pi < 3:
            print(f"      └─ Timing iteration {iter_pi+1} executing...")
    
    # Ensure all GPU operations complete before measuring end time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_timing_time = time.time() - timing_start
    avg_per_iteration = total_timing_time / timing_iters
    
    print(f"      └─ Batch completed: {total_timing_time:.4f}s total")
    
    # For compatibility with existing code, create a list of average times
    pi_update_times = [avg_per_iteration] * timing_iters
    convergence_iter = timing_iters
    
    total_likelihood_time = total_timing_time  # Use the actual measured time
    print(f"\n   📊 Pi Update Summary:")
    print(f"      └─ Total time: {total_likelihood_time:.4f}s")
    print(f"      └─ Total iterations: {convergence_iter}")
    print(f"      └─ Average per iteration: {avg_per_iteration*1000:.2f}ms")
    print(f"      └─ Accurate GPU-synchronized timing")
    print(f"      └─ Warmup completed before measurement")
    
    # ===== PHASE 8: FINAL CALCULATIONS =====
    print("\n🎯 PHASE 8: Final Calculations")
    final_start = time.time()
    
    root_clade_id = get_root_clade_id(ccp)
    log_likelihood = torch.logsumexp(log_Pi[root_clade_id, :], dim=0)
    
    final_time = time.time() - final_start
    print(f"   ⏱️  {final_time:.4f}s")
    print(f"   └─ Log-likelihood: {log_likelihood:.6f}")
    
    # ===== TOTAL TIMING SUMMARY =====
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("📈 COMPONENT TIMING BREAKDOWN")
    print(f"{'='*60}")
    
    components = [
        ("CCP Construction", ccp_time),
        ("Species Setup", species_time),
        ("Mapping Construction", mapping_time), 
        ("CCP Helpers", helpers_time),
        ("Parameter Setup", param_time),
        ("Extinction Computation", extinction_time),
        ("Pi Updates (Total)", total_likelihood_time),
        ("Final Calculations", final_time)
    ]
    
    print(f"{'Component':<25} {'Time (s)':<10} {'Time (ms)':<12} {'% of Total':<12}")
    print("-" * 65)
    
    for name, comp_time in components:
        percentage = (comp_time / total_time) * 100
        print(f"{name:<25} {comp_time:<10.4f} {comp_time*1000:<12.2f} {percentage:<12.1f}%")
    
    print("-" * 65)
    print(f"{'TOTAL':<25} {total_time:<10.4f} {total_time*1000:<12.2f} {'100.0':<12}%")
    
    # ===== PI UPDATE DETAILED ANALYSIS =====
    print(f"\n📊 Pi_update_ccp_log DETAILED ANALYSIS")
    print("-" * 45)
    print(f"Function called {convergence_iter} times:")
    print(f"  • Total time: {total_likelihood_time:.4f}s")
    print(f"  • Average time: {avg_per_iteration*1000:.2f}ms per call")
    print(f"  • GPU-synchronized batch timing (accurate)")
    print(f"  • Time per matrix element: {avg_per_iteration/(C*S)*1e9:.2f}ns")
    print(f"  • Warmup completed before measurement")
    
    # Calculate throughput
    matrix_ops_per_iter = C * S  # Rough estimate of operations
    throughput = matrix_ops_per_iter / avg_per_iteration
    
    print(f"  • Estimated throughput: {throughput/1e6:.2f}M ops/sec")
    
    # ===== MEMORY ANALYSIS =====
    print(f"\n💾 MEMORY USAGE ANALYSIS")
    print("-" * 30)
    log_pi_memory = C * S * 8 / (1024**2)  # float64 = 8 bytes
    splits_memory = len(ccp_helpers['split_parents']) * 4 * 8 / (1024**2)  # 4 arrays × 8 bytes
    
    print(f"  • log_Pi matrix: {log_pi_memory:.2f} MB ({C:,} × {S:,} elements)")
    print(f"  • Split arrays: {splits_memory:.2f} MB ({len(ccp_helpers['split_parents']):,} splits)")
    print(f"  • Total estimated: {log_pi_memory + splits_memory:.2f} MB")
    
    return {
        'total_time': total_time,
        'pi_update_total_time': total_likelihood_time,
        'pi_update_avg_time': avg_per_iteration,
        'pi_update_times': [avg_per_iteration],  # Single accurate measurement
        'convergence_iter': convergence_iter,
        'log_likelihood': float(log_likelihood),
        'components': dict(components),
        'genes': len([c for c in ccp.clades if c.is_leaf()]),
        'species': S,
        'matrix_elements': C * S,
        'splits': len(ccp_helpers['split_parents'])
    }

def main():
    print("🎯 Detailed Log-Space CCP Performance Benchmark")
    print("=" * 70)
    
    # Test cases with different sizes
    test_cases = [
        ("test_trees_1/sp.nwk", "test_trees_1/g.nwk", "Small (8 genes)"),
        ("test_trees_2/sp.nwk", "test_trees_2/g.nwk", "Medium (11 genes)"),
        ("big_test/species_small30.nwk", "big_test/gene_small30.nwk", "Large Sample (41 genes)"),
    ]
    
    results = []
    
    for species_path, gene_path, name in test_cases:
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {name}")
        print(f"   Species: {species_path}")
        print(f"   Gene: {gene_path}")
        print(f"{'='*70}")
        
        try:
            result = benchmark_log_ccp_detailed(
                species_path, gene_path,
                delta=1e-10, tau=0.05, lambda_param=1e-10,
                iters=5
            )
            result['name'] = name
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            continue
    
    # ===== COMPARATIVE SUMMARY =====
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("📊 COMPARATIVE PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n🏃 Pi_update_ccp_log Performance Scaling:")
        print(f"{'Test Case':<20} {'Genes':<8} {'Elements':<12} {'Avg Time':<12} {'Throughput':<15}")
        print("-" * 75)
        
        for result in results:
            genes = result['genes']
            elements = result['matrix_elements']
            avg_time_ms = result['pi_update_avg_time'] * 1000
            throughput = elements / result['pi_update_avg_time'] / 1e6
            
            print(f"{result['name']:<20} {genes:<8} {elements:<12,} {avg_time_ms:<12.2f}ms {throughput:<15.2f}M/s")
        
        print(f"\n⚡ Overall Performance Characteristics:")
        for result in results:
            pi_update_pct = (result['pi_update_total_time'] / result['total_time']) * 100
            print(f"  • {result['name']}: Pi updates = {pi_update_pct:.1f}% of total time")

if __name__ == "__main__":
    main()