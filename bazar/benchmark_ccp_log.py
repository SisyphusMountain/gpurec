#!/usr/bin/env python3
"""
Comprehensive benchmarking script comparing log-space and original CCP implementations.
"""

import sys
sys.path.append('.')

import time
import torch
from tabulate import tabulate

# Import both implementations
from matmul_ale_ccp_log import reconcile_ccp_log
from matmul_ale_ccp import build_ccp_from_single_tree, build_species_helpers, build_clade_species_mapping, build_ccp_helpers, get_root_clade_id, E_step, Pi_update_ccp_parallel

def benchmark_original_ccp(species_tree_path, gene_tree_path, delta=1e-10, tau=0.05, lambda_param=1e-10, 
                           iters=5, device=None, dtype=torch.float64):
    """Benchmark the original CCP implementation."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_start = time.time()
    
    # Phase 1: CCP Construction
    ccp_start = time.time()
    ccp = build_ccp_from_single_tree(gene_tree_path)
    ccp_time = time.time() - ccp_start
    
    # Phase 2: Species Tree Setup
    species_start = time.time()
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    species_time = time.time() - species_start
    
    # Phase 3: Clade-Species Mapping
    mapping_start = time.time()
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    mapping_time = time.time() - mapping_start
    
    # Phase 4: CCP Helpers
    helpers_start = time.time()
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    helpers_time = time.time() - helpers_start
    
    # Phase 5: Parameter Setup
    param_start = time.time()
    rates_sum = 1.0 + delta + tau + lambda_param
    p_S = 1.0 / rates_sum
    p_D = delta / rates_sum
    p_T = tau / rates_sum
    p_L = lambda_param / rates_sum
    
    S = species_helpers["S"]
    C = len(ccp.clades)
    E = torch.zeros(S, dtype=dtype, device=device)
    Pi = torch.zeros((C, S), dtype=dtype, device=device)
    param_time = time.time() - param_start
    
    # Phase 6: Extinction Probabilities
    extinction_start = time.time()
    for iter_e in range(iters):
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    extinction_time = time.time() - extinction_start
    
    # Phase 7: Likelihood Computation
    likelihood_start = time.time()
    for iter_pi in range(iters):
        Pi_new = Pi_update_ccp_parallel(Pi, ccp_helpers, species_helpers, clade_species_map, 
                                       E, Ebar, p_S, p_D, p_T)
        if iter_pi > 0:
            diff = torch.abs(Pi_new - Pi).max()
            if diff < 1e-10:
                break
        Pi = Pi_new
    likelihood_time = time.time() - likelihood_start
    
    # Phase 8: Final Calculations
    final_start = time.time()
    root_clade_id = get_root_clade_id(ccp)
    root_pi_sum = Pi[root_clade_id, :].sum()
    
    # Check for underflow
    if root_pi_sum <= 0:
        log_likelihood = float('-inf')
        underflow = True
    else:
        log_likelihood = torch.log(root_pi_sum)
        underflow = False
        
    final_time = time.time() - final_start
    
    total_time = time.time() - total_start
    
    return {
        'total_time': total_time,
        'component_times': {
            'CCP Construction': ccp_time,
            'Species Setup': species_time,
            'Mapping Construction': mapping_time,
            'Parallel Structures': helpers_time,
            'Parameter Setup': param_time,
            'Extinction Computation': extinction_time,
            'Likelihood Computation': likelihood_time,
            'Final Calculations': final_time
        },
        'log_likelihood': float(log_likelihood),
        'underflow': underflow,
        'genes': len([c for c in ccp.clades if c.is_leaf()]),
        'species': S,
        'matrix_elements': C * S
    }

def benchmark_comparison(species_tree_path, gene_tree_path, delta=1e-10, tau=0.05, lambda_param=1e-10, 
                        iters=5, device=None, dtype=torch.float64):
    """Compare original and log-space CCP implementations."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🔬 CCP Implementation Comparison")
    print(f"{'='*60}")
    print(f"Parameters: δ={delta}, τ={tau}, λ={lambda_param}")
    print(f"Device: {device}, Iterations: {iters}")
    print()
    
    # Benchmark Original Implementation
    print("🧮 Testing Original CCP Implementation...")
    try:
        original_start = time.time()
        original_results = benchmark_original_ccp(
            species_tree_path, gene_tree_path, delta, tau, lambda_param, iters, device, dtype
        )
        original_total = time.time() - original_start
        original_success = True
        print(f"   ✅ Success: {original_total:.4f}s, Log-likelihood: {original_results['log_likelihood']:.6f}")
        if original_results['underflow']:
            print(f"   ⚠️  Numerical underflow detected!")
        
    except Exception as e:
        original_success = False
        original_results = None
        print(f"   ❌ Failed: {e}")
    
    # Benchmark Log-Space Implementation  
    print("📊 Testing Log-Space CCP Implementation...")
    try:
        log_start = time.time()
        log_results = reconcile_ccp_log(
            species_tree_path, gene_tree_path, delta, tau, lambda_param, iters, device, dtype
        )
        log_total = time.time() - log_start
        log_success = True
        print(f"   ✅ Success: {log_total:.4f}s, Log-likelihood: {log_results['log_likelihood']:.6f}")
        
    except Exception as e:
        log_success = False
        log_results = None
        print(f"   ❌ Failed: {e}")
    
    print()
    
    # Detailed Comparison
    if original_success and log_success:
        print(f"📈 DETAILED COMPARISON")
        print(f"{'='*60}")
        
        # Performance comparison
        speedup = original_total / log_total if log_total > 0 else float('inf')
        
        table_data = [
            ["Implementation", "Time (s)", "Log-likelihood", "Status"],
            ["Original CCP", f"{original_total:.4f}", f"{original_results['log_likelihood']:.6f}", 
             "Underflow" if original_results.get('underflow', False) else "✅"],
            ["Log-Space CCP", f"{log_total:.4f}", f"{log_results['log_likelihood']:.6f}", "✅"],
            ["Speedup", f"{speedup:.2f}x" if speedup != float('inf') else "∞", 
             f"Δ = {abs(original_results['log_likelihood'] - log_results['log_likelihood']):.6f}", ""]
        ]
        
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
        print()
        
        # Component timing comparison (if original succeeded)
        if 'component_times' in original_results:
            print(f"⏱️  COMPONENT TIMING COMPARISON")
            print(f"{'-'*60}")
            
            comp_table = [["Component", "Original (s)", "Log-Space (s)", "Ratio"]]
            
            # Note: Log-space doesn't provide component breakdown, so we'll show total vs total
            comp_table.append([
                "Total Runtime",
                f"{original_total:.4f}",
                f"{log_total:.4f}", 
                f"{original_total/log_total:.2f}x" if log_total > 0 else "∞"
            ])
            
            print(tabulate(comp_table, headers="firstrow", tablefmt="grid"))
            print()
        
        # Accuracy comparison
        if not original_results.get('underflow', False):
            diff = abs(original_results['log_likelihood'] - log_results['log_likelihood'])
            rel_error = diff / abs(original_results['log_likelihood']) * 100 if original_results['log_likelihood'] != 0 else 0
            
            print(f"🎯 ACCURACY ANALYSIS")
            print(f"{'-'*30}")
            print(f"Absolute difference: {diff:.6f}")
            print(f"Relative error: {rel_error:.4f}%")
            
            if rel_error < 0.1:
                print("✅ Excellent agreement")
            elif rel_error < 1.0:
                print("✅ Good agreement") 
            elif rel_error < 5.0:
                print("⚠️  Moderate agreement")
            else:
                print("❌ Poor agreement")
        else:
            print(f"🎯 NUMERICAL STABILITY")
            print(f"{'-'*30}")
            print("✅ Log-space prevents underflow that affects original implementation")
            
    elif log_success:
        print(f"📈 LOG-SPACE CCP RESULTS")
        print(f"{'='*60}")
        print(f"Runtime: {log_total:.4f}s")
        print(f"Log-likelihood: {log_results['log_likelihood']:.6f}")
        print("✅ Log-space implementation handles cases where original fails")
        
    return {
        'original_success': original_success,
        'log_success': log_success,
        'original_results': original_results,
        'log_results': log_results,
        'original_time': original_total if original_success else None,
        'log_time': log_total if log_success else None
    }

def main():
    print("🎯 Comprehensive CCP Implementation Benchmark")
    print("=" * 70)
    
    # Test cases with different sizes
    test_cases = [
        ("test_trees_1/sp.nwk", "test_trees_1/g.nwk", "Small (8 genes)"),
        ("test_trees_2/sp.nwk", "test_trees_2/g.nwk", "Medium (11 genes)"),
        ("big_test/species_small30.nwk", "big_test/gene_small30.nwk", "Large Sample (41 genes)"),
        ("big_test/sp.nwk", "big_test/g.nwk", "Massive (5,573 genes)")
    ]
    
    all_results = []
    
    for species_path, gene_path, name in test_cases:
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {name}")
        print(f"   Species: {species_path}")
        print(f"   Gene: {gene_path}")
        print(f"{'='*70}")
        
        try:
            results = benchmark_comparison(
                species_path, gene_path,
                delta=1e-10, tau=0.05, lambda_param=1e-10,
                iters=5
            )
            results['name'] = name
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            continue
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}")
    
    summary_table = [["Test Case", "Original", "Log-Space", "Speedup", "Agreement"]]
    
    for result in all_results:
        name = result['name']
        
        if result['original_success'] and result['log_success']:
            speedup = result['original_time'] / result['log_time']
            diff = abs(result['original_results']['log_likelihood'] - result['log_results']['log_likelihood'])
            rel_error = diff / abs(result['original_results']['log_likelihood']) * 100 if result['original_results']['log_likelihood'] != 0 else 0
            
            if result['original_results'].get('underflow', False):
                agreement = "Underflow→Stable"
            elif rel_error < 1.0:
                agreement = "Excellent"
            else:
                agreement = f"{rel_error:.2f}% error"
                
            summary_table.append([
                name,
                f"{result['original_time']:.2f}s" if not result['original_results'].get('underflow', False) else "Underflow",
                f"{result['log_time']:.2f}s",
                f"{speedup:.2f}x",
                agreement
            ])
        elif result['log_success']:
            summary_table.append([
                name,
                "Failed",
                f"{result['log_time']:.2f}s", 
                "N/A",
                "Log-space only"
            ])
        else:
            summary_table.append([
                name,
                "Failed" if not result['original_success'] else "Success",
                "Failed",
                "N/A", 
                "Both failed"
            ])
    
    print(tabulate(summary_table, headers="firstrow", tablefmt="grid"))
    
    print(f"\n🏆 Key Findings:")
    success_count = sum(1 for r in all_results if r['log_success'])
    underflow_prevented = sum(1 for r in all_results if r['original_success'] and r['original_results'].get('underflow', False) and r['log_success'])
    
    print(f"   • Log-space CCP succeeded on {success_count}/{len(all_results)} test cases")
    if underflow_prevented > 0:
        print(f"   • Prevented numerical underflow in {underflow_prevented} cases")
    print(f"   • Enables analysis of previously intractable phylogenetic problems")

if __name__ == "__main__":
    main()