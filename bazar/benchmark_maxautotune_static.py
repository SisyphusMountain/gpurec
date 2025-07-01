#!/usr/bin/env python3
"""
Benchmark the completely static max-autotune compatible version.
"""

import sys
sys.path.append('.')
import time
import torch
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree, build_species_helpers, 
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, E_step
)
from matmul_ale_ccp_log_maxautotune import (
    Pi_update_ccp_log_maxautotune,
    Pi_update_ccp_log_maxautotune_compiled,
    Pi_update_ccp_log_reduce_overhead_compiled,
    Pi_update_ccp_log_default_compiled,
    create_prealloc_tensors
)

def benchmark_static_mode(mode_func, mode_name, species_tree_path, gene_tree_path, 
                         delta=1e-10, tau=0.05, lambda_param=1e-10,
                         warmup_iters=5, timing_iters=15, device=None, dtype=torch.float32):
    """Benchmark the static max-autotune compatible version."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Testing Static Max-Autotune: '{mode_name}'")
    print(f"{'='*60}")
    
    # Setup data
    print("📊 Setting up benchmark data...")
    ccp = build_ccp_from_single_tree(gene_tree_path)
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
    # Create pre-allocated constant tensors
    prealloc = create_prealloc_tensors(device, dtype)
    
    # Event probabilities
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
    
    # Compute extinction probabilities
    for iter_e in range(5):
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    
    print(f"   ✅ Setup complete: {C} clades × {S} species = {C*S:,} elements")
    
    # Warmup phase
    print(f"🔥 Warmup phase ({warmup_iters} iterations)...")
    warmup_start = time.time()
    
    try:
        for i in range(warmup_iters):
            log_Pi_new = mode_func(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                  E, Ebar, p_S, p_D, p_T,
                                  prealloc['neg_inf_tensor'], prealloc['zero_tensor'], 
                                  prealloc['one_tensor'], prealloc['eps_tensor'])
            log_Pi = log_Pi_new
            if i == 0:
                first_iter_time = time.time() - warmup_start
                print(f"   └─ First iteration: {first_iter_time*1000:.1f}ms (compilation)")
    except Exception as e:
        print(f"❌ Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    warmup_time = time.time() - warmup_start
    print(f"   └─ Warmup complete: {warmup_time:.2f}s total")
    
    # GPU synchronization before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timing phase
    print(f"⏱️  Timing phase ({timing_iters} iterations)...")
    timing_start = time.time()
    
    try:
        for i in range(timing_iters):
            log_Pi_new = mode_func(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                  E, Ebar, p_S, p_D, p_T,
                                  prealloc['neg_inf_tensor'], prealloc['zero_tensor'], 
                                  prealloc['one_tensor'], prealloc['eps_tensor'])
            log_Pi = log_Pi_new
    except Exception as e:
        print(f"❌ Timing failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # GPU synchronization after timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_timing = time.time() - timing_start
    avg_time = total_timing / timing_iters
    
    # Results
    throughput = (C * S) / avg_time / 1e6
    compilation_overhead = first_iter_time / avg_time if 'first_iter_time' in locals() else 0
    
    print(f"📈 Results for '{mode_name}':")
    print(f"   • Average time: {avg_time*1000:.3f}ms per iteration")
    print(f"   • Throughput: {throughput:.1f}M ops/sec")
    print(f"   • Compilation time: {first_iter_time*1000:.1f}ms" if 'first_iter_time' in locals() else "")
    print(f"   • Compilation overhead: {compilation_overhead:.1f}x" if compilation_overhead > 0 else "")
    
    # Verify correctness
    root_clade_id = get_root_clade_id(ccp)
    log_likelihood = torch.logsumexp(log_Pi[root_clade_id, :], dim=0)
    print(f"   • Log-likelihood: {log_likelihood:.6f} ({'finite' if torch.isfinite(log_likelihood) else 'non-finite'})")
    print()
    
    return {
        'mode': mode_name,
        'avg_time': avg_time,
        'throughput': throughput,
        'compilation_time': first_iter_time if 'first_iter_time' in locals() else None,
        'total_timing': total_timing,
        'matrix_elements': C * S,
        'log_likelihood': float(log_likelihood)
    }

def main():
    print("🎯 Static Max-Autotune Compatible Benchmark")
    print("=" * 70)
    print("Testing completely static version designed for CUDAGraphs compatibility")
    print("Key features:")
    print("  • No dynamic tensor creation")
    print("  • Pre-allocated constant tensors")  
    print("  • Manual logsumexp implementation")
    print("  • Matrix operations instead of scatter")
    print("  • Completely static computation graph")
    print()
    
    # Set high precision
    torch.set_float32_matmul_precision('high')
    print(f"Float32 matmul precision: {torch.get_float32_matmul_precision()}")
    print()
    
    # Test modes with static function
    modes = [
        (Pi_update_ccp_log_maxautotune, "uncompiled-static"),
        (Pi_update_ccp_log_default_compiled, "default-static"),
        (Pi_update_ccp_log_reduce_overhead_compiled, "reduce-overhead-static"),
        (Pi_update_ccp_log_maxautotune_compiled, "max-autotune-static")
    ]
    
    # Test cases
    test_cases = [
        ("test_trees_1/sp.nwk", "test_trees_1/g.nwk", "Small (8 genes)"),
        ("test_trees_200/sp.nwk", "test_trees_200/g.nwk", "Large (200 genes)")
    ]
    
    for species_path, gene_path, case_name in test_cases:
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {case_name}")
        print(f"   Species: {species_path}")
        print(f"   Gene: {gene_path}")
        print(f"{'='*70}")
        
        results = []
        
        for mode_func, mode_name in modes:
            try:
                result = benchmark_static_mode(
                    mode_func, mode_name, species_path, gene_path,
                    delta=1e-10, tau=0.05, lambda_param=1e-10,
                    warmup_iters=5, timing_iters=15,
                    dtype=torch.float32
                )
                if result:
                    results.append(result)
                    
            except Exception as e:
                print(f"❌ Error testing mode '{mode_name}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary for this test case
        if results:
            print(f"📊 SUMMARY FOR {case_name.upper()}")
            print(f"{'='*60}")
            print(f"{'Mode':<25} {'Time (ms)':<12} {'Throughput':<15} {'Speedup':<10} {'Status':<10}")
            print("-" * 85)
            
            # Find baseline (uncompiled)
            baseline_time = None
            for r in results:
                if 'uncompiled' in r['mode']:
                    baseline_time = r['avg_time']
                    break
            
            for result in results:
                time_ms = result['avg_time'] * 1000
                throughput = result['throughput']
                speedup = baseline_time / result['avg_time'] if baseline_time else 1.0
                
                # Check if advanced modes worked
                status = "✅" if result['mode'] in ['max-autotune-static', 'reduce-overhead-static'] else "📊"
                
                print(f"{result['mode']:<25} {time_ms:<12.3f} {throughput:<15.1f}M/s {speedup:<10.2f}x {status:<10}")
            
            # Highlight if max-autotune worked
            max_autotune_result = next((r for r in results if r['mode'] == 'max-autotune-static'), None)
            if max_autotune_result:
                print(f"\n🎉 MAX-AUTOTUNE SUCCESS!")
                print(f"   └─ {max_autotune_result['avg_time']*1000:.3f}ms per iteration")
                print(f"   └─ {max_autotune_result['throughput']:.1f}M ops/sec")
                if baseline_time:
                    improvement = (baseline_time - max_autotune_result['avg_time']) / baseline_time * 100
                    print(f"   └─ {improvement:.1f}% faster than uncompiled")
    
    print(f"\n{'='*70}")
    print("🎯 STATIC MAX-AUTOTUNE EXPERIMENT CONCLUSIONS")
    print(f"{'='*70}")
    print("This experiment tests whether a completely static computation graph")
    print("can enable CUDAGraphs-based optimization modes like max-autotune.")
    print("Success would demonstrate deep understanding of CUDAGraphs constraints.")

if __name__ == "__main__":
    main()