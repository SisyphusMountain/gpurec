#!/usr/bin/env python3
"""
Benchmark different torch.compile() modes for Pi_update_ccp_log.
Tests: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs
"""

import sys
sys.path.append('.')
import time
import torch
import importlib

def create_compiled_function(mode):
    """Create a fresh compiled version of Pi_update_ccp_log with specified mode."""
    # Clear compilation cache
    torch._dynamo.reset()
    
    # Re-import the module to get a fresh function
    if 'matmul_ale_ccp_log' in sys.modules:
        del sys.modules['matmul_ale_ccp_log']
    
    # Import and compile with the specified mode
    from matmul_ale_ccp_log import (
        build_ccp_from_single_tree, build_species_helpers, 
        build_clade_species_mapping, build_ccp_helpers,
        get_root_clade_id, E_step
    )
    
    # Remove existing decorator and recompile with new mode
    import matmul_ale_ccp_log
    original_func = matmul_ale_ccp_log.Pi_update_ccp_log
    
    # Remove the existing compile decorator if it exists
    if hasattr(original_func, '_dynamo_compile'):
        original_func = original_func.__wrapped__
    
    # Compile with new mode
    compiled_func = torch.compile(original_func, mode=mode)
    
    return compiled_func, build_ccp_from_single_tree, build_species_helpers, build_clade_species_mapping, build_ccp_helpers, get_root_clade_id, E_step

def benchmark_compile_mode(mode, species_tree_path, gene_tree_path, 
                          delta=1e-10, tau=0.05, lambda_param=1e-10,
                          warmup_iters=5, timing_iters=15, device=None, dtype=torch.float32):
    """Benchmark a specific torch.compile() mode."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Testing torch.compile(mode='{mode}')")
    print(f"{'='*60}")
    
    # Create compiled function with specified mode
    try:
        Pi_update_func, build_ccp, build_species, build_mapping, build_helpers, get_root, E_step = create_compiled_function(mode)
    except Exception as e:
        print(f"❌ Failed to create compiled function: {e}")
        return None
    
    # Setup data
    print("📊 Setting up benchmark data...")
    ccp = build_ccp(gene_tree_path)
    species_helpers = build_species(species_tree_path, device, dtype)
    clade_species_map = build_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_helpers(ccp, device, dtype)
    
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
    
    for i in range(warmup_iters):
        log_Pi_new = Pi_update_func(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                   E, Ebar, p_S, p_D, p_T)
        log_Pi = log_Pi_new
        if i == 0:
            first_iter_time = time.time() - warmup_start
            print(f"   └─ First iteration: {first_iter_time*1000:.1f}ms (compilation)")
    
    warmup_time = time.time() - warmup_start
    print(f"   └─ Warmup complete: {warmup_time:.2f}s total")
    
    # GPU synchronization before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timing phase
    print(f"⏱️  Timing phase ({timing_iters} iterations)...")
    timing_start = time.time()
    
    for i in range(timing_iters):
        log_Pi_new = Pi_update_func(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                   E, Ebar, p_S, p_D, p_T)
        log_Pi = log_Pi_new
    
    # GPU synchronization after timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_timing = time.time() - timing_start
    avg_time = total_timing / timing_iters
    
    # Results
    throughput = (C * S) / avg_time / 1e6
    compilation_overhead = first_iter_time / avg_time if 'first_iter_time' in locals() else 0
    
    print(f"📈 Results for mode='{mode}':")
    print(f"   • Average time: {avg_time*1000:.3f}ms per iteration")
    print(f"   • Throughput: {throughput:.1f}M ops/sec")
    print(f"   • Compilation time: {first_iter_time*1000:.1f}ms" if 'first_iter_time' in locals() else "")
    print(f"   • Compilation overhead: {compilation_overhead:.1f}x" if compilation_overhead > 0 else "")
    print()
    
    return {
        'mode': mode,
        'avg_time': avg_time,
        'throughput': throughput,
        'compilation_time': first_iter_time if 'first_iter_time' in locals() else None,
        'total_timing': total_timing,
        'matrix_elements': C * S
    }

def main():
    print("🎯 Torch.compile() Mode Comparison Benchmark")
    print("=" * 70)
    
    # Set high precision
    torch.set_float32_matmul_precision('high')
    print(f"Float32 matmul precision: {torch.get_float32_matmul_precision()}")
    print()
    
    # Test modes
    modes = [
        "default",
        "reduce-overhead", 
        "max-autotune",
        "max-autotune-no-cudagraphs"
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
        
        for mode in modes:
            try:
                result = benchmark_compile_mode(
                    mode, species_path, gene_path,
                    delta=1e-10, tau=0.05, lambda_param=1e-10,
                    warmup_iters=5, timing_iters=15,
                    dtype=torch.float32
                )
                if result:
                    results.append(result)
                    
            except Exception as e:
                print(f"❌ Error testing mode '{mode}': {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary for this test case
        if results:
            print(f"📊 SUMMARY FOR {case_name.upper()}")
            print(f"{'='*60}")
            print(f"{'Mode':<25} {'Time (ms)':<12} {'Throughput':<15} {'Speedup':<10}")
            print("-" * 70)
            
            # Find baseline (default mode)
            baseline_time = None
            for r in results:
                if r['mode'] == 'default':
                    baseline_time = r['avg_time']
                    break
            
            for result in results:
                time_ms = result['avg_time'] * 1000
                throughput = result['throughput']
                speedup = baseline_time / result['avg_time'] if baseline_time else 1.0
                
                print(f"{result['mode']:<25} {time_ms:<12.3f} {throughput:<15.1f}M/s {speedup:<10.2f}x")
            
            # Find best mode
            best_result = min(results, key=lambda x: x['avg_time'])
            print(f"\n🏆 Best mode for {case_name}: '{best_result['mode']}'")
            print(f"   └─ {best_result['avg_time']*1000:.3f}ms per iteration")
            print(f"   └─ {best_result['throughput']:.1f}M ops/sec")
            
            if baseline_time:
                improvement = (baseline_time - best_result['avg_time']) / baseline_time * 100
                print(f"   └─ {improvement:.1f}% faster than default mode")
    
    print(f"\n{'='*70}")
    print("🎯 TORCH.COMPILE() MODE RECOMMENDATIONS")
    print(f"{'='*70}")
    print("Based on the benchmark results above:")
    print("• Compare time per iteration across modes")
    print("• Consider compilation time vs runtime performance tradeoff") 
    print("• Choose optimal mode for your specific use case")

if __name__ == "__main__":
    main()