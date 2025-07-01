#!/usr/bin/env python3
"""
Benchmark torch.compile() performance improvement for Pi_update_ccp_log.
Includes proper warmup to measure steady-state performance.
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

def benchmark_torch_compile(species_tree_path, gene_tree_path, delta=1e-10, tau=0.05, lambda_param=1e-10, 
                           warmup_iters=10, timing_iters=20, device=None, dtype=torch.float32):
    """
    Benchmark Pi_update_ccp_log with torch.compile, including proper warmup.
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Torch.compile() Performance Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Parameters: δ={delta}, τ={tau}, λ={lambda_param}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Timing iterations: {timing_iters}")
    print()
    
    # ===== SETUP PHASE =====
    print("📊 Setting up benchmark data...")
    
    # Build CCP and helpers
    ccp = build_ccp_from_single_tree(gene_tree_path)
    species_helpers = build_species_helpers(species_tree_path, device, dtype)
    clade_species_map = build_clade_species_mapping(ccp, species_helpers, device, dtype)
    ccp_helpers = build_ccp_helpers(ccp, device, dtype)
    
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
    
    # Compute extinction probabilities
    for iter_e in range(5):
        E_next, E_s1, E_s2, Ebar = E_step(E, species_helpers["s_C1"], species_helpers["s_C2"], 
                                          species_helpers["Recipients_mat"], p_S, p_D, p_T, p_L)
        E = E_next
    
    print(f"   ✅ Setup complete: {C} clades × {S} species = {C*S:,} elements")
    print()
    
    # ===== WARMUP PHASE =====
    print(f"🔥 Torch.compile() warmup phase ({warmup_iters} iterations)...")
    print("   Note: First few iterations will be slow due to compilation")
    
    warmup_times = []
    for i in range(warmup_iters):
        start_time = time.time()
        log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                       E, Ebar, p_S, p_D, p_T)
        elapsed = time.time() - start_time
        warmup_times.append(elapsed)
        
        if i < 5 or i == warmup_iters - 1:  # Show first 5 and last iteration
            print(f"   Warmup {i+1:2d}: {elapsed*1000:6.2f}ms")
        
        log_Pi = log_Pi_new  # Update for next iteration
    
    print(f"   🎯 Warmup complete. Compilation overhead: {warmup_times[0]*1000:.1f}ms → {warmup_times[-1]*1000:.1f}ms")
    print()
    
    # ===== TIMING PHASE =====
    print(f"⏱️  Steady-state timing phase ({timing_iters} iterations)...")
    
    # Clear GPU cache and sync for accurate timing
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    timing_start = time.time()
    timing_times = []
    
    for i in range(timing_iters):
        iter_start = time.time()
        log_Pi_new = Pi_update_ccp_log(log_Pi, ccp_helpers, species_helpers, clade_species_map, 
                                       E, Ebar, p_S, p_D, p_T)
        
        # Ensure GPU computation is complete
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        iter_time = time.time() - iter_start
        timing_times.append(iter_time)
        
        if i < 5 or i % 5 == 0:  # Show progress
            print(f"   Timing {i+1:2d}: {iter_time*1000:6.2f}ms")
        
        log_Pi = log_Pi_new
    
    total_timing = time.time() - timing_start
    
    # ===== PERFORMANCE ANALYSIS =====
    print(f"\n📈 TORCH.COMPILE() PERFORMANCE RESULTS")
    print(f"{'='*60}")
    
    # Statistics
    avg_time = sum(timing_times) / len(timing_times)
    median_time = sorted(timing_times)[len(timing_times)//2]
    min_time = min(timing_times)
    max_time = max(timing_times)
    std_time = (sum((t - avg_time)**2 for t in timing_times) / len(timing_times))**0.5
    
    print(f"📊 Timing Statistics (steady-state):")
    print(f"   • Average time: {avg_time*1000:.3f}ms per iteration")
    print(f"   • Median time:  {median_time*1000:.3f}ms per iteration")
    print(f"   • Min time:     {min_time*1000:.3f}ms per iteration")
    print(f"   • Max time:     {max_time*1000:.3f}ms per iteration")
    print(f"   • Std deviation: {std_time*1000:.3f}ms")
    print(f"   • Time per element: {avg_time/(C*S)*1e9:.2f}ns")
    
    # Throughput
    ops_per_iter = C * S
    throughput = ops_per_iter / avg_time
    print(f"   • Estimated throughput: {throughput/1e6:.2f}M ops/sec")
    
    # Compilation overhead analysis
    compilation_time = warmup_times[0]
    steady_state_time = avg_time
    compilation_overhead = compilation_time / steady_state_time
    
    print(f"\n🔥 Compilation Analysis:")
    print(f"   • First iteration (compilation): {compilation_time*1000:.2f}ms")
    print(f"   • Steady-state average: {steady_state_time*1000:.2f}ms")
    print(f"   • Compilation overhead: {compilation_overhead:.1f}x")
    print(f"   • Break-even point: ~{compilation_overhead:.0f} iterations")
    
    return {
        'avg_time': avg_time,
        'median_time': median_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'throughput': throughput,
        'compilation_time': compilation_time,
        'matrix_elements': C * S,
        'warmup_times': warmup_times,
        'timing_times': timing_times
    }

def main():
    print("🎯 Torch.compile() Performance Analysis")
    print("=" * 70)
    
    # Set high precision for fair comparison
    torch.set_float32_matmul_precision('high')
    print(f"Float32 matmul precision: {torch.get_float32_matmul_precision()}")
    print()
    
    # Test cases
    test_cases = [
        ("test_trees_1/sp.nwk", "test_trees_1/g.nwk", "Small (8 genes)"),
        ("test_trees_200/sp.nwk", "test_trees_200/g.nwk", "Large (200 genes)")
    ]
    
    results = []
    
    for species_path, gene_path, name in test_cases:
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {name}")
        print(f"   Species: {species_path}")
        print(f"   Gene: {gene_path}")
        print(f"{'='*70}")
        
        try:
            result = benchmark_torch_compile(
                species_path, gene_path,
                delta=1e-10, tau=0.05, lambda_param=1e-10,
                warmup_iters=10, timing_iters=20,
                dtype=torch.float32
            )
            result['name'] = name
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final comparison
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("📊 COMPARATIVE ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\n🏃 Pi_update_ccp_log Performance with torch.compile():")
        print(f"{'Test Case':<20} {'Elements':<12} {'Avg Time':<12} {'Throughput':<15}")
        print("-" * 70)
        
        for result in results:
            elements = result['matrix_elements']
            avg_time_ms = result['avg_time'] * 1000
            throughput = result['throughput'] / 1e6
            
            print(f"{result['name']:<20} {elements:<12,} {avg_time_ms:<12.3f}ms {throughput:<15.2f}M/s")
        
        print(f"\n🔥 Compilation Benefits:")
        for result in results:
            speedup = result['compilation_time'] / result['avg_time']
            print(f"  • {result['name']}: {speedup:.1f}x compilation overhead, "
                  f"break-even after ~{speedup:.0f} iterations")

if __name__ == "__main__":
    main()