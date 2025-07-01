#!/usr/bin/env python3
"""
Test float32 vs float64 performance and accuracy with high matmul precision.
"""

import sys
sys.path.append('.')
import torch
from benchmark_log_detailed import benchmark_log_ccp_detailed

# Set high precision matrix multiplication for float32
torch.set_float32_matmul_precision('high')
print(f'Float32 matmul precision set to: {torch.get_float32_matmul_precision()}')

print('🧪 Testing float32 vs float64 with high matmul precision')
print('='*70)

test_cases = [
    ('test_trees_1/sp.nwk', 'test_trees_1/g.nwk', '8 genes'),
    ('test_trees_200/sp.nwk', 'test_trees_200/g.nwk', '200 genes')
]

for species_path, gene_path, name in test_cases:
    print(f'\n📊 Testing {name}:')
    print('-' * 50)
    
    # Test float64 (original)
    print('🔬 Testing float64...')
    try:
        result_64 = benchmark_log_ccp_detailed(
            species_path, gene_path,
            delta=1e-10, tau=0.05, lambda_param=1e-10,
            iters=5, dtype=torch.float64
        )
        print(f'   ✅ float64: {result_64["log_likelihood"]:.8f}, avg time: {result_64["pi_update_avg_time"]*1000:.2f}ms')
        success_64 = True
    except Exception as e:
        print(f'   ❌ float64 failed: {e}')
        success_64 = False
    
    # Test float32
    print('\n🔬 Testing float32...')
    try:
        result_32 = benchmark_log_ccp_detailed(
            species_path, gene_path,
            delta=1e-10, tau=0.05, lambda_param=1e-10,
            iters=5, dtype=torch.float32
        )
        print(f'   ✅ float32: {result_32["log_likelihood"]:.8f}, avg time: {result_32["pi_update_avg_time"]*1000:.2f}ms')
        success_32 = True
    except Exception as e:
        print(f'   ❌ float32 failed: {e}')
        success_32 = False
    
    # Compare results
    if success_64 and success_32:
        diff = abs(result_64['log_likelihood'] - result_32['log_likelihood'])
        if result_64['log_likelihood'] != 0:
            rel_error = diff / abs(result_64['log_likelihood']) * 100
        else:
            rel_error = 0
        speedup = result_64['pi_update_avg_time'] / result_32['pi_update_avg_time']
        
        print(f'\n   📈 COMPARISON:')
        print(f'      Likelihood difference: {diff:.8f}')
        print(f'      Relative error: {rel_error:.6f}%')
        print(f'      Speedup (float32): {speedup:.2f}x')
        
        if rel_error < 0.01:
            print(f'      ✅ Excellent agreement (<0.01% error)')
        elif rel_error < 0.1:
            print(f'      ✅ Good agreement (<0.1% error)')
        elif rel_error < 1.0:
            print(f'      ⚠️  Moderate agreement (<1% error)')
        else:
            print(f'      ❌ Poor agreement (>{rel_error:.2f}% error)')

print(f'\n🎯 FINAL SUMMARY:')
print('='*50)
print('✅ High precision matmul enabled with torch.set_float32_matmul_precision("high")')
print('📊 Comparison tests completed')