#!/usr/bin/env python3
"""
Benchmark the CCP algorithm on smaller test cases for comparison.
"""

import sys
sys.path.append('.')

from benchmark_ccp import benchmark_ccp_algorithm

def main():
    print("🎯 Benchmarking GPU-Parallelized CCP Algorithm - Small Trees")
    print("=" * 60)
    
    # Test cases with different sizes
    test_cases = [
        ("test_trees_1/sp.nwk", "test_trees_1/g.nwk", "Original Small (8 genes)"),
        ("test_trees_2/sp.nwk", "test_trees_2/g.nwk", "Medium (98 genes)"),
        ("big_test/species_small30.nwk", "big_test/gene_small30.nwk", "Small 30-node sample")
    ]
    
    results = []
    
    for species_path, gene_path, name in test_cases:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {name}")
        print(f"   Species: {species_path}")
        print(f"   Gene: {gene_path}")
        print(f"{'='*60}")
        
        try:
            result = benchmark_ccp_algorithm(
                species_path, gene_path,
                delta=1e-10, tau=0.05, lambda_param=1e-10,
                iters=5
            )
            results.append((name, result))
            
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            continue
    
    # Comparison summary
    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Test Case':<25} {'Genes':<8} {'Species':<8} {'Time (s)':<10} {'Rate (elem/s)':<15}")
    print("-" * 80)
    
    for name, result in results:
        rate = result['matrix_elements'] / result['total_time']
        print(f"{name:<25} {result['genes']:<8} {result['species']:<8} {result['total_time']:<10.2f} {rate:<15,.0f}")

if __name__ == "__main__":
    main()