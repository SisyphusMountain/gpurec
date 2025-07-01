#!/usr/bin/env python3
"""
Test script for efficient CCP-based reconciliation.
Compares matrix-based implementation against AleRax results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from matmul_ale_ccp import main_ccp
import time

def test_against_alerax():
    """Test against AleRax reference log-likelihoods."""
    
    # Test cases with known AleRax results
    test_cases = [
        {
            'name': 'test_trees_1',
            'species': 'test_trees_1/sp.nwk',
            'gene': 'test_trees_1/g.nwk',
            'delta': 1e-10,
            'tau': 1e-10,
            'lambda': 1e-10,
            'alerax_loglik': -2.56495
        },
        {
            'name': 'test_trees_2', 
            'species': 'test_trees_2/sp.nwk',
            'gene': 'test_trees_2/g.nwk',
            'delta': 1e-10,
            'tau': 0.0517229,
            'lambda': 1e-10,
            'alerax_loglik': -8.72486
        },
        {
            'name': 'test_trees_3',
            'species': 'test_trees_3/sp.nwk', 
            'gene': 'test_trees_3/g.nwk',
            'delta': 0.0555539,
            'tau': 1e-10,
            'lambda': 1e-10,
            'alerax_loglik': -6.75086
        }
    ]
    
    print("=== Testing CCP Implementation Against AleRax ===\n")
    
    for i, case in enumerate(test_cases):
        print(f"Test {i+1}: {case['name']}")
        print(f"Parameters: delta={case['delta']}, tau={case['tau']}, lambda={case['lambda']}")
        print(f"Expected log-likelihood: {case['alerax_loglik']}")
        
        # Test efficient implementation
        print("\n--- Testing Efficient Implementation ---")
        start_time = time.time()
        try:
            result_efficient = main_ccp(
                case['species'], case['gene'],
                delta=case['delta'], tau=case['tau'], lambda_param=case['lambda'],
                iters=100, use_efficient=True
            )
            efficient_time = time.time() - start_time
            efficient_loglik = result_efficient['log_likelihood']
            print(f"Efficient log-likelihood: {efficient_loglik:.6f}")
            print(f"Efficient time: {efficient_time:.3f}s")
            efficient_diff = abs(efficient_loglik - case['alerax_loglik'])
            print(f"Difference from AleRax: {efficient_diff:.6f}")
        except Exception as e:
            print(f"Efficient implementation failed: {e}")
            efficient_loglik = None
            efficient_time = None
        
        # Test loop-based implementation for comparison
        print("\n--- Testing Loop-based Implementation ---")
        start_time = time.time()
        try:
            result_loop = main_ccp(
                case['species'], case['gene'],
                delta=case['delta'], tau=case['tau'], lambda_param=case['lambda'],
                iters=100, use_efficient=False
            )
            loop_time = time.time() - start_time
            loop_loglik = result_loop['log_likelihood']
            print(f"Loop-based log-likelihood: {loop_loglik:.6f}")
            print(f"Loop-based time: {loop_time:.3f}s")
            loop_diff = abs(loop_loglik - case['alerax_loglik'])
            print(f"Difference from AleRax: {loop_diff:.6f}")
        except Exception as e:
            print(f"Loop-based implementation failed: {e}")
            loop_loglik = None
            loop_time = None
        
        # Compare implementations
        if efficient_loglik is not None and loop_loglik is not None:
            impl_diff = abs(efficient_loglik - loop_loglik)
            print(f"\n--- Implementation Comparison ---")
            print(f"Difference between implementations: {impl_diff:.6f}")
            if efficient_time is not None and loop_time is not None:
                speedup = loop_time / efficient_time if efficient_time > 0 else float('inf')
                print(f"Speedup: {speedup:.2f}x")
        
        print(f"\n{'='*50}\n")

if __name__ == "__main__":
    test_against_alerax()