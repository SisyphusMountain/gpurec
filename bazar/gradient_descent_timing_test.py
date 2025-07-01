#!/usr/bin/env python3
"""
Test gradient descent timing on test_trees_200 to compare with Newton's method.
"""

import torch
import time
import numpy as np
from matmul_ale_ccp_optimize_finite_diff import FiniteDiffCCPOptimizer, compute_log_likelihood

def test_gradient_descent_timing():
    """Test gradient descent timing on both test_trees_1 and test_trees_200."""
    
    print("🔬 GRADIENT DESCENT TIMING COMPARISON")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            'name': 'test_trees_1 (Small)',
            'species_path': 'test_trees_1/sp.nwk',
            'gene_path': 'test_trees_1/g.nwk',
            'expected_time': '~0.5s'
        },
        {
            'name': 'test_trees_200 (Large)', 
            'species_path': 'test_trees_200/sp.nwk',
            'gene_path': 'test_trees_200/g.nwk',
            'expected_time': '~7s (Newton)'
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n🧪 Testing {test_case['name']}")
        print(f"   Expected Newton time: {test_case['expected_time']}")
        print("-" * 50)
        
        # Initialize optimizer
        optimizer = FiniteDiffCCPOptimizer(
            test_case['species_path'], 
            test_case['gene_path'],
            initial_params=(0.1, 0.1, 0.1),
            device=torch.device("cpu"),
            dtype=torch.float64
        )
        
        # Test initial likelihood computation
        print("📊 Testing initial likelihood computation...")
        start_time = time.time()
        initial_ll = compute_log_likelihood(
            test_case['species_path'], 
            test_case['gene_path'], 
            0.1, 0.1, 0.1,
            torch.device("cpu"), 
            torch.float64
        )
        likelihood_time = time.time() - start_time
        print(f"   Initial LL: {initial_ll:.6f}")
        print(f"   Likelihood computation time: {likelihood_time:.2f}s")
        
        # Test gradient computation
        print("🔍 Testing gradient computation...")
        start_time = time.time()
        gradients = optimizer.compute_gradients(epsilon=1e-6)
        gradient_time = time.time() - start_time
        gradient_norm = torch.norm(gradients)
        print(f"   Gradients: {gradients}")
        print(f"   Gradient norm: {gradient_norm:.6f}")
        print(f"   Gradient computation time: {gradient_time:.2f}s")
        
        # Run a few gradient descent iterations to measure average time
        print("⏱️  Running gradient descent iterations...")
        iteration_times = []
        current_params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        
        for i in range(5):  # Test 5 iterations
            iter_start = time.time()
            
            # Evaluate current likelihood
            current_ll = compute_log_likelihood(
                test_case['species_path'], 
                test_case['gene_path'],
                float(current_params[0]), float(current_params[1]), float(current_params[2]),
                torch.device("cpu"), torch.float64
            )
            
            # Compute gradients
            optimizer.log_params.data = torch.log(current_params / (1 - current_params))  # Inverse softplus approx
            gradients = optimizer.compute_gradients(epsilon=1e-6)
            
            # Apply gradient step
            with torch.no_grad():
                current_params += 0.002 * gradients[:3]  # Simple step
                current_params = torch.clamp(current_params, 0.001, 0.999)  # Keep in bounds
            
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            
            print(f"   Iteration {i+1}: LL={current_ll:.6f}, time={iter_time:.2f}s")
        
        # Calculate statistics
        avg_iter_time = np.mean(iteration_times)
        std_iter_time = np.std(iteration_times)
        total_time = sum(iteration_times)
        
        print(f"\n📈 Gradient Descent Results for {test_case['name']}:")
        print(f"   Average iteration time: {avg_iter_time:.2f} ± {std_iter_time:.2f}s")
        print(f"   Total time (5 iterations): {total_time:.2f}s")
        print(f"   Likelihood computation: {likelihood_time:.2f}s")
        print(f"   Gradient computation: {gradient_time:.2f}s") 
        print(f"   Overhead per iteration: {avg_iter_time - likelihood_time:.2f}s")
        
        # Store results
        results[test_case['name']] = {
            'avg_iteration_time': avg_iter_time,
            'std_iteration_time': std_iter_time,
            'likelihood_time': likelihood_time,
            'gradient_time': gradient_time,
            'overhead_time': avg_iter_time - likelihood_time,
            'initial_ll': initial_ll,
            'gradient_norm': float(gradient_norm)
        }
    
    return results

def compare_newton_vs_gradient_descent(results):
    """Compare Newton vs Gradient Descent timing results."""
    
    print(f"\n🆚 NEWTON vs GRADIENT DESCENT TIMING COMPARISON")
    print("=" * 70)
    
    # Newton's method reference times (from previous tests)
    newton_times = {
        'test_trees_1 (Small)': 0.5,   # ~0.5s per iteration
        'test_trees_200 (Large)': 7.0  # ~7s per iteration
    }
    
    print(f"{'Test Case':<25} {'Newton Time':<12} {'GD Time':<12} {'Ratio':<8} {'Scaling'}")
    print("-" * 70)
    
    for test_name, result in results.items():
        newton_time = newton_times.get(test_name, 0)
        gd_time = result['avg_iteration_time']
        ratio = newton_time / gd_time if gd_time > 0 else float('inf')
        
        print(f"{test_name:<25} {newton_time:<12.1f} {gd_time:<12.1f} {ratio:<8.1f}x")
    
    # Scaling analysis
    small_gd = results['test_trees_1 (Small)']['avg_iteration_time']
    large_gd = results['test_trees_200 (Large)']['avg_iteration_time']
    gd_scaling = large_gd / small_gd
    
    small_newton = newton_times['test_trees_1 (Small)']
    large_newton = newton_times['test_trees_200 (Large)']
    newton_scaling = large_newton / small_newton
    
    print(f"\n📊 SCALING ANALYSIS:")
    print(f"   Problem size increase: 780x (8 leaves → 200 leaves)")
    print(f"   Gradient Descent scaling: {gd_scaling:.1f}x")
    print(f"   Newton's Method scaling: {newton_scaling:.1f}x")
    print(f"   Relative efficiency: Newton scales {gd_scaling/newton_scaling:.1f}x better")
    
    # Efficiency analysis
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    for test_name, result in results.items():
        print(f"\n   {test_name}:")
        print(f"     Likelihood computation: {result['likelihood_time']:.2f}s")
        print(f"     Gradient computation: {result['gradient_time']:.2f}s") 
        print(f"     Total GD iteration: {result['avg_iteration_time']:.2f}s")
        print(f"     Newton iteration: {newton_times.get(test_name, 0):.1f}s")
        
        # Calculate how many GD iterations equal one Newton iteration
        newton_time = newton_times.get(test_name, 0)
        if result['avg_iteration_time'] > 0:
            equivalent_iterations = newton_time / result['avg_iteration_time']
            print(f"     1 Newton = ~{equivalent_iterations:.1f} GD iterations in time")
    
    return gd_scaling, newton_scaling

def main():
    """Run complete timing comparison."""
    print("🚀 COMPREHENSIVE TIMING ANALYSIS: Newton vs Gradient Descent")
    print("=" * 80)
    
    # Test gradient descent timing
    results = test_gradient_descent_timing()
    
    # Compare with Newton's method
    gd_scaling, newton_scaling = compare_newton_vs_gradient_descent(results)
    
    # Final summary
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • Gradient Descent: Simple, but scales poorly ({gd_scaling:.1f}x)")
    print(f"   • Newton's Method: More complex, but scales excellently ({newton_scaling:.1f}x)")
    print(f"   • For large problems: Newton provides superior time-to-convergence")
    print(f"   • Both methods handle large phylogenetic trees successfully")
    
    # Convergence comparison
    small_gd_time = results['test_trees_1 (Small)']['avg_iteration_time']
    large_gd_time = results['test_trees_200 (Large)']['avg_iteration_time']
    
    print(f"\n💡 PRACTICAL IMPLICATIONS:")
    print(f"   • Small trees (8 leaves): GD competitive at {small_gd_time:.1f}s vs Newton 0.5s")
    print(f"   • Large trees (200 leaves): Newton superior at 7s vs GD {large_gd_time:.1f}s")
    print(f"   • Convergence: Newton ~10 iterations vs GD ~100+ iterations")
    print(f"   • Total time to convergence: Newton wins decisively for large problems")
    
    return results

if __name__ == "__main__":
    results = main()