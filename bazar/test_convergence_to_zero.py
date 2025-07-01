#!/usr/bin/env python3
"""
Test convergence to optimal parameters near 0 for test_trees_1.
This validates the theoretical prediction that optimal δ,τ,λ should be near 0.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from matmul_ale_ccp_optimize_finite_diff import FiniteDiffCCPOptimizer, compute_log_likelihood

def test_convergence_to_zero():
    """Test that optimization converges to parameters near 0."""
    print("🎯 Testing Convergence to Optimal Parameters Near 0")
    print("=" * 60)
    
    # Test different starting points to ensure robust convergence
    starting_points = [
        (0.1, 0.1, 0.1),    # Original starting point
        (0.5, 0.3, 0.2),    # Higher starting point  
        (0.01, 0.01, 0.01), # Already close to optimal
        (1.0, 0.5, 0.8),    # Much higher starting point
    ]
    
    results = []
    
    for i, (delta_init, tau_init, lambda_init) in enumerate(starting_points):
        print(f"\n📍 Test {i+1}: Starting from δ={delta_init:.3f}, τ={tau_init:.3f}, λ={lambda_init:.3f}")
        
        # Create optimizer
        optimizer = FiniteDiffCCPOptimizer(
            "test_trees_1/sp.nwk", "test_trees_1/g.nwk",
            initial_params=(delta_init, tau_init, lambda_init),
            device=torch.device("cpu"),
            dtype=torch.float64
        )
        
        # Run optimization with more epochs to allow full convergence
        result = optimizer.optimize(
            lr=0.002,  # Smaller learning rate for stability
            epochs=100, 
            epsilon=1e-7,  # Smaller epsilon for better precision
            early_stopping_patience=15,
            min_improvement=1e-8
        )
        
        results.append({
            'starting_params': (delta_init, tau_init, lambda_init),
            'final_params': result['final_params'],
            'final_likelihood': result['best_log_likelihood'],
            'epochs': result['epochs_run'],
            'history': result['history']
        })
        
        final_delta, final_tau, final_lambda = result['final_params']
        print(f"   Final: δ={final_delta:.6f}, τ={final_tau:.6f}, λ={final_lambda:.6f}")
        print(f"   Improvement: {result['best_log_likelihood'] - (-6.45):.6f}")
        print(f"   Distance from 0: {np.sqrt(final_delta**2 + final_tau**2 + final_lambda**2):.6f}")
    
    return results

def analyze_likelihood_landscape():
    """Analyze the likelihood landscape around the optimal point."""
    print(f"\n🔍 Analyzing Likelihood Landscape Around Optimal Point")
    print("=" * 55)
    
    # Test likelihood at various points near 0
    test_points = [
        (0.0, 0.0, 0.0),      # Pure speciation
        (0.001, 0.001, 0.001), # Very small rates
        (0.01, 0.01, 0.01),   # Small rates
        (0.05, 0.05, 0.05),   # Medium-small rates
        (0.1, 0.1, 0.1),      # Original starting point
    ]
    
    likelihoods = []
    
    for delta, tau, lambda_param in test_points:
        ll = compute_log_likelihood(
            "test_trees_1/sp.nwk", "test_trees_1/g.nwk", 
            delta, tau, lambda_param,
            torch.device("cpu"), torch.float64
        )
        likelihoods.append(ll)
        print(f"   δ={delta:.3f}, τ={tau:.3f}, λ={lambda_param:.3f} → LL={ll:.6f}")
    
    # Find the best point
    best_idx = np.argmax(likelihoods)
    best_point = test_points[best_idx]
    best_ll = likelihoods[best_idx]
    
    print(f"\n✨ Best point tested: δ={best_point[0]:.3f}, τ={best_point[1]:.3f}, λ={best_point[2]:.3f}")
    print(f"   Best likelihood: {best_ll:.6f}")
    
    return test_points, likelihoods

def plot_convergence_results(results):
    """Plot convergence trajectories for all starting points."""
    print(f"\n📊 Creating convergence plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Parameter evolution over time
    plt.subplot(2, 3, 1)
    for i, result in enumerate(results):
        history = result['history']
        plt.plot(history['epochs'], history['deltas'], 'o-', label=f'Start {i+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('δ (Delta)')
    plt.title('Delta Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    for i, result in enumerate(results):
        history = result['history']
        plt.plot(history['epochs'], history['taus'], 's-', label=f'Start {i+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('τ (Tau)')
    plt.title('Tau Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    for i, result in enumerate(results):
        history = result['history']
        plt.plot(history['epochs'], history['lambdas'], '^-', label=f'Start {i+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('λ (Lambda)')
    plt.title('Lambda Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood evolution
    plt.subplot(2, 3, 4)
    for i, result in enumerate(results):
        history = result['history']
        plt.plot(history['epochs'], history['log_likelihoods'], 'o-', label=f'Start {i+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Log-Likelihood')
    plt.title('Likelihood Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Distance from origin
    plt.subplot(2, 3, 5)
    for i, result in enumerate(results):
        history = result['history']
        distances = [np.sqrt(d**2 + t**2 + l**2) for d, t, l in 
                    zip(history['deltas'], history['taus'], history['lambdas'])]
        plt.plot(history['epochs'], distances, 'o-', label=f'Start {i+1}', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Distance from Origin')
    plt.title('Convergence to Zero')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Final parameter comparison
    plt.subplot(2, 3, 6)
    final_deltas = [r['final_params'][0] for r in results]
    final_taus = [r['final_params'][1] for r in results]
    final_lambdas = [r['final_params'][2] for r in results]
    
    x = range(len(results))
    width = 0.25
    plt.bar([i - width for i in x], final_deltas, width, label='δ', alpha=0.7)
    plt.bar(x, final_taus, width, label='τ', alpha=0.7)
    plt.bar([i + width for i in x], final_lambdas, width, label='λ', alpha=0.7)
    
    plt.xlabel('Starting Point')
    plt.ylabel('Final Parameter Value')
    plt.title('Final Converged Parameters')
    plt.xticks(x, [f'Start {i+1}' for i in x])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_to_zero_test.png', dpi=150, bbox_inches='tight')
    print(f"   Plot saved as 'convergence_to_zero_test.png'")

def create_summary_report(results, test_points, likelihoods):
    """Create a comprehensive summary report."""
    print(f"\n📋 CONVERGENCE TO ZERO - SUMMARY REPORT")
    print("=" * 60)
    
    # Convergence analysis
    print(f"\n🎯 CONVERGENCE ANALYSIS:")
    all_converged_to_zero = True
    threshold = 0.01  # Consider "near zero" if all params < 0.01
    
    for i, result in enumerate(results):
        delta, tau, lambda_param = result['final_params']
        distance = np.sqrt(delta**2 + tau**2 + lambda_param**2)
        near_zero = all(p < threshold for p in [delta, tau, lambda_param])
        
        print(f"   Starting point {i+1}: δ={result['starting_params'][0]:.3f}, τ={result['starting_params'][1]:.3f}, λ={result['starting_params'][2]:.3f}")
        print(f"   Final parameters: δ={delta:.6f}, τ={tau:.6f}, λ={lambda_param:.6f}")
        print(f"   Distance from 0: {distance:.6f}, Near zero: {'✅' if near_zero else '❌'}")
        print(f"   Final likelihood: {result['final_likelihood']:.6f}")
        print()
        
        if not near_zero:
            all_converged_to_zero = False
    
    # Overall assessment
    print(f"🏆 OVERALL ASSESSMENT:")
    if all_converged_to_zero:
        print(f"   ✅ SUCCESS: All starting points converged to near-zero parameters")
        print(f"   ✅ This validates the theoretical prediction for test_trees_1")
    else:
        print(f"   ⚠️  PARTIAL: Some starting points did not reach near-zero parameters")
        print(f"   💡 This may indicate local optima or need for more optimization")
    
    # Best result
    best_result = max(results, key=lambda r: r['final_likelihood'])
    best_delta, best_tau, best_lambda = best_result['final_params']
    print(f"\n🥇 BEST RESULT:")
    print(f"   Parameters: δ={best_delta:.6f}, τ={best_tau:.6f}, λ={best_lambda:.6f}")
    print(f"   Log-likelihood: {best_result['final_likelihood']:.6f}")
    print(f"   Starting from: {best_result['starting_params']}")
    
    # Landscape analysis
    best_landscape_idx = np.argmax(likelihoods)
    best_landscape_point = test_points[best_landscape_idx]
    print(f"\n🗺️  LANDSCAPE ANALYSIS:")
    print(f"   Best tested point: δ={best_landscape_point[0]:.3f}, τ={best_landscape_point[1]:.3f}, λ={best_landscape_point[2]:.3f}")
    print(f"   Best tested likelihood: {likelihoods[best_landscape_idx]:.6f}")
    
    return all_converged_to_zero

def main():
    """Run the complete convergence test."""
    print("🧬 Phylogenetic Parameter Optimization - Convergence to Zero Test")
    print("=" * 70)
    
    # Test convergence from multiple starting points
    results = test_convergence_to_zero()
    
    # Analyze likelihood landscape
    test_points, likelihoods = analyze_likelihood_landscape()
    
    # Create visualizations
    plot_convergence_results(results)
    
    # Generate summary report
    success = create_summary_report(results, test_points, likelihoods)
    
    print(f"\n🎯 CONCLUSION:")
    if success:
        print(f"   The optimization algorithm successfully converges to near-zero parameters,")
        print(f"   validating the theoretical prediction that optimal δ,τ,λ ≈ 0 for test_trees_1.")
        print(f"   This demonstrates that our gradient descent implementation works correctly!")
    else:
        print(f"   The results show the optimization is working but may need tuning.")
        print(f"   The algorithm is finding improvements but may need more iterations or")
        print(f"   different hyperparameters to reach the theoretical optimum.")
    
    return results, success

if __name__ == "__main__":
    results, success = main()