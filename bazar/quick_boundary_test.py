#!/usr/bin/env python3
"""
Quick Boundary Constraint Test
=============================

Fast test to demonstrate the boundary constraint issue.
"""

import torch
import matplotlib.pyplot as plt
from realistic_optimization_comparison import RealisticOptimizer

def quick_test():
    """Quick test with smaller trees to demonstrate the issue"""
    
    print("🚀 QUICK BOUNDARY CONSTRAINT TEST")
    print("=" * 50)
    
    # Use smaller trees for speed
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    optimizer = RealisticOptimizer(species_path, gene_path)
    
    # Test with and without boundary constraint
    tests = [
        {"name": "With Boundary (min=1e-6)", "min_bound": 1e-6},
        {"name": "Lower Boundary (min=1e-12)", "min_bound": 1e-12},
        {"name": "No Boundary (min=1e-20)", "min_bound": 1e-20}
    ]
    
    results = {}
    
    for test in tests:
        print(f"\n--- {test['name']} ---")
        
        # Reset optimizer state
        optimizer.E = None
        optimizer.Pi = None
        
        params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        learning_rate = 0.01  # Larger for faster convergence
        
        history = []
        
        for iteration in range(5):  # Just 5 iterations for speed
            # Compute likelihood and gradients
            ll_result = optimizer.compute_likelihood_and_timing(
                float(params[0]), float(params[1]), float(params[2])
            )
            
            gradients, _ = optimizer.compute_gradients_realistic(
                float(params[0]), float(params[1]), float(params[2])
            )
            
            # Record
            history.append({
                'iteration': iteration,
                'delta': float(params[0]),
                'tau': float(params[1]),
                'lambda': float(params[2]),
                'log_likelihood': ll_result['log_likelihood'],
                'gradient_norm': torch.norm(gradients).item()
            })
            
            print(f"  Iter {iteration}: δ={params[0]:.6f}, LL={ll_result['log_likelihood']:.4f}, ‖∇‖={torch.norm(gradients):.1f}")
            
            # Update parameters
            params = params + learning_rate * gradients
            
            # Apply boundary constraint
            params_before = params.clone()
            params = torch.clamp(params, min=test['min_bound'])
            
            # Check if clamped
            if not torch.allclose(params_before, params):
                print(f"    ⚠️  Hit boundary: {params_before} → {params}")
        
        results[test['name']] = history
        final_params = params
        final_ll = history[-1]['log_likelihood']
        print(f"  Final: δ={final_params[0]:.2e}, LL={final_ll:.4f}")
    
    return results

def create_quick_plot(results):
    """Create a quick visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Quick Boundary Constraint Test Results', fontsize=14, fontweight='bold')
    
    colors = ['red', 'orange', 'green']
    
    # Plot 1: Parameter evolution
    ax1 = axes[0]
    for i, (name, history) in enumerate(results.items()):
        iterations = [h['iteration'] for h in history]
        deltas = [h['delta'] for h in history]
        ax1.plot(iterations, deltas, 'o-', color=colors[i], label=name, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Duplication Rate (δ)')
    ax1.set_title('Parameter Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Log-likelihood evolution
    ax2 = axes[1]
    for i, (name, history) in enumerate(results.items()):
        iterations = [h['iteration'] for h in history]
        lls = [h['log_likelihood'] for h in history]
        ax2.plot(iterations, lls, 'o-', color=colors[i], label=name, linewidth=2, markersize=4)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Log-Likelihood Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Summary
    ax3 = axes[2]
    ax3.text(0.1, 0.9, 'KEY FINDINGS:', fontsize=12, fontweight='bold', transform=ax3.transAxes)
    
    # Get final results
    final_results = {}
    for name, history in results.items():
        final_results[name] = {
            'delta': history[-1]['delta'],
            'll': history[-1]['log_likelihood']
        }
    
    y_pos = 0.8
    for i, (name, final) in enumerate(final_results.items()):
        ax3.text(0.1, y_pos, f'{name}:', fontsize=10, fontweight='bold', 
                color=colors[i], transform=ax3.transAxes)
        ax3.text(0.1, y_pos-0.05, f'  Final δ: {final["delta"]:.2e}', fontsize=9, transform=ax3.transAxes)
        ax3.text(0.1, y_pos-0.1, f'  Final LL: {final["ll"]:.4f}', fontsize=9, transform=ax3.transAxes)
        y_pos -= 0.2
    
    # Conclusion
    best_ll = max(final['ll'] for final in final_results.values())
    best_method = [name for name, final in final_results.items() if final['ll'] == best_ll][0]
    
    ax3.text(0.1, 0.3, f'BEST: {best_method}', fontsize=11, fontweight='bold', 
            color='green', transform=ax3.transAxes)
    ax3.text(0.1, 0.25, f'Highest LL: {best_ll:.4f}', fontsize=10, transform=ax3.transAxes)
    
    if 'No Boundary' in best_method:
        ax3.text(0.1, 0.15, '✅ Removing boundary improves', fontsize=10, fontweight='bold',
                color='green', transform=ax3.transAxes)
        ax3.text(0.1, 0.1, '   likelihood as expected!', fontsize=10, fontweight='bold',
                color='green', transform=ax3.transAxes)
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_boundary_test_results.png', dpi=300, bbox_inches='tight')
    print(f"📊 Quick test results saved as 'quick_boundary_test_results.png'")

def main():
    """Run the quick test"""
    results = quick_test()
    create_quick_plot(results)
    
    print(f"\n🎯 SUMMARY:")
    for name, history in results.items():
        final = history[-1]
        print(f"{name}: δ={final['delta']:.2e}, LL={final['log_likelihood']:.4f}")

if __name__ == "__main__":
    main()