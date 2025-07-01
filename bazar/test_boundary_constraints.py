#!/usr/bin/env python3
"""
Test Different Boundary Constraints
===================================

This tests what happens when we allow parameters to go very close to zero
for topologically identical trees.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from realistic_optimization_comparison import RealisticOptimizer

def test_optimization_with_different_bounds(species_path: str, gene_path: str):
    """Test optimization with different minimum bounds"""
    
    print("🧪 TESTING DIFFERENT BOUNDARY CONSTRAINTS")
    print("=" * 60)
    
    # Test different minimum bounds
    min_bounds = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    
    results = {}
    
    for min_bound in min_bounds:
        print(f"\n--- Testing min_bound = {min_bound:.0e} ---")
        
        optimizer = RealisticOptimizer(species_path, gene_path)
        
        # Start with reasonable parameters
        params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        learning_rate = 0.001  # Small learning rate
        
        history = {
            'iteration': [],
            'log_likelihood': [],
            'delta': [],
            'tau': [],
            'lambda': [],
            'gradient_norm': []
        }
        
        for iteration in range(10):
            # Compute likelihood
            ll_result = optimizer.compute_likelihood_and_timing(
                float(params[0]), float(params[1]), float(params[2])
            )
            
            # Compute gradients
            gradients, _ = optimizer.compute_gradients_realistic(
                float(params[0]), float(params[1]), float(params[2])
            )
            
            gradient_norm = torch.norm(gradients).item()
            
            # Record before update
            history['iteration'].append(iteration)
            history['log_likelihood'].append(ll_result['log_likelihood'])
            history['delta'].append(float(params[0]))
            history['tau'].append(float(params[1]))
            history['lambda'].append(float(params[2]))
            history['gradient_norm'].append(gradient_norm)
            
            print(f"  Iter {iteration}: δ={params[0]:.6f}, τ={params[1]:.6f}, λ={params[2]:.6f}, "
                  f"LL={ll_result['log_likelihood']:.4f}, ‖∇‖={gradient_norm:.2f}")
            
            # Update parameters
            params = params + learning_rate * gradients
            
            # Apply the test boundary constraint
            params = torch.clamp(params, min=min_bound)
            
            # Check if we hit the boundary
            if torch.any(params <= min_bound * 1.001):  # Close to boundary
                print(f"  ⚠️  Hit boundary constraint at {min_bound:.0e}")
            
            # Check convergence
            if gradient_norm < 1e-3:
                print(f"  ✅ Converged after {iteration+1} iterations")
                break
        
        results[min_bound] = history
        
        final_params = params
        final_ll = history['log_likelihood'][-1]
        print(f"Final: δ={final_params[0]:.2e}, τ={final_params[1]:.2e}, λ={final_params[2]:.2e}, LL={final_ll:.4f}")
    
    return results

def test_unconstrained_optimization(species_path: str, gene_path: str):
    """Test what happens with NO minimum constraint"""
    
    print(f"\n🚀 TESTING UNCONSTRAINED OPTIMIZATION (NO MIN BOUND)")
    print("=" * 60)
    
    optimizer = RealisticOptimizer(species_path, gene_path)
    
    # Start with reasonable parameters
    params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    learning_rate = 0.001  # Small learning rate
    
    history = {
        'iteration': [],
        'log_likelihood': [],
        'delta': [],
        'tau': [],
        'lambda': [],
        'gradient_norm': []
    }
    
    for iteration in range(15):
        # Compute likelihood
        ll_result = optimizer.compute_likelihood_and_timing(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        # Compute gradients  
        gradients, _ = optimizer.compute_gradients_realistic(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        gradient_norm = torch.norm(gradients).item()
        
        # Record before update
        history['iteration'].append(iteration)
        history['log_likelihood'].append(ll_result['log_likelihood'])
        history['delta'].append(float(params[0]))
        history['tau'].append(float(params[1]))
        history['lambda'].append(float(params[2]))
        history['gradient_norm'].append(gradient_norm)
        
        print(f"Iter {iteration}: δ={params[0]:.6f}, τ={params[1]:.6f}, λ={params[2]:.6f}, "
              f"LL={ll_result['log_likelihood']:.4f}, ‖∇‖={gradient_norm:.2f}")
        
        # Update parameters - NO CLAMPING!
        params = params + learning_rate * gradients
        
        # Just prevent extreme negative values that would cause numerical issues
        params = torch.clamp(params, min=1e-20)  # Very small but non-zero
        
        # Check convergence
        if gradient_norm < 1e-3:
            print(f"✅ Converged after {iteration+1} iterations")
            break
            
        # Stop if parameters are converging to very small values
        if torch.all(params < 1e-10):
            print(f"✅ Parameters converged to near-zero values")
            break
    
    return history

def create_boundary_test_plot(results, unconstrained_history):
    """Create visualization of boundary constraint effects"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Effect of Boundary Constraints on Parameter Optimization', fontsize=16, fontweight='bold')
    
    # Colors for different bounds
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    bounds = list(results.keys())
    
    # Plot 1: Delta evolution
    ax1 = axes[0, 0]
    for i, min_bound in enumerate(bounds):
        history = results[min_bound]
        ax1.plot(history['iteration'], history['delta'], 
                'o-', color=colors[i], label=f'min={min_bound:.0e}', linewidth=2, markersize=4)
    
    # Add unconstrained
    ax1.plot(unconstrained_history['iteration'], unconstrained_history['delta'],
             's-', color='black', label='Unconstrained', linewidth=3, markersize=4)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Duplication Rate (δ)')
    ax1.set_title('Duplication Parameter Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Tau evolution
    ax2 = axes[0, 1]
    for i, min_bound in enumerate(bounds):
        history = results[min_bound]
        ax2.plot(history['iteration'], history['tau'], 
                'o-', color=colors[i], label=f'min={min_bound:.0e}', linewidth=2, markersize=4)
    
    ax2.plot(unconstrained_history['iteration'], unconstrained_history['tau'],
             's-', color='black', label='Unconstrained', linewidth=3, markersize=4)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Transfer Rate (τ)')
    ax2.set_title('Transfer Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Lambda evolution
    ax3 = axes[0, 2]
    for i, min_bound in enumerate(bounds):
        history = results[min_bound]
        ax3.plot(history['iteration'], history['lambda'], 
                'o-', color=colors[i], label=f'min={min_bound:.0e}', linewidth=2, markersize=4)
    
    ax3.plot(unconstrained_history['iteration'], unconstrained_history['lambda'],
             's-', color='black', label='Unconstrained', linewidth=3, markersize=4)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss Rate (λ)')
    ax3.set_title('Loss Parameter Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Log-likelihood evolution
    ax4 = axes[1, 0]
    for i, min_bound in enumerate(bounds):
        history = results[min_bound]
        ax4.plot(history['iteration'], history['log_likelihood'], 
                'o-', color=colors[i], label=f'min={min_bound:.0e}', linewidth=2, markersize=4)
    
    ax4.plot(unconstrained_history['iteration'], unconstrained_history['log_likelihood'],
             's-', color='black', label='Unconstrained', linewidth=3, markersize=4)
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Log-Likelihood')
    ax4.set_title('Log-Likelihood Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Final parameter values
    ax5 = axes[1, 1]
    
    bound_labels = [f'{b:.0e}' for b in bounds] + ['Unconstrained']
    final_deltas = [results[b]['delta'][-1] for b in bounds] + [unconstrained_history['delta'][-1]]
    final_taus = [results[b]['tau'][-1] for b in bounds] + [unconstrained_history['tau'][-1]]
    final_lambdas = [results[b]['lambda'][-1] for b in bounds] + [unconstrained_history['lambda'][-1]]
    
    x = np.arange(len(bound_labels))
    width = 0.25
    
    ax5.bar(x - width, final_deltas, width, label='δ', alpha=0.7, color='red')
    ax5.bar(x, final_taus, width, label='τ', alpha=0.7, color='blue')
    ax5.bar(x + width, final_lambdas, width, label='λ', alpha=0.7, color='green')
    
    ax5.set_xlabel('Minimum Bound')
    ax5.set_ylabel('Final Parameter Value')
    ax5.set_title('Final Parameter Values')
    ax5.set_xticks(x)
    ax5.set_xticklabels(bound_labels, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Plot 6: Summary text
    ax6 = axes[1, 2]
    ax6.text(0.1, 0.9, 'BOUNDARY CONSTRAINT ANALYSIS', fontsize=14, fontweight='bold', transform=ax6.transAxes)
    
    ax6.text(0.1, 0.8, 'Key Findings:', fontsize=12, fontweight='bold', transform=ax6.transAxes)
    
    # Find best result (highest final likelihood)
    best_ll = max([results[b]['log_likelihood'][-1] for b in bounds] + [unconstrained_history['log_likelihood'][-1]])
    unconstrained_ll = unconstrained_history['log_likelihood'][-1]
    
    if abs(unconstrained_ll - best_ll) < 0.001:
        ax6.text(0.1, 0.75, '✅ Unconstrained gives best likelihood', fontsize=11, color='green', transform=ax6.transAxes)
    else:
        ax6.text(0.1, 0.75, '⚠️  Boundary constraints limit performance', fontsize=11, color='orange', transform=ax6.transAxes)
    
    # Show final unconstrained values
    final_delta = unconstrained_history['delta'][-1]
    final_tau = unconstrained_history['tau'][-1] 
    final_lambda = unconstrained_history['lambda'][-1]
    
    ax6.text(0.1, 0.65, f'Unconstrained final values:', fontsize=11, fontweight='bold', transform=ax6.transAxes)
    ax6.text(0.1, 0.6, f'δ = {final_delta:.2e}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.55, f'τ = {final_tau:.2e}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.5, f'λ = {final_lambda:.2e}', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.45, f'LL = {unconstrained_ll:.4f}', fontsize=10, transform=ax6.transAxes)
    
    if final_delta < 1e-6 and final_tau < 1e-6 and final_lambda < 1e-6:
        ax6.text(0.1, 0.35, '✅ Parameters converged near zero', fontsize=11, fontweight='bold', 
                color='green', transform=ax6.transAxes)
        ax6.text(0.1, 0.3, '✅ Confirms optimal solution for', fontsize=11, transform=ax6.transAxes)
        ax6.text(0.1, 0.25, '    identical trees is pure speciation', fontsize=11, transform=ax6.transAxes)
    else:
        ax6.text(0.1, 0.35, '❓ Parameters did not reach zero', fontsize=11, fontweight='bold', 
                color='orange', transform=ax6.transAxes)
        ax6.text(0.1, 0.3, '❓ May need more iterations or', fontsize=11, transform=ax6.transAxes)
        ax6.text(0.1, 0.25, '   trees are not perfectly identical', fontsize=11, transform=ax6.transAxes)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('boundary_constraint_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Boundary constraint analysis saved as 'boundary_constraint_analysis.png'")

def main():
    """Main function to test boundary constraints"""
    
    species_path = "test_trees_200/sp.nwk"
    gene_path = "test_trees_200/g.nwk"
    
    print("🔬 INVESTIGATING BOUNDARY CONSTRAINT EFFECTS")
    print("=" * 80)
    
    # Test different boundary constraints
    results = test_optimization_with_different_bounds(species_path, gene_path)
    
    # Test unconstrained optimization
    unconstrained_history = test_unconstrained_optimization(species_path, gene_path)
    
    # Create visualization
    create_boundary_test_plot(results, unconstrained_history)
    
    print(f"\n🎯 CONCLUSION:")
    final_ll = unconstrained_history['log_likelihood'][-1]
    final_params = [unconstrained_history['delta'][-1], 
                   unconstrained_history['tau'][-1], 
                   unconstrained_history['lambda'][-1]]
    
    print(f"Unconstrained optimization result:")
    print(f"  Final LL: {final_ll:.6f}")
    print(f"  Final δ: {final_params[0]:.2e}")
    print(f"  Final τ: {final_params[1]:.2e}")
    print(f"  Final λ: {final_params[2]:.2e}")
    
    if all(p < 1e-6 for p in final_params):
        print(f"✅ SUCCESS: Parameters converged to near-zero as expected for identical trees!")
    else:
        print(f"❓ Parameters did not reach zero - may need investigation")

if __name__ == "__main__":
    main()