#!/usr/bin/env python3
"""
Debug parameter evolution in realistic optimization
=================================================

This version focuses on understanding why parameters aren't evolving properly.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from realistic_optimization_comparison import RealisticOptimizer

def debug_gradient_descent(species_path: str, gene_path: str, max_iterations: int = 5):
    """Debug gradient descent with detailed parameter tracking"""
    
    print(f"🔍 DEBUGGING GRADIENT DESCENT PARAMETER EVOLUTION")
    print(f"=" * 60)
    
    optimizer = RealisticOptimizer(species_path, gene_path)
    
    # Start with reasonable initial parameters (not too small)
    params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    learning_rate = 0.01
    
    print(f"Initial parameters: δ={params[0]:.6f}, τ={params[1]:.6f}, λ={params[2]:.6f}")
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current params: δ={params[0]:.6f}, τ={params[1]:.6f}, λ={params[2]:.6f}")
        
        # Compute current likelihood
        ll_result = optimizer.compute_likelihood_and_timing(
            float(params[0]), float(params[1]), float(params[2])
        )
        print(f"Current LL: {ll_result['log_likelihood']:.6f}")
        
        # Compute gradients with detailed output
        print("Computing gradients...")
        gradients, grad_timing = optimizer.compute_gradients_realistic(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        print(f"Gradients: δ={gradients[0]:.6f}, τ={gradients[1]:.6f}, λ={gradients[2]:.6f}")
        print(f"Gradient norm: {torch.norm(gradients):.6f}")
        
        # Show gradient step before clamping
        gradient_step = learning_rate * gradients
        new_params_unclamped = params + gradient_step
        print(f"Gradient step: δ={gradient_step[0]:.6f}, τ={gradient_step[1]:.6f}, λ={gradient_step[2]:.6f}")
        print(f"Unclamped new params: δ={new_params_unclamped[0]:.6f}, τ={new_params_unclamped[1]:.6f}, λ={new_params_unclamped[2]:.6f}")
        
        # Apply update
        params = params + gradient_step
        
        # Check if clamping will occur
        params_before_clamp = params.clone()
        params = torch.clamp(params, min=1e-6)
        
        clamped = not torch.allclose(params_before_clamp, params)
        if clamped:
            print(f"⚠️  CLAMPING OCCURRED!")
            print(f"Before clamp: δ={params_before_clamp[0]:.6f}, τ={params_before_clamp[1]:.6f}, λ={params_before_clamp[2]:.6f}")
        
        print(f"Final params: δ={params[0]:.6f}, τ={params[1]:.6f}, λ={params[2]:.6f}")
        
        # Check for convergence
        if torch.norm(gradients) < 1e-6:
            print(f"✅ Converged! Gradient norm below threshold")
            break

def create_parameter_evolution_plot():
    """Create detailed parameter evolution plot from the realistic results"""
    
    import json
    
    # Load the results
    with open('realistic_optimization_results.json', 'r') as f:
        results = json.load(f)
    
    gd_history = results['gradient_descent']
    newton_history = results['newton_method']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Evolution Analysis: Newton vs Gradient Descent', fontsize=16, fontweight='bold')
    
    # Colors
    gd_color = '#2E86AB'
    newton_color = '#A23B72'
    
    # Plot 1: Delta evolution
    ax1 = axes[0, 0]
    ax1.plot(gd_history['iteration'], gd_history['delta'], 
             'o-', color=gd_color, label='Gradient Descent δ', linewidth=2, markersize=6)
    ax1.plot(newton_history['iteration'], newton_history['delta'], 
             's-', color=newton_color, label="Newton's Method δ", linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Duplication Rate (δ)')
    ax1.set_title('Duplication Parameter Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Use log scale to see small values
    
    # Plot 2: Tau evolution
    ax2 = axes[0, 1]
    ax2.plot(gd_history['iteration'], gd_history['tau'], 
             'o-', color=gd_color, label='Gradient Descent τ', linewidth=2, markersize=6)
    ax2.plot(newton_history['iteration'], newton_history['tau'], 
             's-', color=newton_color, label="Newton's Method τ", linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Transfer Rate (τ)')
    ax2.set_title('Transfer Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Lambda evolution
    ax3 = axes[0, 2]
    ax3.plot(gd_history['iteration'], gd_history['lambda'], 
             'o-', color=gd_color, label='Gradient Descent λ', linewidth=2, markersize=6)
    ax3.plot(newton_history['iteration'], newton_history['lambda'], 
             's-', color=newton_color, label="Newton's Method λ", linewidth=2, markersize=6)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss Rate (λ)')
    ax3.set_title('Loss Parameter Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: All parameters together (GD)
    ax4 = axes[1, 0]
    ax4.plot(gd_history['iteration'], gd_history['delta'], 
             'o-', color='red', label='δ (duplication)', linewidth=2, markersize=4)
    ax4.plot(gd_history['iteration'], gd_history['tau'], 
             's-', color='blue', label='τ (transfer)', linewidth=2, markersize=4)
    ax4.plot(gd_history['iteration'], gd_history['lambda'], 
             '^-', color='green', label='λ (loss)', linewidth=2, markersize=4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Parameter Value (log scale)')
    ax4.set_title('Gradient Descent: All Parameters')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: All parameters together (Newton)
    ax5 = axes[1, 1]
    ax5.plot(newton_history['iteration'], newton_history['delta'], 
             'o-', color='red', label='δ (duplication)', linewidth=2, markersize=4)
    ax5.plot(newton_history['iteration'], newton_history['tau'], 
             's-', color='blue', label='τ (transfer)', linewidth=2, markersize=4)
    ax5.plot(newton_history['iteration'], newton_history['lambda'], 
             '^-', color='green', label='λ (loss)', linewidth=2, markersize=4)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Parameter Value (log scale)')
    ax5.set_title("Newton's Method: All Parameters")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # Plot 6: Parameter values summary
    ax6 = axes[1, 2]
    
    # Create text summary
    ax6.text(0.1, 0.9, 'PARAMETER EVOLUTION SUMMARY', fontsize=14, fontweight='bold', transform=ax6.transAxes)
    
    ax6.text(0.1, 0.8, 'GRADIENT DESCENT:', fontsize=12, fontweight='bold', color=gd_color, transform=ax6.transAxes)
    ax6.text(0.1, 0.75, f'δ: {gd_history["delta"][0]:.2e} → {gd_history["delta"][-1]:.2e}', fontsize=11, transform=ax6.transAxes)
    ax6.text(0.1, 0.7, f'τ: {gd_history["tau"][0]:.2e} → {gd_history["tau"][-1]:.2e}', fontsize=11, transform=ax6.transAxes)
    ax6.text(0.1, 0.65, f'λ: {gd_history["lambda"][0]:.2e} → {gd_history["lambda"][-1]:.2e}', fontsize=11, transform=ax6.transAxes)
    
    ax6.text(0.1, 0.55, "NEWTON'S METHOD:", fontsize=12, fontweight='bold', color=newton_color, transform=ax6.transAxes)
    ax6.text(0.1, 0.5, f'δ: {newton_history["delta"][0]:.2e} → {newton_history["delta"][-1]:.2e}', fontsize=11, transform=ax6.transAxes)
    ax6.text(0.1, 0.45, f'τ: {newton_history["tau"][0]:.2e} → {newton_history["tau"][-1]:.2e}', fontsize=11, transform=ax6.transAxes)
    ax6.text(0.1, 0.4, f'λ: {newton_history["lambda"][0]:.2e} → {newton_history["lambda"][-1]:.2e}', fontsize=11, transform=ax6.transAxes)
    
    # Check if parameters actually evolved
    gd_delta_change = abs(gd_history["delta"][-1] - gd_history["delta"][0])
    newton_delta_change = abs(newton_history["delta"][-1] - newton_history["delta"][0])
    
    if gd_delta_change < 1e-8 and newton_delta_change < 1e-8:
        ax6.text(0.1, 0.3, '⚠️  WARNING: Parameters not evolving!', fontsize=12, fontweight='bold', 
                color='red', transform=ax6.transAxes)
        ax6.text(0.1, 0.25, 'Possible causes:', fontsize=11, transform=ax6.transAxes)
        ax6.text(0.1, 0.2, '• Clamping to minimum values', fontsize=10, transform=ax6.transAxes)
        ax6.text(0.1, 0.15, '• Gradients pointing wrong direction', fontsize=10, transform=ax6.transAxes)
        ax6.text(0.1, 0.1, '• Learning rate too small', fontsize=10, transform=ax6.transAxes)
        ax6.text(0.1, 0.05, '• Local minimum reached', fontsize=10, transform=ax6.transAxes)
    else:
        ax6.text(0.1, 0.3, '✅ Parameters are evolving', fontsize=12, fontweight='bold', 
                color='green', transform=ax6.transAxes)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('parameter_evolution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Parameter evolution analysis saved as 'parameter_evolution_analysis.png'")
    
    return fig

def main():
    """Main debugging function"""
    
    # First, debug the gradient computation
    species_path = "test_trees_200/sp.nwk"
    gene_path = "test_trees_200/g.nwk"
    
    debug_gradient_descent(species_path, gene_path, max_iterations=3)
    
    print(f"\n")
    
    # Then create the parameter evolution plot
    create_parameter_evolution_plot()

if __name__ == "__main__":
    main()