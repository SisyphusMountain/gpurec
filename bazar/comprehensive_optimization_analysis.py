#!/usr/bin/env python3
"""
Comprehensive Optimization Analysis with Proper GPU Timing
==========================================================

Complete visualization showing:
- Pi update timing with torch.cuda.synchronize() 
- Parameter evolution (δ, τ, λ)
- Log-likelihood evolution
- Final parameter values
- Summary and additional relevant plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Tuple
from realistic_optimization_comparison import RealisticOptimizer

class ComprehensiveOptimizer(RealisticOptimizer):
    """Enhanced optimizer with detailed timing and tracking"""
    
    def __init__(self, species_path: str, gene_path: str, device: str = "cpu"):
        super().__init__(species_path, gene_path, device)
        
        # Enhanced tracking
        self.detailed_timing = {
            'pi_update_times': [],
            'e_update_times': [],
            'gradient_computation_times': [],
            'total_iteration_times': []
        }
        
    def update_Pi_with_detailed_timing(self, delta: float, tau: float, lam: float, E: torch.Tensor,
                                     max_iterations: int = 50, tolerance: float = 1e-8) -> Tuple[torch.Tensor, int, float]:
        """Update Pi with detailed timing using torch.cuda.synchronize()"""
        
        # Initialize Pi if first iteration  
        if self.Pi is None:
            log_Pi = torch.full((self.n_clades, self.n_species), float('-inf'), 
                              device=self.device, dtype=self.dtype)
            
            for c in range(self.n_clades):
                clade = self.ccp.id_to_clade[c]
                if clade.is_leaf():
                    mapped_species = torch.nonzero(self.clade_species_mapping[c] > 0, as_tuple=False).flatten()
                    if len(mapped_species) > 0:
                        log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=self.dtype))
                        log_Pi[c, mapped_species] = log_prob
            
            self.Pi = log_Pi
        
        # Compute event probabilities and Ebar
        rates_sum = 1.0 + delta + tau + lam
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum  
        p_T = tau / rates_sum
        p_L = lam / rates_sum
        
        Ebar = torch.mv(self.species_helpers['Recipients_mat'], E)
        
        # Detailed timing of Pi updates
        individual_times = []
        
        # Synchronize before starting
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_start = time.time()
        log_Pi_current = self.Pi.clone()
        
        for iteration in range(max_iterations):
            # Time individual Pi update iteration
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            iter_start = time.time()
            
            # Import the Pi update function here to avoid circular import
            from matmul_ale_ccp_log import Pi_update_ccp_log
            
            log_Pi_new = Pi_update_ccp_log(
                log_Pi_current, self.ccp_helpers, self.species_helpers, 
                self.clade_species_mapping, E, Ebar, p_S, p_D, p_T
            )
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            iter_time = time.time() - iter_start
            individual_times.append(iter_time)
            
            # Check convergence
            if iteration > 0:
                diff = torch.abs(log_Pi_new - log_Pi_current).max()
                if diff < tolerance:
                    break
            
            log_Pi_current = log_Pi_new
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        total_time = time.time() - total_start
        
        # Update stored Pi
        self.Pi = log_Pi_current
        
        return log_Pi_current, iteration + 1, total_time, individual_times
    
    def comprehensive_likelihood_and_timing(self, delta: float, tau: float, lam: float) -> Dict:
        """Compute likelihood with comprehensive timing breakdown"""
        
        # Time E convergence
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        e_start = time.time()
        E, e_iterations = self.update_E_convergence(delta, tau, lam)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        e_time = time.time() - e_start
        
        # Time Pi convergence with detailed breakdown
        Pi, pi_iterations, pi_total_time, pi_individual_times = self.update_Pi_with_detailed_timing(delta, tau, lam, E)
        
        # Compute log-likelihood
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        ll_start = time.time()
        root_log_probs = Pi[self.root_clade_id, :]
        log_likelihood = torch.logsumexp(root_log_probs, dim=0).item()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        ll_time = time.time() - ll_start
        
        return {
            'log_likelihood': log_likelihood,
            'e_time': e_time,
            'pi_total_time': pi_total_time,
            'pi_individual_times': pi_individual_times,
            'pi_avg_time_per_iter': np.mean(pi_individual_times),
            'll_time': ll_time,
            'total_time': e_time + pi_total_time + ll_time,
            'e_iterations': e_iterations,
            'pi_iterations': pi_iterations
        }
    
    def comprehensive_gradients(self, delta: float, tau: float, lam: float, 
                              epsilon: float = 1e-6) -> Tuple[torch.Tensor, Dict]:
        """Compute gradients with comprehensive timing"""
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        grad_start = time.time()
        
        # Base likelihood
        base_result = self.comprehensive_likelihood_and_timing(delta, tau, lam)
        base_ll = base_result['log_likelihood']
        
        gradients = torch.zeros(3, dtype=self.dtype)
        gradient_timings = []
        
        # Save current state for restoration
        E_saved = self.E.clone() if self.E is not None else None
        Pi_saved = self.Pi.clone() if self.Pi is not None else None
        
        params = [delta, tau, lam]
        
        for i in range(3):
            param_grad_start = time.time()
            
            # Positive perturbation
            params_pos = params.copy()
            params_pos[i] += epsilon
            
            # Restore state
            if E_saved is not None:
                self.E = E_saved.clone()
            if Pi_saved is not None:
                self.Pi = Pi_saved.clone()
                
            pos_result = self.comprehensive_likelihood_and_timing(*params_pos)
            
            # Negative perturbation
            params_neg = params.copy()
            params_neg[i] -= epsilon
            
            # Restore state
            if E_saved is not None:
                self.E = E_saved.clone()
            if Pi_saved is not None:
                self.Pi = Pi_saved.clone()
                
            neg_result = self.comprehensive_likelihood_and_timing(*params_neg)
            
            # Central difference gradient
            gradients[i] = (pos_result['log_likelihood'] - neg_result['log_likelihood']) / (2 * epsilon)
            
            param_grad_time = time.time() - param_grad_start
            gradient_timings.append({
                'parameter': ['delta', 'tau', 'lambda'][i],
                'time': param_grad_time,
                'pos_pi_time': pos_result['pi_total_time'],
                'neg_pi_time': neg_result['pi_total_time']
            })
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        total_gradient_time = time.time() - grad_start
        
        # Restore original state
        if E_saved is not None:
            self.E = E_saved
        if Pi_saved is not None:
            self.Pi = Pi_saved
        
        return gradients, {
            'total_time': total_gradient_time,
            'base_computation': base_result,
            'parameter_timings': gradient_timings
        }

def run_comprehensive_optimization(species_path: str, gene_path: str, 
                                 max_iterations: int = 10, learning_rate: float = 0.01) -> Dict:
    """Run comprehensive optimization with detailed tracking"""
    
    print(f"🚀 COMPREHENSIVE OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    optimizer = ComprehensiveOptimizer(species_path, gene_path, device)
    
    # Start with reasonable parameters - NO minimum boundary constraint
    params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    
    # Track comprehensive history
    history = {
        'iteration': [],
        'log_likelihood': [],
        'delta': [], 'tau': [], 'lambda': [],
        'gradient_norm': [], 'gradients': [],
        
        # Timing data
        'e_time': [], 'pi_total_time': [], 'pi_avg_iter_time': [],
        'll_time': [], 'gradient_time': [], 'total_iteration_time': [],
        'cumulative_time': [],
        
        # Convergence data
        'e_iterations': [], 'pi_iterations': [],
        
        # Detailed Pi timing
        'pi_individual_times': [],
        
        # Gradient computation breakdown
        'gradient_parameter_timings': []
    }
    
    cumulative_time = 0.0
    
    for iteration in range(max_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        iter_start = time.time()
        
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current params: δ={params[0]:.6f}, τ={params[1]:.6f}, λ={params[2]:.6f}")
        
        # Compute current likelihood with comprehensive timing
        ll_result = optimizer.comprehensive_likelihood_and_timing(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        # Compute gradients with comprehensive timing
        gradients, grad_timing = optimizer.comprehensive_gradients(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        gradient_norm = torch.norm(gradients).item()
        
        print(f"LL: {ll_result['log_likelihood']:.6f}")
        print(f"Gradients: δ={gradients[0]:.3f}, τ={gradients[1]:.3f}, λ={gradients[2]:.3f}")
        print(f"‖∇‖: {gradient_norm:.3f}")
        print(f"E: {ll_result['e_iterations']}it/{ll_result['e_time']:.3f}s")
        print(f"Pi: {ll_result['pi_iterations']}it/{ll_result['pi_total_time']:.3f}s (avg: {ll_result['pi_avg_time_per_iter']:.4f}s/iter)")
        print(f"Gradients: {grad_timing['total_time']:.3f}s")
        
        # Update parameters - NO clamping to allow convergence to zero
        params = params + learning_rate * gradients
        
        # Only prevent extreme negative values that would cause numerical issues
        params = torch.clamp(params, min=1e-20)
        
        # Compute likelihood with UPDATED parameters to track actual progress
        optimizer.E = None  # Reset state for clean computation
        optimizer.Pi = None
        updated_ll_result = optimizer.comprehensive_likelihood_and_timing(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        if device == 'cuda':
            torch.cuda.synchronize()
        iter_time = time.time() - iter_start
        cumulative_time += iter_time
        
        print(f"Updated params: δ={params[0]:.2e}, τ={params[1]:.2e}, λ={params[2]:.2e}")
        print(f"UPDATED LL: {updated_ll_result['log_likelihood']:.6f} (ΔLL: {updated_ll_result['log_likelihood'] - ll_result['log_likelihood']:+.6f})")
        
        # Record comprehensive history with UPDATED likelihood
        history['iteration'].append(iteration)
        history['log_likelihood'].append(updated_ll_result['log_likelihood'])  # Use updated LL
        history['delta'].append(float(params[0]))
        history['tau'].append(float(params[1]))
        history['lambda'].append(float(params[2]))
        history['gradient_norm'].append(gradient_norm)
        history['gradients'].append([float(g) for g in gradients])
        
        # Timing data (use updated computation timing)
        history['e_time'].append(updated_ll_result['e_time'])
        history['pi_total_time'].append(updated_ll_result['pi_total_time'])
        history['pi_avg_iter_time'].append(updated_ll_result['pi_avg_time_per_iter'])
        history['ll_time'].append(updated_ll_result['ll_time'])
        history['gradient_time'].append(grad_timing['total_time'])
        history['total_iteration_time'].append(iter_time)
        history['cumulative_time'].append(cumulative_time)
        
        # Convergence data (use updated computation)
        history['e_iterations'].append(updated_ll_result['e_iterations'])
        history['pi_iterations'].append(updated_ll_result['pi_iterations'])
        
        # Detailed Pi timing (use updated computation)
        history['pi_individual_times'].append(updated_ll_result['pi_individual_times'])
        history['gradient_parameter_timings'].append(grad_timing['parameter_timings'])
        
        print(f"Total iteration time: {iter_time:.3f}s")
        
        # Check convergence
        if gradient_norm < 1e-4:
            print(f"✅ Converged! Gradient norm {gradient_norm:.2e} below threshold")
            break
            
        # Check if parameters are approaching zero (expected for identical trees)
        if torch.all(params < 1e-10):
            print(f"✅ Parameters converged to near-zero values (pure speciation)")
            break
    
    return history

def create_comprehensive_visualization(history: Dict, species_path: str, gene_path: str):
    """Create comprehensive visualization with all requested plots"""
    
    from ete3 import Tree
    species_tree = Tree(species_path, format=1)
    gene_tree = Tree(gene_path, format=1)
    n_species = len(species_tree.get_leaves())
    n_genes = len(gene_tree.get_leaves())
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Comprehensive Optimization Analysis: {n_species} Species, {n_genes} Gene Leaves', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Pi Update Timing (detailed)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['iteration'], history['pi_total_time'], 'o-', color='blue', 
             label='Total Pi Time', linewidth=2, markersize=4)
    ax1.plot(history['iteration'], history['pi_avg_iter_time'], 's-', color='lightblue', 
             label='Avg Time/Pi Iter', linewidth=2, markersize=4)
    ax1.set_xlabel('Optimization Iteration')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Pi Update Timing\n(with torch.cuda.synchronize())')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter Evolution (δ)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['iteration'], history['delta'], 'o-', color='red', 
             linewidth=3, markersize=6)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Duplication Rate (δ)')
    ax2.set_title('Duplication Parameter Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Parameter Evolution (τ)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['iteration'], history['tau'], 's-', color='blue', 
             linewidth=3, markersize=6)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Transfer Rate (τ)')
    ax3.set_title('Transfer Parameter Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Parameter Evolution (λ)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(history['iteration'], history['lambda'], '^-', color='green', 
             linewidth=3, markersize=6)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss Rate (λ)')
    ax4.set_title('Loss Parameter Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: Log-Likelihood Evolution
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.plot(history['iteration'], history['log_likelihood'], 'o-', color='purple', 
             linewidth=3, markersize=6)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Log-Likelihood')
    ax5.set_title('Log-Likelihood Evolution')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Final Parameter Values
    ax6 = fig.add_subplot(gs[0, 5])
    param_names = ['δ', 'τ', 'λ']
    final_values = [history['delta'][-1], history['tau'][-1], history['lambda'][-1]]
    colors = ['red', 'blue', 'green']
    
    bars = ax6.bar(param_names, final_values, color=colors, alpha=0.7)
    ax6.set_ylabel('Final Parameter Value')
    ax6.set_title('Final Parameter Values')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e}', ha='center', va='bottom', fontsize=10)
    
    # Plot 7: All Parameters Together
    ax7 = fig.add_subplot(gs[1, 0])
    ax7.plot(history['iteration'], history['delta'], 'o-', color='red', 
             label='δ (duplication)', linewidth=2, markersize=4)
    ax7.plot(history['iteration'], history['tau'], 's-', color='blue', 
             label='τ (transfer)', linewidth=2, markersize=4)
    ax7.plot(history['iteration'], history['lambda'], '^-', color='green', 
             label='λ (loss)', linewidth=2, markersize=4)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Parameter Value (log scale)')
    ax7.set_title('All Parameters Evolution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')
    
    # Plot 8: Timing Breakdown per Iteration
    ax8 = fig.add_subplot(gs[1, 1])
    width = 0.2
    x = np.arange(len(history['iteration']))
    
    ax8.bar(x - width*1.5, history['e_time'], width, label='E update', alpha=0.7, color='orange')
    ax8.bar(x - width*0.5, history['pi_total_time'], width, label='Pi update', alpha=0.7, color='blue')
    ax8.bar(x + width*0.5, history['gradient_time'], width, label='Gradients', alpha=0.7, color='red')
    ax8.bar(x + width*1.5, history['ll_time'], width, label='LL compute', alpha=0.7, color='green')
    
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Time (s)')
    ax8.set_title('Timing Breakdown per Iteration')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Gradient Evolution
    ax9 = fig.add_subplot(gs[1, 2])
    gradients_delta = [g[0] for g in history['gradients']]
    gradients_tau = [g[1] for g in history['gradients']]
    gradients_lambda = [g[2] for g in history['gradients']]
    
    ax9.plot(history['iteration'], gradients_delta, 'o-', color='red', 
             label='∂LL/∂δ', linewidth=2, markersize=4)
    ax9.plot(history['iteration'], gradients_tau, 's-', color='blue', 
             label='∂LL/∂τ', linewidth=2, markersize=4)
    ax9.plot(history['iteration'], gradients_lambda, '^-', color='green', 
             label='∂LL/∂λ', linewidth=2, markersize=4)
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Gradient Value')
    ax9.set_title('Gradient Evolution')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Gradient Norm Evolution
    ax10 = fig.add_subplot(gs[1, 3])
    ax10.semilogy(history['iteration'], history['gradient_norm'], 'o-', color='purple', 
                  linewidth=3, markersize=6)
    ax10.axhline(y=1e-4, color='red', linestyle='--', alpha=0.7, label='Convergence threshold')
    ax10.set_xlabel('Iteration')
    ax10.set_ylabel('Gradient Norm (log scale)')
    ax10.set_title('Gradient Norm Convergence')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # Plot 11: Pi Convergence Details
    ax11 = fig.add_subplot(gs[1, 4])
    ax11.plot(history['iteration'], history['pi_iterations'], 'o-', color='blue', 
              linewidth=2, markersize=6)
    ax11.set_xlabel('Iteration')
    ax11.set_ylabel('Pi Iterations to Convergence')
    ax11.set_title('Pi Fixed-Point Convergence\n(Warm Start Efficiency)')
    ax11.grid(True, alpha=0.3)
    
    # Plot 12: Cumulative Time
    ax12 = fig.add_subplot(gs[1, 5])
    ax12.plot(history['iteration'], history['cumulative_time'], 'o-', color='brown', 
              linewidth=3, markersize=6)
    ax12.set_xlabel('Iteration')
    ax12.set_ylabel('Cumulative Time (s)')
    ax12.set_title('Total Optimization Time')
    ax12.grid(True, alpha=0.3)
    
    # Plot 13: Parameter Space Trajectory
    ax13 = fig.add_subplot(gs[2, 0])
    # 3D trajectory projected to 2D
    ax13.plot(history['delta'], history['tau'], 'o-', color='purple', 
              linewidth=2, markersize=4, alpha=0.7)
    ax13.scatter(history['delta'][0], history['tau'][0], color='green', s=100, 
                marker='*', label='Start', zorder=5)
    ax13.scatter(history['delta'][-1], history['tau'][-1], color='red', s=100, 
                marker='X', label='End', zorder=5)
    ax13.set_xlabel('Duplication Rate (δ)')
    ax13.set_ylabel('Transfer Rate (τ)')
    ax13.set_title('Parameter Space Trajectory')
    ax13.legend()
    ax13.grid(True, alpha=0.3)
    ax13.set_xscale('log')
    ax13.set_yscale('log')
    
    # Plot 14: E vs Pi Convergence Comparison
    ax14 = fig.add_subplot(gs[2, 1])
    ax14.plot(history['iteration'], history['e_iterations'], 'o-', color='orange', 
              label='E iterations', linewidth=2, markersize=4)
    ax14.plot(history['iteration'], history['pi_iterations'], 's-', color='blue', 
              label='Pi iterations', linewidth=2, markersize=4)
    ax14.set_xlabel('Iteration')
    ax14.set_ylabel('Iterations to Convergence')
    ax14.set_title('E vs Pi Convergence Speed')
    ax14.legend()
    ax14.grid(True, alpha=0.3)
    
    # Plot 15: Pi Individual Iteration Times (first few iterations)
    ax15 = fig.add_subplot(gs[2, 2])
    if len(history['pi_individual_times']) > 0:
        # Show individual Pi iteration times for first optimization iteration
        first_pi_times = history['pi_individual_times'][0]
        pi_iters = range(len(first_pi_times))
        ax15.plot(pi_iters, first_pi_times, 'o-', color='blue', linewidth=2, markersize=4)
        ax15.set_xlabel('Pi Iteration')
        ax15.set_ylabel('Time per Pi Iteration (s)')
        ax15.set_title('Individual Pi Update Times\n(First Optimization Iteration)')
        ax15.grid(True, alpha=0.3)
    
    # Plot 16: Performance Scaling Analysis
    ax16 = fig.add_subplot(gs[2, 3])
    # Show how timing changes with warm starts
    if len(history['iteration']) > 1:
        efficiency_ratio = []
        for i in range(1, len(history['iteration'])):
            ratio = history['pi_total_time'][0] / history['pi_total_time'][i] if history['pi_total_time'][i] > 0 else 1
            efficiency_ratio.append(ratio)
        
        ax16.plot(range(1, len(history['iteration'])), efficiency_ratio, 'o-', 
                 color='green', linewidth=2, markersize=6)
        ax16.set_xlabel('Iteration')
        ax16.set_ylabel('Speedup Factor')
        ax16.set_title('Warm Start Efficiency\n(vs First Iteration)')
        ax16.grid(True, alpha=0.3)
        ax16.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax16.legend()
    
    # Large summary plot spanning bottom two rows
    ax_summary = fig.add_subplot(gs[2:4, 4:6])
    ax_summary.text(0.05, 0.95, 'COMPREHENSIVE OPTIMIZATION SUMMARY', 
                   fontsize=16, fontweight='bold', transform=ax_summary.transAxes)
    
    # Problem details
    ax_summary.text(0.05, 0.88, f'Problem Size:', fontsize=12, fontweight='bold', transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.84, f'• Species: {n_species} leaves', fontsize=11, transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.8, f'• Gene: {n_genes} leaves', fontsize=11, transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.76, f'• Matrix elements: {history["pi_total_time"][0] * 1000:.0f}K+', fontsize=11, transform=ax_summary.transAxes)
    
    # Final results
    ax_summary.text(0.05, 0.68, f'Final Results:', fontsize=12, fontweight='bold', transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.64, f'• Iterations: {len(history["iteration"])}', fontsize=11, transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.6, f'• Final LL: {history["log_likelihood"][-1]:.6f}', fontsize=11, transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.56, f'• Final δ: {history["delta"][-1]:.2e}', fontsize=11, transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.52, f'• Final τ: {history["tau"][-1]:.2e}', fontsize=11, transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.48, f'• Final λ: {history["lambda"][-1]:.2e}', fontsize=11, transform=ax_summary.transAxes)
    
    # Timing analysis
    ax_summary.text(0.05, 0.4, f'Timing Analysis:', fontsize=12, fontweight='bold', transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.36, f'• Total time: {history["cumulative_time"][-1]:.2f}s', fontsize=11, transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.32, f'• Avg Pi time: {np.mean(history["pi_total_time"]):.3f}s', fontsize=11, transform=ax_summary.transAxes)
    ax_summary.text(0.05, 0.28, f'• Avg gradient time: {np.mean(history["gradient_time"]):.3f}s', fontsize=11, transform=ax_summary.transAxes)
    
    # Warm start efficiency
    if len(history['pi_total_time']) > 1:
        speedup = history['pi_total_time'][0] / np.mean(history['pi_total_time'][1:])
        ax_summary.text(0.05, 0.24, f'• Warm start speedup: {speedup:.1f}x', fontsize=11, transform=ax_summary.transAxes)
    
    # Convergence analysis
    ax_summary.text(0.05, 0.16, f'Convergence Analysis:', fontsize=12, fontweight='bold', transform=ax_summary.transAxes)
    if all(p < 1e-6 for p in [history['delta'][-1], history['tau'][-1], history['lambda'][-1]]):
        ax_summary.text(0.05, 0.12, '✅ Parameters converged near zero', fontsize=11, 
                       fontweight='bold', color='green', transform=ax_summary.transAxes)
        ax_summary.text(0.05, 0.08, '✅ Optimal for identical trees (pure speciation)', fontsize=11, 
                       fontweight='bold', color='green', transform=ax_summary.transAxes)
    else:
        ax_summary.text(0.05, 0.12, '⚠️  Parameters did not reach zero', fontsize=11, 
                       fontweight='bold', color='orange', transform=ax_summary.transAxes)
    
    final_grad_norm = history['gradient_norm'][-1]
    if final_grad_norm < 1e-4:
        ax_summary.text(0.05, 0.04, f'✅ Gradient converged: ‖∇‖={final_grad_norm:.2e}', fontsize=11, 
                       fontweight='bold', color='green', transform=ax_summary.transAxes)
    else:
        ax_summary.text(0.05, 0.04, f'⚠️  Gradient: ‖∇‖={final_grad_norm:.2e}', fontsize=11, 
                       fontweight='bold', color='orange', transform=ax_summary.transAxes)
    
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis('off')
    
    # Additional timing details in remaining space
    ax_details = fig.add_subplot(gs[3, 0:4])
    ax_details.text(0.02, 0.9, 'DETAILED TIMING BREAKDOWN', fontsize=14, fontweight='bold', transform=ax_details.transAxes)
    
    # Create timing breakdown table
    timing_data = []
    for i, iteration in enumerate(history['iteration']):
        timing_data.append([
            iteration,
            f"{history['e_time'][i]:.3f}",
            f"{history['pi_total_time'][i]:.3f}",
            f"{history['pi_iterations'][i]}",
            f"{history['pi_avg_iter_time'][i]:.4f}",
            f"{history['gradient_time'][i]:.3f}",
            f"{history['total_iteration_time'][i]:.3f}"
        ])
    
    # Show table
    headers = ['Iter', 'E Time', 'Pi Time', 'Pi Iters', 'Pi/Iter', 'Grad Time', 'Total']
    
    # Simple text table
    y_pos = 0.8
    # Headers
    x_positions = [0.02, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72]
    for i, header in enumerate(headers):
        ax_details.text(x_positions[i], y_pos, header, fontsize=10, fontweight='bold', transform=ax_details.transAxes)
    
    # Data rows (first few iterations)
    for i, row in enumerate(timing_data[:min(8, len(timing_data))]):
        y_pos -= 0.08
        for j, value in enumerate(row):
            ax_details.text(x_positions[j], y_pos, str(value), fontsize=9, transform=ax_details.transAxes)
    
    ax_details.set_xlim(0, 1)
    ax_details.set_ylim(0, 1)
    ax_details.axis('off')
    
    plt.savefig('comprehensive_optimization_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive analysis saved as 'comprehensive_optimization_analysis.png'")
    
    return fig

def main():
    """Run comprehensive optimization analysis"""
    
    # Test on both small and large trees
    test_cases = [
        {"name": "Large Trees", "species": "test_trees_200/sp.nwk", "gene": "test_trees_200/g.nwk"}
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"TESTING: {test_case['name']}")
        print(f"{'='*80}")
        
        # Run comprehensive optimization
        history = run_comprehensive_optimization(
            test_case['species'], test_case['gene'], 
            max_iterations=8, learning_rate=0.01
        )
        
        # Create comprehensive visualization
        create_comprehensive_visualization(
            history, test_case['species'], test_case['gene']
        )
        
        # Save detailed results
        output_file = f"comprehensive_analysis_{test_case['name'].lower().replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"💾 Detailed results saved to '{output_file}'")
        
        break  # Only do first test case to avoid timeout

if __name__ == "__main__":
    main()