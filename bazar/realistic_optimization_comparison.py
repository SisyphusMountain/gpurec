#!/usr/bin/env python3
"""
Realistic Newton vs Gradient Descent Comparison for Phylogenetic Optimization
============================================================================

This implements the realistic optimization flow where:
1. One-time setup: Parse trees, build CCP helpers
2. Each iteration: Update E and Pi using previous iteration as warm start
3. Measure only incremental optimization costs, not setup costs
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import time
from typing import Dict, Tuple
from ete3 import Tree

# Import the CCP functions we need
from matmul_ale_ccp_log import (
    build_ccp_from_single_tree, build_species_helpers,
    build_clade_species_mapping, build_ccp_helpers,
    get_root_clade_id, Pi_update_ccp_log
)
from matmul_ale_ccp import E_step

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True

class RealisticOptimizer:
    """Realistic optimizer that reuses matrices between iterations"""
    
    def __init__(self, species_path: str, gene_path: str, device: str = "cpu"):
        print(f"🔧 Setting up realistic optimizer...")
        setup_start = time.time()
        
        self.device = torch.device(device)
        self.dtype = torch.float64
        
        # Store paths
        self.species_path = species_path
        self.gene_path = gene_path
        
        # Parse trees (one-time setup)
        print(f"   📖 Parsing trees...")
        self.species_tree = Tree(species_path, format=1)
        self.gene_tree = Tree(gene_path, format=1)
        
        # Build CCP and helpers (one-time setup)
        print(f"   🔨 Building CCP and helpers...")
        self.ccp = build_ccp_from_single_tree(gene_path)
        self.species_helpers = build_species_helpers(species_path, self.device, self.dtype)
        self.clade_species_mapping = build_clade_species_mapping(self.ccp, self.species_helpers, self.device, self.dtype)
        self.ccp_helpers = build_ccp_helpers(self.ccp, self.device, self.dtype)
        self.root_clade_id = get_root_clade_id(self.ccp)
        
        # Problem dimensions
        self.n_clades = len(self.ccp.clades)
        self.n_species = self.species_helpers["S"]
        
        # Initialize state matrices (will be reused between iterations)
        self.E = None  # Extinction probabilities
        self.Pi = None  # Likelihood matrix
        
        setup_time = time.time() - setup_start
        print(f"   ✅ Setup complete in {setup_time:.2f}s")
        print(f"   📊 Problem size: {self.n_clades} clades × {self.n_species} species = {self.n_clades * self.n_species:,} matrix elements")
    
    def update_E_convergence(self, delta: float, tau: float, lam: float, 
                           max_iterations: int = 50, tolerance: float = 1e-8) -> Tuple[torch.Tensor, int]:
        """Update E to convergence, starting from previous E if available"""
        
        # Initialize E if first iteration
        if self.E is None:
            self.E = torch.zeros(self.n_species, device=self.device, dtype=self.dtype)
        
        # Compute event probabilities
        rates_sum = 1.0 + delta + tau + lam
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum
        p_T = tau / rates_sum
        p_L = lam / rates_sum
        
        # Run E fixed-point iteration starting from current E
        E_current = self.E.clone()
        for iteration in range(max_iterations):
            E_next, E_s1, E_s2, Ebar = E_step(
                E_current,
                self.species_helpers['s_C1'],
                self.species_helpers['s_C2'], 
                self.species_helpers['Recipients_mat'],
                p_S, p_D, p_T, p_L
            )
            
            # Check convergence
            if iteration > 0:
                diff = torch.abs(E_next - E_current).max()
                if diff < tolerance:
                    break
            
            E_current = E_next
        
        # Update stored E and return results
        self.E = E_current
        return E_current, iteration + 1
    
    def update_Pi_convergence(self, delta: float, tau: float, lam: float, E: torch.Tensor,
                            max_iterations: int = 50, tolerance: float = 1e-8) -> Tuple[torch.Tensor, int]:
        """Update Pi to convergence, starting from previous Pi if available"""
        
        # Initialize Pi if first iteration
        if self.Pi is None:
            # Initialize leaf clades
            log_Pi = torch.full((self.n_clades, self.n_species), float('-inf'), 
                              device=self.device, dtype=self.dtype)
            
            # Set leaf probabilities based on clade-species mapping
            for c in range(self.n_clades):
                clade = self.ccp.id_to_clade[c]
                if clade.is_leaf():
                    # Find which species this leaf belongs to
                    mapped_species = torch.nonzero(self.clade_species_mapping[c] > 0, as_tuple=False).flatten()
                    if len(mapped_species) > 0:
                        # Distribute probability uniformly among mapped species
                        log_prob = -torch.log(torch.tensor(len(mapped_species), dtype=self.dtype))
                        log_Pi[c, mapped_species] = log_prob
            
            self.Pi = log_Pi
        
        # Compute event probabilities and Ebar
        rates_sum = 1.0 + delta + tau + lam
        p_S = 1.0 / rates_sum
        p_D = delta / rates_sum  
        p_T = tau / rates_sum
        p_L = lam / rates_sum
        
        # We need Ebar for Pi update
        Ebar = torch.mv(self.species_helpers['Recipients_mat'], E)
        
        # Run Pi fixed-point iteration starting from current Pi
        log_Pi_current = self.Pi.clone()
        for iteration in range(max_iterations):
            log_Pi_new = Pi_update_ccp_log(
                log_Pi_current, self.ccp_helpers, self.species_helpers, 
                self.clade_species_mapping, E, Ebar, p_S, p_D, p_T
            )
            
            # Check convergence in log space
            if iteration > 0:
                diff = torch.abs(log_Pi_new - log_Pi_current).max()
                if diff < tolerance:
                    break
            
            log_Pi_current = log_Pi_new
        
        # Update stored Pi
        self.Pi = log_Pi_current
        return log_Pi_current, iteration + 1
    
    def compute_likelihood_and_timing(self, delta: float, tau: float, lam: float) -> Dict:
        """Compute likelihood with detailed timing breakdown"""
        
        # Time E convergence
        e_start = time.time()
        E, e_iterations = self.update_E_convergence(delta, tau, lam)
        e_time = time.time() - e_start
        
        # Time Pi convergence  
        pi_start = time.time()
        Pi, pi_iterations = self.update_Pi_convergence(delta, tau, lam, E)
        pi_time = time.time() - pi_start
        
        # Compute log-likelihood
        ll_start = time.time()
        root_log_probs = Pi[self.root_clade_id, :]
        log_likelihood = torch.logsumexp(root_log_probs, dim=0).item()
        ll_time = time.time() - ll_start
        
        return {
            'log_likelihood': log_likelihood,
            'e_time': e_time,
            'pi_time': pi_time,
            'll_time': ll_time,
            'total_time': e_time + pi_time + ll_time,
            'e_iterations': e_iterations,
            'pi_iterations': pi_iterations
        }
    
    def compute_gradients_realistic(self, delta: float, tau: float, lam: float, 
                                  epsilon: float = 1e-6) -> Tuple[torch.Tensor, Dict]:
        """Compute gradients using realistic finite differences with warm starts"""
        
        # Base likelihood
        base_result = self.compute_likelihood_and_timing(delta, tau, lam)
        base_ll = base_result['log_likelihood']
        
        gradients = torch.zeros(3, dtype=self.dtype)
        timing_info = {
            'base_time': base_result['total_time'],
            'gradient_times': [],
            'total_gradient_time': 0.0
        }
        
        # Save current state for restoration
        E_saved = self.E.clone() if self.E is not None else None
        Pi_saved = self.Pi.clone() if self.Pi is not None else None
        
        params = [delta, tau, lam]
        param_names = ['delta', 'tau', 'lambda']
        
        grad_start = time.time()
        for i in range(3):
            # Positive perturbation
            params_pos = params.copy()
            params_pos[i] += epsilon
            
            # Restore state for consistent starting point
            if E_saved is not None:
                self.E = E_saved.clone()
            if Pi_saved is not None:
                self.Pi = Pi_saved.clone()
                
            pos_result = self.compute_likelihood_and_timing(*params_pos)
            
            # Negative perturbation
            params_neg = params.copy()
            params_neg[i] -= epsilon
            
            # Restore state for consistent starting point
            if E_saved is not None:
                self.E = E_saved.clone()
            if Pi_saved is not None:
                self.Pi = Pi_saved.clone()
                
            neg_result = self.compute_likelihood_and_timing(*params_neg)
            
            # Central difference gradient
            gradients[i] = (pos_result['log_likelihood'] - neg_result['log_likelihood']) / (2 * epsilon)
            
            # Track timing
            param_grad_time = pos_result['total_time'] + neg_result['total_time']
            timing_info['gradient_times'].append(param_grad_time)
        
        timing_info['total_gradient_time'] = time.time() - grad_start
        
        # Restore original state
        if E_saved is not None:
            self.E = E_saved
        if Pi_saved is not None:
            self.Pi = Pi_saved
        
        return gradients, timing_info

def run_realistic_gradient_descent(species_path: str, gene_path: str, 
                                 max_iterations: int = 15, learning_rate: float = 0.01) -> Dict:
    """Run realistic gradient descent optimization"""
    
    print(f"🚀 REALISTIC GRADIENT DESCENT OPTIMIZATION")
    print(f"=" * 60)
    
    optimizer = RealisticOptimizer(species_path, gene_path)
    
    # Initialize parameters
    params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    
    # Track optimization history
    history = {
        'iteration': [],
        'log_likelihood': [],
        'delta': [], 'tau': [], 'lambda': [],
        'gradient_norm': [],
        'e_time': [], 'pi_time': [], 'll_time': [], 'gradient_time': [],
        'total_iteration_time': [],
        'cumulative_time': [],
        'e_iterations': [], 'pi_iterations': []
    }
    
    cumulative_time = 0.0
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Compute current likelihood with timing
        ll_result = optimizer.compute_likelihood_and_timing(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        # Compute gradients with timing
        gradients, grad_timing = optimizer.compute_gradients_realistic(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        gradient_norm = torch.norm(gradients).item()
        
        # Gradient descent update
        params = params + learning_rate * gradients
        params = torch.clamp(params, min=1e-6)  # Keep positive
        
        iter_time = time.time() - iter_start
        cumulative_time += iter_time
        
        # Record history
        history['iteration'].append(iteration)
        history['log_likelihood'].append(ll_result['log_likelihood'])
        history['delta'].append(float(params[0]))
        history['tau'].append(float(params[1]))
        history['lambda'].append(float(params[2]))
        history['gradient_norm'].append(gradient_norm)
        history['e_time'].append(ll_result['e_time'])
        history['pi_time'].append(ll_result['pi_time'])
        history['ll_time'].append(ll_result['ll_time'])
        history['gradient_time'].append(grad_timing['total_gradient_time'])
        history['total_iteration_time'].append(iter_time)
        history['cumulative_time'].append(cumulative_time)
        history['e_iterations'].append(ll_result['e_iterations'])
        history['pi_iterations'].append(ll_result['pi_iterations'])
        
        print(f"Iter {iteration:2d}: LL={ll_result['log_likelihood']:8.4f}, "
              f"δ={params[0]:.4f}, τ={params[1]:.4f}, λ={params[2]:.4f}, "
              f"‖∇‖={gradient_norm:.2e}, "
              f"E:{ll_result['e_iterations']:2d}it/{ll_result['e_time']:.2f}s, "
              f"Pi:{ll_result['pi_iterations']:2d}it/{ll_result['pi_time']:.2f}s, "
              f"total:{iter_time:.1f}s")
        
        # Check convergence
        if gradient_norm < 1e-6:
            print(f"✅ Converged! Gradient norm {gradient_norm:.2e} below threshold")
            break
    
    return history

def run_realistic_newton_method(species_path: str, gene_path: str,
                               max_iterations: int = 15, learning_rate: float = 1.0) -> Dict:
    """Run realistic Newton's method optimization"""
    
    print(f"🚀 REALISTIC NEWTON'S METHOD OPTIMIZATION")
    print(f"=" * 60)
    
    optimizer = RealisticOptimizer(species_path, gene_path)
    
    # Initialize parameters
    params = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    
    # Track optimization history
    history = {
        'iteration': [],
        'log_likelihood': [],
        'delta': [], 'tau': [], 'lambda': [],
        'gradient_norm': [],
        'e_time': [], 'pi_time': [], 'll_time': [], 'gradient_time': [], 'hessian_time': [],
        'total_iteration_time': [],
        'cumulative_time': [],
        'e_iterations': [], 'pi_iterations': [],
        'll_improvement': []
    }
    
    cumulative_time = 0.0
    previous_ll = None
    
    for iteration in range(max_iterations):
        iter_start = time.time()
        
        # Compute current likelihood with timing
        ll_result = optimizer.compute_likelihood_and_timing(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        # Compute gradients with timing
        gradients, grad_timing = optimizer.compute_gradients_realistic(
            float(params[0]), float(params[1]), float(params[2])
        )
        
        # Compute diagonal Hessian approximation
        hess_start = time.time()
        hessian_diag = torch.zeros_like(params)
        epsilon = 1e-6
        
        # Save state for Hessian computation
        E_saved = optimizer.E.clone() if optimizer.E is not None else None
        Pi_saved = optimizer.Pi.clone() if optimizer.Pi is not None else None
        
        for i in range(3):
            # Forward difference for second derivative
            params_pos = params.clone()
            params_pos[i] += epsilon
            
            # Restore state
            if E_saved is not None:
                optimizer.E = E_saved.clone()
            if Pi_saved is not None:
                optimizer.Pi = Pi_saved.clone()
            ll_pos = optimizer.compute_likelihood_and_timing(
                float(params_pos[0]), float(params_pos[1]), float(params_pos[2])
            )['log_likelihood']
            
            params_neg = params.clone()
            params_neg[i] -= epsilon
            
            # Restore state
            if E_saved is not None:
                optimizer.E = E_saved.clone()
            if Pi_saved is not None:
                optimizer.Pi = Pi_saved.clone()
            ll_neg = optimizer.compute_likelihood_and_timing(
                float(params_neg[0]), float(params_neg[1]), float(params_neg[2])
            )['log_likelihood']
            
            # Second derivative approximation
            hessian_diag[i] = (ll_pos - 2*ll_result['log_likelihood'] + ll_neg) / (epsilon**2)
        
        # Restore original state
        if E_saved is not None:
            optimizer.E = E_saved
        if Pi_saved is not None:
            optimizer.Pi = Pi_saved
        
        hess_time = time.time() - hess_start
        
        gradient_norm = torch.norm(gradients).item()
        
        # Newton update with regularization
        hessian_diag_reg = torch.where(torch.abs(hessian_diag) < 1e-8,
                                     torch.sign(hessian_diag) * 1e-8,
                                     hessian_diag)
        newton_step = gradients / hessian_diag_reg
        params = params + learning_rate * newton_step
        params = torch.clamp(params, min=1e-6)  # Keep positive
        
        iter_time = time.time() - iter_start
        cumulative_time += iter_time
        
        # Compute log-likelihood improvement
        ll_improvement = ll_result['log_likelihood'] - previous_ll if previous_ll is not None else 0.0
        
        # Record history
        history['iteration'].append(iteration)
        history['log_likelihood'].append(ll_result['log_likelihood'])
        history['delta'].append(float(params[0]))
        history['tau'].append(float(params[1]))
        history['lambda'].append(float(params[2]))
        history['gradient_norm'].append(gradient_norm)
        history['e_time'].append(ll_result['e_time'])
        history['pi_time'].append(ll_result['pi_time'])
        history['ll_time'].append(ll_result['ll_time'])
        history['gradient_time'].append(grad_timing['total_gradient_time'])
        history['hessian_time'].append(hess_time)
        history['total_iteration_time'].append(iter_time)
        history['cumulative_time'].append(cumulative_time)
        history['e_iterations'].append(ll_result['e_iterations'])
        history['pi_iterations'].append(ll_result['pi_iterations'])
        history['ll_improvement'].append(ll_improvement)
        
        print(f"Iter {iteration:2d}: LL={ll_result['log_likelihood']:8.4f}, "
              f"δ={params[0]:.4f}, τ={params[1]:.4f}, λ={params[2]:.4f}, "
              f"‖∇‖={gradient_norm:.2e}, ΔLL={ll_improvement:.2e}, "
              f"E:{ll_result['e_iterations']:2d}it/{ll_result['e_time']:.2f}s, "
              f"Pi:{ll_result['pi_iterations']:2d}it/{ll_result['pi_time']:.2f}s, "
              f"total:{iter_time:.1f}s")
        
        # Check convergence
        if previous_ll is not None and abs(ll_improvement) < 1e-8:
            print(f"✅ Converged! LL improvement {abs(ll_improvement):.2e} below threshold")
            break
            
        if gradient_norm < 1e-6:
            print(f"✅ Converged! Gradient norm {gradient_norm:.2e} below threshold")
            break
        
        previous_ll = ll_result['log_likelihood']
    
    return history

def create_realistic_comparison_plot(gd_history: Dict, newton_history: Dict, 
                                   species_path: str, gene_path: str):
    """Create comprehensive realistic comparison plot"""
    
    species_tree = Tree(species_path, format=1)
    gene_tree = Tree(gene_path, format=1)
    n_species = len(species_tree.get_leaves())
    n_genes = len(gene_tree.get_leaves())
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f'Realistic Optimization: Newton vs Gradient Descent\n'
                f'(Species: {n_species} leaves, Gene: {n_genes} leaves)', 
                fontsize=16, fontweight='bold')
    
    # Colors
    gd_color = '#2E86AB'
    newton_color = '#A23B72'
    
    # Plot 1: Log-likelihood convergence
    ax1 = axes[0, 0]
    ax1.plot(gd_history['iteration'], gd_history['log_likelihood'], 
             'o-', color=gd_color, label='Gradient Descent', linewidth=2, markersize=4)
    ax1.plot(newton_history['iteration'], newton_history['log_likelihood'], 
             's-', color=newton_color, label="Newton's Method", linewidth=2, markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: E convergence time per iteration
    ax2 = axes[0, 1]
    ax2.plot(gd_history['iteration'], gd_history['e_time'], 
             'o-', color=gd_color, label='GD E time', linewidth=2, markersize=4)
    ax2.plot(newton_history['iteration'], newton_history['e_time'], 
             's-', color=newton_color, label='Newton E time', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('E Convergence Time (s)')
    ax2.set_title('Extinction Probability Update Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Pi convergence time per iteration
    ax3 = axes[0, 2]
    ax3.plot(gd_history['iteration'], gd_history['pi_time'], 
             'o-', color=gd_color, label='GD Pi time', linewidth=2, markersize=4)
    ax3.plot(newton_history['iteration'], newton_history['pi_time'], 
             's-', color=newton_color, label='Newton Pi time', linewidth=2, markersize=4)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Pi Convergence Time (s)')
    ax3.set_title('Likelihood Matrix Update Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: E iterations per optimization step
    ax4 = axes[1, 0]
    ax4.plot(gd_history['iteration'], gd_history['e_iterations'], 
             'o-', color=gd_color, label='GD E iterations', linewidth=2, markersize=4)
    ax4.plot(newton_history['iteration'], newton_history['e_iterations'], 
             's-', color=newton_color, label='Newton E iterations', linewidth=2, markersize=4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('E Iterations to Convergence')
    ax4.set_title('E Fixed-Point Iterations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Pi iterations per optimization step
    ax5 = axes[1, 1]
    ax5.plot(gd_history['iteration'], gd_history['pi_iterations'], 
             'o-', color=gd_color, label='GD Pi iterations', linewidth=2, markersize=4)
    ax5.plot(newton_history['iteration'], newton_history['pi_iterations'], 
             's-', color=newton_color, label='Newton Pi iterations', linewidth=2, markersize=4)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Pi Iterations to Convergence')
    ax5.set_title('Pi Fixed-Point Iterations')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Total iteration time breakdown
    ax6 = axes[1, 2]
    width = 0.35
    x = np.arange(min(len(gd_history['iteration']), len(newton_history['iteration'])))
    
    # Stack E, Pi, and gradient times
    gd_e_times = gd_history['e_time'][:len(x)]
    gd_pi_times = gd_history['pi_time'][:len(x)]
    gd_grad_times = gd_history['gradient_time'][:len(x)]
    
    newton_e_times = newton_history['e_time'][:len(x)]
    newton_pi_times = newton_history['pi_time'][:len(x)]
    newton_grad_times = newton_history['gradient_time'][:len(x)]
    
    ax6.bar(x - width/2, gd_e_times, width, label='GD E time', color=gd_color, alpha=0.7)
    ax6.bar(x - width/2, gd_pi_times, width, bottom=gd_e_times, label='GD Pi time', color=gd_color, alpha=0.5)
    ax6.bar(x - width/2, gd_grad_times, width, 
            bottom=[e+p for e,p in zip(gd_e_times, gd_pi_times)], label='GD Grad time', color=gd_color, alpha=0.3)
    
    ax6.bar(x + width/2, newton_e_times, width, label='Newton E time', color=newton_color, alpha=0.7)
    ax6.bar(x + width/2, newton_pi_times, width, bottom=newton_e_times, label='Newton Pi time', color=newton_color, alpha=0.5)
    ax6.bar(x + width/2, newton_grad_times, width,
            bottom=[e+p for e,p in zip(newton_e_times, newton_pi_times)], label='Newton Grad time', color=newton_color, alpha=0.3)
    
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Time (s)')
    ax6.set_title('Time Breakdown per Iteration')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Cumulative time comparison
    ax7 = axes[2, 0]
    ax7.plot(gd_history['iteration'], gd_history['cumulative_time'], 
             'o-', color=gd_color, label='Gradient Descent', linewidth=2, markersize=4)
    ax7.plot(newton_history['iteration'], newton_history['cumulative_time'], 
             's-', color=newton_color, label="Newton's Method", linewidth=2, markersize=4)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Cumulative Time (s)')
    ax7.set_title('Total Time Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Parameter evolution - Delta
    ax8 = axes[2, 1]
    ax8.plot(gd_history['iteration'], gd_history['delta'], 
             'o-', color=gd_color, label='GD δ', linewidth=2, markersize=4)
    ax8.plot(newton_history['iteration'], newton_history['delta'], 
             's-', color=newton_color, label='Newton δ', linewidth=2, markersize=4)
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Duplication Rate (δ)')
    ax8.set_title('Parameter Evolution: δ')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Gradient norm convergence
    ax9 = axes[2, 2]
    ax9.semilogy(gd_history['iteration'], gd_history['gradient_norm'], 
                 'o-', color=gd_color, label='Gradient Descent', linewidth=2, markersize=4)
    ax9.semilogy(newton_history['iteration'], newton_history['gradient_norm'], 
                 's-', color=newton_color, label="Newton's Method", linewidth=2, markersize=4)
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Gradient Norm (log scale)')
    ax9.set_title('Gradient Norm Convergence')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Efficiency metrics
    ax10 = axes[3, 0]
    methods = ['Gradient\nDescent', 'Newton\nMethod']
    total_times = [gd_history['cumulative_time'][-1], newton_history['cumulative_time'][-1]]
    iterations = [len(gd_history['iteration']), len(newton_history['iteration'])]
    
    bars = ax10.bar(methods, total_times, color=[gd_color, newton_color], alpha=0.7)
    ax10.set_ylabel('Total Time (s)')
    ax10.set_title('Time to Convergence')
    ax10.grid(True, alpha=0.3)
    
    # Add iteration counts as text
    for bar, iter_count in zip(bars, iterations):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                f'{iter_count} iterations', ha='center', va='bottom', fontsize=10)
    
    # Plot 11: Warm start efficiency (E and Pi iterations over time)
    ax11 = axes[3, 1]
    ax11.plot(gd_history['iteration'][1:], gd_history['e_iterations'][1:], 
             'o-', color=gd_color, label='GD E iterations', linewidth=2, markersize=4, alpha=0.7)
    ax11.plot(gd_history['iteration'][1:], gd_history['pi_iterations'][1:], 
             '^-', color=gd_color, label='GD Pi iterations', linewidth=2, markersize=4, alpha=0.5)
    ax11.plot(newton_history['iteration'][1:], newton_history['e_iterations'][1:], 
             's-', color=newton_color, label='Newton E iterations', linewidth=2, markersize=4, alpha=0.7)
    ax11.plot(newton_history['iteration'][1:], newton_history['pi_iterations'][1:], 
             'd-', color=newton_color, label='Newton Pi iterations', linewidth=2, markersize=4, alpha=0.5)
    ax11.set_xlabel('Iteration')
    ax11.set_ylabel('Iterations to Convergence')
    ax11.set_title('Warm Start Efficiency (iterations 1+)')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # Plot 12: Summary statistics
    ax12 = axes[3, 2]
    ax12.text(0.1, 0.9, 'REALISTIC OPTIMIZATION SUMMARY', fontsize=14, fontweight='bold', transform=ax12.transAxes)
    ax12.text(0.1, 0.8, f'Problem: {n_species} species, {n_genes} genes', fontsize=12, transform=ax12.transAxes)
    
    ax12.text(0.1, 0.7, 'GRADIENT DESCENT:', fontsize=12, fontweight='bold', color=gd_color, transform=ax12.transAxes)
    ax12.text(0.1, 0.65, f'• Total time: {gd_history["cumulative_time"][-1]:.1f}s', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.6, f'• Iterations: {len(gd_history["iteration"])}', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.55, f'• Avg E iters: {np.mean(gd_history["e_iterations"][1:]):.1f}', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.5, f'• Avg Pi iters: {np.mean(gd_history["pi_iterations"][1:]):.1f}', fontsize=11, transform=ax12.transAxes)
    
    ax12.text(0.1, 0.4, "NEWTON'S METHOD:", fontsize=12, fontweight='bold', color=newton_color, transform=ax12.transAxes)
    ax12.text(0.1, 0.35, f'• Total time: {newton_history["cumulative_time"][-1]:.1f}s', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.3, f'• Iterations: {len(newton_history["iteration"])}', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.25, f'• Avg E iters: {np.mean(newton_history["e_iterations"][1:]):.1f}', fontsize=11, transform=ax12.transAxes)
    ax12.text(0.1, 0.2, f'• Avg Pi iters: {np.mean(newton_history["pi_iterations"][1:]):.1f}', fontsize=11, transform=ax12.transAxes)
    
    speedup = gd_history["cumulative_time"][-1] / newton_history["cumulative_time"][-1]
    ax12.text(0.1, 0.1, f'SPEEDUP: {speedup:.1f}x faster', fontsize=12, fontweight='bold', 
              color='green', transform=ax12.transAxes)
    
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)
    ax12.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    output_name = f'realistic_optimization_comparison_{n_genes}_leaves.png'
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"📊 Realistic optimization comparison saved as '{output_name}'")
    
    return fig

def main():
    """Main function to run realistic optimization comparison"""
    
    print("🚀 REALISTIC OPTIMIZATION COMPARISON: Newton vs Gradient Descent")
    print("=" * 80)
    
    # Test on test_trees_200 (large trees)
    species_path = "test_trees_200/sp.nwk"
    gene_path = "test_trees_200/g.nwk"
    
    print(f"🧪 Testing on {species_path}")
    print("-" * 50)
    
    # Run realistic gradient descent
    gd_history = run_realistic_gradient_descent(species_path, gene_path, 
                                              max_iterations=10, learning_rate=0.01)
    
    print(f"\n")
    
    # Run realistic Newton's method
    newton_history = run_realistic_newton_method(species_path, gene_path,
                                                max_iterations=10, learning_rate=1.0)
    
    print(f"\n")
    
    # Create comparison visualization
    create_realistic_comparison_plot(gd_history, newton_history, species_path, gene_path)
    
    # Print detailed summary
    print("📈 REALISTIC OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    print(f"Gradient Descent:")
    print(f"  • Total time: {gd_history['cumulative_time'][-1]:.1f}s")
    print(f"  • Iterations: {len(gd_history['iteration'])}")
    print(f"  • Final LL: {gd_history['log_likelihood'][-1]:.6f}")
    print(f"  • Avg E iterations (after 1st): {np.mean(gd_history['e_iterations'][1:]):.1f}")
    print(f"  • Avg Pi iterations (after 1st): {np.mean(gd_history['pi_iterations'][1:]):.1f}")
    print(f"  • Total E time: {sum(gd_history['e_time']):.1f}s")
    print(f"  • Total Pi time: {sum(gd_history['pi_time']):.1f}s")
    
    print(f"\nNewton's Method:")
    print(f"  • Total time: {newton_history['cumulative_time'][-1]:.1f}s")
    print(f"  • Iterations: {len(newton_history['iteration'])}")
    print(f"  • Final LL: {newton_history['log_likelihood'][-1]:.6f}")
    print(f"  • Avg E iterations (after 1st): {np.mean(newton_history['e_iterations'][1:]):.1f}")
    print(f"  • Avg Pi iterations (after 1st): {np.mean(newton_history['pi_iterations'][1:]):.1f}")
    print(f"  • Total E time: {sum(newton_history['e_time']):.1f}s")
    print(f"  • Total Pi time: {sum(newton_history['pi_time']):.1f}s")
    
    speedup = gd_history['cumulative_time'][-1] / newton_history['cumulative_time'][-1]
    print(f"\n🏆 Newton's method is {speedup:.1f}x faster!")
    
    # Demonstrate warm start efficiency
    print(f"\n🔥 WARM START EFFICIENCY:")
    if len(gd_history['e_iterations']) > 1:
        print(f"  GD: E iterations drop from {gd_history['e_iterations'][0]} to {np.mean(gd_history['e_iterations'][1:]):.1f} avg")
        print(f"  GD: Pi iterations drop from {gd_history['pi_iterations'][0]} to {np.mean(gd_history['pi_iterations'][1:]):.1f} avg")
    if len(newton_history['e_iterations']) > 1:
        print(f"  Newton: E iterations drop from {newton_history['e_iterations'][0]} to {np.mean(newton_history['e_iterations'][1:]):.1f} avg")
        print(f"  Newton: Pi iterations drop from {newton_history['pi_iterations'][0]} to {np.mean(newton_history['pi_iterations'][1:]):.1f} avg")
    
    # Save detailed results
    results = {
        'gradient_descent': gd_history,
        'newton_method': newton_history,
        'summary': {
            'speedup': speedup,
            'problem_size': {
                'species_leaves': len(Tree(species_path, format=1).get_leaves()),
                'gene_leaves': len(Tree(gene_path, format=1).get_leaves())
            }
        }
    }
    
    with open('realistic_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Detailed results saved to 'realistic_optimization_results.json'")

if __name__ == "__main__":
    main()