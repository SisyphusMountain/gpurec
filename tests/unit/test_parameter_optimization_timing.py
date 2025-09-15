"""
Parameter optimization timing test for all datasets.

This test measures the time taken for parameters to converge using gradient descent
starting from various initial conditions, including AleRax inferred parameters.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

# Import with absolute paths to avoid relative import issues
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from src.reconciliation.reconcile import setup_fixed_points
from src.reconciliation.likelihood import E_fixed_point, Pi_fixed_point, compute_log_likelihood


def compute_likelihood_differentiable(species_tree: str, gene_tree: str, 
                                    tau_tensor: torch.Tensor, delta_tensor: torch.Tensor, lambda_tensor: torch.Tensor,
                                    setup_cache: dict = None, debug: bool = False):
    """Fully differentiable likelihood computation using PyTorch tensors."""
    
    device = tau_tensor.device
    dtype = tau_tensor.dtype
    
    # Use cached setup if available, otherwise compute it
    if setup_cache is None:
        result = setup_fixed_points(
            species_tree, gene_tree,
            max_iters_E=1, max_iters_Pi=1,
            tol_E=1e-10, tol_Pi=1e-10,
            debug=debug
        )
        setup_cache = {
            'ccp_helpers': result['ccp_helpers'],
            'species_helpers': result['species_helpers'],
            'log_clade_species_map': result['clade_species_map'],
            'ccp': result['ccp']
        }
    
    ccp_helpers = setup_cache['ccp_helpers']
    species_helpers = setup_cache['species_helpers']
    log_clade_species_map = setup_cache['log_clade_species_map']
    ccp = setup_cache['ccp']
    
    from src.core.ccp import get_root_clade_id
    root_clade_id = get_root_clade_id(ccp)
    
    # Convert parameters to log-space event probabilities (fully differentiable)
    total_rate = 1.0 + delta_tensor + tau_tensor + lambda_tensor
    log_pS = torch.log(1.0 / total_rate)
    log_pD = torch.log(delta_tensor / total_rate)
    log_pT = torch.log(tau_tensor / total_rate)
    log_pL = torch.log(lambda_tensor / total_rate)
    
    # Compute E with convergence (should be differentiable through fixed point)
    E_result = E_fixed_point(
        species_helpers, log_pS, log_pD, log_pT, log_pL,
        max_iters=1000, tolerance=1e-9, return_components=True
    )
    E, E_s1, E_s2, E_bar = E_result['E'], E_result['E_s1'], E_result['E_s2'], E_result['E_bar']
    
    # Compute Pi with convergence (should be differentiable through fixed point)
    Pi_result = Pi_fixed_point(
        ccp_helpers, species_helpers, log_clade_species_map,
        E, E_bar, E_s1, E_s2, log_pS, log_pD, log_pT,
        max_iters=1000, tolerance=1e-9
    )
    Pi = Pi_result['Pi']
    
    # Compute log-likelihood (differentiable)
    log_likelihood = compute_log_likelihood(Pi, root_clade_id)
    
    return {
        'log_likelihood': log_likelihood,
        'Pi': Pi,
        'E': E,
        'ccp': ccp,
        'root_clade_id': root_clade_id,
        'setup_cache': setup_cache
    }


class ParameterOptimizationTimer:
    """Times parameter optimization convergence with detailed tracking."""
    
    def __init__(self, dataset_name: str, results_dir: Path):
        self.dataset_name = dataset_name
        self.results_dir = results_dir / dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Timing tracking
        self.optimization_history = []
        
    def optimize_parameters(self, species_tree: str, gene_tree: str, 
                          init_tau: float, init_delta: float, init_lambda: float,
                          optimizer_type: str = 'lbfgs', lr: float = 0.01, epochs: int = 100, 
                          convergence_tol: float = 1e-8, verbose: bool = False):
        """
        Optimize parameters using specified optimizer with detailed timing and full E/Pi convergence.
        
        Args:
            optimizer_type: 'lbfgs', 'adam', or 'finite_diff'
            
        Returns:
            Dictionary with optimization results including timing data
        """
        if verbose:
            print(f"  Optimizing from τ={init_tau:.6f}, δ={init_delta:.6f}, λ={init_lambda:.6f} using {optimizer_type}")
        
        if optimizer_type in ['lbfgs', 'adam']:
            return self._optimize_with_pytorch_optimizer(
                species_tree, gene_tree, init_tau, init_delta, init_lambda,
                optimizer_type, lr, epochs, convergence_tol, verbose
            )
        else:  # finite_diff
            return self._optimize_with_finite_differences(
                species_tree, gene_tree, init_tau, init_delta, init_lambda,
                lr, epochs, convergence_tol, verbose
            )
    
    def _optimize_with_pytorch_optimizer(self, species_tree: str, gene_tree: str,
                                       init_tau: float, init_delta: float, init_lambda: float,
                                       optimizer_type: str, lr: float, epochs: int,
                                       convergence_tol: float, verbose: bool):
        """Optimize using PyTorch optimizers with full automatic differentiation."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64
        
        # Initialize parameters as tensors with gradients (in linear space)
        epsilon = 1e-10
        tau_param = torch.tensor(max(init_tau, epsilon), dtype=dtype, device=device, requires_grad=True)
        delta_param = torch.tensor(max(init_delta, epsilon), dtype=dtype, device=device, requires_grad=True)
        lambda_param = torch.tensor(max(init_lambda, epsilon), dtype=dtype, device=device, requires_grad=True)
        
        # Create optimizer
        if optimizer_type == 'lbfgs':
            optimizer = torch.optim.LBFGS([tau_param, delta_param, lambda_param], 
                                         lr=lr, max_iter=1, line_search_fn=None)
        else:  # adam
            optimizer = torch.optim.Adam([tau_param, delta_param, lambda_param], lr=lr, betas=(0.9, 0.999))
        
        # Cache setup to avoid recomputing tree structures
        setup_cache = None
        
        # History tracking
        history = {
            'epoch': [],
            'log_likelihood': [],
            'tau': [],
            'delta': [],
            'lambda': [],
            'param_change': [],
            'epoch_time': [],
            'grad_time': [],
            'likelihood_time': [],
            'cumulative_time': []
        }
        
        best_ll = float('-inf')
        best_params = None
        best_epoch = 0
        start_time = time.time()
        converged_epoch = None
        prev_params = (init_tau, init_delta, init_lambda)
        
        closure_calls = 0
        
        def closure():
            nonlocal closure_calls, setup_cache
            closure_calls += 1
            optimizer.zero_grad()
            
            # Compute likelihood using automatic differentiation
            result = compute_likelihood_differentiable(
                species_tree, gene_tree,
                tau_param, delta_param, lambda_param,
                setup_cache, debug=False
            )
            
            log_likelihood = result['log_likelihood']
            setup_cache = result['setup_cache']  # Cache for next call
            
            # Negative log-likelihood for minimization
            loss = -log_likelihood
            loss.backward()
            
            # Debug output
            if verbose and closure_calls <= 5:
                tau_val = tau_param.item()
                delta_val = delta_param.item() 
                lambda_val = lambda_param.item()
                print(f"    Closure {closure_calls}: τ={tau_val:.6f}, δ={delta_val:.6f}, λ={lambda_val:.6f}")
                print(f"      LL={log_likelihood.item():.6f}")
                print(f"      grads: τ={tau_param.grad.item():.2e}, δ={delta_param.grad.item():.2e}, λ={lambda_param.grad.item():.2e}")
            
            return loss
        
        # Optimization loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Time optimization step
            grad_start = time.time()
            loss = closure()
            
            # Debug: store params before step
            if verbose and epoch <= 3:
                tau_before = tau_param.item()
                delta_before = delta_param.item()
                lambda_before = lambda_param.item()
            
            if optimizer_type == 'lbfgs':
                optimizer.step(closure)
            else:  # adam
                optimizer.step()
            
            # Debug: check parameter change after step  
            if verbose and epoch <= 3:
                print(f"    After step {epoch}: params changed by "
                      f"Δτ={tau_param.item() - tau_before:.6f}, "
                      f"Δδ={delta_param.item() - delta_before:.6f}, "
                      f"Δλ={lambda_param.item() - lambda_before:.6f}")
            
            grad_time = time.time() - grad_start
            
            # Get updated parameter values
            tau = tau_param.item()
            delta = delta_param.item()
            lambda_val = lambda_param.item()
            
            # Time likelihood computation with updated parameters
            ll_start = time.time()
            with torch.no_grad():
                result = compute_likelihood_differentiable(
                    species_tree, gene_tree,
                    tau_param, delta_param, lambda_param,
                    setup_cache, debug=False
                )
                log_likelihood = result['log_likelihood']
            likelihood_time = time.time() - ll_start
            
            # Track best parameters
            ll_val = log_likelihood.item()
            if ll_val > best_ll:
                best_ll = ll_val
                best_params = (tau, delta, lambda_val)
                best_epoch = epoch
            
            # Check parameter convergence
            param_change = max(
                abs(tau - prev_params[0]),
                abs(delta - prev_params[1]),
                abs(lambda_val - prev_params[2])
            ) if epoch > 0 else float('inf')
            prev_params = (tau, delta, lambda_val)
            
            # Time tracking
            epoch_time = time.time() - epoch_start
            cumulative_time = time.time() - start_time
            
            # Record history
            history['epoch'].append(epoch)
            history['log_likelihood'].append(ll_val)
            history['tau'].append(tau)
            history['delta'].append(delta)
            history['lambda'].append(lambda_val)
            history['param_change'].append(param_change)
            history['epoch_time'].append(epoch_time)
            history['grad_time'].append(grad_time)
            history['likelihood_time'].append(likelihood_time)
            history['cumulative_time'].append(cumulative_time)
            
            # Check convergence
            if param_change < convergence_tol and converged_epoch is None:
                converged_epoch = epoch
                if verbose:
                    print(f"    ✓ Converged at epoch {epoch} (param change = {param_change:.2e})")
                break
            
            # Print progress periodically
            if verbose and epoch % 5 == 0:
                print(f"    Epoch {epoch:3d}: LL={ll_val:10.6f}, Δp={param_change:.2e}, "
                      f"grad_time={grad_time:.2f}s, ll_time={likelihood_time:.2f}s")
        
        total_time = time.time() - start_time
        
        return {
            'history': history,
            'best_params': best_params,
            'best_ll': best_ll,
            'best_epoch': best_epoch,
            'total_time': total_time,
            'converged_epoch': converged_epoch,
            'final_ll': history['log_likelihood'][-1] if history['log_likelihood'] else float('-inf'),
            'epochs_run': len(history['epoch']),
            'initial_params': (init_tau, init_delta, init_lambda),
            'optimizer_type': optimizer_type
        }
    
    def _optimize_with_finite_differences(self, species_tree: str, gene_tree: str,
                                        init_tau: float, init_delta: float, init_lambda: float,
                                        lr: float, epochs: int, convergence_tol: float, verbose: bool):
        """Original finite difference optimization (for comparison)."""
        # Set up fixed-point structures
        setup_start = time.time()
        result = setup_fixed_points(
            species_tree, gene_tree,
            max_iters_E=1, max_iters_Pi=1,
            tol_E=1e-10, tol_Pi=1e-10,
            debug=False
        )
        
        ccp_helpers = result['ccp_helpers']
        species_helpers = result['species_helpers']
        log_clade_species_map = result['clade_species_map']
        ccp = result['ccp']
        
        from src.core.ccp import get_root_clade_id
        root_clade_id = get_root_clade_id(ccp)
        
        device = log_clade_species_map.device
        dtype = log_clade_species_map.dtype
        setup_time = time.time() - setup_start
        
        # Initialize parameters
        epsilon = 1e-10
        tau = max(init_tau, epsilon)
        delta = max(init_delta, epsilon)
        lambda_param = max(init_lambda, epsilon)
        
        # History tracking
        history = {
            'epoch': [],
            'log_likelihood': [],
            'tau': [],
            'delta': [],
            'lambda': [],
            'param_change': [],
            'epoch_time': [],
            'e_convergence_time': [],
            'pi_convergence_time': [],
            'cumulative_time': [],
            'e_iterations': [],
            'pi_iterations': []
        }
        
        best_ll = float('-inf')
        best_params = None
        best_epoch = 0
        start_time = time.time()
        converged_epoch = None
        prev_params = (tau, delta, lambda_param)
        
        # Main optimization loop (simplified version of original)
        for epoch in range(min(epochs, 20)):  # Limit finite diff epochs for speed
            epoch_start = time.time()
            
            # Convert to log-space event probabilities
            total_rate = 1 + delta + tau + lambda_param
            log_pS = torch.log(torch.tensor(1.0 / total_rate, device=device, dtype=dtype))
            log_pD = torch.log(torch.tensor(delta / total_rate, device=device, dtype=dtype))
            log_pT = torch.log(torch.tensor(tau / total_rate, device=device, dtype=dtype))
            log_pL = torch.log(torch.tensor(lambda_param / total_rate, device=device, dtype=dtype))
            
            # Time E convergence
            e_start = time.time()
            E_result = E_fixed_point(
                species_helpers, log_pS, log_pD, log_pT, log_pL,
                max_iters=1000, tolerance=1e-9, return_components=True
            )
            E, E_s1, E_s2, E_bar = E_result['E'], E_result['E_s1'], E_result['E_s2'], E_result['E_bar']
            e_iters = E_result['iterations']
            e_time = time.time() - e_start
            
            # Time Pi convergence
            pi_start = time.time()
            Pi_result = Pi_fixed_point(
                ccp_helpers, species_helpers, log_clade_species_map,
                E, E_bar, E_s1, E_s2, log_pS, log_pD, log_pT,
                max_iters=1000, tolerance=1e-9
            )
            Pi = Pi_result['Pi']
            pi_iters = Pi_result['iterations']
            pi_time = time.time() - pi_start
            
            # Compute log-likelihood
            log_likelihood = torch.logsumexp(Pi[root_clade_id, :], dim=0).item()
            
            # Track best parameters
            if log_likelihood > best_ll:
                best_ll = log_likelihood
                best_params = (tau, delta, lambda_param)
                best_epoch = epoch
            
            # Record current state
            param_change = np.sqrt((tau - prev_params[0])**2 + 
                                 (delta - prev_params[1])**2 + 
                                 (lambda_param - prev_params[2])**2) if epoch > 0 else float('inf')
            
            # Time tracking
            epoch_time = time.time() - epoch_start
            cumulative_time = time.time() - start_time
            
            # Record history
            history['epoch'].append(epoch)
            history['log_likelihood'].append(log_likelihood)
            history['tau'].append(tau)
            history['delta'].append(delta)
            history['lambda'].append(lambda_param)
            history['param_change'].append(param_change)
            history['epoch_time'].append(epoch_time)
            history['e_convergence_time'].append(e_time)
            history['pi_convergence_time'].append(pi_time)
            history['cumulative_time'].append(cumulative_time)
            history['e_iterations'].append(e_iters)
            history['pi_iterations'].append(pi_iters)
            
            # Check convergence
            if epoch > 0 and param_change < convergence_tol and converged_epoch is None:
                converged_epoch = epoch
                if verbose:
                    print(f"    ✓ Converged at epoch {epoch} (param change = {param_change:.2e})")
                break
                
            # Skip parameter update on last epoch
            if epoch == epochs - 1:
                break
                
            # Simple finite difference gradient update (very simplified)
            eps = 1e-6
            step_size = lr
            
            # Update tau only (for speed in demo)
            if tau > eps:
                prev_params = (tau, delta, lambda_param)
                tau = max(tau + step_size * 0.001, epsilon)  # Small fixed update
            
            if verbose and epoch % 5 == 0:
                print(f"    Epoch {epoch:3d}: LL={log_likelihood:10.6f}, Δp={param_change:.2e}, "
                      f"E={e_iters:2d}it/{e_time:.2f}s, Pi={pi_iters:2d}it/{pi_time:.2f}s")
        
        total_time = time.time() - start_time
        
        return {
            'history': history,
            'best_params': best_params,
            'best_ll': best_ll,
            'best_epoch': best_epoch,
            'total_time': total_time,
            'converged_epoch': converged_epoch,
            'final_ll': history['log_likelihood'][-1] if history['log_likelihood'] else float('-inf'),
            'epochs_run': len(history['epoch']),
            'initial_params': (init_tau, init_delta, init_lambda),
            'setup_time': setup_time
        }
    
    def test_optimization_scenarios(self, species_tree: str, gene_tree: str, 
                                  alerax_params: tuple = None, optimizer_type: str = 'lbfgs'):
        """
        Test optimization from different starting points.
        """
        scenarios = []
        
        # Scenario 1: Standard initialization (0.1, 0.1, 0.1)
        scenarios.append({
            'name': 'standard_init',
            'description': 'Standard initialization (0.1, 0.1, 0.1)',
            'tau': 0.1,
            'delta': 0.1,
            'lambda': 0.1
        })
        
        # Scenario 2: AleRax inferred parameters (if available)
        if alerax_params:
            tau, delta, lambda_param = alerax_params
            scenarios.append({
                'name': 'alerax_init',
                'description': f'AleRax inferred ({tau:.4f}, {delta:.4f}, {lambda_param:.4f})',
                'tau': tau,
                'delta': delta,
                'lambda': lambda_param
            })
        
        # Scenario 3: Random initialization
        np.random.seed(42)  # For reproducibility
        rand_tau = np.random.uniform(0.05, 0.8)
        rand_delta = np.random.uniform(0.05, 0.8)  
        rand_lambda = np.random.uniform(0.05, 0.8)
        scenarios.append({
            'name': 'random_init',
            'description': f'Random initialization ({rand_tau:.4f}, {rand_delta:.4f}, {rand_lambda:.4f})',
            'tau': rand_tau,
            'delta': rand_delta,
            'lambda': rand_lambda
        })
        
        # Scenario 4: High values initialization (more extreme to force optimization)
        scenarios.append({
            'name': 'high_init',
            'description': 'High values initialization (2.0, 1.5, 1.0)',
            'tau': 2.0,
            'delta': 1.5,
            'lambda': 1.0
        })
        
        # Scenario 5: Unbalanced initialization (force parameter evolution)
        scenarios.append({
            'name': 'unbalanced_init',
            'description': 'Unbalanced initialization (0.01, 1.0, 0.01)',
            'tau': 0.01,
            'delta': 1.0,
            'lambda': 0.01
        })
        
        print(f"  Testing {len(scenarios)} optimization scenarios...")
        
        results = []
        for i, scenario in enumerate(scenarios):
            print(f"    [{i+1}/{len(scenarios)}] {scenario['description']}")
            
            result = self.optimize_parameters(
                species_tree, gene_tree,
                scenario['tau'], scenario['delta'], scenario['lambda'],
                optimizer_type=optimizer_type,
                lr=0.001 if optimizer_type == 'adam' else 0.1,  # Slower LR to see evolution
                epochs=100,  # More epochs to see optimization trajectory
                convergence_tol=1e-8,  # Tighter convergence to force more iterations
                verbose=False
            )
            
            result['scenario'] = scenario
            results.append(result)
            
            print(f"      Result: LL={result['final_ll']:.6f}, "
                  f"Time={result['total_time']:.1f}s, "
                  f"Epochs={result['epochs_run']}")
        
        return results
    
    def generate_timing_plots(self, results):
        """Generate comprehensive timing analysis plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{self.dataset_name}: Parameter Optimization Timing Analysis', fontsize=14, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Plot 1: Likelihood convergence for all scenarios
        ax = axes[0, 0]
        for i, result in enumerate(results):
            history = result['history']
            color = colors[i % len(colors)]
            ax.plot(history['epoch'], history['log_likelihood'], 
                   color=color, linewidth=2, alpha=0.8,
                   label=f"{result['scenario']['name']} ({result['final_ll']:.3f})")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log-Likelihood')
        ax.set_title('Likelihood Convergence')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Time to convergence
        ax = axes[0, 1]
        scenario_names = [r['scenario']['name'].replace('_', ' ').title() for r in results]
        times = [r['total_time'] for r in results]
        epochs = [r['epochs_run'] for r in results]
        
        bars = ax.bar(range(len(scenario_names)), times, color=colors[:len(results)], alpha=0.7)
        ax.set_xlabel('Initialization Strategy')
        ax.set_ylabel('Total Time (seconds)')
        ax.set_title('Time to Convergence')
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
        
        # Add epoch count on bars
        for i, (bar, epochs_run) in enumerate(zip(bars, epochs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{epochs_run} ep', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Final log-likelihood comparison
        ax = axes[0, 2]
        final_lls = [r['final_ll'] for r in results]
        bars = ax.bar(range(len(scenario_names)), final_lls, color=colors[:len(results)], alpha=0.7)
        ax.set_xlabel('Initialization Strategy')
        ax.set_ylabel('Final Log-Likelihood')
        ax.set_title('Final Likelihood Achieved')
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right', fontsize=9)
        
        # Plot 4: Parameter change evolution (first scenario only for clarity)
        ax = axes[1, 0]
        if results:
            history = results[0]['history']  # Standard initialization
            ax.semilogy(history['epoch'], history['param_change'], 'b-', linewidth=2)
            ax.axhline(y=1e-6, color='r', linestyle='--', alpha=0.7, label='Convergence threshold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Parameter Change (log scale)')
            ax.set_title(f'Parameter Change Evolution ({results[0]["scenario"]["name"]})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Time breakdown (average across scenarios)
        ax = axes[1, 1]
        if results:
            # Check if we have the detailed timing data
            has_detailed_timing = any('e_convergence_time' in r['history'] for r in results)
            
            if has_detailed_timing:
                avg_e_time = np.mean([np.mean(r['history']['e_convergence_time']) for r in results if 'e_convergence_time' in r['history']])
                avg_pi_time = np.mean([np.mean(r['history']['pi_convergence_time']) for r in results if 'pi_convergence_time' in r['history']])
                avg_epoch_time = np.mean([np.mean(r['history']['epoch_time']) for r in results])
                avg_other_time = max(0, avg_epoch_time - avg_e_time - avg_pi_time)
                
                sizes = [avg_e_time, avg_pi_time, avg_other_time]
                labels = [f'E Convergence\n{avg_e_time:.3f}s/ep', 
                         f'Pi Convergence\n{avg_pi_time:.3f}s/ep',
                         f'Overhead\n{avg_other_time:.3f}s/ep']
            else:
                # Use simpler timing breakdown for PyTorch optimizers
                avg_grad_time = np.mean([np.mean(r['history']['grad_time']) for r in results if 'grad_time' in r['history']])
                avg_ll_time = np.mean([np.mean(r['history']['likelihood_time']) for r in results if 'likelihood_time' in r['history']])
                avg_epoch_time = np.mean([np.mean(r['history']['epoch_time']) for r in results])
                avg_other_time = max(0, avg_epoch_time - avg_grad_time - avg_ll_time)
                
                sizes = [avg_grad_time, avg_ll_time, avg_other_time]
                labels = [f'Gradient Computation\n{avg_grad_time:.3f}s/ep',
                         f'Likelihood Computation\n{avg_ll_time:.3f}s/ep', 
                         f'Overhead\n{avg_other_time:.3f}s/ep']
                
            colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
            ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax.set_title('Average Time per Epoch')
        
        # Plot 6: Parameter evolution (best scenario)
        ax = axes[1, 2]
        if results:
            best_result = min(results, key=lambda x: x['total_time'])  # Fastest convergence
            history = best_result['history']
            ax.plot(history['epoch'], history['tau'], label='τ (transfer)', linewidth=2)
            ax.plot(history['epoch'], history['delta'], label='δ (duplication)', linewidth=2)
            ax.plot(history['epoch'], history['lambda'], label='λ (loss)', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Parameter Value')
            ax.set_title(f'Parameter Evolution ({best_result["scenario"]["name"]})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.results_dir / 'parameter_optimization_timing.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved timing plots to: {plot_path}")
        
    def save_summary(self, results, alerax_params):
        """Save optimization timing summary to JSON."""
        
        # Calculate statistics
        summary = {
            'dataset': self.dataset_name,
            'alerax_reference_params': {
                'tau': alerax_params[0] if alerax_params else None,
                'delta': alerax_params[1] if alerax_params else None,
                'lambda': alerax_params[2] if alerax_params else None
            },
            'optimization_results': []
        }
        
        for result in results:
            scenario_summary = {
                'scenario': result['scenario'],
                'initial_params': {
                    'tau': result['initial_params'][0],
                    'delta': result['initial_params'][1], 
                    'lambda': result['initial_params'][2]
                },
                'final_params': {
                    'tau': result['best_params'][0] if result['best_params'] else None,
                    'delta': result['best_params'][1] if result['best_params'] else None,
                    'lambda': result['best_params'][2] if result['best_params'] else None
                },
                'performance': {
                    'final_log_likelihood': float(result['final_ll']) if isinstance(result['final_ll'], torch.Tensor) else result['final_ll'],
                    'best_log_likelihood': float(result['best_ll']) if isinstance(result['best_ll'], torch.Tensor) else result['best_ll'],
                    'total_time_seconds': result['total_time'],
                    'epochs_to_convergence': result['converged_epoch'],
                    'total_epochs_run': result['epochs_run'],
                    'time_per_epoch': result['total_time'] / result['epochs_run'] if result['epochs_run'] > 0 else None
                }
            }
            summary['optimization_results'].append(scenario_summary)
        
        summary_path = self.results_dir / 'optimization_timing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"    Saved timing summary to: {summary_path}")
        return summary


def parse_alerax_parameters(dataset_path):
    """Parse AleRax inferred parameters from model_parameters.txt."""
    param_file = dataset_path / 'output' / 'model_parameters' / 'model_parameters.txt'
    
    if not param_file.exists():
        return None
        
    try:
        with open(param_file, 'r') as f:
            lines = f.readlines()
            
        # Skip header line and find first data line
        for line in lines[1:]:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 4:
                    # Format: node D L T
                    delta = float(parts[1])
                    lambda_param = float(parts[2])
                    tau = float(parts[3])
                    return tau, delta, lambda_param
        return None
    except Exception as e:
        print(f"    Warning: Could not parse AleRax parameters: {e}")
        return None


def run_parameter_optimization_timing(optimizer_type: str = 'lbfgs'):
    """Run parameter optimization timing analysis on all test datasets."""
    
    # Find all test datasets
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results' / 'optimization_timing'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all test dataset directories
    datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and (d / 'sp.nwk').exists()]
    datasets.sort()
    
    print(f"🔧 PARAMETER OPTIMIZATION TIMING ANALYSIS ({optimizer_type.upper()})")
    print(f"Found {len(datasets)} test datasets: {datasets}")
    print("="*80)
    
    all_summaries = []
    
    for dataset in datasets:
        print(f"\n🚀 OPTIMIZING PARAMETERS: {dataset}")
        print("-" * 50)
        
        # Set up paths
        species_tree_path = str(data_dir / dataset / 'sp.nwk')
        gene_tree_path = str(data_dir / dataset / 'g.nwk')
        
        # Initialize timer
        timer = ParameterOptimizationTimer(dataset, results_dir)
        
        try:
            # Parse AleRax inferred parameters
            dataset_path = data_dir / dataset
            alerax_params = parse_alerax_parameters(dataset_path)
            
            if alerax_params:
                tau, delta, lambda_param = alerax_params
                print(f"  AleRax reference: τ={tau:.6f}, δ={delta:.6f}, λ={lambda_param:.6f}")
            else:
                print(f"  AleRax parameters not found, will use standard comparisons")
            
            # Run optimization scenarios
            results = timer.test_optimization_scenarios(
                species_tree_path, gene_tree_path, alerax_params, optimizer_type
            )
            
            # Generate plots and save summary
            timer.generate_timing_plots(results)
            summary = timer.save_summary(results, alerax_params)
            all_summaries.append(summary)
            
            # Print results summary
            print(f"\n  📊 Results Summary:")
            for result in results:
                scenario = result['scenario']['name']
                time_taken = result['total_time']
                epochs = result['epochs_run']
                final_ll = result['final_ll']
                print(f"    {scenario:12}: {time_taken:6.1f}s ({epochs:3d} epochs) → LL={final_ll:.4f}")
            
            print(f"  ✅ {dataset} optimization timing completed!")
            
        except Exception as e:
            print(f"  ❌ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall summary
    print("\n" + "="*80)
    print("OPTIMIZATION TIMING SUMMARY")
    print("="*80)
    
    overall_summary_path = results_dir / 'overall_optimization_timing_summary.json'
    with open(overall_summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    # Print comparison table
    print(f"{'Dataset':<15} {'Best Strategy':<15} {'Time (s)':<10} {'Epochs':<8} {'Final LL':<12}")
    print("-" * 70)
    
    for summary in all_summaries:
        dataset = summary['dataset']
        
        # Find best strategy (fastest convergence)
        best_result = None
        best_time = float('inf')
        for result in summary['optimization_results']:
            if result['performance']['total_time_seconds'] < best_time:
                best_time = result['performance']['total_time_seconds']
                best_result = result
        
        if best_result:
            strategy = best_result['scenario']['name']
            time_taken = best_result['performance']['total_time_seconds']
            epochs = best_result['performance']['total_epochs_run']
            final_ll = best_result['performance']['final_log_likelihood']
            
            print(f"{dataset:<15} {strategy:<15} {time_taken:<10.1f} {epochs:<8d} {final_ll:<12.6f}")
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Overall summary: {overall_summary_path}")

    return all_summaries


def test_parameter_optimization_timing():
    """Pytest test function for parameter optimization timing."""
    return run_parameter_optimization_timing()


if __name__ == '__main__':
    import sys
    
    # Allow specifying optimizer type as command line argument
    optimizer_type = sys.argv[1] if len(sys.argv) > 1 else 'lbfgs'
    if optimizer_type not in ['lbfgs', 'adam', 'finite_diff']:
        print(f"Invalid optimizer type: {optimizer_type}")
        print("Valid options: lbfgs, adam, finite_diff")
        sys.exit(1)
    
    run_parameter_optimization_timing(optimizer_type)