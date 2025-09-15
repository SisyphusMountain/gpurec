"""
Comprehensive convergence analysis test for all datasets.

This test runs reconciliation on all test datasets with detailed convergence tracking:
- High iteration limits with custom convergence criteria (1e-9 max elementwise improvement)
- Detailed timing analysis for E and Pi fixed-point iterations
- Plots showing convergence evolution over iterations
- Results saved to organized folder structure in tests/results/
"""

import os
import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

# Import with absolute paths to avoid relative import issues
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from src.reconciliation.reconcile import setup_fixed_points
from src.reconciliation.likelihood import E_fixed_point, Pi_fixed_point


class ConvergenceAnalyzer:
    """Analyzes convergence behavior of fixed-point iterations with detailed tracking."""
    
    def __init__(self, dataset_name: str, results_dir: Path):
        self.dataset_name = dataset_name
        self.results_dir = results_dir / dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convergence tracking
        self.E_convergence = []  # List of (iteration, max_change, time) tuples
        self.Pi_convergence = [] # List of (iteration, max_change, time) tuples
        
        # Overall timing
        self.total_time = 0
        self.E_total_time = 0
        self.Pi_total_time = 0
        
    def analyze_E_convergence(self, species_helpers, log_pS, log_pD, log_pT, log_pL, 
                             max_iters=10000, tolerance=1e-9):
        """
        Run E fixed-point with detailed convergence tracking.
        """
        print(f"  Analyzing E convergence (max {max_iters} iters, tol {tolerance})...")
        
        start_time = time.time()
        device = log_pS.device
        dtype = log_pS.dtype
        
        # Initialize E
        S = species_helpers['S']
        log_E = torch.full((S,), -1.0, device=device, dtype=dtype)  # Initial guess
        
        self.E_convergence = []
        
        for iteration in range(max_iters):
            iter_start = time.time()
            
            # Store previous value for convergence check
            log_E_prev = log_E.clone()
            
            # Run one step of E fixed-point
            result = E_fixed_point(
                species_helpers, log_pS, log_pD, log_pT, log_pL,
                max_iters=1,  # Single iteration
                tolerance=1e-15,  # Very tight tolerance to get exact single step
                return_components=True,
                warm_start_E=log_E
            )
            
            log_E = result['E']
            iter_time = time.time() - iter_start
            
            # Compute max elementwise change
            max_change = torch.max(torch.abs(log_E - log_E_prev)).item()
            
            # Track convergence
            self.E_convergence.append((iteration, max_change, iter_time))
            
            if iteration % 100 == 0:
                print(f"    E iter {iteration:4d}: max_change={max_change:.2e}, time={iter_time:.4f}s")
            
            # Check convergence
            if max_change < tolerance:
                print(f"    E converged at iteration {iteration} (max_change={max_change:.2e})")
                break
        
        self.E_total_time = time.time() - start_time
        return log_E, result['E_s1'], result['E_s2'], result['E_bar']
        
    def analyze_Pi_convergence(self, ccp_helpers, species_helpers, log_clade_species_map,
                              log_E, log_Ebar, log_E_s1, log_E_s2, 
                              log_pS, log_pD, log_pT, max_iters=10000, tolerance=1e-9):
        """
        Run Pi fixed-point with detailed convergence tracking.
        """
        print(f"  Analyzing Pi convergence (max {max_iters} iters, tol {tolerance})...")
        
        start_time = time.time()
        device = log_pS.device
        dtype = log_pS.dtype
        
        # Initialize Pi
        C, S = ccp_helpers['C'], species_helpers['S']
        log_Pi = torch.full((C, S), -5.0, device=device, dtype=dtype)  # Initial guess
        
        # Set leaf probabilities based on clade-species mapping
        clade_species_map = torch.exp(log_clade_species_map)
        for c in range(C):
            for s in range(S):
                if clade_species_map[c, s] > 0:
                    log_Pi[c, s] = 0.0  # log(1) for valid mappings
        
        self.Pi_convergence = []
        
        for iteration in range(max_iters):
            iter_start = time.time()
            
            # Store previous value for convergence check
            log_Pi_prev = log_Pi.clone()
            
            # Run one step of Pi fixed-point
            Pi_result = Pi_fixed_point(
                ccp_helpers, species_helpers, log_clade_species_map,
                log_E, log_Ebar, log_E_s1, log_E_s2, 
                log_pS, log_pD, log_pT,
                max_iters=1,  # Single iteration
                tolerance=1e-15,  # Very tight tolerance to get exact single step
                warm_start_Pi=log_Pi
            )
            log_Pi = Pi_result['Pi']
            
            iter_time = time.time() - iter_start
            
            # Compute max elementwise change
            max_change = torch.max(torch.abs(log_Pi - log_Pi_prev)).item()
            
            # Track convergence
            self.Pi_convergence.append((iteration, max_change, iter_time))
            
            if iteration % 100 == 0:
                print(f"    Pi iter {iteration:4d}: max_change={max_change:.2e}, time={iter_time:.4f}s")
            
            # Check convergence
            if max_change < tolerance:
                print(f"    Pi converged at iteration {iteration} (max_change={max_change:.2e})")
                break
        
        self.Pi_total_time = time.time() - start_time
        return log_Pi
        
    def generate_plots(self):
        """Generate convergence plots and save them."""
        print(f"  Generating convergence plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Convergence Analysis: {self.dataset_name}', fontsize=16)
        
        # E convergence plot
        if self.E_convergence:
            E_iters = [x[0] for x in self.E_convergence]
            E_changes = [x[1] for x in self.E_convergence]
            
            ax1.semilogy(E_iters, E_changes, 'b.-', linewidth=2, markersize=4)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Max |E_new - E_old|')
            ax1.set_title('E Fixed-Point Convergence')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1e-9, color='r', linestyle='--', alpha=0.7, label='Target tolerance')
            ax1.legend()
        
        # Pi convergence plot
        if self.Pi_convergence:
            Pi_iters = [x[0] for x in self.Pi_convergence]
            Pi_changes = [x[1] for x in self.Pi_convergence]
            
            ax2.semilogy(Pi_iters, Pi_changes, 'g.-', linewidth=2, markersize=4)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Max |Pi_new - Pi_old|')
            ax2.set_title('Pi Fixed-Point Convergence')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1e-9, color='r', linestyle='--', alpha=0.7, label='Target tolerance')
            ax2.legend()
        
        # Timing per iteration
        if self.E_convergence:
            E_times = [x[2] for x in self.E_convergence]
            ax3.plot(E_iters, E_times, 'b.-', linewidth=2, markersize=4, label='E iteration time')
            
        if self.Pi_convergence:
            Pi_times = [x[2] for x in self.Pi_convergence]
            ax3.plot(Pi_iters, Pi_times, 'g.-', linewidth=2, markersize=4, label='Pi iteration time')
            
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Time per iteration (s)')
        ax3.set_title('Timing Analysis')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Combined convergence rate comparison
        if self.E_convergence and self.Pi_convergence:
            # Normalize iteration counts for comparison
            max_E_iter = max(E_iters) if E_iters else 1
            max_Pi_iter = max(Pi_iters) if Pi_iters else 1
            max_iter = max(max_E_iter, max_Pi_iter)
            
            E_normalized = [(i/max_E_iter)*max_iter for i in E_iters]
            Pi_normalized = [(i/max_Pi_iter)*max_iter for i in Pi_iters]
            
            ax4.semilogy(E_normalized, E_changes, 'b.-', linewidth=2, markersize=3, 
                        label=f'E (converged in {len(E_iters)} iters)', alpha=0.7)
            ax4.semilogy(Pi_normalized, Pi_changes, 'g.-', linewidth=2, markersize=3, 
                        label=f'Pi (converged in {len(Pi_iters)} iters)', alpha=0.7)
            ax4.set_xlabel('Normalized Iteration')
            ax4.set_ylabel('Max Change')
            ax4.set_title('Convergence Rate Comparison')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=1e-9, color='r', linestyle='--', alpha=0.7, label='Target tolerance')
            ax4.legend()
        
        plt.tight_layout()
        plot_path = self.results_dir / 'convergence_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved plot to: {plot_path}")
        
    def save_summary(self, log_likelihood):
        """Save convergence analysis summary to JSON."""
        
        # Calculate statistics
        E_total_iters = len(self.E_convergence)
        Pi_total_iters = len(self.Pi_convergence)
        
        E_avg_time = np.mean([x[2] for x in self.E_convergence]) if self.E_convergence else 0
        Pi_avg_time = np.mean([x[2] for x in self.Pi_convergence]) if self.Pi_convergence else 0
        
        E_final_change = self.E_convergence[-1][1] if self.E_convergence else float('inf')
        Pi_final_change = self.Pi_convergence[-1][1] if self.Pi_convergence else float('inf')
        
        summary = {
            'dataset': self.dataset_name,
            'log_likelihood': log_likelihood,
            'convergence': {
                'E': {
                    'total_iterations': E_total_iters,
                    'total_time': self.E_total_time,
                    'avg_time_per_iter': E_avg_time,
                    'final_max_change': E_final_change,
                    'converged': E_final_change < 1e-9
                },
                'Pi': {
                    'total_iterations': Pi_total_iters,
                    'total_time': self.Pi_total_time,
                    'avg_time_per_iter': Pi_avg_time,
                    'final_max_change': Pi_final_change,
                    'converged': Pi_final_change < 1e-9
                }
            }
        }
        
        summary_path = self.results_dir / 'convergence_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"    Saved summary to: {summary_path}")
        return summary


def parse_alerax_parameters(dataset_path):
    """Parse AleRax inferred parameters from model_parameters.txt.
    
    Returns tuple of (tau, delta, lambda_param) or None if file doesn't exist.
    """
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


def run_convergence_analysis():
    """Run convergence analysis on all test datasets."""
    
    # Find all test datasets
    data_dir = Path(__file__).parent.parent / 'data'
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Get all test dataset directories
    datasets = [d.name for d in data_dir.iterdir() if d.is_dir() and (d / 'sp.nwk').exists()]
    datasets.sort()
    
    print(f"Found {len(datasets)} test datasets: {datasets}")
    print("="*80)
    
    all_summaries = []
    
    for dataset in datasets:
        print(f"\n🔬 ANALYZING DATASET: {dataset}")
        print("-" * 50)
        
        # Set up paths
        species_tree_path = str(data_dir / dataset / 'sp.nwk')
        gene_tree_path = str(data_dir / dataset / 'g.nwk')
        
        # Initialize analyzer
        analyzer = ConvergenceAnalyzer(dataset, results_dir)
        
        try:
            # Set up fixed-point structures
            print("  Setting up tree structures...")
            setup_start = time.time()
            
            result = setup_fixed_points(
                species_tree_path, gene_tree_path,
                max_iters_E=1, max_iters_Pi=1,  # Just setup, no solving
                tol_E=1e-10, tol_Pi=1e-10,
                debug=False
            )
            
            setup_time = time.time() - setup_start
            print(f"    Setup completed in {setup_time:.2f}s")
            
            # Extract components
            ccp_helpers = result['ccp_helpers']
            species_helpers = result['species_helpers']
            log_clade_species_map = result['clade_species_map']
            
            # Get root clade ID
            from src.core.ccp import get_root_clade_id
            ccp = result['ccp']
            root_clade_id = get_root_clade_id(ccp)
            
            device = log_clade_species_map.device
            dtype = log_clade_species_map.dtype
            
            print(f"    Dataset size: {ccp_helpers['C']} clades, {species_helpers['S']} species")
            
            # Parse AleRax inferred parameters or use defaults
            dataset_path = data_dir / dataset
            alerax_params = parse_alerax_parameters(dataset_path)
            
            if alerax_params:
                tau, delta, lambda_param = alerax_params
                print(f"    Using AleRax inferred parameters: τ={tau:.6f}, δ={delta:.6f}, λ={lambda_param:.6f}")
            else:
                tau, delta, lambda_param = 0.1, 0.1, 0.1
                print(f"    AleRax parameters not found, using defaults: τ={tau}, δ={delta}, λ={lambda_param}")
            
            # Convert to log-space event probabilities
            total_rate = 1 + delta + tau + lambda_param
            log_pS = torch.log(torch.tensor(1.0 / total_rate, device=device, dtype=dtype))
            log_pD = torch.log(torch.tensor(delta / total_rate, device=device, dtype=dtype))
            log_pT = torch.log(torch.tensor(tau / total_rate, device=device, dtype=dtype))
            log_pL = torch.log(torch.tensor(lambda_param / total_rate, device=device, dtype=dtype))
            
            print(f"    Parameters: τ={tau}, δ={delta}, λ={lambda_param}")
            print(f"    Event probabilities: pS={torch.exp(log_pS):.4f}, pD={torch.exp(log_pD):.4f}, "
                  f"pT={torch.exp(log_pT):.4f}, pL={torch.exp(log_pL):.4f}")
            
            # Analyze E convergence
            log_E, log_E_s1, log_E_s2, log_Ebar = analyzer.analyze_E_convergence(
                species_helpers, log_pS, log_pD, log_pT, log_pL,
                max_iters=10000, tolerance=1e-9
            )
            
            # Analyze Pi convergence
            log_Pi = analyzer.analyze_Pi_convergence(
                ccp_helpers, species_helpers, log_clade_species_map,
                log_E, log_Ebar, log_E_s1, log_E_s2, 
                log_pS, log_pD, log_pT,
                max_iters=10000, tolerance=1e-9
            )
            
            # Compute final log-likelihood  
            log_likelihood = torch.logsumexp(log_Pi[root_clade_id, :], dim=0).item()
            
            print(f"    Final log-likelihood: {log_likelihood:.6f}")
            print(f"    Total time: E={analyzer.E_total_time:.2f}s, Pi={analyzer.Pi_total_time:.2f}s")
            
            # Generate plots and save summary
            analyzer.generate_plots()
            summary = analyzer.save_summary(log_likelihood)
            
            # Update summary with actual parameters used
            summary['parameters'] = {
                'tau': tau,
                'delta': delta,
                'lambda': lambda_param,
                'source': 'alerax_inferred' if alerax_params else 'default'
            }
            all_summaries.append(summary)
            
            print(f"✅ {dataset} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    overall_summary_path = results_dir / 'overall_summary.json'
    with open(overall_summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    # Print summary table
    print(f"{'Dataset':<15} {'Log-Likelihood':<15} {'E Iters':<10} {'Pi Iters':<10} {'E Time':<10} {'Pi Time':<10} {'Parameters (τ,δ,λ)':<25}")
    print("-" * 115)
    
    for summary in all_summaries:
        dataset = summary['dataset']
        ll = summary['log_likelihood']
        E_iters = summary['convergence']['E']['total_iterations']
        Pi_iters = summary['convergence']['Pi']['total_iterations']
        E_time = summary['convergence']['E']['total_time']
        Pi_time = summary['convergence']['Pi']['total_time']
        params = summary['parameters']
        param_str = f"({params['tau']:.4f},{params['delta']:.4f},{params['lambda']:.4f})"
        if params['source'] == 'alerax_inferred':
            param_str += " *"
            
        print(f"{dataset:<15} {ll:<15.6f} {E_iters:<10d} {Pi_iters:<10d} {E_time:<10.2f} {Pi_time:<10.2f} {param_str:<25}")
    
    print(f"\n* = AleRax inferred parameters")
    print(f"\nResults saved to: {results_dir}")
    print(f"Overall summary: {overall_summary_path}")


def test_convergence_analysis():
    """Pytest test function for convergence analysis."""
    run_convergence_analysis()


if __name__ == '__main__':
    run_convergence_analysis()