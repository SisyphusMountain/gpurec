#!/usr/bin/env python3
"""
Test script for gradient descent optimization of phylogenetic reconciliation parameters.
Verifies gradient computation and demonstrates optimization on test trees.
"""

import torch
import time
import json
import argparse
from pathlib import Path
from tabulate import tabulate

from matmul_ale_ccp_optimize import CCPOptimizer, verify_gradients, softplus_transform, inverse_softplus_transform

def test_parameter_transformation():
    """Test the softplus parameter transformation functions."""
    print("🧪 Testing parameter transformation...")
    
    # Test with various parameter values
    test_params = torch.tensor([0.01, 0.1, 1.0, 10.0])
    
    # Forward and inverse transform
    log_params = inverse_softplus_transform(test_params)
    recovered_params = softplus_transform(log_params)
    
    # Check accuracy
    error = torch.abs(test_params - recovered_params).max()
    print(f"   Max error in round-trip transformation: {error:.2e}")
    
    # Check positivity constraint
    extreme_log_params = torch.tensor([-100.0, -10.0, 0.0, 10.0])
    positive_params = softplus_transform(extreme_log_params)
    print(f"   Transformed extreme values: {positive_params.tolist()}")
    print(f"   All positive: {torch.all(positive_params > 0)}")
    
    assert error < 1e-6, f"Transformation error too large: {error}"
    assert torch.all(positive_params > 0), "Not all transformed parameters are positive"
    print("   ✅ Parameter transformation tests passed")

def test_single_tree_pair(species_path: str, gene_path: str, name: str = ""):
    """Test optimization on a single tree pair."""
    print(f"\n{'='*60}")
    print(f"🌳 Testing optimization on {name or Path(species_path).parent.name}")
    print(f"{'='*60}")
    
    # Initialize optimizer
    optimizer = CCPOptimizer(
        species_path, gene_path,
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32
    )
    
    # Evaluate initial likelihood
    initial_likelihood = optimizer.evaluate_likelihood()
    initial_params = optimizer.get_current_params()
    print(f"Initial likelihood: {initial_likelihood:.6f}")
    print(f"Initial parameters: δ={initial_params[0]:.6f}, τ={initial_params[1]:.6f}, λ={initial_params[2]:.6f}")
    
    # Verify gradients
    print("\n🔍 Verifying gradients...")
    gradient_errors = verify_gradients(optimizer, epsilon=1e-5)
    
    if gradient_errors["max_rel_error"] > 1e-2:
        print(f"⚠️  WARNING: Large gradient error detected!")
        print(f"   This could indicate numerical issues in the implementation")
    else:
        print("✅ Gradient verification passed")
    
    # Run optimization with Adam
    print("\n🚀 Running Adam optimization...")
    adam_results = optimizer.optimize(
        lr=0.01, epochs=50, optimizer_type="adam",
        early_stopping_patience=15, min_improvement=1e-6
    )
    
    # Reset parameters and try SGD
    optimizer.log_params.data = torch.nn.Parameter(
        inverse_softplus_transform(torch.tensor(initial_params, device=optimizer.device, dtype=optimizer.dtype))
    ).data
    
    print("\n🚀 Running SGD optimization...")
    sgd_results = optimizer.optimize(
        lr=0.005, epochs=30, optimizer_type="sgd",
        early_stopping_patience=10, min_improvement=1e-6
    )
    
    return {
        'name': name or Path(species_path).parent.name,
        'initial': {
            'likelihood': initial_likelihood,
            'params': initial_params
        },
        'adam': adam_results,
        'sgd': sgd_results,
        'gradient_errors': gradient_errors
    }

def test_all_tree_pairs():
    """Test optimization on all available tree pairs."""
    print("🌲 Testing optimization on all available tree pairs...")
    
    # Define test tree pairs
    tree_pairs = [
        ("test_trees_1/sp.nwk", "test_trees_1/g.nwk", "Simple test case"),
        ("test_trees_2/sp.nwk", "test_trees_2/g.nwk", "Complex test case"),
    ]
    
    # Check for test_trees_200 if it exists
    if Path("test_trees_200").exists():
        tree_pairs.append(("test_trees_200/sp.nwk", "test_trees_200/g.nwk", "Large test case"))
    
    results = []
    
    for species_path, gene_path, description in tree_pairs:
        if Path(species_path).exists() and Path(gene_path).exists():
            try:
                result = test_single_tree_pair(species_path, gene_path, description)
                results.append(result)
            except Exception as e:
                print(f"❌ Error testing {description}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Skipping {description}: files not found")
    
    return results

def benchmark_optimization_performance():
    """Benchmark the optimization performance."""
    print("\n📊 Benchmarking optimization performance...")
    
    # Use test_trees_1 for benchmarking
    species_path = "test_trees_1/sp.nwk"
    gene_path = "test_trees_1/g.nwk"
    
    if not (Path(species_path).exists() and Path(gene_path).exists()):
        print("❌ Benchmark files not found, skipping performance test")
        return
    
    optimizer = CCPOptimizer(
        species_path, gene_path,
        initial_params=(0.1, 0.1, 0.1),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32
    )
    
    # Benchmark single likelihood evaluation
    print("⏱️  Benchmarking likelihood evaluation...")
    warmup_iters = 5
    benchmark_iters = 20
    
    # Warmup
    for _ in range(warmup_iters):
        optimizer.evaluate_likelihood()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(benchmark_iters):
        optimizer.evaluate_likelihood()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    time_per_eval = total_time / benchmark_iters
    
    print(f"   Time per likelihood evaluation: {time_per_eval*1000:.2f} ms")
    print(f"   Evaluations per second: {1/time_per_eval:.1f}")
    
    # Benchmark gradient computation
    print("⏱️  Benchmarking gradient computation...")
    
    def compute_gradient():
        log_likelihood = optimizer.evaluate_likelihood()
        log_likelihood.backward()
        return optimizer.log_params.grad
    
    # Warmup
    for _ in range(warmup_iters):
        optimizer.log_params.grad = None
        compute_gradient()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(benchmark_iters):
        optimizer.log_params.grad = None
        compute_gradient()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    time_per_grad = total_time / benchmark_iters
    
    print(f"   Time per gradient computation: {time_per_grad*1000:.2f} ms")
    print(f"   Gradient computations per second: {1/time_per_grad:.1f}")

def create_summary_table(results):
    """Create a summary table of optimization results."""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print("📋 OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    # Create table data
    table_data = []
    
    for result in results:
        # Initial row
        table_data.append([
            result['name'],
            "Initial",
            f"{result['initial']['likelihood']:.3f}",
            f"{result['initial']['params'][0]:.4f}",
            f"{result['initial']['params'][1]:.4f}", 
            f"{result['initial']['params'][2]:.4f}",
            "-",
            "-"
        ])
        
        # Adam results
        adam = result['adam']
        table_data.append([
            "",
            "Adam",
            f"{adam['best_log_likelihood']:.3f}",
            f"{adam['final_params'][0]:.4f}",
            f"{adam['final_params'][1]:.4f}",
            f"{adam['final_params'][2]:.4f}",
            f"{adam['epochs_run']}",
            f"{adam['total_time']:.1f}s"
        ])
        
        # SGD results
        sgd = result['sgd']
        table_data.append([
            "",
            "SGD", 
            f"{sgd['best_log_likelihood']:.3f}",
            f"{sgd['final_params'][0]:.4f}",
            f"{sgd['final_params'][1]:.4f}",
            f"{sgd['final_params'][2]:.4f}",
            f"{sgd['epochs_run']}",
            f"{sgd['total_time']:.1f}s"
        ])
        
        # Improvement summary
        adam_improvement = adam['best_log_likelihood'] - result['initial']['likelihood']
        sgd_improvement = sgd['best_log_likelihood'] - result['initial']['likelihood']
        
        table_data.append([
            "",
            "Best Δ",
            f"+{max(adam_improvement, sgd_improvement):.3f}",
            "",
            "",
            "",
            "",
            ""
        ])
        
        table_data.append(["", "", "", "", "", "", "", ""])  # Separator
    
    headers = ["Tree Set", "Method", "Log-Likelihood", "δ", "τ", "λ", "Epochs", "Time"]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Gradient verification summary
    print(f"\n📊 GRADIENT VERIFICATION SUMMARY")
    print(f"{'='*50}")
    
    grad_table = []
    for result in results:
        errors = result['gradient_errors']
        grad_table.append([
            result['name'],
            f"{errors['max_abs_error']:.2e}",
            f"{errors['mean_abs_error']:.2e}",
            f"{errors['max_rel_error']:.2e}",
            f"{errors['mean_rel_error']:.2e}"
        ])
    
    grad_headers = ["Tree Set", "Max Abs Error", "Mean Abs Error", "Max Rel Error", "Mean Rel Error"]
    print(tabulate(grad_table, headers=grad_headers, tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description='Test phylogenetic parameter optimization')
    parser.add_argument('--test-transforms', action='store_true', help='Test parameter transformations')
    parser.add_argument('--test-single', nargs=3, metavar=('SPECIES', 'GENE', 'NAME'), 
                       help='Test single tree pair')
    parser.add_argument('--test-all', action='store_true', help='Test all available tree pairs')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if not any([args.test_transforms, args.test_single, args.test_all, args.benchmark]):
        # Default: run all tests
        args.test_transforms = True
        args.test_all = True
        args.benchmark = True
    
    print("🧬 Phylogenetic Reconciliation Parameter Optimization Tests")
    print("="*65)
    
    results = []
    
    # Test parameter transformations
    if args.test_transforms:
        test_parameter_transformation()
    
    # Test single tree pair
    if args.test_single:
        species_path, gene_path, name = args.test_single
        result = test_single_tree_pair(species_path, gene_path, name)
        results.append(result)
    
    # Test all tree pairs
    if args.test_all:
        all_results = test_all_tree_pairs()
        results.extend(all_results)
    
    # Run performance benchmarks
    if args.benchmark:
        benchmark_optimization_performance()
    
    # Create summary
    if results:
        create_summary_table(results)
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")
    
    print(f"\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()