#!/usr/bin/env python3
"""
Integration tests comparing likelihoods between AleRax and refactored CCP reconciliation code.

This test verifies that the refactored implementation produces identical results
to the reference AleRax implementation across multiple test datasets.
"""

import torch
import sys
import pytest
from pathlib import Path
import subprocess
import tempfile
import re
from typing import Dict, Optional
from tabulate import tabulate

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reconciliation.reconcile import setup_fixed_points


@pytest.fixture(scope="class")
def test_setup():
    """Setup test data paths and check AleRax availability."""
    test_data_dir = Path(__file__).parent.parent / "data"
    
    # Check if test data directory exists
    if not test_data_dir.exists():
        pytest.skip(f"Test data directory not found: {test_data_dir}")
    
    # Try to find AleRax binary
    possible_paths = [
        "AleRax_modified/build/bin/alerax",
        "../AleRax_modified/build/bin/alerax",
        "../../AleRax_modified/build/bin/alerax"
    ]
    
    repo_root = Path(__file__).parent.parent.parent
    alerax_binary = None
    
    for rel_path in possible_paths:
        full_path = repo_root / rel_path
        if full_path.exists():
            alerax_binary = str(full_path)
            break
    
    return {
        'test_data_dir': test_data_dir,
        'alerax_binary': alerax_binary
    }


class TestLikelihoodComparison:
    """Test class for comparing likelihoods between AleRax and refactored code."""
    
    # Available test datasets - parameters are read dynamically from AleRax output files
    AVAILABLE_DATASETS = [
        'test_trees_1', 'test_trees_2', 'test_trees_3', 
        'test_trees_200', 'test_mixed_200'
    ]
        
    def _get_alerax_result_from_file(self, test_setup, dataset_name: str) -> Optional[float]:
        """Extract AleRax likelihood from existing output files."""
        dataset_dir = test_setup['test_data_dir'] / dataset_name / "output"
        likelihood_file = dataset_dir / "per_fam_likelihoods.txt"
        
        if not likelihood_file.exists():
            return None
        
        try:
            with open(likelihood_file, 'r') as f:
                content = f.read().strip()
                # Extract numerical value - file should contain just the likelihood
                match = re.search(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?', content)
                if match:
                    return float(match.group())
        except (FileNotFoundError, ValueError):
            pass
        
        return None
    
    def _run_alerax(self, test_setup, dataset_name: str) -> Optional[float]:
        """Run AleRax on a dataset and return the likelihood (with parameter optimization)."""
        if not test_setup['alerax_binary']:
            return None
        
        dataset_dir = test_setup['test_data_dir'] / dataset_name
        families_file = dataset_dir / "families.txt"
        species_tree = dataset_dir / "sp.nwk"
        
        if not (families_file.exists() and species_tree.exists()):
            return None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Run AleRax with parameter optimization (no fixed rates specified)
                cmd = [
                    test_setup['alerax_binary'],
                    "-f", str(families_file),
                    "-s", str(species_tree), 
                    "-p", temp_dir,
                    "--gene-tree-samples", "0",
                    "--species-tree-search", "SKIP"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Extract likelihood from output file
                    likelihood_file = Path(temp_dir) / "per_fam_likelihoods.txt"
                    if likelihood_file.exists():
                        with open(likelihood_file, 'r') as f:
                            content = f.read().strip()
                            match = re.search(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?', content)
                            if match:
                                return float(match.group())
                
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return None
    
    def _get_alerax_parameters(self, test_setup, dataset_name: str) -> Dict[str, float]:
        """Extract AleRax parameters from model_parameters.txt file."""
        dataset_dir = test_setup['test_data_dir'] / dataset_name / "output"
        params_file = dataset_dir / "model_parameters" / "model_parameters.txt"
        
        if not params_file.exists():
            raise FileNotFoundError(f"AleRax model parameters file not found: {params_file}")
        
        try:
            with open(params_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header line, use first species branch parameters as representative
            # Format: "node_name D L T"
            for line in lines[1:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:
                        # Extract D, L, T values (columns 1, 2, 3)
                        delta = float(parts[1])
                        lambda_param = float(parts[2])  # L comes before T in AleRax format
                        tau = float(parts[3])
                        return {'delta': delta, 'tau': tau, 'lambda': lambda_param}
            
            raise ValueError(f"No valid parameter lines found in {params_file}")
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse AleRax parameters from {params_file}: {e}")
    
    def _run_refactored_code(self, test_setup, dataset_name: str) -> float:
        """Run refactored reconciliation code and return likelihood."""
        dataset_dir = test_setup['test_data_dir'] / dataset_name
        species_tree = dataset_dir / "sp.nwk"
        gene_tree = dataset_dir / "g.nwk"
        
        if not (species_tree.exists() and gene_tree.exists()):
            pytest.skip(f"Required files not found in {dataset_dir}")
        
        # Use AleRax parameters instead of hardcoded ones
        alerax_params = self._get_alerax_parameters(test_setup, dataset_name)
        
        # Run reconciliation with same parameters as AleRax
        result = setup_fixed_points(
            str(species_tree),
            str(gene_tree),
            delta=alerax_params['delta'],
            tau=alerax_params['tau'],
            lambda_param=alerax_params['lambda'],
            max_iters_E=100,
            max_iters_Pi=100,
            device=torch.device('cpu'),  # Use CPU for consistent results
            dtype=torch.float64,
            debug=False
        )
        
        return result['log_likelihood']
    
    @pytest.mark.integration
    @pytest.mark.parametrize("dataset_name", ["test_trees_1", "test_trees_2", "test_trees_3"])
    def test_likelihood_comparison_individual(self, test_setup, dataset_name: str):
        """Test individual datasets against AleRax."""
        # Run refactored code with AleRax parameters
        refactored_likelihood = self._run_refactored_code(test_setup, dataset_name)
        
        # Get AleRax parameters for display
        alerax_params = self._get_alerax_parameters(test_setup, dataset_name)
        
        # Try to get AleRax result from existing output file first
        alerax_likelihood = self._get_alerax_result_from_file(test_setup, dataset_name)
        
        if alerax_likelihood is None:
            # Try running AleRax if binary is available (with parameter optimization)
            alerax_likelihood = self._run_alerax(test_setup, dataset_name)
        
        if alerax_likelihood is None:
            pytest.skip(f"Could not obtain AleRax result for {dataset_name}")
        
        # Compare results
        diff = abs(refactored_likelihood - alerax_likelihood)
        tolerance = 1e-5
        
        # Print nice comparison table
        print(f"\n🧪 Likelihood Comparison for {dataset_name}")
        print("=" * 80)
        
        # Parameters table
        param_data = [
            ["Parameter", "Value"],
            ["δ (duplication)", f"{alerax_params['delta']:.2e}"],
            ["τ (transfer)", f"{alerax_params['tau']:.2e}"],
            ["λ (loss)", f"{alerax_params['lambda']:.2e}"]
        ]
        print("📊 Model Parameters:")
        print(tabulate(param_data, headers="firstrow", tablefmt="grid"))
        
        # Results comparison table
        comparison_data = [
            ["Implementation", "Log-Likelihood", "Difference", "Status"],
            ["AleRax (Reference)", f"{alerax_likelihood:.6f}", "-", "✓"],
            ["Refactored Code", f"{refactored_likelihood:.6f}", f"{diff:.2e}", "✓" if diff < tolerance else "❌"]
        ]
        print("\n🎯 Likelihood Comparison:")
        print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))
        
        if diff < tolerance:
            print(f"✅ PASS: Difference {diff:.2e} < tolerance {tolerance:.2e}")
        else:
            print(f"❌ FAIL: Difference {diff:.2e} >= tolerance {tolerance:.2e}")
        
        print("=" * 80)
        
        assert diff < tolerance, \
            f"Likelihood mismatch for {dataset_name}: " \
            f"refactored={refactored_likelihood:.6f}, " \
            f"alerax={alerax_likelihood:.6f}, " \
            f"diff={diff:.2e}"
    
    @pytest.mark.integration
    def test_known_alerax_results(self, test_setup):
        """Test against actual AleRax results from output files."""
        for dataset_name in ['test_trees_1', 'test_trees_2', 'test_trees_3', 'test_trees_200']:
            if dataset_name in self.AVAILABLE_DATASETS:
                # Get expected likelihood from AleRax output file
                expected_likelihood = self._get_alerax_result_from_file(test_setup, dataset_name)
                
                if expected_likelihood is None:
                    pytest.skip(f"No AleRax output found for {dataset_name}")
                
                # Run refactored code with AleRax parameters
                actual_likelihood = self._run_refactored_code(test_setup, dataset_name)
                
                diff = abs(actual_likelihood - expected_likelihood)
                tolerance = 1e-5
                
                assert diff < tolerance, \
                    f"Result mismatch for {dataset_name}: " \
                    f"expected={expected_likelihood:.6f}, actual={actual_likelihood:.6f}, " \
                    f"diff={diff:.2e}"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_all_datasets_consistency(self, test_setup):
        """Test that all available datasets produce consistent results."""
        results = {}
        
        for dataset_name in self.AVAILABLE_DATASETS:
            try:
                # Run with AleRax parameters
                likelihood = self._run_refactored_code(test_setup, dataset_name)
                results[dataset_name] = likelihood
            except Exception as e:
                pytest.fail(f"Failed to run {dataset_name}: {e}")
        
        # Basic sanity checks
        assert len(results) > 0, "No datasets could be processed"
        
        # All likelihoods should be negative (log probabilities)
        for dataset_name, likelihood in results.items():
            assert likelihood < 0, f"Likelihood should be negative for {dataset_name}: {likelihood}"
            assert not torch.isnan(torch.tensor(likelihood)), f"Likelihood is NaN for {dataset_name}"
            assert torch.isfinite(torch.tensor(likelihood)), f"Likelihood is not finite for {dataset_name}"
    
    def test_refactored_code_basic_functionality(self, test_setup):
        """Basic smoke test for refactored code."""
        # Use the simplest dataset
        dataset_name = 'test_trees_1'
        
        try:
            # Run with AleRax parameters
            likelihood = self._run_refactored_code(test_setup, dataset_name)
            
            # Basic checks
            assert isinstance(likelihood, float), "Likelihood should be a float"
            assert likelihood < 0, "Log likelihood should be negative"
            assert not torch.isnan(torch.tensor(likelihood)), "Likelihood should not be NaN"
            assert torch.isfinite(torch.tensor(likelihood)), "Likelihood should be finite"
            
        except Exception as e:
            pytest.fail(f"Basic functionality test failed: {e}")


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])