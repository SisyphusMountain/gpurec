#!/usr/bin/env python3
"""
Test suite for the CLI reconcile.py script.
"""

import os
import sys
import pytest
import subprocess
import tempfile
from pathlib import Path

# Ensure we can import the CLI module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import torch for CUDA availability check
try:
    import torch
except ImportError:
    torch = None

# Path to test data
TEST_DATA_DIR = Path(__file__).parent.parent / "data"
CLI_SCRIPT = Path(__file__).parent.parent.parent / "src" / "cli" / "reconcile.py"


class TestReconcileCLI:
    """Test suite for reconcile.py CLI."""
    
    def test_cli_help(self):
        """Test that help flag works."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "reconciliation" in result.stdout.lower()
        assert "--species" in result.stdout
        assert "--gene" in result.stdout
        assert "--delta" in result.stdout
        assert "--tau" in result.stdout
        assert "--lambda" in result.stdout
    
    def test_cli_missing_required_args(self):
        """Test that missing required arguments causes error."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT)],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower()
    
    def test_cli_with_test_trees_1(self):
        """Test reconciliation with test_trees_1 dataset."""
        species_file = TEST_DATA_DIR / "test_trees_1" / "sp.nwk"
        gene_file = TEST_DATA_DIR / "test_trees_1" / "g.nwk"
        
        assert species_file.exists(), f"Species file not found: {species_file}"
        assert gene_file.exists(), f"Gene file not found: {gene_file}"
        
        result = subprocess.run(
            [
                sys.executable, str(CLI_SCRIPT),
                "--species", str(species_file),
                "--gene", str(gene_file),
                "--delta", "0.1",
                "--tau", "0.1",
                "--lambda", "0.1",
                "--iters", "10",
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        assert "Final Results" in result.stdout
        assert "Log-likelihood" in result.stdout
        
        # Extract log-likelihood value and verify it's a reasonable number
        for line in result.stdout.split('\n'):
            if "Log-likelihood" in line:
                ll_str = line.split(':')[-1].strip()
                ll_value = float(ll_str)
                assert ll_value < 0, "Log-likelihood should be negative"
                assert ll_value > -1000, "Log-likelihood seems too small"
    
    def test_cli_with_test_trees_2(self):
        """Test reconciliation with test_trees_2 dataset."""
        species_file = TEST_DATA_DIR / "test_trees_2" / "sp.nwk"
        gene_file = TEST_DATA_DIR / "test_trees_2" / "g.nwk"
        
        if not species_file.exists() or not gene_file.exists():
            pytest.skip("test_trees_2 data not found")
        
        result = subprocess.run(
            [
                sys.executable, str(CLI_SCRIPT),
                "--species", str(species_file),
                "--gene", str(gene_file),
                "--delta", "0.2",
                "--tau", "0.15",
                "--lambda", "0.1",
                "--iters", "10",
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        assert "Log-likelihood" in result.stdout
    
    def test_cli_with_debug_flag(self):
        """Test reconciliation with debug output enabled."""
        species_file = TEST_DATA_DIR / "test_trees_1" / "sp.nwk"
        gene_file = TEST_DATA_DIR / "test_trees_1" / "g.nwk"
        
        result = subprocess.run(
            [
                sys.executable, str(CLI_SCRIPT),
                "--species", str(species_file),
                "--gene", str(gene_file),
                "--delta", "0.1",
                "--tau", "0.1",
                "--lambda", "0.1",
                "--iters", "5",
                "--device", "cpu",
                "--debug"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        # Debug mode should produce more verbose output
        assert len(result.stdout) > 100, "Debug output seems too short"
    
    def test_cli_parameter_variations(self):
        """Test various parameter combinations."""
        species_file = TEST_DATA_DIR / "test_trees_1" / "sp.nwk"
        gene_file = TEST_DATA_DIR / "test_trees_1" / "g.nwk"
        
        test_params = [
            # (delta, tau, lambda)
            (1e-10, 1e-10, 1e-10),  # Very small rates
            (0.01, 0.01, 0.01),      # Small rates
            (0.5, 0.3, 0.4),         # Large rates
        ]
        
        for delta, tau, lambda_param in test_params:
            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--species", str(species_file),
                    "--gene", str(gene_file),
                    "--delta", str(delta),
                    "--tau", str(tau),
                    "--lambda", str(lambda_param),
                    "--iters", "5",
                    "--device", "cpu"
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, \
                f"Failed with params ({delta}, {tau}, {lambda_param}): {result.stderr}"
            assert "Log-likelihood" in result.stdout
    
    def test_cli_invalid_file_path(self):
        """Test error handling for invalid file paths."""
        result = subprocess.run(
            [
                sys.executable, str(CLI_SCRIPT),
                "--species", "nonexistent_species.nwk",
                "--gene", "nonexistent_gene.nwk",
                "--device", "cpu"
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0
        assert "Error" in result.stdout or "Error" in result.stderr
    
    @pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA not available")
    def test_cli_with_cuda(self):
        """Test reconciliation using CUDA if available."""
        
        species_file = TEST_DATA_DIR / "test_trees_1" / "sp.nwk"
        gene_file = TEST_DATA_DIR / "test_trees_1" / "g.nwk"
        
        result = subprocess.run(
            [
                sys.executable, str(CLI_SCRIPT),
                "--species", str(species_file),
                "--gene", str(gene_file),
                "--delta", "0.1",
                "--tau", "0.1",
                "--lambda", "0.1",
                "--iters", "10",
                "--device", "cuda"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"CUDA execution failed: {result.stderr}"
        assert "Log-likelihood" in result.stdout


class TestReconcileLargeDatasets:
    """Test with larger datasets if available."""
    
    @pytest.mark.slow
    def test_cli_with_test_trees_200(self):
        """Test with larger tree dataset (200 leaves)."""
        species_file = TEST_DATA_DIR / "test_trees_200" / "sp.nwk"
        gene_file = TEST_DATA_DIR / "test_trees_200" / "g.nwk"
        
        if not species_file.exists() or not gene_file.exists():
            pytest.skip("test_trees_200 data not found")
        
        result = subprocess.run(
            [
                sys.executable, str(CLI_SCRIPT),
                "--species", str(species_file),
                "--gene", str(gene_file),
                "--delta", "0.1",
                "--tau", "0.1",
                "--lambda", "0.1",
                "--iters", "5",  # Fewer iterations for large trees
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=120  # Longer timeout for large trees
        )
        
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        assert "Log-likelihood" in result.stdout
    
    @pytest.mark.slow
    def test_cli_with_mixed_200(self):
        """Test with mixed 200 dataset."""
        species_file = TEST_DATA_DIR / "test_mixed_200" / "sp.nwk"
        gene_file = TEST_DATA_DIR / "test_mixed_200" / "g.nwk"
        
        if not species_file.exists() or not gene_file.exists():
            pytest.skip("test_mixed_200 data not found")
        
        result = subprocess.run(
            [
                sys.executable, str(CLI_SCRIPT),
                "--species", str(species_file),
                "--gene", str(gene_file),
                "--delta", "0.1",
                "--tau", "0.1",
                "--lambda", "0.1",
                "--iters", "5",
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"
        assert "Log-likelihood" in result.stdout


def test_cli_script_exists():
    """Verify the CLI script exists and is executable."""
    assert CLI_SCRIPT.exists(), f"CLI script not found at {CLI_SCRIPT}"
    
    # Check it's a Python file
    with open(CLI_SCRIPT, 'r') as f:
        first_line = f.readline()
        assert "python" in first_line.lower() or first_line.startswith("#!")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])