"""Shared fixtures and automatic test markers for gpurec."""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"

GPU_TEST_FILES = {
    "test_adaptive_iterations.py",
    "test_gene_recon_model.py",
    "test_hogenom_alerax_input.py",
    "test_specieswise_uniform.py",
    "test_stochastic_backtracking.py",
    "test_uniform_chunked_model.py",
    "test_wave_step_uniform_forward_kernel.py",
}

SLOW_TEST_PATTERNS = (
    "test_specieswise_uniform_matches_alerax_specieswise_reference",
    "test_hogenom_alerax_family_file_likelihood_matches_reference",
    "test_chunked_uniform_matches_resident_global_model",
)


@pytest.fixture
def data_dir():
    """Path to the shared test data directory."""
    return DATA_DIR


def pytest_collection_modifyitems(config, items):
    """Auto-apply coarse markers that are easy to miss in module fixtures."""
    gpu_marker = pytest.mark.gpu
    slow_marker = pytest.mark.slow
    for item in items:
        if item.path.name in GPU_TEST_FILES:
            item.add_marker(gpu_marker)
        if any(pattern in item.nodeid for pattern in SLOW_TEST_PATTERNS):
            item.add_marker(slow_marker)
        for marker in item.iter_markers("skipif"):
            reason = marker.kwargs.get("reason", "")
            if "cuda" in reason.lower():
                item.add_marker(gpu_marker)
                break
