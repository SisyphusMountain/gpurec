"""Shared fixtures and automatic test markers for gpurec."""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    """Path to the shared test data directory."""
    return DATA_DIR


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``gpu`` marker to tests that skip without CUDA."""
    gpu_marker = pytest.mark.gpu
    for item in items:
        for marker in item.iter_markers("skipif"):
            reason = marker.kwargs.get("reason", "")
            if "cuda" in reason.lower():
                item.add_marker(gpu_marker)
                break
