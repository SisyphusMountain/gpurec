"""Shared fixtures and automatic coarse test markers for gpurec."""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


def _tree_data_dir(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not found")
    return path


@pytest.fixture
def data_dir():
    """Path to the shared test data directory."""
    return DATA_DIR


@pytest.fixture(scope="module")
def data_dir_100():
    """Path to the 100-family AleRax test tree fixture."""
    return _tree_data_dir("test_trees_100")


@pytest.fixture(scope="module")
def data_dir_1000():
    """Path to the 1000-family AleRax test tree fixture."""
    return _tree_data_dir("test_trees_1000")


def pytest_collection_modifyitems(config, items):
    """Auto-apply directory-level markers."""
    unit_marker = pytest.mark.unit
    integration_marker = pytest.mark.integration
    kernel_marker = pytest.mark.kernel
    for item in items:
        try:
            test_section = item.path.relative_to(DATA_DIR.parent).parts[0]
        except ValueError:
            test_section = ""
        if test_section == "unit":
            item.add_marker(unit_marker)
        elif test_section == "integration":
            item.add_marker(integration_marker)
        elif test_section == "kernels":
            item.add_marker(integration_marker)
            item.add_marker(kernel_marker)
