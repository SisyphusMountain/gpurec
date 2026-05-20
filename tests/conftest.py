"""Shared fixtures and automatic coarse test markers for gpurec."""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    """Path to the shared test data directory."""
    return DATA_DIR


def pytest_collection_modifyitems(config, items):
    """Auto-apply directory-level markers."""
    unit_marker = pytest.mark.unit
    integration_marker = pytest.mark.integration
    for item in items:
        try:
            test_section = item.path.relative_to(DATA_DIR.parent).parts[0]
        except ValueError:
            test_section = ""
        if test_section == "unit":
            item.add_marker(unit_marker)
        elif test_section in {"integration", "kernels"}:
            item.add_marker(integration_marker)
