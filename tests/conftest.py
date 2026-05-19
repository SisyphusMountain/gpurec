"""Shared fixtures and automatic coarse test markers for gpurec."""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"

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
    """Auto-apply directory and known slow markers."""
    slow_marker = pytest.mark.slow
    unit_marker = pytest.mark.unit
    integration_marker = pytest.mark.integration
    for item in items:
        try:
            test_section = item.path.relative_to(DATA_DIR.parent).parts[0]
        except ValueError:
            test_section = ""
        if test_section == "unit":
            item.add_marker(unit_marker)
        elif test_section == "integration":
            item.add_marker(integration_marker)
        if any(pattern in item.nodeid for pattern in SLOW_TEST_PATTERNS):
            item.add_marker(slow_marker)
