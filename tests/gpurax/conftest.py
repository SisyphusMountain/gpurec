import json
import pathlib

import pytest


@pytest.fixture
def fixture_dir():
    return pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def generax_ref(fixture_dir):
    return json.loads((fixture_dir / "generax_ref.json").read_text())
