"""Dataset paths shared by developer verification gates."""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "GPUREC_DATA_ROOT",
        REPO_ROOT / "data" / "external" / "benchmarks" / "large_dataset_capacity" / "datasets",
    )
).expanduser().resolve()
HOGENOM_ROOT = DATA_ROOT / "alerax_hogenom_core" / "hogenom"
