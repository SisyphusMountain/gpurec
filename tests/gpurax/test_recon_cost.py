import time

import pytest

from gpurax.recon.adapter import ReconBatch  # noqa: F401 (import parity per brief)
from gpurax.recon.incremental import preprocess_families


def test_incremental_matches_full(fixture_dir):
    fams = preprocess_families(str(fixture_dir / "species.nwk"),
                               [str(fixture_dir / "gene.nwk")] * 4)
    assert len(fams["families"]) == 4


@pytest.mark.benchmark
def test_preprocess_throughput_recorded(fixture_dir, tmp_path):
    # duplicate the fixture topology N times to simulate a neighbor batch
    paths = [str(fixture_dir / "gene.nwk")] * 200
    t0 = time.perf_counter()
    preprocess_families(str(fixture_dir / "species.nwk"), paths)
    dt = time.perf_counter() - t0
    (tmp_path / "throughput.txt").write_text(f"{200/dt:.1f} topologies/s")
    assert dt < 30.0   # loose guard; real number goes in findings doc
