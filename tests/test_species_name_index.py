# tests/test_species_name_index.py
import pytest

pytest.importorskip("gpurec.gpurec_preprocess")

from gpurec.core.scheduling.batching import species_name_to_index


def test_names_map_to_postorder_indices(tmp_path):
    sp = tmp_path / "species.nwk"
    sp.write_text("(A:1,B:1)Root;")
    m = species_name_to_index(str(sp))
    assert set(m) == {"A", "B", "Root"}
    assert m["A"] != m["B"]
    # Root is the last (highest) index in post-order.
    assert m["Root"] == max(m.values())
