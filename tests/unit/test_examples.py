from __future__ import annotations

from pathlib import Path

from gpurec.core.model import parse_alerax_family_file
from gpurec.workflow.config import RunConfig


ROOT = Path(__file__).resolve().parents[2]


def test_minimal_run_config_example_loads_and_points_to_tiny_fixture():
    config = RunConfig.from_json(ROOT / "examples" / "minimal-run-config.json")

    assert config.species_tree == (
        ROOT / "examples" / "tiny" / "species.nwk"
    ).resolve()
    assert config.families_file == (
        ROOT / "examples" / "tiny" / "families.txt"
    ).resolve()
    assert config.mode == "genewise"
    assert config.device == "cuda"
    assert config.max_families == 1
    assert config.steps == 10

    names, tree_paths, leaf_maps = parse_alerax_family_file(config.families_file)
    assert names == ["tiny_family"]
    assert tree_paths == [
        [str((ROOT / "examples" / "tiny" / "gene.nwk").resolve())]
    ]
    assert leaf_maps == [{"a": "A", "b": "B"}]
