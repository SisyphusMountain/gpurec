import pathlib

from gpurax.cli import main

FX = pathlib.Path(__file__).parent / "fixtures"
SIX = FX / "six"


def test_cli_runs(tmp_path):
    rc = main([
        "-f", str(SIX / "families.txt"),
        "-s", str(SIX / "species.nwk"),
        "-r", "UndatedDTL",
        "--max-spr-radius", "2",
        "-p", str(tmp_path),
        "--device", "cuda",
    ])
    assert rc == 0
    assert (tmp_path / "scores.tsv").exists()
