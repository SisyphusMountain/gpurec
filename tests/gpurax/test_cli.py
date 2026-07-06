import pathlib

from gpurax.cli import main

FX = pathlib.Path(__file__).parent / "fixtures"
SIX = FX / "six"


def _write_families_file(path):
    path.write_text(
        "[FAMILIES]\n"
        "- fam1\n"
        f"starting_gene_tree = {SIX / 'start_gene.nwk'}\n"
        f"alignment = {SIX / 'aln.fasta'}\n"
        f"mapping = {SIX / 'map.link'}\n"
        "subst_model = GTR\n"
    )


def test_cli_runs(tmp_path):
    families_file = tmp_path / "families.txt"
    _write_families_file(families_file)

    rc = main([
        "-f", str(families_file),
        "-s", str(SIX / "species.nwk"),
        "-r", "UndatedDTL",
        "--max-spr-radius", "2",
        "-p", str(tmp_path),
        "--device", "cuda",
    ])
    assert rc == 0
    assert (tmp_path / "scores.tsv").exists()
