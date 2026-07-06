import gpurax


def test_version():
    assert gpurax.__version__


def test_fixtures_exist(fixture_dir):
    for f in ["species.nwk", "gene.nwk", "aln.fasta", "map.link"]:
        assert (fixture_dir / f).exists()
