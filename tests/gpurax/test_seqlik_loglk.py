import math
import pathlib

import pytest

from gpurax._seqlik import SeqFamily


def test_seq_loglk_matches_generax(fixture_dir, generax_ref):
    if not generax_ref.get("available", True):
        pytest.xfail("no GeneRax binary")
    newick = (fixture_dir / "gene.nwk").read_text().strip()
    fam = SeqFamily(newick, str(fixture_dir / "aln.fasta"), "GTR")
    ll = fam.optimize_all()  # optimized model + branch lengths, nats
    assert ll == pytest.approx(generax_ref["seq_loglk_nats"], rel=1e-4)


def test_seq_loglk_is_finite_and_negative():
    # self-consistency, always runs
    newick = "((a_A:0.1,b_B:0.1):0.1,(c_C:0.1,d_D:0.1):0.1);"
    fx = pathlib.Path(__file__).parent / "fixtures"
    fam = SeqFamily(newick, str(fx / "aln.fasta"), "GTR")
    ll = fam.seq_loglk(opt_bl=False)
    assert math.isfinite(ll) and ll < 0
