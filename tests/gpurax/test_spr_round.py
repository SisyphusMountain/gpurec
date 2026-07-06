import pathlib

from gpurax._seqlik import SeqFamily
from gpurax.search.scorer import Candidate, JointScorer
from gpurax.search.spr import spr_round

FX = pathlib.Path(__file__).parent / "fixtures" / "six"


def test_round_never_decreases_joint(tmp_path):
    newick = (FX / "start_gene.nwk").read_text().strip()
    fam = SeqFamily(newick, str(FX / "aln.fasta"), "GTR")
    sc = JointScorer(str(FX / "species.nwk"), device="cuda")
    rates = (0.2, 0.3, 0.1)

    seq0 = fam.seq_loglk(False)
    start_path = str(tmp_path / "start.nwk")
    pathlib.Path(start_path).write_text(fam.newick() + "\n")
    j0 = sc.score_candidates([Candidate(seq0, start_path, rates)])[0]

    improved, new_joint = spr_round(fam, rates, sc, radius=1, tmpdir=tmp_path, current_joint=j0)

    assert new_joint >= j0 - 1e-9
    # NOTE: spr_neighbors on balanced trees was fixed (see test_seqlik_spr.py
    # B2 fix #2 regression tests) so six/start_gene.nwk now genuinely
    # enumerates candidates at radius=1. spr_round must actually find one
    # that improves on the starting joint (this closes the vacuous-pass gap
    # from D2, where no improving move was ever available to find).
    assert improved is True
    assert new_joint > j0
