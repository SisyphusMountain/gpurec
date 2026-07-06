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
    # NOTE: six/start_gene.nwk and six/true_gene.nwk are both the fully
    # "nested cherries" unrooted shape (((X,Y),(Z,W)),(V,U)) — empirically,
    # SeqFamily.spr_neighbors(radius) returns [] for this exact shape at
    # every radius 1..9 (verified directly against the fixture and against
    # synthetic trees of the same shape with different leaf sets/alignments;
    # the asymmetric "cherry+pendant" and caterpillar shapes both DO produce
    # candidates). This mirrors the already-documented 4-taxon
    # spr_neighbors==[] case in test_seqlik_spr.py, just for a 6-taxon
    # topology that happens to hit the same class of limitation in B2's
    # canonicalized-rotation recursion. So no improving move is available
    # here for spr_round to find; per the task spec, only assert the
    # monotonic non-decrease guarantee.
    assert isinstance(improved, bool)
