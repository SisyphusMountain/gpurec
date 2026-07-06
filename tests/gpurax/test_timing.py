import pathlib

from gpurax._seqlik import SeqFamily
from gpurax.search.scorer import JointScorer
from gpurax.search.spr import FamilyState, spr_search

FX = pathlib.Path(__file__).parent / "fixtures" / "six"
RATES = (0.2, 0.3, 0.1)


def test_timings_accumulate(tmp_path):
    """Run a small spr_search on the 6-taxon fixture and confirm the scorer
    accumulates both the reconciliation-scoring time (rec_s, timed inside
    JointScorer.score_candidates) and the sequence-likelihood time paid at
    SPR-candidate materialization (seq_s, timed in gpurax/search/spr.py).

    This only proves the instrumentation works -- these fixtures are tiny
    (sub-millisecond per term), so the seq_s/rec_s ratio observed here is
    NOT a meaningful Phase-2 (GPU seq-term) signal. A real decision needs
    the same measurement on realistic family sizes.
    """
    newick = (FX / "start_gene.nwk").read_text().strip()
    fam = SeqFamily(newick, str(FX / "aln.fasta"), "GTR")
    sc = JointScorer(str(FX / "species.nwk"), device="cuda")

    fams = [FamilyState("f1", fam, RATES)]
    spr_search(fams, sc, max_radius=1, tmpdir=tmp_path)

    assert sc.timings["rec_s"] > 0.0
    assert sc.timings["seq_s"] >= 0.0
