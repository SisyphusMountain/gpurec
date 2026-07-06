import pathlib

import pytest

from gpurax.search.scorer import JointScorer, Candidate
from gpurax._seqlik import SeqFamily
from gpurax.recon.parity import recon_loglk

FX = pathlib.Path(__file__).parent / "fixtures"


def test_joint_is_seq_plus_recweight_rec():
    newick = (FX / "gene.nwk").read_text().strip()
    fam = SeqFamily(newick, str(FX / "aln.fasta"), "GTR")
    seq = fam.seq_loglk(False)

    cand = Candidate(seq_loglk=seq, newick_path=str(FX / "gene.nwk"), rates=(0.2, 0.3, 0.1))

    sc = JointScorer(str(FX / "species.nwk"), rec_weight=1.0, device="cuda")
    joint = sc.score_candidates([cand])[0]

    rec = recon_loglk(str(FX / "species.nwk"), str(FX / "gene.nwk"), 0.2, 0.3, 0.1, device="cuda")

    assert joint == pytest.approx(seq + rec, rel=1e-6)
