import pathlib

import pytest

from gpurax._seqlik import SeqFamily
from gpurax.search.scorer import Candidate, JointScorer
from gpurax.search.spr import FamilyState, spr_round, spr_search

FX = pathlib.Path(__file__).parent / "fixtures" / "six"

RATES = (0.2, 0.3, 0.1)
MAX_RADIUS = 2


def _fresh_seqfam():
    newick = (FX / "start_gene.nwk").read_text().strip()
    return SeqFamily(newick, str(FX / "aln.fasta"), "GTR")


def _score_current(fam, scorer, tmpdir, tag, warmup=False):
    """Score a family's current (unmoved) topology, matching the exact
    scoring path spr_round/spr_search use for the starting joint."""
    seq = 0.0 if warmup else fam.seq_loglk(False)
    path = str(pathlib.Path(tmpdir) / f"cur_{tag}.nwk")
    pathlib.Path(path).write_text(fam.newick() + "\n")
    return scorer.score_candidates([Candidate(seq, path, RATES)])[0]


def _independent_search(fam, scorer, tmpdir, tag):
    """Reference per-family search: run spr_round to convergence at each
    radius 1..MAX_RADIUS, exactly mirroring the trajectory spr_search should
    reproduce when families are batched together."""
    current = _score_current(fam, scorer, tmpdir, f"{tag}_init")
    for radius in range(1, MAX_RADIUS + 1):
        while True:
            improved, current = spr_round(
                fam, RATES, scorer, radius, tmpdir, current
            )
            if not improved:
                break
    return current


def test_two_families_converge_equivalently_to_per_family(tmp_path):
    fams = [
        FamilyState("f1", _fresh_seqfam(), RATES),
        FamilyState("f2", _fresh_seqfam(), RATES),
    ]
    sc = JointScorer(str(FX / "species.nwk"), device="cuda")

    start_hash = fams[0].seqfam.tree_hash()
    assert fams[1].seqfam.tree_hash() == start_hash

    j0 = {
        fs.name: _score_current(fs.seqfam, sc, tmp_path, f"start_{fs.name}")
        for fs in fams
    }

    spr_search(fams, sc, max_radius=MAX_RADIUS, tmpdir=tmp_path)

    # search terminates and leaves each family a valid tree
    for fs in fams:
        assert fs.seqfam.newick().count("(") >= 1

    # joint never decreased for either family
    for fs in fams:
        j_final = _score_current(fs.seqfam, sc, tmp_path, f"final_{fs.name}")
        assert j_final >= j0[fs.name] - 1e-6

    # KEY PROPERTY: batched search reaches the SAME final topology as running
    # each family's search independently (single-family spr_round loop to
    # convergence over the same radii).
    ref1 = _fresh_seqfam()
    ref_hash1 = _independent_search(ref1, sc, tmp_path, "ref1") and ref1.tree_hash()

    ref2 = _fresh_seqfam()
    ref_hash2 = _independent_search(ref2, sc, tmp_path, "ref2") and ref2.tree_hash()

    assert fams[0].seqfam.tree_hash() == ref_hash1
    assert fams[1].seqfam.tree_hash() == ref_hash2


def test_warmup_mode_runs_and_terminates(tmp_path):
    # warmup=True forces the seq term to 0.0 (reconciliation-only SPR,
    # mirrors GeneRax --rec-radius). This is just a smoke test: it must run
    # to completion without error and leave a valid tree per family.
    fams = [FamilyState("f1", _fresh_seqfam(), RATES)]
    sc = JointScorer(str(FX / "species.nwk"), device="cuda")

    spr_search(fams, sc, max_radius=MAX_RADIUS, tmpdir=tmp_path, warmup=True)

    assert fams[0].seqfam.newick().count("(") >= 1
