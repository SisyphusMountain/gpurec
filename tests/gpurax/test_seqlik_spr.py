import pathlib

import pytest

from gpurax._seqlik import SeqFamily

FX = pathlib.Path(__file__).parent / "fixtures"


def make():
    newick = (FX / "gene.nwk").read_text().strip()
    return SeqFamily(newick, str(FX / "aln.fasta"), "GTR")


def test_seqfamily_still_loads_the_shared_fixture():
    # Sanity check that spr_neighbors/tree_hash don't crash on the existing
    # B1 fixture. gene.nwk is a 4-taxon tree: ((A_a,B_b),(C_c,D_d)); its only
    # internal edge has two leaves as immediate neighbors on each side, so
    # corax_utree_spr (which requires an inner-node prune point, and rejects
    # regrafting onto an immediate neighbor as a no-op) structurally has no
    # non-identity move available anywhere in this tree, at any radius. So
    # this only checks spr_neighbors runs cleanly and tree_hash is stable;
    # the real neighbor-enumeration/apply/rollback checks below use a
    # 6-taxon tree that actually has room for a non-trivial SPR move.
    fam = make()
    assert fam.spr_neighbors(1) == []
    assert isinstance(fam.tree_hash(), int)


# gene.nwk (4 taxa) can't exercise a real SPR move (see above), so build a
# slightly bigger 6-taxon tree/alignment on the fly (not a committed
# fixture) with an internal edge (the AB-cherry <-> the root edge) that has
# an internal neighbor to regraft past.
SPR_NEWICK = "(((A,B),C),((D,E),F));"
SPR_ALIGNMENT = (
    ">A\n"
    "ACGTACGTACGTAAGT\n"
    ">B\n"
    "ACGTACGTACGTAAGT\n"
    ">C\n"
    "ACGTTCGTACGTTAGT\n"
    ">D\n"
    "ACGTTCGTACGTTAGT\n"
    ">E\n"
    "ACGTACGTACGTTAGT\n"
    ">F\n"
    "ACGTTCGTACGTAAGT\n"
)


def make_spr_family(tmp_path):
    aln_path = tmp_path / "spr_aln.fasta"
    aln_path.write_text(SPR_ALIGNMENT)
    return SeqFamily(SPR_NEWICK, str(aln_path), "GTR")


def test_neighbors_nonempty_at_radius1(tmp_path):
    fam = make_spr_family(tmp_path)
    nbrs = fam.spr_neighbors(1)
    assert len(nbrs) > 0
    p, r, path = nbrs[0]
    assert isinstance(p, int) and isinstance(r, int) and isinstance(path, list)


def test_apply_then_rollback_restores_tree(tmp_path):
    fam = make_spr_family(tmp_path)
    h0 = fam.tree_hash()
    p, r, _ = fam.spr_neighbors(1)[0]
    fam.apply_spr(p, r)
    assert fam.tree_hash() != h0
    fam.rollback()
    assert fam.tree_hash() == h0


# --- Review findings regression tests ---------------------------------


def test_no_identity_moves(tmp_path):
    # HIGH finding: sprYeldsSameTree's pointer-identity check missed no-op
    # moves once regraft/prune were canonicalized to array-stored rotations
    # (~0.9% of radius-1 candidates were identity moves in the buggy build).
    # Gate: every move returned by spr_neighbors(radius) must actually
    # change the topology, and must cleanly round-trip via rollback().
    for radius in (1, 2):
        fam = make_spr_family(tmp_path)
        h0 = fam.tree_hash()
        moves = fam.spr_neighbors(radius)
        assert len(moves) > 0
        for p, r, _ in moves:
            fam.apply_spr(p, r)
            assert fam.tree_hash() != h0, (
                f"identity move slipped through at radius={radius}: "
                f"(prune={p}, regraft={r})"
            )
            fam.rollback()
            assert fam.tree_hash() == h0


def test_apply_spr_out_of_range_raises(tmp_path):
    # CRITICAL finding: apply_spr had no bounds-checking and segfaulted on
    # out-of-range indices. Must now raise instead of crashing.
    fam = make_spr_family(tmp_path)
    with pytest.raises(RuntimeError):
        fam.apply_spr(10**9, 0)
    with pytest.raises(RuntimeError):
        fam.apply_spr(0, 10**9)


def test_rollback_without_move_raises(tmp_path):
    # MEDIUM finding: _lastRollback was uninitialized, so calling rollback()
    # before any apply_spr() ran corax_tree_rollback() on garbage (UB). Must
    # now raise cleanly instead.
    fam = make_spr_family(tmp_path)
    with pytest.raises(RuntimeError):
        fam.rollback()


def test_neighbors_deterministic(tmp_path):
    # LOW finding: spr_neighbors' internal getBranches() is an
    # unordered_set keyed on pointer address, so the raw enumeration order
    # is nondeterministic; the output must be sorted before returning.
    fam_a = make_spr_family(tmp_path)
    fam_b = make_spr_family(tmp_path)
    assert fam_a.spr_neighbors(1) == fam_a.spr_neighbors(1)
    assert fam_a.spr_neighbors(1) == fam_b.spr_neighbors(1)
