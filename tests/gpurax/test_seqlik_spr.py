import pathlib

import pytest

from gpurax._seqlik import SeqFamily

FX = pathlib.Path(__file__).parent / "fixtures"


def make():
    newick = (FX / "gene.nwk").read_text().strip()
    return SeqFamily(newick, str(FX / "aln.fasta"), "GTR")


def test_seqfamily_still_loads_the_shared_fixture():
    # Sanity check that spr_neighbors/tree_hash don't crash on the existing
    # B1 fixture. gene.nwk is a 4-taxon tree: ((A_a,B_b),(C_c,D_d)). An
    # unrooted 4-taxon tree has three possible topologies (AB|CD, AC|BD,
    # AD|BC), so SPR CAN reach the two others; the earlier `== []` assertion
    # here encoded the very over-rejection bug this task fixes (rotations
    # collapsed to logical nodes, so no move was ever emitted). Correct
    # behavior: spr_neighbors is non-empty and every returned move genuinely
    # changes the topology and round-trips via rollback().
    fam = make()
    h0 = fam.tree_hash()
    assert isinstance(h0, int)
    moves = fam.spr_neighbors(1)
    assert len(moves) > 0
    for p, r, _ in moves:
        fam.apply_spr(p, r)
        assert fam.tree_hash() != h0
        fam.rollback()
        assert fam.tree_hash() == h0


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


# --- Balanced/symmetric-tree enumeration (Task B2 fix #2) --------------
#
# The prior no-op filter collapsed prune/regraft to one canonical rotation
# per logical node (a logical-array-position index space), which discarded
# 2/3 of all prune and regraft targets and over-rejected genuine moves whose
# logical node coincided (under the tree's symmetry) with a no-op edge's
# logical node. Balanced 6-taxon trees therefore enumerated ZERO neighbors.
# The fix works at directed-edge (rotation) granularity like GeneRax; these
# tests gate it against an independent brute-force ground truth.

SIX_ALIGNMENT = (
    ">A_a\nACGTACGTACGTAAGT\n"
    ">B_b\nACGTACGTACGTAAGT\n"
    ">C_c\nACGTTCGTACGTTAGT\n"
    ">D_d\nACGTTCGTACGTTAGT\n"
    ">E_e\nACGTACGTACGTTAGT\n"
    ">F_f\nACGTTCGTACGTAAGT\n"
)

BALANCED_6 = "(((A_a,B_b),(C_c,D_d)),(E_e,F_f));"
CATERPILLAR_6 = "(((((A_a,B_b),C_c),D_d),E_e),F_f);"

# Any radius >= tree diameter covers the whole tree; 6 taxa need far less.
FULL_RADIUS = 10
# Node-index space of a 6-taxon unrooted tree tops out well under this; the
# brute force just skips indices that don't resolve to a directed subnode.
INDEX_UPPER_BOUND = 40


def _six_family(tmp_path, newick):
    aln_path = tmp_path / "six_aln.fasta"
    aln_path.write_text(SIX_ALIGNMENT)
    return SeqFamily(newick, str(aln_path), "GTR")


def _brute_force_neighbor_hashes(fam):
    # Independent ground truth: for every ordered pair of node indices, try
    # apply_spr (skipping pairs that raise — out-of-range indices, tip
    # prunes, no-op moves, and regrafts inside the pruned subtree all raise,
    # never crash), record the resulting tree_hash if it differs from the
    # original, then roll back. The ground-truth set is the DISTINCT set of
    # resulting topology hashes. Shape-agnostic.
    h0 = fam.tree_hash()
    gt = set()
    for p in range(INDEX_UPPER_BOUND):
        for r in range(INDEX_UPPER_BOUND):
            try:
                fam.apply_spr(p, r)
            except RuntimeError:
                continue
            h = fam.tree_hash()
            if h != h0:
                gt.add(h)
            fam.rollback()
    return gt


def _enumerated_neighbor_hashes(fam, radius):
    h0 = fam.tree_hash()
    hashes = set()
    for p, r, _ in fam.spr_neighbors(radius):
        fam.apply_spr(p, r)
        h = fam.tree_hash()
        assert h != h0, f"identity move slipped through: (prune={p}, regraft={r})"
        hashes.add(h)
        fam.rollback()
    return hashes


@pytest.mark.parametrize("newick", [BALANCED_6, CATERPILLAR_6])
def test_balanced_tree_has_neighbors(tmp_path, newick):
    # A balanced 6-taxon tree used to enumerate ZERO neighbors; both shapes
    # must now yield a non-empty set, and spr_neighbors at a radius covering
    # the whole tree must reproduce EXACTLY the brute-force ground-truth set
    # of distinct-topology moves (no genuine neighbor missing, no identity
    # move included).
    ground_truth = _brute_force_neighbor_hashes(_six_family(tmp_path, newick))
    assert len(ground_truth) > 0

    enumerated = _enumerated_neighbor_hashes(
        _six_family(tmp_path, newick), FULL_RADIUS
    )
    assert len(enumerated) > 0
    assert enumerated == ground_truth, (
        f"missing={len(ground_truth - enumerated)} "
        f"extra={len(enumerated - ground_truth)}"
    )


def test_start_can_reach_true():
    # The real fixtures are balanced; the search must be able to move from
    # start_gene toward true_gene. start and true are RF=4 apart, reachable
    # by <= 2 SPR moves. Assert true's topology hash is reachable within two
    # applied SPR moves (each drawn from spr_neighbors of the current tree).
    six = FX / "six"
    aln = str(six / "aln.fasta")
    start = (six / "start_gene.nwk").read_text().strip()
    true = (six / "true_gene.nwk").read_text().strip()
    target = SeqFamily(true, aln, "GTR").tree_hash()

    assert SeqFamily(start, aln, "GTR").tree_hash() != target

    def neighbor_newicks(newick):
        fam = SeqFamily(newick, aln, "GTR")
        out = []
        for p, r, _ in fam.spr_neighbors(3):
            fam.apply_spr(p, r)
            out.append(fam.newick())
            fam.rollback()
        return out

    depth1 = neighbor_newicks(start)
    assert len(depth1) > 0
    reachable = False
    for nw in depth1:
        if SeqFamily(nw, aln, "GTR").tree_hash() == target:
            reachable = True
            break
        for nw2 in neighbor_newicks(nw):
            if SeqFamily(nw2, aln, "GTR").tree_hash() == target:
                reachable = True
                break
        if reachable:
            break
    assert reachable, "true_gene topology not reachable from start_gene within 2 SPR moves"
