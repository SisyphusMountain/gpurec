import pathlib

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
