import pytest
from gpurax.recon.parity import recon_loglk

# 3 distinct 4-taxon gene-tree topologies (species-prefixed leaves).
T = {
    "T1": "((A_a,B_b),(C_c,D_d));",   # matches species tree ((A,B),(C,D)) -> pure speciation
    "T2": "((A_a,C_c),(B_b,D_d));",
    "T3": "((A_a,D_d),(B_b,C_c));",
}


def _ll(tmp_path, species, nwk, name):
    p = tmp_path / f"{name}.nwk"
    p.write_text(nwk + "\n")
    return recon_loglk(species, str(p), 0.2, 0.3, 0.1, device="cuda")


def test_gpurec_ranks_species_matching_topology_best(tmp_path, fixture_dir):
    sp = str(fixture_dir / "species.nwk")
    lls = {k: _ll(tmp_path, sp, v, k) for k, v in T.items()}
    # T1 (matches species tree) is most likely; T2==T3 by the C<->D species-tree symmetry.
    assert lls["T1"] > lls["T2"] and lls["T1"] > lls["T3"]
    assert lls["T2"] == pytest.approx(lls["T3"], rel=1e-4)
    # GeneRax ranks identically (T1 best) — cross-checked in scratchpad/gpurax/offset-constancy.md.
