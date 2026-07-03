import math
from pathlib import Path
from gpurec.bench.alerax_io import parse_alerax_likelihoods

DATA = Path(__file__).parent / "data" / "alerax"
FIXTURES = ["test_trees_1", "test_trees_2", "test_trees_3", "test_trees_200", "test_mixed_200"]


def _ll(path):
    return parse_alerax_likelihoods(path)["my_family"]


def _n_species(fixture):
    # rooted binary Newick with S leaves has S-1 commas
    return (DATA / fixture / "sp.nwk").read_text().count(",") + 1


def test_stock_equals_fixed_plus_branch_term():
    """Shipped stock AleRax == AleRax_fixed + ln(2S-1) (the origination branch-count term)."""
    for f in FIXTURES:
        stock = _ll(DATA / f / "per_fam_likelihoods.txt")
        ref = _ll(DATA / f / "ref" / "per_fam_likelihoods.txt")
        S = _n_species(f)
        assert abs((stock - ref) - math.log(2 * S - 1)) < 1e-2, (
            f"{f}: stock={stock} ref={ref} S={S} ln(2S-1)={math.log(2*S-1)}")


def test_test_trees_1_known_values():
    assert _ll(DATA / "test_trees_1" / "per_fam_likelihoods.txt") == -2.56495
    assert abs(_ll(DATA / "test_trees_1" / "ref" / "per_fam_likelihoods.txt") - (-5.27300)) < 1e-3
