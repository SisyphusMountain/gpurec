import pytest
from pathlib import Path
from gpurec.bench.alerax_io import parse_alerax_likelihoods, parse_alerax_parameters, global_rates

DATA = Path(__file__).parent / "data" / "alerax"
EXPECTED_LL = {
    "test_trees_1": -2.56495, "test_trees_2": -8.72486, "test_trees_3": -6.75086,
    "test_trees_200": -5.98394, "test_mixed_200": -6215.73,
}


@pytest.mark.parametrize("fixture,expected", list(EXPECTED_LL.items()))
def test_fixture_likelihood_value(fixture, expected):
    ll = parse_alerax_likelihoods(DATA / fixture / "per_fam_likelihoods.txt")
    assert ll["my_family"] == pytest.approx(expected)


@pytest.mark.parametrize("fixture", list(EXPECTED_LL))
def test_fixture_parameters_are_global(fixture):
    r = global_rates(parse_alerax_parameters(DATA / fixture / "model_parameters.txt"))
    assert r.D > 0 and r.L > 0 and r.T > 0


def test_archaea60_species_tree_present():
    tree = DATA.parent / "archaea60" / "reference_species_tree.newick"
    assert tree.stat().st_size > 0
    assert tree.read_text().count(",") >= 59        # >=60 tips -> >=59 commas
