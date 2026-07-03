import math
import pytest
from gpurec.bench.alerax_io import (
    parse_alerax_likelihoods, parse_alerax_parameters, norm_family_name,
    AleraxRates, global_rates,
)


def test_parse_likelihoods(tmp_path):
    p = tmp_path / "per_fam_likelihoods.txt"
    p.write_text("my_family -2.56495\nother.ale -8.5\nbad line only\n")
    assert parse_alerax_likelihoods(p) == {"my_family": -2.56495, "other": -8.5}


def test_parse_parameters_DLT_order(tmp_path):
    p = tmp_path / "model_parameters.txt"
    p.write_text("# node D L T\n5 0.1 0.2 0.3\nn6 0.1 0.2 0.3\n")
    out = parse_alerax_parameters(p)
    # AleRax columns are D L T, the SAME order as gpurec theta = [D, L, T] (no reorder).
    assert out["5"] == AleraxRates(D=0.1, L=0.2, T=0.3)


def test_norm_family_name():
    assert norm_family_name("my_family.ale") == "my_family"
    assert norm_family_name("some/path/fam.newick") == "fam"
    assert norm_family_name("plain") == "plain"


def test_global_rates_asserts_uniform(tmp_path):
    p = tmp_path / "mp.txt"
    p.write_text("# node D L T\n5 0.1 0.2 0.3\n6 0.1 0.2 0.3\n")
    assert global_rates(parse_alerax_parameters(p)) == AleraxRates(0.1, 0.2, 0.3)
    p2 = tmp_path / "mp2.txt"
    p2.write_text("# node D L T\n5 0.1 0.2 0.3\n6 0.9 0.2 0.3\n")
    with pytest.raises(ValueError):
        global_rates(parse_alerax_parameters(p2))
