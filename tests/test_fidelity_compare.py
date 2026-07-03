import math
from gpurec.bench.fidelity import compare, FidelityReport


def test_compare_matches_and_deltas():
    rep = compare({"a.ale": -2.0, "b": -3.0}, {"a": -2.0005, "b.txt": -2.9990})
    assert isinstance(rep, FidelityReport)
    assert rep.n_matched == 2
    assert rep.max_abs_delta < 1e-3 + 1e-9
    assert rep.pearson_r is None            # n <= 2 -> undefined


def test_compare_no_overlap():
    rep = compare({"x": -1.0}, {"y": -1.0})
    assert rep.n_matched == 0
    assert math.isnan(rep.mean_abs_delta)


def test_compare_pearson_when_enough_points():
    g = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
    a = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
    rep = compare(g, a)
    assert rep.n_matched == 4
    assert rep.pearson_r is not None and rep.pearson_r > 0.999
