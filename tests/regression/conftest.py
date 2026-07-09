# tests/regression/conftest.py
import pytest


@pytest.fixture(scope="session")
def _sim_root(tmp_path_factory):
    return tmp_path_factory.mktemp("gpurec_regression_datasets")


@pytest.fixture(scope="session")
def dataset(_sim_root):
    """Session-cached regen: dataset(mode) -> (species_path, [gene_paths]) at FULL scale."""
    pytest.importorskip("rustree")
    from gpurec.bench.simulate import simulate_dataset, SIM_PARAMS
    cache = {}

    def _get(mode):
        if mode not in cache:
            p = SIM_PARAMS[mode]
            cache[mode] = simulate_dataset(mode, _sim_root / mode, n_species=p["n_species"],
                                           n_families=p["n_families"], dtl=p["dtl"], seed=p["seed"])
        return cache[mode]

    return _get
