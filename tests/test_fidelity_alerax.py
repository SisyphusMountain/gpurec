import pytest
from pathlib import Path
from gpurec.bench.alerax_io import parse_alerax_likelihoods, parse_alerax_parameters, global_rates
from gpurec.bench.fidelity import reconcile_at_alerax_rates, compare

DATA = Path(__file__).parent / "data" / "alerax"
FIXTURES = ["test_trees_1", "test_trees_2", "test_trees_3", "test_trees_200", "test_mixed_200"]
TOL_NATS = 1e-3


@pytest.mark.gpu
@pytest.mark.parametrize("fixture", FIXTURES)
def test_fidelity_matches_alerax_fixed(fixture):
    import torch
    ref = DATA / fixture / "ref"
    rates = global_rates(parse_alerax_parameters(ref / "model_parameters.txt"))
    alerax_ll = parse_alerax_likelihoods(ref / "per_fam_likelihoods.txt")
    gpurec_ll = reconcile_at_alerax_rates(
        str(DATA / fixture / "sp.nwk"), [str(DATA / fixture / "g.nwk")], rates,
        device="cuda", dtype=torch.float64, mode="global")
    # reconcile_at_alerax_rates keys its result by the gene-tree FILE basename
    # (g.nwk -> "g"), while ref/per_fam_likelihoods.txt keys by the AleRax family
    # NAME ("my_family"). These fixtures are single-family, so remap gpurec's
    # single value onto the ref's family key before comparing. (The real
    # archaea60-scale dataset names its gene files after the family, so there
    # the two keys already align naturally and no remap is needed.)
    assert len(gpurec_ll) == 1 and len(alerax_ll) == 1
    gpurec_ll = {k: v for k, v in zip(alerax_ll, gpurec_ll.values())}
    rep = compare(gpurec_ll, alerax_ll)
    assert rep.n_matched == 1, f"{fixture}: name mismatch {gpurec_ll} vs {alerax_ll}"
    assert rep.mean_abs_delta <= TOL_NATS, (
        f"{fixture}: |Δ|={rep.mean_abs_delta:.3e} nats > {TOL_NATS} "
        f"(gpurec={gpurec_ll}, alerax_fixed={alerax_ll})")
