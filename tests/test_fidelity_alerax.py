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


TOL_NATS_FP64 = 1e-8


@pytest.mark.gpu
@pytest.mark.parametrize("fixture", FIXTURES)
def test_fidelity_float64_reaches_machine_precision(fixture):
    """float64 reconciliation must be GENUINE, not floored by a float32 batch static.

    The CCP split log-probs (``log_split_probs``) and the transfer row-max
    (``unnorm_row_max``) are computed in float64 by the Rust preprocessor. If either is
    quantized through float32 before an fp64 tensor is created, float64 reconciliation floors
    at ~1e-5 nats instead of reaching f64 round-off. This guards that regression: Rust retains
    f64 values through serialization, then batching materializes each tensor directly in its
    owning model or accumulator dtype. Every fixture lands within 1e-8 nats of AleRax_fixed
    (observed <=3e-12). The looser 1e-3 gate above
    would not catch a re-truncation; this one will.
    """
    import torch
    ref = DATA / fixture / "ref"
    rates = global_rates(parse_alerax_parameters(ref / "model_parameters.txt"))
    alerax_ll = parse_alerax_likelihoods(ref / "per_fam_likelihoods.txt")
    gpurec_ll = reconcile_at_alerax_rates(
        str(DATA / fixture / "sp.nwk"), [str(DATA / fixture / "g.nwk")], rates,
        device="cuda", dtype=torch.float64, mode="global")
    assert len(gpurec_ll) == 1 and len(alerax_ll) == 1
    gpurec_ll = {k: v for k, v in zip(alerax_ll, gpurec_ll.values())}
    rep = compare(gpurec_ll, alerax_ll)
    assert rep.mean_abs_delta <= TOL_NATS_FP64, (
        f"{fixture}: float64 |Δ|={rep.mean_abs_delta:.3e} nats > {TOL_NATS_FP64} "
        f"— a float32 batch static is likely flooring float64 "
        f"(gpurec={gpurec_ll}, alerax_fixed={alerax_ll})")
