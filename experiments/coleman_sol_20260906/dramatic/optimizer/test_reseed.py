from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import torch


THIS = Path(__file__).resolve().parent
REPO = THIS.parents[3]
HYBRID = REPO / "experiments/coleman_sol_20260906/geometry/hybrid"
sys.path.insert(0, str(HYBRID))

import gpurec.fit.genewise_fit as production
from fresh_seed import FreshReseeder
from reseed_adapter import compile_hierarchical_reseed, compile_native_reseed


def _inputs():
    artifact = torch.load(
        REPO / "experiments/coleman_sol_20260906/em/hybrid_shared_200_v2.pt",
        map_location="cpu", weights_only=False,
    )
    trace = torch.load(
        REPO / "experiments/coleman_sol_20260906/geometry/hybrid/diagnosis/trace200.pt",
        map_location="cpu", weights_only=False,
    )
    candidates = torch.load(
        REPO / "experiments/coleman_sol_20260906/em/hybrid_diagnosis/latest_secant_first_step.pt",
        map_location="cpu", weights_only=False,
    )
    return artifact, trace, candidates


def test_fresh_native_seed_matches_independent_candidate():
    artifact, trace, candidates = _inputs()
    lo, hi = artifact["metadata"]["rate_box_log2"]
    reseeder = FreshReseeder(
        coordinate="native", theta1=artifact["theta_native"]["theta1"],
        gradient1=artifact["gradient_theta"]["g1"], counts1=artifact["event_counts"]["N1"],
        lo=lo, hi=hi,
    )
    proposal = trace["arms"]["native"]["trace"]["proposals"][0]
    seed = reseeder(artifact["theta_native"]["theta2"].float(), proposal["gradient_theta"])
    torch.testing.assert_close(
        seed, candidates["candidates"]["latest_native"]["seed"], rtol=0, atol=0,
    )
    assert reseeder.calls == 1


def test_fresh_hierarchical_seed_matches_independent_candidate():
    artifact, trace, candidates = _inputs()
    lo, hi = artifact["metadata"]["rate_box_log2"]
    reseeder = FreshReseeder(
        coordinate="hierarchical", theta1=artifact["theta_native"]["theta1"],
        gradient1=artifact["gradient_theta"]["g1"], counts1=artifact["event_counts"]["N1"],
        lo=lo, hi=hi,
    )
    proposal = trace["arms"]["hierarchical_native_metric"]["trace"]["proposals"][0]
    seed = reseeder(artifact["theta_native"]["theta2"].float(), proposal["gradient_theta"])
    torch.testing.assert_close(
        seed, candidates["candidates"]["latest_hierarchical"]["seed"], rtol=0, atol=0,
    )
    assert reseeder.calls == 1


def test_source_adapters_compile_from_unchanged_production():
    artifact, _trace, _candidates = _inputs()
    lo, hi = artifact["metadata"]["rate_box_log2"]
    before = inspect.getsource(production.fit_genewise)
    source_hash = hashlib.sha256(before.encode()).hexdigest()
    native = FreshReseeder(
        coordinate="native", theta1=artifact["theta_native"]["theta1"],
        gradient1=artifact["gradient_theta"]["g1"], counts1=artifact["event_counts"]["N1"],
        lo=lo, hi=hi,
    )
    hierarchy = FreshReseeder(
        coordinate="hierarchical", theta1=artifact["theta_native"]["theta1"],
        gradient1=artifact["gradient_theta"]["g1"], counts1=artifact["event_counts"]["N1"],
        lo=lo, hi=hi,
    )
    _fit_n, hash_n = compile_native_reseed(native)
    _fit_h, hash_h = compile_hierarchical_reseed(hierarchy)
    assert hash_n == hash_h == source_hash
    assert inspect.getsource(production.fit_genewise) == before

