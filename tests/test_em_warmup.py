"""Optimality and boundary checks for the complete-history warm-up."""
import math
import importlib
import inspect

import pytest
import torch

from gpurec.core.scheduling import batching
from gpurec.fit.em_warmup import boxed_em_m_step, complete_information, event_probabilities
from gpurec.fit.genewise_fit import TRUST_TEST_OFF, fit_genewise


LO, HI = math.log2(1e-6), math.log2(2.0)


def _fit_validation_kwargs(**overrides):
    kwargs = dict(
        warmup_method="em", init_curvature="adam_bfgs", curvature_update="bfgs",
        min_drop=1, rebuild_frac=0.25, hessian_refresh=15, trust_max=8.0,
        certify_curvature=False, init_log2_rates=(-1.0, -1.0, -1.0), stall_patience=1,
        step_extrapolation=1.0, step_model="quadratic", stop_nll_bits=0.0,
        approach_pruning_threshold=0.0, targeted_hessian=(0, 0.0),
        coordinate_staging=(0, 0), trust_test=TRUST_TEST_OFF,
    )
    kwargs.update(overrides)
    return kwargs


def test_m_step_recovers_interior_count_ratios():
    counts = torch.tensor([[50.0, 10.0, 20.0, 5.0]], dtype=torch.float64)
    theta = boxed_em_m_step(counts, LO, HI)
    torch.testing.assert_close(torch.exp2(theta), counts[:, 1:] / counts[:, :1])


def test_m_step_releases_a_bound_after_another_rate_is_pinned():
    # Both D/S and T/S exceed two initially. Pinning D increases the
    # multinomial normalizer enough that T must be released from its cap.
    counts = torch.tensor([[1.0, 100.0, 0.2, 8.0]], dtype=torch.float64)
    theta = boxed_em_m_step(counts, LO, HI)
    expected = torch.tensor([[2.0, 0.2 * 3 / 101, 8.0 * 3 / 101]], dtype=torch.float64)
    torch.testing.assert_close(torch.exp2(theta), expected)


def test_zero_counts_and_boundary_heavy_counts_satisfy_kkt():
    generator = torch.Generator().manual_seed(183)
    random_counts = torch.exp(torch.randn((256, 4), generator=generator, dtype=torch.float64) * 8)
    counts = torch.cat((random_counts, torch.eye(4, dtype=torch.float64)), dim=0)
    theta = boxed_em_m_step(counts, LO, HI)
    assert bool(((theta >= LO - 1e-12) & (theta <= HI + 1e-12)).all())
    score = counts[:, 1:] - counts.sum(dim=1, keepdim=True) * event_probabilities(theta)[:, 1:]
    violation = torch.where(theta <= LO + 1e-10, score.relu(),
                            torch.where(theta >= HI - 1e-10, (-score).relu(), score.abs()))
    assert float((violation / counts.sum(dim=1, keepdim=True)).max()) < 3e-12


def test_complete_information_matches_surrogate_hessian():
    counts = torch.tensor([[17.0, 3.0, 8.0, 5.0]], dtype=torch.float64)
    theta = torch.tensor([-2.0, -1.0, -3.0], dtype=torch.float64)

    def objective(x):
        return -(counts[0] * torch.log2(event_probabilities(x[None])[0])).sum()

    reference = torch.autograd.functional.hessian(objective, theta)
    torch.testing.assert_close(complete_information(theta[None], counts)[0], reference)


@pytest.mark.parametrize("counts", [
    [[1.0, -1.0, 1.0, 1.0]],
    [[0.0, 0.0, 0.0, 0.0]],
    [[1.0, float("nan"), 1.0, 1.0]],
])
def test_m_step_rejects_invalid_counts(counts):
    with pytest.raises(ValueError):
        boxed_em_m_step(torch.tensor(counts, dtype=torch.float64), LO, HI)


@pytest.mark.parametrize("init_curvature", ["exact", torch.eye(3).expand(1, 3, 3)])
def test_em_fit_rejects_incompatible_curvature_seed(init_curvature):
    with pytest.raises(ValueError, match=r'requires init_curvature="adam_bfgs"'):
        fit_genewise("unused", [], **_fit_validation_kwargs(init_curvature=init_curvature))


@pytest.mark.parametrize("curvature_update", ["sr1", "multisecant"])
def test_em_fit_rejects_non_bfgs_curvature_update(curvature_update):
    with pytest.raises(ValueError, match=r'requires curvature_update="bfgs"'):
        fit_genewise("unused", [], **_fit_validation_kwargs(curvature_update=curvature_update))


@pytest.mark.parametrize(("min_rate", "max_rate"), [
    (0.0, 2.0), (1e-6, 0.0), (float("nan"), 2.0),
    (1e-6, float("inf")), (1e-6, None),
])
def test_em_fit_rejects_nonfinite_or_nonpositive_bounds(min_rate, max_rate):
    with pytest.raises(ValueError, match="requires finite positive min_rate and max_rate"):
        fit_genewise("unused", [], **_fit_validation_kwargs(min_rate=min_rate, max_rate=max_rate))


def test_em_fit_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="requires max_rate >= min_rate"):
        fit_genewise("unused", [], **_fit_validation_kwargs(min_rate=2.0, max_rate=1.0))


@pytest.mark.parametrize("steps", [0, 1, 4, True, 2.0])
def test_em_fit_rejects_unsupported_step_counts(steps):
    with pytest.raises(ValueError, match="requires em_steps=2 or em_steps=3"):
        fit_genewise("unused", [], **_fit_validation_kwargs(em_steps=steps))


@pytest.mark.parametrize("method", [None, "adam", "em"])
def test_public_fit_threads_warmup_without_changing_recipe(monkeypatch, method):
    entry = importlib.import_module("gpurec.fit.dtl_fit")
    seen = {}

    def fake_fit(*args, **kwargs):
        seen.update(kwargs)
        theta = torch.zeros(1, 3)
        return dict(theta=theta, rates=torch.exp2(theta), loss_bits=3.0, n_families=1)

    monkeypatch.setattr(entry, "fit_genewise", fake_fit)
    options = {} if method is None else {"genewise_warmup_method": method}
    result = entry.fit_dtl("unused", ["unused"], "genewise", genewise_em_steps=3, **options)
    assert seen["warmup_method"] == (method or "adam")
    assert seen["em_steps"] == 3
    assert seen["adam_steps"] == 3
    assert seen["hessian_refresh"] == 15
    assert seen["curvature_update"] == "bfgs"
    assert seen["certify"] is True
    assert seen["approach_pruning_threshold"] == 0.0
    assert result["nll_bits"] == 3.0
    assert inspect.signature(fit_genewise).parameters["warmup_method"].default == "adam"


def _require_native_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    try:
        batching._load_native_module()
    except Exception as exc:
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")


@pytest.mark.gpu
@pytest.mark.parametrize("em_steps", [2, 3])
def test_fit_genewise_em_warmup_reuses_initial_model_and_records_work(tmp_path, em_steps):
    _require_native_cuda()
    species = tmp_path / "species.nwk"
    gene = tmp_path / "gene.nwk"
    species.write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
    gene.write_text("(A_1:1,(B_1:1,C_1:1)X:1)R:1;\n", encoding="utf-8")
    production_start = (math.log2(0.01), math.log2(0.1), math.log2(0.01))

    result = fit_genewise(
        species,
        [gene],
        **_fit_validation_kwargs(
            device="cuda", dtype=torch.float32, em_steps=em_steps, adam_steps=3,
            pi_tiers=(16,), neu_opt=16, neu_cert=16, clade_budget=1000,
            init_log2_rates=production_start, max_iter=1, certify=False,
        ),
    )

    assert tuple(result["theta"].shape) == (1, 3)
    assert tuple(result["curvature"].shape) == (1, 3, 3)
    assert bool(torch.isfinite(result["theta"]).all())
    assert bool(torch.isfinite(result["curvature"]).all())
    assert bool(((result["theta"] >= LO) & (result["theta"] <= HI)).all())
    assert result["n_builds"] == 1
    assert [row["phase"] for row in result["gradient_work"]] == ["em"] * em_steps + ["newton"]
    assert all(row["families"] == 1 and row["clades"] > 0 for row in result["gradient_work"])
    assert result["warmup_method"] == "em"
    assert result["em_steps"] == em_steps
    assert result["adam_seconds"] == 0.0
    assert result["em_seconds"] > 0.0
