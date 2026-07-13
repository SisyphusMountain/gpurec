"""Survival normalizer must be computed without linear-space cancellation.

``survival = 1 - sum_s w_s 2^{E_s}`` was historically computed as ``1 - torch.exp2(E).mean()``
(+ ``clamp_min(tiny)``).  That round-trips log -> linear -> log and catastrophically cancels:
once ``2^{E_s}`` rounds to ``1.0`` (fp32: extinction within ~1.2e-7 of 1), the mean is ``1.0`` and
``1 - mean`` collapses to ``0`` (clamped), destroying any survival probability below ~1e-6 and
producing spurious ~1e36 gradients.  The compensated form
``sum_s w_s (1 - 2^{E_s}) = sum_s w_s (-expm1(E_s ln2))`` cannot cancel (weighted sum of
non-negative terms) and matches fp64 to the last digit.
"""
import math
from types import SimpleNamespace

import pytest
import torch

from gpurec.core.inference.logspace import log2_survival, logsumexp2, survival_from_E
from gpurec.core.inference.solver import nll_from_root_rows
from gpurec.api._implicit_grad import (
    _likelihood_log2_survival,
    _likelihood_root_seed,
)
from gpurec.solver import ggn

LN2 = math.log(2.0)


def _ref_survival(E, weights=None):
    """Exact fp64 survival via the compensated identity (naive 1-mean(2^E) cancels even in fp64)."""
    s = -torch.expm1(E.double() * LN2)
    return s.mean(dim=-1) if weights is None else (weights.double() * s).sum(dim=-1)


def _E_for_survival(target, S, seed):
    """A length-S log2-extinction vector whose survival ~ ``target`` (built in fp64, no cancellation)."""
    g = torch.Generator().manual_seed(seed)
    v = (target / LN2) * (0.5 + torch.rand(S, generator=g, dtype=torch.float64))  # per-species survival scale
    return -v  # E_s = log2(extinction) <= 0


@pytest.mark.parametrize("target", [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-20, 1e-30])
def test_survival_matches_fp64(target):
    E64 = _E_for_survival(target, S=64, seed=0)
    ref = float(_ref_survival(E64))
    got = survival_from_E(E64.to(torch.float32))
    assert torch.isfinite(got).all()
    assert abs(float(got) - ref) <= 1e-4 * ref + 1e-40, (float(got), ref)
    got_l2 = float(log2_survival(E64.to(torch.float32)))
    assert abs(got_l2 - math.log2(ref)) <= 1e-3 * abs(math.log2(ref)) + 1e-3


def test_weighted_survival_matches_fp64():
    S = 48
    E64 = _E_for_survival(1e-9, S=S, seed=3)
    g = torch.Generator().manual_seed(7)
    logits = torch.rand(S, generator=g, dtype=torch.float64)
    w = torch.softmax(logits, dim=0)  # origination probs sum to 1
    ref = float(_ref_survival(E64, w))
    got = float(survival_from_E(E64.to(torch.float32), w.to(torch.float32)))
    assert math.isfinite(got)
    assert abs(got - ref) <= 1e-4 * ref, (got, ref)


@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_nll_head_uses_configured_accumulator(accumulator_dtype):
    root = torch.linspace(-308.0, -295.0, 1331, dtype=torch.float32).reshape(1, -1)
    extinction = torch.linspace(-2.0, -0.1, 1331, dtype=torch.float32).reshape(1, -1)

    got = nll_from_root_rows(
        root,
        extinction,
        accumulator_dtype=accumulator_dtype,
    )
    reference = nll_from_root_rows(
        root.to(dtype=accumulator_dtype),
        extinction.to(dtype=accumulator_dtype),
    )

    assert got.dtype == accumulator_dtype
    torch.testing.assert_close(got, reference, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_uniform_likelihood_adjoint_head_uses_configured_accumulator(
    accumulator_dtype,
):
    root = torch.linspace(-308.0, -295.0, 1331, dtype=torch.float32).reshape(1, -1)
    actual_seed = _likelihood_root_seed(
        root,
        None,
        accumulator_dtype=accumulator_dtype,
    )
    root_acc = root.to(dtype=accumulator_dtype)
    reference_seed = -torch.exp2(
        root_acc - logsumexp2(root_acc, dim=-1, keepdim=True)
    ).float()
    torch.testing.assert_close(actual_seed, reference_seed, rtol=0.0, atol=0.0)

    extinction = torch.full((1, 1331), -1e-6, dtype=torch.float32, requires_grad=True)
    actual_survival = _likelihood_log2_survival(
        extinction,
        None,
        accumulator_dtype=accumulator_dtype,
    )
    (actual_grad,) = torch.autograd.grad(actual_survival.sum(), extinction)

    extinction_acc = extinction.detach().to(dtype=accumulator_dtype).requires_grad_(True)
    reference_survival = _likelihood_log2_survival(extinction_acc, None)
    (reference_grad,) = torch.autograd.grad(reference_survival.sum(), extinction_acc)
    torch.testing.assert_close(actual_grad, reference_grad.float(), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("accumulator_dtype", [torch.float32, torch.float64])
def test_ggn_fisher_head_uses_configured_accumulator(
    monkeypatch,
    accumulator_dtype,
):
    root = torch.tensor(
        [[-300.125, -299.75, -301.5]],
        dtype=torch.float32,
    )
    tangent = torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float32)
    static = SimpleNamespace(
        species_helpers={"S": 3},
        wave_layout={"root_clade_ids": torch.tensor([0])},
        accumulator_dtype=accumulator_dtype,
    )
    saved = {"pi_wave": root}
    captured = {}

    monkeypatch.setattr(ggn, "jvp_root_scores", lambda *_args, **_kwargs: tangent)

    def fake_vjp(*args, **_kwargs):
        captured["seed"] = args[2]
        return torch.zeros((3, 3), dtype=torch.float32), None

    monkeypatch.setattr(ggn, "vjp_root_to_theta", fake_vjp)
    hvp = ggn.make_ggn_hvp(
        static,
        torch.zeros((3, 3), dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
        saved,
    )
    hvp(torch.zeros(9, dtype=torch.float32))

    root_acc = root.to(dtype=accumulator_dtype)
    tangent_acc = tangent.to(dtype=accumulator_dtype)
    q = torch.exp2(root_acc - logsumexp2(root_acc, dim=-1, keepdim=True))
    expected = LN2 * q * (
        tangent_acc - (q * tangent_acc).sum(dim=-1, keepdim=True)
    )
    assert captured["seed"].dtype == accumulator_dtype
    torch.testing.assert_close(captured["seed"], expected, rtol=0.0, atol=0.0)


def test_gradient_is_finite_and_matches_fp64():
    """The gradient d log2(survival)/dE at ordinary small survival must be moderate, not ~1e36/NaN."""
    E = _E_for_survival(1e-8, S=64, seed=1)
    E64 = E.clone().requires_grad_(True)
    log2_survival(E64).backward()
    ref_grad = E64.grad.clone()
    E32 = E.to(torch.float32).requires_grad_(True)
    log2_survival(E32).backward()
    assert torch.isfinite(E32.grad).all()
    assert float(E32.grad.abs().max()) < 1e9  # true grad ~1e6, NOT the fabricated ~1e36
    torch.testing.assert_close(E32.grad.double(), ref_grad, rtol=2e-3, atol=1e-2)


def test_receiver_normalizer_no_cancellation():
    """The transfer receiver normalizer (``1 - ancestor_mass``) is the same cancellation and must
    be computed as a sum over non-excluded recipients, not ``1 - excluded_mass``."""
    from gpurec.core.parameters.extract_parameters import receiver_valid_log_normalizer
    parent = torch.tensor([1, 2, 3, 4, 5, 6, 7, -1], dtype=torch.long)  # chain, root = node 7
    # receiver weight concentrated on nodes 3..7 (= node 3's self+ancestors) -> tiny valid mass for donor 3
    logits = torch.tensor([0., 0., 0., 26., 26., 26., 26., 26.], dtype=torch.float64)
    rlp64 = torch.log_softmax(logits * LN2, dim=0) / LN2  # log2 receiver probs (sum to 1)
    depth, node = 8, 3
    ref = receiver_valid_log_normalizer(rlp64, parent, depth)
    got = receiver_valid_log_normalizer(rlp64.to(torch.float32), parent, depth)
    assert torch.isfinite(got[node])
    assert float(ref[node]) > 20.0  # valid_mass ~9e-9 -> normalizer ~26.7 (the cancellation regime)
    assert abs(float(got[node]) - float(ref[node])) <= 1e-3 * abs(float(ref[node])), (float(got[node]), float(ref[node]))
    # old form: 1 - excluded_mass collapses in fp32 (excluded mass rounds to 1) -> clamp -> ~126 bits
    rp32 = torch.exp2(rlp64.to(torch.float32))
    excluded_mass = rp32[torch.tensor([3, 4, 5, 6, 7])].sum()
    naive = float(-torch.log2((1 - excluded_mass).clamp_min(torch.finfo(torch.float32).tiny)))
    assert naive > 100.0  # collapsed vs the true ~26.7 -> documents the cancellation


def test_naive_formula_cancels_but_fix_recovers():
    """Regression guard: the old formula destroys an ordinary survival ~1e-9; the fix recovers it."""
    S = 64
    E = torch.full((S,), math.log2(1.0 - 1e-9), dtype=torch.float32)  # extinction 1-1e-9, survival ~1e-9
    ref = float(_ref_survival(E.double()))
    naive = float((1 - torch.exp2(E).mean()).clamp_min(torch.finfo(torch.float32).tiny))
    assert naive <= 1e-3 * ref  # naive collapses (cancellation) -> off by many orders of magnitude
    got = float(survival_from_E(E))
    assert abs(got - ref) <= 1e-3 * ref  # fix recovers the true value
