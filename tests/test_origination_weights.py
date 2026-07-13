"""Optimizable origination probabilities: correctness (FD gradcheck), gauge, and descent.

Origination weights are a per-species softmax distribution over the candidate origination branches.
They enter ONLY the final NLL aggregation (not the fixed-point solve or kernels), so:
  * the default (uniform) path is byte-identical to the legacy uniform-origination likelihood, and
  * their gradient is exact and cheap.
These tests verify fp64 finite differences and fp32-to-fp64 head alignment.
"""
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.api._execution import _origination_log_probs
from gpurec.core.inference.solver import (
    nll_vector_from_root_rows,
    origination_grad_from_root_rows,
)
from gpurec.core.parameters.extract_parameters import origination_log_probs_from_weights
from gpurec.core.scheduling import batching
from gpurec.fit.optimize import first_order

DT = torch.float64


def _require_cuda_triton() -> torch.device:
    pytest.importorskip("triton")
    try:
        batching._load_native_module()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gpurec-preprocess extension is unavailable: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        torch.empty(1, device="cuda")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return torch.device("cuda")


def _tiny_model(tmp_path: Path, device):
    sp = tmp_path / "species.nwk"
    gt = tmp_path / "family_0.nwk"
    sp.write_text("((A:1,B:1)AB:1,C:1)Root:1;\n", encoding="utf-8")
    gt.write_text("((A_1:1,B_1:1)g:1,C_1:1)GeneRoot:1;\n", encoding="utf-8")
    so = SolverOptions(pi_iters=160, neumann_terms=96, e_max_iter=400, e_tol=1e-14)
    return GeneReconModel(
        sp, [gt], mode="specieswise", device=str(device), family_chunk_size=1,
        clade_budget=None, batch_packing="sequential", max_wave_size=4, solver_options=so,
    )


def test_weighted_nll_reduces_to_uniform():
    """The weighted aggregation is exactly the uniform one when weights are equal."""
    torch.manual_seed(0)
    s, nfam = 7, 5
    rr = torch.randn(nfam, s, dtype=DT)
    e = -torch.rand(nfam, s, dtype=DT) * 3.0
    uniform = nll_vector_from_root_rows(rr, e)
    olp = origination_log_probs_from_weights(torch.zeros(s, dtype=DT))
    weighted = nll_vector_from_root_rows(rr, e, origination_log_probs=olp, origination_probs=torch.exp2(olp))
    assert float((uniform - weighted).abs().max()) < 1e-12


def test_fp32_origination_gradient_matches_promoted_head():
    """The direct gradient differentiates the same fp64 head used by fp32 forward."""
    torch.manual_seed(7)
    root = torch.randn(3, 11, dtype=torch.float32) - 20.0
    extinction = -torch.rand(3, 11, dtype=torch.float32) * 3.0
    weights = torch.randn(11, dtype=torch.float32)

    promoted = weights.double().requires_grad_(True)
    log_probs = origination_log_probs_from_weights(promoted)
    loss = nll_vector_from_root_rows(
        root,
        extinction,
        origination_log_probs=log_probs,
        origination_probs=torch.exp2(log_probs),
    ).sum()
    (reference,) = torch.autograd.grad(loss, promoted)

    actual = origination_grad_from_root_rows(root, extinction, weights)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference.float(), rtol=0.0, atol=0.0)


def test_fp32_execution_promotes_raw_origination_weights_for_head():
    weights = torch.linspace(-2.0, 1.0, 17, dtype=torch.float32)
    log_probs, probabilities = _origination_log_probs(
        weights,
        like=torch.empty((), dtype=torch.float32),
    )
    reference = origination_log_probs_from_weights(weights.double())

    assert log_probs.dtype == torch.float64
    assert probabilities.dtype == torch.float64
    torch.testing.assert_close(log_probs, reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(probabilities, torch.exp2(reference), rtol=0.0, atol=0.0)


def test_origination_gradients_match_finite_differences(tmp_path: Path):
    device = _require_cuda_triton()
    torch.manual_seed(0)
    model = _tiny_model(tmp_path, device)
    s = int(model.species_helpers["S"])
    theta = 0.3 * torch.randn(s, 3, dtype=DT, device=device)
    recv = torch.zeros(s, dtype=DT, device=device)
    ow = 0.7 * torch.randn(s, dtype=DT, device=device)
    ow = ow - ow.mean()

    def loss_at(t, o):
        with torch.no_grad():
            model.clear_warm_starts()
            return float(model(t.contiguous(), recv, o.contiguous()))

    # default (self.origination_weights == zeros) is byte-identical to explicit uniform origination
    with torch.no_grad():
        model.clear_warm_starts()
        l_default = float(model(theta, recv))
    l_uniform = loss_at(theta, torch.zeros(s, dtype=DT, device=device))
    assert abs(l_default - l_uniform) < 1e-9

    tp = theta.clone().requires_grad_(True)
    op = ow.clone().requires_grad_(True)
    model.clear_warm_starts()
    model(tp, recv, op).backward()
    g_orig, g_theta = op.grad.detach(), tp.grad.detach()

    eps = 1e-6
    fd_orig = torch.zeros_like(ow)
    for i in range(s):
        d = torch.zeros_like(ow); d[i] = eps
        fd_orig[i] = (loss_at(theta, ow + d) - loss_at(theta, ow - d)) / (2 * eps)
    assert float((g_orig - fd_orig).abs().max()) < 1e-6

    # theta gradient under NON-uniform origination exercises the weighted backward seeds
    fd_theta = torch.zeros_like(theta)
    for i in range(s):
        for j in range(3):
            d = torch.zeros_like(theta); d[i, j] = eps
            fd_theta[i, j] = (loss_at(theta + d, ow) - loss_at(theta - d, ow)) / (2 * eps)
    assert float((g_theta - fd_theta).abs().max()) < 1e-5

    # softmax gauge: gradient along the all-ones direction is zero
    assert abs(float(g_orig.sum())) < 1e-9


def test_first_order_optimizes_origination(tmp_path: Path):
    device = _require_cuda_triton()
    torch.manual_seed(0)
    model = _tiny_model(tmp_path, device)
    s = int(model.species_helpers["S"])
    theta0 = (0.3 * torch.randn(s, 3, device=device)).float()
    recv = torch.zeros(s, device=device)

    model.clear_warm_starts()
    l_init = float(model(theta0, recv, torch.zeros(s, device=device)))
    (theta_out, orig_out), _hist, _warm = first_order(
        model.batch_statics, theta0, recv, with_origination=True,
        optimizer="adam", lr0=0.2, max_steps=60, verbose=False, early_stop=False,
    )
    model.clear_warm_starts()
    l_final = float(model(theta_out, recv, orig_out))
    assert l_final < l_init - 1e-3            # origination + theta jointly reduce the NLL
    assert abs(float(orig_out.mean())) < 1e-4  # gauge stays mean-zero
