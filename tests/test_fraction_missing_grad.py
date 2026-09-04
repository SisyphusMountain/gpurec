import torch, pytest
pytestmark = pytest.mark.gpu
from gpurec.api.model import GeneReconModel


def _build(tmp_path, mode, fm):
    sp = tmp_path / "s.nwk"; sp.write_text("((A:1,B:1)AB:1,C:1)Root;")
    g = tmp_path / "g.nwk"; g.write_text("((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;")
    return GeneReconModel(str(sp), [str(g)], mode=mode, device="cuda",
                          dtype=torch.float64, fraction_missing=fm)


@pytest.mark.parametrize("mode", ["global", "specieswise"])
def test_grad_matches_central_fd_with_fraction_missing(tmp_path, mode):
    m = _build(tmp_path, mode, 0.3)
    loss = m(); loss.backward()
    g = m.theta.grad.detach().clone()
    eps = 1e-5
    max_rel = 0.0
    with torch.no_grad():
        flat = m.theta.view(-1)
        for i in range(flat.numel()):
            base = flat[i].item()
            flat[i] = base + eps; fp = float(m().item())
            flat[i] = base - eps; fmv = float(m().item())
            flat[i] = base
            fd = (fp - fmv) / (2 * eps)
            ga = float(g.view(-1)[i])
            rel = abs(fd - ga) / max(1.0, abs(fd))
            max_rel = max(max_rel, rel)
            assert rel < 1e-3, (
                f"mode={mode} i={i}: analytic={ga:.8e} fd={fd:.8e} rel={rel:.3e}"
            )
    print(f"mode={mode} max_rel={max_rel:.3e}")


@pytest.mark.parametrize("mode", ["global", "specieswise", "genewise"])
def test_streaming_grad_matches_central_fd_with_fraction_missing(tmp_path, mode):
    """The batch-evaluation entry point must carry fraction_missing into the E-adjoint too.

    ``GeneReconModel.forward()`` + ``.backward()`` (the test above) goes through
    ``gpurec/api/_autograd.py``, which passes ``leaf_fm_log`` to the implicit-gradient call.
    Every optimizer and benchmark instead goes through ``gpurec/api/_execution.py``
    (``evaluate_static_loss_grad`` / ``stream_batches`` / ``stream_genewise_loss_vector_grad``),
    which used to omit it. The forward extinction fixed point was then solved WITH the missing
    fractions while the extinction-adjoint linear system was built from the E-step WITHOUT them:
    the wrong linear system, giving a silently wrong gradient (the loss-rate component came out
    -0.937 where the true value is -0.041 on one Coleman family at fraction_missing 0.5) and, when
    that wrong system stopped being a contraction, the "E-adjoint Neumann series failed to
    converge" error.

    A non-default theta and non-uniform receiver weights are used because the model's default
    theta on this fixture sits at a stationary point where every gradient is ~1e-10 and any
    comparison is vacuous.
    """
    from gpurec.api._execution import evaluate_static_loss_grad

    model = _build(tmp_path, mode, 0.5)
    static = model.batch_statics[0]
    species_count = int(model.species_helpers["S"])
    theta = torch.full_like(model.theta.detach(), -1.5)
    theta[..., 2] = 0.5
    receiver_weights = torch.linspace(
        -0.5, 0.5, species_count, device="cuda", dtype=torch.float64)
    origination_weights = torch.zeros(species_count, device="cuda", dtype=torch.float64)

    static.warm_E = None
    _, grad_theta, _, _ = evaluate_static_loss_grad(
        static, theta, receiver_weights, origination_weights, need_grad=True)
    analytic = grad_theta.reshape(-1)
    assert torch.isfinite(analytic).all()
    assert float(analytic.abs().max()) > 1e-3, "degenerate probe: the gradient is ~0 everywhere"

    step = 1e-5
    flat = theta.reshape(-1)
    worst = 0.0
    for index in range(flat.numel()):
        shifted = flat.clone()
        shifted[index] = flat[index] + step
        static.warm_E = None
        up = float(evaluate_static_loss_grad(
            static, shifted.view_as(theta), receiver_weights, origination_weights,
            need_grad=False)[0])
        shifted[index] = flat[index] - step
        static.warm_E = None
        down = float(evaluate_static_loss_grad(
            static, shifted.view_as(theta), receiver_weights, origination_weights,
            need_grad=False)[0])
        finite = (up - down) / (2 * step)
        relative = abs(finite - float(analytic[index])) / max(1.0, abs(finite))
        worst = max(worst, relative)
        assert relative < 1e-5, (
            f"mode={mode} component {index}: analytic={float(analytic[index]):.8e} "
            f"finite differences={finite:.8e} relative gap={relative:.3e}"
        )
    print(f"mode={mode} largest relative gap {worst:.3e}")
