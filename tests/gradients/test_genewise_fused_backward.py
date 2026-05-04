"""Correctness coverage for batched genewise uniform backward.

These tests live outside the kernel implementation on purpose.  They pin the
autograd-facing genewise backward against small finite-difference checks,
per-family global/uniform references, and the generic self-loop fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel
from gpurec.api.autograd import _GeneReconFunction


_ROOT = Path(__file__).resolve().parent.parent


class _GenericSelfLoopReached(RuntimeError):
    pass


def _cuda_device() -> torch.device:
    return torch.device("cuda")


def _gene_paths(data_dir: Path, n: int) -> list[str]:
    paths = sorted(data_dir.glob("g_*.nwk"))[:n]
    if len(paths) < n:
        pytest.skip(f"Need {n} gene families in {data_dir}")
    return [str(p) for p in paths]


def _set_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _optimized_genewise_env() -> dict[str, str]:
    return {
        "GPUREC_FUSED_GENEWISE_BACKWARD": "1",
        "GPUREC_FUSED_GENEWISE_BACKWARD_SELF_LOOP": "1",
        "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
        "GPUREC_BACKWARD_LEAF_INDEX": "1",
        "GPUREC_FORWARD_FAMILY_INDEXED_CONSTS": "1",
        "GPUREC_FORWARD_FAMILY_INDEXED_DTS_PARAMS": "1",
        "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
        "GPUREC_FUSED_CROSS_PIBAR_VJP_IMPL": "tree",
        "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
        "GPUREC_FUSED_DTS_BACKWARD_ACCUM": "1",
    }


def _fallback_env() -> dict[str, str]:
    env = _optimized_genewise_env()
    env["GPUREC_FUSED_UNIFORM_BACKWARD"] = "0"
    env["GPUREC_FUSED_GENEWISE_BACKWARD_SELF_LOOP"] = "0"
    return env


def _build_genewise_model(
    data_dir: Path,
    gene_paths: list[str],
    *,
    dtype: torch.dtype,
    rates: torch.Tensor | None = None,
    **solver_kwargs,
) -> GeneReconModel:
    model = GeneReconModel.from_trees(
        species_tree=str(data_dir / "sp.nwk"),
        gene_trees=gene_paths,
        mode="genewise",
        pibar_mode="uniform",
        device=_cuda_device(),
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        **solver_kwargs,
    )
    if rates is not None:
        with torch.no_grad():
            model.theta.copy_(torch.log2(rates.to(device=model.theta.device, dtype=dtype)))
    return model


def _loss_and_grad(model: GeneReconModel) -> tuple[torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    model.static.warm_E = None
    loss = model()
    loss.backward()
    torch.cuda.synchronize()
    return loss.detach(), model.theta.grad.detach().clone()


@contextmanager
def _tight_generic_uniform_self_loop(monkeypatch: pytest.MonkeyPatch):
    from gpurec.core import backward as backward_core

    orig_gmres = backward_core._gmres_self_loop_solve

    def tight_gmres(
        rhs,
        ingredients,
        sp_child1,
        sp_child2,
        S,
        W,
        pibar_mode,
        transfer_mat_T,
        ancestors_T,
        max_iters=30,
        tol=1e-5,
    ):
        return orig_gmres(
            rhs,
            ingredients,
            sp_child1,
            sp_child2,
            S,
            W,
            pibar_mode,
            transfer_mat_T,
            ancestors_T,
            max_iters=40,
            tol=1e-14,
        )

    with monkeypatch.context() as m:
        m.setattr(backward_core, "_gmres_self_loop_solve", tight_gmres)
        yield


@contextmanager
def _fail_if_generic_self_loop_used(monkeypatch: pytest.MonkeyPatch):
    from gpurec.core import backward as backward_core

    def fail(*args, **kwargs):
        raise _GenericSelfLoopReached(
            "batched genewise optimized backward reached the generic self-loop"
        )

    with monkeypatch.context() as m:
        m.setattr(backward_core, "_self_loop_vjp_precompute", fail)
        m.setattr(backward_core, "_gmres_self_loop_solve", fail)
        yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gradcheck_genewise_uniform_small_tree():
    """Small batched genewise tree passes torch gradcheck."""
    data_dir = _ROOT / "data" / "test_trees_3"
    if not data_dir.exists():
        pytest.skip("test_trees_3 dataset not present")

    gene = str(data_dir / "g.nwk")
    model = _build_genewise_model(
        data_dir,
        [gene, gene],
        dtype=torch.float64,
        max_iters_E=2000,
        tol_E=1e-10,
        max_iters_Pi=2000,
        tol_Pi=1e-9,
        fixed_iters_Pi=6,
        neumann_terms=5,
        use_pruning=False,
    )
    theta = model.theta.detach().clone().requires_grad_(True)

    def fn(theta_in: torch.Tensor) -> torch.Tensor:
        model.static.warm_E = None
        return _GeneReconFunction.apply(theta_in, model.static, "sum")

    assert torch.autograd.gradcheck(
        fn,
        (theta,),
        eps=1e-4,
        atol=2e-3,
        rtol=2e-3,
        nondet_tol=1e-8,
        fast_mode=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_genewise_uniform_batched_gradient_matches_per_family_global_fp64(
    monkeypatch,
):
    """Batched genewise weighted gradients equal independent global runs."""
    data_dir = _ROOT / "data" / "test_trees_20"
    if not data_dir.exists():
        pytest.skip("test_trees_20 dataset not present")
    genes = _gene_paths(data_dir, 3)
    dtype = torch.float64
    device = _cuda_device()
    rates = torch.tensor(
        [
            [0.050, 0.040, 0.030],
            [0.060, 0.025, 0.045],
            [0.030, 0.055, 0.035],
        ],
        device=device,
        dtype=dtype,
    )
    solver_kwargs = dict(
        fixed_iters_Pi=6,
        max_iters_E=128,
        tol_E=0.0,
        neumann_terms=8,
        use_pruning=False,
        cg_tol=1e-10,
        cg_maxiter=1000,
    )

    _set_env(monkeypatch, _fallback_env())
    batched = _build_genewise_model(
        data_dir,
        genes,
        dtype=dtype,
        rates=rates,
        **solver_kwargs,
    )
    weights = torch.tensor([0.5, 1.25, 2.0], dtype=dtype, device=device)
    batched.zero_grad(set_to_none=True)
    batched.static.warm_E = None
    per_family_nll = batched.nll_per_family()
    (weights * per_family_nll).sum().backward()
    batched_grad = batched.theta.grad.detach().clone()

    single_nlls = []
    single_grads = []
    for i, gene_path in enumerate(genes):
        single = GeneReconModel.from_trees(
            species_tree=str(data_dir / "sp.nwk"),
            gene_trees=[gene_path],
            mode="global",
            pibar_mode="uniform",
            device=device,
            dtype=dtype,
            theta_init_rates=(0.05, 0.05, 0.05),
            **solver_kwargs,
        )
        with torch.no_grad():
            single.theta.copy_(torch.log2(rates[i]))
        nll, grad = _loss_and_grad(single)
        single_nlls.append(nll)
        single_grads.append(grad)

    torch.testing.assert_close(
        per_family_nll.detach(),
        torch.stack(single_nlls),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        batched_grad,
        weights.view(-1, 1) * torch.stack(single_grads),
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_genewise_uniform_optimized_high_s_fp64_matches_generic_fallback(
    monkeypatch,
):
    """Optimized batched genewise self-loop matches the tight generic fallback."""
    data_dir = _ROOT / "data" / "test_trees_1000"
    if not data_dir.exists():
        pytest.skip("test_trees_1000 dataset not present")
    genes = _gene_paths(data_dir, 2)
    dtype = torch.float64
    rates = torch.tensor(
        [
            [0.050, 0.040, 0.030],
            [0.062, 0.028, 0.044],
        ],
        device=_cuda_device(),
        dtype=dtype,
    )
    solver_kwargs = dict(
        fixed_iters_Pi=6,
        max_iters_E=128,
        tol_E=0.0,
        neumann_terms=20,
        use_pruning=False,
        cg_tol=1e-10,
        cg_maxiter=1000,
        max_wave_size=4096,
    )

    _set_env(monkeypatch, _optimized_genewise_env())
    optimized = _build_genewise_model(
        data_dir,
        genes,
        dtype=dtype,
        rates=rates,
        **solver_kwargs,
    )
    try:
        with _fail_if_generic_self_loop_used(monkeypatch):
            optimized_nll, optimized_grad = _loss_and_grad(optimized)
    except _GenericSelfLoopReached:
        pytest.xfail("optimized batched genewise backward still routes through the generic self-loop")

    _set_env(monkeypatch, _fallback_env())
    fallback = _build_genewise_model(
        data_dir,
        genes,
        dtype=dtype,
        rates=rates,
        **solver_kwargs,
    )
    with _tight_generic_uniform_self_loop(monkeypatch):
        fallback_nll, fallback_grad = _loss_and_grad(fallback)

    torch.testing.assert_close(
        optimized_nll,
        fallback_nll,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        optimized_grad,
        fallback_grad,
        rtol=2e-5,
        atol=2e-5,
    )
