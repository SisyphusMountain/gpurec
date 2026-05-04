"""Specieswise uniform forward/backward optimization parity tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel
from gpurec.api.autograd import _extract_parameters
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import (
    E_fixed_point,
    compute_log_likelihood,
    compute_log_likelihood_root_rows,
)


_ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-3


@pytest.fixture(scope="module")
def data_dir_100():
    d = _ROOT / "data" / "test_trees_100"
    if not d.exists():
        pytest.skip("test_trees_100 not found")
    return d


@pytest.fixture(scope="module")
def data_dir_1000():
    d = _ROOT / "data" / "test_trees_1000"
    if not d.exists():
        pytest.skip("test_trees_1000 not found")
    return d


def _genes(data_dir: Path, n: int) -> list[str]:
    genes = sorted(data_dir.glob("g_*.nwk"))[:n]
    if len(genes) < n:
        pytest.skip(f"need {n} gene trees in {data_dir}")
    return [str(g) for g in genes]


def _specieswise_theta(S: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    base = torch.log2(torch.tensor([0.05, 0.05, 0.05], dtype=dtype, device=device))
    offsets = torch.linspace(-0.08, 0.08, S, dtype=dtype, device=device)
    theta = base.unsqueeze(0).expand(S, 3).clone()
    theta[:, 0] += offsets
    theta[:, 1] -= 0.5 * offsets
    theta[:, 2] += 0.25 * torch.sin(offsets * math.pi)
    return theta


def _constant_specieswise_theta(
    S: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.log2(torch.tensor([0.04, 0.06, 0.05], dtype=dtype, device=device))
    return theta, theta.unsqueeze(0).expand(S, -1).clone()


def _make_model(
    data_dir: Path,
    genes: list[str],
    *,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    use_pruning: bool = False,
) -> GeneReconModel:
    return GeneReconModel.from_trees(
        str(data_dir / "sp.nwk"),
        genes,
        mode=mode,
        pibar_mode="uniform",
        device=device,
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_Pi=3,
        max_iters_Pi=500,
        tol_Pi=TOL,
        max_iters_E=500,
        tol_E=TOL,
        neumann_terms=2,
        use_pruning=use_pruning,
        preprocess_cache_dir="/tmp/gpurec_preprocess_cache",
    )


def _prepare_forward(model: GeneReconModel):
    static = model.static
    params = _extract_parameters(model.theta.detach(), static)
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = params
    E_out = E_fixed_point(
        species_helpers=static.species_helpers,
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=transfer_mat,
        max_transfer_mat=max_transfer_vec,
        max_iters=static.max_iters_E,
        tolerance=static.tol_E,
        warm_start_E=None,
        dtype=static.dtype,
        device=static.device,
        pibar_mode=static.pibar_mode,
        ancestors_T=static.ancestors_T,
    )
    return E_out, params


def _run_forward(
    model: GeneReconModel,
    E_out: dict,
    params: tuple,
    *,
    need_pibar: bool = True,
    root_rows: bool = False,
):
    static = model.static
    log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = params
    pi_out = Pi_wave_forward(
        wave_layout=static.wave_layout,
        species_helpers=static.species_helpers,
        E=E_out["E"],
        Ebar=E_out["E_bar"],
        E_s1=E_out["E_s1"],
        E_s2=E_out["E_s2"],
        log_pS=log_pS,
        log_pD=log_pD,
        log_pL=log_pL,
        transfer_mat=transfer_mat,
        max_transfer_mat=max_transfer_vec,
        device=static.device,
        dtype=static.dtype,
        local_iters=static.max_iters_Pi,
        local_tolerance=static.tol_Pi,
        fixed_iters=static.fixed_iters_Pi,
        pibar_mode=static.pibar_mode,
        return_original=False,
        need_pibar=need_pibar,
        return_root_rows=root_rows,
        family_idx=None,
    )
    if root_rows:
        nll = compute_log_likelihood_root_rows(pi_out["Pi_root_rows"], E_out["E"]).sum()
    else:
        nll = compute_log_likelihood(
            pi_out["Pi_wave_ordered"],
            E_out["E"],
            static.wave_layout["root_clade_ids"],
        ).sum()
    return nll, pi_out


def _set_reference_env(monkeypatch: pytest.MonkeyPatch) -> None:
    flags = {
        "GPUREC_FORWARD_LEAF_INDEX": "0",
        "GPUREC_FORWARD_PARENT_REDUCED_DTS": "0",
        "GPUREC_FORWARD_TOPOLOGY_INT32": "0",
        "GPUREC_FUSED_UNIFORM_BACKWARD": "0",
        "GPUREC_KERNELIZED_BACKWARD_DTS": "0",
        "GPUREC_BACKWARD_PARENT_REDUCED_DTS": "0",
        "GPUREC_FUSED_CROSS_PIBAR_VJP": "0",
        "GPUREC_BACKWARD_LEAF_INDEX": "0",
        "GPUREC_DTS_PIBAR_UD_FUSION": "0",
    }
    for key, value in flags.items():
        monkeypatch.setenv(key, value)


def _set_optimized_env(monkeypatch: pytest.MonkeyPatch) -> None:
    flags = {
        "GPUREC_UNIFORM_PINGPONG": "1",
        "GPUREC_FORWARD_LEAF_INDEX": "1",
        "GPUREC_FORWARD_PARENT_REDUCED_DTS": "1",
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_IMPL": "tiled",
        "GPUREC_FORWARD_PARENT_REDUCED_DTS_GE2_ONLY": "1",
        "GPUREC_FORWARD_TOPOLOGY_INT32": "1",
        "GPUREC_FUSED_UNIFORM_BACKWARD": "1",
        "GPUREC_KERNELIZED_BACKWARD_DTS": "1",
        "GPUREC_BACKWARD_PARENT_REDUCED_DTS": "tiled",
        "GPUREC_BACKWARD_PARENT_REDUCED_DTS_MIN_SPLITS": "0",
        "GPUREC_FUSED_CROSS_PIBAR_VJP": "1",
        "GPUREC_BACKWARD_LEAF_INDEX": "1",
        "GPUREC_DTS_PIBAR_UD_FUSION": "1",
    }
    for key, value in flags.items():
        monkeypatch.setenv(key, value)


def _loss_and_grad(
    model: GeneReconModel,
    set_env,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    set_env(monkeypatch)
    model.static.warm_E = None
    model.zero_grad(set_to_none=True)
    loss = model()
    loss.backward()
    torch.cuda.synchronize()
    assert model.theta.grad is not None
    return loss.detach().clone(), model.theta.grad.detach().clone()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_specieswise_uniform_forward_optimized_matches_reference(data_dir_100, monkeypatch):
    """Optimized specieswise uniform Pi/Pibar matches the reference toggles."""
    device = torch.device("cuda")
    dtype = torch.float32
    genes = _genes(data_dir_100, 3)
    model = _make_model(data_dir_100, genes, mode="specieswise", dtype=dtype, device=device)
    with torch.no_grad():
        model.theta.copy_(_specieswise_theta(model.n_species, dtype=dtype, device=device))

    E_out, params = _prepare_forward(model)

    _set_reference_env(monkeypatch)
    ref_nll, ref_out = _run_forward(model, E_out, params, need_pibar=True, root_rows=False)
    torch.cuda.synchronize()

    _set_optimized_env(monkeypatch)
    opt_nll, opt_out = _run_forward(model, E_out, params, need_pibar=True, root_rows=False)
    torch.cuda.synchronize()

    assert torch.allclose(opt_nll, ref_nll, atol=1e-4, rtol=1e-5)
    assert torch.allclose(
        opt_out["Pi_wave_ordered"],
        ref_out["Pi_wave_ordered"],
        atol=1e-4,
        rtol=1e-5,
    )
    assert ref_out["Pibar_wave_ordered"] is not None
    assert opt_out["Pibar_wave_ordered"] is not None
    assert torch.allclose(
        opt_out["Pibar_wave_ordered"],
        ref_out["Pibar_wave_ordered"],
        atol=1e-4,
        rtol=1e-5,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_specieswise_uniform_backward_optimized_matches_reference(data_dir_100, monkeypatch):
    """Optimized specieswise uniform backward matches the generic/reference path."""
    device = torch.device("cuda")
    dtype = torch.float32
    genes = _genes(data_dir_100, 2)
    ref_model = _make_model(data_dir_100, genes, mode="specieswise", dtype=dtype, device=device)
    opt_model = _make_model(data_dir_100, genes, mode="specieswise", dtype=dtype, device=device)
    theta = _specieswise_theta(ref_model.n_species, dtype=dtype, device=device)
    with torch.no_grad():
        ref_model.theta.copy_(theta)
        opt_model.theta.copy_(theta)

    ref_loss, ref_grad = _loss_and_grad(ref_model, _set_reference_env, monkeypatch)
    opt_loss, opt_grad = _loss_and_grad(opt_model, _set_optimized_env, monkeypatch)

    assert torch.allclose(opt_loss, ref_loss, atol=2e-3, rtol=1e-5)
    assert torch.allclose(opt_grad, ref_grad, atol=3e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_constant_specieswise_matches_global_loss_and_gradient_semantics(
    data_dir_100, monkeypatch
):
    """Constant specieswise rates have global-mode loss and summed gradients."""
    device = torch.device("cuda")
    dtype = torch.float32
    genes = _genes(data_dir_100, 2)
    global_model = _make_model(data_dir_100, genes, mode="global", dtype=dtype, device=device)
    species_model = _make_model(data_dir_100, genes, mode="specieswise", dtype=dtype, device=device)
    theta_global, theta_species = _constant_specieswise_theta(
        species_model.n_species, dtype=dtype, device=device
    )
    with torch.no_grad():
        global_model.theta.copy_(theta_global)
        species_model.theta.copy_(theta_species)

    global_loss, global_grad = _loss_and_grad(global_model, _set_optimized_env, monkeypatch)
    species_loss, species_grad = _loss_and_grad(species_model, _set_optimized_env, monkeypatch)

    assert torch.allclose(species_loss, global_loss, atol=2e-3, rtol=1e-5)
    assert torch.allclose(species_grad.sum(dim=0), global_grad, atol=5e-2, rtol=2e-2)
