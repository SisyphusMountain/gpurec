"""Specieswise uniform forward/backward optimization parity tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel
from gpurec.api.autograd import _extract_parameters
from gpurec.core.model import GeneDataset
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.likelihood import (
    E_fixed_point,
    compute_log_likelihood,
    compute_log_likelihood_root_rows,
)
from gpurec.core.species import species_child_arrays_from_helpers


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


def _alerax_label_by_species_index(species_helpers: dict) -> list[str]:
    """Replicate AleRax internal-node labels in gpurec species order."""
    S = int(species_helpers["S"])
    names = list(species_helpers["names"])
    child1, child2 = species_child_arrays_from_helpers(species_helpers, S=S)
    parent = [-1] * S
    for p, (c1, c2) in enumerate(zip(child1, child2)):
        if c1 < S:
            parent[c1] = p
        if c2 < S:
            parent[c2] = p
    root = parent.index(-1)

    post_order: list[int] = []

    def visit(idx: int) -> None:
        if child1[idx] < S:
            visit(child1[idx])
        if child2[idx] < S:
            visit(child2[idx])
        post_order.append(idx)

    def is_numeric(text: str) -> bool:
        if not text:
            return False
        try:
            float(text)
            return True
        except ValueError:
            return False

    def unique_label(seen: set[str], base: str) -> str:
        suffix = 0
        while True:
            label = f"{base}_{suffix}"
            if label not in seen:
                return label
            suffix += 1

    visit(root)
    seen: set[str] = set()
    any_leaf_label = [""] * S
    labels = [""] * S
    for idx in post_order:
        name = names[idx]
        if child1[idx] >= S:
            label = name
            any_leaf_label[idx] = name
        else:
            any_leaf_label[idx] = any_leaf_label[child1[idx]]
            label = name or ""
            if (not label) or label in seen or is_numeric(label):
                base = f"Node_{any_leaf_label[child1[idx]]}_{any_leaf_label[child2[idx]]}"
                label = unique_label(seen, base)
        labels[idx] = label
        seen.add(label)
    return labels


def _load_alerax_specieswise_theta(
    data_dir: Path,
    species_helpers: dict,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    params_path = data_dir / "output_specieswise" / "model_parameters" / "model_parameters.txt"
    labels = _alerax_label_by_species_index(species_helpers)
    index_by_label = {label: idx for idx, label in enumerate(labels)}
    rates = torch.empty((len(labels), 3), dtype=dtype, device=device)
    seen: set[str] = set()
    for raw in params_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("node"):
            continue
        label, d_rate, l_rate, t_rate = line.split()[:4]
        idx = index_by_label[label]
        rates[idx] = torch.tensor(
            [float(d_rate), float(l_rate), float(t_rate)],
            dtype=dtype,
            device=device,
        )
        seen.add(label)
    assert seen == set(labels)
    return torch.log2(rates)


def _load_alerax_per_family_likelihoods(data_dir: Path) -> dict[int, float]:
    output_dir = data_dir / "output_specieswise"
    path = output_dir / "per_fam_likelihoods.txt"
    if not path.exists():
        path = output_dir / "per_fam_likelihood.txt"
    result: dict[int, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        family_name, value = line.split()[:2]
        family_idx = int(family_name.split("_")[1])
        result[family_idx] = float(value)
    return result


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
        device=device,
        dtype=dtype,
        theta_init_rates=(0.05, 0.05, 0.05),
        fixed_iters_Pi=4,
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
        ancestors_T=static.ancestors_T,
    )
    return E_out, params


def _run_forward(
    model: GeneReconModel,
    E_out: dict,
    params: tuple,
    *,
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
        max_transfer_mat=max_transfer_vec,
        device=static.device,
        dtype=static.dtype,
        fixed_iters=static.fixed_iters_Pi,
        return_original=False,
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
        "GPUREC_SELF_LOOP_2D_TRITON": "0",
    }
    for key, value in flags.items():
        monkeypatch.setenv(key, value)


def _set_optimized_env(monkeypatch: pytest.MonkeyPatch) -> None:
    flags = {
        "GPUREC_SELF_LOOP_2D_TRITON": "auto",
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
    ref_nll, ref_out = _run_forward(model, E_out, params, root_rows=False)
    torch.cuda.synchronize()

    _set_optimized_env(monkeypatch)
    opt_nll, opt_out = _run_forward(model, E_out, params, root_rows=False)
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_specieswise_uniform_matches_alerax_specieswise_reference(
    data_dir_100,
    monkeypatch,
    tmp_path,
):
    """Specieswise uniform likelihood matches the AleRax PER-SPECIES fixture."""
    device = torch.device("cuda")
    dtype = torch.float64
    genes = _genes(data_dir_100, 100)
    ds = GeneDataset(
        str(data_dir_100 / "sp.nwk"),
        genes,
        genewise=False,
        specieswise=True,
        dtype=dtype,
        device=device,
        preprocess_cache_dir=tmp_path / "preprocess",
    )
    theta = _load_alerax_specieswise_theta(
        data_dir_100,
        ds.species_helpers,
        dtype=dtype,
        device=device,
    )
    for family in ds.families:
        family["theta"] = theta.clone()

    _set_optimized_env(monkeypatch)
    gpurec_nll_bits = ds.compute_likelihood_batch(
        indices=list(range(100)),
        max_iters_E=4000,
        tol_E=1e-10,
        device=device,
        dtype=dtype,
        chunk_size=100,
    )
    torch.cuda.synchronize()

    alerax_log_likelihood_nats = _load_alerax_per_family_likelihoods(data_dir_100)
    assert set(alerax_log_likelihood_nats) == set(range(100))

    gpurec_log_likelihood_nats = torch.tensor(
        [-float(nll) * math.log(2.0) for nll in gpurec_nll_bits],
        dtype=torch.float64,
    )
    reference = torch.tensor(
        [alerax_log_likelihood_nats[i] for i in range(100)],
        dtype=torch.float64,
    )
    max_abs = (gpurec_log_likelihood_nats - reference).abs().max()
    assert max_abs < 5e-3, f"max specieswise AleRax likelihood diff: {max_abs.item():.6g}"
