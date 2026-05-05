"""Integration tests for production global-rate optimization."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel
from gpurec.optimization import optimize_global_rates_lbfgs


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data" / "test_trees_100"
MIN_RATE = 1e-10
INTERIOR_INIT_RATES = (0.05, 0.05, 0.05)
NLL_ATOL_BITS = 5e-2
RATE_RTOL = 1e-2


def _gene_paths(root: Path) -> list[str]:
    paths = sorted(root.glob("g_*.nwk"))
    if not paths:
        pytest.skip(f"no gene trees found in {root}")
    if len(paths) != 100:
        pytest.skip(f"expected 100 gene trees in {root}, got {len(paths)}")
    return [str(p) for p in paths]


def _reference_rates(root: Path) -> torch.Tensor:
    path = root / "output_global" / "model_parameters" / "model_parameters.txt"
    if not path.exists():
        pytest.skip(f"missing AleRax model parameters: {path}")
    rates = []
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4:
            rates.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not rates:
        raise AssertionError(f"could not parse rates from {path}")
    first = rates[0]
    for row in rates[1:]:
        assert row == first, f"expected global rates, got {first} and {row}"
    return torch.tensor(first, dtype=torch.float64)


def _reference_nll_bits(root: Path) -> float:
    path = root / "output_global" / "per_fam_likelihoods.txt"
    if not path.exists():
        pytest.skip(f"missing AleRax per-family likelihoods: {path}")
    total_ll_nats = 0.0
    rows = 0
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            total_ll_nats += float(parts[1])
            rows += 1
    if rows != 100:
        raise AssertionError(f"expected 100 per-family likelihoods, got {rows}")
    return -total_ll_nats / math.log(2.0)


def _make_model(root: Path, *, init_rate: float, cache_dir: Path) -> GeneReconModel:
    if not root.exists():
        pytest.skip(f"dataset not present: {root}")
    return GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=_gene_paths(root),
        mode="global",
        pibar_mode="uniform",
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(init_rate, init_rate, init_rate),
        preprocess_cache_dir=str(cache_dir),
        fixed_iters_Pi=6,
        max_wave_size=32768,
        neumann_terms=3,
        use_pruning=True,
        pruning_threshold=1e-6,
    )


def test_global_lbfgs_reaches_alerax_reference(tmp_path):
    target_rates = _reference_rates(DATA_DIR)
    target_nll = _reference_nll_bits(DATA_DIR)
    model = _make_model(
        DATA_DIR,
        init_rate=INTERIOR_INIT_RATES[0],
        cache_dir=tmp_path / "cache",
    )

    result = optimize_global_rates_lbfgs(
        model,
        min_rate=MIN_RATE,
        interior_init_rates=INTERIOR_INIT_RATES,
        steps=12,
        max_eval=60,
        tolerance_grad=1e-3,
        tolerance_change=1e-7,
    )

    rates = result["rates"].detach().to(device="cpu", dtype=torch.float64)
    target = target_rates.to(dtype=torch.float64)
    max_rel_err = torch.max(
        torch.abs(rates - target) / torch.clamp(target.abs(), min=1e-30)
    ).item()
    nll_gap = result["negative_log_likelihood"] - target_nll

    print(
        "global_lbfgs",
        f"evals={result['timing']['evaluations']}",
        f"elapsed_s={result['timing']['total_s']:.3f}",
        f"nll={result['negative_log_likelihood']:.6f}",
        f"nll_gap_bits={nll_gap:.6f}",
        f"rate_rel_err={max_rel_err:.6e}",
        f"initialization={result['initialization']}",
    )

    assert result["initialization"] == "used_model_theta"
    assert result["timing"]["evaluations"] <= 20
    assert math.isfinite(result["negative_log_likelihood"])
    assert abs(nll_gap) <= NLL_ATOL_BITS
    assert max_rel_err <= RATE_RTOL


def test_global_lbfgs_moves_floor_initialization_into_interior(tmp_path):
    model = _make_model(DATA_DIR, init_rate=MIN_RATE, cache_dir=tmp_path / "cache")

    result = optimize_global_rates_lbfgs(
        model,
        min_rate=MIN_RATE,
        interior_init_rates=INTERIOR_INIT_RATES,
        steps=1,
        max_eval=2,
    )

    assert result["initialization"] == "overrode_model_floor_init"
    assert result["history"]
    assert all(math.isfinite(record["nll"]) for record in result["history"])
    assert result["history"][0]["rates"].tolist() == pytest.approx(
        INTERIOR_INIT_RATES
    )
    assert torch.all(result["rates"].detach().cpu() > MIN_RATE)
