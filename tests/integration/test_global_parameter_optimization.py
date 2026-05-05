"""Integration tests for production global-rate optimization."""
from __future__ import annotations

import inspect
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
BF16_SMALL_FAMILIES = 3
BF16_NLL_CLOSE_ATOL_BITS = 5.0
BF16_RATE_CLOSE_RTOL = 0.75


def _gene_paths(root: Path, *, limit: int | None = None) -> list[str]:
    paths = sorted(root.glob("g_*.nwk"))
    if not paths:
        pytest.skip(f"no gene trees found in {root}")
    if len(paths) != 100:
        pytest.skip(f"expected 100 gene trees in {root}, got {len(paths)}")
    if limit is not None:
        paths = paths[:limit]
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


def _make_small_model(
    root: Path, *, init_rate: float, cache_dir: Path
) -> GeneReconModel:
    if not root.exists():
        pytest.skip(f"dataset not present: {root}")
    return GeneReconModel.from_trees(
        species_tree=str(root / "sp.nwk"),
        gene_trees=_gene_paths(root, limit=BF16_SMALL_FAMILIES),
        mode="global",
        pibar_mode="uniform",
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(init_rate, init_rate, init_rate),
        preprocess_cache_dir=str(cache_dir),
        fixed_iters_Pi=4,
        max_wave_size=32768,
        neumann_terms=2,
        use_pruning=True,
        pruning_threshold=1e-6,
    )


def _first_present(names: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _bf16_start_kwargs() -> dict[str, object]:
    names = set(inspect.signature(optimize_global_rates_lbfgs).parameters)
    kwargs: dict[str, object] = {}

    start_flag = _first_present(
        names,
        (
            "bf16_start",
            "use_bf16_start",
            "bf16_warmup",
            "mixed_precision_bf16_start",
        ),
    )
    start_dtype = _first_present(names, ("start_dtype", "warmup_dtype"))
    polish_dtype = _first_present(names, ("polish_dtype", "final_dtype"))
    bf16_steps = _first_present(
        names,
        ("bf16_warmup_steps", "bf16_start_steps", "bf16_steps", "warmup_steps"),
    )

    if start_flag is not None:
        kwargs[start_flag] = True
    elif start_dtype is not None:
        kwargs[start_dtype] = torch.bfloat16
    elif bf16_steps is None:
        pytest.skip("optimize_global_rates_lbfgs does not expose a bf16-start path")

    if polish_dtype is not None:
        kwargs[polish_dtype] = torch.float32
    fp32_polish = _first_present(names, ("fp32_polish", "use_fp32_polish"))
    if fp32_polish is not None:
        kwargs[fp32_polish] = True

    if bf16_steps is not None:
        kwargs[bf16_steps] = 1
    bf16_max_eval = _first_present(
        names,
        ("bf16_warmup_max_eval", "bf16_start_max_eval", "bf16_max_eval"),
    )
    if bf16_max_eval is not None:
        kwargs[bf16_max_eval] = 2

    fp32_steps = _first_present(
        names,
        ("fp32_polish_steps", "polish_steps", "final_steps"),
    )
    if fp32_steps is not None:
        kwargs[fp32_steps] = 2
    fp32_max_eval = _first_present(
        names,
        ("fp32_polish_max_eval", "polish_max_eval", "final_max_eval"),
    )
    if fp32_max_eval is not None:
        kwargs[fp32_max_eval] = 4

    return kwargs


def _phase_names(result: dict[str, object]) -> list[str]:
    names: list[str] = []
    timing = result.get("timing", {})
    if isinstance(timing, dict):
        for phase in timing.get("phases", []):
            if isinstance(phase, dict):
                names.append(str(phase.get("phase", "")))
    for record in result.get("history", []):
        if isinstance(record, dict):
            names.append(str(record.get("phase", "")))
    return names


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


def test_global_lbfgs_bf16_start_fp32_polish_matches_fp32_short_run(tmp_path):
    bf16_kwargs = _bf16_start_kwargs()
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")

    fp32_model = _make_small_model(
        DATA_DIR,
        init_rate=INTERIOR_INIT_RATES[0],
        cache_dir=tmp_path / "fp32-cache",
    )
    fp32_result = optimize_global_rates_lbfgs(
        fp32_model,
        min_rate=MIN_RATE,
        interior_init_rates=INTERIOR_INIT_RATES,
        steps=3,
        max_eval=6,
        tolerance_grad=1e-3,
        tolerance_change=1e-7,
    )

    bf16_model = _make_small_model(
        DATA_DIR,
        init_rate=INTERIOR_INIT_RATES[0],
        cache_dir=tmp_path / "bf16-cache",
    )
    try:
        bf16_result = optimize_global_rates_lbfgs(
            bf16_model,
            min_rate=MIN_RATE,
            interior_init_rates=INTERIOR_INIT_RATES,
            steps=3,
            max_eval=6,
            tolerance_grad=1e-3,
            tolerance_change=1e-7,
            **bf16_kwargs,
        )
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        if "bf16" in str(exc).lower() or "bfloat16" in str(exc).lower():
            pytest.skip(f"bf16-start optimizer path is not supported: {exc}")
        raise

    phase_names = [name.lower() for name in _phase_names(bf16_result)]
    assert any("bf16" in name or "bfloat" in name for name in phase_names)
    assert any("fp32" in name or "float32" in name for name in phase_names)

    assert math.isfinite(fp32_result["negative_log_likelihood"])
    assert math.isfinite(bf16_result["negative_log_likelihood"])
    assert torch.isfinite(fp32_result["rates"]).all()
    assert torch.isfinite(bf16_result["rates"]).all()

    nll_gap = abs(
        bf16_result["negative_log_likelihood"]
        - fp32_result["negative_log_likelihood"]
    )
    assert nll_gap <= BF16_NLL_CLOSE_ATOL_BITS

    fp32_rates = fp32_result["rates"].detach().to(device="cpu", dtype=torch.float64)
    bf16_rates = bf16_result["rates"].detach().to(device="cpu", dtype=torch.float64)
    max_rel_err = torch.max(
        torch.abs(bf16_rates - fp32_rates) / torch.clamp(fp32_rates.abs(), min=1e-30)
    ).item()
    assert max_rel_err <= BF16_RATE_CLOSE_RTOL

    assert bf16_model.theta.dtype == torch.float32
    assert bf16_result["theta"].dtype == torch.float32
    assert bf16_result["rates"].dtype == torch.float32
