# tests/regression/test_perf_regression.py
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rustree")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tests.regression.mint_goldens import fit_mode

GOLDENS = Path(__file__).parent / "goldens"
REPORTS = Path(__file__).parent / "reports"


def _record(mode, wall_s, baseline):
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / "wall_times.jsonl").open("a") as f:
        f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
                            "wall_s": wall_s, "baseline_s": baseline}) + "\n")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize("mode", ["global", "genewise", "specieswise"])
def test_golden(mode, dataset):
    gpath = GOLDENS / f"{mode}.json"
    if not gpath.exists():
        pytest.skip(f"golden not minted: {gpath} (run: python -m tests.regression.mint_goldens --mode {mode})")
    golden = json.loads(gpath.read_text())
    sp, genes = dataset(mode)
    nll, rates, wall = fit_mode(mode, sp, genes)
    _record(mode, wall, golden["recorded_wall_s"])            # logged, never asserted

    tol = golden["tolerances"]
    assert np.isclose(nll, golden["nll"], rtol=tol["nll_rtol"]), \
        f"{mode}: NLL {nll} vs golden {golden['nll']} (rtol={tol['nll_rtol']})"
    exp = np.asarray(golden["rates"]); got = np.asarray(rates)
    assert got.shape == exp.shape, f"{mode}: rate shape {got.shape} vs {exp.shape}"
    assert np.allclose(got, exp, rtol=tol["rates_rtol"], atol=tol["rates_atol"]), (
        f"{mode}: max rel rate delta "
        f"{np.abs(got - exp).max() / (np.abs(exp).mean() + 1e-30):.2e} exceeds rtol={tol['rates_rtol']}")
