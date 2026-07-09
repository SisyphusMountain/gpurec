# Perf / Golden Non-Regression Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three reproducible golden non-regression tests (`global`, `genewise`, `specieswise`) that fit a fixed rustree-simulated dataset and assert the final likelihood and full fitted-rate vector still match a committed golden, while logging (not asserting) wall time.

**Architecture:** A seeded `rustree` simulator (`gpurec/bench/simulate.py`) produces a gpurec-compatible dataset on demand (regen-only; trees never committed). A `mint_goldens.py` script fits each mode via `GeneReconModel` + `optimize`, measures run-to-run spread, and writes committed golden JSONs with empirically-set tolerances. Three `@gpu @slow` pytest tests regenerate, fit, and assert against the goldens; they skip cleanly when `rustree`/CUDA are absent.

**Tech Stack:** Python 3.12, PyTorch (CUDA/fp32 + fp64), Triton, `rustree` (PyO3/maturin), pytest.

## Global Constraints

- Branch: **`dev`** (all work committed here; `main` is library-only and untouched).
- No `clamp`/`clip`/`min`/`max` used to mask numerical values (project rule).
- No wall-time **assertion** — time is recorded to a gitignored report only.
- Golden `rates` are stored/compared as the **full** fitted-rate vector (`allclose`), never summary stats.
- theta layout is `[log2 D, log2 L, log2 T]` (order **D, L, T**); linear rates are `2.0 ** theta`.
- Fit driver is `GeneReconModel` + `gpurec.fit.optimize.optimize` **as-is** (genewise uses `optimize` with `group_index`; see Task 4 note).
- Simulation targets: species `n=500` extant leaves, `1000` families, DTL rate scale `0.05`. Small-scale variants (e.g. `n=20`, `5` families) are used only in fast plumbing/smoke tests.
- rustree is **deterministic** for a fixed seed (verified) — this is what makes the committed golden valid under regen-only.
- New pytest marker: `regression`.

---

### Task 0: Make `rustree` importable in the gpurec env + register the `regression` marker

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options] markers`)
- Env: gpurec `.venv` (install `rustree`)

**Interfaces:**
- Produces: a gpurec-env `python` where `import rustree` succeeds; a registered `regression` marker.

- [ ] **Step 1: Confirm rustree is absent, then install it into the gpurec venv**

Run (from repo root `/home/enzo/Documents/git/gpurec/gpurec`):
```bash
.venv/bin/python -c "import rustree" 2>&1 | tail -1   # expect ModuleNotFoundError
# Preferred: build+install from source (maturin backend, release):
.venv/bin/pip install /home/enzo/Documents/git/rustree
```
Expected: install succeeds. **Fallback if the build fails** (a verified-working prebuilt cp312 package exists): symlink just the package (not all of miniforge site-packages, which would shadow torch):
```bash
ln -s /home/enzo/miniforge3/lib/python3.12/site-packages/rustree \
      "$(.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')/rustree"
```

- [ ] **Step 2: Verify import works in the gpurec env**

Run:
```bash
.venv/bin/python -c "import rustree, torch; print('rustree', hasattr(rustree,'simulate_species_tree'), 'torch', torch.cuda.is_available())"
```
Expected: `rustree True torch True` (torch still resolves to gpurec's build).

- [ ] **Step 3: Register the `regression` marker**

In `pyproject.toml`, under `[tool.pytest.ini_options]`, extend `markers`:
```toml
markers = [
    "gpu: requires CUDA + Triton (deselect with -m 'not gpu')",
    "slow: long-running test",
    "regression: golden non-regression benchmark (rustree-simulated; skips if rustree/CUDA absent)",
]
```

- [ ] **Step 4: Verify pytest sees the marker with no warnings**

Run:
```bash
.venv/bin/python -m pytest --markers 2>&1 | grep regression
```
Expected: the `regression` marker line prints.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "test(regression): register 'regression' marker; document rustree env prereq"
```

---

### Task 1: `gpurec/bench/simulate.py` — deterministic, gpurec-compatible dataset generation

The single riskiest piece: producing rustree output that gpurec's preprocessor accepts. The **acceptance gate is the real invariant** — `GeneReconModel(...)` must construct and yield a finite likelihood — not any newick-string heuristic. This task establishes the exact extant-pruning recipe by making that gate pass at small scale.

**Files:**
- Create: `gpurec/bench/simulate.py`
- Test: `tests/regression/test_simulate.py`
- Create: `tests/regression/__init__.py` (empty)

**Interfaces:**
- Produces:
  - `SIM_PARAMS: dict` — canonical params per mode (seeds, `n_species`, `n_families`, `dtl`).
  - `def simulate_dataset(mode: str, out_dir: str | Path, *, n_species: int, n_families: int, dtl: float, seed: int) -> tuple[str, list[str]]` — returns `(species_nwk_path, [gene_nwk_path, ...])`. Writes `species.nwk` and `fam_000000.nwk …` under `out_dir`. For `genewise`, per-family (D,T,L) are drawn from a seeded RNG around `dtl`; for `specieswise`, per-species branch-rate vectors around `dtl`; for `global`, the scalar `dtl`. Deterministic for fixed `seed`.
  - `def true_rates(mode, ...) -> "np.ndarray"` — the ground-truth simulation rates used by the Task 4 statistical fallback (shape `[3]`, `[G,3]`, or `[S,3]`).

- [ ] **Step 1: Write the failing acceptance test (small scale, all three modes)**

```python
# tests/regression/test_simulate.py
import math
from pathlib import Path
import pytest

rustree = pytest.importorskip("rustree")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from gpurec.bench.simulate import simulate_dataset
from gpurec.api.model import GeneReconModel


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["global", "genewise", "specieswise"])
def test_simulated_dataset_is_gpurec_compatible(mode, tmp_path):
    sp, genes = simulate_dataset(mode, tmp_path, n_species=20, n_families=5, dtl=0.05, seed=1)
    assert Path(sp).exists() and len(genes) == 5 and all(Path(g).exists() for g in genes)
    model = GeneReconModel(sp, genes, mode=mode, device="cuda", dtype=torch.float32)
    loss = float(model())                      # NLL in bits
    assert math.isfinite(loss), f"{mode}: non-finite likelihood {loss}"


@pytest.mark.gpu
def test_simulation_is_deterministic(tmp_path):
    a = simulate_dataset("global", tmp_path / "a", n_species=20, n_families=5, dtl=0.05, seed=7)
    b = simulate_dataset("global", tmp_path / "b", n_species=20, n_families=5, dtl=0.05, seed=7)
    assert [Path(g).read_text() for g in a[1]] == [Path(g).read_text() for g in b[1]]
    assert Path(a[0]).read_text() == Path(b[0]).read_text()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/regression/test_simulate.py -x -q`
Expected: FAIL (`ModuleNotFoundError: gpurec.bench.simulate` / `simulate_dataset` undefined).

- [ ] **Step 3: Implement `simulate.py`, establishing the extant recipe against the gate**

Write `gpurec/bench/simulate.py` below. The extant-pruning recipe is chosen to make Step-1's `GeneReconModel` gate pass; start with `forest.sample_extant()` and, if the gate rejects (non-finite / preprocess error) because gene tip-species fall outside the species tree, switch the written species tree to the pruned forest's own `species_tree` (`fe.species_tree`) so labels are consistent. The gate — not a regex — decides correctness.

```python
"""Deterministic rustree simulation of gpurec-compatible DTL datasets (regen-only benchmark).

Trees are never committed; callers regenerate from a fixed seed. rustree is imported lazily so
the rest of gpurec does not depend on it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Canonical parameters for the committed goldens (full-scale). Seeds are arbitrary but fixed.
SIM_PARAMS = {
    "global":      {"seed": 20260709, "n_species": 500, "n_families": 1000, "dtl": 0.05},
    "genewise":    {"seed": 20260710, "n_species": 500, "n_families": 1000, "dtl": 0.05},
    "specieswise": {"seed": 20260711, "n_species": 500, "n_families": 1000, "dtl": 0.05},
}


def _extant_species_tree_and_forest(sp_full, forest):
    """Return (species_tree_to_write, pruned_forest) whose gene tip-species are a subset of the
    species tree's leaves. Uses the pruned forest's carried species_tree so labels stay consistent."""
    fe = forest.sample_extant()
    sp_out = fe.species_tree  # carries the same node labeling as the pruned gene tips
    return sp_out, fe


def _write(sp_out, pruned_forest, out_dir: Path) -> tuple[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sp_path = out_dir / "species.nwk"
    sp_out.save_newick(str(sp_path))
    gene_paths = []
    for i, g in enumerate(pruned_forest):
        p = out_dir / f"fam_{i:06d}.nwk"
        nwk = g.to_newick()
        p.write_text(nwk if nwk.rstrip().endswith(";") else nwk + ";")
        gene_paths.append(str(p))
    return str(sp_path), gene_paths


def simulate_dataset(mode, out_dir, *, n_species, n_families, dtl, seed):
    import rustree
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    sp_full = rustree.simulate_species_tree(n_species, 1.0, 0.5, seed=seed)

    if mode == "global":
        forest = sp_full.simulate_dtl_batch(n_families, dtl, dtl, dtl, seed=seed)
    elif mode == "genewise":
        # per-family (D,T,L) ~ lognormal around `dtl` (positive, ~0.5 dex spread); fixed by rng.
        rates = dtl * np.exp(rng.normal(0.0, 0.5, size=(n_families, 3)))
        trees = [sp_full.simulate_dtl(float(d), float(t), float(l), seed=int(seed + 1 + i))
                 for i, (d, t, l) in enumerate(rates)]
        forest = rustree.GeneForest.from_gene_trees(trees) if hasattr(rustree.GeneForest, "from_gene_trees") \
            else _forest_from_trees(sp_full, trees)
    elif mode == "specieswise":
        n_branch = sp_full.num_nodes()
        d = (dtl * np.exp(rng.normal(0.0, 0.5, n_branch))).tolist()
        t = (dtl * np.exp(rng.normal(0.0, 0.5, n_branch))).tolist()
        l = (dtl * np.exp(rng.normal(0.0, 0.5, n_branch))).tolist()
        orig = np.zeros(n_branch).tolist()
        forest = sp_full.simulate_dtl_batch_with_branch_rates(n_families, d, t, l, orig, seed=seed)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    sp_out, pruned = _extant_species_tree_and_forest(sp_full, forest)
    return _write(sp_out, pruned, out_dir)


def _forest_from_trees(sp_full, trees):
    """Fallback wrapper exposing len()/iteration/sample_extant()/species_tree over a tree list,
    used only if rustree lacks GeneForest.from_gene_trees. Establish the concrete shape in Step 3."""
    raise NotImplementedError  # resolved empirically in Step 3 against the acceptance gate
```

Resolve the two empirically-determined points **in this step** by making Step-1 pass:
1. **genewise forest assembly** — if `GeneForest.from_gene_trees` does not exist, either simulate genewise per-family with `simulate_dtl` and prune each tree individually (write directly, skipping the forest wrapper), or discover the correct rustree constructor. Whichever makes the gate pass.
2. **specieswise branch-rate array length** — `n_branch = sp_full.num_nodes()`; if `simulate_dtl_batch_with_branch_rates` raises a length error, adjust to the length it validates (it reports the expected size).

- [ ] **Step 4: Run the acceptance + determinism tests until green**

Run: `.venv/bin/python -m pytest tests/regression/test_simulate.py -x -q`
Expected: PASS (3 compatibility params + 1 determinism = 4 passed). Iterate Step 3 until green — the finite-likelihood gate is the definition of "compatible".

- [ ] **Step 5: Commit**

```bash
git add gpurec/bench/simulate.py tests/regression/__init__.py tests/regression/test_simulate.py
git commit -m "feat(bench): deterministic rustree DTL dataset simulation (3 modes), gpurec-compat gated"
```

---

### Task 2: `tests/regression/conftest.py` — session-scoped regen fixtures + gitignore

**Files:**
- Create: `tests/regression/conftest.py`
- Modify: `.gitignore` (ignore `tests/regression/reports/`)

**Interfaces:**
- Consumes: `simulate_dataset`, `SIM_PARAMS` (Task 1).
- Produces: pytest fixtures `dataset(mode)` → `(species_path, gene_paths)` regenerated **once per session per mode** into a session tmp dir; `require_rustree` autouse skip guard.

- [ ] **Step 1: Write the fixtures**

```python
# tests/regression/conftest.py
import pytest


@pytest.fixture(scope="session")
def _sim_root(tmp_path_factory):
    return tmp_path_factory.mktemp("gpurec_regression_datasets")


@pytest.fixture(scope="session")
def dataset(_sim_root):
    """Session-cached regen: dataset(mode) -> (species_path, [gene_paths]) at FULL scale."""
    pytest.importorskip("rustree")
    from gpurec.bench.simulate import simulate_dataset, SIM_PARAMS
    cache = {}

    def _get(mode):
        if mode not in cache:
            p = SIM_PARAMS[mode]
            cache[mode] = simulate_dataset(mode, _sim_root / mode, n_species=p["n_species"],
                                           n_families=p["n_families"], dtl=p["dtl"], seed=p["seed"])
        return cache[mode]

    return _get
```

- [ ] **Step 2: Add reports/ to .gitignore**

Append to `.gitignore`:
```
# regression benchmark wall-time logs (machine-specific, not asserted)
tests/regression/reports/
```

- [ ] **Step 3: Verify the fixture imports (collection-only, no full run)**

Run: `.venv/bin/python -m pytest tests/regression/ --collect-only -q`
Expected: collection succeeds (no errors); tests from Task 1 are listed.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/conftest.py .gitignore
git commit -m "test(regression): session-scoped regen dataset fixture; gitignore reports/"
```

---

### Task 3: `mint_goldens.py` — fit each mode, measure spread, write goldens

**Files:**
- Create: `tests/regression/mint_goldens.py`

**Interfaces:**
- Consumes: `simulate_dataset`, `SIM_PARAMS` (Task 1); `GeneReconModel`; `gpurec.fit.optimize.optimize`, `final_eval`.
- Produces: `def fit_mode(mode, species_path, gene_paths) -> tuple[float, "np.ndarray", float]` returning `(nll_bits, rates_vec, wall_s)`; writes `tests/regression/goldens/{mode}.json` with fields per the spec (`mode`, `provenance`, `nll`, `rates`, `recorded_wall_s`, `tolerances`). CLI: `python -m tests.regression.mint_goldens [--mode MODE] [--repeats N]`.

- [ ] **Step 1: Implement the fit driver + minting**

```python
# tests/regression/mint_goldens.py
"""Mint committed goldens: fit each mode, measure run-to-run spread, set tolerances = 3x spread.
Run ONCE on the target GPU (RTX 4090). Re-run to re-mint if rustree/gpurec change."""
from __future__ import annotations

import argparse, json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gpurec.api.model import GeneReconModel
from gpurec.fit.optimize import optimize, final_eval
from gpurec.bench.simulate import simulate_dataset, SIM_PARAMS

GOLDENS = Path(__file__).parent / "goldens"


def _git_rev(cwd):
    try:
        return subprocess.check_output(["git", "-C", str(cwd), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def fit_mode(mode, species_path, gene_paths):
    model = GeneReconModel(species_path, gene_paths, mode=mode, device="cuda", dtype=torch.float32)
    t0 = time.perf_counter()
    theta_hat, _hist = optimize(
        model.batch_statics, model.theta.detach(), model.receiver_weights.detach(),
        group_index=model.rate_family_idx, verbose=False)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    nll_bits, _gn = final_eval(model.batch_statics, theta_hat, model.receiver_weights.detach())
    rates = (2.0 ** theta_hat.detach().float().cpu()).numpy()   # [.,3] order D,L,T
    return float(nll_bits), rates, float(wall_s)


def mint(mode, repeats=5):
    p = SIM_PARAMS[mode]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sp, genes = simulate_dataset(mode, Path(d) / mode, n_species=p["n_species"],
                                     n_families=p["n_families"], dtl=p["dtl"], seed=p["seed"])
        nlls, rate_runs, walls = [], [], []
        for _ in range(repeats):
            nll, rates, wall = fit_mode(mode, sp, genes)
            nlls.append(nll); rate_runs.append(rates); walls.append(wall)

    nlls = np.array(nlls); R = np.stack(rate_runs)          # [repeats, ., 3]
    nll_spread = float(nlls.std()) if repeats > 1 else 0.0
    rate_spread_rel = float(np.abs(R - R.mean(0)).max() / (np.abs(R).mean() + 1e-30)) if repeats > 1 else 0.0
    nll_rtol = max(3 * nll_spread / (abs(nlls.mean()) + 1e-30), 1e-3)
    rates_rtol = max(3 * rate_spread_rel, 1e-2)

    golden = {
        "mode": mode,
        "provenance": {
            "rustree_commit": _git_rev("/home/enzo/Documents/git/rustree"),
            "gpurec_commit": _git_rev(Path(__file__).resolve().parents[2]),
            "seed": p["seed"], "n_species": p["n_species"], "n_families": p["n_families"],
            "dtl": p["dtl"], "device": torch.cuda.get_device_name(0), "dtype": "float32",
            "repeats": repeats, "nll_spread": nll_spread, "rate_spread_rel": rate_spread_rel,
            "minted_utc": datetime.now(timezone.utc).isoformat(),
        },
        "nll": float(nlls.mean()),
        "rates": R.mean(0).tolist(),
        "recorded_wall_s": float(np.median(walls)),
        "tolerances": {"nll_rtol": nll_rtol, "rates_rtol": rates_rtol, "rates_atol": 1e-4},
    }
    GOLDENS.mkdir(exist_ok=True)
    (GOLDENS / f"{mode}.json").write_text(json.dumps(golden, indent=2))
    print(f"[{mode}] nll={golden['nll']:.6f} rtol={nll_rtol:.2e} rates_rtol={rates_rtol:.2e} "
          f"wall~{golden['recorded_wall_s']:.1f}s")
    return golden


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(SIM_PARAMS), default=None)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    for m in ([a.mode] if a.mode else list(SIM_PARAMS)):
        mint(m, repeats=a.repeats)
```

- [ ] **Step 2: Smoke-run minting at small scale (fast, does not commit goldens yet)**

Temporarily verify the driver end-to-end without the full 1000×500 cost by running a tiny mint from a Python one-liner (does not touch `goldens/`):
```bash
.venv/bin/python - <<'PY'
import tempfile, torch
from pathlib import Path
from gpurec.bench.simulate import simulate_dataset
from tests.regression.mint_goldens import fit_mode
with tempfile.TemporaryDirectory() as d:
    sp, g = simulate_dataset("global", Path(d), n_species=20, n_families=10, dtl=0.05, seed=1)
    nll, rates, wall = fit_mode("global", sp, g)
    print("nll", nll, "rates", rates.tolist(), "wall", round(wall,2))
    assert torch.isfinite(torch.tensor(nll))
PY
```
Expected: prints a finite `nll`, a 3-vector of positive rates, and a wall time. (Run from repo root so `tests` is importable, or `PYTHONPATH=. `.)

- [ ] **Step 3: Commit the minting script (goldens minted later, Task 5)**

```bash
git add tests/regression/mint_goldens.py
git commit -m "feat(regression): mint_goldens driver (fit via optimize, spread-based tolerances)"
```

---

### Task 4: `test_perf_regression.py` — the three golden tests + time report

**Files:**
- Create: `tests/regression/test_perf_regression.py`

**Interfaces:**
- Consumes: `dataset` fixture (Task 2); `fit_mode` (Task 3); committed `goldens/{mode}.json` (Task 5).
- Produces: 3 parametrized tests `test_golden[global|genewise|specieswise]`, each appending one line to `tests/regression/reports/wall_times.jsonl`.

- [ ] **Step 1: Write the tests (skip if the golden is not yet minted)**

```python
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
```

- [ ] **Step 2: Verify collection + clean skip (goldens absent)**

Run: `.venv/bin/python -m pytest tests/regression/test_perf_regression.py -q`
Expected: 3 SKIPPED with "golden not minted" (goldens are committed in Task 5).

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_perf_regression.py
git commit -m "test(regression): three golden non-regression tests (NLL + full rate vector), time logged"
```

> **Genewise driver note (spec review item):** genewise is driven through `optimize(..., group_index=model.rate_family_idx)` per the user directive, not `fit_genewise`. If Task 5 minting shows genewise does not converge sensibly through `optimize` (e.g. `||g||` not descending, wildly unstable `rates` across repeats → `rates_rtol` blows up), surface it and switch this one test to `gpurec.fit.genewise_fit.fit_genewise`; record the decision in the golden's provenance.

---

### Task 5: Mint the real goldens on the 4090 and commit them

**Files:**
- Create: `tests/regression/goldens/global.json`, `genewise.json`, `specieswise.json` (committed artifacts)

**Interfaces:**
- Consumes: everything above.
- Produces: committed goldens that flip Task 4's tests from SKIP to active.

- [ ] **Step 1: Mint all three at full scale**

Run (from repo root; this is the one intentionally heavy step — 3 modes × 5 repeats × (1000 families, 500 leaves)):
```bash
.venv/bin/python -m tests.regression.mint_goldens --repeats 5
```
Expected: three lines printed with finite `nll`, sensible `rtol`s, and per-mode wall times; three JSONs written under `tests/regression/goldens/`. **Ask the user before launching** (long-running benchmark, per project rule).

- [ ] **Step 2: Sanity-check the goldens, then run the real regression tests once**

Run:
```bash
.venv/bin/python -m pytest tests/regression/test_perf_regression.py -q -rA
```
Expected: 3 PASSED (a fresh regen + fit reproduces the just-minted goldens within the spread-based tolerances). If genewise fails on instability, apply the Task 4 genewise note and re-mint that mode.

- [ ] **Step 3: Commit the goldens**

```bash
git add tests/regression/goldens/*.json
git commit -m "test(regression): mint global/genewise/specieswise goldens on RTX 4090"
```

- [ ] **Step 4: Full-suite regression sanity (deselect slow to confirm no collateral breakage)**

Run:
```bash
.venv/bin/python -m pytest -q -m "not slow"
```
Expected: existing suite still green; regression tests deselected (they are `slow`).

---

## Self-Review

**Spec coverage:**
- Three modes / one test each → Tasks 1, 3, 4, 5. ✓
- rustree simulation (500 leaves, 1000 fam, DTL 0.05), all three rate structures → Task 1 (`SIM_PARAMS`, `simulate_dataset`). ✓
- Regen-only, trees not committed, skip if rustree absent → Tasks 1–2 (`importorskip`), `.gitignore`. ✓
- Golden = NLL + **full** rate vector; `allclose` → Task 4. ✓
- Time recorded, never asserted → Task 4 `_record`. ✓
- Tolerances set empirically (spread×3) with NLL-tight/θ-statistical fallback → Task 3 + Task 4 genewise note. ✓
- `GeneReconModel` + `optimize` as-is; theta order D,L,T → Tasks 3–4, Global Constraints. ✓
- Provenance (rustree/gpurec commit, seeds, device) → Task 3 golden JSON. ✓
- `@gpu @slow`, new `regression` marker → Task 0 + Task 4. ✓

**Placeholder scan:** The only deferred specifics are the two empirically-nailed points in Task 1 (genewise forest assembly; branch-rate array length), which are resolved *within* Task 1 against a hard finite-likelihood gate — legitimate TDD, not hand-waving. `_forest_from_trees` raises `NotImplementedError` deliberately as a marker to resolve in Step 3, and Step 4 will not pass until it is.

**Type consistency:** `fit_mode` returns `(nll_bits: float, rates: np.ndarray, wall_s: float)` — consumed identically in Tasks 3 and 4. `simulate_dataset(...) -> (str, list[str])` — consumed identically in Tasks 1, 2, 3. Golden JSON keys (`nll`, `rates`, `recorded_wall_s`, `tolerances.{nll_rtol,rates_rtol,rates_atol}`) written in Task 3, read in Task 4 — consistent.
