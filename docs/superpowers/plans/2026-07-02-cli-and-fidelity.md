# `gpurec` CLI + AleRax Fidelity Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `gpurec` command-line interface (`reconcile` + `fit`) and an AleRax reference-fidelity kit (parsers, vendored fixtures, a runnable fidelity gate, and a cluster scale-benchmark scaffold) to `consolidate-release`.

**Architecture:** A new Triton-free `gpurec/bench/` subpackage (pure-text AleRax parsers + a fidelity comparator) and a new `gpurec/cli/` package (argparse, heavy imports lazy) built entirely on the base's own surface — `GeneReconModel`, `SolverOptions`, `gpurec.optim.optimize`/`final_eval`, and `fit_genewise`. Because the shipped stock AleRax numbers omit the `−ln(2S−1)` origination term (off by `+ln(2S−1)` per family), the batch rebuilds `AleRax_fixed` (which computes gpurec's likelihood convention) and generates references under `ref/`, then a GPU gate proves gpurec matches them per-family to ≤1e-3 nats. A repo-root `benchmark/` directory ports gergely's scale kit, rewired to the CLI/optimizers, runnable on the cluster.

**Tech Stack:** Python 3.10+, argparse, PyTorch + Triton (GPU forward), pytest (`gpu` marker), Rust preprocessor (already built); AleRax reference build via cmake + MPI/g++ (verified present).

## Global Constraints

- Base repo: `consolidate-release`. Branch `feat/cli-and-fidelity` off `main` @ `2dee4d9c`.
- **CWD for every command is `/home/enzo/Documents/git/gpurec/consolidate-release`; Python is `.venv/bin/python`** (a sibling `/home/enzo/Documents/git/gpurec/gpurec/` tree exists — a wrong CWD resolves `import gpurec` to the wrong package).
- **Units:** the base forward returns **NLL in bits (log₂)**. AleRax likelihoods are **nats (ln)**. Convert everywhere: `logL_nats = -loss_bits * math.log(2)`. `theta` is log₂-rates.
- **Rate order (verified in code):** gpurec `theta = [log₂ D, log₂ L, log₂ T]` — `theta[2]` is transfer (`extract_parameters.py` reads `log_pT = result[..., 3]`). AleRax `model_parameters.txt` columns are also `D L T`, so **map straight to theta with NO reordering**. CLI `--delta/--tau/--lambda` (D/T/L) → `theta = [log₂ D, log₂ L, log₂ T]`. (An earlier `[D, T, L]` draft swapped T/L — invisible when T=L, ~20-nat error when transfer is active. See memory `gpurec-theta-order-dlt`.)
- **Lazy heavy imports:** `gpurec/cli/*` must NOT import `torch`, `GeneReconModel`, or any forward/optim module at module top level — only inside functions — so `--help`/arg-parsing need no GPU work. (`import gpurec` itself still pulls Triton; that is the base's existing behavior and is fine on this Linux+Triton box.)
- **No edits to any existing forward/optimize/solver/kernel path.** This batch only ADDS `gpurec/cli/`, `gpurec/bench/`, tests, `tests/data/` fixtures, `benchmark/`, and one `[project.scripts]` line + one `.gitignore` change.
- **Git hygiene:** never `git add -A`/`git add .`. Stage only the files each task creates. Do NOT touch pre-existing uncommitted work: `crates/gpurec-backtrack/src/lib.rs`, `crates/gpurec-preprocess/src/lib.rs` (modified), and untracked `experiments/ghost_lineages/`, `ghost_experiments/`, `paper.pdf`.
- **GPU marker:** GPU tests use `@pytest.mark.gpu` (registered in `pyproject.toml`). CPU suite = `-m "not gpu"`; GPU suite = `-m gpu`. This box has CUDA, so GPU tests run here.
- **Reference oracle = AleRax_fixed (critical):** gpurec computes the AleRax supplementary likelihood `L = (Σ_s p^O_s Π_{Γ,s})/(Σ_s p^O_s(1−E_s))` (`solver.py:158–169`) and matches **AleRax_fixed** to ~1e-5 nats. Stock AleRax v1.3.0 (the shipped `output/` numbers) omits the `−ln(2S−1)` origination branch-count term, so it is off by `+ln(2S−1)` per family — a normalization difference, not convergence (at these fixtures' low fitted rates AleRax_fixed is flat across `--fixed-point-iterations`; at high rates the default under-converges, see the convergence note below, which is why Task 6 pins 64 iters). Gate against AleRax_fixed references, never the shipped stock numbers.
- **AleRax build (Task 6):** source = `agent-worktrees/*/AleRax_fixed/AleRax` (verified builds + runs on this box; binary already present). Copy to stable `/home/enzo/Documents/git/gpurec/tools/AleRax_fixed/` (worktrees are ephemeral; never build inside them). Generate references at `--fixed-point-iterations 64` (converged in any rate regime; the default 4 under-converges at high rates — measured 87 nats at D/L/T≈1.5). The AleRax build lives OUTSIDE the tracked repo — do NOT `git add` it.
- **Oracle-wins guardrail:** if the fidelity gate (Task 7) fails, DO NOT loosen the tolerance — record the actual delta and escalate. A ~20-nat miss on the transfer fixture (`test_trees_2`) specifically means the theta order is wrong (T/L swapped).
- **Convergence (rate-dependent):** at the toy fixtures' low fitted rates AleRax's default fixed-point iterations already converge, so no gpurec self-convergence test is added this batch. But at HIGH rates the default 4 iters under-converge (measured 87 nats at fixed D/L/T≈1.5/1.6/1.7 on test_mixed_200; gpurec matches AleRax_fixed only at the converged value), so Task 6 pins `--fixed-point-iterations 64` when generating references. Broader convergence testing on larger/higher-rate families is a follow-up.

---

### Task 1: AleRax I/O parsers (`gpurec/bench/alerax_io.py`)

**Files:**
- Create: `gpurec/bench/__init__.py`, `gpurec/bench/alerax_io.py`
- Test: `tests/test_alerax_io.py`

**Interfaces:**
- Produces:
  - `AleraxRates = namedtuple("AleraxRates", ["D", "L", "T"])` — maps straight to gpurec `theta = [log₂ D, log₂ L, log₂ T]` (same column order; no reorder helper).
  - `parse_alerax_likelihoods(path) -> dict[str, float]` (normalized family → logL nats)
  - `parse_alerax_parameters(path) -> dict[str, AleraxRates]` (node → rates, D L T order)
  - `norm_family_name(name: str) -> str`
  - `global_rates(params: dict[str, AleraxRates]) -> AleraxRates`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alerax_io.py`:
```python
import math
import pytest
from gpurec.bench.alerax_io import (
    parse_alerax_likelihoods, parse_alerax_parameters, norm_family_name,
    AleraxRates, global_rates,
)


def test_parse_likelihoods(tmp_path):
    p = tmp_path / "per_fam_likelihoods.txt"
    p.write_text("my_family -2.56495\nother.ale -8.5\nbad line only\n")
    assert parse_alerax_likelihoods(p) == {"my_family": -2.56495, "other": -8.5}


def test_parse_parameters_DLT_order(tmp_path):
    p = tmp_path / "model_parameters.txt"
    p.write_text("# node D L T\n5 0.1 0.2 0.3\nn6 0.1 0.2 0.3\n")
    out = parse_alerax_parameters(p)
    # AleRax columns are D L T, the SAME order as gpurec theta = [D, L, T] (no reorder).
    assert out["5"] == AleraxRates(D=0.1, L=0.2, T=0.3)


def test_norm_family_name():
    assert norm_family_name("my_family.ale") == "my_family"
    assert norm_family_name("some/path/fam.newick") == "fam"
    assert norm_family_name("plain") == "plain"


def test_global_rates_asserts_uniform(tmp_path):
    p = tmp_path / "mp.txt"
    p.write_text("# node D L T\n5 0.1 0.2 0.3\n6 0.1 0.2 0.3\n")
    assert global_rates(parse_alerax_parameters(p)) == AleraxRates(0.1, 0.2, 0.3)
    p2 = tmp_path / "mp2.txt"
    p2.write_text("# node D L T\n5 0.1 0.2 0.3\n6 0.9 0.2 0.3\n")
    with pytest.raises(ValueError):
        global_rates(parse_alerax_parameters(p2))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /home/enzo/Documents/git/gpurec/consolidate-release && .venv/bin/python -m pytest tests/test_alerax_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpurec.bench'`.

- [ ] **Step 3: Create the package + parser module**

Create `gpurec/bench/__init__.py`:
```python
"""Benchmark + AleRax-fidelity helpers (Triton-free, CPU-importable)."""
from gpurec.bench.alerax_io import (
    AleraxRates,
    parse_alerax_likelihoods,
    parse_alerax_parameters,
    norm_family_name,
    global_rates,
)

__all__ = [
    "AleraxRates",
    "parse_alerax_likelihoods",
    "parse_alerax_parameters",
    "norm_family_name",
    "global_rates",
]
```

Create `gpurec/bench/alerax_io.py`:
```python
"""Parsers for AleRax reference output files.

Pure text; no Triton/CUDA import. All log-likelihoods are NATS (ln), matching AleRax.
The rate columns in ``model_parameters.txt`` are ordered ``D L T`` — the SAME order as
gpurec ``theta = [log2 D, log2 L, log2 T]``, so they map straight to theta (no reorder).
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

AleraxRates = namedtuple("AleraxRates", ["D", "L", "T"])
# AleRax columns D L T == gpurec theta order [log2 D, log2 L, log2 T]; no reorder helper needed.

_SUFFIXES = (".ale", ".txt", ".ufboot", ".newick", ".nwk", ".treelist")


def norm_family_name(name: str) -> str:
    """Strip a known extension and directory so gpurec and AleRax names align."""
    base = str(name).strip().rsplit("/", 1)[-1]
    for suf in _SUFFIXES:
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def parse_alerax_likelihoods(path) -> dict:
    """Parse ``per_fam_likelihoods.txt`` -> {normalized_family: logL_nats}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            out[norm_family_name(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return out


def parse_alerax_parameters(path) -> dict:
    """Parse ``model_parameters.txt`` -> {node: AleraxRates(D, L, T)}.

    Skips the ``# node D L T`` comment header. Node labels may be numeric or strings.
    """
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            d, l, t = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        out[parts[0]] = AleraxRates(D=d, L=l, T=t)
    return out


def global_rates(params: dict) -> AleraxRates:
    """Return the common (D, L, T) triple for a GLOBAL-mode AleRax run.

    Raises ValueError if the run is empty or has per-node rate variation.
    """
    if not params:
        raise ValueError("no rate rows parsed")
    vals = list(params.values())
    first = vals[0]
    for r in vals[1:]:
        if r != first:
            raise ValueError("rates are not global (per-node variation present)")
    return first
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_alerax_io.py -v -W error`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add gpurec/bench/__init__.py gpurec/bench/alerax_io.py tests/test_alerax_io.py
git commit -m "feat(bench): AleRax output parsers (likelihoods, parameters, name norm)"
```

---

### Task 2: CLI common helpers + `main` + `reconcile`

**Files:**
- Create: `gpurec/cli/__init__.py`, `gpurec/cli/_common.py`, `gpurec/cli/main.py`, `gpurec/cli/reconcile.py`
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Test: `tests/test_cli.py` (CPU parts) + one `@pytest.mark.gpu` smoke

**Interfaces:**
- Consumes: `gpurec.bench.alerax_io.norm_family_name` (Task 1); base `gpurec.api.model.GeneReconModel`, `gpurec.api.solver_options.SolverOptions`, `gpurec.optim.genewise_fit._resolve_gene_trees`.
- Produces:
  - `gpurec.cli.main.build_parser() -> argparse.ArgumentParser`, `gpurec.cli.main.main(argv=None) -> int`
  - `gpurec.cli._common`: `bits_to_nats(x)`, `add_common_args(parser)`, `resolve_gene_trees(values) -> list`, `make_solver_options(args)`, `make_dtype(name)`, `build_model(args) -> (model, genes_list)`
  - `gpurec.cli.reconcile.add_args(parser)`, `gpurec.cli.reconcile.run_reconcile(args) -> int`

- [ ] **Step 1: Write the failing CPU tests**

Create `tests/test_cli.py`:
```python
import math
import pytest
from gpurec.cli.main import build_parser, main
from gpurec.cli import _common


def test_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--help"])
    assert e.value.code == 0


def test_no_subcommand_returns_2():
    assert main([]) == 2


def test_reconcile_arg_parsing():
    args = build_parser().parse_args(
        ["reconcile", "--species", "sp.nwk", "--gene", "g.nwk",
         "--delta", "0.1", "--tau", "0.2", "--lambda", "0.3"])
    assert args.command == "reconcile"
    assert args.species == "sp.nwk" and args.gene == ["g.nwk"]
    assert (args.delta, args.tau, args.lambda_) == (0.1, 0.2, 0.3)
    assert args.mode == "global" and args.dtype == "float64"


def test_reconcile_requires_species():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["reconcile", "--gene", "g.nwk"])


def test_bits_to_nats():
    assert math.isclose(_common.bits_to_nats(1.0), math.log(2.0))
    assert math.isclose(_common.bits_to_nats(3.0), 3.0 * math.log(2.0))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpurec.cli'`.

- [ ] **Step 3: Create the CLI package**

Create `gpurec/cli/__init__.py`:
```python
"""gpurec command-line interface."""
```

Create `gpurec/cli/_common.py`:
```python
"""Shared CLI helpers. Heavy imports (torch, GeneReconModel) are done lazily inside
functions so ``gpurec --help`` and argument parsing need no GPU work."""
from __future__ import annotations

import math


def bits_to_nats(x: float) -> float:
    """Convert a log2-space value to natural-log space."""
    return x * math.log(2.0)


def add_common_args(parser) -> None:
    parser.add_argument("--species", required=True, help="species tree (Newick)")
    parser.add_argument("--gene", required=True, nargs="+",
                        help="gene tree file(s), a glob, a directory, or an AleRax [FAMILIES] listfile")
    parser.add_argument("--mode", choices=["global", "specieswise", "genewise"], default="global")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--pi-iters", type=int, default=64)
    parser.add_argument("--neumann-terms", type=int, default=64)
    parser.add_argument("--e-max-iter", type=int, default=2000)


def resolve_gene_trees(values) -> list:
    """Resolve --gene values (list / glob / dir / listfile) to gene-tree paths."""
    from gpurec.optim.genewise_fit import _resolve_gene_trees
    return _resolve_gene_trees(values)


def make_dtype(name: str):
    import torch
    return torch.float64 if name == "float64" else torch.float32


def make_solver_options(args):
    from gpurec.api.solver_options import SolverOptions
    return SolverOptions(e_max_iter=args.e_max_iter, pi_iters=args.pi_iters,
                         neumann_terms=args.neumann_terms)


def build_model(args):
    """Build a GeneReconModel from parsed args (lazy heavy import). Returns (model, genes)."""
    from gpurec.api.model import GeneReconModel
    genes = resolve_gene_trees(args.gene)
    model = GeneReconModel(args.species, genes, mode=args.mode, device=args.device,
                           solver_options=make_solver_options(args))
    return model, genes
```

Create `gpurec/cli/reconcile.py`:
```python
"""`gpurec reconcile` — marginal log-likelihood at fixed DTL rates."""
from __future__ import annotations

import math
import sys

from . import _common


def add_args(parser) -> None:
    _common.add_common_args(parser)
    parser.add_argument("--delta", type=float, default=1e-10, help="duplication rate D")
    parser.add_argument("--tau", type=float, default=1e-10, help="transfer rate T")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-10, help="loss rate L")
    parser.add_argument("--out", default=None, help="write per-family logL (AleRax per_fam format)")


def _set_rates(model, D, T, L):
    import torch
    # gpurec theta order is [log2 D, log2 L, log2 T] (theta[2] = transfer)
    triple = torch.tensor([math.log2(D), math.log2(L), math.log2(T)],
                          dtype=model.theta.dtype, device=model.theta.device)
    with torch.no_grad():
        if model.theta.dim() == 1:            # global (3,)
            model.theta.copy_(triple)
        else:                                  # (S,3) or (F,3): broadcast the global triple
            model.theta.copy_(triple.expand_as(model.theta))


def run_reconcile(args) -> int:
    import torch
    from gpurec.bench.alerax_io import norm_family_name

    model, genes = _common.build_model(args)
    _set_rates(model, args.delta, args.tau, args.lambda_)

    rows = []
    with torch.no_grad():
        if args.mode == "genewise":
            loss_vec = model.genewise_loss_vector()          # [F] NLL bits
            for path, loss_bits in zip(genes, loss_vec.tolist()):
                rows.append((norm_family_name(path), -_common.bits_to_nats(loss_bits)))
        else:
            loss_bits = float(model())                        # summed NLL bits
            fam = norm_family_name(genes[0]) if len(genes) == 1 else "all"
            rows.append((fam, -_common.bits_to_nats(loss_bits)))

    for fam, ll in rows:
        if not math.isfinite(ll):
            print(f"error: non-finite log-likelihood for {fam}", file=sys.stderr)
            return 1

    text = "".join(f"{fam} {ll:.6f}\n" for fam, ll in rows)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    sys.stdout.write(text)
    return 0
```

Create `gpurec/cli/main.py`:
```python
"""gpurec command-line entry point."""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpurec", description="GPU DTL gene/species-tree reconciliation")
    sub = parser.add_subparsers(dest="command")
    from . import reconcile, fit
    reconcile.add_args(sub.add_parser("reconcile", help="marginal log-likelihood at fixed rates"))
    fit.add_args(sub.add_parser("fit", help="optimize DTL rates"))
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "reconcile":
        from . import reconcile
        return reconcile.run_reconcile(args)
    if args.command == "fit":
        from . import fit
        return fit.run_fit(args)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

**NOTE:** `main.py`'s `build_parser` imports `from . import reconcile, fit`, so `gpurec/cli/fit.py` must exist as an importable module (created in Task 3). For THIS task, create a minimal `gpurec/cli/fit.py` stub so imports resolve:
```python
"""`gpurec fit` — optimize DTL rates (implemented in Task 3)."""
from __future__ import annotations

from . import _common


def add_args(parser) -> None:
    _common.add_common_args(parser)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--init-rate", type=float, default=None)
    parser.add_argument("--out", default=None)


def run_fit(args) -> int:
    raise NotImplementedError("gpurec fit is implemented in Task 3")
```

- [ ] **Step 4: Register the console script**

Add to `pyproject.toml` a top-level `[project.scripts]` table (place it directly after the `[project]` table's closing content, before `[build-system]` or `[tool.*]`):
```toml
[project.scripts]
gpurec = "gpurec.cli.main:main"
```

- [ ] **Step 5: Run to verify CPU tests pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Add a GPU reconcile smoke test**

Append to `tests/test_cli.py`:
```python
@pytest.mark.gpu
def test_reconcile_smoke_gpu(tmp_path, capsys):
    from pathlib import Path
    data = Path(__file__).parent / "data" / "alerax" / "test_trees_1"
    if not data.exists():
        pytest.skip("fixtures not vendored yet (Task 5)")
    rc = main(["reconcile", "--species", str(data / "sp.nwk"),
               "--gene", str(data / "g.nwk"), "--device", "cuda"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "my_family" in out
```
(This is skipped until Task 5 vendors the fixtures; it becomes live then.)

- [ ] **Step 7: Verify the entry point installs**

Run: `.venv/bin/pip install -e . >/dev/null 2>&1 && .venv/bin/gpurec --help`
Expected: prints usage with `reconcile` and `fit` subcommands, exit 0.

- [ ] **Step 8: Commit**

```bash
git add gpurec/cli/__init__.py gpurec/cli/_common.py gpurec/cli/main.py gpurec/cli/reconcile.py gpurec/cli/fit.py pyproject.toml tests/test_cli.py
git commit -m "feat(cli): gpurec reconcile subcommand + console entry point"
```

---

### Task 3: CLI `fit` subcommand (all three modes)

**Files:**
- Modify: `gpurec/cli/fit.py` (replace the Task-2 stub)
- Test: `tests/test_cli.py` (append `_write_outputs` CPU test + GPU fit smoke)

**Interfaces:**
- Consumes: base `gpurec.optim.optimize.optimize`/`final_eval`, `gpurec.optim.genewise_fit.fit_genewise`, `gpurec.cli._common`, `gpurec.bench.alerax_io.parse_alerax_parameters`.
- Produces: `gpurec.cli.fit.run_fit(args) -> int`, `gpurec.cli.fit._write_outputs(out, node_rates, nll_bits, nll_nats, elapsed_s, mode, n_families)` where `node_rates` is `list[(node_label, D, L, T)]` (gpurec order == AleRax column order) and the file is written verbatim as AleRax `D L T`.

- [ ] **Step 1: Write the failing CPU test for the output writer**

Append to `tests/test_cli.py`:
```python
def test_write_outputs_alerax_format_and_sidecar(tmp_path):
    import json
    from gpurec.cli.fit import _write_outputs
    from gpurec.bench.alerax_io import parse_alerax_parameters, AleraxRates
    out = tmp_path / "rates.txt"
    # node_rates rows are (node, D, L, T) — gpurec order == AleRax D L T column order
    _write_outputs(str(out), [("GLOBAL", 0.1, 0.2, 0.3)],
                   nll_bits=-1.0, nll_nats=-0.6931, elapsed_s=1.0, mode="global", n_families=1)
    parsed = parse_alerax_parameters(out)          # reads columns as D L T
    assert parsed["GLOBAL"] == AleraxRates(D=0.1, L=0.2, T=0.3)   # written verbatim, no reorder
    side = json.loads((tmp_path / "rates.txt.json").read_text())
    assert side["mode"] == "global" and side["nll_nats"] == -0.6931 and side["n_families"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py::test_write_outputs_alerax_format_and_sidecar -v`
Expected: FAIL — `ImportError: cannot import name '_write_outputs'` (stub has no such function).

- [ ] **Step 3: Implement `gpurec/cli/fit.py`**

Replace `gpurec/cli/fit.py` with:
```python
"""`gpurec fit` — optimize DTL rates in global / specieswise / genewise mode.

global / specieswise -> gpurec.optim.optimize + final_eval (writes AleRax-format rates).
genewise            -> gpurec.optim.genewise_fit.fit_genewise (writes a JSON sidecar).
"""
from __future__ import annotations

import json
import math
import time

from . import _common


def add_args(parser) -> None:
    _common.add_common_args(parser)
    parser.add_argument("--steps", type=int, default=300, help="optimizer steps (global/specieswise)")
    parser.add_argument("--init-rate", type=float, default=None, help="initial rate for D, T, L")
    parser.add_argument("--out", default=None,
                        help="write fitted rates (AleRax '# node D L T') + <out>.json sidecar")


def _write_outputs(out, node_rates, nll_bits, nll_nats, elapsed_s, mode, n_families):
    """node_rates: list[(node, D, L, T)] (gpurec order == AleRax cols); file uses 'node D L T'."""
    with open(out, "w") as fh:
        fh.write("# node D L T\n")
        for node, D, L, T in node_rates:
            fh.write(f"{node} {D:.10g} {L:.10g} {T:.10g}\n")
    sidecar = {"mode": mode, "nll_bits": nll_bits, "nll_nats": nll_nats,
               "elapsed_s": elapsed_s, "n_families": n_families}
    with open(f"{out}.json", "w") as fh:
        json.dump(sidecar, fh, indent=2)


def _write_genewise(out, res, nll_bits, nll_nats, elapsed_s):
    theta, rates = res.get("theta"), res.get("rates")
    payload = {"mode": "genewise", "nll_bits": nll_bits, "nll_nats": nll_nats,
               "elapsed_s": elapsed_s, "n_families": int(res.get("n_families", 0)),
               "theta_log2": theta.tolist() if hasattr(theta, "tolist") else theta,
               "rates": rates.tolist() if hasattr(rates, "tolist") else rates}
    with open(f"{out}.json", "w") as fh:
        json.dump(payload, fh, indent=2)


def _set_init_rate(model, rate):
    import torch
    lr = math.log2(rate)
    with torch.no_grad():
        model.theta.fill_(lr)


def _global_node_rates(theta_hat, mode, S):
    """Turn a fitted theta into list[(label, D, L, T)] rows (gpurec order == AleRax cols)."""
    t = theta_hat.detach().cpu()
    if t.dim() == 1:                         # global (3,)
        D, L, T = (2.0 ** t).tolist()
        return [("GLOBAL", D, L, T)]
    rows = []                                # specieswise (S,3)
    for i in range(t.shape[0]):
        D, L, T = (2.0 ** t[i]).tolist()
        rows.append((f"s{i}", D, L, T))
    return rows


def run_fit(args) -> int:
    t0 = time.perf_counter()

    if args.mode == "genewise":
        from gpurec.optim.genewise_fit import fit_genewise
        res = fit_genewise(args.species, args.gene, device=args.device,
                           dtype=_common.make_dtype(args.dtype), certify=True)
        nll_nats = float(res.get("loss_nats", float("nan")))
        nll_bits = nll_nats / math.log(2.0)
        n_fam = int(res.get("n_families", 0))
        elapsed = time.perf_counter() - t0
        print(f"fit(genewise): final NLL = {nll_nats:.6f} nats ({n_fam} families, {elapsed:.1f}s)")
        if args.out:
            _write_genewise(args.out, res, nll_bits, nll_nats, elapsed)
        return 0

    # global / specieswise
    from gpurec.optim.optimize import optimize, final_eval
    model, genes = _common.build_model(args)
    if args.init_rate is not None:
        _set_init_rate(model, args.init_rate)
    theta_hat, _hist = optimize(model.batch_statics, model.theta.detach(),
                                model.receiver_weights.detach(),
                                optimizer="adam", schedule="adaptive", max_steps=args.steps)
    loss_bits, gnorm = final_eval(model.batch_statics, theta_hat, model.receiver_weights.detach())
    loss_bits = float(loss_bits)
    nll_nats = _common.bits_to_nats(loss_bits)
    elapsed = time.perf_counter() - t0
    print(f"fit({args.mode}): final NLL = {nll_nats:.6f} nats "
          f"(grad_norm {float(gnorm):.2e}, {elapsed:.1f}s)")
    if args.out:
        S = model.theta.shape[0] if model.theta.dim() > 1 else 1
        _write_outputs(args.out, _global_node_rates(theta_hat, args.mode, S),
                       loss_bits, nll_nats, elapsed, args.mode, len(genes))
    return 0
```

- [ ] **Step 4: Run the CPU writer test**

Run: `.venv/bin/python -m pytest tests/test_cli.py::test_write_outputs_alerax_format_and_sidecar -v -W error`
Expected: PASS.

- [ ] **Step 5: Add a GPU fit smoke test**

Append to `tests/test_cli.py`:
```python
@pytest.mark.gpu
def test_fit_global_reduces_nll_gpu(tmp_path):
    from pathlib import Path
    data = Path(__file__).parent / "data" / "alerax" / "test_mixed_200"
    if not data.exists():
        pytest.skip("fixtures not vendored yet (Task 5)")
    out = tmp_path / "fit_rates.txt"
    rc = main(["fit", "--species", str(data / "sp.nwk"), "--gene", str(data / "g.nwk"),
               "--mode", "global", "--steps", "5", "--device", "cuda", "--out", str(out)])
    assert rc == 0
    import json
    side = json.loads((tmp_path / "fit_rates.txt.json").read_text())
    assert math.isfinite(side["nll_nats"]) and side["mode"] == "global"
```

- [ ] **Step 6: Run the full CLI CPU suite**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v -m "not gpu"`
Expected: PASS (all CPU tests).

- [ ] **Step 7: Commit**

```bash
git add gpurec/cli/fit.py tests/test_cli.py
git commit -m "feat(cli): gpurec fit subcommand (global/specieswise/genewise)"
```

---

### Task 4: Fidelity harness (`gpurec/bench/fidelity.py`)

**Files:**
- Create: `gpurec/bench/fidelity.py`
- Modify: `gpurec/bench/__init__.py` (add exports)
- Test: `tests/test_fidelity_compare.py` (CPU: `compare`)

**Interfaces:**
- Consumes: `gpurec.bench.alerax_io.norm_family_name`; base `GeneReconModel`.
- Produces:
  - `FidelityReport = namedtuple(..., ["n_matched","mean_abs_delta","rmse","max_abs_delta","total_delta","pearson_r"])`
  - `compare(gpurec_ll: dict, alerax_ll: dict) -> FidelityReport`
  - `reconcile_at_alerax_rates(species, genes, rates, *, device="cuda", dtype=None, mode="global", solver_options=None) -> dict[str, float]` — `rates` is `AleraxRates`/`(D,L,T)`.

- [ ] **Step 1: Write the failing CPU test for `compare`**

Create `tests/test_fidelity_compare.py`:
```python
import math
from gpurec.bench.fidelity import compare, FidelityReport


def test_compare_matches_and_deltas():
    rep = compare({"a.ale": -2.0, "b": -3.0}, {"a": -2.0005, "b.txt": -2.9990})
    assert isinstance(rep, FidelityReport)
    assert rep.n_matched == 2
    assert rep.max_abs_delta < 1e-3 + 1e-9
    assert rep.pearson_r is None            # n <= 2 -> undefined


def test_compare_no_overlap():
    rep = compare({"x": -1.0}, {"y": -1.0})
    assert rep.n_matched == 0
    assert math.isnan(rep.mean_abs_delta)


def test_compare_pearson_when_enough_points():
    g = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
    a = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0}
    rep = compare(g, a)
    assert rep.n_matched == 4
    assert rep.pearson_r is not None and rep.pearson_r > 0.999
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fidelity_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpurec.bench.fidelity'`.

- [ ] **Step 3: Implement `gpurec/bench/fidelity.py`**

Create `gpurec/bench/fidelity.py`:
```python
"""Run gpurec at AleRax's fitted rates and compare per-family log-likelihood.

``compare`` is pure (CPU). ``reconcile_at_alerax_rates`` builds a GeneReconModel and
needs Triton/CUDA. All log-likelihoods are NATS.
"""
from __future__ import annotations

import math
from collections import namedtuple

FidelityReport = namedtuple(
    "FidelityReport",
    ["n_matched", "mean_abs_delta", "rmse", "max_abs_delta", "total_delta", "pearson_r"],
)


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def compare(gpurec_ll: dict, alerax_ll: dict) -> FidelityReport:
    from gpurec.bench.alerax_io import norm_family_name
    g = {norm_family_name(k): v for k, v in gpurec_ll.items()}
    a = {norm_family_name(k): v for k, v in alerax_ll.items()}
    keys = sorted(set(g) & set(a))
    n = len(keys)
    if n == 0:
        nan = float("nan")
        return FidelityReport(0, nan, nan, nan, nan, None)
    deltas = [g[k] - a[k] for k in keys]
    abs_d = [abs(x) for x in deltas]
    r = _pearson([g[k] for k in keys], [a[k] for k in keys]) if n > 2 else None
    return FidelityReport(n, sum(abs_d) / n, math.sqrt(sum(x * x for x in deltas) / n),
                          max(abs_d), sum(deltas), r)


def reconcile_at_alerax_rates(species, genes, rates, *, device="cuda", dtype=None,
                              mode="global", solver_options=None) -> dict:
    """rates: AleraxRates/(D, L, T) in AleRax order. Returns {family: logL_nats}."""
    import torch
    from gpurec.api.model import GeneReconModel
    from gpurec.bench.alerax_io import norm_family_name

    if dtype is None:
        dtype = torch.float64
    D, L, T = float(rates[0]), float(rates[1]), float(rates[2])
    gene_list = genes if isinstance(genes, list) else [genes]
    model = GeneReconModel(species, gene_list, mode=mode, device=device,
                           solver_options=solver_options)
    # gpurec theta order is [log2 D, log2 L, log2 T] == AleRax (D, L, T) column order
    triple = torch.tensor([math.log2(D), math.log2(L), math.log2(T)],
                          dtype=model.theta.dtype, device=model.theta.device)
    with torch.no_grad():
        if model.theta.dim() == 1:
            model.theta.copy_(triple)
        else:
            model.theta.copy_(triple.expand_as(model.theta))
        loss_bits = float(model())
    fam = norm_family_name(gene_list[0])
    return {fam: -loss_bits * math.log(2.0)}
```

Add to `gpurec/bench/__init__.py` imports + `__all__`:
```python
from gpurec.bench.fidelity import FidelityReport, compare, reconcile_at_alerax_rates
```
(append `"FidelityReport", "compare", "reconcile_at_alerax_rates"` to `__all__`.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fidelity_compare.py -v -W error`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add gpurec/bench/fidelity.py gpurec/bench/__init__.py tests/test_fidelity_compare.py
git commit -m "feat(bench): fidelity comparator + reconcile-at-AleRax-rates harness"
```

---

### Task 5: Vendor AleRax fixtures + un-ignore `tests/data`

**Files:**
- Create: `tests/data/alerax/<fixture>/{sp.nwk,g.nwk,families.txt,per_fam_likelihoods.txt,model_parameters.txt}` for the 5 fixtures; `tests/data/archaea60/reference_species_tree.newick`
- Modify: `.gitignore` (line 23 `tests/data/`)
- Test: `tests/test_fixtures.py` (CPU: parse real fixture files, spot-check oracle values)

**Interfaces:**
- Consumes: Task 1 parsers.
- Produces: on-disk fixtures at `tests/data/alerax/` consumed by Task 6.

- [ ] **Step 1: Copy the fixture text assets**

Run (from the base repo root):
```bash
SRC=/home/enzo/Documents/git/gpurec/gergely_version/tests/data
DST=/home/enzo/Documents/git/gpurec/consolidate-release/tests/data/alerax
for f in test_trees_1 test_trees_2 test_trees_3 test_trees_200 test_mixed_200; do
  mkdir -p "$DST/$f"
  cp "$SRC/$f/sp.nwk" "$SRC/$f/g.nwk" "$SRC/$f/families.txt" "$DST/$f/"
  cp "$SRC/$f/output/per_fam_likelihoods.txt" "$DST/$f/"
  cp "$SRC/$f/output/model_parameters/model_parameters.txt" "$DST/$f/"
done
mkdir -p /home/enzo/Documents/git/gpurec/consolidate-release/tests/data/archaea60
cp /home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_archaea_davin2017/species_reference/reference_species_tree.newick \
   /home/enzo/Documents/git/gpurec/consolidate-release/tests/data/archaea60/reference_species_tree.newick
```
Expected: 5 fixture dirs each with 5 files + 1 archaea60 tree. Verify:
`find tests/data -type f | sort` shows 26 files.

- [ ] **Step 2: Un-ignore the vendored fixtures**

In `.gitignore`, replace the line `tests/data/` with:
```
tests/data/*
!tests/data/alerax/
!tests/data/archaea60/
```
(`tests/data/*` keeps ignoring arbitrary future content under `tests/data` while the two `!` re-includes make the vendored fixture subtrees tracked; a bare `tests/data/` would prevent git from descending, so the `/*`+`!` form is required.)

Verify the fixtures are now trackable:
`git check-ignore -v tests/data/alerax/test_trees_1/sp.nwk` → prints nothing (not ignored); exit 1.

- [ ] **Step 3: Write the real-file parser test**

Create `tests/test_fixtures.py`:
```python
import pytest
from pathlib import Path
from gpurec.bench.alerax_io import parse_alerax_likelihoods, parse_alerax_parameters, global_rates

DATA = Path(__file__).parent / "data" / "alerax"
EXPECTED_LL = {
    "test_trees_1": -2.56495, "test_trees_2": -8.72486, "test_trees_3": -6.75086,
    "test_trees_200": -5.98394, "test_mixed_200": -6215.73,
}


@pytest.mark.parametrize("fixture,expected", list(EXPECTED_LL.items()))
def test_fixture_likelihood_value(fixture, expected):
    ll = parse_alerax_likelihoods(DATA / fixture / "per_fam_likelihoods.txt")
    assert ll["my_family"] == pytest.approx(expected)


@pytest.mark.parametrize("fixture", list(EXPECTED_LL))
def test_fixture_parameters_are_global(fixture):
    r = global_rates(parse_alerax_parameters(DATA / fixture / "model_parameters.txt"))
    assert r.D > 0 and r.L > 0 and r.T > 0


def test_archaea60_species_tree_present():
    tree = DATA.parent / "archaea60" / "reference_species_tree.newick"
    assert tree.stat().st_size > 0
    assert tree.read_text().count(",") >= 59        # >=60 tips -> >=59 commas
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fixtures.py -v -W error`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add .gitignore tests/data/alerax tests/data/archaea60 tests/test_fixtures.py
git commit -m "test(data): vendor AleRax toy fixtures + archaea60 species tree; un-ignore tests/data"
```

---

### Task 6: Build AleRax_fixed + generate reference likelihoods (`ref/`)

**Files:**
- Create: `benchmark/regen_alerax_refs.sh`
- Create (data): `tests/data/alerax/<fixture>/ref/{per_fam_likelihoods.txt,model_parameters.txt}` (5 fixtures)
- Create (OUTSIDE the repo, NOT committed): `/home/enzo/Documents/git/gpurec/tools/AleRax_fixed/` (AleRax build)
- Test: `tests/test_alerax_convention.py` (CPU)

**Interfaces:**
- Consumes: Task 5 shipped fixtures (`sp.nwk`/`g.nwk`/`families.txt`), Task 1 `parse_alerax_likelihoods`.
- Produces: `tests/data/alerax/<fixture>/ref/` references (the AleRax_fixed convention gpurec shares), consumed by Task 7's gate.

**Why:** the shipped `per_fam_likelihoods.txt` (stock AleRax v1.3.0) omit the `−ln(2S−1)` origination branch-count term, so each family is off by `+ln(2S−1)` from gpurec's likelihood (`test_trees_1`, S=8: shipped −2.56495 vs gpurec/AleRax_fixed −5.27300; ln(2·8−1)=ln 15=2.70805). `AleRax_fixed` computes gpurec's convention (the AleRax supplementary formula `L = (Σ pᴼ Π_Γ)/(Σ pᴼ(1−E))`), so its output is the correct oracle. The stock-vs-fixed gap here is the `ln(2S−1)` term, **not** convergence — at these fixtures' low/moderate fitted rates the value is flat across `--fixed-point-iterations` (verified: test_mixed_200 fitted D≈0.16/T≈0.16 gives −6221.02 at both 4 and 16 iters, and −6221.02 = stock −6215.73 − ln 199). **But convergence is rate-dependent:** at high rates AleRax's default 4 iterations under-converge badly (at fixed D/L/T = 1.5/1.6/1.7 the default is −8736.45 vs the converged −8649.06, an 87-nat gap that gpurec reproduces only against the converged value). So the regen script pins `--fixed-point-iterations 64` to guarantee converged references in any rate regime — cheap insurance the current fixtures don't strictly need but future higher-rate references would. (Full reproducible evidence for all these numbers: `experiments/alerax_convergence/` — `reproduce.py` re-runs both engines and self-checks; `README.md` writes it up.)

- [ ] **Step 1: Write the regeneration script**

Create `benchmark/regen_alerax_refs.sh`:
```bash
#!/usr/bin/env bash
# Generate AleRax_fixed reference likelihoods for the toy fixtures. AleRax_fixed computes
# gpurec's likelihood convention (includes the -ln(2S-1) origination term the stock v1.3.0
# shipped numbers omit). Copies + builds AleRax_fixed to a stable location OUTSIDE the tracked
# repo, runs each fixture, and vendors ref/{per_fam_likelihoods.txt,model_parameters.txt}.
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DATA="$REPO/tests/data/alerax"
ALERAX_SRC="${ALERAX_SRC:-$(ls -d /home/enzo/Documents/git/gpurec/agent-worktrees/*/AleRax_fixed/AleRax 2>/dev/null | head -1)}"
ALERAX_DST="${ALERAX_DST:-/home/enzo/Documents/git/gpurec/tools/AleRax_fixed}"
BIN="$ALERAX_DST/build/bin/alerax"

# 1) copy + build (idempotent; build lives OUTSIDE the tracked repo)
if [ ! -x "$BIN" ]; then
  [ -n "$ALERAX_SRC" ] || { echo "AleRax_fixed source not found; set ALERAX_SRC"; exit 1; }
  mkdir -p "$ALERAX_DST"
  rsync -a --exclude build --exclude .git "$ALERAX_SRC/" "$ALERAX_DST/"
  mkdir -p "$ALERAX_DST/build"
  ( cd "$ALERAX_DST/build" && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)" )
fi
# confirm this is AleRax_fixed (has the flag), not stock — we DO pass it (see below)
"$BIN" --help 2>&1 | grep -q "fixed-point-iterations" || { echo "not AleRax_fixed (no flag)"; exit 1; }

# 2) run each fixture at 64 fixed-point iterations (converged for any rate regime; the default 4
#    under-converges at high rates), vendor ref/
ITERS="${ITERS:-64}"
for f in test_trees_1 test_trees_2 test_trees_3 test_trees_200 test_mixed_200; do
  d="$DATA/$f"
  work="$(mktemp -d)"
  cp "$d/sp.nwk" "$d/g.nwk" "$d/families.txt" "$work/"
  ( cd "$work" && "$BIN" -f families.txt -s sp.nwk -p out --gene-tree-samples 0 \
      --species-tree-search SKIP --fixed-point-iterations "$ITERS" >alerax.log 2>&1 )
  mkdir -p "$d/ref"
  cp "$work/out/per_fam_likelihoods.txt" "$d/ref/"
  cp "$work/out/model_parameters/model_parameters.txt" "$d/ref/"
  echo "$f -> $(cat "$d/ref/per_fam_likelihoods.txt")"
  rm -rf "$work"
done
```

- [ ] **Step 2: Run it to generate the references**

Run: `chmod +x benchmark/regen_alerax_refs.sh && bash benchmark/regen_alerax_refs.sh`
Expected: builds AleRax_fixed once, then prints one logL per fixture, e.g. `test_trees_1 -> my_family -5.27300` (the shipped stock value is −2.56495; the two differ by ln 15 = 2.70805). Creates `tests/data/alerax/<fixture>/ref/` for all 5.

- [ ] **Step 3: Write the convention test**

Create `tests/test_alerax_convention.py`:
```python
import math
from pathlib import Path
from gpurec.bench.alerax_io import parse_alerax_likelihoods

DATA = Path(__file__).parent / "data" / "alerax"
FIXTURES = ["test_trees_1", "test_trees_2", "test_trees_3", "test_trees_200", "test_mixed_200"]


def _ll(path):
    return parse_alerax_likelihoods(path)["my_family"]


def _n_species(fixture):
    # rooted binary Newick with S leaves has S-1 commas
    return (DATA / fixture / "sp.nwk").read_text().count(",") + 1


def test_stock_equals_fixed_plus_branch_term():
    """Shipped stock AleRax == AleRax_fixed + ln(2S-1) (the origination branch-count term)."""
    for f in FIXTURES:
        stock = _ll(DATA / f / "per_fam_likelihoods.txt")
        ref = _ll(DATA / f / "ref" / "per_fam_likelihoods.txt")
        S = _n_species(f)
        assert abs((stock - ref) - math.log(2 * S - 1)) < 1e-2, (
            f"{f}: stock={stock} ref={ref} S={S} ln(2S-1)={math.log(2*S-1)}")


def test_test_trees_1_known_values():
    assert _ll(DATA / "test_trees_1" / "per_fam_likelihoods.txt") == -2.56495
    assert abs(_ll(DATA / "test_trees_1" / "ref" / "per_fam_likelihoods.txt") - (-5.27300)) < 1e-3
```

- [ ] **Step 4: Run the convention test**

Run: `.venv/bin/python -m pytest tests/test_alerax_convention.py -v -W error`
Expected: PASS (2 tests) — documents that the shipped stock numbers differ from the AleRax_fixed references by exactly ln(2S−1) per family.

- [ ] **Step 5: Commit the refs + script + test (NOT the AleRax build)**

```bash
git add benchmark/regen_alerax_refs.sh tests/test_alerax_convention.py tests/data/alerax/*/ref
git commit -m "test(fidelity): generate AleRax_fixed refs; document +ln(2S-1) stock offset"
```
(The AleRax build under `/home/enzo/Documents/git/gpurec/tools/` is external — never `git add` it.)

---

### Task 7: Fidelity gate — gpurec vs AleRax_fixed references (GPU)

**Files:**
- Create: `tests/test_fidelity_alerax.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Consumes: Task 1 parsers, Task 4 `reconcile_at_alerax_rates`/`compare`, Task 5 fixtures, Task 6 `ref/` references; base `GeneReconModel`.

- [ ] **Step 1: Write the fidelity gate**

Create `tests/test_fidelity_alerax.py`:
```python
import pytest
from pathlib import Path
from gpurec.bench.alerax_io import parse_alerax_likelihoods, parse_alerax_parameters, global_rates
from gpurec.bench.fidelity import reconcile_at_alerax_rates, compare

DATA = Path(__file__).parent / "data" / "alerax"
FIXTURES = ["test_trees_1", "test_trees_2", "test_trees_3", "test_trees_200", "test_mixed_200"]
TOL_NATS = 1e-3


@pytest.mark.gpu
@pytest.mark.parametrize("fixture", FIXTURES)
def test_fidelity_matches_alerax_fixed(fixture):
    import torch
    ref = DATA / fixture / "ref"
    rates = global_rates(parse_alerax_parameters(ref / "model_parameters.txt"))
    alerax_ll = parse_alerax_likelihoods(ref / "per_fam_likelihoods.txt")
    gpurec_ll = reconcile_at_alerax_rates(
        str(DATA / fixture / "sp.nwk"), [str(DATA / fixture / "g.nwk")], rates,
        device="cuda", dtype=torch.float64, mode="global")
    rep = compare(gpurec_ll, alerax_ll)
    assert rep.n_matched == 1, f"{fixture}: name mismatch {gpurec_ll} vs {alerax_ll}"
    assert rep.mean_abs_delta <= TOL_NATS, (
        f"{fixture}: |Δ|={rep.mean_abs_delta:.3e} nats > {TOL_NATS} "
        f"(gpurec={gpurec_ll}, alerax_fixed={alerax_ll})")
```

- [ ] **Step 2: Run the gate — the substantive result of the batch**

Run: `.venv/bin/python -m pytest tests/test_fidelity_alerax.py -v -m gpu`
Expected: PASS (5 params). gpurec reproduces the AleRax_fixed per-family logL to ≤1e-3 nats (observed ~1e-5 on the toy fixtures during design), closing prerequisite #1 (base == AleRax) at toy scale against a correct-convention oracle.

**If any fixture fails:** DO NOT raise `TOL_NATS`. Record the per-fixture `mean_abs_delta` and both logL values from the assertion message and escalate as a finding. A ~20-nat miss on `test_trees_2` (the only fixture with transfer active) specifically means the theta order is wrong (T/L swapped — see Global Constraints). Any other real gap points at a genuine base-vs-AleRax model difference (rooting, origination normalization, CCP construction) worth surfacing precisely.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fidelity_alerax.py
git commit -m "test(fidelity): gpurec vs AleRax_fixed gate (<=1e-3 nats) — prerequisite #1 at toy scale"
```

---

### Task 8: Scale benchmark kit (`benchmark/`, cluster-run, CI-smoke only)

**Files:**
- Create: `benchmark/config.sh`, `benchmark/lib.sh`, `benchmark/bin/00_preflight.sh`, `benchmark/bin/40_run.sh`, `benchmark/bin/70_check_fidelity.sh`, `benchmark/bench_gpurec_fit.py`, `benchmark/eval_at_alerax_rates.py`, `benchmark/slurm/fidelity_saion.sbatch`, `benchmark/README.md`
- Test: `tests/test_benchmark_smoke.py` (CPU)

**Interfaces:**
- Consumes: the `gpurec` CLI (Task 2/3), `gpurec.bench.fidelity` (Task 4), `gpurec.optim.optimize`.
- Produces: a runnable-on-Saion scale kit + a CI smoke that only checks scripts parse and Python drivers import.

**Scope note (YAGNI):** this is a faithful-but-slimmed port of gergely's `benchmark/` — the honest, base-wired subset (preflight, run, fidelity, one Python fit driver, one Python eval driver, one slurm template). It is NOT executed in CI (no Williams data locally; archaea60 has no AleRax reference). Only the CPU smoke below runs.

- [ ] **Step 1: Write the failing CPU smoke test**

Create `tests/test_benchmark_smoke.py`:
```python
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).parent.parent / "benchmark"


def test_shell_scripts_parse():
    for sh in ["config.sh", "lib.sh", "bin/00_preflight.sh", "bin/40_run.sh",
               "bin/70_check_fidelity.sh"]:
        r = subprocess.run(["bash", "-n", str(BENCH / sh)], capture_output=True, text=True)
        assert r.returncode == 0, f"{sh}: {r.stderr}"


def test_python_drivers_help():
    for py in ["bench_gpurec_fit.py", "eval_at_alerax_rates.py"]:
        r = subprocess.run([sys.executable, str(BENCH / py), "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{py}: {r.stderr}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_benchmark_smoke.py -v`
Expected: FAIL — `benchmark/` files do not exist.

- [ ] **Step 3: Create the Python fit driver**

Create `benchmark/bench_gpurec_fit.py`:
```python
#!/usr/bin/env python
"""Fit gpurec DTL rates on a dataset and write rates (AleRax format) + a JSON sidecar.

Thin wrapper over the `gpurec fit` library path so the scale kit and the CLI share code.
"""
import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(description="gpurec rate-fit benchmark driver")
    p.add_argument("--species", required=True)
    p.add_argument("--gene", required=True, nargs="+")
    p.add_argument("--mode", choices=["global", "specieswise", "genewise"], default="global")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from gpurec.cli.fit import run_fit
    return run_fit(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Create the Python fidelity-eval driver**

Create `benchmark/eval_at_alerax_rates.py`:
```python
#!/usr/bin/env python
"""Evaluate gpurec per-family logL at AleRax's fitted GLOBAL rates and report fidelity.

Reads an AleRax run directory (model_parameters.txt + per_fam_likelihoods.txt), runs
gpurec at those rates over the given families, and prints matched count, mean/RMSE/max
|Δ| (nats) and Pearson r.
"""
import argparse
import sys
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description="gpurec-vs-AleRax global fidelity")
    p.add_argument("--species", required=True)
    p.add_argument("--gene", required=True, nargs="+")
    p.add_argument("--alerax-dir", required=True,
                   help="AleRax output dir with model_parameters/ and per_fam_likelihoods.txt")
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from gpurec.bench.alerax_io import (parse_alerax_likelihoods, parse_alerax_parameters,
                                        global_rates)
    from gpurec.bench.fidelity import reconcile_at_alerax_rates, compare
    d = Path(args.alerax_dir)
    rates = global_rates(parse_alerax_parameters(d / "model_parameters" / "model_parameters.txt"))
    alerax_ll = parse_alerax_likelihoods(d / "per_fam_likelihoods.txt")
    gpurec_ll = {}
    for g in args.gene:
        gpurec_ll.update(reconcile_at_alerax_rates(args.species, [g], rates, device=args.device))
    rep = compare(gpurec_ll, alerax_ll)
    print(f"matched={rep.n_matched} mean|Δ|={rep.mean_abs_delta:.3e} rmse={rep.rmse:.3e} "
          f"max|Δ|={rep.max_abs_delta:.3e} pearson_r={rep.pearson_r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Create the shell orchestration**

Create `benchmark/config.sh`:
```bash
#!/usr/bin/env bash
# Scale-benchmark configuration. Edit for your cluster/dataset.
set -euo pipefail

# Dataset selector: "williams" (Zenodo download) or "archaea" (local sibling repo).
DATASET="${DATASET:-archaea}"

# archaea60 lives in a sibling repo; the 60-taxon tree is vendored into the base.
ARCHAEA_DATA_DIR="${ARCHAEA_DATA_DIR:-/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_archaea_davin2017}"
ARCHAEA_SPECIES_TREE="${ARCHAEA_SPECIES_TREE:-$(git rev-parse --show-toplevel)/tests/data/archaea60/reference_species_tree.newick}"

# Williams et al. 2017 (headline fidelity dataset) is Zenodo-only.
ZENODO_TARBALL_URL="${ZENODO_TARBALL_URL:-https://zenodo.org/record/17360806/files/williams2017.tar.gz}"

# gpurec mode -> AleRax parametrization.
MODE="${MODE:-global}"          # global | specieswise | genewise
case "$MODE" in
  global)      ALERAX_PARAM="GLOBAL" ;;
  specieswise) ALERAX_PARAM="PER-SPECIES" ;;
  genewise)    ALERAX_PARAM="PER-FAMILY" ;;
esac

RESULTS_DIR="${RESULTS_DIR:-$(git rev-parse --show-toplevel)/benchmark/results}"
PY="${PY:-python}"
```

Create `benchmark/lib.sh`:
```bash
#!/usr/bin/env bash
# Shared helpers for the benchmark scripts.
set -euo pipefail

log() { printf '[bench] %s\n' "$*" >&2; }

require_file() { [ -f "$1" ] || { log "missing file: $1"; exit 1; }; }
```

Create `benchmark/bin/00_preflight.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

log "dataset=$DATASET mode=$MODE alerax_param=$ALERAX_PARAM"
"$PY" -c "import gpurec; print('gpurec import OK')"
if [ "$DATASET" = "archaea" ]; then
  require_file "$ARCHAEA_SPECIES_TREE"
  [ -d "$ARCHAEA_DATA_DIR" ] || { log "archaea data dir absent: $ARCHAEA_DATA_DIR"; exit 1; }
  log "archaea60 ready: $(find "$ARCHAEA_DATA_DIR/ale_gene_tree_distributions" -name '*.ale' | wc -l) .ale families"
else
  log "williams: fetch $ZENODO_TARBALL_URL (not automated here)"
fi
```

Create `benchmark/bin/40_run.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"
mkdir -p "$RESULTS_DIR"

if [ "$DATASET" != "archaea" ]; then log "only archaea wired for 40_run"; exit 1; fi
GENES="$ARCHAEA_DATA_DIR/ale_gene_tree_distributions/main_families_ge4seq"
log "fitting gpurec ($MODE) on archaea60 main families -> $RESULTS_DIR/rates.txt"
"$PY" "$HERE/bench_gpurec_fit.py" --species "$ARCHAEA_SPECIES_TREE" \
  --gene "$GENES" --mode "$MODE" --steps 300 --out "$RESULTS_DIR/rates.txt"
```

Create `benchmark/bin/70_check_fidelity.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

# Requires an AleRax output dir ($ALERAX_DIR) to correlate against.
: "${ALERAX_DIR:?set ALERAX_DIR to an AleRax output directory}"
GENES="${GENES:?set GENES to the gene-tree files/dir}"
log "checking gpurec-vs-AleRax fidelity against $ALERAX_DIR"
"$PY" "$HERE/eval_at_alerax_rates.py" --species "$ARCHAEA_SPECIES_TREE" \
  --gene $GENES --alerax-dir "$ALERAX_DIR"
```

Create `benchmark/slurm/fidelity_saion.sbatch`:
```bash
#!/usr/bin/env bash
#SBATCH --job-name=gpurec-fidelity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate 2>/dev/null || true
DATASET=archaea MODE=global bash benchmark/bin/70_check_fidelity.sh
```

Create `benchmark/README.md`:
```markdown
# gpurec scale benchmark kit

Runnable on a GPU cluster (Saion A100). NOT run in CI — Williams 2017 is a Zenodo
download and archaea60 ships no AleRax reference (only ALEml `.uml_rec`).

## Layout
- `config.sh` — dataset/mode/paths (`DATASET=archaea|williams`, `MODE=global|specieswise|genewise`).
- `bin/00_preflight.sh` — import + data checks.
- `bin/40_run.sh` — `gpurec fit` over a dataset -> `results/rates.txt` + `.json` sidecar.
- `bin/70_check_fidelity.sh` — correlate gpurec per-family logL at AleRax rates vs an AleRax output dir.
- `bench_gpurec_fit.py` / `eval_at_alerax_rates.py` — the Python drivers (share code with `gpurec.cli`/`gpurec.bench`).
- `slurm/fidelity_saion.sbatch` — example A100 job.

## Fidelity semantics
The in-repo CI gate (`tests/test_fidelity_alerax.py`) proves base==AleRax at toy scale
(≤1e-3 nats) against the **AleRax_fixed** references (`tests/data/alerax/*/ref/`, generated by
`benchmark/regen_alerax_refs.sh`). This kit reproduces the paper's large-scale Pearson-r
headline on the cluster, where thousands of families make correlation meaningful.

**Important:** the shipped stock AleRax v1.3.0 numbers omit the `−ln(2S−1)` origination
branch-count term, so they are off by `+ln(2S−1)` per family from gpurec's convention. Any
AleRax run used as a fidelity reference here MUST be produced by `AleRax_fixed` (which computes
the same supplementary-formula likelihood) with a high `--fixed-point-iterations` (≥16; the toy
regen pins 64 — the default 4 under-converges at high rates) — otherwise the correlation is against
a biased target.

## archaea60
Data: `$ARCHAEA_DATA_DIR` (sibling repo). Species tree vendored at
`tests/data/archaea60/reference_species_tree.newick`. It has NO AleRax reference; use it as
a runtime/scale target for `fit`, or supply your own AleRax output dir to `70_check_fidelity.sh`.
```

Make scripts executable:
```bash
chmod +x benchmark/bin/*.sh benchmark/*.py benchmark/*.sh benchmark/slurm/*.sbatch
```

- [ ] **Step 6: Run the CPU smoke test**

Run: `.venv/bin/python -m pytest tests/test_benchmark_smoke.py -v`
Expected: PASS (2 tests — scripts parse, drivers `--help`).

- [ ] **Step 7: Commit**

```bash
git add benchmark tests/test_benchmark_smoke.py
git commit -m "feat(benchmark): scale kit (archaea60/williams) wired to gpurec CLI + bench helpers"
```

---

### Task 9: Docs, exports & full-suite verification

**Files:**
- Create: `docs/cli.md`, `docs/benchmark.md`
- Test: `tests/test_bench_exports.py`

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Write the export test**

Create `tests/test_bench_exports.py`:
```python
def test_bench_public_exports():
    import gpurec.bench as b
    for name in ["AleraxRates", "parse_alerax_likelihoods", "parse_alerax_parameters",
                 "norm_family_name", "global_rates", "FidelityReport", "compare",
                 "reconcile_at_alerax_rates"]:
        assert hasattr(b, name), name


def test_cli_entrypoint_importable():
    from gpurec.cli.main import main, build_parser
    assert callable(main) and build_parser().prog == "gpurec"
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bench_exports.py -v -W error`
Expected: PASS (2 tests).

- [ ] **Step 3: Write `docs/cli.md`**

Create `docs/cli.md`:
```markdown
# gpurec CLI

Installed as the `gpurec` console script (`pip install -e .`). Requires CUDA + Triton
for the forward path (`--help` and argument parsing do not).

## reconcile — log-likelihood at fixed rates
```
gpurec reconcile --species sp.nwk --gene g.nwk [g2.nwk ...] \
  --delta 0.1 --tau 0.05 --lambda 0.1 [--mode global|genewise] \
  [--device cuda] [--dtype float64] [--out per_fam.txt]
```
Prints `<family> <logL_nats>`. `global` mode prints one total line; `genewise` prints one
line per family. `--out` writes the AleRax `per_fam_likelihoods.txt` format.

**Units:** log-likelihood is reported in **nats** (AleRax-comparable). Internally the model
returns NLL in bits; the CLI converts (`logL_nats = -loss_bits * ln 2`).

**Rate order:** `--delta/--tau/--lambda` are D/T/L; `theta = [log₂ D, log₂ L, log₂ T]` (theta[2] = transfer; same column order as AleRax `D L T`).

## fit — optimize rates
```
gpurec fit --species sp.nwk --gene DIR_OR_GLOB \
  --mode global|specieswise|genewise [--steps 300] [--init-rate 0.1] \
  [--device cuda] [--dtype float64] [--out rates.txt]
```
`global`/`specieswise` use `gpurec.optim.optimize` + `final_eval`; `genewise` uses
`fit_genewise`. `--out` writes fitted rates (AleRax `# node D L T` order) plus a
`<out>.json` sidecar (`nll_bits`, `nll_nats`, `elapsed_s`, `mode`, `n_families`).
Specieswise rows are labelled `s{i}` by species index (mapping to AleRax node labels is
the scale kit's job).
```

- [ ] **Step 4: Write `docs/benchmark.md`**

Create `docs/benchmark.md`:
```markdown
# AleRax fidelity & scale benchmark

## Why AleRax's shipped numbers are not the reference
Stock AleRax v1.3.0 omits the `−ln(2S−1)` origination branch-count term when forming L from
Π and E, so each shipped `per_fam_likelihoods.txt` value is off by `+ln(2S−1)` from gpurec's
likelihood (`test_trees_1`, S=8: shipped −2.56495 vs correct −5.27300; ln 15 = 2.70805). This
ln(2S−1) gap is a normalization difference, not convergence (flat across iterations at the
fixtures' low fitted rates). Convergence is nonetheless **rate-dependent** — at high rates AleRax's
default 4 iterations under-converge (measured 87 nats at fixed D/L/T=1.5/1.6/1.7 on test_mixed_200),
so the regen script pins `--fixed-point-iterations 64`. `benchmark/regen_alerax_refs.sh` rebuilds
`AleRax_fixed` (which computes the AleRax supplementary-formula likelihood gpurec shares) and
generates references under `tests/data/alerax/*/ref/`.

## In-repo fidelity gates (CI, GPU)
- `tests/test_fidelity_alerax.py` — the gate: `gpurec reconcile` at each fixture's AleRax_fixed
  rates matches the AleRax_fixed per-family logL ≤ 1e-3 nats (observed ~1e-5). The first
  checked-in proof that the base computes AleRax's likelihood — "prerequisite #1" — at toy scale.
- `tests/test_alerax_convention.py` (CPU) — documents that the shipped stock numbers differ from
  the AleRax_fixed references by exactly `ln(2S−1)` per family.

Parsers live in `gpurec/bench/alerax_io.py`; the harness in `gpurec/bench/fidelity.py`.

## Scale kit (cluster, not CI)
`benchmark/` reproduces the paper's large-scale fidelity/speed numbers on Saion. See
`benchmark/README.md`. Williams 2017 is a Zenodo download; archaea60 is local (sibling repo)
but ships only an ALEml reference, so it is a runtime/scale target unless you supply an
AleRax output directory.

## Fixtures
Vendored under `tests/data/alerax/<fixture>/` (text only — `sp.nwk`, `g.nwk`, `families.txt`,
shipped stock `per_fam_likelihoods.txt`/`model_parameters.txt`, and `ref/` with the AleRax_fixed
references). The 60-taxon archaea species tree is at
`tests/data/archaea60/reference_species_tree.newick`.
```

- [ ] **Step 5: Run the full CPU + GPU suites**

Run: `.venv/bin/python -m pytest tests/ -m "not gpu" -q`
Expected: all CPU tests pass (existing base tests + the new `test_alerax_io`, `test_cli` CPU, `test_fidelity_compare`, `test_fixtures`, `test_alerax_convention`, `test_benchmark_smoke`, `test_bench_exports`).

Run: `.venv/bin/python -m pytest tests/ -m gpu -q`
Expected: GPU tests pass, including `test_fidelity_alerax` (5 params), the reconcile/fit smokes, and the base's existing GPU tests. If any `test_fidelity_alerax` param fails, apply the Task-7 guidance (record the delta, escalate — do not loosen the tolerance).

- [ ] **Step 6: Commit**

```bash
git add docs/cli.md docs/benchmark.md tests/test_bench_exports.py
git commit -m "docs(cli,bench): CLI + benchmark usage; bench export test; full-suite verified"
```

---

## Plan self-review notes

- **Spec coverage:** CLI reconcile (Task 2) + fit all-modes (Task 3); parsers (Task 1); fidelity harness (Task 4); vendored shipped fixtures + gitignore (Task 5); AleRax_fixed build + `ref/` references (Task 6); direct fidelity gate vs AleRax_fixed (Task 7); scale kit incl. archaea60 config (Task 8); docs + exports + verification (Task 9). All spec sections mapped.
- **Units/order:** every logL conversion is `-loss_bits * ln2`; gpurec theta and AleRax columns are BOTH `D L T`, so rates map straight through (no reorder), round-trip tested (Tasks 1, 3).
- **Lazy imports:** no `gpurec/cli/*` module imports torch/GeneReconModel at top level; CPU tests (Tasks 2, 3) exercise `--help`/parsing without the forward path.
- **No-op on existing paths:** only additive files + one `pyproject.toml` scripts line + one `.gitignore` change. The AleRax build (Task 6) lives outside the tracked repo and is never staged.
- **Reference oracle:** the gate (Task 7) runs against AleRax_fixed references (Task 6) — the −ln(2S−1) convention gpurec shares — never the shipped stock numbers (off by +ln(2S−1)).
- **Oracle-wins:** Task 7 forbids loosening the tolerance; a failure is a reported finding.
- **Type consistency:** `AleraxRates(D,L,T)` maps straight to `theta = [log₂ D, log₂ L, log₂ T]`, `FidelityReport` fields, and `_write_outputs(node,D,L,T)` are used identically across Tasks 1/3/4/6.
