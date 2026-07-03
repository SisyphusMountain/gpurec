# `gpurec` CLI + AleRax fidelity kit — Design Spec

> **Status:** approved 2026-07-02. Branch: `feat/cli-and-fidelity` (off `main` @ `2dee4d9c`).
> Base = `consolidate-release`. Source of algorithms/assets = `gergely_version` (reference,
> **not** a numeric oracle — the numeric oracle here is *AleRax itself*).

## Goal

Give the base two Tier-1 "publication-critical" capabilities it currently lacks:

1. **A `gpurec` command-line interface** with two subcommands — `reconcile` (compute the marginal
   log-likelihood of one or more gene families against a species tree at fixed DTL rates) and
   `fit` (optimize DTL rates in any of the three param modes) — so the tool has a one-line,
   reproducible entry point for the paper.
2. **An AleRax reference-fidelity kit** — vendored AleRax ground-truth fixtures + parsers + a
   runnable fidelity gate that proves the base computes the same per-family log-likelihood as
   AleRax, plus a scale benchmark kit (Williams 2017 / archaea60) wired to the base API for
   cluster runs.

Together these close the report's **prerequisite #1** ("no one has ever shown the base computes
the same likelihood as AleRax") at small scale, and scaffold the large-scale fidelity/speed
numbers the paper cites.

## Non-negotiable principles

- **Rebuild on the base's surface; do not port gergely's code.** The reference CLI targets the old
  `GeneDataset`/C++-JIT API and `experiments/run_undine_branchwise.py`; the base has none of that.
  The CLI is written against `GeneReconModel`, `SolverOptions`, `fit_genewise`,
  `gpurec.optim.optimize`, and `_resolve_gene_trees` — **base names only**.
- **Units are explicit and correct.** The base forward returns **NLL in bits (log₂)**. AleRax
  reference likelihoods are in **nats (ln)**. Every user-facing / comparison path converts:
  `logL_nats = -loss_bits * ln 2`. `theta` is log₂-rates in both trees.
- **Rate-component order is `[D, L, T]` — verified in code, matches AleRax.** gpurec
  `theta = [log₂ D, log₂ L, log₂ T]` — Duplication, **Loss, Transfer** (`theta[2]` is transfer;
  `extract_parameters.py` takes `log_pT = result[..., 3]` from `theta[2]`). AleRax
  `model_parameters.txt` columns are also `D L T`, so **AleRax rates map straight to gpurec theta
  with NO reordering.** CLI `--delta/--tau/--lambda` (D/T/L) → `theta = [log₂ D, log₂ L, log₂ T]`.
  (An earlier draft used `[D, T, L]`; the T/L swap is invisible whenever T=L and caused a spurious
  ~20-nat "transfer discrepancy" — see memory `gpurec-theta-order-dlt`.)
- **Heavy imports are lazy.** The CLI must run `--help`, argument parsing, and argument validation
  **without importing the Triton/CUDA forward path**, so those layers are unit-testable on a
  CPU/no-GPU box. `GeneReconModel` and optimizers are imported *inside* the command functions.
- **Stock AleRax's shipped likelihood uses a different normalization; the oracle is AleRax_fixed.**
  gpurec computes the AleRax supplementary-material likelihood
  `L = (Σ_s p^O_s Π_{Γ,s}) / (Σ_s p^O_s (1−E_s))` (verified in `solver.py:158–169`). Stock AleRax
  v1.3.0 (which produced the shipped `tests/data/**/output/`) omits the `−ln(2S−1)` origination
  branch-count term, so it differs from gpurec/AleRax_fixed by exactly `+ln(2S−1)` per family
  (verified across S=8 ×3 and S=200). **`AleRax_fixed`** (built from `agent-worktrees/*/AleRax_fixed`)
  restores that term and matches gpurec to ~1e-5 nats. So the fidelity reference is generated with
  **AleRax_fixed**, not the shipped stock numbers. (This has nothing to do with fixed-point
  iterations — values are flat across `--fixed-point-iterations` 1→4096. Whether stock under-converges
  on larger/transfer-heavy families is a separate, unverified follow-up.)
- **The fidelity gate is a direct, checked-in oracle.** For each fixture, gpurec `reconcile` at the
  fixture's rates matches the **AleRax_fixed** per-family logL to ≤1e-3 nats (observed ~1e-5). This
  is prerequisite-#1 closure at toy scale — already demonstrated during design. GPU-marked; must pass
  on this box.
- **No change to default numerics.** No edits to any existing forward/optimize/solver path. This
  batch only *adds* modules (`gpurec/cli/`, `gpurec/bench/`), tests, fixtures, a `benchmark/`
  directory, and a `[project.scripts]` line in `pyproject.toml`.
- **Do not touch pre-existing uncommitted work** (`crates/*/src/lib.rs`, `experiments/ghost_lineages/`,
  `ghost_experiments/`, `paper.pdf`). Never `git add -A`/`git add .`; stage only files each task creates.

## Scope decisions (resolved 2026-07-02)

- **Fidelity kit = in-hand gate + scaffolded scale kit.** Vendored toy-fixture gate runs in CI (on
  GPU); the `benchmark/` scale scripts (Williams via Zenodo, archaea60 via config path) are ported
  and wired to the base API, runnable on Saion, **not** executed in CI.
- **AleRax_fixed reference generated in-batch.** Copy `AleRax_fixed`
  (`agent-worktrees/*/AleRax_fixed/AleRax`) to a stable location
  `/home/enzo/Documents/git/gpurec/tools/AleRax_fixed/`, build it (cmake+make; toolchain verified,
  binary already builds), and run the 5 fixtures to generate `per_fam_likelihoods.txt` +
  `model_parameters.txt` in gpurec's convention, vendored under `tests/data/alerax/<fixture>/ref/`.
  Iteration count is irrelevant (values flat 1→4096); use the default. A committed
  `benchmark/regen_alerax_refs.sh` makes it reproducible. The AleRax build lives OUTSIDE the tracked
  repo (not committed).
- **Convergence check deferred.** No gpurec self-convergence / AleRax-under-convergence testing in
  this batch (it was a red herring for these fixtures); noted as a follow-up on larger families.
- **`fit` covers all three modes:** `global`/`specieswise` → `gpurec.optim.optimize` + `final_eval`;
  `genewise` → `fit_genewise`.
- **archaea60 = scale/runtime target.** Vendor the 60-taxon species tree into the base; point the
  scale kit at the sibling dataset (`/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_archaea_davin2017/`)
  via a config variable; use it as the `fit` runtime/scale benchmark (ms/family), with the ALEml
  `.uml_rec` `>logl` as an optional secondary reference. Its lack of an AleRax reference means it
  is **not** part of the AleRax fidelity gate.

## Module layout

| File | Status | Owns |
|---|---|---|
| `gpurec/cli/__init__.py` | **new** | package marker |
| `gpurec/cli/main.py` | **new** | `main(argv=None)`: argparse w/ `reconcile`/`fit` subparsers; dispatch; registered console entry |
| `gpurec/cli/_common.py` | **new** | shared arg group (`--species/--gene/--device/--dtype/--mode/--solver knobs`), `resolve_gene_trees`, `rates_to_theta`, `bits_to_nats`, model builder |
| `gpurec/cli/reconcile.py` | **new** | `run_reconcile(args)`: build model at fixed rates → per-family/total logL → stdout or `--out` (AleRax `per_fam_likelihoods.txt` format) |
| `gpurec/cli/fit.py` | **new** | `run_fit(args)`: mode-dispatched optimize → fitted rates (AleRax `# node D L T` format) + JSON sidecar (NLL bits+nats, elapsed) |
| `gpurec/bench/__init__.py` | **new** | package marker |
| `gpurec/bench/alerax_io.py` | **new** | pure-text parsers: `parse_alerax_likelihoods`, `parse_alerax_parameters`, `norm_family_name`; `AleraxRates` triple |
| `gpurec/bench/fidelity.py` | **new** | `reconcile_at_alerax_rates(species, genes, rates, ...) -> {family: logL_nats}` + `compare(gpurec, alerax) -> FidelityReport` (matched, Δ mean/rmse/max, Pearson r when >2 pts) |
| `pyproject.toml` | edit | add `[project.scripts] gpurec = "gpurec.cli.main:main"` |
| `.gitignore` | edit (if needed) | ensure `tests/data/alerax/` is not excluded (verify first — currently `tests/data` is NOT ignored) |
| `tests/data/alerax/<fixture>/…` | **new** | fixture inputs: `sp.nwk`, `g.nwk`, `families.txt` |
| `tests/data/alerax/<fixture>/ref/…` | **new** | AleRax_fixed reference (the gate oracle, gpurec's convention): `per_fam_likelihoods.txt`, `model_parameters.txt` |
| `tests/data/archaea60/reference_species_tree.newick` | **new** | vendored 60-taxon species tree (866 B) |
| `benchmark/regen_alerax_refs.sh` | **new** | reproducible: copy+build `AleRax_fixed`, run 5 fixtures → `ref/` |
| `tests/test_alerax_io.py` | **new** | CPU parser unit tests |
| `tests/test_cli.py` | **new** | CPU arg/dispatch/unit-conversion tests + GPU reconcile/fit smoke |
| `tests/test_fidelity_alerax.py` | **new** | GPU fidelity gate on vendored fixtures |
| `benchmark/` (repo root) | **new** | scale kit: `config.sh`, `lib.sh`, `bin/00..70`, `bench_gpurec_fit.py`, `slurm/*.sbatch` — wired to base API |
| `docs/cli.md`, `docs/benchmark.md` | **new** | usage docs |

## Data flow

```
CLI reconcile:
  --species,--gene,--delta/--tau/--lambda,--mode,--device,--dtype,--iters
    → _common.resolve_gene_trees / rates_to_theta / SolverOptions
    → GeneReconModel(species, genes, mode, device, solver_options)
    → set model.theta = [log2 D, log2 L, log2 T]   (theta[2] = transfer)
    → global: logL_total_nats = -float(model()) * ln2
      genewise: logL_per_family_nats = -model.genewise_loss_vector() * ln2
    → stdout ("<family> <logL_nats>") or --out (AleRax per_fam format)

CLI fit:
  --species,--gene,--mode,--steps,--device,--dtype,[--origination/reg knobs],--out
    → genewise: fit_genewise(...) -> {theta[F,3], rates, loss_nats}
      global/specieswise: optimize(model.batch_statics, theta0, receiver_weights, ...)
                          → final_eval(...) -> (loss_bits, gnorm)
    → write rates (AleRax "# node D L T") + JSON sidecar (nll_bits, nll_nats, elapsed_s)

Fidelity gate (test):
  fixture ref/ → parse_alerax_parameters(ref/model_parameters.txt) -> (D,L,T)
    → bench.fidelity.reconcile_at_alerax_rates(sp, g, rates)  [D,L,T maps straight to theta]
    → parse_alerax_likelihoods(ref/per_fam_likelihoods.txt) -> {fam: logL_nats}
    → compare(): assert per-fixture |Δ| ≤ 1e-3 nats  (observed ~1e-5)
```

## Component detail

### `gpurec/cli/_common.py`
- `add_common_args(parser)`: `--species` (required), `--gene` (required, `nargs="+"`; also accepts a
  dir/glob/AleRax listfile — resolved by `resolve_gene_trees`), `--mode {global,specieswise,genewise}`
  (default `global`), `--device {cpu,cuda}` (default `cuda`), `--dtype {float32,float64}`
  (default `float64`), `--pi-iters`/`--neumann-terms`/`--e-max-iter` (→ `SolverOptions`).
- `resolve_gene_trees(values) -> list[str]`: thin wrapper over
  `gpurec.optim.genewise_fit._resolve_gene_trees` (handles list/glob/dir/`.ale`/`.newick`/listfile).
  Import lazily.
- `rates_to_theta(D, T, L, mode, S=None, F=None) -> torch.Tensor`: returns log₂-rate tensor of the
  right shape for the mode `(3,)`/`(S,3)`/`(F,3)` (broadcast the global triple). **Order `[D, L, T]`**
  (theta[2] is transfer) — CLI D/T/L flags → `[log₂ D, log₂ L, log₂ T]`.
- `bits_to_nats(x) -> x * math.log(2)`.
- `build_model(args) -> GeneReconModel`: lazy import; construct with resolved genes + `SolverOptions`.

### `gpurec/cli/reconcile.py`
- `--delta/--tau/--lambda` (float, default `1e-10`), `--out` (optional path).
- Global mode → single total logL (nats). Genewise mode → per-family vector. Print
  `f"{family}\t{logL_nats:.6f}"` (family name from the resolved gene-tree basenames; single-family
  fixtures → `my_family`). `--out` writes AleRax `per_fam_likelihoods.txt` format (`<family> <logL>`).
- Exit non-zero on non-finite logL.

### `gpurec/cli/fit.py`
- `--mode` dispatch: `genewise` → `fit_genewise(species, genes, device, dtype, certify=True, ...)`;
  `global`/`specieswise` → build model, `optimize(model.batch_statics, model.theta.detach(),
  model.receiver_weights.detach(), optimizer="adam", schedule="adaptive", max_steps=--steps)` then
  `final_eval(...)`.
- `--steps` (default 300), `--init-rate`, optional origination-regularizer knobs (base has them from
  the regularizers batch; expose `--origination-ridge`/`--dirichlet-c` as pass-throughs, default off).
- `--out` writes fitted rates in AleRax `# node D L T` order + a `<out>.json` sidecar
  (`{"nll_bits":…, "nll_nats":…, "elapsed_s":…, "mode":…, "n_families":…}`), matching the shape the
  scale kit's `bench_gpurec_fit.py` consumes. Print final NLL to stdout.

### `gpurec/bench/alerax_io.py` (pure text, no Triton — CPU-importable)
- `parse_alerax_likelihoods(path) -> dict[str, float]`: `<family> <logL_nats>` per line; skip
  non-numeric. (From the reference's `_parse_alerax_perfam`.)
- `parse_alerax_parameters(path) -> dict[str, tuple[float,float,float]]`: skip `#`-comment header,
  read `<node> <D> <L> <T>` → `{node: (D, L, T)}`. **Column order is D L T.**
- `norm_family_name(s) -> str`: strip `.ale/.txt/.ufboot/.newick/.nwk` suffix (from `_norm_name`).
- `AleraxRates = namedtuple("AleraxRates", "D L T")`. AleRax's `D L T` columns are already gpurec's
  theta order `[D, L, T]`, so the triple maps straight to `theta` — **no reorder helper** (an earlier
  draft's `as_gpurec_dtl` swap was itself the bug).

### `gpurec/bench/fidelity.py`
- `reconcile_at_alerax_rates(species, genes, rates, *, device, dtype, mode="global", solver_options=None)
  -> dict[str, float]`: lazy-import `GeneReconModel`; set θ from `rates` directly as
  `[log₂ D, log₂ L, log₂ T]` (no reorder); return `{family: logL_nats}`. For single-family fixtures
  the dict has one entry keyed by the normalized family name.
- `compare(gpurec_ll, alerax_ll) -> FidelityReport`: align on normalized names; fields `n_matched`,
  `mean_abs_delta`, `rmse`, `max_abs_delta`, `total_delta`, and `pearson_r` (only meaningful when
  `n_matched > 2`, else `None`). Nats throughout.

### CLI wiring (`gpurec/cli/main.py`)
- argparse with `subparsers = parser.add_subparsers(dest="command")`; `reconcile` and `fit`
  subparsers each get the common args + their specific args. `main(argv=None)` parses, dispatches to
  `run_reconcile`/`run_fit`, returns an int exit code. Bare `gpurec` with no subcommand prints help
  and exits 2. `pyproject.toml`: `[project.scripts] gpurec = "gpurec.cli.main:main"`.

### Vendored fixtures
Copy from `gergely_version/tests/data/` the **text** assets only (no binary `.ccp`): for each of
`test_trees_1`, `test_trees_2`, `test_trees_3`, `test_trees_200`, `test_mixed_200` →
`sp.nwk`, `g.nwk`, `families.txt`, and `output/per_fam_likelihoods.txt` +
`output/model_parameters/model_parameters.txt`. Layout under `tests/data/alerax/<fixture>/`. Also
vendor `alerax_archaea_davin2017/species_reference/reference_species_tree.newick` →
`tests/data/archaea60/reference_species_tree.newick`. Total added data is small (largest text asset
`test_mixed_200/g.nwk` ≈ 84 KB).

### `benchmark/` scale kit (ported, base-wired, cluster-run)
Port gergely's `benchmark/` structure — `config.sh`, `lib.sh`, `bin/00_preflight..70_check_fidelity`,
`bench_gpurec_fit.py`, `slurm/*.sbatch` — with these rewirings:
- `bench_gpurec_fit.py` → call `gpurec.optim.optimize` / `fit_genewise` (not gergely's optimizers);
  write rates + JSON sidecar via `gpurec/bench` helpers.
- Fidelity stages (`70_check_fidelity`, `eval_at_alerax_rates`, `eval_branchwise_perfam`) → call
  `gpurec/bench/fidelity.py` + the `gpurec` CLI.
- `config.sh` → dataset selector `{williams|archaea}`; Williams via `ZENODO_TARBALL_URL`; archaea via
  `ARCHAEA_DATA_DIR` (default the sibling repo path) + the vendored species tree; `15_setup` builds
  the Rust extensions (`pip install -e .`), not C++ JIT.
- Mode→AleRax parametrization map (`GLOBAL/PER-SPECIES/PER-FAMILY`) preserved.
No `benchmark/` stage runs in CI. A CPU dry-run test asserts the scripts parse and the Python
drivers import + build argument parsers (no dataset, no GPU).

## Testing strategy

- **Parsers (`test_alerax_io.py`, CPU, `-W error`):** `parse_alerax_likelihoods` /
  `parse_alerax_parameters` on vendored fixture files return expected dict sizes + a spot-checked
  value (e.g. `test_trees_1` logL `-2.56495`); comment header skipped; `norm_family_name` strips
  suffixes; `AleraxRates` parses `D L T` in column order (which is already gpurec's theta order — no reorder).
- **CLI parsing (`test_cli.py`, CPU):** `main(["--help"])` / `main([])` behavior; `reconcile`/`fit`
  subcommand routing; `rates_to_theta` shapes per mode; `bits_to_nats`; missing-required-arg errors;
  `resolve_gene_trees` on a glob/dir. All without importing the forward path (assert no CUDA needed).
- **CLI end-to-end (`test_cli.py`, `@pytest.mark.gpu`):** `gpurec reconcile` on a tiny vendored
  fixture prints a finite logL; `gpurec fit --mode global --steps 3` reduces NLL vs the initial rate.
- **Fidelity gate (`test_fidelity_alerax.py`, `@pytest.mark.gpu`):** for each fixture,
  `reconcile_at_alerax_rates` at the fixture's `ref/model_parameters.txt` rates (mapped to gpurec
  theta `[D, L, T]` with no reordering) matches `ref/per_fam_likelihoods.txt` within
  `mean_abs_delta ≤ 1e-3` nats (observed ~1e-5). Parametrized over the 5 fixtures. A wrong theta
  order fails this on the transfer fixture (`test_trees_2`) by ~20 nats — the test's built-in trip
  wire.
- **Convention check (`test_alerax_convention.py`, CPU):** the shipped stock reference equals the
  AleRax_fixed reference `+ ln(2S−1)` per fixture (documents the origination-normalization gap; pure
  parse+compare, no gpurec).
- **Scale kit (`test_cli.py` or `test_benchmark_smoke.py`, CPU):** `bench_gpurec_fit.py` and the
  fidelity drivers `--help`/import cleanly; `config.sh` sources without error (`bash -n`).
- **Full suite:** CPU `-m "not gpu"` green; GPU `-m gpu` green on this box.

## Out of scope (explicitly)

- Running Williams 2017 or archaea60 at scale inside CI (no data locally / cluster-only).
- Item #5's other validation oracles (full-matrix solver, dense pibar, `torch.func` reference VJPs,
  deep kernel/grad tests) — a separate port.
- DDP multi-GPU sharding, on-disk preprocessing cache, pytest auto-marker conftest.
- A Python `io/ale.py` loader — the base already loads `.ale`/Newick via the Rust preprocessor.
- Running AleRax to *generate* new reference data for archaea60 (would need a cluster AleRax run).
- Per-branch (specieswise) AleRax fidelity beyond the global toy fixtures — the scale kit's
  `eval_branchwise_perfam` path is ported but only exercised on the cluster.
