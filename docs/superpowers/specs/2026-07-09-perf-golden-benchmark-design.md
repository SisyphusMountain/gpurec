# Perf / golden non-regression benchmark (three fit modes)

**Date:** 2026-07-09
**Branch:** `dev`
**Status:** design approved, pending implementation plan

## Goal

Solidify `gpurec` with an automatic, reproducible non-regression benchmark that, for each of
the three fit modes (`global`, `genewise`, `specieswise`), fits a fixed simulated dataset and
asserts the fit still lands on the **same final likelihood and the same fitted rate parameters**
as a committed golden fixture. Wall-clock time is **recorded** to a report each run but never
asserted (it is machine-specific).

This is deliberately scoped to **(B)** the perf/golden benchmark. A follow-up spec will cover
**(A)** the full test-coverage audit.

## Non-goals

- No CI wiring (there is no CI today; tests are marked `@gpu @slow`, opt-in, and skip off-CUDA).
- No wall-time assertion — timing is logged for trend-watching only.
- No committing of the simulated trees (regen-only, see Dataset delivery).

## Datasets (simulated with `rustree`, fixed seeds)

Species tree: `rustree.simulate_species_tree(n=500, lambda_, mu, seed=…)` (≈500 extant leaves,
so `S ≈ 999` internal+leaf species nodes). DTL rates centered on **0.05**. 1000 families each.

| Mode | Simulation call | Rate structure |
|------|-----------------|----------------|
| `global` | `sp.simulate_dtl_batch(1000, 0.05, 0.05, 0.05, seed=S_g)` | one shared (D,T,L) |
| `genewise` | loop `sp.simulate_dtl(D_i, T_i, L_i, seed=…)` ×1000 | per-family (D,T,L) drawn around 0.05 (fixed per-family seeds) |
| `specieswise` | `sp.simulate_dtl_batch_with_branch_rates(1000, λ_d=[…], λ_t=[…], λ_l=[…], origination_probability=[…], seed=S_s)` | per-species rate vectors around 0.05 |

`rustree` is **verified deterministic** for a fixed seed (identical species + gene-tree hashes
across repeated runs on this machine/build), which is what makes a committed golden meaningful
under regen-only delivery.

### Dataset delivery: regen-only

- Commit the **seeded regen module** (`gpurec/bench/simulate.py`) and the **golden fixtures**;
  do **not** commit the trees.
- At test time, a session-scoped fixture regenerates each mode's dataset once (into a
  gitignored scratch dir), then all assertions run against it.
- Tests **skip cleanly** if `import rustree` fails (rustree currently lives in a separate
  venv; installing it into the gpurec env is documented but not required for the rest of the
  suite). This is the explicit tradeoff of regen-only: a fresh clone without rustree gets no
  non-regression coverage from these three tests, but nothing heavy enters git.

## Fit driver

For each mode:

```python
model = GeneReconModel(species_tree, gene_trees, mode=MODE, device="cuda", dtype=torch.float32)
theta_hat, hist = optimize(
    model.batch_statics,
    model.theta.detach(),
    model.receiver_weights.detach(),
    group_index=model.rate_family_idx,   # for genewise/specieswise grouping
    verbose=False,
)
nll = final_eval(model.batch_statics, theta_hat, model.receiver_weights.detach())  # fp64 NLL + ||g||
rates = 2.0 ** theta_hat                  # log2 → linear D,T,L
```

Use the `GeneReconModel` + `optimize` entry point **as-is** (user directive).

**Open nuance for spec review:** `genewise`'s *canonical* library entry is `fit_genewise`
(independent per-family LBFGS), not `optimize` (whose Newton polish assumes a single shared
smooth optimum). Driving genewise through `optimize` may converge differently or worse. The
plan will drive genewise through `optimize` as directed, but if minting reveals it does not
converge sensibly, that finding will be surfaced (candidate fix: switch the genewise test to
`fit_genewise`).

## Golden fixtures

One JSON per mode in `tests/regression/goldens/{global,genewise,specieswise}.json`:

```jsonc
{
  "mode": "global",
  "provenance": {
    "rustree_commit": "<git rev>",
    "species_seed": …, "gene_seed": …,
    "n_species": 500, "n_families": 1000, "dtl": 0.05,
    "gpurec_commit": "<git rev at mint time>",
    "device": "RTX 4090", "dtype": "float32",
    "minted_utc": "…"
  },
  "nll": <float>,                 // fp64 final_eval NLL
  "rates": [[D,T,L], …],          // FULL fitted-rate vector: [1] triple (global),
                                  //   [G,3] (genewise), or [S,3] (specieswise)
  "recorded_wall_s": <float>,     // informational baseline; NOT asserted
  "tolerances": { "nll_rtol": …, "rates_rtol": …, "rates_atol": … }
}
```

Assertions per test:
- `final_eval` NLL ≈ `golden.nll` within `nll_rtol`.
- fitted `rates` full vector ≈ `golden.rates` via `torch.allclose(rtol, atol)`.

## Tolerances & the reproducibility risk

The golden is a **self-golden**: "this code, on this seeded data, produces these numbers." The
governing risk is **run-to-run fp32 fit reproducibility** — nondeterministic atomic reductions
perturb the optimization *trajectory*, and fitted θ at a flat optimum can drift more than a
single forward eval. Existing precedent (`test_optim_golden.py`): NLL `rtol=1e-3`, grad
`rtol=2e-3`.

**Tolerances are set empirically at mint time**, not guessed: mint repeats each fit N times
(e.g. N=5), measures the observed spread of NLL and θ, and sets `rtol = k × observed_spread`
(k≈3). Expected outcome: NLL tight (~1e-3); θ looser. **Fallback if θ is too unstable to pin:**
pin NLL tightly and assert θ against the *true* simulation rates with a statistical tolerance
instead of against a pinned vector — the plan will record which choice was made per mode, with
the measured spreads, in the golden's `provenance`.

## Layout

```
gpurec/bench/simulate.py                       # seeded rustree → (species_tree_path, gene_tree_paths) per mode
tests/regression/
  __init__.py
  conftest.py                                  # session-scoped regen fixtures; skip if rustree/CUDA absent
  goldens/{global,genewise,specieswise}.json   # COMMITTED goldens
  test_perf_regression.py                      # 3 tests (one per mode), @gpu @slow
  mint_goldens.py                              # (re)generate goldens on the 4090; measures spread, sets tolerances
  reports/                                     # gitignored: append-only per-run wall-time log
```

New pytest marker `regression` registered in `pyproject.toml` alongside `gpu`/`slow`.

## Testing the benchmark itself

- `mint_goldens.py` is run once on the 4090 to establish committed goldens (and again to
  re-mint if rustree/gpurec change).
- The 3 regression tests are the deliverable; each is self-checking against its golden.
- A tiny smoke check (fast, non-`slow`): `simulate.py` produces a well-formed dataset at a
  small size (e.g. 20 leaves / 5 families) so import/plumbing regressions are caught without a
  full fit. Skips if rustree absent.

## Provenance & re-minting

Goldens are tied to the pinned `rustree` commit and seeds. If rustree's simulator output
changes, regen-only means the golden must be re-minted; `mint_goldens.py` documents and
automates this. `gpurec_commit` in provenance records which gpurec produced the pinned numbers.
