# Does the new gpurec reproduce the original code's fitted likelihoods?

Two datasets the new code was **not** tuned on -- archaea (60 species, 5446 gene families) and
HOGENOM (666 species, up to 10,869 gene families) -- were fitted twice: once with the original code
and once with the current branch head, each at its own default recipe. Both fits were then scored by
one common solver, so any difference in likelihood below is about where the two fits *landed*, not
about how they were evaluated.

Short answer: **yes on HOGENOM, and yes on archaea apart from the families that carry no
information.** Details and numbers below.

## What was compared

| | original code | new code |
|---|---|---|
| checkout | `/sps/biometr/emarsot/gpurec_base`, commit `817007e6` | `/sps/biometr/emarsot/gpurec`, commit `a8598cee` (branch `perf/genewise-cc-h100`) |
| genewise starting rates | all three rates = 1.0 (`theta` = 0) | D = 0.01, L = 0.1, T = 0.01 (`fit_dtl`'s default) |
| forward / adjoint self-loop | the original kernels | `exact` for both (the new library default) |
| genewise certificate | includes the interior positive-definite count | omits it (`certify_curvature=False`) |
| memory policy | `GPUREC_MEMORY_POLICY_FRACTION=0.3`, as in the earlier runs on this machine | left at its default |

Nothing else was changed. Both codes read **the same family list files in the same order**, so
per-family results line up row by row.

Hardware: one NVIDIA H100 NVL (94 GiB) per job on CC-IN2P3, `gpu_h100` partition, PyTorch
2.13.0+cu130. **The nodes were shared with other jobs throughout, so the wall times below say which
code is roughly faster, not how fast either one is.**

## The family lists

`benchmark/cc/ab_make_lists.py` built them and printed every family it dropped and why.

| list | families | note |
|---|---:|---|
| archaea, all `.ale` files | 5446 | every file in `main_families_ge4seq`; **none failed to preprocess** |
| archaea, >= 4 covered species | 3946 | built for reference only; the fits used all 5446 |
| HOGENOM 1055-family subset | 1055 | `experiments/sanderson_cv/families_1055.txt`; **none failed to preprocess** |
| HOGENOM full matched set | 10869 | every family named in the AleRax likelihood file; all present on disk |

Nothing failed to preprocess anywhere. Two counts look like problems and are not:

- **1500 of the 5446 archaea families cover fewer than 4 species.** The directory is named
  `main_families_ge4seq` -- four *sequences*, not four *species* -- and four sequences can all come
  from the same organism. Those families were kept: both codes get the identical list either way.
  They turn out to be exactly where the two codes disagree (see below).
- **13 of the 1055 HOGENOM families are absent from the AleRax likelihood file**, and they are
  exactly the 13 that cover fewer than 4 species -- AleRax dropped them for the same reason. The
  HOGENOM-1055 cross-check is therefore over 1042 families, not 1055.

## Runs 1, 3 and 4: genewise fits

NLL = negative log-likelihood; **lower is better**. "converged", "bound-active" and "unconverged"
are the fit's own certificate; a bound-active family is one whose fitted rate ended sitting on the
edge of the allowed rate box (D, L and T are confined to [1e-6, 2.0] relative to speciation).

| dataset | code | families | wall (s) | NLL (bits) | NLL (nats) | converged | bound-active | unconverged | max &#124;Pg&#124; | peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| archaea 5446 fam | old | 5446 | 768.1 | 360167.917 | 249649.376 | 5446 | 1275 | 0 | 0.001 | 8.15 |
| archaea 5446 fam | new | 5446 | 95.1 | 360336.068 | 249765.929 | 5446 | 869 | 0 | 0.001 | 3.38 |
| HOGENOM 1055 fam | old | 1055 | 815.3 | 578249.750 | 400812.184 | 1054 | 97 | 1 | 0.00445 | 33.07 |
| HOGENOM 1055 fam | new | 1055 | 65.2 | 578249.677 | 400812.133 | 1054 | 97 | 1 | 0.0166 | 11.44 |
| HOGENOM 10869 fam | new | 10869 | 260.0 | 1906508.952 | 1321491.305 | 10869 | 724 | 0 | 0.001 | 34.49 |

The old code was not run on the full 10,869-family HOGENOM set: it took 815 s on the 1055-family
subset, so the full set would have run past the 2-hour job limit.

### Per-family comparison, both fits scored by one solver

`compare_fit_thetas.py` re-scores both fitted rate sets with the same converged solver (64 Pi
iterations, 64 Neumann terms, exact forward self-loop), so the numbers isolate where the fits landed.
The re-scored totals reproduce each run's own reported NLL: exactly for the new code (360336.0676
vs 360336.068 on archaea, 578249.6771 vs 578249.677 on HOGENOM-1055, 1906508.9525 vs 1906508.952 on
the full set) and to within 0.004-0.006 bits for the old code (360167.9205 vs 360167.917;
578249.7556 vs 578249.750), because the new code's own final evaluation already uses this solver.
Those 0.005 bits are 30,000 times smaller than the 168-bit archaea gap, so the scoring is neutral.

**archaea, 5446 families** (`cmp_arch_old_gw__arch_new_gw.txt`):

| | value |
|---|---:|
| total NLL difference, new minus old | **+168.147 bits** over 5446 families = **+0.031 bits/family** |
| as a fraction of the total | 4.7e-5 (0.0047%) |
| families agreeing within 0.01 bits | 5008 of 5446 (92.0%) |
| families differing by more than 0.1 bits | 424 (414 worse for new, 10 better) |
| families differing by more than 1 bit | 7 |
| largest single family difference | +2.261 bits worse, -2.079 bits better |
| share of the gap carried by the 10 worst families | 4.3% |

So the archaea gap is **not** a handful of blown-up families and it is **not** a systematic shift
across the set: it is spread over about 430 families, each contributing a few tenths of a bit.

Who are those 430 families? (`split_archaea.txt`)

```
[split] 5446 families, threshold 0.01 bits
[split] disagree: 438 families (8.0%) -- 407 (92.9%) cover < 4 species,
        409 (93.4%) have a rate pinned to a bound in at least one fit, median covered species 1
[split] agree:  5008 families (92.0%) -- 1093 (21.8%) cover < 4 species,
        870 (17.4%) have a rate pinned to a bound in at least one fit, median covered species 5
[split] the 438 disagreeing families carry +168.248 bits of the +168.147 bit total
```

**Every bit of the archaea gap sits on families whose gene tree covers a median of ONE species.**
A single-species family cannot distinguish a duplication from a transfer from a loss, so its
likelihood surface is nearly flat over a huge region and the two optimizers stop in different places
inside it. The fitted rates make this obvious: for the worst family the original code stopped at
D = 2.0 (the box's upper edge), L = 1.70, T = 1.6e-4, while the new code stopped at D = 0.87,
L = 1.3e-3, T = 0.067 -- wildly different rates worth 2.3 bits of each other. The old code starts
every rate at 1.0 and walks up to the upper bound; the new code starts at D = 0.01, L = 0.1,
T = 0.01 and walks down. On a flat surface the starting point decides the answer.

**HOGENOM, 1055 families** (`cmp_hog1055_old_gw__hog1055_new_gw.txt`):

| | value |
|---|---:|
| total NLL difference, new minus old | **-0.0786 bits** over 1055 families (the new fit is *better*) |
| as a fraction of the total | 1.4e-7 |
| families agreeing within 0.01 bits | 1054 of 1055 |
| families differing by more than 0.1 bits | 0 |
| largest single family difference | 0.032 bits better, 0.007 bits worse |

This is reproduction to the numerical floor: one family out of 1055 moves by more than a hundredth
of a bit.

## Run 2: global mode (one shared D/L/T for every family)

| dataset | code | families | wall (s) | D | L | T | NLL (bits) | NLL (nats) | steps | max &#124;Pg&#124; |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| archaea 5446 fam | old | 5446 | 233.4 | 0.182717 | 0.430304 | 0.268559 | 383695.374 | 265957.367 | 6 | 0.0588 |
| archaea 5446 fam | new | 5446 | 57.1 | 0.182717 | 0.430304 | 0.268558 | 383695.386 | 265957.375 | 6 | 0.0547 |
| HOGENOM 1055 fam | old | 1055 | 472.5 | 0.182655 | 0.56111 | 0.189046 | 594992.684 | 412417.501 | 7 | 0.00943 |
| HOGENOM 1055 fam | new | 1055 | 53.8 | 0.182655 | 0.56111 | 0.189046 | 594992.674 | 412417.494 | 7 | 0.0163 |

Global mode reproduces essentially exactly on both datasets: the three fitted rates agree to six or
seven significant figures, and the totals differ by 0.012 bits out of 383,695 on archaea (3e-8
relative) and 0.010 bits out of 594,993 on HOGENOM (1.6e-8 relative). Both codes take the same
number of Newton steps. The HOGENOM-1055 global pair was not one of the four requested runs; it was
added because of the smoke-run caveat two paragraphs down.

### A bug found on the way: global mode did not run at all, in either checkout

`gpurec/fit/global_fit.py` builds its per-tier solver settings as

```python
return SolverOptions(pi_iters=pi_iters, neumann_terms=neumann_terms, e_adjoint_solver="neumann")
```

but `SolverOptions` has no field named `e_adjoint_solver`, so every call raises
`TypeError: SolverOptions.__init__() got an unexpected keyword argument 'e_adjoint_solver'` before
any work happens. The line is **byte-identical in the original checkout (817007e6) and at branch
head (a8598cee)** -- this is not a regression, it is a long-standing dead caller. The E-adjoint
solve is Neumann-only now, so the keyword names the only behaviour there is and removing it changes
nothing numerically.

`benchmark/cc/run_global.py` installs a one-function shim that drops the keyword, **identically for
both checkouts**, and records the fact in every global result JSON (`"fit_global_shim"`). The
library itself was not touched. The one-line fix belongs in `gpurec/fit/global_fit.py`: delete the
`e_adjoint_solver="neumann"` argument.

### A caveat on the 40-family global smoke run

At 40 archaea families the two codes agreed to 1e-6 relative, but at 40 HOGENOM families they did
not: the old code reached NLL 22992.474 bits with a final projected gradient of 0.088, while the new
code stopped after 15 steps at NLL 23073.340 bits with a projected gradient of 218 --
`fit_global`'s loss-plateau stopping rule fired early. **At the real sizes (1055 and 5446 families)
this does not happen and the two agree to seven figures**, so it is a small-sample artifact of the
plateau rule, not a difference between the codes. It is recorded here because a stopping rule that
exits at a projected gradient of 218 is worth knowing about on its own.

## AleRax cross-check

Per-family log-likelihood in nats (higher = better fit), gpurec at its own fitted rates against
AleRax's reported per-family values in
`benchmarks/hogenom-cpu-vs-gpu/results/alerax_hogenom_combined_likelihoods.txt`, restricted to the
families present in both.

| fit | matched families | Pearson r | mean gpurec - AleRax | median | mean abs | max abs | gpurec total NLL | AleRax total NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HOGENOM 1055, old code | 1042 | 0.9999572 | +0.9121 | +0.1856 | 0.9122 | 75.6135 | 400356.72 | 401307.11 |
| HOGENOM 1055, new code | 1042 | 0.9999572 | +0.9121 | +0.1856 | 0.9122 | 75.6135 | 400356.66 | 401307.11 |
| HOGENOM 10869, new code | 10869 | 0.9999169 | +0.4195 | +0.0015 | 0.4257 | 163.3699 | 1321491.31 | 1326050.75 |

The old and new cross-check statistics on the 1055-family subset agree to six decimal places -- the
two fits are indistinguishable as far as AleRax is concerned. On both sets gpurec assigns a higher
likelihood than AleRax reports: 950.4 nats over 1042 families (+0.912/family) and 4559.4 nats over
10,869 families (+0.4195/family). The single worst family is `CLU_004282_0_0_C` at 163.4 nats.

### Against the recorded full-set numbers

`benchmarks/hogenom-cpu-vs-gpu/results/xcheck_headline_fullset.txt` recorded, for the same 10,869
families:

| quantity | recorded | this run (new code) | difference |
|---|---:|---:|---:|
| gpurec total NLL at gpurec's own rates (nats) | 1321460 | 1321491.31 | +31 nats (2.4e-5 relative) |
| AleRax reported total NLL (nats) | 1326051 | 1326050.75 | same file, same number |
| reported margin, gpurec vs AleRax | 4591 nats = +0.422/family | 4559 nats = +0.4195/family | -32 nats, -0.003/family |
| median per-family difference (nats) | +0.0018 | +0.0015 | -0.0003 |

The new code's own optimum is 31 nats worse than the recorded one out of 1.32 million nats -- a
relative difference of 0.0000237 -- and the headline margin over AleRax is preserved (+0.42
nats/family both times). Note the recorded median (+0.0018) came from scoring *AleRax's* rates in
gpurec, whereas this one scores *gpurec's* rates; that the two medians land within 0.0003 nats of
each other says the two tools' fitted rates are nearly the same for the typical family.

## Speed (shared nodes -- read as ratios, not absolutes)

| run | old (s) | new (s) | new is |
|---|---:|---:|---:|
| archaea genewise, 5446 families | 768.1 | 95.1 | 8.1x faster |
| archaea global, 5446 families | 233.4 | 57.1 | 4.1x faster |
| HOGENOM genewise, 1055 families | 815.3 | 65.2 | 12.5x faster |
| HOGENOM global, 1055 families | 472.5 | 53.8 | 8.8x faster |
| HOGENOM genewise, 10869 families | not run | 260.0 | -- |

Peak GPU memory also drops: 8.15 -> 3.38 GiB on archaea genewise and 33.07 -> 11.44 GiB on
HOGENOM-1055 genewise.

## Reading

**On HOGENOM the new code reproduces the original optimum exactly.** On the 1055-family subset the
two fits differ by 0.079 bits out of 578,250 (1.4e-7), with a single family moving by more than a
hundredth of a bit; the AleRax cross-check statistics agree to six decimals. On the full 10,869
families the new code lands 31 nats from the previously recorded optimum out of 1.32 million
(2.4e-5) and keeps the +0.42 nats/family margin over AleRax.

**On archaea it reproduces the optimum for every family that carries information, and lands
somewhere else on the ones that do not.** 5008 of the 5446 families agree within 0.01 bits. The
whole 168-bit gap (0.031 bits/family, 0.0047% of the total) comes from 438 families whose gene trees
cover a median of one species. On those the likelihood surface is close to flat and the answer is
decided by where the optimizer starts -- the old code starts every rate at 1.0, the new one at
D = 0.01, L = 0.1, T = 0.01. Both fits certify all 5446 families converged with a projected gradient
below 0.001, so both are at a legitimate stationary point; they are simply different ones. On the
3946-family subset that covers at least 4 species this difference would largely disappear; that list
is already built, but the fits reported here deliberately used all 5446 families.

**Global mode reproduces on both datasets to six or seven significant figures**, once the
long-standing `e_adjoint_solver` crash is worked around -- and that crash is in both checkouts, so
it is not something this branch introduced.

## Files

Fit results (`benchmark/cc/results/`): `arch_old_gw.json`, `arch_new_gw.json`, `arch_old_gl.json`,
`arch_new_gl.json`, `hog1055_old_gw.json`, `hog1055_new_gw.json`, `hog1055_old_gl.json`,
`hog1055_new_gl.json`, `hogfull_new_gw.json`, plus the 40-family smoke runs `sm_*.json`.
Comparisons: `cmp_arch_old_gw__arch_new_gw.txt`, `cmp_hog1055_old_gw__hog1055_new_gw.txt`,
`split_archaea.txt`. Cross-checks: `xcheck_hog1055_old_gw.json`, `xcheck_hog1055_new_gw.json`,
`xcheck_hogfull_new_gw.json`.

Drivers (`benchmark/cc/`): `ab_make_lists.py` (family lists), `ab_run.sh` (one arm), `ab_submit.sh`
(one job, several arms), `run_global.py` (global-mode driver and the `fit_global` shim),
`ab_compare.sh` with `compare_fit_thetas.py` (score two fits with one solver), `ab_xcheck_single.sh`
with `score_per_family.py` (score one fit), `xcheck_alerax.py` (AleRax comparison),
`ab_disagreement.py` (who carries the gap), `ab_bound_active.py`, `ab_report.py` (regenerates the
tables above from the JSONs).

## Third round (2026-09-04): HOGENOM full set on the local RTX 4090, exact solves vs iterated solvers

After the rounding-floor fix (every transfer sum built by addition; see `docs/genewise_h100_runtime.md`),
the full 10,869-family HOGENOM set (666-species AleRax starting tree, the same list as above) was fitted
twice on one RTX 4090 (24 GB) with the current code, once with the exact tree solves (the defaults, the
configuration that took the Coleman fit under 800 s) and once with the iterated solvers (log-space sweeps,
Neumann-series adjoint, two tiers) as a control. The adjoint ran cold on both (its warm cache does not
fit in 24 GB).

| solver | wall | reported NLL (bits) | Newton steps | certified | bound-active | uncertified | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact forward + exact adjoint | 179 s | 1906508.955 | 93 | 10869 | 724 | 0 | 12.2 |
| log sweeps + series adjoint (two tiers) | 654 s | 1906508.734 | 224 | 10431 | 731 | 438 | 12.2 |

Both rate sets re-scored by one converged solver (64 sweeps, 64 terms, exact forward;
`compare_fit_thetas.py`, `cmp_hogenom_full_round3.txt`): no family differs by more than 0.01 bits
(largest 0.004); total 1906508.952 (exact) vs 1906508.758 (iterated), 0.19 bits apart over 10,869
families. The exact fit before the fix (cluster, commit a8598cee) re-scores to 1906508.953: 0.001 bits
from the new one, largest per-family difference 0.0004 bits. Against the original code's fit on the
1042 shared families of the 1055 subset: one family differs by more than 0.01 bits (better by 0.03),
total 0.07 bits better. AleRax cross-check (`xcheck_hog_exact_local.json`, `xcheck_hog_log_local.json`):
10,869 of 10,869 families matched, Pearson r = 0.999917 for both, mean gpurec minus AleRax +0.4195
nats per family, total margin +4559.4 nats, identical to the earlier cluster fit and the same for both
solver paths. The 438 families the iterated control leaves uncertified sit at the same optimum (their
re-scored likelihoods match the exact fit's within 0.004 bits); the control's accurate tier simply
runs out of its 120 iterations on them.
