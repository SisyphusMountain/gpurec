# Genewise adaptive-rebatch + bounded convergence (hogenom 1055)

Reproducible run of `experiments/sanderson_cv/bench_genewise_adaptive.py`: drive every per-family
genewise rate vector `theta[F,3]` (rate = `2^theta` = P(category)/P(speciation)) to a certified
per-family minimum, with **box-bounded rates** and **adaptive pi-tier rebatching by convergence
difficulty**.

## Recipe
- **Bounds:** rate in `[1e-6, 2]` -> theta in `[log2(1e-6), log2(2)] = [-19.93, 1.0]`. Runaway
  (non-identifiable) families park *at* the bound (projected gradient -> 0 = constrained KKT min)
  instead of diverging to `theta = +/-inf`.
- **Adaptive rebatch:** all families start at `pi=16`; each tier runs a bounded **active-set projected
  trust-region Newton** (reused 3x3 FD Hessian, no line search) over the still-active families, then a
  family GRADUATES only if its projected `|g| < 1e-3` at BOTH the current pi AND the next pi (stability
  check -> not truncation-biased). The rest are rebatched into the next, higher-pi tier
  (16 -> 32 -> 64 -> 128 -> 256). Only the shrinking hard set pays for high pi.
- **Per-family 3x3 PD cert by FD** (6 grad evals, no HVP), authoritative at `pi=256`.

## THE BUG THIS RUN FIXES (active-set Newton)
A naive bounded "projected Newton" (full 3x3 Hessian step, then clamp theta) **diverges** on families
with a bound-active coordinate coupled to a free one. Duplication & Loss are collinear (the "confounded
pair"); when D is pinned at the rate=2 bound, the full-Hessian Newton direction tries to move along
(D-L) -- D can't move, so L's step comes out with the WRONG sign and walks uphill into the wall
(loss 134 -> 153, `|g|` 3 -> 18). With no line search, nothing rejects it. ~110 families got stuck at
projected `|g| ~ 75`, dead-constant across all pi tiers.

**Fix:** active-set / reduced-Hessian Newton -- a coord is FIXED if it sits at a bound with the gradient
pushing further out (KKT-binding); solve the Newton system only on the FREE coords (zero the fixed
rows/cols). The clamped D can no longer corrupt L's step. The stuck families then converge in ~3 steps.

## Result (pinned: hogenom_1055.json)
```
CONVERGED (|Pg|<1e-3) = 1046/1055     (was: 110 diverging at |Pg|=75)
  interior PD minima        = 785
  bound-active (rate@2)     =  97      (ex-runaway families, now well-defined)
  still-unconverged         =   9      (|Pg| 2-8e-3, interior, at the FD-Newton precision floor)
graduated by tier: pi16=984  pi32=52  pi64=7  pi128=3  pi256=1
rate range: [3.4e-6, 2.0]    TOTAL = 720s (~12 min, RTX 4090)
```

## Reproduce
From the worktree root, miniforge3 python (torch 2.11+cu130):
```bash
WT=$(git rev-parse --show-toplevel)
GPUREC_PREPROCESS_PATH=$WT/crates/gpurec-preprocess/target/release/libgpurec_preprocess.so \
PYTHONPATH=$WT \
DATASET=hogenom FAMILIES=all PIS=16,32,64,128,256 MIN_RATE=1e-6 MAX_RATE=2 TOL=1e-3 \
ADAM=20 TIER_NEWTON=35 HESS_EVERY=5 SEED=0 \
OUT_JSON=experiments/sanderson_cv/_artifacts/genewise_adaptive/hogenom_1055.json \
python -u experiments/sanderson_cv/bench_genewise_adaptive.py
```
Data: `tests/data/alerax_hogenom_core/hogenom/` + `experiments/sanderson_cv/families_1055.txt`.
For archaea: `DATASET=archaea` (5446 families, S=119).

**Determinism:** theta=0 init + (Adam, FD-Hessian, reduced-Newton) are deterministic, but the genewise
backward uses atomic accumulation (run-to-run gradient noise ~2e-4), so the per-family counts reproduce
to within a few families, not bit-exactly. The 9 stragglers sit at that floor.
