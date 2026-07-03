# gpurec vs AleRax at fixed rates — and AleRax's rate-dependent under-convergence

**Date:** 2026-07-03 · **Box:** RTX 4090 (CUDA), base `.venv` (torch 2.11 + triton) ·
**AleRax:** `AleRax_fixed` (exposes `--fix-rates` and `--fixed-point-iterations`) ·
**Fixture:** `test_mixed_200` (100 species, one family `my_family`, large gene-tree distribution).

## Question

Do gpurec and AleRax agree on the marginal reconciliation log-likelihood at **genuinely
non-trivial rates** — all of D, L, T distinct *and large* — and is AleRax's shipped default a
converged reference? The toy fixtures only ever exercised a single process at a small fitted rate
(transfer ≈ 0.05, or floored), so the [D, L, T] mapping and the high-transfer regime were untested.

## Method

Both engines evaluate at the **same fixed rates** — no rate optimisation on either side:

- **AleRax_fixed**: `--fix-rates --d D --l L --t T --fixed-point-iterations N`
  (defaults otherwise: `UndatedDTL`, `--transfer-constraint PARENTS`, uniform origination,
  `--species-tree-search SKIP` — matching how the shipped fixtures were produced).
- **gpurec**: `theta = [log₂ D, log₂ L, log₂ T]` (its native rate order; `theta[2]` is transfer).

Log-likelihoods are reported in **nats** (gpurec internally returns NLL in bits; `logL = −loss·ln2`).

## Result 1 — high rates D/L/T = 1.5 / 1.6 / 1.7

| engine | setting | logL (nats) |
|---|---|---|
| AleRax_fixed | N = 4 (**default**) | **−8736.45** |
| AleRax_fixed | N = 16 | **−8649.06** |
| AleRax_fixed | N = 64 | **−8649.06** (plateaued) |
| **gpurec** | converged (e=2000, pi=64) | **−8649.05735** |
| gpurec | heavy (e=6000, pi=256) | −8649.05735 (identical → converged) |
| gpurec | light (e=50, pi=8) | −8650.78 (deliberately under-iterated) |

**gpurec and AleRax_fixed agree to ~0.003 nats** (−8649.057 vs −8649.06, within AleRax's 2-decimal
print precision) at a rate triple where the [D,L,T] order, the transfer-to-parents constraint, and
the survival/origination normalization all have to be right — any one wrong would blow up at T=1.7.

But the match only appears **once AleRax runs enough fixed-point iterations**. Its default (N=4) is
**87 nats short**; increasing N moves it monotonically onto gpurec's independently-converged value,
and it has plateaued by N=16. gpurec is the well-converged side (default == 256-iter to 5 decimals).

## Result 2 — the fixture's fitted moderate rates (D≈0.16, T≈0.16), and the convention

| quantity | logL (nats) |
|---|---|
| AleRax_fixed N = 4 | −6221.02 |
| AleRax_fixed N = 16 | −6221.02 (**already converged at N=4**) |
| gpurec converged | −6221.02 |
| stock shipped (AleRax v1.3.0, 4-iter) | −6215.73 |
| ln(2S−1) = ln(199) | 5.29330 |

At these moderate rates N=4 is **already converged** (N4 == N16), so under-convergence is invisible
here. The stock-vs-fixed gap is purely the **branch-count normalization**:

```
stock − AleRax_fixed = −6215.73 − (−6221.02) = 5.29  ≈  ln(2S−1) = 5.293   ✓
```

Stock AleRax omits the `−ln(2S−1)` origination-branch term; `AleRax_fixed` (and gpurec) include it.

## Conclusion

1. **gpurec ≡ AleRax_fixed(converged)** at both low fitted rates (~1e-5 nats, from the toy fixtures)
   and high rates 1.5/1.6/1.7 (~0.003 nats, AleRax's print floor). The model, rate order, transfer
   constraint, and normalization all match.
2. **AleRax's fixed-point under-convergence is rate-dependent.** Negligible at the toy fixtures'
   low fitted rates (which is why an earlier look saw it "flat across iterations"), but **87 nats at
   high rates**. The original instinct to make the iteration count adjustable was correct — it just
   doesn't bite until rates get large.
3. **Two distinct effects, don't conflate them:** the stock-vs-fixed `ln(2S−1)` gap is a
   *normalization convention* (rate-independent); the N=4-vs-converged gap is *under-convergence*
   (rate-dependent).

### Implication for the fidelity kit

Generate AleRax reference likelihoods with a **high `--fixed-point-iterations` (≥16; the plan pins
64)** — never the default 4 — so the references are converged in any rate regime. See
`docs/superpowers/plans/2026-07-02-cli-and-fidelity.md`, Task 6.

## Reproduce

```bash
cd /home/enzo/Documents/git/gpurec/consolidate-release
.venv/bin/python experiments/alerax_convergence/reproduce.py            # full (~4–5 min)
.venv/bin/python experiments/alerax_convergence/reproduce.py --quick    # skip slow N=64
```

`reproduce.py` locates an `AleRax_fixed` binary (auto under `agent-worktrees/`, or set
`ALERAX_BIN`), runs both engines at both rate sets, prints the tables above, writes `results.json`,
and ends with PASS/FAIL self-checks (converged agreement, gpurec self-convergence, the 87-nat
default gap, and the `ln(2S−1)` convention identity). Requirements: a CUDA GPU + the base env, an
`AleRax_fixed` binary, and the fixture (override with `FIXTURE=...`).
