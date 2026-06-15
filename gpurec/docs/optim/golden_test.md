# Golden regression test for the optimization layer

`tests/test_optim_golden.py` locks the **likelihood (value + gradient)** of the `gpurec/optim`
layer against future codebase changes, so a correctness bug introduced by editing a kernel, the
solver, or the optim layer is caught immediately. It runs on the frozen kernel-bench **666x80**
fixture.

## What it checks

At **two thetas** — the initial fixture theta *and* the fitted checkpoint `theta_hat` (from
kernel-bench `data/666x80/newton_golden_fp32.pt`) — and **two solver regimes** — the capture's
truncated `pi_iters=16` and a converged `pi_iters=64` — it asserts two layers:

1. **Self-golden** — gpurec loss/grad vs pinned, verified-correct values in
   `gpurec/optim/_golden/checkpoint_666x80.pt` (loss `rtol=1e-3`, grad rel-L2 `< 2e-3`). Catches
   any change that shifts gpurec's output. The fixture golden is kernel-bench's own canonical
   `whole.pt` `golden` (loss `170130.40625` + grad); the checkpoint goldens are gpurec values
   re-verified bit-exact against live kernel-bench.
2. **Cross-parity** — gpurec loss `==` **live kernel-bench** loss, **bit-exact** (`rtol=1e-6`), on
   identical inputs/hardware/kernels. This is the merge-boundary guard: if a future change diverges
   gpurec from the frozen kernel-bench solver, it fails here. Skipped only if the kernel-bench tree
   is not importable.

Tolerance rationale: the stale-golden discrepancy below is `4e-3` relative; atomic-reduction noise
is `~1e-5`. `rtol=1e-3` cleanly separates a real correctness bug from benign numerics. The
cross-parity check is essentially exact (same kernels, same GPU) so it uses `1e-6`.

## Why the checkpoint matters (the finding)

The fixture-theta parity (NLL `170130`) alone is a weak guard — a bug could be orthogonal to the
gradient at the init point. Verifying at a **fitted** `theta_hat` (a non-trivial, low-loss point) is
much stronger. Result (RTX 4090, fp32):

| theta | pi/neumann | gpurec | live kbench | gp − kb |
|---|---|---|---|---|
| fixture | 16/16 | 170130.40625 | 170130.40625 | **0.0** |
| checkpoint | 16/16 | 143463.54688 | 143463.54688 | **0.0** |
| checkpoint | 64/64 (converged) | 143462.07812 | 143462.07812 | **0.0** |

gpurec reproduces kernel-bench **bit-for-bit** at the checkpoint — the port is correct away from the
init, not just at it.

## ⚠ Stale recorded `FN` — do NOT use it as a golden

The value *recorded* in `newton_golden_fp32.pt` is `FN = 144033.16`, but re-evaluating the forward at
that exact `theta_hat` gives **~143462** (converged) in **both** gpurec and current kernel-bench — a
`~571` NLL gap that is **identical in both codebases**. Its history shows `loss == F` (so `FN` is the
bare NLL, not a MAP objective), and no `pi_iters` reproduces it. The golden file is dated `2026-06-14`
and the fixture-theta loss is unchanged while the `theta_hat` loss shifted — i.e. the recorded `FN`
is **stale** relative to the current kernel-bench forward, a pre-existing kernel-bench issue
independent of this merge. The golden test therefore pins the **re-verified** values, never `FN`.
(Open follow-up: git-bisect kernel-bench to pinpoint what shifted the `theta_hat` forward loss before
trusting any saved `*_fit.pt` checkpoint.)

## Running

```bash
# the 3.4 GB capture is external (kernel-bench, not in-repo). Auto-discovered if kernel-bench is an
# ancestor sibling; otherwise point at it:
export GPUREC_KB_CAPTURE=/path/to/kernel-bench/data/666x80/whole.pt
python -m pytest tests/test_optim_golden.py -v
```

Triton is a **hard dependency** (imported at module load) — a broken triton **errors**, never skips.
The test skips only on genuine environment gaps: no CUDA hardware, or the external capture absent.

## Regenerating the golden artifact

If a deliberate, reviewed change legitimately alters the numerics (e.g. a solver default), regenerate
`gpurec/optim/_golden/checkpoint_666x80.pt` — but **only after** confirming gpurec still matches live
kernel-bench bit-exact (the cross-parity test). The generator is the snippet in the commit that added
this file; it stores `theta_fixture`, `theta_hat`, and the loss/grad at pi=16 and converged pi=64.
