# Analytic HVP as the default genewise Hessian

## Context

`gpurec/fit/genewise_fit.py`'s `fit_genewise` (the production per-family DTL rate recipe) has
always used a forward-difference (FD) Hessian in its trust-region Newton step — 3 extra nonlinear
forward+backward evaluations per Hessian rebuild. Earlier work this session built and validated an
analytic-HVP alternative: `gpurec.solver.hvp_exact.make_exact_hvp`, exploiting genewise's exact
block-diagonal Hessian structure via 3 broadcast unit-theta-component probes (the same trick as
`tests/test_genewise_hvp.py`'s `genewise_hessian_blocks`), plus the warm-starting built in the
immediately preceding plan (`docs/superpowers/plans/2026-07-21-hvp-warm-start.md`).

A scratchpad prototype (`fit_genewise_analytic.py`, a standalone copy of the recipe with only the
Hessian source swapped) was benchmarked against production `fit_genewise` on real archaea60 data:

- **200 families**: analytic 9% faster (22.8s vs 25.1s), both converge 200/200, matching NLL.
- **Full 5446 families, 26 batches** (confirmed via direct model-construction check — this is the
  real multi-batch production scale, not an artificially favorable single-batch case): analytic
  **9% faster** (322.4s vs 353.9s) and converges **5446/5446** vs FD's 5444/5446, matching NLL to
  ~1e-7 relative.

This is now good enough evidence to make analytic HVP the default, not just a validated
alternative.

## Goals

- Replace FD with analytic HVP in `gpurec/fit/genewise_fit.py`'s `fit_genewise`, in **both** places
  it's used: the optimization loop's trust-region Newton step, and the `certify=True` diagnostic's
  final cold Hessian. (User's explicit call: certify switches too, retiring FD from the file
  entirely — not kept as an independent cross-check, since removing FD everywhere was preferred
  over keeping two Hessian code paths.)
- Remove `fd_eps` from `fit_genewise`'s signature and from `GENEWISE_REFERENCE` — it becomes
  genuinely dead code, not just deprioritized. This is a breaking change to the public signature.
- Apply the identical swap to `experiments/sanderson_cv/bench_genewise_warm_rebatch.py` (a separate,
  independent reimplementation of the same recipe used for larger-scale benchmark runs), so the two
  don't diverge into "one FD, one analytic" and cause confusion about which is authoritative.
- Preserve everything else about the recipe unchanged: `mu` (eigenvalue-floor convexification, still
  needed regardless of Hessian source), the outer trust-region/rebatching/tiering logic, `tol`,
  `max_iter`, `check_every`, `drop_frac`, `trust`.

## Non-goals

- Porting this to `main` is a separate, later phase. `main`'s current `genewise_fit.py` wasn't part
  of the Epoch-4 file comparison already done this session (that covered `_implicit_grad.py`,
  `solver_options.py`, `hvp_exact.py`, `cg.py`, `dtl_fit.py`, `global_fit.py`, `map_cv.py`,
  `newton_cg.py`, `optimize.py`, `specieswise_fit.py`, `memory_policy.py`, `batched_lbfgs.py`,
  `pyproject.toml` — not `genewise_fit.py` itself), so its actual gap against `dev` is unknown and
  needs checking once the dev-side change lands.
- Not touching the joint (theta+omega+alpha) CG-based Newton solvers (`newton_joint_genewise`,
  `origination_curvature.newton_joint`) — those already use analytic HVP via a different code path
  and are out of scope here.

## Design

### Shared Hessian-construction helper

Both places `genewise_fit.py` currently builds an FD Hessian (the optimization loop's per-iteration
rebuild, and the `certify` block's one-time cold rebuild) need the identical single-batch /
multi-batch analytic construction. Factor this into one module-level helper, `_analytic_hessian`,
mirroring the already-validated prototype's structure exactly:

```python
def _analytic_hessian(m, theta, pi_cur):
    """Per-family [G,3,3] curvature via 3 broadcast analytic-HVP probes, warm-started via
    probe_id. Mirrors hvp_exact.py's _make_exact_hvp_streaming genewise gather/scatter for
    single-batch (which the top-level make_exact_hvp does NOT do for you), and lets it handle
    multi-batch itself."""
    G = theta.shape[0]
    dev, dtype = theta.device, theta.dtype
    rw = m.receiver_weights.detach()
    if len(m.batch_statics) > 1:
        hvp = make_exact_hvp(m.batch_statics, theta, rw, None, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u = torch.zeros(G, 3, device=dev, dtype=dtype); u[:, j] = 1.0
            cols.append(hvp(u.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3))
        H = torch.stack(cols, dim=-1)
    else:
        static = m.batch_statics[0]
        fam = static.family_index_tensor.to(dev)
        theta_b = theta.index_select(0, fam).contiguous()
        _l, sv = forward_solve(m.batch_statics, theta, rw)
        hvp = make_exact_hvp(m.batch_statics, theta_b, rw, sv, tangent_self_iters=pi_cur)
        cols = []
        for j in range(3):
            u_b = torch.zeros(G, 3, device=dev, dtype=dtype); u_b[:, j] = 1.0
            out_b = hvp(u_b.reshape(-1), probe_id=j)[: G * 3].reshape(G, 3)
            col = torch.zeros(G, 3, device=dev, dtype=dtype)
            col.index_add_(0, fam, out_b)
            cols.append(col)
        H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))
```

New imports needed in `genewise_fit.py`: `from gpurec.solver.value_and_grad import forward_solve`
and `from gpurec.solver.hvp_exact import make_exact_hvp`.

**The `probe_id=j` on each probe is the detail most likely to get silently dropped** if this is
re-derived from scratch rather than copied — without it, the tangent-adjoint warm-start never
activates and the whole speed benefit disappears (correctness would be unaffected, just the
validated performance win).

### Optimization loop's Hessian rebuild

Currently (the FD block, inside the per-iteration `if it % 5 == 0 or Hd is None or ...:` guard):

```python
                if it % 5 == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
                    H = torch.zeros(sub.shape[0], 3, 3, device=dev, dtype=dtype)
                    for j in range(3):
                        tp = sub.clone(); tp[:, j] += fd_eps; _, gp = lg(m, tp)
                        H[:, :, j] = (gp - g) / fd_eps            # forward difference (reuse base g) -> 3 evals
                    H = 0.5 * (H + H.transpose(1, 2))
                    e, V = torch.linalg.eigh(H)
                    Hd = V @ torch.diag_embed(e.clamp(min=mu)) @ V.transpose(1, 2)   # convexify -> PD
```

Becomes:

```python
                if it % 5 == 0 or Hd is None or Hd.shape[0] != sub.shape[0]:
                    H = _analytic_hessian(m, sub, pi_cur)
                    e, V = torch.linalg.eigh(H)
                    Hd = V @ torch.diag_embed(e.clamp(min=mu)) @ V.transpose(1, 2)   # convexify -> PD
```

`g` (the base gradient, still computed via `lg(m, sub)` just above this block for the Newton
right-hand-side) is unaffected — only the Hessian source changes. `pi_cur` is already in scope (the
current pi-tier loop variable).

### Certify block's Hessian

Currently (central-difference FD, computed once over all families at the cert tier):

```python
            H = torch.zeros(F_all, 3, 3, device=dev, dtype=dtype)
            for j in range(3):
                tp = theta.clone(); tp[:, j] += fd_eps; _, gp = lg(mfull, tp)
                tm = theta.clone(); tm[:, j] -= fd_eps; _, gm = lg(mfull, tm)
                H[:, :, j] = (gp - gm) / (2 * fd_eps)
            H = 0.5 * (H + H.transpose(1, 2))
```

Becomes:

```python
            H = _analytic_hessian(mfull, theta, cert_pi)
```

(`cert_pi = max(pis)` is already computed earlier in the function and is what `mfull` was built
with, matching the tier the analytic HVP's `tangent_self_iters` should use.)

### Signature and `GENEWISE_REFERENCE` changes

- Remove `fd_eps: float = 1e-2,` from `fit_genewise`'s signature.
- Remove `fd_eps=1e-2,` from `GENEWISE_REFERENCE`'s dict literal.
- Remove the module docstring's mention of `fd_eps` if present, and update the docstring's
  description of step 2 ("forward-difference Hessian") to describe the analytic HVP instead.
- `mu` stays in both places, unchanged.

### `experiments/sanderson_cv/bench_genewise_warm_rebatch.py`

This file independently reimplements the same recipe (its own `FD_EPS = 1e-2` module constant, its
own inline FD Hessian construction inside its own optimization loop — no shared code with
`genewise_fit.py`; confirmed it imports nothing from `genewise_fit.py` today, by design — it's meant
to be a standalone reimplementation for benchmarking outside the library, same convention as the
earlier scratchpad prototype). Add a duplicate `_analytic_hessian` helper inline in this file too
(not imported from `genewise_fit.py`, to preserve that standalone-ness), replace its FD block with
the same swap, remove its `FD_EPS` constant.

## Testing plan

1. **Update existing signature/config-consistency tests** that currently assert on `fd_eps`:
   - `tests/test_reference_defaults.py` — check for any assertion tied to `fd_eps` via its
     `_check(fit_genewise, GENEWISE_REFERENCE)` helper; since both the signature default and the
     dict entry are removed together, this should stay consistent automatically, but verify.
   - `tests/test_config_rates.py::test_fit_genewise_signature_defaults_come_from_genewise_preset`
     and `::test_fit_genewise_still_uses_1e6_2p0_box` — confirm neither references `fd_eps`
     directly; if either does, update.
   - `tests/test_config_wiring.py` — its stubbed-`GeneReconModel` tests use `adam_steps=0`; confirm
     they don't exercise the Hessian-construction code path at all (if they do, they may need
     updating to not reference `fd_eps` either).

2. **Add a direct correctness test for `fit_genewise` itself** (doesn't exist today — current
   coverage is all signature/config-wiring, not behavior). Run it on a small real fixture (e.g.
   `tests/data/alerax/test_trees_200`, already used elsewhere in the genewise HVP tests) with a
   handful of families, confirm it converges (final `|Pg| < tol` via `certify=True`) and produces a
   finite loss.

3. **Re-run the archaea60 benchmark through the actual merged `fit_genewise`** (not the scratchpad
   prototype) — both the 200-family sanity check and the full 26-batch/5446-family run — to confirm
   the validated numbers still hold once this is the real production code path, not an isolated
   copy. Report actual measured wall-clock and convergence counts; do not assume the prototype's
   numbers carry over unchanged.

4. **Run the benchmark driver script** (`bench_genewise_warm_rebatch.py`) once at a modest scale
   (not necessarily the full 12408-family hogenom set) to confirm its swap also works correctly.

5. **Full test suite run** (`pytest tests/ -m gpu`) to catch anything else touching `fit_genewise`
   or `GENEWISE_REFERENCE` that wasn't anticipated.
