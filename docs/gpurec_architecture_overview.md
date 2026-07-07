# gpurec — Architecture Overview

_Snapshot: 2026-07-07. Package at `consolidate-release/gpurec/`._

## What it is

A GPU-accelerated **phylogenetic reconciliation engine**: it fits DTL
(Duplication / Transfer / Loss) rates by maximum likelihood, reconciling gene
trees against a species tree. The heavy math is custom **Triton kernels**
solving two coupled fixed points (extinction `E`, then per-wave `Pi`),
differentiated **implicitly** (adjoint method, not autograd-through-iterations).
Two Rust `.abi3.so` extensions handle preprocessing and backtracking.
Everything internal is in **log2 (bits)**.

## Size snapshot

| Scope | Python LOC | Files |
|---|---:|---:|
| **`gpurec/` package total** | **17,240** | 71 |
| `optim/` | 7,219 | 34 |
| `core/` (of which `kernels/` = 5,537) | 6,646 | 19 |
| `api/` | 1,703 | 6 |
| root (`batched_lbfgs.py` etc.) | 1,261 | 4 |
| `cli/` | 250 | 5 |
| `bench/` | 161 | 3 |

Plus: **~3,000 LOC markdown docs** (`docs/optim/`), a real **`tests/` dir at the
repo root** (48 files, ~3,986 LOC), and the Rust sources in `crates/`. Zero
TODO/FIXME/HACK markers — the code is mature and comment-clean.

## Subsystem map

**`api/` — the public surface (1,703 LOC).** `GeneReconModel` (nn.Module) is
what users touch: `loss = model()` → `_execution.stream_batches` runs the forward
solve, then `loss.backward()` fires the implicit VJP in `_implicit_grad.py` (the
adjoint core, with a matrix-free BiCGSTAB). `_autograd.py` bridges the
hand-written gradient into torch. Clean, well-factored.

**`core/inference/` — the solver (366 LOC).** `solve_resident_e_pi` = E fixed
point → Pi wave-forward → NLL. Small and tight.

**`core/kernels/` — the compute (5,537 LOC).** The engine. The naming pattern is
a **derivative trinity**: each primal kernel has a `_tangent` (forward-mode JVP)
and `_so` (second-order) variant:

| primal | JVP | 2nd-order |
|---|---|---|
| `e_step` | `e_step_tangent` | `e_step_so` |
| `wave_step` (+ `wave_backward`) | `wave_tangent` | `wave_so` |
| `dts_fused` | `dts_tangent` | `dts_so` |

That's exactly the value / gradient / Hessian-vector-product set. This structure
is intentional and good — it is the price of exact Newton on GPU.

**`optim/` — the fitting layer (7,219 LOC, the sprawl).** Two clusters:
**optimizers** (`optimize`, `newton_cg`, `cg`, `map_fit`, `map_cv`,
`genewise_fit`, `value_and_grad`) built on a single `make_value_and_grad`
foundation; and **curvature/HVP** (`hvp_exact`, `ggn`, and three `*_curvature`
modules).

**Root files.** `batched_lbfgs.py` (1,076 LOC) is a specialized per-row batched
L-BFGS for the genewise `[G,…]` case — does **not** overlap `optim/optimize.py`
(different problem shape). `distributed.py` is manual family-sharded
data-parallel. `optimization.py` is a thin re-export façade.

## Pruning & refactoring assessment (ranked by payoff)

Headline: **~13% of the package is test/research scaffolding that ships in the
wheel**, and the `optim/` curvature code has real triplicate duplication.

### P1 — Relocate scaffolding out of the package (~2,300 LOC, mechanical)
Runnable scripts with `__main__` and "gate/parity/FD-test" docstrings living
*inside* the importable package, even though a proper `tests/` exists at the
repo root:
- **11 `_verify_*` / `_parity_kbench` / `_fit_kbench` / `_test_first_order_recv`
  scripts** = 1,744 LOC. They only import each other; production never imports
  them.
- **`optim/diagnostics/` — 7 files, 558 LOC** — one-off "specieswise basin
  investigation (2026-06-15)" scripts with stale `python -m newton.…` paths.

### P2 — Unify the three `*_curvature.py` modules (1,346 LOC → est. ~700)
`receiver_curvature.py` (437), `origination_curvature.py` (345), and
`genewise_curvature.py` (564) are parallel copies of one pattern —
`build_joint_hvp` → gauge-project → Lanczos PD certificate → `newton_joint` —
for different parameter blocks `(θ,α)`, `(θ,α,ω)`, and per-family θ.
`origination_curvature` largely supersedes `receiver_curvature`.

### P3 — Deduplicate the backward
`ggn.py::vjp_root_to_theta` is, per its own docstring, a "faithful copy of
`implicit_grad_loglik_vjp_wave`" (in `api/_implicit_grad.py`);
`_e_adjoint_and_theta_vjp` is defined in both.

### P4 — Split `core/kernels/wave_backward.py`
2,446 LOC — the single largest file by 2×. Not duplication, just big; split the
retained-VJP fast path from the setup/layout code.

### P5 — Remove test-only code from a production module
`genewise_curvature.py::genewise_hessian_blocks` is docstring-flagged
TEST-ONLY ("production genewise fit never assembles blocks").

### Housekeeping
Stale `newton/` / `kbench` module references litter `optim/` docstrings and
`docs/optim/*.md` from the port. Two `# unused` locals in `wave_tangent.py`
(lines 247, 428).

**Bottom line:** the `core/` engine and `api/` are lean and well-architected.
All the weight is in `optim/`, which is ~⅓ scaffolding and has a triplicated
curvature framework. P1 + P2 alone would cut the package by ~2,500–3,000 LOC
(~15–17%) with no loss of capability.
