# Fraction-Missing (per-species leaf-observation boundary) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AleRax-style *fraction missing* (per-species probability that a gene is present but unobserved at a leaf) to gpurec, flowing through the forward likelihood, the first-order gradient, and the second-order (Newton/HVP) curvature, exposed via the Python API, config TOML, and CLI.

**Architecture:** A single per-species tensor `leaf_fm_log[S] = log2(fraction_missing_s)` (`−inf` at internal species and fully-observed leaves) is threaded, off-by-default, into two model boundaries: (1) the **E-step extinction** recurrence — at a species leaf the terminal speciation term becomes the single factor `p^S_l · fraction_missing_l`, realized as `E_s1 = log2(fm_l)`, `E_s2 = 0`; and (2) the **Pi reconciliation** recurrence — a leaf-species column that a clade does not map to carries an extra "present-but-unobserved" baseline `log_pS[s] + log2(fm_s)` instead of `−inf`. Both are pure additions inside existing Triton kernels; the default (`leaf_fm_log = None`) reproduces current behaviour byte-for-byte.

**Tech Stack:** Python 3.12, PyTorch, Triton kernels (`gpurec/core/kernels/*`), Rust preprocessor (`crates/gpurec-preprocess`, PyO3), `dataclasses` + `tomllib` config, `argparse` CLI, `pytest`.

## Global Constraints

- **No `clamp` / `clamp_` anywhere.** Compute `log2` only on strictly-positive entries via `torch.where(mask, torch.log2(x.masked_fill(~mask, 1.0)), NEG_INF)` (masked_fill to `1.0` → `log2(1)=0`, never NaN). This is a hard project rule (silent corruption).
- **All log-space is base-2 (`log2` / `exp2`).** New tensors are `log2`, `−inf`-guarded, no `clamp`.
- **Off-by-default and byte-identical.** With `fraction_missing` unset/zero, every code path must produce results bit-identical to today. `fraction_missing = 0.0` MUST equal the no-`fraction_missing` path exactly (regression gate in every phase).
- **`fraction_missing` is a fixed input, never a fitted parameter.** It carries no gradient; it only changes the forward map and therefore the gradient/Hessian w.r.t. the D/T/L rates `theta`.
- **`fraction_missing_s ∈ [0, 1)`.** Validate; reject `>= 1` or `< 0`. Internal (non-leaf) species always carry no missing term.
- **Species-tree leaf set** is `species_helpers["sp_child1"] == species_helpers["S"]` (the Rust sentinel; there is no leaf-mask key).
- **Test command:** `pytest -m "not gpu"` for CPU-only tests; GPU/kernel tests are marked `@pytest.mark.gpu` and run with `pytest -m gpu` on a CUDA box. Test sources live flat in `tests/` (they were dropped from the tree in a chore commit — new test files are created fresh here). Keep test-module basenames unique (no top-level `conftest.py`).
- **Do not touch** the `origination_weights` / survival-normalizer head: `fraction_missing` enters E and Pi only, and the survival term consumes the already-modified `E` automatically.

---

## Shared Contract (read once before any task)

### The math (for `fraction_missing_l = 1 − p_obs_l`)

- **E-step, at a species leaf `l`.** Today the terminal speciation term is `p^S_l · E_s1 · E_s2` with `E_s1 = E_s2 = −inf` (children are sentinels → no S-term). Fraction-missing replaces it with the single factor `p^S_l · fm_l`, i.e. set `E_s1 = log2(fm_l)`, `E_s2 = log2(1) = 0`. Terminal fixed point becomes:
  `E_l = logsumexp2( log_pS_l + log2(fm_l),  log_pD_l + 2·E_l,  E_l + Ebar_l,  log_pL_l )`.
- **Pi recurrence, at a leaf-species column `s`.** `Pi_{l,γ} = σ + (1−σ)·fm_s` where `σ = 1` for the clade mapped to leaf `s` (kept as the existing `leaf_hit` term `log_pS[s]`) and `σ = 0` otherwise. The `σ=0` (off-hit) leaf-species columns get baseline `log_pS[s] + log2(fm_s)` instead of `−inf`.

### The carrier tensor

`leaf_fm_log`: `torch.float`-tensor shape `[S]`, in `log2`, value `log2(fraction_missing_s)` for species-tree leaves with `fraction_missing_s > 0`, else `−inf`. `None` (the default) means "no missing → existing fast path". It is per-species (never per-family): the E-step and Pi kernels index it by species column `s_offs`, broadcasting across families/genes.

### Kernel work matrix (derived from the codebase scan)

| Concern | File : anchor | Change |
|---|---|---|
| E forward | `core/kernels/e_step.py:19` `_update_extinction_log_probabilities_kernel` | **MODIFY**: inject `E_s1=log2(fm)`, `E_s2=0` at leaf species |
| E VJP (1st-order) | `core/kernels/e_step.py:111` | **NONE** (reads *saved* `E_s1/E_s2`; grad_log_pS emerges automatically) |
| E tangent (JVP) | `core/kernels/e_step_tangent.py:15` | **MODIFY**: inject primal `E_s1=log2(fm)`, `E_s2=0`; tangents stay 0 |
| E second-order | `core/kernels/e_step_so.py:13` | **NONE** (reads *passed* `E_s1/E_s2/dE_s1/dE_s2`) |
| Pi forward | `core/kernels/pi_forward.py:547` (+ leaf initializer ~`:214`) | **MODIFY**: off-hit leaf baseline |
| Pi backward | `core/kernels/wave_backward_kernels.py:185` and `:725` (both kernels) | **MODIFY**: off-hit leaf baseline (grad_log_pS auto) |
| Pi tangent | `core/kernels/wave_tangent.py:202` and `:389` (both kernels) | **MODIFY**: off-hit leaf baseline + tangent |
| Pi second-order | `core/kernels/wave_so.py:215` | **MODIFY**: off-hit leaf baseline |
| DTS tangent/SO | `core/kernels/dts_tangent.py`, `dts_so.py` | **NONE** (no leaf boundary) |

### Integration touchpoints (thread `leaf_fm_log` through)

- `core/inference/solver.py:71,86` `solve_resident_e_pi` → `e_fixed_point_triton`, `pi_wave_forward`.
- `api/_batch_state.py:11` `_BatchStatic` (new field) + `:36` `build_batch_static`.
- `api/model.py:35` `GeneReconModel.__init__` (new `fraction_missing=` arg; build + store `self.leaf_fm_log`).
- `api/_autograd.py:33,103` `_GeneReconFunction.forward/backward` → read `static.leaf_fm_log`.
- `api/_implicit_grad.py:153` `implicit_grad_loglik_vjp_wave` → pass to the `solve_reconciliation_wave_vjp` call (`:332`). (The E-adjoint `_e_adjoint_and_theta_vjp` calls the E-step VJP, which needs nothing — see matrix — but the Pi backward needs it.)
- `solver/ggn.py:55` `vjp_root_to_theta` → forward into `implicit_grad_loglik_vjp_wave`.
- `solver/hvp_exact.py` → `forward_tangent.py` (`e_tangent_fixed_point`, `compute_wave_step_tangent*`), `wave_backward_so` (`:569`), `e_step_backward_so` (`:764/781/807`, no-op but pass for symmetry), and the direct `e_step_triton_autograd` callsites (`:303/343`).

### Validation oracles (no legacy path exists in this repo)

1. **fm=0 regression** — exact equality vs the no-fm path. Cheapest, strongest; every phase.
2. **E self-consistency** — the converged `E` at a leaf satisfies the terminal fixed point above to `|resid| < 1e-9`. Self-contained (needs only `e_fixed_point_triton` output). Phase 1.
3. **Pure-torch oracle** (Task 1.4) — a CPU reference E + Pi fixed point with the fm boundary, ported clamp-free from the upstream `oliver-schick/gpurec@fraction-missing` legacy path, used only in tests. Independent forward check.
4. **grad vs central finite-difference** — Phase 2.
5. **HVP vs finite-difference-of-gradient** (directional) — Phase 3.
6. **uniform == dense** pibar-mode agreement with fm active — Phases 1–2.

---

## Phase 0 — Foundation (host-side `leaf_fm_log`, no kernels yet)

### Task 0.1: Species-leaf mask + `build_fraction_missing_tensors`

**Files:**
- Create: `gpurec/core/parameters/fraction_missing.py`
- Test: `tests/test_fraction_missing_tensors.py`

**Interfaces:**
- Produces:
  - `species_leaf_mask(species_helpers: dict) -> torch.Tensor` → `[S]` bool, `True` at species-tree leaves (`sp_child1 == S`).
  - `build_fraction_missing_tensors(species_helpers: dict, *, fraction_missing, species_name_to_index: dict[str,int] | None = None, device, dtype) -> torch.Tensor | None` → `leaf_fm_log` `[S]` (`log2`, `−inf` off-leaf/observed) or `None`. Accepts `None | float | dict[str,float] | Sequence|Tensor[S]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fraction_missing_tensors.py
import math
import torch
import pytest
from gpurec.core.parameters.fraction_missing import (
    species_leaf_mask, build_fraction_missing_tensors,
)

NEG_INF = float("-inf")

def _tiny_species_helpers():
    # 3 species nodes in postorder: leaves 0,1 ; internal root 2. S=3.
    # sentinel for "no child" is S (==3). Root's children are leaves 0 and 1.
    S = 3
    return {
        "S": S,
        "sp_child1": torch.tensor([S, S, 0], dtype=torch.int32),
        "sp_child2": torch.tensor([S, S, 1], dtype=torch.int32),
        "sp_parent": torch.tensor([2, 2, -1], dtype=torch.int32),
    }

def test_leaf_mask_picks_leaves_via_sentinel():
    sh = _tiny_species_helpers()
    m = species_leaf_mask(sh)
    assert m.tolist() == [True, True, False]

def test_none_returns_none():
    sh = _tiny_species_helpers()
    assert build_fraction_missing_tensors(
        sh, fraction_missing=None, device="cpu", dtype=torch.float64) is None

def test_all_zero_returns_none():
    sh = _tiny_species_helpers()
    assert build_fraction_missing_tensors(
        sh, fraction_missing=0.0, device="cpu", dtype=torch.float64) is None

def test_scalar_applies_to_every_leaf_only():
    sh = _tiny_species_helpers()
    out = build_fraction_missing_tensors(
        sh, fraction_missing=0.3, device="cpu", dtype=torch.float64)
    assert out.shape == (3,)
    assert out[0].item() == pytest.approx(math.log2(0.3))
    assert out[1].item() == pytest.approx(math.log2(0.3))
    assert out[2].item() == NEG_INF  # internal species: never missing

def test_dict_by_name_maps_to_index():
    sh = _tiny_species_helpers()
    out = build_fraction_missing_tensors(
        sh, fraction_missing={"A": 0.5}, device="cpu", dtype=torch.float64,
        species_name_to_index={"A": 0, "B": 1})
    assert out[0].item() == pytest.approx(math.log2(0.5))
    assert out[1].item() == NEG_INF  # B absent from dict -> observed
    assert out[2].item() == NEG_INF

def test_tensor_per_species_ignores_internal():
    sh = _tiny_species_helpers()
    out = build_fraction_missing_tensors(
        sh, fraction_missing=[0.1, 0.2, 0.9], device="cpu", dtype=torch.float64)
    assert out[0].item() == pytest.approx(math.log2(0.1))
    assert out[1].item() == pytest.approx(math.log2(0.2))
    assert out[2].item() == NEG_INF  # index 2 is internal -> forced off

def test_out_of_range_raises():
    sh = _tiny_species_helpers()
    with pytest.raises(ValueError):
        build_fraction_missing_tensors(sh, fraction_missing=1.0, device="cpu", dtype=torch.float64)
    with pytest.raises(ValueError):
        build_fraction_missing_tensors(sh, fraction_missing=-0.1, device="cpu", dtype=torch.float64)

def test_no_clamp_used():
    import inspect, gpurec.core.parameters.fraction_missing as m
    assert "clamp" not in inspect.getsource(m)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fraction_missing_tensors.py -q`
Expected: FAIL (module `gpurec.core.parameters.fraction_missing` does not exist).

- [ ] **Step 3: Write minimal implementation**

```python
# gpurec/core/parameters/fraction_missing.py
"""Host-side construction of the per-species fraction-missing leaf boundary.

``fraction_missing_s = 1 - p_obs_s`` is the probability that a gene present in
species ``s`` is unobserved at that leaf. It enters the E-step and Pi recurrence
as a leaf boundary (see docs/superpowers/plans/2026-07-22-fraction-missing.md).
"""
from __future__ import annotations

import torch

NEG_INF = float("-inf")


def species_leaf_mask(species_helpers: dict) -> torch.Tensor:
    """[S] bool: True at species-tree leaves.

    A species node is a leaf iff its first child is the Rust sentinel ``S``
    (internal nodes get real child indices ``< S``).
    """
    S = int(species_helpers["S"])
    sp_child1 = species_helpers["sp_child1"]
    return sp_child1.to(torch.long) == S


def build_fraction_missing_tensors(
    species_helpers: dict,
    *,
    fraction_missing,
    species_name_to_index: dict[str, int] | None = None,
    device,
    dtype,
) -> torch.Tensor | None:
    """Map ``fraction_missing`` to ``leaf_fm_log`` ([S], log2, -inf off-leaf/observed).

    ``fraction_missing`` may be:
      * ``None``                       -> every gene observed (returns ``None``);
      * ``float``                      -> same value at every species-tree leaf;
      * ``dict {species_name: frac}``  -> per-leaf by name (absent names = observed);
      * length-S sequence/tensor       -> per-species (internal entries forced off).

    Returns ``None`` when the result is empty (all observed) so callers keep the
    no-overhead default path. Values must lie in ``[0, 1)``.
    """
    if fraction_missing is None:
        return None

    S = int(species_helpers["S"])
    leaf_mask = species_leaf_mask(species_helpers).to(device)
    fm = torch.zeros(S, dtype=dtype, device=device)

    if isinstance(fraction_missing, dict):
        if species_name_to_index is None:
            raise ValueError(
                "fraction_missing as a {name: frac} dict requires species_name_to_index "
                "(load it via gpurec.core.scheduling.batching.species_name_to_index)."
            )
        for name, val in fraction_missing.items():
            if name not in species_name_to_index:
                raise KeyError(f"fraction_missing species name {name!r} not in species tree")
            fm[int(species_name_to_index[name])] = float(val)
    elif isinstance(fraction_missing, (int, float)):
        fm[leaf_mask] = float(fraction_missing)
    else:
        t = torch.as_tensor(fraction_missing, dtype=dtype, device=device).reshape(-1)
        if t.numel() != S:
            raise ValueError(
                f"fraction_missing tensor has length {t.numel()}, expected S={S} "
                "(per-species) or pass a dict keyed by species name."
            )
        fm = t.clone()

    # Internal species never carry a missing term.
    fm = torch.where(leaf_mask, fm, torch.zeros((), dtype=dtype, device=device))

    fmax = float(fm.max())
    if fmax <= 0.0:
        return None  # nothing missing -> keep the fast default path
    if fmax >= 1.0 or float(fm.min()) < 0.0:
        raise ValueError("fraction_missing values must lie in [0, 1).")

    positive = fm > 0
    # Clamp-free log2: take log2 only on the strictly-positive branch; the
    # masked_fill(1.0) makes the off-branch log2(1)=0 (finite, never NaN), then
    # torch.where selects -inf there.
    leaf_fm_log = torch.where(
        positive,
        torch.log2(fm.masked_fill(~positive, 1.0)),
        torch.full_like(fm, NEG_INF),
    )
    return leaf_fm_log
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fraction_missing_tensors.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add gpurec/core/parameters/fraction_missing.py tests/test_fraction_missing_tensors.py
git commit -m "feat(fraction-missing): host-side leaf_fm_log builder + species leaf mask"
```

### Task 0.2: Expose species-name→index from the Rust preprocessor

**Files:**
- Modify: `gpurec/core/scheduling/batching.py` (add a thin Python wrapper)
- Test: `tests/test_species_name_index.py`

**Interfaces:**
- Consumes: the already-exported (but unused-from-Python) Rust pyfunction `species_leaf_index_map(species_path) -> JSON {name: gp_index}` (`crates/gpurec-preprocess/src/lib.rs:2288`).
- Produces: `species_name_to_index(species_tree_path: str) -> dict[str, int]` in `gpurec/core/scheduling/batching.py`, using the same post-order indexing as `preprocess_dataset`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_species_name_index.py
import pytest
from gpurec.core.scheduling.batching import species_name_to_index

def test_names_map_to_postorder_indices(tmp_path):
    sp = tmp_path / "species.nwk"
    sp.write_text("(A:1,B:1)Root;")
    m = species_name_to_index(str(sp))
    assert set(m) == {"A", "B", "Root"}
    assert m["A"] != m["B"]
    # Root is the last (highest) index in post-order.
    assert m["Root"] == max(m.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_species_name_index.py -q`
Expected: FAIL (`ImportError: cannot import name 'species_name_to_index'`).

- [ ] **Step 3: Write minimal implementation**

Add to `gpurec/core/scheduling/batching.py` (near the existing `preprocess_dataset`; reuse the same native module import already present at the top of that file — grep for the `import gpurec_preprocess` / `_native` symbol it already uses and call `.species_leaf_index_map`):

```python
import json as _json  # if not already imported

def species_name_to_index(species_tree_path: str) -> dict[str, int]:
    """{species_name: post-order gp index}, matching preprocess_dataset's ordering.

    Wraps the Rust ``species_leaf_index_map`` pyfunction. Use to resolve a
    ``{species_name: fraction_missing}`` dict onto the model's species indices.
    """
    # ``_native`` is the same module object preprocess_dataset already calls into.
    raw = _native.species_leaf_index_map(str(species_tree_path))
    return {str(k): int(v) for k, v in (raw.items() if isinstance(raw, dict) else _json.loads(raw).items())}
```

Note for the implementer: open `batching.py` first and match the exact native-module handle it already uses (the scan shows `preprocess_dataset` invokes `gpurec_preprocess` / `libgpurec_preprocess`); reuse that same handle rather than re-importing.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_species_name_index.py -q`
Expected: PASS. (If the native extension is not built in the environment, mark the test `@pytest.mark.gpu`-free but skip when the module is absent — follow the skip pattern already used by `preprocess_dataset` tests.)

- [ ] **Step 5: Commit**

```bash
git add gpurec/core/scheduling/batching.py tests/test_species_name_index.py
git commit -m "feat(fraction-missing): expose species_name_to_index from the Rust preprocessor"
```

### Task 0.3: Carry `leaf_fm_log` on `_BatchStatic` and the model

**Files:**
- Modify: `gpurec/api/_batch_state.py:11-33` (`_BatchStatic`), `:36-79` (`build_batch_static`)
- Modify: `gpurec/api/model.py:35-50` (`__init__` signature), `:110-128` (build + store)
- Test: `tests/test_model_fraction_missing_wiring.py`

**Interfaces:**
- Consumes: `build_fraction_missing_tensors` (Task 0.1), `species_name_to_index` (Task 0.2).
- Produces: `_BatchStatic.leaf_fm_log: torch.Tensor | None = None`; `GeneReconModel.__init__(..., fraction_missing=None)`; `self.leaf_fm_log`; every `static.leaf_fm_log` set to that tensor.

- [ ] **Step 1: Write the failing test** (CPU-safe: only checks wiring, not kernels)

```python
# tests/test_model_fraction_missing_wiring.py
import math, torch, pytest
from gpurec.api.model import GeneReconModel

def _tiny(tmp_path):
    sp = tmp_path / "species.nwk"; sp.write_text("(A:1,B:1)Root;")
    g = tmp_path / "g.nwk"; g.write_text("(A_1:1,B_1:1)GeneRoot;")
    return str(sp), [str(g)]

def test_default_leaf_fm_log_is_none(tmp_path):
    sp, g = _tiny(tmp_path)
    m = GeneReconModel(sp, g, mode="global", device="cpu")
    assert m.leaf_fm_log is None
    assert all(s.leaf_fm_log is None for s in m.batch_statics)

def test_scalar_fraction_missing_populates_statics(tmp_path):
    sp, g = _tiny(tmp_path)
    m = GeneReconModel(sp, g, mode="global", device="cpu", fraction_missing=0.3)
    assert m.leaf_fm_log is not None
    assert m.leaf_fm_log.shape == (int(m.species_helpers["S"]),)
    # leaves carry log2(0.3); internal carries -inf
    leaves = m.leaf_fm_log[torch.isfinite(m.leaf_fm_log)]
    assert torch.allclose(leaves, torch.full_like(leaves, math.log2(0.3)))
    assert all(s.leaf_fm_log is m.leaf_fm_log for s in m.batch_statics)
```

(If constructing a CPU model requires CUDA in this repo, mark both tests `@pytest.mark.gpu` and assert the same on a GPU box.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_fraction_missing_wiring.py -q`
Expected: FAIL (`__init__` has no `fraction_missing`; `.leaf_fm_log` missing).

- [ ] **Step 3: Implement — `_BatchStatic` field**

In `gpurec/api/_batch_state.py`, add the field to the dataclass (after `warm_scratch_reserved_bytes` at line 33):

```python
    # Per-species fraction-missing leaf boundary (log2(fraction_missing_s), -inf
    # off-leaf/observed). Shared across batches; None => every gene observed.
    leaf_fm_log: torch.Tensor | None = None
```

Add a parameter to `build_batch_static` (keyword, after `max_wave_size`):

```python
    max_wave_size: int,
    leaf_fm_log: torch.Tensor | None = None,
) -> _BatchStatic:
```

and pass it into the `_BatchStatic(...)` constructor (after `accumulator_dtype=accumulator_dtype,`):

```python
        accumulator_dtype=accumulator_dtype,
        leaf_fm_log=leaf_fm_log,
    )
```

- [ ] **Step 4: Implement — model construction**

In `gpurec/api/model.py`, add the constructor argument (after `per_family_origination: bool = False,` at line 49):

```python
        per_family_origination: bool = False,
        fraction_missing=None,
```

After `self.species_helpers = species_helpers` (line 110), build and store the tensor (needs `species_helpers` + the species tree path for name resolution):

```python
        from gpurec.core.parameters.fraction_missing import build_fraction_missing_tensors
        from gpurec.core.scheduling.batching import species_name_to_index as _species_name_to_index
        name_to_index = (
            _species_name_to_index(str(species_tree))
            if isinstance(fraction_missing, dict) else None
        )
        self.leaf_fm_log = build_fraction_missing_tensors(
            species_helpers,
            fraction_missing=fraction_missing,
            species_name_to_index=name_to_index,
            device=device,
            dtype=dtype,
        )
```

Pass it into `_make_batch_static`. Update `_make_batch_static` (line 170) to forward it:

```python
    def _make_batch_static(self, batch, plan=None, *, device: torch.device, max_wave_size: int) -> _BatchStatic:
        return build_batch_static(
            ...
            max_wave_size=max_wave_size,
            leaf_fm_log=self.leaf_fm_log,
        )
```

Ensure `self.leaf_fm_log` is assigned **before** the `self.batch_statics = [...]` comprehension at line 120 (move the build block above it).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_model_fraction_missing_wiring.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gpurec/api/_batch_state.py gpurec/api/model.py tests/test_model_fraction_missing_wiring.py
git commit -m "feat(fraction-missing): thread leaf_fm_log onto _BatchStatic and GeneReconModel"
```

---

## Phase 1 — Forward likelihood (E-step + Pi kernels)

> After Phase 1 the model computes the correct fraction-missing NLL but has no gradient yet. It is a natural review/merge checkpoint.

### Task 1.1: E-step forward kernel — leaf extinction injection

**Files:**
- Modify: `gpurec/core/kernels/e_step.py` — kernel `_update_extinction_log_probabilities_kernel:19`, launcher `_launch_e_step_forward_2d:307`, autograd `_TritonEStep2D:353`, wrapper `e_step_triton_autograd:484`, driver `e_fixed_point_triton:510`.
- Modify: `gpurec/core/inference/solver.py:71` (pass `leaf_fm_log`).
- Test: `tests/test_fraction_missing_e_step.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Consumes: `static.leaf_fm_log` (Task 0.3).
- Produces: `e_fixed_point_triton(..., leaf_fm_log: torch.Tensor | None = None)` and `e_step_triton_autograd(..., leaf_fm_log=None)`; converged `E` satisfies the fraction-missing terminal fixed point at leaves.

- [ ] **Step 1: READ FIRST** — reread `e_step.py:70-85` (the `c1/c2` load and `speciation_log_term`) and `:307-330` (the launcher's grid/constexpr wiring) so the new pointer + constexpr match the existing call convention.

- [ ] **Step 2: Write the failing test** (E self-consistency + fm=0 regression)

```python
# tests/test_fraction_missing_e_step.py
import math, torch, pytest
pytestmark = pytest.mark.gpu

from gpurec.core.kernels.e_step import e_fixed_point_triton

def _tiny_species():
    # S=3: leaves 0,1 ; root 2. sentinel = S = 3.
    S = 3
    dev = "cuda"; dt = torch.float64
    sp_parent = torch.tensor([2, 2, -1], dtype=torch.int32, device=dev)
    sp_child1 = torch.tensor([3, 3, 0], dtype=torch.int32, device=dev)
    sp_child2 = torch.tensor([3, 3, 1], dtype=torch.int32, device=dev)
    return S, dev, dt, sp_parent, sp_child1, sp_child2

def _run(leaf_fm_log):
    S, dev, dt, sp_parent, sp_child1, sp_child2 = _tiny_species()
    logp = lambda v: torch.full((1, S), math.log2(v), dtype=dt, device=dev)
    E0 = torch.full((1, S), torch.finfo(dt).min, dtype=dt, device=dev)
    E, E_s1, E_s2, Ebar = e_fixed_point_triton(
        E0, log_pS=logp(0.2), log_pD=logp(0.2), log_pL=logp(0.2),
        max_transfer=torch.zeros((1, S), dtype=dt, device=dev),
        receiver_log_probs=torch.full((S,), -math.log2(S), dtype=dt, device=dev),
        species_parent=sp_parent, species_child1=sp_child1, species_child2=sp_child2,
        max_ancestor_depth=3, max_iter=4000, tol=1e-13, use_receiver_weights=False,
        leaf_fm_log=leaf_fm_log,
    )
    return E, Ebar

def test_fm_zero_matches_no_fm_exactly():
    E_none, _ = _run(None)
    S = E_none.shape[-1]
    off = torch.full((S,), float("-inf"), dtype=E_none.dtype, device=E_none.device)  # all -inf => no missing
    E_fm0, _ = _run(off)
    assert torch.equal(E_none, E_fm0)

def test_leaf_self_consistency_with_fraction_missing():
    S, dev, dt, *_ = _tiny_species()
    fm = 0.3
    leaf_fm_log = torch.tensor([math.log2(fm), math.log2(fm), float("-inf")], dtype=dt, device=dev)
    E, Ebar = _run(leaf_fm_log)
    l = 0
    pS = math.log2(0.2); pD = math.log2(0.2); pL = math.log2(0.2)
    E_l = E[0, l]; Eb_l = Ebar[0, l]
    terms = torch.stack([
        torch.tensor(pS + math.log2(fm), dtype=dt, device=dev),  # p^S * fm_l  (single factor)
        pD + 2 * E_l,
        E_l + Eb_l,
        torch.tensor(pL, dtype=dt, device=dev),
    ])
    m = terms.max()
    rhs = m + torch.log2(torch.exp2(terms - m).sum())
    assert abs(float(rhs - E_l)) < 1e-9
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_fraction_missing_e_step.py -q`
Expected: FAIL (`e_fixed_point_triton` has no `leaf_fm_log`).

- [ ] **Step 4: Implement — kernel injection**

In `_update_extinction_log_probabilities_kernel` (`e_step.py:19`): add a pointer arg `leaf_fm_log_ptr` (after `species_child2_ptr`) and a constexpr `USE_FRACTION_MISSING: tl.constexpr` (after `USE_RECEIVER_WEIGHTS`). Replace the child load block at lines 70-75 so leaf species get the fm injection:

```python
    c1 = tl.load(species_child1_ptr + offs, mask=mask, other=-1)
    c2 = tl.load(species_child2_ptr + offs, mask=mask, other=-1)
    c1_valid = mask & (c1 >= 0) & (c1 < S)
    c2_valid = mask & (c2 >= 0) & (c2 < S)
    E_s1 = tl.load(E_ptr + base + c1, mask=c1_valid, other=neg_inf)
    E_s2 = tl.load(E_ptr + base + c2, mask=c2_valid, other=neg_inf)
    if USE_FRACTION_MISSING:
        # At a leaf species (child sentinels) with fraction_missing>0, the terminal
        # speciation term is the single factor p^S * fm_l: E_s1=log2(fm_l), E_s2=0.
        fm_log = tl.load(leaf_fm_log_ptr + offs, mask=mask, other=neg_inf)
        is_missing_leaf = mask & (fm_log > neg_inf)
        E_s1 = tl.where(is_missing_leaf, fm_log, E_s1)
        E_s2 = tl.where(is_missing_leaf, tl.zeros([BLOCK_S], dtype=DTYPE), E_s2)
```

The rest (`speciation_log_term = speciation_log_probability + E_s1 + E_s2`, line 85) is unchanged and now correct at leaves. `E_s1/E_s2` are stored as forward outputs (lines 103-104), so the first-order VJP and second-order kernels pick up the right values for free.

- [ ] **Step 5: Implement — launcher + Function + driver plumbing**

- `_launch_e_step_forward_2d:307`: add `leaf_fm_log=None` kwarg; pass `leaf_fm_log if leaf_fm_log is not None else <dummy 1-elem tensor>` as the new pointer and `USE_FRACTION_MISSING=leaf_fm_log is not None` to the kernel launch (`:327`). Use a `.contiguous()` `[S]` tensor; when `None`, pass a zero-size or `S`-length `−inf` placeholder guarded by the constexpr (the constexpr short-circuits the load, so a valid-but-unused pointer is enough — allocate `torch.empty(1)` and gate on the constexpr).
- `_TritonEStep2D:353`: add `leaf_fm_log` as a trailing input to `.apply(...)`; in `setup_context` it does **not** need saving (constant, no grad); in `backward` (`:389`, return tuple `:469-481`) append **one** `None` gradient slot for `leaf_fm_log`.
- `e_step_triton_autograd:484`: add `leaf_fm_log=None` and forward to `.apply`.
- `e_fixed_point_triton:510`: add `leaf_fm_log=None`; pass to every `_launch_e_step_forward_2d` call (`:559`).

- `gpurec/core/inference/solver.py:71`: pass `leaf_fm_log=getattr(static, "leaf_fm_log", None)` into `e_fixed_point_triton`.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_fraction_missing_e_step.py -q`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add gpurec/core/kernels/e_step.py gpurec/core/inference/solver.py tests/test_fraction_missing_e_step.py
git commit -m "feat(fraction-missing): E-step forward leaf-extinction injection"
```

### Task 1.2: Pi forward kernel — off-hit leaf baseline

**Files:**
- Modify: `gpurec/core/kernels/pi_forward.py` — main kernel `_update_reconciliation_wave_step_kernel` (leaf block `:547-555`, row leaf load `:470-478`), leaf-initializer kernel (`:214-268`) and its wrapper `compute_leaf_initial_wave_step:1063`, `compute_wave_step:1133`.
- Modify: `gpurec/core/inference/forward.py:13` `pi_wave_forward` (new arg) and its kernel-wrapper calls (`:120-141`, `:168-173`).
- Modify: `gpurec/core/inference/solver.py:86` (pass `leaf_fm_log`).
- Test: `tests/test_fraction_missing_forward.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Consumes: `static.leaf_fm_log`.
- Produces: `pi_wave_forward(..., leaf_fm_log=None)`; forward NLL with the Pi leaf baseline active.

- [ ] **Step 1: READ FIRST** — reread `pi_forward.py:470-484` (row leaf load + `leaf_frame_shift`) and `:547-555` (the leaf-term `tl.where`). Confirm whether the **leaf-initializer** kernel (`:214-268`) contributes to the same leaf-speciation term; if it only seeds leaf-clade rows (not the per-column baseline), it needs **no** change — verify by checking that the baseline is applied in the main wave-step recurrence (`:547`). Record the finding in the commit message.

- [ ] **Step 2: Write the failing test** — forward NLL: fm=0 regression + fm>0 effect + uniform==dense

```python
# tests/test_fraction_missing_forward.py
import torch, pytest
pytestmark = pytest.mark.gpu
from gpurec.api.model import GeneReconModel

def _tiny(tmp_path):
    sp = tmp_path / "species.nwk"; sp.write_text("((A:1,B:1)AB:1,C:1)Root;")
    g = tmp_path / "g.nwk"; g.write_text("((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;")
    return str(sp), [str(g)]

def _nll(model):
    with torch.no_grad():
        return float(model().item())

def test_forward_fm0_equals_no_fm(tmp_path):
    sp, g = _tiny(tmp_path)
    a = _nll(GeneReconModel(sp, g, mode="global", device="cuda", dtype=torch.float64))
    b = _nll(GeneReconModel(sp, g, mode="global", device="cuda", dtype=torch.float64, fraction_missing=0.0))
    assert a == b

def test_forward_fm_changes_nll(tmp_path):
    sp, g = _tiny(tmp_path)
    base = _nll(GeneReconModel(sp, g, mode="global", device="cuda", dtype=torch.float64))
    fm = _nll(GeneReconModel(sp, g, mode="global", device="cuda", dtype=torch.float64, fraction_missing=0.3))
    assert abs(fm - base) > 1e-6
```

(If `pibar_mode` is selectable on the constructor in this repo, add a `uniform == dense` cross-check with `fraction_missing=0.3`; otherwise defer it to Task 2.3 where both modes are exercised.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_fraction_missing_forward.py -q`
Expected: FAIL (either `pi_wave_forward` has no `leaf_fm_log`, or the NLL does not change).

- [ ] **Step 4: Implement — main wave-step kernel**

In `_update_reconciliation_wave_step_kernel`, add pointer `leaf_fm_log_ptr` + constexpr `USE_FRACTION_MISSING`. Replace the leaf block at `:547-555`:

```python
        if USE_LEAF_INDEX:
            leaf_hit = mask & (leaf_species == s_offs)
            mapped_term = leaf_observation_log_probability + leaf_frame_shift
            if USE_FRACTION_MISSING:
                # Off-hit leaf-species columns carry the "present-but-unobserved"
                # baseline log_pS[s] + log2(fm_s); non-leaf/observed columns stay -inf.
                leaf_logp_col = tl.load(leaf_logp_ptr + family_const * S + s_offs, mask=mask, other=NEG_LARGE)
                fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=mask, other=NEG_LARGE)
                baseline = leaf_logp_col + fm_col + leaf_frame_shift  # -inf where fm_col is -inf
                leaf_observation_log_term = tl.where(leaf_hit, mapped_term, baseline)
            else:
                leaf_observation_log_term = tl.where(leaf_hit, mapped_term, NEG_LARGE)
        else:
            leaf_observation_log_term = tl.full([BLOCK_S], value=NEG_LARGE, dtype=DTYPE)
```

- [ ] **Step 5: Implement — wrappers + forward + solver plumbing**

- `compute_wave_step:1133` and `compute_leaf_initial_wave_step:1063`: add `leaf_fm_log=None` kwarg; pass the pointer + `USE_FRACTION_MISSING=leaf_fm_log is not None` to the launch (mirror the `leaf_species_idx`/`leaf_logp` plumbing). If Step-1 found the initializer does not carry the baseline, still add the kwarg for signature symmetry but leave its kernel unchanged.
- `pi_wave_forward` (`forward.py:13`): add `leaf_fm_log: torch.Tensor | None = None`; forward it into the two wrapper calls (`:120-141`, `:168-173`).
- `solver.py:86`: pass `leaf_fm_log=getattr(static, "leaf_fm_log", None)` into `pi_wave_forward`.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_fraction_missing_forward.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add gpurec/core/kernels/pi_forward.py gpurec/core/inference/forward.py gpurec/core/inference/solver.py tests/test_fraction_missing_forward.py
git commit -m "feat(fraction-missing): Pi forward off-hit leaf baseline"
```

### Task 1.3: Pure-torch CPU oracle (test-only) for cross-validation

**Files:**
- Create: `tests/_fraction_missing_oracle.py` (test helper, not shipped code)
- Test: `tests/test_fraction_missing_oracle.py` (`@pytest.mark.gpu` — compares wave vs oracle)

**Interfaces:**
- Produces: `oracle_nll(species_nwk, gene_nwk, D, L, T, fraction_missing) -> float`, a clamp-free CPU port of the upstream legacy `E_fixed_point` + `Pi_fixed_point` (the fraction-missing math), used to independently confirm the wave NLL.

- [ ] **Step 1: Write the oracle helper** — port the E and Pi fixed points from `oliver-schick/gpurec@fraction-missing` (`gpurec/core/likelihood.py::E_step`/`E_fixed_point` and `gpurec/core/legacy.py::Pi_fixed_point`), reusing this repo's own preprocessor for the topology. Implement the two leaf boundaries exactly as Shared Contract, **clamp-free** (`torch.where` + `masked_fill`, never `clamp_min`). Keep it self-contained CPU float64.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_fraction_missing_oracle.py
import torch, pytest
pytestmark = pytest.mark.gpu
from gpurec.api.model import GeneReconModel
from tests._fraction_missing_oracle import oracle_nll

def test_wave_matches_cpu_oracle(tmp_path):
    sp = tmp_path / "s.nwk"; sp.write_text("((A:1,B:1)AB:1,C:1)Root;")
    g = tmp_path / "g.nwk"; g.write_text("((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;")
    D=L=T=0.1; fm=0.3
    ref = oracle_nll(str(sp), str(g), D, L, T, fm)
    m = GeneReconModel(str(sp), [str(g)], mode="global", device="cuda",
                       dtype=torch.float64, theta_init_rates=(D, L, T),
                       fraction_missing=fm)
    with torch.no_grad():
        got = float(m().item())
    assert abs(got - ref) < 1e-6
```

(Adjust `theta_init_rates=` to this repo's actual constructor knob for setting D/L/T; if none exists, set `m.theta` directly from `log2([D,L,T])` before the forward.)

- [ ] **Step 3: Run — expect the wave/oracle deltas to match** and fix the oracle until `|delta| < 1e-6`.

Run: `pytest tests/test_fraction_missing_oracle.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/_fraction_missing_oracle.py tests/test_fraction_missing_oracle.py
git commit -m "test(fraction-missing): CPU fixed-point oracle cross-checks the wave forward"
```

---

## Phase 2 — First-order gradient (Pi backward; E-step VJP is automatic)

> **Key result from the scan:** the E-step VJP and SO kernels consume *saved/passed* `E_s1/E_s2`, so once Task 1.1 makes the forward save `E_s1=log2(fm)`, `E_s2=0`, `grad_log_pS` at leaves is already exact — **no E-step VJP kernel edit**. Only the Pi backward needs the baseline mirrored (Task 2.1) plus threading (Task 2.2).

### Task 2.1: Pi backward kernels — mirror the off-hit leaf baseline

**Files:**
- Modify: `gpurec/core/kernels/wave_backward_kernels.py` — `_prepare_reconciliation_self_loop_vjp_kernel` (leaf block `:185-226`, params `:57-59`, constexprs `:74-76`) and `_accumulate_reconciliation_event_vjp_kernel` (leaf block `:725-749`, params `:607-609`, `grad_log_pS_ptr:612`).
- Modify: `gpurec/core/kernels/wave_backward.py` — `solve_reconciliation_wave_vjp:653` (+ `_solve_reconciliation_wave_vjp_2d:198`), both kernel launches (`:381-429`, `:579-629`).
- Test: `tests/test_fraction_missing_grad.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Consumes: `leaf_fm_log` passed by `implicit_grad_loglik_vjp_wave` (Task 2.2).
- Produces: `solve_reconciliation_wave_vjp(..., leaf_fm_log=None)`; the leaf baseline enters `local_event_scaled_mass` so `grad_log_pS` picks up the off-hit derivative automatically.

- [ ] **Step 1: READ FIRST** — reread both leaf blocks (`wave_backward_kernels.py:185-226` and `:725-749`) and the `grad_log_pS` accumulation at `:786-813`. Confirm the baseline's `∂/∂log_pS[s] = 1` means no new gradient buffer (Shared Contract). Note the `LEAF_LOGP_MODE` branch (`:189/192/195/202`): the baseline is well-defined per-column only in mode 0 (`[S]`) and mode 2 (`[G,S]`); for modes 1/3 the per-column `log_pS[s]` is unavailable, so **guard**: if `leaf_fm_log is not None and leaf_logp_mode in (1, 3)`, raise a clear `NotImplementedError` in `solve_reconciliation_wave_vjp` (fraction-missing requires per-species `log_pS`). Verify which mode the uniform/dense forward actually uses at this scale and record it.

- [ ] **Step 2: Write the failing test** (analytic grad vs central finite difference)

```python
# tests/test_fraction_missing_grad.py
import torch, pytest
pytestmark = pytest.mark.gpu
from gpurec.api.model import GeneReconModel

def _build(tmp_path, mode, fm):
    sp = tmp_path / "s.nwk"; sp.write_text("((A:1,B:1)AB:1,C:1)Root;")
    g = tmp_path / "g.nwk"; g.write_text("((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;")
    return GeneReconModel(str(sp), [str(g)], mode=mode, device="cuda",
                          dtype=torch.float64, fraction_missing=fm)

@pytest.mark.parametrize("mode", ["global", "specieswise"])
def test_grad_matches_central_fd_with_fraction_missing(tmp_path, mode):
    m = _build(tmp_path, mode, 0.3)
    loss = m(); loss.backward()
    g = m.theta.grad.detach().clone()
    eps = 1e-5
    with torch.no_grad():
        flat = m.theta.view(-1); i = 0
        base = flat[i].item()
        flat[i] = base + eps; fp = float(m().item())
        flat[i] = base - eps; fmv = float(m().item())
        flat[i] = base
    fd = (fp - fmv) / (2 * eps)
    ga = float(g.view(-1)[i])
    assert abs(fd - ga) / max(1.0, abs(fd)) < 1e-3
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_fraction_missing_grad.py -q`
Expected: FAIL (`solve_reconciliation_wave_vjp` has no `leaf_fm_log`, or grad mismatches FD).

- [ ] **Step 4: Implement — both backward kernels**

In each of the two leaf blocks, extend the `USE_LEAF_INDEX` branch so off-hit leaf-species columns carry the baseline instead of `NEG_LARGE`. For the per-species mode (`LEAF_LOGP_MODE == 0`), the existing `leaf_logp[:, None]` at `:203`/`:743` is already `log_pS[s]`; add `leaf_fm_log[s]`:

```python
    else:  # LEAF_LOGP_MODE == 0  (per-species [S])
        leaf_logp = tl.load(leaf_logp_ptr + s_offs, mask=species_valid, other=NEG_LARGE)
        if USE_FRACTION_MISSING:
            fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=species_valid, other=NEG_LARGE)
            baseline = (leaf_logp + fm_col)[:, None]  # -inf where fm_col is -inf
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[:, None], baseline)
        else:
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp[:, None], NEG_LARGE)
```

Mirror the analogous change for `LEAF_LOGP_MODE == 2` (`[G,S]`, `:195/:735`). The existing `+= leaf_offset_corr` frame correction (`:209/:749`) applies to the whole term unchanged. No new `grad_*` store is added — `grad_log_pS` at `:806/:813` receives the off-hit contribution because `leaf_observation_mass` now flows through `leaf_observation_event_vjp` (`:786`).

Add pointer `leaf_fm_log_ptr` + constexpr `USE_FRACTION_MISSING` to both kernel signatures (`:57-59`/`:74-76` and `:607-609`/`:630-637`).

- [ ] **Step 5: Implement — launcher plumbing**

In `wave_backward.py`: add `leaf_fm_log=None` to `solve_reconciliation_wave_vjp:653` and `_solve_reconciliation_wave_vjp_2d:198`; add the mode 1/3 guard (Step 1); pass the pointer + `USE_FRACTION_MISSING=leaf_fm_log is not None` to both launches (`:381-429`, `:579-629`).

- [ ] **Step 6: Run test to verify it passes** (after Task 2.2 wires it — if run standalone now it still fails at the API level; either implement 2.2 first or land 2.1+2.2 together and run the gate once).

- [ ] **Step 7: Commit**

```bash
git add gpurec/core/kernels/wave_backward_kernels.py gpurec/core/kernels/wave_backward.py
git commit -m "feat(fraction-missing): Pi backward off-hit leaf baseline (grad_log_pS auto)"
```

### Task 2.2: Thread `leaf_fm_log` through the implicit-grad + autograd path

**Files:**
- Modify: `gpurec/api/_implicit_grad.py:153` `implicit_grad_loglik_vjp_wave` (+ the `solve_reconciliation_wave_vjp` call `:332`).
- Modify: `gpurec/api/_autograd.py:134-166` (`backward` → pass `static.leaf_fm_log`).
- Modify: `gpurec/solver/ggn.py:55` `vjp_root_to_theta` (forward it).
- Test: reuse `tests/test_fraction_missing_grad.py` from Task 2.1.

**Interfaces:**
- Consumes: `static.leaf_fm_log`.
- Produces: `implicit_grad_loglik_vjp_wave(..., leaf_fm_log=None)` reaching the Pi backward.

- [ ] **Step 1: Implement**

- `implicit_grad_loglik_vjp_wave:153`: add keyword `leaf_fm_log: torch.Tensor | None = None`; pass it into the `solve_reconciliation_wave_vjp(...)` call at `:332`. (The E-adjoint `_e_adjoint_and_theta_vjp` and its `e_step_triton_autograd` calls need **nothing** — the E-step VJP reads saved `E_s1/E_s2`.)
- `_autograd.py` `_GeneReconFunction.backward` (`:134-166`): add `leaf_fm_log=static.leaf_fm_log` to the `implicit_grad_loglik_vjp_wave(...)` kwargs.
- `ggn.py` `vjp_root_to_theta:30`: add `leaf_fm_log=None` and forward it into `implicit_grad_loglik_vjp_wave` at `:55`; the caller (`make_ggn_hvp`, `hvp_exact.build_point_cache`) passes `static.leaf_fm_log`.

- [ ] **Step 2: Run the Task 2.1 gate**

Run: `pytest tests/test_fraction_missing_grad.py -q`
Expected: PASS (global + specieswise).

- [ ] **Step 3: Commit**

```bash
git add gpurec/api/_implicit_grad.py gpurec/api/_autograd.py gpurec/solver/ggn.py tests/test_fraction_missing_grad.py
git commit -m "feat(fraction-missing): thread leaf_fm_log through implicit-grad + autograd + ggn"
```

### Task 2.3: Genewise mode + uniform/dense regression sweep

**Files:**
- Test: `tests/test_fraction_missing_grad_modes.py` (`@pytest.mark.gpu`)

- [ ] **Step 1: Write the test** — grad-vs-FD for `mode="genewise"`, plus a `fraction_missing=0.0` exact-equality check of both loss AND `theta.grad` against the no-fm model, for all three modes. (Guards that the default path is still byte-identical after the backward edits.)

- [ ] **Step 2: Run**

Run: `pytest tests/test_fraction_missing_grad_modes.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fraction_missing_grad_modes.py
git commit -m "test(fraction-missing): genewise grad + fm=0 grad regression across modes"
```

---

## Phase 3 — Second-order (Newton / HVP)

> After Phase 3 the Newton fit recipes (`map_cv`, `genewise`) work with `fraction_missing`. Uses the JVP + SO kernels; DTS needs nothing.

### Task 3.1: E-step tangent (JVP) kernel — primal leaf injection

**Files:**
- Modify: `gpurec/core/kernels/e_step_tangent.py` — kernel `_update_extinction_log_probabilities_jvp_kernel:15` (primal `E_s1/E_s2` load `:110-117`), launcher `_launch_e_step_tangent_2d:205`, driver `e_tangent_fixed_point:255`.
- Test: `tests/test_fraction_missing_jvp.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Produces: `e_tangent_fixed_point(..., leaf_fm_log=None)`; the JVP primal `speciation_log_term` matches the forward at leaves (tangents stay 0 because `fm` is constant).

- [ ] **Step 1: READ FIRST** — reread `e_step_tangent.py:110-117` (primal + tangent child loads) and `:153,184` (primal/tangent speciation terms). Confirm only the **primal** `E_s1/E_s2` needs the fm injection; `dE_s1/dE_s2` remain 0 at leaves (sentinel path already yields `other=0.0`).

- [ ] **Step 2: Write the failing test** — directional JVP of `E` vs central FD of `E`, with fm active:

```python
# tests/test_fraction_missing_jvp.py
import math, torch, pytest
pytestmark = pytest.mark.gpu
from gpurec.core.kernels.e_step import e_fixed_point_triton
from gpurec.core.kernels.e_step_tangent import e_tangent_fixed_point
# Build E*(theta) on the tiny 3-species tree (reuse helpers from
# tests/test_fraction_missing_e_step.py), pick a direction dlog_pS, and assert
# e_tangent_fixed_point(...) ~= (E(theta+eps*d) - E(theta-eps*d)) / (2 eps),
# both with leaf_fm_log = log2(0.3) at the two leaves. rel tol 1e-3.
```

(Flesh out with the same `_tiny_species` setup as Task 1.1; compare `dE` against a central FD of `e_fixed_point_triton`.)

- [ ] **Step 3: Run — expect FAIL** (`e_tangent_fixed_point` has no `leaf_fm_log`; JVP primal wrong at leaves).

- [ ] **Step 4: Implement** — in `_update_extinction_log_probabilities_jvp_kernel`, add `leaf_fm_log_ptr` + `USE_FRACTION_MISSING`; after the primal child loads (`:116-117`), inject the same primal `E_s1=log2(fm)`, `E_s2=0` at leaf species as Task 1.1 (leave `dE_s1/dE_s2` untouched — they are already 0 at leaves). Thread through `_launch_e_step_tangent_2d:205` and `e_tangent_fixed_point:255`.

- [ ] **Step 5: Run test to verify it passes.** Commit:

```bash
git add gpurec/core/kernels/e_step_tangent.py tests/test_fraction_missing_jvp.py
git commit -m "feat(fraction-missing): E-step tangent primal leaf injection"
```

### Task 3.2: Pi tangent + second-order kernels — off-hit baseline

**Files:**
- Modify: `gpurec/core/kernels/wave_tangent.py` — kernels `_update_reconciliation_likelihood_jvp_kernel:28` (leaf block `:202-214`) and `_apply_reconciliation_self_loop_jvp_iterations_kernel:273` (leaf block `:389-401`); wrappers `compute_wave_step_tangent_selfloop:559`, `compute_wave_step_tangent:646`.
- Modify: `gpurec/core/kernels/wave_so.py` — kernel `_reconciliation_vjp_directional_derivative_kernel:18` (leaf block `:215-226`); wrapper `wave_backward_so:476`.
- Test: covered by Task 3.3's HVP-vs-FD gate.

**Interfaces:**
- Produces: `compute_wave_step_tangent*(..., leaf_fm_log=None)`, `wave_backward_so(..., leaf_fm_log=None)`; off-hit leaf columns carry the baseline in both the primal and (for tangent) `d_leaf` terms.

- [ ] **Step 1: READ FIRST** — reread the three leaf blocks. They already load `leaf_logp` and `d_leaf_logp` per-column (`wave_tangent.py:205-206`, `wave_so.py:218-219`). The baseline is `leaf_logp[s] + leaf_fm_log[s]` (primal) with tangent `d_leaf_logp[s]` (since `∂baseline/∂theta = ∂log_pS[s]/∂theta = d_leaf_logp[s]`; `log2(fm)` is constant). Confirm the frame-shift handling (`wave_tangent.py:208` adds `(0.0 - pi_offset)`) applies to the baseline too.

- [ ] **Step 2: Implement — tangent kernels.** In each tangent leaf block, extend the off-hit branch:

```python
    if USE_LEAF_INDEX:
        leaf_species = tl.load(leaf_species_ptr + ws + w)
        leaf_hit = mask & (leaf_species == s_offs)
        leaf_logp = tl.load(leaf_logp_ptr + family * S + s_offs, mask=mask, other=NEG)
        d_leaf_logp = tl.load(d_leaf_logp_ptr + family * S + s_offs, mask=mask, other=0.0)
        leaf_logp = leaf_logp + (0.0 - pi_offset).to(DTYPE)   # existing frame shift
        if USE_FRACTION_MISSING:
            fm_col = tl.load(leaf_fm_log_ptr + s_offs, mask=mask, other=NEG)
            baseline = leaf_logp + fm_col                      # -inf where fm_col is -inf
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, baseline)
            d_leaf_observation_log_term = tl.where(leaf_hit, d_leaf_logp, d_leaf_logp)
        else:
            leaf_observation_log_term = tl.where(leaf_hit, leaf_logp, NEG)
            d_leaf_observation_log_term = tl.where(leaf_hit, d_leaf_logp, zero)
```

(Note the off-hit tangent is `d_leaf_logp`, not `zero` — the baseline depends on `log_pS[s]`; where `fm_col` is `−inf` the primal is `−inf` so the term's mass is 0 and the tangent contribution vanishes regardless.) Add `leaf_fm_log_ptr` + `USE_FRACTION_MISSING` to both tangent kernels + wrappers.

- [ ] **Step 3: Implement — SO kernel.** In `_reconciliation_vjp_directional_derivative_kernel:215-226`, apply the same primal baseline (the SO kernel's `d_leaf_observation_log_term` mirrors the tangent). Add the pointer + constexpr to `wave_backward_so:476`.

- [ ] **Step 4: Commit** (the gate is Task 3.3):

```bash
git add gpurec/core/kernels/wave_tangent.py gpurec/core/kernels/wave_so.py
git commit -m "feat(fraction-missing): Pi tangent + second-order off-hit leaf baseline"
```

### Task 3.3: Thread `leaf_fm_log` through the HVP driver + gate

**Files:**
- Modify: `gpurec/solver/hvp_exact.py` — source `leaf_fm_log` near `:242`; pass to `jvp_root_scores`/`forward_tangent` (`:458`), `wave_backward_so` (`:569`), and the direct `e_step_triton_autograd` callsites (`:303,343`). `e_step_backward_so` (`:764/781/807`) takes no new arg (no-op) — leave unless Step-1 of 3.1 found otherwise.
- Modify: `gpurec/core/inference/forward_tangent.py` — pass `leaf_fm_log` into `e_tangent_fixed_point` (`:150`) and `compute_wave_step_tangent*` (`:277,342`); build the leaf tangent constants (`:118,194`) unchanged (the `d_leaf_logp` already = `dlog_pS`).
- Test: `tests/test_fraction_missing_hvp.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Consumes: `static.leaf_fm_log`.
- Produces: exact HVP with `fraction_missing` active.

- [ ] **Step 1: Write the failing test** — HVP vs central finite-difference of the gradient along a random direction `u`:

```python
# tests/test_fraction_missing_hvp.py
import torch, pytest
pytestmark = pytest.mark.gpu
from gpurec.api.model import GeneReconModel
from gpurec.solver.hvp_exact import make_exact_hvp_single  # adjust import to the real entry

def _grad(m):
    m.zero_grad(set_to_none=True); m().backward(); return m.theta.grad.detach().clone()

def test_hvp_matches_fd_of_grad(tmp_path):
    sp = tmp_path / "s.nwk"; sp.write_text("((A:1,B:1)AB:1,C:1)Root;")
    g = tmp_path / "g.nwk"; g.write_text("((A_1:1,B_1:1)x:1,C_1:1)GeneRoot;")
    m = GeneReconModel(str(sp), [str(g)], mode="specieswise", device="cuda",
                       dtype=torch.float64, fraction_missing=0.3)
    hvp = make_exact_hvp_single(m.batch_statics[0], m.theta.detach(), m.receiver_weights.detach())
    u = torch.randn_like(m.theta.view(-1)); u /= u.norm()
    hu = hvp(u).view(-1)
    eps = 1e-5
    with torch.no_grad():
        base = m.theta.detach().clone()
        m.theta.copy_((base.view(-1) + eps * u).view_as(base)); gp = _grad(m).view(-1)
        m.theta.copy_((base.view(-1) - eps * u).view_as(base)); gm = _grad(m).view(-1)
        m.theta.copy_(base)
    fd = (gp - gm) / (2 * eps)
    assert (hu - fd).norm() / max(1.0, fd.norm()) < 1e-2
```

(Adjust `make_exact_hvp_single`'s real signature/entry — confirm from `hvp_exact.py:185`.)

- [ ] **Step 2: Run — expect FAIL** (HVP does not see `leaf_fm_log`).

- [ ] **Step 3: Implement the threading** (hvp_exact.py + forward_tangent.py per Files).

- [ ] **Step 4: Run test to verify it passes.**

Run: `pytest tests/test_fraction_missing_hvp.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurec/solver/hvp_exact.py gpurec/core/inference/forward_tangent.py tests/test_fraction_missing_hvp.py
git commit -m "feat(fraction-missing): thread leaf_fm_log through the exact HVP driver"
```

### Task 3.4: fm=0 HVP regression + Newton smoke test

**Files:**
- Test: `tests/test_fraction_missing_newton.py` (`@pytest.mark.gpu`)

- [ ] **Step 1: Write the test** — (a) with `fraction_missing=0.0`, HVP output is bit-identical to the no-fm HVP for a fixed `u`; (b) a short Newton fit (few steps, via the real `fit`/`map_cv` entry) on the tiny tree with `fraction_missing=0.3` runs to completion and decreases the loss.

- [ ] **Step 2: Run**

Run: `pytest tests/test_fraction_missing_newton.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fraction_missing_newton.py
git commit -m "test(fraction-missing): HVP fm=0 regression + Newton fit smoke"
```

---

## Phase 4 — Config + CLI exposure

### Task 4.1: Config `[data] fraction_missing` scalar

**Files:**
- Create: `gpurec/config/data.py` (`DataOptions` dataclass)
- Modify: `gpurec/config/gpurec_config.py:108-121` (compose it), `:141-148`/`:73-105` (merge already generic)
- Modify: `gpurec/config/defaults.toml` (add `[data]` section)
- Modify: `gpurec/config/__init__.py` (export `DataOptions` if the package re-exports sub-configs)
- Test: `tests/test_config_data.py`

**Interfaces:**
- Produces: `DataOptions(fraction_missing: float = 0.0)`; `GpurecConfig().data.fraction_missing`. (A **scalar** global fraction only; per-species/name stays API/CLI-file — TOML has no clean per-species map here.)

- [ ] **Step 1: Write the failing tests** — mirror the existing config guard conventions:

```python
# tests/test_config_data.py
from gpurec.config import GpurecConfig
from gpurec.config.data import DataOptions

def test_default_data_options():
    assert GpurecConfig().data == DataOptions()
    assert GpurecConfig().data.fraction_missing == 0.0

def test_from_dict_sets_fraction_missing():
    c = GpurecConfig.from_dict({"data": {"fraction_missing": 0.2}})
    assert c.data.fraction_missing == 0.2

def test_unknown_data_key_raises():
    import pytest
    with pytest.raises(ValueError):
        GpurecConfig.from_dict({"data": {"nope": 1}})

def test_defaults_toml_still_matches_dataclass():
    from gpurec.config.gpurec_config import GpurecConfig as G, DEFAULTS_TOML_PATH
    assert G.from_toml(DEFAULTS_TOML_PATH) == G()
```

- [ ] **Step 2: Run — expect FAIL** (`DataOptions` missing; `defaults.toml` lacks `[data]`).

- [ ] **Step 3: Implement**

```python
# gpurec/config/data.py
from dataclasses import dataclass

@dataclass
class DataOptions:
    """Dataset-level options folded into GpurecConfig.

    fraction_missing: global UndatedDTL fraction-missing (1 - p_obs) applied to
    every species-tree leaf. 0.0 => every gene observed. Per-species / per-name
    fractions are set via the Python API or the CLI --fraction-missing-file.
    """
    fraction_missing: float = 0.0

    def validate(self) -> None:
        if not (0.0 <= self.fraction_missing < 1.0):
            raise ValueError("data.fraction_missing must lie in [0, 1).")
```

Compose in `gpurec_config.py`: import `DataOptions`, add `data: DataOptions = field(default_factory=DataOptions)` to `GpurecConfig`, and call `self.data.validate()` in `GpurecConfig.validate`. Add to `defaults.toml`:

```toml
[data]
fraction_missing = 0.0
```

- [ ] **Step 4: Run tests to verify they pass** (including the existing `tests/test_config_toml.py` guard if present — re-add it if the suite needs it).

Run: `pytest tests/test_config_data.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gpurec/config/data.py gpurec/config/gpurec_config.py gpurec/config/defaults.toml gpurec/config/__init__.py tests/test_config_data.py
git commit -m "feat(fraction-missing): [data] fraction_missing config field + defaults mirror"
```

### Task 4.2: Consume config `fraction_missing` in the constructor

**Files:**
- Modify: `gpurec/api/model.py` (resolve `config.data.fraction_missing` when the explicit `fraction_missing=` arg is unset)
- Test: `tests/test_model_config_fraction_missing.py`

**Interfaces:**
- Precedence: explicit `fraction_missing=` arg > `config.data.fraction_missing` > `None`.

- [ ] **Step 1: Write the failing test** — a model built with `config=GpurecConfig.from_dict({"data": {"fraction_missing": 0.3}})` and no explicit arg has non-`None` `leaf_fm_log`; an explicit `fraction_missing=0.0` overrides the config back to `None`.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — in `__init__`, before building `self.leaf_fm_log`:

```python
        if fraction_missing is None and config is not None:
            cfg_fm = getattr(getattr(config, "data", None), "fraction_missing", 0.0)
            fraction_missing = cfg_fm if cfg_fm else None
```

- [ ] **Step 4: Run test to verify it passes. Commit:**

```bash
git add gpurec/api/model.py tests/test_model_config_fraction_missing.py
git commit -m "feat(fraction-missing): resolve fraction_missing from config.data in the constructor"
```

### Task 4.3: CLI flags `--fraction-missing` / `--fraction-missing-file`

**Files:**
- Modify: `gpurec/cli/_common.py:13-29` (add args), `:69-80` (`build_model` passes it)
- Modify: `gpurec/cli/fit.py:63-110` (`run_fit` passes it to `fit_dtl`) and the `fit` recipe entry if it constructs the model
- Test: `tests/test_cli_fraction_missing.py`

**Interfaces:**
- `--fraction-missing FLOAT` → scalar; `--fraction-missing-file PATH` → TSV `species_name<TAB>frac` parsed to a dict. Mutually exclusive; both optional.

- [ ] **Step 1: Write the failing test** — argparse round-trip: `build_parser().parse_args([...])` accepts the flags; `build_model` receives the resolved `fraction_missing`. Use a fake tiny tree; assert `model.leaf_fm_log is not None` when `--fraction-missing 0.3` is given, and equals the config path otherwise.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — add the two args (mutually exclusive group) in `_common.add_common_args`; a small `resolve_fraction_missing(args)` returning `None | float | dict` (parse the TSV file clamp-free); thread into `GeneReconModel(..., fraction_missing=...)` at `_common.build_model:76-79` and into `fit.run_fit`'s `fit_dtl(...)` call (add a `fraction_missing=` param to the recipe if needed).

- [ ] **Step 4: Run test to verify it passes. Commit:**

```bash
git add gpurec/cli/_common.py gpurec/cli/fit.py tests/test_cli_fraction_missing.py
git commit -m "feat(fraction-missing): CLI --fraction-missing / --fraction-missing-file"
```

---

## Phase 5 — Integration & docs

### Task 5.1: End-to-end fit regression

**Files:**
- Test: `tests/test_fraction_missing_e2e.py` (`@pytest.mark.gpu`)

- [ ] **Step 1: Write the test** — a full `fit` on a small multi-family fixture with `fraction_missing=0.3`: runs to completion, loss decreases, recovered `theta` differs from the `fraction_missing=0` fit (sanity that it changes inference), and `fraction_missing=0.0` reproduces the baseline fit trajectory bit-for-bit at step 0.

- [ ] **Step 2: Run**

Run: `pytest tests/test_fraction_missing_e2e.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fraction_missing_e2e.py
git commit -m "test(fraction-missing): end-to-end fit regression"
```

### Task 5.2: Documentation

**Files:**
- Modify: `docs/gpurec_architecture_overview.md` (a "Fraction missing" subsection), `gpurec/api/README.md` (the `fraction_missing=` arg), `docs/config_convention.md` (the `[data]` section).

- [ ] **Step 1:** Document the model (both leaf boundaries + the `1 − p_obs` convention), the three API input forms, the config scalar limitation (global-only), and the CLI flags. Cross-link this plan.

- [ ] **Step 2: Commit**

```bash
git add docs/gpurec_architecture_overview.md gpurec/api/README.md docs/config_convention.md
git commit -m "docs(fraction-missing): model, API, config, and CLI usage"
```

---

## Self-Review

**Spec coverage vs the chosen scope (Full Newton/HVP · all three input forms · API+config+CLI):**
- Full Newton/HVP → Phases 1 (forward), 2 (first-order), 3 (JVP+SO+HVP). ✓ DTS explicitly excluded (no leaf boundary). ✓ E-step VJP/SO explicitly need no edit (consume saved values) — documented in the matrix and Phase 2 header. ✓
- All three input forms → Task 0.1 (`float | dict | tensor`) + Task 0.2 (`species_name_to_index` for the dict). ✓
- API+config+CLI → Task 0.3 (API arg), Tasks 4.1–4.2 (config), Task 4.3 (CLI). ✓
- Validation with no legacy path → oracles enumerated in Shared Contract; used in Tasks 1.1 (self-consistency), 1.3 (CPU oracle), 2.1 (grad-FD), 3.1 (JVP-FD), 3.3 (HVP-FD). ✓
- No-clamp rule → Task 0.1 uses `masked_fill`, asserted by `test_no_clamp_used`; oracle (1.3) and CLI file parse (4.3) called out clamp-free. ✓
- Off-by-default/byte-identical → regression gates in Tasks 1.1, 1.2, 2.3, 3.4, 5.1. ✓

**Type/name consistency:** the carrier is `leaf_fm_log` (`[S]`, log2, `−inf` off) everywhere; the new kernel constexpr is `USE_FRACTION_MISSING` everywhere; the new `_BatchStatic` field and model attribute are both `leaf_fm_log`. The builder is `build_fraction_missing_tensors`; the name map is `species_name_to_index`.

**Open items the implementer MUST confirm against live code (flagged in-task, not placeholders):**
1. `LEAF_LOGP_MODE` actually used by the uniform/dense forward at scale (Task 2.1 Step 1) — the mode 1/3 guard depends on it.
2. The Pi leaf **initializer** kernel's role (Task 1.2 Step 1) — whether it needs the baseline or only the main wave-step kernel does.
3. Real constructor knob for setting D/L/T in tests (`theta_init_rates=` vs direct `m.theta` write) — Tasks 1.3, 3.x.
4. Exact `make_exact_hvp_single` entry signature (Task 3.3) — from `hvp_exact.py:185`.
5. Whether CPU model construction is possible or every kernel test must be `@pytest.mark.gpu` (Tasks 0.3, 1.x).

---

## Notes

- **Early merge point:** Phases 0–2 deliver a fully usable *first-order* fraction-missing feature (forward + gradient), validated end-to-end. If Newton support is deferred, add a guard in the Newton/`map_cv` entry that raises when `static.leaf_fm_log is not None` until Phase 3 lands, so no silent wrong-Hessian fit is possible.
- **Reference:** upstream `oliver-schick/gpurec@fraction-missing` (commits `1f3efc0`, `03bced3`, by G. Szöllősi + Claude) is the pure-torch reference for the math; it stops at first-order and uses a dense `clade_species_map` this repo does not have — hence this in-kernel port.
