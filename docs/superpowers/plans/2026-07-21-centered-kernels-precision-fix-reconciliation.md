# Centered-Kernels Precision-Fix Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a complete, already-tested-but-uncommitted precision fix (dual fp32/fp64
materialization of preprocessed split-probability statics, replacing per-call-site `.to(dtype)`
casts) plus a closed-form origination-gradient rewrite onto `dev`'s current tip, reconciling it
against identifier-renaming commits (`1605fc5d`/`d5af8a10`) that landed on `dev` after the WIP
forked off.

**Architecture:** This is a **reconciliation task, not new design**. The WIP (found complete and
passing 97/97 tests in a stale sibling worktree, `gpurec-centered-kernels`, forked from `dev` at
commit `2000e16b`) already fully specifies the target behavior. `dev`'s tip has since renamed
nearly every local variable/kwarg in the touched kernel files (cosmetic, no behavior change) and
independently fixed one unrelated bug (a tangent-default in `wave_tangent.py`). Every task below
re-derives the WIP's semantic change using **dev's current identifier names**, verified file-by-file
against the actual current source (not assumed from a rename table). The clean (non-conflicting)
portion of the WIP — docs, `_batch_state.py`, `logspace.py`, `solver.py`'s origination-gradient
rewrite, `batching.py`, and 7 test files — is **already committed** as `bb7aed8a` on branch
`land-centered-kernels-precision-fix` in worktree
`/home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port`. This plan covers the 11
remaining files that had real merge conflicts.

**Tech Stack:** PyTorch, Triton kernels, Python.

## Global Constraints

- **Transcribe code blocks verbatim.** Every "New" code block below already uses dev's current,
  correct identifier names (verified against the live file, not inferred from a rename table). Do
  not "helpfully" restore an old name you recognize from the WIP's original design, and do not
  invent substitute names — if a block looks unfamiliar, that is dev's actual current code, not an
  error in the plan.
- **Two files contain a silent-trap risk**: `wave_backward.py` (Task 10) and `wave_so.py` (Task 8)
  each had a spot where a naive patch application would reference undefined (stale, pre-rename)
  identifiers with **zero merge-tool conflict** (the surrounding context matched, so the hunk would
  "apply cleanly" even though the referenced names don't exist) — this raises `NameError` only at
  first real invocation, never at import time or via `git apply --check`. These two tasks carry
  extra emphasis for this reason; there is nothing exotic about the actual fix, just heightened
  care in transcription.
- **Order matters.** Task 1 (`pi_forward.py`) defines two new helpers,
  `_select_log_split_probs(meta, dtype)` and `_validate_residual_tensors(reference, /, **tensors)`,
  that Tasks 2–10 import and call. Do Task 1 first.
- **Do not touch** `gpurec/core/kernels/wave_backward_kernels.py`'s 11 arithmetic-expression
  `.to(DTYPE)` casts (frame/offset corrections) — only casts directly suffixing a `tl.load(...)`
  call are in scope for that file (Task 11).
- **All work happens in** the existing worktree/branch
  `/home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port`
  (branch `land-centered-kernels-precision-fix`, currently at commit `bb7aed8a`). Do not create a
  new worktree.
- **GPU test marker**: this repo's tests use `pytest.mark.gpu` (`markers = ["gpu: requires CUDA +
  Triton (deselect with -m 'not gpu')"]` in `pyproject.toml`). A CUDA GPU (RTX 4090) is available in
  this environment — do not skip GPU tests.
- Each task's edits are surgical: only the shown code blocks change. Do not reformat, rename, or
  "clean up" surrounding code you were not asked to touch.

---

### Task 1: `gpurec/core/kernels/pi_forward.py` — define the two shared helpers

**Files:**
- Modify: `gpurec/core/kernels/pi_forward.py`

**Interfaces:**
- Produces: `_select_log_split_probs(meta, dtype)` — returns the preprocessing-owned
  split-probability tensor already materialized at the requested residual dtype (falls back to
  `meta.get("log_split_probs")`, or zeros keyed off `meta["sl"]`, if no per-dtype variant is
  present under `meta["_log_split_probs_by_dtype"]`).
- Produces: `_validate_residual_tensors(reference, /, **tensors)` — raises `TypeError`/`ValueError`
  if any non-`None` tensor's dtype/device doesn't match `reference`'s. `None` values are skipped
  (not required).
- Consumed by: Tasks 2–10 (`dts_so.py`, `dts_tangent.py`, `_implicit_grad.py`, `forward.py`,
  `forward_tangent.py`, `hvp_exact.py`, `wave_so.py`, `wave_tangent.py`, `wave_backward.py`).

- [ ] **Step 1: Update `__all__`**

  Before (lines 5-13):
  ```python
  __all__ = [
      "compute_dts_forward",
      "compute_leaf_initial_wave_step",
      "compute_wave_step",
      "_load_event_log_probability",
      "_prepare_wave_launch",
      "_tl_float_dtype",
      "_validate_offset_tensor",
  ]
  ```

  After:
  ```python
  __all__ = [
      "compute_dts_forward",
      "compute_leaf_initial_wave_step",
      "compute_wave_step",
      "_load_event_log_probability",
      "_prepare_wave_launch",
      "_select_log_split_probs",
      "_tl_float_dtype",
      "_validate_offset_tensor",
      "_validate_residual_tensors",
  ]
  ```

- [ ] **Step 2: Insert the two new helpers** between `_validate_offset_tensor` and
      `_prepare_wave_launch`

  Current (lines ~28-45):
  ```python
  def _validate_offset_tensor(
      name, value, *, rows, device, dtype=None, residual_dtype=None
  ):
      """Return a contiguous row-offset tensor after structural dtype checks."""
      if value.ndim != 1 or int(value.shape[0]) != int(rows):
          raise ValueError(f"{name} must have shape [{int(rows)}]")
      if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
          raise TypeError(f"{name} must use torch.float32 or torch.float64")
      if residual_dtype == torch.float64 and value.dtype != torch.float64:
          raise TypeError(f"{name} must not be narrower than the residual dtype")
      if dtype is not None and value.dtype != dtype:
          raise TypeError(f"{name} must match accumulator dtype {dtype}")
      if value.device != device:
          raise ValueError(f"{name} must be on {device}")
      return value.contiguous()


  def _prepare_wave_launch(S: int, const_tensor) -> tuple[int, int]:
  ```

  New (insert between them, `_validate_offset_tensor`'s body unchanged):
  ```python
  def _validate_offset_tensor(
      name, value, *, rows, device, dtype=None, residual_dtype=None
  ):
      """Return a contiguous row-offset tensor after structural dtype checks."""
      if value.ndim != 1 or int(value.shape[0]) != int(rows):
          raise ValueError(f"{name} must have shape [{int(rows)}]")
      if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
          raise TypeError(f"{name} must use torch.float32 or torch.float64")
      if residual_dtype == torch.float64 and value.dtype != torch.float64:
          raise TypeError(f"{name} must not be narrower than the residual dtype")
      if dtype is not None and value.dtype != dtype:
          raise TypeError(f"{name} must match accumulator dtype {dtype}")
      if value.device != device:
          raise ValueError(f"{name} must be on {device}")
      return value.contiguous()


  def _validate_residual_tensors(reference, /, **tensors) -> None:
      """Require dense kernel tensors to share the model dtype and device.

      Centered offsets are intentionally excluded: they use the accumulator
      dtype and are validated separately by :func:`_validate_offset_tensor`.
      """
      if not torch.is_tensor(reference):
          raise TypeError("residual reference must be a tensor")
      if reference.dtype not in _SUPPORTED_FLOAT_DTYPES:
          raise TypeError("residual tensors must use torch.float32 or torch.float64")
      for name, value in tensors.items():
          if value is None:
              continue
          if not torch.is_tensor(value):
              raise TypeError(f"{name} must be a tensor")
          if value.dtype != reference.dtype:
              raise TypeError(
                  f"{name} must match residual dtype {reference.dtype}, got {value.dtype}"
              )
          if value.device != reference.device:
              raise ValueError(f"{name} must be on {reference.device}")


  def _select_log_split_probs(meta, dtype):
      """Return preprocessing-owned split probabilities for a kernel dtype."""
      variants = meta.get("_log_split_probs_by_dtype")
      if variants is not None:
          try:
              return variants[dtype]
          except KeyError as exc:
              raise TypeError(f"no split-probability tensor for residual dtype {dtype}") from exc
      value = meta.get("log_split_probs")
      if value is None and "sl" in meta:
          value = torch.zeros(
              int(meta["sl"].numel()), device=meta["sl"].device, dtype=dtype
          )
      return value


  def _prepare_wave_launch(S: int, const_tensor) -> tuple[int, int]:
  ```

  (`meta`'s dict keys, e.g. `"sl"`, `"_log_split_probs_by_dtype"`, `"log_split_probs"`, were never
  renamed by dev's identifier pass — only Python variable/kwarg names were — so this helper needs no
  adaptation beyond what's shown.)

- [ ] **Step 3: `_initialize_leaf_reconciliation_likelihood_kernel`** (was
      `_leaf_initial_wave_step_kernel`) — 2 cast removals

  Current (lines 193-197):
  ```python
      if USE_RECEIVER_WEIGHTS:
          leaf_receiver_log_probability = tl.load(receiver_log_probs_ptr + leaf_species).to(DTYPE)
      else:
          leaf_receiver_log_probability = tl.zeros((), dtype=DTYPE)
      leaf_observation_log_probability = tl.load(leaf_logp_ptr + family * S + leaf_species).to(DTYPE)
  ```

  New:
  ```python
      if USE_RECEIVER_WEIGHTS:
          leaf_receiver_log_probability = tl.load(receiver_log_probs_ptr + leaf_species)
      else:
          leaf_receiver_log_probability = tl.zeros((), dtype=DTYPE)
      leaf_observation_log_probability = tl.load(leaf_logp_ptr + family * S + leaf_species)
  ```

- [ ] **Step 4: `_update_reconciliation_likelihood_kernel`** (was `_wave_step_kernel`) — 2 cast
      removals

  Current (lines ~404-411):
  ```python
          reconciliation_residual_shift = tl.max(
              tl.where(
                  shift_source != NEG_LARGE,
                  shift_source,
                  tl.zeros_like(shift_source),
              ),
              axis=0,
          ).to(DTYPE)
  ```

  New:
  ```python
          reconciliation_residual_shift = tl.max(
              tl.where(
                  shift_source != NEG_LARGE,
                  shift_source,
                  tl.zeros_like(shift_source),
              ),
              axis=0,
          )
  ```

  Current (lines ~430-433):
  ```python
      if USE_LEAF_INDEX:
          leaf_species = tl.load(leaf_species_ptr + global_row)
          leaf_observation_log_probability = tl.load(
              leaf_logp_ptr + family_const * S + leaf_species
          ).to(DTYPE)
  ```

  New:
  ```python
      if USE_LEAF_INDEX:
          leaf_species = tl.load(leaf_species_ptr + global_row)
          leaf_observation_log_probability = tl.load(
              leaf_logp_ptr + family_const * S + leaf_species
          )
  ```

- [ ] **Step 5: `compute_leaf_initial_wave_step`** — insert a validate call at the top

  Current (lines 1022-1050):
  ```python
  def compute_leaf_initial_wave_step(
      Pi_out,
      Pi_out_offset,
      ws,
      W,
      S,
      max_transfer_mat,
      duplication_loss_const,
      Ebar,
      E,
      speciation_child1_const,
      speciation_child2_const,
      receiver_log_probs,
      species_child1,
      species_child2,
      species_subtree_start,
      species_subtree_end,
      leaf_species_idx,
      leaf_logp,
      family_idx,
      use_receiver_weights=True,
  ):
      Pi_out_offset = _validate_offset_tensor(
          "Pi_out_offset",
          Pi_out_offset,
          rows=Pi_out.shape[0],
          device=Pi_out.device,
          residual_dtype=Pi_out.dtype,
      )
  ```

  New (insert the validate call before `Pi_out_offset = _validate_offset_tensor(...)`):
  ```python
  def compute_leaf_initial_wave_step(
      Pi_out,
      Pi_out_offset,
      ws,
      W,
      S,
      max_transfer_mat,
      duplication_loss_const,
      Ebar,
      E,
      speciation_child1_const,
      speciation_child2_const,
      receiver_log_probs,
      species_child1,
      species_child2,
      species_subtree_start,
      species_subtree_end,
      leaf_species_idx,
      leaf_logp,
      family_idx,
      use_receiver_weights=True,
  ):
      _validate_residual_tensors(
          Pi_out,
          max_transfer_mat=max_transfer_mat,
          duplication_loss_const=duplication_loss_const,
          Ebar=Ebar,
          E=E,
          speciation_child1_const=speciation_child1_const,
          speciation_child2_const=speciation_child2_const,
          receiver_log_probs=receiver_log_probs,
          leaf_logp=leaf_logp,
      )
      Pi_out_offset = _validate_offset_tensor(
          "Pi_out_offset",
          Pi_out_offset,
          rows=Pi_out.shape[0],
          device=Pi_out.device,
          residual_dtype=Pi_out.dtype,
      )
  ```

- [ ] **Step 6: `compute_wave_step`** — insert a validate call at the top

  Current (lines 1081-1116):
  ```python
  def compute_wave_step(
      Pi_in,
      Pi_in_offset,
      Pi_out,
      Pi_out_offset,
      Pibar,
      Pibar_offset,
      ws,
      W,
      S,
      max_transfer_mat,
      duplication_loss_const,
      Ebar,
      E,
      speciation_child1_const,
      speciation_child2_const,
      receiver_log_probs,
      species_child1,
      species_child2,
      species_parent,
      max_ancestor_depth,
      gene_split_log_likelihood=None,
      gene_split_offset=None,
      gene_split_center_offset=None,
      *,
      leaf_species_idx,
      leaf_logp,
      family_idx,
      pibar_row_max,
      store_final_pibar=False,
      has_leaf_term=True,
      input_ws=None,
      use_receiver_weights=True,
      pi_residual_out=None,
  ):
      Pi_in_offset = _validate_offset_tensor(
          "Pi_in_offset",
          Pi_in_offset,
          rows=Pi_in.shape[0],
          device=Pi_in.device,
          residual_dtype=Pi_in.dtype,
      )
  ```

  New (insert the validate call right before `Pi_in_offset = _validate_offset_tensor(...)`; the
  rest of the function body is unchanged):
  ```python
      _validate_residual_tensors(
          Pi_in,
          Pi_out=Pi_out,
          Pibar=Pibar,
          max_transfer_mat=max_transfer_mat,
          duplication_loss_const=duplication_loss_const,
          Ebar=Ebar,
          E=E,
          speciation_child1_const=speciation_child1_const,
          speciation_child2_const=speciation_child2_const,
          receiver_log_probs=receiver_log_probs,
          leaf_logp=leaf_logp,
          pibar_row_max=pibar_row_max,
          gene_split_log_likelihood=gene_split_log_likelihood,
      )
      Pi_in_offset = _validate_offset_tensor(
          "Pi_in_offset",
          Pi_in_offset,
          rows=Pi_in.shape[0],
          device=Pi_in.device,
          residual_dtype=Pi_in.dtype,
      )
  ```

- [ ] **Step 7: `compute_dts_forward`** — insert a validate call, drop the cast on `log_split_probs`

  Current (lines 1227-1277):
  ```python
  def compute_dts_forward(
      Pi,
      Pi_offset,
      Pibar,
      Pibar_offset,
      split_left_rows,
      split_right_rows,
      species_child1,
      species_child2,
      W,
      reduce_idx,
      log_pD_vec,
      log_pS_vec,
      family_idx,
      *,
      log_split_probs=None,
      n_single_split_parents=None,
      single_split_parent_rows=None,
      multiple_split_group_ptr=None,
      multiple_split_parent_rows=None,
      max_splits_per_multiple_parent=None,
      active_parent_rows=None,
      family_offset=0,
  ):
      N = int(split_left_rows.shape[0])
      S = int(Pi.shape[1])
      Pi_offset = _validate_offset_tensor(
          "Pi_offset",
          Pi_offset,
          rows=Pi.shape[0],
          device=Pi.device,
          residual_dtype=Pi.dtype,
      )
      accumulator_dtype = Pi_offset.dtype
      Pibar_offset = _validate_offset_tensor(
          "Pibar_offset",
          Pibar_offset,
          rows=Pibar.shape[0],
          device=Pi.device,
          dtype=accumulator_dtype,
      )
      gene_split_log_likelihood = torch.full((W, S), float("-inf"), device=Pi.device, dtype=Pi.dtype)
      gene_split_offset = torch.zeros((W,), device=Pi.device, dtype=accumulator_dtype)
      if N == 0:
          return gene_split_log_likelihood, gene_split_offset
      if log_split_probs is None:
          log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
      else:
          # Preprocessing owns split statics. Match the residual compute dtype at
          # the DTS boundary so they do not change the loop-carried arithmetic.
          log_split_probs = log_split_probs.reshape(N).to(Pi.dtype).contiguous()
  ```

  New:
  ```python
  def compute_dts_forward(
      Pi,
      Pi_offset,
      Pibar,
      Pibar_offset,
      split_left_rows,
      split_right_rows,
      species_child1,
      species_child2,
      W,
      reduce_idx,
      log_pD_vec,
      log_pS_vec,
      family_idx,
      *,
      log_split_probs=None,
      n_single_split_parents=None,
      single_split_parent_rows=None,
      multiple_split_group_ptr=None,
      multiple_split_parent_rows=None,
      max_splits_per_multiple_parent=None,
      active_parent_rows=None,
      family_offset=0,
  ):
      N = int(split_left_rows.shape[0])
      S = int(Pi.shape[1])
      _validate_residual_tensors(
          Pi,
          Pibar=Pibar,
          log_pD_vec=log_pD_vec,
          log_pS_vec=log_pS_vec,
          log_split_probs=log_split_probs,
      )
      Pi_offset = _validate_offset_tensor(
          "Pi_offset",
          Pi_offset,
          rows=Pi.shape[0],
          device=Pi.device,
          residual_dtype=Pi.dtype,
      )
      accumulator_dtype = Pi_offset.dtype
      Pibar_offset = _validate_offset_tensor(
          "Pibar_offset",
          Pibar_offset,
          rows=Pibar.shape[0],
          device=Pi.device,
          dtype=accumulator_dtype,
      )
      gene_split_log_likelihood = torch.full((W, S), float("-inf"), device=Pi.device, dtype=Pi.dtype)
      gene_split_offset = torch.zeros((W,), device=Pi.device, dtype=accumulator_dtype)
      if N == 0:
          return gene_split_log_likelihood, gene_split_offset
      if log_split_probs is None:
          log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
      else:
          log_split_probs = log_split_probs.reshape(N).contiguous()
  ```

  (the rest of the function — the `n_single_split_parents is None` fallback block onward — is
  unchanged)

- [ ] **Step 8: Sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -c "from gpurec.core.kernels.pi_forward import _select_log_split_probs, _validate_residual_tensors"
  git add gpurec/core/kernels/pi_forward.py
  git commit -m "Reconcile precision-fix WIP: pi_forward.py (defines shared helpers)"
  ```

---

### Task 2: `gpurec/core/kernels/dts_so.py`

**Files:**
- Modify: `gpurec/core/kernels/dts_so.py`

**Interfaces:**
- Consumes: `_select_log_split_probs`, `_validate_residual_tensors` from Task 1.

- [ ] **Step 1: Imports**

  Before (lines 9-13):
  ```python
  from gpurec.core.kernels.pi_forward import (
      _load_event_log_probability,
      _tl_float_dtype,
      _validate_offset_tensor,
  )
  ```

  After:
  ```python
  from gpurec.core.kernels.pi_forward import (
      _load_event_log_probability,
      _select_log_split_probs,
      _tl_float_dtype,
      _validate_offset_tensor,
      _validate_residual_tensors,
  )
  ```

- [ ] **Step 2: `_gene_split_event_vjp_directional_derivative_kernel`** (was `_dts_split_so_kernel`)
      — 2 cast removals

  Current (lines 234-235):
  ```python
      left_pibar_row_max = tl.load(pibar_row_max_ptr + left_clade_row).to(DTYPE)
      right_pibar_row_max = tl.load(pibar_row_max_ptr + right_clade_row).to(DTYPE)
  ```

  New:
  ```python
      left_pibar_row_max = tl.load(pibar_row_max_ptr + left_clade_row)
      right_pibar_row_max = tl.load(pibar_row_max_ptr + right_clade_row)
  ```

- [ ] **Step 3: `_transfer_subtree_vjp_directional_derivative_kernel`** (was `_dts_tree_so_kernel`)
      — 1 cast removal

  Current (line 306):
  ```python
      receiver_mass_log_scale = tl.load(pibar_row_max_ptr + child).to(DTYPE)
  ```

  New:
  ```python
      receiver_mass_log_scale = tl.load(pibar_row_max_ptr + child)
  ```

- [ ] **Step 4: `dts_backward_so`** — insert validate call, drop cast, remove the now-redundant
      `meta.get` reassignment

  Current (lines 448-492):
  ```python
  def dts_backward_so(
      Pi, dPi, Pibar, dPibar, v, ws, meta, S,
      log_pD_param, log_pS_param, dlog_pD_param, dlog_pS_param,
      max_transfer, d_max_transfer,
      receiver_log_probs, species_child1, species_child2, pibar_row_max, family_idx,
      d_rhs, d_grad_pD, d_grad_pS, d_grad_max_transfer,
      d_grad_receiver_log_probs,
      *, compact_level_ptr=None, compact_level_parents=None,
      compact_level_child1=None, compact_level_child2=None,
      use_receiver_weights=False, dreceiver_log_probs=None, pi_offset, pibar_offset,
  ):
      """Accumulate the DTS second-order contraction documented in LaTeX."""
      split_left_rows = meta["sl"]
      split_right_rows = meta["sr"]
      n_splits = int(split_left_rows.numel())
      device, dtype = Pi.device, Pi.dtype
      expected_rows = int(Pi.shape[0])
      pi_offset = _validate_offset_tensor(
          "pi_offset",
          pi_offset,
          rows=expected_rows,
          device=device,
          residual_dtype=Pi.dtype,
      )
      pibar_offset = _validate_offset_tensor(
          "pibar_offset",
          pibar_offset,
          rows=expected_rows,
          device=device,
          dtype=pi_offset.dtype,
      )
      if n_splits == 0:
          return
      split_log_priors = meta.get("log_split_probs")
      if split_log_priors is None:
          split_log_priors = torch.zeros(
              (n_splits,), device=Pi.device, dtype=Pi.dtype
          )
      else:
          # Scheduling retains split priors at accumulator precision. DTS event
          # probabilities are residual-state quantities, so convert once at this
          # mathematical boundary; preprocessing cannot choose the model dtype.
          split_log_priors = (
              split_log_priors.reshape(n_splits).to(Pi.dtype).contiguous()
          )
  ```

  New:
  ```python
  def dts_backward_so(
      Pi, dPi, Pibar, dPibar, v, ws, meta, S,
      log_pD_param, log_pS_param, dlog_pD_param, dlog_pS_param,
      max_transfer, d_max_transfer,
      receiver_log_probs, species_child1, species_child2, pibar_row_max, family_idx,
      d_rhs, d_grad_pD, d_grad_pS, d_grad_max_transfer,
      d_grad_receiver_log_probs,
      *, compact_level_ptr=None, compact_level_parents=None,
      compact_level_child1=None, compact_level_child2=None,
      use_receiver_weights=False, dreceiver_log_probs=None, pi_offset, pibar_offset,
  ):
      """Accumulate the DTS second-order contraction documented in LaTeX."""
      split_left_rows = meta["sl"]
      split_right_rows = meta["sr"]
      n_splits = int(split_left_rows.numel())
      device, dtype = Pi.device, Pi.dtype
      split_log_priors = _select_log_split_probs(meta, Pi.dtype)
      _validate_residual_tensors(
          Pi,
          dPi=dPi,
          Pibar=Pibar,
          dPibar=dPibar,
          v=v,
          log_pD_param=log_pD_param,
          log_pS_param=log_pS_param,
          dlog_pD_param=dlog_pD_param,
          dlog_pS_param=dlog_pS_param,
          max_transfer=max_transfer,
          d_max_transfer=d_max_transfer,
          receiver_log_probs=receiver_log_probs,
          pibar_row_max=pibar_row_max,
          d_rhs=d_rhs,
          d_grad_pD=d_grad_pD,
          d_grad_pS=d_grad_pS,
          d_grad_max_transfer=d_grad_max_transfer,
          d_grad_receiver_log_probs=d_grad_receiver_log_probs,
          dreceiver_log_probs=dreceiver_log_probs,
          log_split_probs=split_log_priors,
      )
      expected_rows = int(Pi.shape[0])
      pi_offset = _validate_offset_tensor(
          "pi_offset",
          pi_offset,
          rows=expected_rows,
          device=device,
          residual_dtype=Pi.dtype,
      )
      pibar_offset = _validate_offset_tensor(
          "pibar_offset",
          pibar_offset,
          rows=expected_rows,
          device=device,
          dtype=pi_offset.dtype,
      )
      if n_splits == 0:
          return
      if split_log_priors is None:
          split_log_priors = torch.zeros(
              (n_splits,), device=Pi.device, dtype=Pi.dtype
          )
      else:
          split_log_priors = (
              split_log_priors.reshape(n_splits).contiguous()
          )
  ```

  (everything after this point in the function — `by_species = ...` onward — is unchanged)

- [ ] **Step 5: Sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -c "import gpurec.core.kernels.dts_so"
  git add gpurec/core/kernels/dts_so.py
  git commit -m "Reconcile precision-fix WIP: dts_so.py"
  ```

---

### Task 3: `gpurec/core/kernels/dts_tangent.py`

**Files:**
- Modify: `gpurec/core/kernels/dts_tangent.py`

**Interfaces:**
- Consumes: `_validate_residual_tensors` from Task 1.

- [ ] **Step 1: Imports**

  Before (lines 9-13):
  ```python
  from gpurec.core.kernels.pi_forward import (
      _load_event_log_probability,
      _tl_float_dtype,
      _validate_offset_tensor,
  )
  ```

  After:
  ```python
  from gpurec.core.kernels.pi_forward import (
      _load_event_log_probability,
      _tl_float_dtype,
      _validate_offset_tensor,
      _validate_residual_tensors,
  )
  ```

- [ ] **Step 2: `compute_dts_tangent`** — insert validate call, drop cast

  Current (lines 182-225):
  ```python
  def compute_dts_tangent(
      Pi, Pibar, dPi, dPibar, split_left_rows, split_right_rows,
      species_child1, species_child2, W, reduce_idx,
      log_pD_vec, log_pS_vec, dlog_pD_vec, dlog_pS_vec,
      gene_split_log_likelihood, family_idx,
      *, log_split_probs=None, family_offset=0,
      pi_offset, pibar_offset, gene_split_offset,
  ):
      """Return the DTS JVP; see ``docs/latex/kernel_mathematics.tex``."""
      N = int(split_left_rows.shape[0])
      S = int(Pi.shape[1])
      d_gene_split_log_likelihood = torch.zeros(
          (W, S), device=Pi.device, dtype=Pi.dtype
      )
      C = int(Pi.shape[0])
      pi_offset = _validate_offset_tensor(
          "pi_offset",
          pi_offset,
          rows=C,
          device=Pi.device,
          residual_dtype=Pi.dtype,
      )
      accumulator_dtype = pi_offset.dtype
      pibar_offset = _validate_offset_tensor(
          "pibar_offset",
          pibar_offset,
          rows=C,
          device=Pi.device,
          dtype=accumulator_dtype,
      )
      gene_split_offset = _validate_offset_tensor(
          "gene_split_offset",
          gene_split_offset,
          rows=W,
          device=Pi.device,
          dtype=accumulator_dtype,
      )
      if N == 0:
          return d_gene_split_log_likelihood
      if log_split_probs is None:
          log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
      else:
          # Batch static -> compute dtype at the canonical forward boundary.
          log_split_probs = log_split_probs.reshape(N).to(Pi.dtype).contiguous()
  ```

  New:
  ```python
  def compute_dts_tangent(
      Pi, Pibar, dPi, dPibar, split_left_rows, split_right_rows,
      species_child1, species_child2, W, reduce_idx,
      log_pD_vec, log_pS_vec, dlog_pD_vec, dlog_pS_vec,
      gene_split_log_likelihood, family_idx,
      *, log_split_probs=None, family_offset=0,
      pi_offset, pibar_offset, gene_split_offset,
  ):
      """Return the DTS JVP; see ``docs/latex/kernel_mathematics.tex``."""
      N = int(split_left_rows.shape[0])
      S = int(Pi.shape[1])
      _validate_residual_tensors(
          Pi,
          Pibar=Pibar,
          dPi=dPi,
          dPibar=dPibar,
          log_pD_vec=log_pD_vec,
          log_pS_vec=log_pS_vec,
          dlog_pD_vec=dlog_pD_vec,
          dlog_pS_vec=dlog_pS_vec,
          gene_split_log_likelihood=gene_split_log_likelihood,
          log_split_probs=log_split_probs,
      )
      d_gene_split_log_likelihood = torch.zeros(
          (W, S), device=Pi.device, dtype=Pi.dtype
      )
      C = int(Pi.shape[0])
      pi_offset = _validate_offset_tensor(
          "pi_offset",
          pi_offset,
          rows=C,
          device=Pi.device,
          residual_dtype=Pi.dtype,
      )
      accumulator_dtype = pi_offset.dtype
      pibar_offset = _validate_offset_tensor(
          "pibar_offset",
          pibar_offset,
          rows=C,
          device=Pi.device,
          dtype=accumulator_dtype,
      )
      gene_split_offset = _validate_offset_tensor(
          "gene_split_offset",
          gene_split_offset,
          rows=W,
          device=Pi.device,
          dtype=accumulator_dtype,
      )
      if N == 0:
          return d_gene_split_log_likelihood
      if log_split_probs is None:
          log_split_probs = torch.zeros((N,), device=Pi.device, dtype=Pi.dtype)
      else:
          log_split_probs = log_split_probs.reshape(N).contiguous()
  ```

  (the rest of the function is unchanged)

- [ ] **Step 3: Sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -c "import gpurec.core.kernels.dts_tangent"
  git add gpurec/core/kernels/dts_tangent.py
  git commit -m "Reconcile precision-fix WIP: dts_tangent.py"
  ```

---

### Task 4: `gpurec/api/_implicit_grad.py`

**Files:**
- Modify: `gpurec/api/_implicit_grad.py`

**Interfaces:**
- Consumes: `_select_log_split_probs` from Task 1.

- [ ] **Step 1: Import**

  Before (line 8):
  ```python
  from gpurec.core.kernels.pi_forward import compute_dts_forward
  ```

  After:
  ```python
  from gpurec.core.kernels.pi_forward import _select_log_split_probs, compute_dts_forward
  ```

- [ ] **Step 2: First call site** — `compute_dts_forward` (line ~581)

  Current:
  ```python
                  log_split_probs=meta.get("log_split_probs"),
                  n_single_split_parents=meta.get("n_eq1"),
  ```

  New:
  ```python
                  log_split_probs=_select_log_split_probs(meta, Pi_star_wave.dtype),
                  n_single_split_parents=meta.get("n_eq1"),
  ```

  (only the `log_split_probs=` line changes; everything else in this `compute_dts_forward(...)`
  call is unchanged)

- [ ] **Step 3: Second call site** — `accumulate_gene_split_event_vjp` (lines ~722-731)

  Current:
  ```python
              (
                  donor_adjoint,
                  total_donor_adjoint,
                  active_donor_side,
                  _duplication_parameter_vjp,
                  _speciation_parameter_vjp,
              ) = accumulate_gene_split_event_vjp(
                  Pi_star_wave,
                  Pibar_star_wave,
                  v_k,
                  ws,
                  split_left_rows,
                  split_right_rows,
                  meta["reduce_idx"],
                  meta.get(
                      "log_split_probs",
                      split_left_rows.new_zeros(
                          (int(split_left_rows.numel()),),
                          dtype=Pi_star_wave.dtype,
                      ),
                  ),
                  log_pD_param,
                  log_pS_param,
  ```

  New:
  ```python
              (
                  donor_adjoint,
                  total_donor_adjoint,
                  active_donor_side,
                  _duplication_parameter_vjp,
                  _speciation_parameter_vjp,
              ) = accumulate_gene_split_event_vjp(
                  Pi_star_wave,
                  Pibar_star_wave,
                  v_k,
                  ws,
                  split_left_rows,
                  split_right_rows,
                  meta["reduce_idx"],
                  _select_log_split_probs(meta, Pi_star_wave.dtype),
                  log_pD_param,
                  log_pS_param,
  ```

- [ ] **Step 4: Sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -c "import gpurec.api._implicit_grad"
  git add gpurec/api/_implicit_grad.py
  git commit -m "Reconcile precision-fix WIP: _implicit_grad.py"
  ```

---

### Task 5: `gpurec/core/inference/forward.py`

**Files:**
- Modify: `gpurec/core/inference/forward.py`

**Interfaces:**
- Consumes: `_select_log_split_probs` from Task 1.

- [ ] **Step 1: Import**

  Before (lines 4-8):
  ```python
  from ..kernels.pi_forward import (
      compute_dts_forward,
      compute_leaf_initial_wave_step,
      compute_wave_step,
  )
  ```

  After:
  ```python
  from ..kernels.pi_forward import (
      compute_dts_forward,
      compute_leaf_initial_wave_step,
      compute_wave_step,
      _select_log_split_probs,
  )
  ```

- [ ] **Step 2: `compute_dts_forward` call site** (line ~95)

  Current:
  ```python
                  log_split_probs=meta.get("log_split_probs"),
                  n_single_split_parents=meta.get("n_eq1"),
  ```

  New:
  ```python
                  log_split_probs=_select_log_split_probs(meta, pi.dtype),
                  n_single_split_parents=meta.get("n_eq1"),
  ```

- [ ] **Step 3: Sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -c "import gpurec.core.inference.forward"
  git add gpurec/core/inference/forward.py
  git commit -m "Reconcile precision-fix WIP: inference/forward.py"
  ```

---

### Task 6: `gpurec/solver/forward_tangent.py`

**Files:**
- Modify: `gpurec/solver/forward_tangent.py`

**Interfaces:**
- Consumes: `_select_log_split_probs` from Task 1.

- [ ] **Step 1: Import**

  Before (line 28):
  ```python
  from gpurec.core.kernels.pi_forward import compute_dts_forward
  ```

  After:
  ```python
  from gpurec.core.kernels.pi_forward import (
      _select_log_split_probs,
      compute_dts_forward,
  )
  ```

- [ ] **Step 2: Both call sites** (lines ~299-321)

  Current:
  ```python
              gene_split_log_likelihood, gene_split_offset = compute_dts_forward(
                  pi, pi_offset, pibar, pibar_offset,
                  meta["sl"], meta["sr"], species_child1, species_child2,
                  W, meta["reduce_idx"],
                  base["duplication_log_probability_param"],
                  base["speciation_log_probability_param"], family_idx=family_idx,
                  log_split_probs=meta.get("log_split_probs"), n_single_split_parents=meta.get("n_eq1"),
                  single_split_parent_rows=meta.get("eq1_reduce_idx"), multiple_split_group_ptr=meta.get("ge2_ptr"),
                  multiple_split_parent_rows=meta.get("ge2_parent_ids"),
                  max_splits_per_multiple_parent=meta.get("ge2_max_fanout"), family_offset=ws,
              )
              d_gene_split_log_likelihood = compute_dts_tangent(
                  pi, pibar, dpi, dpibar, meta["sl"], meta["sr"],
                  species_child1, species_child2, W, meta["reduce_idx"],
                  base["duplication_log_probability_param"],
                  base["speciation_log_probability_param"],
                  tangent_constants["d_duplication_log_probability_param"],
                  tangent_constants["d_speciation_log_probability_param"],
                  gene_split_log_likelihood, family_idx,
                  log_split_probs=meta.get("log_split_probs"), family_offset=ws,
                  pi_offset=pi_offset, pibar_offset=pibar_offset,
                  gene_split_offset=gene_split_offset,
              )
  ```

  New:
  ```python
              gene_split_log_likelihood, gene_split_offset = compute_dts_forward(
                  pi, pi_offset, pibar, pibar_offset,
                  meta["sl"], meta["sr"], species_child1, species_child2,
                  W, meta["reduce_idx"],
                  base["duplication_log_probability_param"],
                  base["speciation_log_probability_param"], family_idx=family_idx,
                  log_split_probs=_select_log_split_probs(meta, pi.dtype), n_single_split_parents=meta.get("n_eq1"),
                  single_split_parent_rows=meta.get("eq1_reduce_idx"), multiple_split_group_ptr=meta.get("ge2_ptr"),
                  multiple_split_parent_rows=meta.get("ge2_parent_ids"),
                  max_splits_per_multiple_parent=meta.get("ge2_max_fanout"), family_offset=ws,
              )
              d_gene_split_log_likelihood = compute_dts_tangent(
                  pi, pibar, dpi, dpibar, meta["sl"], meta["sr"],
                  species_child1, species_child2, W, meta["reduce_idx"],
                  base["duplication_log_probability_param"],
                  base["speciation_log_probability_param"],
                  tangent_constants["d_duplication_log_probability_param"],
                  tangent_constants["d_speciation_log_probability_param"],
                  gene_split_log_likelihood, family_idx,
                  log_split_probs=_select_log_split_probs(meta, pi.dtype), family_offset=ws,
                  pi_offset=pi_offset, pibar_offset=pibar_offset,
                  gene_split_offset=gene_split_offset,
              )
  ```

- [ ] **Step 3: Sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -c "import gpurec.solver.forward_tangent"
  git add gpurec/solver/forward_tangent.py
  git commit -m "Reconcile precision-fix WIP: solver/forward_tangent.py"
  ```

---

### Task 7: `gpurec/solver/hvp_exact.py`

**Files:**
- Modify: `gpurec/solver/hvp_exact.py`

**Interfaces:**
- Consumes: `_select_log_split_probs` from Task 1.

- [ ] **Step 1: Import** (insert after the `receiver_weights_are_uniform` import, before
      `from gpurec.core.kernels.wave_backward import (`)

  Before:
  ```python
  from gpurec.core.inference.solver import receiver_weights_are_uniform
  from gpurec.core.kernels.dts_so import dts_backward_so
  from gpurec.core.kernels.e_step import e_step_triton_autograd
  from gpurec.core.kernels.e_step_so import e_step_backward_so
  from gpurec.core.kernels.wave_backward import (
  ```

  After:
  ```python
  from gpurec.core.inference.solver import receiver_weights_are_uniform
  from gpurec.core.kernels.dts_so import dts_backward_so
  from gpurec.core.kernels.e_step import e_step_triton_autograd
  from gpurec.core.kernels.e_step_so import e_step_backward_so
  from gpurec.core.kernels.pi_forward import _select_log_split_probs
  from gpurec.core.kernels.wave_backward import (
  ```

- [ ] **Step 2: `compute_dts_tangent` call site** (line ~549)

  Current:
  ```python
                          log_split_probs=meta.get("log_split_probs"), family_offset=ws,
  ```

  New:
  ```python
                          log_split_probs=_select_log_split_probs(meta, sv["pi_wave"].dtype), family_offset=ws,
  ```

- [ ] **Step 3: `accumulate_gene_split_event_vjp` call site** (line ~684)

  Current:
  ```python
                      ) = accumulate_gene_split_event_vjp(
                          sv["pi_wave"], sv["pibar_wave"], dv, ws, meta["sl"], meta["sr"],
                          meta["reduce_idx"],
                          meta.get("log_split_probs", meta["sl"].new_zeros((int(meta["sl"].numel()),), dtype=dtype)),
                          wave_constants["duplication_log_probability_param"],
                          wave_constants["speciation_log_probability_param"],
  ```

  New:
  ```python
                      ) = accumulate_gene_split_event_vjp(
                          sv["pi_wave"], sv["pibar_wave"], dv, ws, meta["sl"], meta["sr"],
                          meta["reduce_idx"],
                          _select_log_split_probs(meta, dtype),
                          wave_constants["duplication_log_probability_param"],
                          wave_constants["speciation_log_probability_param"],
  ```

- [ ] **Step 4: Sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -c "import gpurec.solver.hvp_exact"
  git add gpurec/solver/hvp_exact.py
  git commit -m "Reconcile precision-fix WIP: solver/hvp_exact.py"
  ```

---

### Task 8: `gpurec/core/kernels/wave_so.py` — silent-trap file, extra care

**Files:**
- Modify: `gpurec/core/kernels/wave_so.py`

**Interfaces:**
- Consumes: `_validate_residual_tensors` from Task 1.

**Trap note:** `wave_backward_so`'s validate call must NOT reference `mc`/`max_transfer` — dev's
current `wave_backward_so` signature has no such parameter at all (verify:
`grep -n "max_transfer" gpurec/core/kernels/wave_so.py` returns nothing before your edit). This is
a genuine functional difference from the WIP's fork point (not a naming collision) — omit it
entirely from the validate call below; do not invent a substitute.

- [ ] **Step 1: Verify the trap precondition**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  grep -n "max_transfer" gpurec/core/kernels/wave_so.py
  ```
  Expected: no output (confirms `wave_backward_so` genuinely has no `max_transfer`-family
  parameter — the validate call below correctly omits it).

- [ ] **Step 2: Import**

  Before (lines 9-13):
  ```python
  from gpurec.core.kernels.pi_forward import (
      _prepare_wave_launch,
      _tl_float_dtype,
      _validate_offset_tensor,
  )
  ```

  After:
  ```python
  from gpurec.core.kernels.pi_forward import (
      _prepare_wave_launch,
      _tl_float_dtype,
      _validate_offset_tensor,
      _validate_residual_tensors,
  )
  ```

- [ ] **Step 3: `_reconciliation_vjp_directional_derivative_kernel`** (was `_wave_so_kernel`) — 1
      cast removal

  Current (line 68):
  ```python
      receiver_mass_log_scale = tl.load(pibar_row_max_ptr + ws + w).to(DTYPE)
  ```

  New:
  ```python
      receiver_mass_log_scale = tl.load(pibar_row_max_ptr + ws + w)
  ```

- [ ] **Step 4: `wave_backward_so`** — insert validate call (no `mc`/`max_transfer` kwarg — see
      trap note above)

  Current (lines 475-493):
  ```python
  def wave_backward_so(
      Pi_star, dPi, Pibar_star, dPibar, v, ws, W, S,
      pibar_row_max,
      duplication_loss_const, d_duplication_loss_const,
      Ebar, dEbar, E, dE,
      speciation_child1_const, d_speciation_child1_const,
      speciation_child2_const, d_speciation_child2_const,
      receiver_log_probs, species_child1, species_child2, species_parent, max_ancestor_depth,
      gene_split_log_likelihood=None, d_gene_split_log_likelihood=None,
      *, leaf_species_idx, leaf_logp, d_leaf_logp, family_idx, has_leaf_term=True,
      use_receiver_weights=False, d_rhs=None, dreceiver_log_probs=None,
      pi_offset, pibar_offset, gene_split_offset=None,
  ):
      """Return the wave second-order contraction documented in LaTeX."""
      has_splits = gene_split_log_likelihood is not None
      fold_rhs = d_rhs is not None
      _, const_row_stride = _prepare_wave_launch(S, duplication_loss_const)
      block_s = int(triton.next_power_of_2(S))
      device, dtype = Pi_star.device, Pi_star.dtype
      expected_rows = int(Pi_star.shape[0])
  ```

  New (insert the validate call between `device, dtype = ...` and `expected_rows = ...`; the rest
  of the function is unchanged):
  ```python
      device, dtype = Pi_star.device, Pi_star.dtype
      _validate_residual_tensors(
          Pi_star,
          dPi=dPi,
          Pibar_star=Pibar_star,
          dPibar=dPibar,
          v=v,
          pibar_row_max=pibar_row_max,
          duplication_loss_const=duplication_loss_const,
          d_duplication_loss_const=d_duplication_loss_const,
          Ebar=Ebar,
          dEbar=dEbar,
          E=E,
          dE=dE,
          speciation_child1_const=speciation_child1_const,
          d_speciation_child1_const=d_speciation_child1_const,
          speciation_child2_const=speciation_child2_const,
          d_speciation_child2_const=d_speciation_child2_const,
          receiver_log_probs=receiver_log_probs,
          dreceiver_log_probs=dreceiver_log_probs,
          leaf_logp=leaf_logp,
          d_leaf_logp=d_leaf_logp,
          gene_split_log_likelihood=gene_split_log_likelihood,
          d_gene_split_log_likelihood=d_gene_split_log_likelihood,
          d_rhs=d_rhs,
      )
      expected_rows = int(Pi_star.shape[0])
  ```

- [ ] **Step 5: Post-edit trap check + sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  grep -n "mc=\|mc,\| mc \b" gpurec/core/kernels/wave_so.py || echo "clean: no stale mc reference"
  python -c "import gpurec.core.kernels.wave_so"
  git add gpurec/core/kernels/wave_so.py
  git commit -m "Reconcile precision-fix WIP: wave_so.py (trap avoided: no max_transfer kwarg)"
  ```

---

### Task 9: `gpurec/core/kernels/wave_tangent.py`

**Files:**
- Modify: `gpurec/core/kernels/wave_tangent.py`

**Interfaces:**
- Consumes: `_validate_residual_tensors` from Task 1.
- Produces: `_validate_wave_tangent_inputs(...)` — new helper, local to this file, no other callers
  to keep compatible.

**Note:** dev independently fixed a pre-existing bug in both call sites below (the
`dreceiver_log_probs` default used to alias the *primal* `receiver_log_probs` as its own tangent;
dev's current code correctly defaults to `torch.zeros_like(receiver_log_probs)`). That fix is
unrelated to this precision work — the code blocks below already include dev's fix; keep it exactly
as shown, do not revert it.

- [ ] **Step 1: Import**

  Before (lines 11-15):
  ```python
  from gpurec.core.kernels.pi_forward import (
      _prepare_wave_launch,
      _tl_float_dtype,
      _validate_offset_tensor,
  )
  ```

  After:
  ```python
  from gpurec.core.kernels.pi_forward import (
      _prepare_wave_launch,
      _tl_float_dtype,
      _validate_offset_tensor,
      _validate_residual_tensors,
  )
  ```

- [ ] **Step 2: New helper** — insert right after `_prepare_wave_offsets`, before
      `compute_wave_step_tangent_selfloop`

  Current (lines 485-509, showing the insertion point):
  ```python
  def _prepare_wave_offsets(Pi_in, pi_offset, gene_split_offset, has_splits, W):
      """Validate the reconciliation and gene-split gauges."""
      pi_offset = _validate_offset_tensor(
          "pi_offset",
          pi_offset,
          rows=Pi_in.shape[0],
          device=Pi_in.device,
          residual_dtype=Pi_in.dtype,
      )
      if has_splits:
          if gene_split_offset is None:
              raise ValueError("gene_split_offset is required for a split wave")
          gene_split_offset = _validate_offset_tensor(
              "gene_split_offset",
              gene_split_offset,
              rows=W,
              device=Pi_in.device,
              dtype=pi_offset.dtype,
          )
      elif gene_split_offset is not None:
          raise ValueError("gene_split_offset is only valid for a split wave")
      return pi_offset, (gene_split_offset if has_splits else pi_offset)


  def compute_wave_step_tangent_selfloop(
  ```

  New (insert the helper between them; `_prepare_wave_offsets`'s body unchanged):
  ```python
  def _prepare_wave_offsets(Pi_in, pi_offset, gene_split_offset, has_splits, W):
      """Validate the reconciliation and gene-split gauges."""
      pi_offset = _validate_offset_tensor(
          "pi_offset",
          pi_offset,
          rows=Pi_in.shape[0],
          device=Pi_in.device,
          residual_dtype=Pi_in.dtype,
      )
      if has_splits:
          if gene_split_offset is None:
              raise ValueError("gene_split_offset is required for a split wave")
          gene_split_offset = _validate_offset_tensor(
              "gene_split_offset",
              gene_split_offset,
              rows=W,
              device=Pi_in.device,
              dtype=pi_offset.dtype,
          )
      elif gene_split_offset is not None:
          raise ValueError("gene_split_offset is only valid for a split wave")
      return pi_offset, (gene_split_offset if has_splits else pi_offset)


  def _validate_wave_tangent_inputs(
      Pi_in,
      *,
      dPi,
      max_transfer_mat,
      dmax_transfer,
      duplication_loss_const,
      d_duplication_loss_const,
      Ebar,
      dEbar,
      E,
      dE,
      speciation_child1_const,
      d_speciation_child1_const,
      speciation_child2_const,
      d_speciation_child2_const,
      receiver_log_probs,
      leaf_logp,
      d_leaf_logp,
      gene_split_log_likelihood,
      d_gene_split_log_likelihood,
      dreceiver_log_probs,
      dPibar_out,
  ):
      _validate_residual_tensors(
          Pi_in,
          dPi=dPi,
          max_transfer_mat=max_transfer_mat,
          dmax_transfer=dmax_transfer,
          duplication_loss_const=duplication_loss_const,
          d_duplication_loss_const=d_duplication_loss_const,
          Ebar=Ebar,
          dEbar=dEbar,
          E=E,
          dE=dE,
          speciation_child1_const=speciation_child1_const,
          d_speciation_child1_const=d_speciation_child1_const,
          speciation_child2_const=speciation_child2_const,
          d_speciation_child2_const=d_speciation_child2_const,
          receiver_log_probs=receiver_log_probs,
          leaf_logp=leaf_logp,
          d_leaf_logp=d_leaf_logp,
          gene_split_log_likelihood=gene_split_log_likelihood,
          d_gene_split_log_likelihood=d_gene_split_log_likelihood,
          dreceiver_log_probs=dreceiver_log_probs,
          dPibar_out=dPibar_out,
      )


  def compute_wave_step_tangent_selfloop(
  ```

- [ ] **Step 3: `compute_wave_step_tangent_selfloop`** — insert a call, right after the
      `dreceiver_log_probs` reassignment

  Current (lines 509-540):
  ```python
  def compute_wave_step_tangent_selfloop(
      Pi_in, dPi_io, ws, W, S, n_iters,
      max_transfer_mat, dmax_transfer,
      duplication_loss_const, d_duplication_loss_const,
      Ebar, dEbar, E, dE,
      speciation_child1_const, d_speciation_child1_const,
      speciation_child2_const, d_speciation_child2_const,
      receiver_log_probs, species_child1, species_child2, species_parent, max_ancestor_depth,
      gene_split_log_likelihood=None, d_gene_split_log_likelihood=None,
      *, leaf_species_idx, leaf_logp, d_leaf_logp, family_idx,
      dPibar_out=None, has_leaf_term=True, use_receiver_weights=True,
      dreceiver_log_probs=None,
      pi_offset, gene_split_offset=None,
  ):
      """Run fixed-count wave-tangent iterations; see the LaTeX reference."""
      has_splits = gene_split_log_likelihood is not None
      pi_offset_arg, gene_split_offset_arg = _prepare_wave_offsets(
          Pi_in, pi_offset, gene_split_offset, has_splits, W
      )
      _, const_row_stride = _prepare_wave_launch(S, duplication_loss_const)
      block_s = int(triton.next_power_of_2(S))
      store_pibar = dPibar_out is not None
      dPi_out_rows = dPi_io.narrow(0, int(ws), int(W))
      dummy = Pi_in
      dreceiver_log_probs = (
          torch.zeros_like(receiver_log_probs)
          if dreceiver_log_probs is None
          else dreceiver_log_probs.to(device=Pi_in.device, dtype=Pi_in.dtype)
          .reshape(S)
          .contiguous()
      )
      _apply_reconciliation_self_loop_jvp_iterations_kernel[(int(W),)](
  ```

  New (insert the call between the `dreceiver_log_probs = (...)` block and the kernel launch; the
  function signature and everything after the kernel launch are unchanged):
  ```python
      dreceiver_log_probs = (
          torch.zeros_like(receiver_log_probs)
          if dreceiver_log_probs is None
          else dreceiver_log_probs.to(device=Pi_in.device, dtype=Pi_in.dtype)
          .reshape(S)
          .contiguous()
      )
      _validate_wave_tangent_inputs(
          Pi_in,
          dPi=dPi_io,
          max_transfer_mat=max_transfer_mat,
          dmax_transfer=dmax_transfer,
          duplication_loss_const=duplication_loss_const,
          d_duplication_loss_const=d_duplication_loss_const,
          Ebar=Ebar,
          dEbar=dEbar,
          E=E,
          dE=dE,
          speciation_child1_const=speciation_child1_const,
          d_speciation_child1_const=d_speciation_child1_const,
          speciation_child2_const=speciation_child2_const,
          d_speciation_child2_const=d_speciation_child2_const,
          receiver_log_probs=receiver_log_probs,
          leaf_logp=leaf_logp,
          d_leaf_logp=d_leaf_logp,
          gene_split_log_likelihood=gene_split_log_likelihood,
          d_gene_split_log_likelihood=d_gene_split_log_likelihood,
          dreceiver_log_probs=dreceiver_log_probs,
          dPibar_out=dPibar_out,
      )
      _apply_reconciliation_self_loop_jvp_iterations_kernel[(int(W),)](
  ```

- [ ] **Step 4: `compute_wave_step_tangent`** — insert 2 calls, right after its own
      `dreceiver_log_probs` reassignment

  Current (lines 573-604):
  ```python
  def compute_wave_step_tangent(
      Pi_in, dPi_in, dPi_out, ws, W, S,
      max_transfer_mat, dmax_transfer,
      duplication_loss_const, d_duplication_loss_const,
      Ebar, dEbar, E, dE,
      speciation_child1_const, d_speciation_child1_const,
      speciation_child2_const, d_speciation_child2_const,
      receiver_log_probs, species_child1, species_child2, species_parent, max_ancestor_depth,
      gene_split_log_likelihood=None, d_gene_split_log_likelihood=None,
      *, leaf_species_idx, leaf_logp, d_leaf_logp, family_idx,
      dPibar_out=None, has_leaf_term=True, input_ws=None,
      use_receiver_weights=True, dreceiver_log_probs=None,
      pi_offset, gene_split_offset=None,
  ):
      """Apply the wave-step JVP documented in the LaTeX reference."""
      has_splits = gene_split_log_likelihood is not None
      pi_offset_arg, gene_split_offset_arg = _prepare_wave_offsets(
          Pi_in, pi_offset, gene_split_offset, has_splits, W
      )
      _, const_row_stride = _prepare_wave_launch(S, duplication_loss_const)
      block_s = int(triton.next_power_of_2(S))
      store_pibar = dPibar_out is not None
      dPi_out_rows = dPi_out.narrow(0, int(ws), int(W))
      dummy = Pi_in  # unused placeholder for None pointers
      dreceiver_log_probs = (
          torch.zeros_like(receiver_log_probs)
          if dreceiver_log_probs is None
          else dreceiver_log_probs.to(device=Pi_in.device, dtype=Pi_in.dtype)
          .reshape(S)
          .contiguous()
      )
      _update_reconciliation_likelihood_jvp_kernel[(int(W),)](
  ```

  New (only the two new calls are inserted; nothing else in the function changes):
  ```python
      dreceiver_log_probs = (
          torch.zeros_like(receiver_log_probs)
          if dreceiver_log_probs is None
          else dreceiver_log_probs.to(device=Pi_in.device, dtype=Pi_in.dtype)
          .reshape(S)
          .contiguous()
      )
      _validate_residual_tensors(Pi_in, dPi_out=dPi_out)
      _validate_wave_tangent_inputs(
          Pi_in,
          dPi=dPi_in,
          max_transfer_mat=max_transfer_mat,
          dmax_transfer=dmax_transfer,
          duplication_loss_const=duplication_loss_const,
          d_duplication_loss_const=d_duplication_loss_const,
          Ebar=Ebar,
          dEbar=dEbar,
          E=E,
          dE=dE,
          speciation_child1_const=speciation_child1_const,
          d_speciation_child1_const=d_speciation_child1_const,
          speciation_child2_const=speciation_child2_const,
          d_speciation_child2_const=d_speciation_child2_const,
          receiver_log_probs=receiver_log_probs,
          leaf_logp=leaf_logp,
          d_leaf_logp=d_leaf_logp,
          gene_split_log_likelihood=gene_split_log_likelihood,
          d_gene_split_log_likelihood=d_gene_split_log_likelihood,
          dreceiver_log_probs=dreceiver_log_probs,
          dPibar_out=dPibar_out,
      )
      _update_reconciliation_likelihood_jvp_kernel[(int(W),)](
  ```

- [ ] **Step 5: Sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -c "import gpurec.core.kernels.wave_tangent"
  git add gpurec/core/kernels/wave_tangent.py
  git commit -m "Reconcile precision-fix WIP: wave_tangent.py"
  ```

---

### Task 10: `gpurec/core/kernels/wave_backward.py` — highest-risk file, silent-trap fix

**Files:**
- Modify: `gpurec/core/kernels/wave_backward.py`

**Interfaces:**
- Consumes: `_validate_residual_tensors` from Task 1.

**This is the highest-risk file in the whole plan.** A naive patch application would insert a
`_validate_residual_tensors(...)` call referencing stale names (`dts_r`, `mt_squeezed`, `DL_const`,
`SL1_const`, `SL2_const`) that do not exist anywhere in this file — `git apply --3way` reports this
hunk as clean (zero conflict markers) because the surrounding context matches; the bug is a
`NameError` at first real invocation only, caught by no merge tool and no import-time check. All
code blocks below already use the correct current names — transcribe them exactly, do not
substitute anything that "looks more familiar."

- [ ] **Step 1: Import**

  Before (line 12):
  ```python
  from gpurec.core.kernels.pi_forward import _validate_offset_tensor
  ```

  After:
  ```python
  from gpurec.core.kernels.pi_forward import (
      _validate_offset_tensor,
      _validate_residual_tensors,
  )
  ```

- [ ] **Step 2: `_solve_reconciliation_wave_vjp_2d`** (was `_wave_backward_uniform_2d`) — insert
      validate call **(the trap)**

  Current (lines 195-256, showing signature and insertion point):
  ```python
  def _solve_reconciliation_wave_vjp_2d(
      Pi_star, Pibar_star, ws, W, S,
      gene_split_log_likelihood,
      rhs,
      max_transfer, duplication_loss_const, Ebar, E, speciation_child1_const, speciation_child2_const,
      receiver_log_probs,
      species_child1, species_child2, leaf_term_wt,
      *,
      neumann_terms,
      leaf_species_idx,
      leaf_logp,
      has_leaf_term,
      active_mask,
      species_parent,
      max_ancestor_depth,
      pibar_row_max,
      family_idx,
      const_layout,
      leaf_logp_mode,
      use_leaf_index,
      compact_level_ptr,
      compact_level_parents,
      compact_level_child1,
      compact_level_child2,
      grad_receiver_log_probs=None,
      use_receiver_weights=True,
      self_loop_grad_targets=None,
      initial_v=None,
      return_last_increment=False,
      reserved_scratch_bytes=None,
      pi_offset,
      pibar_offset,
      gene_split_offset=None,
  ):
      """Retained 2D row-block/full-species tree-reduction self-loop."""
      if Pi_star.device.type != "cuda":
          raise RuntimeError("GPUREC self-loop 2D fast path requires CUDA tensors")
      if Pi_star.dtype not in (torch.float32, torch.float64):
          raise RuntimeError(
              "2D self-loop fast path currently supports fp32/fp64 only",
          )
      fits_memory_budget, required_bytes, budget_bytes = proposal0_memory_gate(
          W,
          S,
          Pi_star.dtype,
          device=Pi_star.device,
          reserved_scratch_bytes=reserved_scratch_bytes,
      )
      if not fits_memory_budget:
          raise RuntimeError(
              "2D self-loop fast path estimated scratch "
              f"{required_bytes / (1024 ** 3):.2f} GiB above memory budget "
              f"{(budget_bytes or 0) / (1024 ** 3):.2f} GiB",
          )
      if const_layout not in (0, 1, 2):
          raise RuntimeError("unsupported self-loop constant layout")
      if use_leaf_index and leaf_logp_mode not in (0, 1, 2, 3):
          raise RuntimeError("unsupported leaf log-probability layout")

      device = Pi_star.device
      dtype = Pi_star.dtype
      expected_rows = int(Pi_star.shape[0])
  ```

  New (insert the validate call between `dtype = Pi_star.dtype` and `expected_rows = ...`; the rest
  of the function is unchanged):
  ```python
      device = Pi_star.device
      dtype = Pi_star.dtype
      grad_targets = {}
      if self_loop_grad_targets is not None:
          target_names = (
              "grad_log_pD",
              "grad_log_pS",
              "grad_E",
              "grad_Ebar",
              "grad_E_s1",
              "grad_E_s2",
              "grad_max_transfer",
          )
          grad_targets = dict(zip(target_names, self_loop_grad_targets[:7]))
      _validate_residual_tensors(
          Pi_star,
          Pibar_star=Pibar_star,
          gene_split_log_likelihood=gene_split_log_likelihood,
          rhs=rhs,
          max_transfer=max_transfer,
          duplication_loss_const=duplication_loss_const,
          Ebar=Ebar,
          E=E,
          speciation_child1_const=speciation_child1_const,
          speciation_child2_const=speciation_child2_const,
          receiver_log_probs=receiver_log_probs,
          leaf_term_wt=leaf_term_wt,
          leaf_logp=leaf_logp,
          pibar_row_max=pibar_row_max,
          grad_receiver_log_probs=grad_receiver_log_probs,
          initial_v=initial_v,
          **grad_targets,
      )
      expected_rows = int(Pi_star.shape[0])
  ```

  `self_loop_grad_targets` is an 8-tuple at dev's tip — 7 gradient-target tensors followed by a
  `param_grad_vector` boolean flag; `self_loop_grad_targets[:7]` correctly takes only the 7 tensors,
  matching `target_names`'s length. Do **not** "fix" this to `[:8]` or add an 8th name — the boolean
  flag must never be passed into `_validate_residual_tensors`, which requires every value to be a
  tensor.

- [ ] **Step 3: Same function — 3 cast removals**

  Current (line 284):
  ```python
      receiver_log_probs = receiver_log_probs.to(device=device, dtype=dtype).contiguous()
  ```
  New:
  ```python
      receiver_log_probs = receiver_log_probs.contiguous()
  ```

  Current (line 326):
  ```python
      pibar_row_max = pibar_row_max.to(device=device, dtype=dtype).contiguous()
  ```
  New:
  ```python
      pibar_row_max = pibar_row_max.contiguous()
  ```

  Current (line 404):
  ```python
          v_k.copy_(initial_v.to(device=device, dtype=dtype).contiguous())
  ```
  New:
  ```python
          v_k.copy_(initial_v.contiguous())
  ```

- [ ] **Step 4: `solve_reconciliation_wave_vjp`** (was `wave_backward_uniform_fused`) — 1 cast
      removal

  Current (line 706):
  ```python
      pibar_row_max = pibar_row_max.to(device=Pi_star.device, dtype=Pi_star.dtype).contiguous()
  ```
  New:
  ```python
      pibar_row_max = pibar_row_max.contiguous()
  ```

- [ ] **Step 5: `accumulate_gene_split_event_vjp`** (was `dts_cross_backward_accum_fused`) — add
      `.contiguous()` on the squeeze, plus a validate call

  Current (line 807):
  ```python
      split_log_priors = log_split_probs.squeeze(-1) if log_split_probs.ndim > 1 else log_split_probs
  ```
  New:
  ```python
      split_log_priors = (log_split_probs.squeeze(-1) if log_split_probs.ndim > 1 else log_split_probs).contiguous()
  ```

  Current (lines 854-857, insertion point — right before this existing check):
  ```python
      else:
          max_transfer_layout = 0
      if output_donor_adjoint and pibar_row_max.numel() < Pi_star.shape[0]:
          raise ValueError("pibar_row_max must contain one row-max value per Pi row")
  ```

  New (insert the validate call between them):
  ```python
      else:
          max_transfer_layout = 0
      _validate_residual_tensors(
          Pi_star,
          Pibar_star=Pibar_star,
          v_k=v_k,
          log_split_probs=split_log_priors,
          log_pD=log_pD_arg,
          log_pS=log_pS_arg,
          accumulated_rhs=accumulated_rhs,
          grad_log_pD=grad_log_pD,
          grad_log_pS=grad_log_pS,
          grad_max_transfer=grad_max_transfer,
          max_transfer=max_transfer if output_donor_adjoint else None,
          pibar_row_max=pibar_row_max if output_donor_adjoint else None,
      )
      if output_donor_adjoint and pibar_row_max.numel() < Pi_star.shape[0]:
          raise ValueError("pibar_row_max must contain one row-max value per Pi row")
  ```

- [ ] **Step 6: `accumulate_transfer_complement_vjp_from_donor_adjoint`** (was
      `uniform_cross_pibar_vjp_tree_from_ud_fused`) — validate call + cast removal

  Current (lines 1082-1087):
  ```python
      compact_level_ptr = compact_level_ptr.contiguous()
      compact_level_parents = compact_level_parents.contiguous()
      compact_level_child1 = compact_level_child1.contiguous()
      compact_level_child2 = compact_level_child2.contiguous()
      receiver_log_probs = receiver_log_probs.to(device=Pi_star.device, dtype=Pi_star.dtype).contiguous()
      receiver_grad_arg = (
  ```

  New:
  ```python
      compact_level_ptr = compact_level_ptr.contiguous()
      compact_level_parents = compact_level_parents.contiguous()
      compact_level_child1 = compact_level_child1.contiguous()
      compact_level_child2 = compact_level_child2.contiguous()
      _validate_residual_tensors(
          Pi_star,
          receiver_log_probs=receiver_log_probs,
          donor_adjoint=donor_adjoint,
          total_donor_adjoint=total_donor_adjoint,
          pibar_row_max=pibar_row_max,
          accumulated_rhs=accumulated_rhs,
          grad_receiver_log_probs=grad_receiver_log_probs,
      )
      receiver_log_probs = receiver_log_probs.contiguous()
      receiver_grad_arg = (
  ```

- [ ] **Step 7: Trap-detector grep + sanity check + commit**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  grep -nE '\bdts_r\b|\bmt_squeezed\b|\bDL_const\b|\bSL1_const\b|\bSL2_const\b' gpurec/core/kernels/wave_backward.py
  ```
  Expected: no output. If anything prints, a stale identifier survived — fix it before continuing.
  ```bash
  python -c "import gpurec.core.kernels.wave_backward"
  git add gpurec/core/kernels/wave_backward.py
  git commit -m "Reconcile precision-fix WIP: wave_backward.py (fixes silent NameError trap)"
  ```

---

### Task 11: `gpurec/core/kernels/wave_backward_kernels.py` — mechanical cast-stripping

**Files:**
- Modify: `gpurec/core/kernels/wave_backward_kernels.py`

No new imports or helper calls — every one of the WIP's 102 changed lines in this file is a
`.to(DTYPE)` cast removal from a `tl.load(...)` call (confirmed exhaustively against the raw WIP
patch — no other content). The current (dev-tip) file has 113 occurrences of `.to(DTYPE)` in total;
**11 of those must NOT be touched** — they are arithmetic frame/offset-correction casts
(subtracting/negating already-loaded accumulator-dtype values into the kernel's working `DTYPE`),
not loads of a residual-precision static. `113 - 11 = 102`, matching the WIP's count exactly.

- [ ] **Step 1: The rule**

  Remove `.to(DTYPE)` from every expression of the form `tl.load(...).to(DTYPE)`, whether the
  `tl.load(...)` call is on one line or split across several (in which case `.to(DTYPE)` sits alone
  on the closing line, immediately after the load's closing `)`). Do **not** touch:
  - `.to(tl.int64)` / `.to(tl.int32)` (index/pointer casts — out of scope)
  - `.to(ACC_DTYPE)` (accumulator-dtype casts — none currently exist in this file; leave any you
    find)
  - The 11 arithmetic-expression casts listed in Step 2 (identifiable by the right-hand side being
    a parenthesized subtraction/addition/negation of variables, not a direct `tl.load(...)` call)

- [ ] **Step 2: The 11 lines to leave untouched** — verify each is present, unmodified, before and
      after your edit

  This exact 3-line group appears **twice** in the file, in two different kernels — both
  occurrences stay untouched:
  ```python
      pibar_offset_corr = (pibar_row_offset - pi_row_offset).to(DTYPE)
      leaf_offset_corr = (-pi_row_offset).to(DTYPE)
          gene_split_frame_shift = (gene_split_row_offset - pi_row_offset).to(DTYPE)
  ```

  This 5-line group appears **once**:
  ```python
      child_pair_frame_shift = (left_pi_offset + right_pi_offset - parent_pi_offset).to(DTYPE)
      left_transfer_frame_shift = (left_pi_offset + right_pibar_offset - parent_pi_offset).to(DTYPE)
      right_transfer_frame_shift = (right_pi_offset + left_pibar_offset - parent_pi_offset).to(DTYPE)
      left_exclusion_frame_shift = (left_pi_offset - left_pibar_offset).to(DTYPE)
      right_exclusion_frame_shift = (right_pi_offset - right_pibar_offset).to(DTYPE)
  ```

- [ ] **Step 3: Three worked examples** (representative of the ~50 remaining single- and
      multi-line cases — apply the same transform throughout the file)

  Single-line load, before:
  ```python
      row_max = tl.load(Pibar_row_max_ptr + row_global, mask=row_valid, other=NEG_LARGE).to(DTYPE)
  ```
  after:
  ```python
      row_max = tl.load(Pibar_row_max_ptr + row_global, mask=row_valid, other=NEG_LARGE)
  ```

  Multi-line load, before:
  ```python
      extinction_complement_log_probability = tl.load(
          Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
      ).to(DTYPE)
  ```
  after:
  ```python
      extinction_complement_log_probability = tl.load(
          Ebar_ptr + const_offsets, mask=const_mask, other=NEG_LARGE
      )
  ```

  Adjacent arithmetic cast that must survive unchanged (contrast case, do **not** edit):
  ```python
      pibar_offset_corr = (pibar_row_offset - pi_row_offset).to(DTYPE)
  ```

- [ ] **Step 4: Verification**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  grep -c '\.to(DTYPE)' gpurec/core/kernels/wave_backward_kernels.py
  ```
  Must print exactly `11`. Then:
  ```bash
  grep -n '\.to(DTYPE)' gpurec/core/kernels/wave_backward_kernels.py
  ```
  Every line printed must start with one of: `pibar_offset_corr = `, `leaf_offset_corr = `,
  `gene_split_frame_shift = `, `child_pair_frame_shift = `, `left_transfer_frame_shift = `,
  `right_transfer_frame_shift = `, `left_exclusion_frame_shift = `,
  `right_exclusion_frame_shift = `. If any other line still has `.to(DTYPE)`, or if any of these 11
  got accidentally stripped, fix it before proceeding.

- [ ] **Step 5: Sanity check + commit**

  ```bash
  python -c "import gpurec.core.kernels.wave_backward_kernels"
  git add gpurec/core/kernels/wave_backward_kernels.py
  git commit -m "Reconcile precision-fix WIP: wave_backward_kernels.py (102 cast removals)"
  ```

---

### Task 12: Full validation

**Files:** none modified — this task only runs verification.

- [ ] **Step 1: Trap-detector grep across the whole reconciled set** (not just the files with
      visible conflict markers — the same pattern that caused Task 10's silent trap could in
      principle recur anywhere)

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  grep -rnE '\bdts_r\b|\bmt_squeezed\b|\bDL_const\b|\bSL1_const\b|\bSL2_const\b|\bmc_item\b|\bcol_log_probs\b|\bitem_idx\b|\bsp_parent\b|\bitem_offset\b' \
    gpurec/core/kernels/pi_forward.py gpurec/core/kernels/dts_so.py gpurec/core/kernels/dts_tangent.py \
    gpurec/api/_implicit_grad.py gpurec/core/inference/forward.py gpurec/solver/forward_tangent.py \
    gpurec/solver/hvp_exact.py gpurec/core/kernels/wave_so.py gpurec/core/kernels/wave_tangent.py \
    gpurec/core/kernels/wave_backward.py gpurec/core/kernels/wave_backward_kernels.py
  ```
  Expected: no output. Any match is a stale identifier that must be fixed before proceeding.

- [ ] **Step 2: Run the direct regression tests**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -m pytest tests/test_centered_pi_forward.py tests/test_centered_pi_so.py \
    tests/test_centered_tangent.py tests/test_config_wiring.py tests/test_fidelity_alerax.py \
    tests/test_origination_weights.py tests/test_public_api_integration.py -v
  ```
  All tests must pass. These are the tests that merged cleanly in commit `bb7aed8a` and directly
  exercise dtype selection, HVP second-order kernels, and the closed-form origination gradient.

- [ ] **Step 3: Full test suite**

  ```bash
  cd /home/enzo/Documents/git/gpurec/gpurec/.worktrees/centered-kernels-port
  python -m pytest tests/ -m gpu -x -q
  ```
  Report actual pass/fail counts — do not assume prior numbers carry over.

- [ ] **Step 4: Report**

  Summarize: which tasks needed any deviation from this plan's exact code blocks (there should be
  none), full test suite pass/fail counts, and confirm the branch
  `land-centered-kernels-precision-fix` is ready for the next phase (folding into the larger
  centered-kernels-epoch main-port effort — out of scope for this plan).
