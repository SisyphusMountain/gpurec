# Uniform mode execution in gpurec

This document explains the execution path used when `pibar_mode="uniform"` is selected. In gpurec this is the default fast path for global, specieswise, and genewise non-pairwise transfer models. It is separate from the high-level API `mode` names `global`, `specieswise`, and `genewise`.

The short version is: uniform mode never materializes the dense `[S, S]` transfer matrix during the main forward or backward pass. It uses the fact that each donor species has a uniform distribution over recipient species that are not ancestral to the donor. The aggregate transfer term `Pibar` can therefore be computed from a row sum over `Pi` minus an ancestor correction.

## Source map

| Area | Main functions |
| --- | --- |
| High-level module API | `gpurec.api.model.GeneReconModel.forward`, `gpurec.api.autograd._GeneReconFunction.forward`, `gpurec.api.autograd._GeneReconFunction.backward` |
| Core inference API | `gpurec.core.model.GeneDataset.compute_likelihood_batch`, `GeneDataset._extract_batch_params`, `GeneDataset._species_helpers_for_mode` |
| Parameter extraction | `gpurec.core.extract_parameters.extract_parameters_uniform` |
| E fixed point | `gpurec.core.likelihood.E_fixed_point`, `gpurec.core.likelihood.E_step` |
| Pi forward | `gpurec.core.forward.Pi_wave_forward`, `_compute_Pibar_inline`, `_compute_dts_cross`, `gpurec.core.kernels.dts_fused.dts_fused`, `gpurec.core.kernels.wave_step.wave_step_fused` |
| Pi backward | `gpurec.core.backward.Pi_wave_backward`, `_self_loop_vjp_precompute`, `_gmres_self_loop_solve`, `_self_loop_Jt_apply`, `_dts_cross_differentiable` |
| Fused backward kernels | `gpurec.core.kernels.wave_backward.wave_backward_uniform_fused`, `dts_cross_backward_fused` |
| Full theta gradient | `gpurec.optimization.implicit_grad.implicit_grad_loglik_vjp_wave`, `_e_adjoint_and_theta_vjp` |
| Optimizer path | `gpurec.optimization.wave_optimizer.optimize_theta_wave`, `gpurec.optimization.genewise_optimizer.optimize_theta_genewise` |

## What uniform means

`uniform` refers to the computation of the transfer aggregate `Pibar`, not to whether all model rates are shared. The high-level API has three exposed parameter granularities:

| API mode | Flags | Theta shape |
| --- | --- | --- |
| `global` | `genewise=False`, `specieswise=False`, `pairwise=False` | `[3]` |
| `specieswise` | `genewise=False`, `specieswise=True`, `pairwise=False` | `[S, 3]` |
| `genewise` | `genewise=True`, `specieswise=False`, `pairwise=False` | `[G, 3]` |

The lower-level extraction function also supports `genewise=True` with `specieswise=True`, using theta shape `[G, S, 3]`. The notebook-friendly `GeneReconModel` mode map does not currently expose that as a named mode.

Pairwise transfer is not supported by `GeneReconModel`, and `pibar_mode="uniform"` is only valid for non-pairwise transfer matrices. If the recipient weights are truly non-uniform per donor-recipient pair, use `dense` or `topk`.

## Species preprocessing used by uniform mode

The C++ preprocessing builds two related `[S, S]` matrices in `compute_ancestors_and_recipients`:

| Matrix | Meaning |
| --- | --- |
| `ancestors_dense[descendant, ancestor]` | `1` if `ancestor` is on the path from `descendant` to the root, including the descendant itself |
| `Recipients_mat[donor, recipient]` | uniform over recipients where `recipient` is not ancestral to `donor`, zero otherwise |

`GeneDataset.__init__` stores `tr_mat_unnormalized = log2(Recipients_mat)` and `unnorm_row_max = tr_mat_unnormalized.max(dim=-1).values`. In uniform mode, the full `Recipients_mat` is skipped when moving helper tensors to the device. The code keeps `ancestors_dense`, builds `ancestors_T = ancestors_dense.T.to_sparse_coo()`, and carries `unnorm_row_max`.

This is the core memory choice: the dense recipient matrix is `O(S^2)`, while `ancestors_T` is sparse and the runtime transfer aggregate is `O(W*S + W*nnz_ancestors_per_row)` per wave.

## Uniform parameter extraction

All probabilities in the reconciliation dynamic program are represented in base-2 log space. `extract_parameters_uniform(theta, unnorm_row_max, specieswise, genewise=False)` computes:

```text
result = log2_softmax(concat(0, theta))
log_pS = result[..., 0]
log_pD = result[..., 1]
log_pL = result[..., 2]
log_pT = result[..., 3]
```

The function then returns `transfer_mat=None` and only keeps:

```text
max_transfer_mat = log_pT + unnorm_row_max
```

For genewise scalar rates this broadcasts as `[G, S]`. For specieswise rates this is already `[S]` or `[G, S]`. The dense linear-space transfer matrix is intentionally not produced.

`max_transfer_mat` is named after the dense path, where it stores the row maxima used to stabilize log-space matrix multiplication. In uniform mode it still has the same mathematical role: it contains the transfer-rate log probability plus the row normalization for the uniform recipient distribution.

## Forward pass call graph

The autograd API forward path is:

```text
GeneReconModel.forward
_GeneReconFunction.apply
_GeneReconFunction.forward
_extract_parameters
extract_parameters_uniform
E_fixed_point
E_step
Pi_wave_forward
_compute_dts_cross
dts_fused
seg_logsumexp
_compute_Pibar_inline
wave_step_fused
compute_log_likelihood
```

The core inference path is:

```text
GeneDataset.compute_likelihood_batch
GeneDataset._extract_batch_params
extract_parameters_uniform
GeneDataset._species_helpers_for_mode
GeneDataset._solve_e_fixed_point
E_fixed_point
E_step
compute_clade_waves
collate_wave
build_wave_layout
Pi_wave_forward
compute_log_likelihood
```

The direct optimizer path uses the same computational blocks:

```text
optimize_theta_wave
_extract
extract_parameters_uniform
E_fixed_point
Pi_wave_forward
compute_log_likelihood
implicit_grad_loglik_vjp_wave
```

## Forward pass details

### 1. Static state construction

`GeneReconModel` builds a `ReconStaticState` once. It collates all gene families, computes cross-family wave scheduling, moves species helpers to the target device, and stores `ancestors_T` for uniform mode. It also sets `transfer_mat_unnormalized=None` for uniform mode.

`GeneDataset.compute_likelihood_batch` performs the same work per call rather than caching it in a module object.

### 2. Parameter extraction

`_GeneReconFunction.forward` calls `_extract_parameters`, which dispatches to `extract_parameters_uniform` when `static.pibar_mode == "uniform"`. The returned `transfer_mat` is `None`; downstream functions rely on `max_transfer_mat` and `ancestors_T`.

### 3. E fixed point

`E_fixed_point` iterates `E_step` until convergence or `max_iters_E`. In uniform mode, `E_step` computes the transfer extinction aggregate `Ebar` without a matrix-vector product:

```text
max_E = max(E)
expE = exp2(E - max_E)
row_sum = sum_j expE[j]
ancestor_sum[s] = sum_{j ancestor of s, including s} expE[j]
Ebar[s] = safe_log2(row_sum - ancestor_sum[s]) + max_E + max_transfer_mat[s]
```

For batched genewise E, the same formula is applied row-wise with shape `[G, S]`.

`E_step` then evaluates four extinction alternatives:

| Term | Code expression |
| --- | --- |
| Speciation | `log_pS + E_s1 + E_s2` |
| Duplication | `log_pD + 2 * E` |
| Transfer | `E + Ebar` |
| Loss | `log_pL` |

The next `E` is `logsumexp2` over those four alternatives. `_safe_log2` is used because `row_sum - ancestor_sum` can become slightly negative in float32 from cancellation; the safe path returns `-inf` instead of `NaN`.

### 4. Pi wave initialization

`Pi_wave_forward` creates wave-ordered tensors:

```text
Pi:    [C, S], initialized to very negative values
Pibar: [C, S], initialized to -inf
```

Leaf clades are initialized with log probability `0.0` at their known species column. All other species entries remain unreachable until filled by the dynamic program.

The function constructs `sp_child1` and `sp_child2` vectors from the species helper parent-child arrays. A sentinel species index `S` represents "no child" for leaf species.

### 5. Uniform setup inside Pi forward

When `pibar_mode == "uniform"`, `Pi_wave_forward`:

```text
ancestors_T_mat = species_helpers["ancestors_dense"].T.to_sparse_coo()
transfer_mat_T = None
transfer_mat_c = None
use_global_pibar = True
```

The current forward implementation does not call `wave_step_uniform_fused`. It computes `Pibar` with `_compute_Pibar_inline` and then calls the generic `wave_step_fused` Triton kernel. The uniform-specific fused forward kernel exists in `gpurec.core.kernels.wave_step`, but it is not wired into `Pi_wave_forward` in the current code.

### 6. Cross-clade DTS for each wave

For each wave, if the wave has CCP splits, `Pi_wave_forward` calls `_compute_dts_cross`. That function calls the Triton kernel wrapper `dts_fused`, which computes five cross-clade terms for each split:

| Cross-clade term | Expression |
| --- | --- |
| Duplication | `log_pD + Pi_left + Pi_right` |
| Transfer left to right | `Pi_left + Pibar_right` |
| Transfer right to left | `Pi_right + Pibar_left` |
| Speciation aligned | `log_pS + Pi_left[child1] + Pi_right[child2]` |
| Speciation swapped | `log_pS + Pi_right[child1] + Pi_left[child2]` |

The split log probability is added, and splits belonging to the same parent clade are reduced with segmented `logsumexp`.

### 7. Uniform Pibar for each wave iteration

For every local fixed-point iteration of a wave, `_compute_Pibar_inline` evaluates:

```text
Pi_max[c] = max_j Pi[c, j]
p_prime[c, j] = exp2(Pi[c, j] - Pi_max[c])
row_sum[c] = sum_j p_prime[c, j]
ancestor_sum[c, s] = sum_{j ancestor of s, including s} p_prime[c, j]
Pibar[c, s] = safe_log2(row_sum[c] - ancestor_sum[c, s]) + Pi_max[c] + mt[c, s]
```

For non-batched global/specieswise cases, `mt[c, s]` is just `max_transfer_mat[s]`. For genewise batched cases, `mt[c, s]` is gathered by family id.

This exactly matches the preprocessed uniform recipient matrix because allowed recipients for donor `s` are all species that are not ancestors of `s`. Since the recipient weights are constant across those allowed recipients, the dense weighted sum collapses to "sum all `Pi` mass minus ancestor mass, then apply the row normalization".

### 8. Wave self-loop update

After `Pibar_W` is computed, `Pi_wave_forward` calls `wave_step_fused`, which launches `_wave_step_kernel`. For each clade/species entry, the kernel evaluates six local terms:

| Local term | Expression |
| --- | --- |
| Duplication/loss continuation | `DL_const + Pi` where `DL_const = 1.0 + log_pD + E` |
| Transfer extinction on recipient side | `Pi + Ebar` |
| Transfer aggregate from donor side | `Pibar + E` |
| Speciation child 1 | `SL1_const + Pi[species_child1]` |
| Speciation child 2 | `SL2_const + Pi[species_child2]` |
| Leaf compatibility | `leaf_wt` |

If cross-clade DTS exists for the wave, it is included as an additional term. The kernel returns `Pi_new = logsumexp2(all_terms)`.

The wave iterates until the significant entries satisfy the local tolerance after a small warmup, or until `local_iters` is reached. Converged `Pi` and `Pibar` are stored in wave order for the backward pass.

### 9. Likelihood or loss

`compute_log_likelihood(Pi, E, root_clade_ids)` computes a per-family negative log-likelihood despite its name:

```text
root_term = logsumexp2(Pi[root, :]) - log2(S)
denominator = log2(1 - mean(exp2(E)))
nll = -(root_term - denominator)
```

The autograd bridge returns NLL directly, so users can call:

```python
loss = model()
loss.backward()
```

## Backward pass call graph

The autograd API backward path is:

```text
_GeneReconFunction.backward
implicit_grad_loglik_vjp_wave
Pi_wave_backward
_e_adjoint_and_theta_vjp
```

For `genewise=True`, `_GeneReconFunction.backward` calls these two pieces directly:

```text
_GeneReconFunction.backward
Pi_wave_backward
_e_adjoint_and_theta_vjp(genewise=True)
```

Inside `Pi_wave_backward`, the generic uniform path is:

```text
Pi_wave_backward
_dts_cross_differentiable
_self_loop_vjp_precompute
_gmres_self_loop_solve
_self_loop_Jt_apply
generic cross-clade VJP
uniform Pibar VJP correction
```

When the fused fast path is available, the self-loop and scalar-parameter cross-clade work can dispatch to:

```text
Pi_wave_backward
wave_backward_uniform_fused
dts_cross_backward_fused
uniform Pibar VJP correction
```

## Backward pass details

### 1. Saved forward state

`_GeneReconFunction.forward` saves `theta`, `Pi_wave_ordered`, `Pibar_wave_ordered`, `E`, `E_s1`, `E_s2`, `Ebar`, `log_pS`, `log_pD`, `log_pL`, and `max_transfer_mat`. In uniform mode, `ctx.transfer_mat` is `None`.

These are fixed-point values, so the backward pass uses implicit differentiation rather than differentiating through every forward iteration.

### 2. Pi adjoint initialization

`Pi_wave_backward` initializes `accumulated_rhs` from the NLL root term. For each root clade `r`:

```text
lse = logsumexp2(Pi_star_wave[r])
accumulated_rhs[r] = -exp2(Pi_star_wave[r] - lse)
```

The negative sign comes from differentiating NLL rather than log-likelihood.

### 3. Reverse wave traversal

The backward pass processes waves from root to leaves, which is reverse order relative to the forward wave schedule. Per wave it:

```text
rhs_k = accumulated_rhs[wave_rows]
active_mask = max_abs(rhs_k per clade) >= pruning_threshold
```

Inactive clades can be skipped. This is gradient pruning, not forward pruning.

### 4. Self-loop adjoint solve

Each wave forward step is a fixed point:

```text
Pi_W = g(Pi_W, frozen_cross_clade_terms, theta, E)
```

The adjoint for the self-loop must solve:

```text
(I - J_self^T) v_k = rhs_k
```

For the generic uniform path, `_self_loop_vjp_precompute` caches softmax weights for the six local terms and the uniform `Pibar` denominator:

```text
pibar_denom[c, s] = row_sum[c] - ancestor_sum[c, s]
pibar_inv_denom = 1 / pibar_denom where positive, else 0
```

`_gmres_self_loop_solve` then applies GMRES with `_self_loop_Jt_apply` as the matrix-vector product. The reason is practical: the comments note that the spectral radius of `J_self^T` can be close to one for uniform mode, so a short Neumann series can diverge or converge slowly.

For dense and topk modes, the generic path uses a finite Neumann series. Uniform is the exception in the generic path.

### 5. Uniform self-loop J transpose

The uniform `Pibar` contribution to `_self_loop_Jt_apply` is:

```text
v_Pibar = alpha * weight_of_Pibar_term
u_d = v_Pibar / pibar_denom
A = sum_s u_d[s]
correction[j] = sum_{s where j ancestor of s} u_d[s]
grad_Pi[j] += p_prime[j] * (A - correction[j])
```

This is the transpose of:

```text
Pibar[s] = log2(sum_j p_prime[j] - sum_{j ancestor of s} p_prime[j]) + constants
```

The same `ancestors_T` sparse matrix drives both the forward ancestor subtraction and the backward correction.

### 6. Fused uniform backward fast path

`Pi_wave_backward` uses `wave_backward_uniform_fused` only when all of these are true:

```text
Triton backward kernels import successfully
G == 1
pibar_mode == "uniform"
dtype == torch.float32
device.type == "cuda"
S > 256
```

This fused path computes self-loop softmax weights, a Neumann approximation for the self-loop adjoint, and per-element parameter-gradient contributions in one kernel per wave. It returns:

```text
v_k, aw0, aw1, aw2, aw345, aw3, aw4
```

Those `aw*` tensors are reduced into gradients for `log_pD`, `log_pS`, `E`, `Ebar`, `E_s1`, `E_s2`, and `max_transfer_mat`.

The fused path is not the same solve strategy as the generic uniform path. It uses `neumann_terms`, while the generic uniform path uses GMRES with a small iteration cap.

### 7. Cross-clade backward

If the wave had split terms, `Pi_wave_backward` propagates gradients through the cross-clade DTS computation.

When the fused path is active and `log_pD` and `log_pS` are scalar, it calls `dts_cross_backward_fused`. Otherwise it uses a PyTorch implementation in the main function.

The cross-clade VJP computes gradients for:

```text
Pi_left
Pi_right
Pibar_left
Pibar_right
log_pD
log_pS
```

Direct `Pi_left` and `Pi_right` gradients are added into `accumulated_rhs` for child clades. `Pibar_left` and `Pibar_right` are not final; they must be propagated through the uniform `Pibar(Pi)` formula.

### 8. Cross-clade Pibar gradient propagation

For child clades receiving a `Pibar` gradient from transfer terms, `Pi_wave_backward` applies the same uniform VJP formula used in the self-loop:

```text
p_prime = exp2(Pi_child - max(Pi_child))
denom = sum_j p_prime[j] - ancestor_sum
u = grad_Pibar / denom
A = sum_s u[s]
correction = ancestors_T @ u.T
pi_from_pibar = p_prime * (A - correction)
```

`pi_from_pibar` is added to the child clade rows in `accumulated_rhs`, so earlier waves see the complete adjoint.

### 9. Pi backward return values

`Pi_wave_backward` returns a dictionary with:

```text
v_Pi
grad_E
grad_Ebar
grad_E_s1
grad_E_s2
grad_log_pD
grad_log_pS
grad_max_transfer_mat
n_waves_total
n_waves_skipped
n_waves_processed
n_clades_total
n_clades_skipped
n_clades_active
```

There is no `grad_transfer_mat` in uniform mode because `transfer_mat` is not materialized and is not a theta-dependent dense object. The only transfer-related parameter gradient at this level is through `max_transfer_mat`.

### 10. E adjoint and theta VJP

`implicit_grad_loglik_vjp_wave` calls `_e_adjoint_and_theta_vjp` after `Pi_wave_backward`.

`_e_adjoint_and_theta_vjp` first forms the right-hand side for the E fixed point adjoint:

```text
q_E = pi_bwd["grad_E"] + direct_dNLL_dE_from_likelihood_denominator
```

It then chains `grad_Ebar` through the uniform `Ebar` computation by recomputing the formula under autograd. It also chains `grad_E_s1` and `grad_E_s2` through `gather_E_children`.

The E fixed point adjoint solve is:

```text
(I - G_E^T) wE = q_E
```

`G_E` is `E_step` viewed as a function of `E` only. The code builds a VJP closure with `torch.func.vjp`, solves with conjugate gradient, and falls back to GMRES if CG fails.

Finally, theta gradients are computed in two pieces.

`grad_theta_pi` comes from a scalar `param_loss` over the Pi-side accumulated parameter gradients:

```text
param_loss =
  sum(log_pS * grad_log_pS) +
  sum(log_pD * grad_log_pD) +
  sum(max_transfer_mat * (grad_max_transfer_mat + grad_Ebar))
```

The VJP for that scalar is taken through `extract_parameters_uniform`.

`gtheta_E` comes from differentiating one `E_step` at `E_star` with respect to theta-derived parameters, again using `extract_parameters_uniform` and `pibar_mode="uniform"`.

The final gradient returned to PyTorch is:

```text
grad_theta = grad_theta_pi + gtheta_E
```

## Technical choices in uniform mode

The dense transfer matrix is avoided. `transfer_mat` is `None` from parameter extraction onward, and `Recipients_mat` is skipped during device movement in the API and optimizer setup.

The transfer aggregate is exact for gpurec's preprocessed uniform recipient matrix. The shortcut is not a probabilistic approximation when the recipient weights are uniform over non-ancestors.

Ancestor exclusion includes self. `ancestors_dense[descendant, descendant]` is `1`, so self-transfer is excluded by `row_sum - ancestor_sum`.

The implementation uses base-2 log space throughout. Stabilized calculations subtract row maxima, use `exp2`, and then re-add the maxima.

`_safe_log2` is used in the subtraction formulas. This guards against float32 cancellation in `row_sum - ancestor_sum`.

Forward uniform `Pibar` currently uses a PyTorch sparse matrix multiply plus a generic Triton wave-step kernel. The uniform-specific fused forward kernel exists but is not called by `Pi_wave_forward`.

Backward uniform uses two different self-loop solve strategies. The generic path uses GMRES; the fused CUDA float32 large-S path uses a Neumann approximation in `wave_backward_uniform_fused`.

The sparse ancestor matrix is used in both directions. Forward computes ancestor sums as `Pi_exp @ ancestors_T`; backward computes the correction as `(ancestors_T @ u.T).T`.

## Practical caveats

`compute_log_likelihood` returns NLL by sign, despite its name. The autograd bridge explicitly treats it as NLL.

Uniform mode does not compute gradients for a dense recipient matrix. If recipient weights are parameters, this mode is the wrong path.

The public `GeneReconModel` does not support pairwise mode and does not expose a named full `genewise+specieswise` mode, although lower-level functions contain support for that shape.

If a senior engineer is debugging numerical differences, the first places to compare are `extract_parameters_uniform` versus dense extraction, `_compute_Pibar_inline` versus dense `Pi @ transfer_mat.T`, and `_e_adjoint_and_theta_vjp` where `grad_Ebar` is added into the transfer normalization gradient.
