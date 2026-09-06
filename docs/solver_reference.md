# gpurec/solver/ — architecture reference

_Compiled from a set of independent per-file technical reviews. Covers every file in
`gpurec/solver/` and its `hvp/` and `curvature/` subfolders._

## The big picture

`gpurec/solver/` is the part of gpurec responsible for **fitting** the DTL model's parameters —
computing the loss (negative log-likelihood: a number saying how badly the current parameter
guess explains the observed gene trees, lower is better) and its derivatives, and running the
actual optimization loops. The parameters being fit are:

- **`theta`** — duplication/loss/transfer rates, stored as log-base-2 numbers.
- **`receiver_weights`** (called `alpha` inside the curvature code) — one raw score per species,
  turned into probabilities via a **softmax** (a formula that converts a list of raw numbers into
  probabilities that sum to 1, by dividing each number's "size" by the total of all of them),
  controlling how likely each species is to receive a horizontal gene transfer.
- **`origination_weights`** (called `omega`) — the same idea, for which species a gene family's
  history is inferred to have started at.

Computing the likelihood itself requires an iterative numerical solve: two coupled tables, `E`
(extinction probability) and `Pi`/`Pibar` (reconciliation-history probability), each found by
**fixed-point iteration** — repeatedly applying an update formula until it stops changing.

A recurring theme throughout this directory is **gauge-fixing**. Because `receiver_weights` and
`origination_weights` each pass through their own softmax, and a softmax's output doesn't change
at all if you add the same number to every one of its raw inputs, there's a whole direction in
parameter space that changes nothing about the model — physicists call a "direction that changes
nothing" a **gauge direction**, and the practice of explicitly stepping around it is called
**gauge-fixing**. This matters because the more advanced fitting routines here use **Newton's
method** — a fitting method that uses **curvature** (how sharply the error surface bends, not just
which way it slopes) to converge faster and more reliably than plain gradient descent — and along
a gauge direction the curvature is exactly zero, which breaks the linear algebra Newton's method
needs to solve at every step. So before Newton's method can run safely on these parameters, the
gauge direction(s) have to be explicitly projected out, every time.

### Directory map

```
gpurec/solver/
    value_and_grad.py     — base "given theta, what's the loss and its gradient" (first-order only)
    penalties.py           — optional regularization terms
    krylov.py               — generic, domain-agnostic iterative linear-algebra solvers
    hvp/
        forward_tangent.py   — forward-mode directional derivative of the root scores ("J")
        vjp.py                  — root-score cotangent adapter over the production backward
        exact.py                 — the fully exact curvature (analytic Hessian-vector product)
    curvature/
        gauge.py               — shared gauge-fixing + damped-Newton machinery
        receiver.py              — joint Newton fit of theta + receiver_weights
        origination.py             — joint Newton fit of theta + receiver_weights + origination_weights
        genewise.py                  — same as origination.py, but theta and omega are per-gene-family
```

Roughly: `value_and_grad.py` + `penalties.py` + `krylov.py` are foundational, reusable primitives.
`hvp/` holds two different ways to compute a **Hessian-vector product** (HVP) — "if I nudge the
parameters in this direction, how does the curvature respond," computed without ever writing out
the full curvature matrix (the Hessian), which would be too large. `curvature/` holds the four
actual Newton's-method fitting loops that consume those HVPs.

---

## `value_and_grad.py` — the base loss-and-gradient computation

Every optimizer in `solver/` — plain gradient-based and curvature-aware alike — needs the same one
thing repeatedly: a function that takes a candidate parameter setting and returns the loss and its
gradient (a vector saying, for every individual parameter number, whether increasing or decreasing
it would improve the loss, and roughly by how much). This file is where that function is built. It
doesn't run the DTL fixed-point solve itself (`gpurec/core/inference/solver.py` does that) and it
doesn't hand-differentiate the model either (a separate hand-written backward pass in
`gpurec/api/_execution.py`/`_implicit_grad.py` does that) — this file is the glue: call those
pieces, add any optional penalty terms on top, handle joint optimization of `receiver_weights`/
`origination_weights` alongside `theta`, and hand back one clean `(loss, gradient)` pair in the
shape every optimizer expects. Per its own docstring, it's a "port" of an earlier codebase
(kernel-bench's `newton/vg.py`), re-pointed at gpurec's multi-batch machinery.

**`free_cuda_cache_if_tight(min_free_gib=4.0)`** — PyTorch normally keeps freed GPU memory in a
private pool for fast reuse rather than handing it back to the driver immediately, which can make
the driver's own "free memory" number look artificially low. If that number drops below the
threshold, this forces PyTorch to actually release cached memory (`torch.cuda.empty_cache()`), so
downstream memory-budget checks aren't fooled by a false alarm.

**`forward_solve(batch_statics, theta, receiver_weights, *, warm_E=None)`** — runs only the forward
direction (loss, no gradient). For a single batch (the path the curvature code uses), it calls
`solve_resident_e_pi` — the actual DTL fixed-point solver — with a **warm start** (seeding the
iterative solver with a previous nearby answer instead of starting from scratch, so it converges in
far fewer steps), collects the 13 returned tensors into a `saved` dictionary the second-order code
needs to reuse, and computes the loss. For multiple batches it sums each batch's loss via
`stream_batches` and returns `saved=None`, since there's no single well-defined intermediate state
across independently-solved batches.

**`make_value_and_grad(batch_statics, receiver_weights, *, theta_shape=None, grad_avg_K=1,
prior=None, tree_penalty=None, optimize_receiver=False, origination_weights=None,
optimize_origination=False, tv_penalty=None, origination_penalty=None, group_index=None)`** — the
file's main function, a **closure factory**: it doesn't compute a loss/gradient itself, it builds
and returns another function `f` that "remembers" all these settings and is what an optimizer
actually calls, repeatedly, at different candidate parameter values.

Setup done once: resolves `theta_shape` (default `(S,3)`), precomputes the ridge-prior reference
value if `prior=(lam, theta_ref)` was given, precomputes the species-tree parent/child edge lists
if `tree_penalty` was given, and resolves grouping info if `group_index` was given (a scheme where
several species share one rate parameter).

The returned closure `f(theta_vec, *, warm_E=None, want_grad=True)`:
1. Unpacks the flat input `zvec` into `[theta][receiver logits, if optimizing][origination logits,
   if optimizing]` blocks — the codebase's convention for this combined vector is `z`.
2. Expands grouped `theta` back to one row per species (`group_expand`, a no-op if not grouping).
3. Calls `stream_batches` (`gpurec/api/_execution.py`) to run the DTL solve and, if a gradient was
   requested, the hand-written backward pass, summing across batches.
4. Adds each active penalty's *value* on top: ridge (`0.5 * lam * sum((theta-theta_ref)^2)`), tree
   penalty (`0.5 * lam_tree * sum((theta[child]-theta[parent])^2)` over every tree edge), TV penalty
   (via `tv_prior_and_grad`), and — if origination is being optimized — the origination penalty.
5. If a gradient was requested, optionally re-runs `stream_batches` `grad_avg_K - 1` more times and
   averages the gradients together (not the loss). This exists because the hand-written backward
   pass uses GPU "atomic" operations whose combining order varies run to run, adding tiny
   floating-point noise to each individual gradient evaluation; averaging several evaluations at
   the same point reduces that noise. Default 1 = no extra cost; turned up by the curvature code
   where a cleaner gradient matters more than speed.
6. Adds each active penalty's *gradient*: the ridge gradient `lam*(theta-theta_ref)`; the tree
   penalty's gradient, built with `index_add_` so a parent with several children correctly
   accumulates each child's contribution; the TV penalty's gradient.
7. Folds grouped gradients back down (`group_reduce`) and appends the receiver/origination gradient
   blocks if they were being jointly optimized.

**The math.** The core loss computation is *not* done via PyTorch's automatic differentiation —
everything runs under `torch.no_grad()`. Instead a separate, hand-written backward pass directly
implements the calculus, exploiting the DTL model's specific structure for speed. This file never
touches that calculus for the core loss; it only receives the finished gradient pieces back. The
one place real calculus happens directly in this file is the penalty terms — plain sums of squared
differences, whose derivative (`(a-b)^2` differentiates to `2*(a-b)`) is applied by hand.
`receiver_weights`/`origination_weights` are raw, pre-softmax numbers; this file passes them
through unchanged and relies on the backward pass having already differentiated through the softmax
correctly — it must never touch those gradient blocks with its own penalty math.

**Connections.** Depends on `gpurec/api/_batch_state.py` (`_BatchStatic`), `gpurec/api/_execution.py`
(`stream_batches`), `gpurec/config/memory.py` (numeric defaults), `gpurec/core/inference/solver.py`
(`solve_resident_e_pi`, `nll_from_root_rows`), and `penalties.py` (all four penalty helpers,
`group_expand`/`group_reduce`). Everything else in `solver/` — the `fit/` optimizers, all three
`curvature/` Newton loops, and `hvp/exact.py` (for the single-batch `saved` snapshot) — builds on
top of `make_value_and_grad`/`forward_solve`. Heavily exercised by the test/gate suite (finite-
difference gradient checks, HVP checks, memory-gate tests), evidence it's treated as foundational.

---

## `penalties.py` — optional regularization terms

A toolbox of penalty terms addable on top of the main loss, all built so that turned off (strength
zero or `None`) they're a byte-identical no-op. Everything is plain PyTorch, log2-space, no custom
GPU kernels.

- **`tv_prior_and_grad(theta, sp_parent, lam, eps=1e-3)`** — the main regularizer: a "total-variation"
  / "fused-lasso" prior saying neighboring species on the tree should have similar rates. For every
  non-root species it computes the difference between its rate and its tree-parent's rate, applies
  a **pseudo-Huber** smoothing function `rho(d) = sqrt(d² + eps²) - eps`, sums it (weighted by
  `lam`), and hand-computes the exact gradient (`+` to the child, `-` to the parent, combined
  correctly across multiple children via `index_add_`).

  *Why pseudo-Huber specifically*: a plain squared-difference ("L2") penalty barely pulls small
  differences toward zero and over-punishes one big real jump; a plain absolute-difference ("L1")
  penalty applies a constant pull that *can* force differences to exactly zero (useful — it creates
  flat, identical-rate blocks across the tree) but has a sharp, non-smooth corner exactly at zero.
  Pseudo-Huber behaves like L1 far from zero (`sqrt(d²+eps²) ≈ |d|` when `|d| >> eps`, so large real
  jumps aren't over-punished) and like a smooth quadratic bowl close to zero (no sharp corner,
  well-defined slope everywhere) — `eps` sets the width of that smoothing window.

- **`origination_log_pO(omega)` / `origination_log_pO_floored(omega, floor)`** — turn raw origination
  scores into `log2` probabilities via softmax; the floored version blends in a small uniform-over-
  species probability so the optimizer can never drive any one species' probability all the way to
  zero (which would be numerically unstable and an implausibly overconfident belief).

- **`OriginationPenalty`** (dataclass) — bundles: `l2` (ridge on raw `omega`), `depth_lambda`
  (penalizes putting origination probability far from the tree root, using `species_node_depths`),
  `root_lambda` (rewards putting probability mass specifically on the root), `dirichlet_c` +
  `barrier_kind` (an anti-concentration term discouraging the distribution from collapsing onto a
  few species — three variants: `"meanlog"` cross-entropy against a reference distribution,
  `"simpson"` sum-of-squared-probabilities, `"renyi2"` its log), `floor`.

- **`origination_penalty_value`/`origination_penalty_and_grad`** — compute the penalty's value, and
  (unlike the TV prior) its gradient via PyTorch autograd rather than by hand, specifically because
  the floored-probability chain rule is easy to get wrong by hand across every option combination.

- **`PenaltyOptions`** (dataclass) — the object actually wired into `GpurecConfig.regularizer`
  (default constructed via a *lazily*-imported helper in `gpurec/config/gpurec_config.py`,
  specifically to avoid a circular-import chain: `gpurec.solver` → `value_and_grad` →
  `gpurec.api._execution` → `_implicit_grad` → `gpurec.config`). Its defaults live in
  `gpurec/config/defaults.toml`'s `[regularizer]`/`[regularizer.origination]` sections, all zero
  (off) out of the box.

- **`group_expand`/`group_reduce`** — reshaping helpers for a mode where several species share one
  rate parameter: expand a `[G,K]` group-parameter tensor out to `[S,K]` per-species (copy each
  group's row to every species in it), and reduce a `[S,K]` gradient back down to `[G,K]` (sum each
  group's member gradients — the standard calculus rule for a value copied into several places).

**Connections.** `gpurec/fit/optimize.py` reads `config.regularizer.origination` as the default
`origination_penalty` (an explicit argument still wins) and requires callers to build/pass
`tv_penalty` explicitly — it's not auto-sourced from config. `value_and_grad.py` is the direct,
line-level consumer of every function here.

---

## `krylov.py` — generic iterative linear algebra

Zero domain knowledge of DTL, species trees, or gene trees — pure numerical building blocks that
only ever need a function `Av` (given a vector, return "the matrix times that vector") supplied by
the caller. This is what lets Newton's method work on problems with tens of thousands of parameters
without ever building or storing the full curvature matrix (which would need memory quadratic in
the parameter count). Its own docstring says it was "copied verbatim from kernel-bench `newton/cg.py`."

- **`_lanczos_tridiag(Av, p, m, ...)`** (private) — the shared engine behind the two eigenvalue
  routines below. Builds a sequence of `m` vectors (the **Lanczos algorithm**: each new vector is
  `Av` applied to the previous one, with everything already explained by earlier vectors subtracted
  off, plus a "full reorthogonalization" correction against *every* previous vector to counteract
  floating-point drift). This spans a **Krylov subspace** — the space reachable by repeatedly
  applying the matrix to a starting vector — and produces a small tridiagonal matrix (nonzero only
  on the main diagonal and its two neighbors) that cheaply summarizes the big matrix's behavior in
  that subspace. Stops early if a newly built vector becomes negligibly small relative to the
  working floating-point precision and the scale of the numbers seen so far.

- **`lanczos_extremes(Av, p, m=40, ...)`** — estimates the smallest and largest **eigenvalues**
  (numbers describing the matrix's stretching/curving strength along the directions it doesn't
  rotate) by handing the small tridiagonal matrix to an exact solver (`scipy.linalg.eigh_tridiagonal`).
  The smallest eigenvalue is the harder one to pin down (can even get the sign wrong below ~40 steps).

- **`lanczos_min_eigpair(Av, p, m=120, ...)`** — same idea, but returns the smallest eigenvalue
  *and* its associated direction, needing more steps (~120) since the relevant region of eigenvalues
  is numerically "clustered" (close together, hard to separate).

- **`steihaug_cg(Av, b, delta, ...)`** — a **trust-region** solver: approximately minimizes a
  quadratic model of the loss while keeping the step within a ball of radius `delta` around the
  start (because a curvature model built locally stops being believable once you move too far). If
  it detects **negative curvature** along the current search direction (the surface actually curves
  *downward* that way — like walking off a mountain pass rather than settling into a bowl — meaning
  the quadratic model has no bottom in that direction), it stops and jumps straight to the trust-
  region boundary along that direction rather than trying to "solve" a direction with no minimum.

- **`cg_witness(Av, b, ...)`** — a plain conjugate-gradient solver (an iterative technique for
  solving "matrix times x equals b," building up the answer direction by direction, each new
  direction chosen not to undo progress made along earlier ones) that, on hitting negative
  curvature, doesn't just fail — it returns a **certificate**: a number (a Rayleigh quotient) that
  mathematically proves how negative the true, undamped matrix's smallest eigenvalue is, letting the
  caller grow its Levenberg-Marquardt damping (adding a positive number to the matrix's diagonal to
  force better behavior) by exactly the right amount and retry.

- **`cg_solve(Av, b, ..., x0=None)`** — the simplest solver, meant for when the caller trusts the
  matrix is genuinely positive-definite (curves upward everywhere, i.e. has a real unique minimum).
  Notably, if it *does* hit non-positive curvature (which "should" be impossible for its intended
  use), it deliberately does **not** try to hack around it — an explicit code comment documents and
  rejects an old, dangerous fallback (`max(rayleigh, 1e-12)`, which always collapsed to dividing by
  `1e-12` and produced a step ~10¹² times too large) in favor of honestly reporting failure and
  letting the caller fall back to a safe steepest-descent step.

**Why "negative curvature" matters, in one picture.** Standing at the bottom of a bowl, every
direction curves back up toward you (positive curvature — there's a real minimum nearby). Standing
at a mountain pass or hilltop, some directions curve *downward* (negative curvature) — following
them, the model would predict the loss dropping forever with no bottom. Newton's method, which
solves "where does the curved model bottom out," has no sane answer to give along such a direction;
blindly trusting the algebra sends the optimizer flying off in a meaningless direction. Every
function above explicitly checks for this at every iteration and reacts differently: cap the step
(`steihaug_cg`), report a certificate (`cg_witness`), or admit failure (`cg_solve`).

**Connections.** `curvature/receiver.py` and `curvature/origination.py` import `cg_solve`;
`curvature/gauge.py` imports `cg_witness`, `lanczos_extremes`, `lanczos_min_eigpair`;
`curvature/genewise.py` reuses these "verbatim" per its own comment (no direct import found).
`steihaug_cg` did not turn up in a grep of current callers within `solver/` — it may be legacy,
held in reserve, or invoked indirectly.

---

## `hvp/` — two ways to compute a Hessian-vector product

### `forward_tangent.py` — the forward half ("J")

Computes, for one chosen nudge direction `v` in `theta` (and optionally `alpha`), how *every*
downstream quantity in the whole forward solve responds — ending in the root-level scores
`Pi_root`. This is **forward-mode differentiation**: start from one nudge to the inputs and push it
forward through the calculation, collecting how every output responds — as opposed to reverse-mode
(used everywhere else in the codebase for the ordinary gradient), which starts from one output and
works backward to find how every input affects it. Reverse mode is efficient for "one output,
sensitivity to every input" (exactly what the ordinary loss gradient needs); forward mode is
efficient for "many outputs, sensitivity to one input direction" — exactly what's needed here, since
Hessian-vector products are evaluated one direction at a time.

- **`param_jvp_uniform`/`param_jvp_weighted`** — differentiate the one-shot (non-iterative) formula
  that turns `theta` (and `alpha`, in the weighted case) into event log-probabilities, using
  PyTorch's built-in forward-mode autodiff (`torch.func.jvp`) directly on the same
  `extract_parameters_*` formula the real forward solve uses. The weighted version additionally
  carries a coupling term the uniform version doesn't need: when transfer recipients aren't equally
  likely, the "how much probability mass lands on valid recipients" correction itself depends on
  `alpha`, so skipping that dependency would understate `alpha`'s effect on the transfer rate.

- **`jvp_root_scores(static, theta, v, sv, ...)`** — the main entry point. Walks the *same*
  wave-by-wave order (tips-to-root batches of gene-tree clades) the real solve uses, computing the
  cross-wave tangent then the within-wave "self-loop" tangent per wave, and reads off the root rows
  at the end.

**The core idea — threading a tangent through a fixed-point iteration.** The forward solve isn't one
formula computed once; the E-table and the Pi/Pibar self-loop are each **fixed points**: `x* =
f(theta, x*)`, i.e. the answer is defined as "the value that, fed back into the formula, reproduces
itself." Nudging `theta` shifts `x*` two ways at once — directly (the formula itself moves) and
indirectly (the shifted `x*` feeds back into the same formula) — so the true tangent obeys its own
self-referential equation with the unknown tangent on both sides; you can't read it off from
evaluating the formula once. The E tangent uses its contraction iteration; the Pi tangent solves
the differentiated tree system by exact elimination. Rows outside the scaled-linear dynamic range
use the same masked iterative fallback as the primal.

**Connections.** Uses `gpurec/core/inference/solver.py`, `gpurec/core/parameters/extract_parameters.py`,
and several `gpurec/core/kernels/*_tangent.py` kernels. Called by `exact.py` with
`return_full=True`, because the second-order sweep needs every intermediate tangent.

### `exact.py` — the fully exact curvature (largest file in `solver/`, ~1000 lines)

Computes the *true* Hessian-vector product by differentiating the entire solve-then-gradient
pipeline a second time — "**forward-over-reverse**" differentiation. It is used directly by the
Newton fitting and curvature-certification paths.

- **`build_point_cache(static, theta, col_weights, sv, ...)`** — runs the *existing*, already-
  verified first-order backward pass (`vjp_root_to_theta`) exactly once, but with a `cache`
  dictionary supplied so it stashes, per wave, its solved adjoint value `v_k`, the split-likelihood
  numbers, and the pruning mask, plus the E-side solved quantity `wE` — instead of throwing them
  away once added to the running gradient total (which is what happens normally).

- **`make_exact_hvp`/`make_exact_hvp_single`** — dispatch (single- vs multi-batch) and the main
  routine: builds/reuses the point cache, does one-time direction-independent setup, and returns a
  closure `hvp(u_vec)` a Krylov solver calls repeatedly at different directions.

- **`_make_exact_hvp_streaming(batch_statics, ...)`** — the multi-batch version. Since the total
  loss sums over families and batches split families into non-overlapping groups, the true HVP is
  exactly the sum of each batch's own HVP. Every call loops over every batch, redoes that batch's
  forward solve and cache from scratch, applies it, adds the contribution, and frees memory before
  moving on — deliberately paying extra compute per call in exchange for bounded memory (holding
  every batch's cache resident at once would exceed GPU memory).

**The algorithm, in four steps.** (1) *Forward tangent* — `jvp_root_scores` (from `forward_tangent.py`)
computes how every intermediate quantity moves for a chosen nudge `u`. (2) *Forward-over-reverse* —
the code differentiates the *backward pass itself* along that same tangent: wherever
the ordinary backward pass combined a frozen value (`Pi`, `Pibar`, `E`, `v_k`) into a gradient
contribution, this file calls a matching "second-order" kernel (`wave_backward_so`,
`dts_backward_so`, `e_step_backward_so` — literally named "directional derivative" kernels in their
own source) that takes both the frozen value *and* its tangent and produces the directional
derivative of the original formula. (3) *A fresh linear solve, per wave* — because the backward pass
itself contained a linear solve per wave, differentiating it again produces a *new* linear system
with a new right-hand side built from the tangent quantities, solved with the same solver
(`solve_reconciliation_wave_vjp`) — the "tangent-adjoint sweep." The same happens on the E side (a
Neumann-series solve — repeatedly applying a shrinking operator and summing — redone with a tangent
right-hand side). (4) *The smooth parameter head* — `theta`/`receiver_weights` first pass through a
small, smooth, non-iterative softmax-like formula before hitting the big iterative solve; rather
than hand-differentiate that formula twice, the code builds its computation graph once
(`create_graph=True`) and lets PyTorch's autograd differentiate it a second time using the
accumulated tangent cotangents.

**Why the point cache matters.** An iterative Newton solver calls `hvp(u)` many times *in a row at
the same `theta`* — only `u` changes; `theta` is fixed for the whole inner loop. Everything in
`build_point_cache` (every wave's `v_k`, pruning masks, accumulated totals, `wE`, the smooth head's
graph) depends only on `theta`, never on `u`. Without caching, every single `hvp(u)` call would redo
that *entire* first-order backward pass. With it, that work happens once per outer Newton point, and
each subsequent call only pays for the genuinely `u`-dependent work — matching the file's own
docstring: "theta is fixed across all CG iterations, so the cache amortizes."

**Connections.** Uses `forward_tangent.py` (`jvp_root_scores`), `vjp.py`
(`vjp_root_to_theta`), `value_and_grad.py` (`forward_solve`, `free_cuda_cache_if_tight`), and several
`gpurec/core/kernels/*_so.py` second-order kernels. Called by all three `curvature/` Newton loops
(`receiver.py`'s and `origination.py`'s `build_joint_hvp`, `genewise.py`'s per-batch construction)
and directly by `gpurec/fit/optimize.py`'s Newton-polish stage and ridge-selection logic.

---

## `curvature/` — the four joint Newton fitting loops

### `gauge.py` — shared gauge-fixing + damped-Newton machinery

Factors out everything that's identical across `receiver.py`'s one-gauge-direction case and
`origination.py`'s two-gauge-direction case (and, by extension, `genewise.py`'s many-gauge-direction
case): the generic damped-Newton loop and the generic "is this really a minimum" certificate. Works
entirely on a generic vector `z`, a generic projector `proj` (the caller's "kill the gauge
direction" operation), and generic caller-supplied `hvp`/`vg` functions — it has zero built-in
knowledge of species trees, gene trees, or rates.

- **`resolve_newton(newton, **overrides)`** — resolves a `NewtonOptions` settings object from an
  optional base plus explicit-kwarg overrides, a backward-compatibility shim for old call sites that
  passed individual settings directly.

- **`gauge_operator(hvp, proj, ..., penalty_hvp=None, ridge=0.0)`** — builds `A_z(v) = P_z (H +
  penalty)(P_z v) + ridge * P_z v`: project the input through `proj`, apply the raw curvature (plus
  any penalty curvature), project the *result* again, optionally add a uniform "ridge" shift for
  numerical robustness. Projecting on both sides guarantees the gauge direction maps to exactly zero
  regardless of what the raw `hvp` does with it.

- **`certify_min(hvp, proj, p, ...)`** — is the curvature genuinely positive everywhere that
  matters (a true minimum), or does it curve the wrong way somewhere (a saddle point)? Estimates the
  smallest eigenvalue after **deflation**: the known-pointless gauge direction is artificially made
  to look enormously curved (`Av_deflated(v) = Av(v) + shift_C * (v - proj(v))`, where `shift_C` is
  set safely above the legitimate spectrum), so a search for "the smallest curvature value anywhere"
  can never mistake the gauge direction (whose true curvature is exactly zero) for the answer.
  Returns the smallest genuine eigenvalue, a residual measuring trustworthiness, the worst-case
  direction, and a sanity check that that direction really did land in the gauge-fixed subspace.

- **`newton_min(z, p_dim, proj, vg, build_hvp, ...)`** — the actual loop. Each iteration: solve the
  **Levenberg-Marquardt-damped** system (curvature-plus-`lam_damp`-times-identity, times step, equals
  minus gradient) via `cg_witness`; if negative curvature is witnessed, grow `lam_damp` by exactly
  the amount the witness certificate proves is needed and retry (up to `max_bumps` times, then fall
  back to plain steepest descent); check the step actually helps via **Armijo backtracking** (try the
  full step; if it doesn't beat a small guaranteed fraction of the straight-line prediction, halve it
  and retry, up to `ls_max` times); after an accepted step, relax the damping (if the full step was
  accepted outright) or tighten it (if backtracking was needed), and re-project `z` back onto the
  gauge-fixed subspace to stop floating-point drift from accumulating. Stops on small gradient, on a
  stalled improvement twice in a row, or on damping hitting its ceiling with no accepted step.

**How the three siblings plug in.** Each of `receiver.py`/`origination.py`/`genewise.py` supplies its
own `proj` (mean-subtract however many softmax blocks it has), its own `build_hvp` (rerun the DTL
forward solve at a point and wrap `hvp/exact.py`'s exact HVP with `gauge_operator`), and its own `vg`
(built from `make_value_and_grad`, folding in whatever penalties are active) — then calls `gauge.py`'s
`newton_min`/`certify_min` unchanged. None of the Newton-loop mechanics (damping schedule, negative-
curvature recovery, line search, stall detection) is duplicated in any of the three.

### `receiver.py` — joint fit of theta + receiver_weights

Fits `theta` and `alpha` (receiver weights) together because they're genuinely entangled — a
transfer event's probability depends jointly on both — so fitting them in separate alternating
rounds is blind to how a change in one would have changed the best value of the other, and ignores
curvature's speed/reliability advantage entirely.

- **`proj_alpha(g_alpha)` / `proj_z(u, theta_numel)`** — the one-gauge-direction projector:
  `theta` untouched, `alpha` mean-subtracted.
- **`build_joint_hvp(static, theta, alpha, ...)`** — runs `forward_solve`, builds/reuses the point
  cache (`hvp/exact.py`), and returns the exact joint HVP. Explicitly rejects a perfectly uniform
  `alpha` (`receiver_weights_are_uniform`) — at a perfectly flat starting point the softmax's own
  derivative is zero, so the curvature calculation would silently degenerate rather than raise a
  clear error; the docstring tells callers to nudge `alpha` off-uniform first (e.g. a short warmup).
- **`make_gauge_operator`** — one-line wrapper of `gauge.gauge_operator` supplying `proj_z`.
- **`certify_joint_min`** — wraps `gauge.certify_min` with this file's projector and HVP.
- **`receiver_information(static, theta, alpha, ...)`** — computes standard errors on the receiver
  weights at a fitted point, using the observed curvature as the (inverse) covariance — the standard
  "Fisher information" idea from statistics: sharper curvature around the fitted value means it's
  more precisely pinned down. Solves one column of the covariance matrix per species via `cg_solve`;
  because each solve uses the *full joint* system, the result correctly accounts for `alpha`'s
  entanglement with `theta` ("Schur-complement-correct"). Also converts raw-score uncertainty into
  probability-space uncertainty via the delta method (using the softmax's own local slope).
- **`newton_joint(static, theta0, alpha0, ...)`** — the fitting entry point: builds the combined
  vector, wires in optional ridge/tree-smoothness penalties on `theta` (with matching penalty
  curvature via the private `_penalty_hvp`), and calls `gauge.newton_min`.

**Why cross terms matter.** The full curvature matrix has four blocks: theta-vs-theta,
alpha-vs-alpha, and two cross blocks (theta-vs-alpha and its mirror) describing how nudging one
changes the *slope* with respect to the other. A theta-only Hessian can never see the cross block —
exactly the blindness that alternating (fit-theta-then-alpha) fitting has too. `hvp/exact.py`
computes the full matrix's effect, cross terms included, in one analytic pass.

**Connections.** The only one of the three joint-fit files whose `newton_joint` was found actually
wired into a production `fit/` driver — `gpurec/fit/newton_cg.py` calls it. `_penalty_hvp` and
`_tree_edges` are reused verbatim by `origination.py`.

### `origination.py` — joint fit of theta + receiver_weights + origination_weights

Same idea as `receiver.py`, extended to a third block, `omega` (origination weights), giving **two**
gauge directions to remove instead of one.

- **`proj_z(u, theta_numel, S)`** — leaves `theta` untouched, mean-subtracts the `alpha` block and,
  separately, the `omega` block.
- **`build_joint_hvp`** — same shape as `receiver.py`'s, extended with origination log-probabilities
  and checked against both `receiver_weights_are_uniform` and `origination_weights_are_uniform`.
- **`origination_information`** — the origination-weight analogue of `receiver_information`.
- **`newton_joint(static, theta0, alpha0, omega0, ...)`** — reuses `receiver.py`'s `_penalty_hvp` and
  `_tree_edges` verbatim (the module docstring calls these "IDENTICAL" needs regardless of how many
  gauge blocks there are), and otherwise mirrors `receiver.py`'s structure with the extra block
  threaded through.

**A notable gap found in this review:** `penalties.py`'s dedicated `OriginationPenalty`/
`origination_penalty_and_grad` machinery (the anti-collapse/depth/root-mass penalties described
above) is **not** currently wired into this file's `newton_joint` — its call to `make_value_and_grad`
passes `prior` and `tree_penalty` (both theta-only) but never `origination_penalty`. So despite
living in the same package, the origination-specific regularizer and the joint Newton fit are not
currently connected; the loss this file minimizes includes only the two theta-only penalties.

**Connections.** `genewise.py` reuses this file's `build_joint_hvp` verbatim for its single-batch
case. Like `receiver.py`'s `newton_joint`, this file's own `newton_joint` was **not** found to be
called by any `fit/` driver in this review — it appears to be validated, tested library code not yet
plugged into a production recipe.

### `genewise.py` — same as `origination.py`, but per-gene-family (largest file in `curvature/`)

`theta` here has shape `[G,3]` (one rate triple per gene family) and `omega` has shape `[G,S]` (one
origination distribution per family) instead of shared `[3]`/`[S]`. `origination.py` can't be reused
directly because it assumes exactly one shared `omega`, and — critically — each family's `omega` row
has its **own** softmax, so each family contributes its **own** pointless gauge direction: `G`
separate zero-curvature directions instead of one shared one.

- **`proj_z_genewise(u, theta_numel, S, G)`** — `theta` untouched, `alpha` mean-subtracted once
  (shared), and **each** of the `G` `omega` rows mean-subtracted **independently** — removing all
  `G+1` gauge directions.
- **`newton_joint_genewise(static, theta0, alpha0, omega0, ...)`** — the fitting entry point,
  requiring both `alpha0` and `omega0` to start non-uniform, running entirely in double precision.
- **`make_multibatch_joint_hvp_genewise`/`multibatch_joint_vg_genewise`** — when the gene-family set
  doesn't fit in one GPU batch, these gather each batch's own `theta`/`omega` rows, build/run a
  per-batch HVP or loss-and-grad, and scatter results back (`index_add_` for `theta`/`omega`, since
  each family lives in exactly one batch and there's no double-counting; a plain running sum for
  `alpha`, which every family shares). This works because the total curvature/gradient over the
  whole dataset is exactly the sum of each family's own contribution.
- **`_assemble_dense_arrowhead`/`newton_step_joint`** — explicitly marked **TEST-ONLY**, not called
  by production. These demonstrate and exploit a real mathematical structure — an **arrowhead
  matrix**: family `g`'s own theta/omega block has no *direct* curvature connection to family `g'`'s
  block (block-diagonal), and the only thing linking every family is the shared `alpha` (the
  "spine" of the arrow, since `alpha` affects every family's loss). Solving with this structure
  costs roughly `G × (block size)³` plus one shared-block solve, instead of one dense solve over the
  whole `3G + S + GS`-sized system. But the file's own comment block and warning docstrings are
  explicit that this structured solver is **not** wired into the real fit: it assumes the
  per-family omega-curvature block only curves upward (positive semi-definite), which the real one
  is not guaranteed to (it can be indefinite), and it uses a different flat variable ordering than
  the real fit. **Production instead runs matrix-free conjugate gradient directly against the full
  gauge-projected HVP** (via `gauge.newton_min`'s `cg_witness`) — it gets its speed from never
  building the dense Hessian at all, not from hand-exploiting the arrowhead shape.

**Connections.** Reuses `origination.py`'s `build_joint_hvp` verbatim for the single-batch case, but
**cannot** reuse `origination.py`'s `make_value_and_grad`-based loss/gradient path (which hard-codes
a single global `omega[S]`) — hence its own `evaluate_static_loss_grad`/`multibatch_joint_vg_genewise`
path. A documented "API asymmetry" the file has to track carefully: `forward_solve` takes the
*full* `[G,3]` theta and self-selects a batch's rows; `make_exact_hvp` takes *already-selected*
per-batch tensors; `evaluate_static_loss_grad` takes an already-selected theta but the *full*
`omega[G,S]` (it self-selects omega's rows internally). **As of this review, `newton_joint_genewise`
was found to be called only from tests, not from `gpurec/fit/genewise_fit.py`** (which has its own,
separate fitting loop built directly on `make_exact_hvp`) — validated, tested library code not yet
plugged into the genewise fitting pipeline, the same situation as `origination.py`'s `newton_joint`.

---

## Cross-cutting findings from this review

- **Only `receiver.py`'s `newton_joint` is currently used by a production `fit/` driver**
  (`gpurec/fit/newton_cg.py`). `origination.py`'s and `genewise.py`'s `newton_joint`/
  `newton_joint_genewise` are validated and tested (finite-difference gradient/HVP checks, gauge
  certification) but not called from any `fit/` recipe found in this review.
- **`origination.py`'s `newton_joint` doesn't wire in `OriginationPenalty`**, despite
  `penalties.py` having dedicated machinery for exactly this parameter block.
- **`krylov.py`'s `steihaug_cg`** (the trust-region solver) didn't turn up in a grep of current
  callers within `solver/`.
- **`genewise.py`'s arrowhead-structured direct solver is explicit test-only scaffolding** — a
  real, verified mathematical structure, deliberately not used in production because its
  positive-semi-definite assumption doesn't hold for the real per-family omega curvature.

None of the above are bugs — everything described as "not wired in" is working, tested code sitting
one call away from being used, not broken code. They're flagged here simply because a systematic
per-file review surfaces exactly this kind of "built but not yet connected" gap.
