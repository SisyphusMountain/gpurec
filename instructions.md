Here’s a **VJP/adjoint (reverse-mode)** version of your prompt, tailored for getting $\nabla_{\tau,\delta,\lambda} L(\Pi^*)$ efficiently without unrolling the fixed-point iterations.

---

# Task: Implicit **VJP** differentiation of fixed-point (log) updates for $E$ and $\Pi$

## Context

We extend an AleRax-like reconciliation tool. In `src/reconciliation/likelihood.py`:

* `log_E_step` updates $E$ (log-extinction),
* `Pi_update_ccp_log` updates $\Pi$ (log clade probabilities).

Both are applied by fixed-point iteration to convergence. Parameters:

* $\tau$ (transfer), $\delta$ (duplication), $\lambda$ (loss).

We want **gradients of a scalar objective** $\ell(\tau,\delta,\lambda) := L(\Pi^*)$ using **reverse-mode** implicit differentiation with **VJP**s (`torch.func.vjp`). **Do not unroll** the fixed-point iterations.

---

## Notation & assumptions

* $E\in\mathbb{R}^{n_E},\; \Pi\in\mathbb{R}^{n_\Pi}$, parameters $\theta=(\tau,\delta,\lambda)$.
* Update maps (log-space; “log” omitted in notation):

  * $G(E;\tau,\delta,\lambda)$ = `log_E_step` (contractive in $E$),
  * $F(\Pi,E;\tau,\delta)$ = `Pi_update_ccp_log` (contractive in $\Pi$).
* Fixed points (unique by Banach):

  $$
  E^*=G(E^*;\tau,\delta,\lambda),\qquad
  \Pi^*=F(\Pi^*,E^*;\tau,\delta).
  $$
* Define Jacobians at the fixed point:

  $$
  G_E=\partial_E G,\quad
  F_\Pi=\partial_\Pi F,\quad
  F_E=\partial_E F,\quad
  F_\tau=\partial_\tau F,\; F_\delta=\partial_\delta F,\quad
  G_\tau,\; G_\delta,\; G_\lambda.
  $$

  Contraction $\Rightarrow$ $I-G_E$ and $I-F_\Pi$ are invertible.

We optimize $\ell(\tau,\delta,\lambda)=L(\Pi^*)$ where $L:\mathbb{R}^{n_\Pi}\to\mathbb{R}$ is $\mathcal{C}^1$.

---

## Adjoint (implicit) gradient formula

Let $\phi:=\partial_\Pi L(\Pi^*)$.

1. Solve for $v\in\mathbb{R}^{n_\Pi}$:

$$
\boxed{(I - F_\Pi^\top)\,v = \phi}
$$

2. Define $q := F_E^\top v \in \mathbb{R}^{n_E}$.

3. Solve for $w\in\mathbb{R}^{n_E}$:

$$
\boxed{(I - G_E^\top)\,w = q}
$$

4. Gradients (one readout each from the pullbacks):

$$
\boxed{
\nabla_\tau \ell = F_\tau^\top v + G_\tau^\top w,\quad
\nabla_\delta \ell = F_\delta^\top v + G_\delta^\top w,\quad
\nabla_\lambda \ell = G_\lambda^\top w
}
$$

If $L$ also depends on $E^*$: add $\partial_E L$ to $q$.
If $F$ depends on $\lambda$: add $F_\lambda^\top v$ to $\nabla_\lambda \ell$.

These require only **two linear solves** of sizes $n_\Pi$ and $n_E$, plus **one** VJP evaluation per parameter block to read out the terms.

---

## High-level algorithm (PyTorch, **VJP**)

We work **at the converged fixed points** $(E^*,\Pi^*)$. Use **persistent VJP closures** to avoid recomputing forwards.

1. Build a pullback for $F$ at $(\Pi^*,E^*,\tau,\delta)$:

```python
_, vjpF = torch.func.vjp(lambda Pi, E, t, d: F(Pi, E, t, d),
                         Pi_star, E_star, tau, delta)
```

Calling `vjpF(g)` returns the tuple
$((\partial_\Pi F)^\top g,\; (\partial_E F)^\top g,\; (\partial_\tau F)^\top g,\; (\partial_\delta F)^\top g)$.

2. Build a pullback for $G$ at $(E^*,\tau,\delta,\lambda)$:

```python
_, vjpG = torch.func.vjp(lambda E, t, d, l: G(E, t, d, l),
                         E_star, tau, delta, lam)
```

Calling `vjpG(h)` returns
$((\partial_E G)^\top h,\; (\partial_\tau G)^\top h,\; (\partial_\delta G)^\top h,\; (\partial_\lambda G)^\top h)$.

3. Compute the seed $\phi=\partial_\Pi L(\Pi^*)$ with `autograd.grad`.

4. Solve the two adjoint linear systems **matrix-free** using Picard/Neumann or GMRES, with operators:

* $v \mapsto (I - F_\Pi^\top) v = v - (\partial_\Pi F)^\top v$ via `vjpF(v)[0]`,
* $w \mapsto (I - G_E^\top) w = w - (\partial_E G)^\top w$ via `vjpG(w)[0]`.

5. Read out parameter gradients in one shot:

* From `vjpF(v)` take the $\tau,\delta$ components,
* From `vjpG(w)` take the $\tau,\delta,\lambda$ components and add appropriately.

---

## Implementation plan (PyTorch 2.x)

### File layout

Create `src/reconciliation/implicit_vjp.py`:

* `solve_adjoint_fixedpoint(vjp_op, rhs, ...)`  — matrix-free linear solver for $(I - J^\top)x=rhs$.
* `implicit_grad_L_vjp(F, G, L, Pi_star, E_star, tau, delta, lam, ...)` — returns $\nabla_\tau \ell,\nabla_\delta \ell,\nabla_\lambda \ell$.

### Core code

```python
import torch
from torch import func as tfunc

@torch.no_grad()
def solve_adjoint_fixedpoint(vjp_apply, rhs, max_iter=200, tol=1e-8, damping=1.0):
    """
    Solve (I - J^T) x = rhs with Picard iteration:
        x_{k+1} = rhs + damping * (J^T x_k)
    where vjp_apply(x) returns (J^T x).
    Converges if spectral radius < 1 (contraction).
    """
    x = torch.zeros_like(rhs)
    for _ in range(max_iter):
        x_next = rhs + damping * vjp_apply(x)
        if (x_next - x).norm() <= tol * (1 + x.norm()):
            return x_next
        x = x_next
    return x

def implicit_grad_L_vjp(F, G, L, Pi_star, E_star, tau, delta, lam,
                        tol=1e-8, max_iter=200, damping=1.0):
    """
    Compute gradients of scalar L(Pi_star) w.r.t. (tau, delta, lam)
    using adjoint implicit differentiation via VJP closures at the fixed points.
    Assumes Pi_star, E_star are already converged for (tau, delta, lam).
    """

    # 0) Seed adjoint: phi = dL/dPi at Pi*
    Pi_req = Pi_star.detach().requires_grad_(True)
    loss = L(Pi_req)
    (phi,) = torch.autograd.grad(loss, Pi_req)
    del Pi_req, loss

    # 1) Persistent VJP closures at the fixed points
    _, vjpF = tfunc.vjp(lambda Pi, E, t, d: F(Pi, E, t, d),
                        Pi_star, E_star, tau, delta)
    _, vjpG = tfunc.vjp(lambda E, t, d, l: G(E, t, d, l),
                        E_star, tau, delta, lam)

    # (I - F_Pi^T) v = phi  -> use vjpF for (∂F/∂Pi)^T
    v = solve_adjoint_fixedpoint(lambda x: vjpF(x)[0], phi,
                                 max_iter=max_iter, tol=tol, damping=damping)

    # q = F_E^T v; also get parameter pieces from F in one call
    gPi_v, q, g_tau_F, g_delta_F = vjpF(v)  # gPi_v unused; serves as internal check if needed

    # (I - G_E^T) w = q  -> use vjpG for (∂G/∂E)^T
    w = solve_adjoint_fixedpoint(lambda x: vjpG(x)[0], q,
                                 max_iter=max_iter, tol=tol, damping=damping)

    # Parameter pieces from G in one call
    _, g_tau_G, g_delta_G, g_lambda_G = vjpG(w)

    # Combine per the formulas
    g_tau   = g_tau_F   + g_tau_G
    g_delta = g_delta_F + g_delta_G
    g_lam   = g_lambda_G

    return g_tau, g_delta, g_lam, v, w
```

> **Memory/speed knobs:** Using persistent `vjp` closures reuses saved activations (fast). If memory is tight, rebuild the closures periodically or switch to recomputing forwards; if convergence is slow, reduce `damping` or implement GMRES with matrix-free operator $x\mapsto x - \text{vjp}(x)$.

### Integration

* Treat `log_E_step` as $G$ and `Pi_update_ccp_log` as $F$.
* Ensure inputs share device/dtype; `Pi_star` and `E_star` should be **detached** (no need for tapes).
* Add a convenience wrapper:

```python
def implicit_param_grads(Pi_star, E_star, tau, delta, lam, L):
    return implicit_grad_L_vjp(
        F=Pi_update_ccp_log, G=log_E_step, L=L,
        Pi_star=Pi_star, E_star=E_star, tau=tau, delta=delta, lam=lam
    )
```

---

## Testing & validation

1. **Finite differences (small dims):**

   * Converge $(E^*,\Pi^*)$ at $(\tau,\delta,\lambda)$.
   * Compute grads $g_\tau,g_\delta,g_\lambda$ via `implicit_grad_L_vjp`.
   * For a small $\epsilon$, perturb one param block, re-solve fixed points, evaluate $L(\Pi^*_{\text{new}})$, and check

     $$
     \frac{L(\Pi^*_{\text{new}})-L(\Pi^*)}{\epsilon}
     \approx \langle g_{\text{block}},\; \Delta\theta/\epsilon \rangle.
     $$

   Use `float64` for this test.

2. **Cross-check (debug):**

   * On a tiny instance, unroll a few iterations with `create_graph=True` and backprop; compare to implicit grads (they should match the fixed-point differential as the unroll depth grows).

---

## Notes / edge cases

* If $L=L(\Pi^*,E^*)$, compute $\phi_E=\partial_E L$ and replace $q\leftarrow F_E^\top v + \phi_E$.
* If $F$ actually depends on $\lambda$, add the $\lambda$-component from `vjpF(v)` into $\nabla_\lambda \ell$.
* For near-marginal contractions, prefer **GMRES** with matrix-free operator $x\mapsto x - \text{vjp}(x)$ and a modest restart (e.g. 20).

---
