You want a **batched L-BFGS**, not multiple PyTorch `LBFGS` objects.

The key issue in the implementation you pasted is that it flattens **all parameters into one vector**, then uses global dot products like

```python
ys = y.dot(s)
gtd = flat_grad.dot(d)
```

and a single scalar `loss`, scalar `t`, scalar Wolfe test, scalar history, and scalar stopping condition. That couples all (i)’s together. The implementation also explicitly has only one parameter group and stores one global state for the first parameter. 

You should instead reinterpret the first dimension of `theta` as the independent optimization index:

```python
theta.shape == [B, ...]
```

and make every L-BFGS scalar quantity become shape `[B]`.

## Core idea

Let

```python
f_vec = closure(theta)      # shape [B]
grad = d f_vec.sum() / d theta
```

This is valid if

[
f_i = f_i(x_i,\theta_i)
]

with no cross-batch dependence. Then

[
\nabla_{\theta_i}\sum_j f_j(x_j,\theta_j)
=========================================

\nabla_{\theta_i}f_i(x_i,\theta_i),
]

because for (j\neq i),

[
\nabla_{\theta_i}f_j(x_j,\theta_j)=0.
]

So you can evaluate all (f_i) at once, call `.sum().backward()` or `torch.autograd.grad(f_vec.sum(), theta)`, and still get the correct per-(i) gradients.

## Replace global flattening with batched flattening

Instead of

```python
flat_grad.shape == [total_num_params]
```

use

```python
flat_grad.shape == [B, P]
```

where

```python
B = theta.shape[0]
P = theta[0].numel()
```

Helpers:

```python
def flatten_batch(z):
    return z.reshape(z.shape[0], -1)

def unflatten_batch(z_flat, like):
    return z_flat.reshape_as(like)

def bdot(a, b):
    # rowwise dot product
    return (a * b).sum(dim=1)
```

So every place in the original implementation that has

```python
x.dot(y)
```

should become

```python
bdot(x, y)
```

if `x` and `y` are batched.

## Minimal batched L-BFGS skeleton

This keeps `theta` as one batched `nn.Parameter`.

```python
import torch


class BatchedLBFGS:
    def __init__(
        self,
        theta,
        lr=1.0,
        history_size=10,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    ):
        """
        theta: torch.nn.Parameter with shape [B, ...]
        """
        self.theta = theta
        self.lr = lr
        self.history_size = history_size
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change

        self.state = {
            "n_iter": 0,
            "d": None,                 # [B, P]
            "t": None,                 # [B]
            "prev_grad": None,         # [B, P]
            "prev_loss": None,         # [B]
            "old_dirs": [],            # list of [B, P]
            "old_stps": [],            # list of [B, P]
            "ro": [],                  # list of [B]
            "H_diag": None,            # [B]
        }

    def _value_and_grad(self, closure, theta_value):
        """
        closure(theta_value) must return shape [B].
        """
        with torch.enable_grad():
            z = theta_value.detach().clone().requires_grad_(True)
            f = closure(z)                       # [B]
            g, = torch.autograd.grad(f.sum(), z) # [B, ...]
        return f.detach(), g.detach().reshape(g.shape[0], -1)

    @torch.no_grad()
    def step(self, closure, max_iter=1):
        """
        closure(theta) -> f_vec of shape [B]
        """

        B = self.theta.shape[0]
        theta_shape = self.theta.shape
        theta_flat = self.theta.detach().reshape(B, -1)

        loss, flat_grad = self._value_and_grad(closure, self.theta)

        grad_norm = flat_grad.abs().amax(dim=1)  # [B]
        active = grad_norm > self.tolerance_grad

        if not active.any():
            return loss

        state = self.state

        for _ in range(max_iter):
            state["n_iter"] += 1

            d_prev = state["d"]
            t_prev = state["t"]
            prev_grad = state["prev_grad"]
            prev_loss = state["prev_loss"]

            old_dirs = state["old_dirs"]
            old_stps = state["old_stps"]
            ro = state["ro"]

            # ------------------------------------------------------------
            # L-BFGS two-loop recursion, independently per batch row
            # ------------------------------------------------------------
            if prev_grad is None:
                d = -flat_grad
                H_diag = torch.ones(B, device=theta_flat.device, dtype=theta_flat.dtype)
            else:
                y = flat_grad - prev_grad             # [B, P]
                s = d_prev * t_prev[:, None]          # [B, P]

                ys = bdot(y, s)                       # [B]
                yy = bdot(y, y).clamp_min(1e-30)      # [B]

                valid = ys > 1e-10

                # Store masked history. Invalid rows get harmless values.
                y_store = torch.where(valid[:, None], y, torch.zeros_like(y))
                s_store = torch.where(valid[:, None], s, torch.zeros_like(s))
                ro_store = torch.where(valid, 1.0 / ys.clamp_min(1e-30), torch.zeros_like(ys))

                if len(old_dirs) == self.history_size:
                    old_dirs.pop(0)
                    old_stps.pop(0)
                    ro.pop(0)

                old_dirs.append(y_store)
                old_stps.append(s_store)
                ro.append(ro_store)

                H_diag = torch.where(
                    valid,
                    ys / yy,
                    state["H_diag"] if state["H_diag"] is not None else torch.ones_like(ys),
                )

                q = -flat_grad
                alphas = []

                for y_k, s_k, ro_k in zip(reversed(old_dirs), reversed(old_stps), reversed(ro)):
                    a = bdot(s_k, q) * ro_k           # [B]
                    q = q - a[:, None] * y_k
                    alphas.append(a)

                r = H_diag[:, None] * q

                for y_k, s_k, ro_k, a in zip(old_dirs, old_stps, ro, reversed(alphas)):
                    beta = bdot(y_k, r) * ro_k        # [B]
                    r = r + (a - beta)[:, None] * s_k

                d = r

            # Save previous quantities before moving
            state["prev_grad"] = flat_grad.clone()
            state["prev_loss"] = loss.clone()
            state["H_diag"] = H_diag

            # ------------------------------------------------------------
            # Per-row step size
            # ------------------------------------------------------------
            if state["t"] is None:
                t = torch.minimum(
                    torch.ones(B, device=theta_flat.device, dtype=theta_flat.dtype),
                    1.0 / flat_grad.abs().sum(dim=1).clamp_min(1e-30),
                ) * self.lr
            else:
                t = torch.full((B,), self.lr, device=theta_flat.device, dtype=theta_flat.dtype)

            gtd = bdot(flat_grad, d)                  # [B]
            descent = gtd < -self.tolerance_change

            active = active & descent

            if not active.any():
                break

            # ------------------------------------------------------------
            # Simple independent backtracking Armijo line search
            # ------------------------------------------------------------
            # This is much easier to batch than full strong Wolfe.
            c1 = 1e-4
            shrink = 0.5
            max_ls = 20

            theta0_flat = theta_flat.clone()
            loss0 = loss.clone()

            accepted = torch.zeros(B, device=theta_flat.device, dtype=torch.bool)

            for _ls in range(max_ls):
                trial_flat = theta0_flat + t[:, None] * d
                trial = trial_flat.reshape(theta_shape)

                loss_new, grad_new = self._value_and_grad(closure, trial)

                armijo = loss_new <= loss0 + c1 * t * gtd
                ok = active & armijo

                # Accept rows independently.
                theta_flat = torch.where(ok[:, None], trial_flat, theta_flat)
                loss = torch.where(ok, loss_new, loss)
                flat_grad = torch.where(ok[:, None], grad_new, flat_grad)

                accepted |= ok

                still_searching = active & ~accepted
                if not still_searching.any():
                    break

                t = torch.where(still_searching, shrink * t, t)

            # For rows that never accepted, leave theta unchanged.
            state["d"] = d
            state["t"] = t

            step_norm = (t[:, None] * d).abs().amax(dim=1)
            active = active & (step_norm > self.tolerance_change)

            if prev_loss is not None:
                loss_change = (loss - prev_loss).abs()
                active = active & (loss_change > self.tolerance_change)

            if not active.any():
                break

        self.theta.copy_(theta_flat.reshape(theta_shape))
        return loss


def bdot(a, b):
    return (a * b).sum(dim=1)
```

Usage:

```python
theta = torch.nn.Parameter(theta0.clone())  # shape [B, ...]

def closure(theta_value):
    # return shape [B], not a scalar
    return f_batched(x_batch, theta_value)

opt = BatchedLBFGS(theta, lr=1.0, history_size=20)

for _ in range(num_steps):
    losses = opt.step(closure, max_iter=5)
```

## What this accomplishes

Each (i) now has its own:

```python
loss[i]
grad[i]
d[i]
t[i]
H_diag[i]
old_dirs[:, i]
old_stps[:, i]
ro[:, i]
stopping_condition[i]
line_search_acceptance[i]
```

but all expensive evaluations still happen as:

```python
f_batched(x_batch, theta_trial)
```

not as a Python loop over `i`.

## Why I used Armijo instead of strong Wolfe

The pasted implementation’s `_strong_wolfe` is scalar throughout: scalar `t`, scalar `f`, scalar `gtd`, scalar bracket endpoints, scalar interpolation, and scalar accept/reject logic. 

You *can* batch strong Wolfe, but it is noticeably more annoying because each row has its own bracket, zoom phase, interpolation, and termination time. Armijo backtracking is far easier to vectorize and usually works well for this use case.

If you really want batched strong Wolfe, the same principle applies:

```python
t.shape == [B]
f.shape == [B]
gtd.shape == [B]
bracket_low.shape == [B]
bracket_high.shape == [B]
done.shape == [B]
```

and every scalar branch becomes a boolean mask.

## Most important assumption

This batched trick is correct only if `f_batched` does not couple the batch elements. For example, this is okay:

```python
f_vec[i] = f_i(x[i], theta[i])
```

This is not okay:

```python
f_vec = normalize_across_batch(raw_losses)
```

or anything where `f_i` depends on `theta_j` for (j\neq i). In that case,

[
\nabla_{\theta_i}\sum_j f_j
]

is no longer equal to

[
\nabla_{\theta_i}f_i.
]

So the efficient solution is: **single batched `theta`, vector-valued closure, rowwise L-BFGS state, rowwise dot products, rowwise line search.**
