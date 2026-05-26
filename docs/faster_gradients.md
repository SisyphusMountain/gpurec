Let the fixed point be

[
x^\star(\theta)=f(x^\star(\theta),\theta),
]

with (f) contractive in (x). Let the objective be

[
L(\theta)=\ell(x^\star(\theta),\theta).
]

The implicit-gradient calculation usually gives

[
\nabla_\theta L
=

\partial_\theta \ell
+
\lambda^\top \partial_\theta f,
]

where the adjoint (\lambda) solves

[
(I-J_x f^\top)\lambda = \nabla_x \ell.
]

Equivalently,

[
\lambda
=

(I-J_x f^\top)^{-1}\nabla_x \ell.
]

If (\rho(J_x f)<1), then

[
(I-J_x f^\top)^{-1}
=

\sum_{k=0}^{\infty}(J_x f^\top)^k,
]

so

[
\lambda
=

\sum_{k=0}^{\infty}(J_x f^\top)^k \nabla_x \ell.
]

This is the Neumann series you are computing by repeated VJPs.

(I-A)^{-1}b=\sum_{k=0}^{\infty}A^k b,\qquad \rho(A)<1

Here (A=J_x f^\top) and (b=\nabla_x\ell).

The basic iteration is

[
\lambda_{k+1}=b+A\lambda_k,
]

or, with (\lambda_0=0),

[
\lambda_K=\sum_{k=0}^{K-1}A^k b.
]

Its error is

[
\lambda-\lambda_K
=

# \sum_{k=K}^{\infty}A^k b

A^K(I-A)^{-1}b.
]

Thus convergence is controlled by (\rho(A)=\rho(J_x f)). If the contraction factor is close to (1), plain Neumann iteration is slow.

The main acceleration options are:

## 1. Solve the linear system directly with Krylov methods

Instead of summing the Neumann series, solve

[
(I-A)\lambda=b
]

using a matrix-free linear solver. You only need products of the form

[
A u = J_x f^\top u,
]

which are exactly VJPs.

Good choices:

* **GMRES** for general nonsymmetric (I-A).
* **BiCGSTAB** if memory is an issue.
* **CG** only if (I-A) is symmetric positive definite, which is uncommon for neural fixed-point maps.

This is often the best practical improvement. The Neumann iteration is just stationary Richardson iteration; GMRES uses the whole Krylov space

[
\mathcal K_K(A,b)
=

\operatorname{span}{b,Ab,A^2b,\dots,A^{K-1}b}
]

and chooses the best approximation in that subspace.

So instead of

[
\lambda_K=\sum_{k=0}^{K-1}A^k b,
]

use

[
\lambda_K \in \mathcal K_K(A,b)
]

chosen to minimize the residual

[
r_K=b-(I-A)\lambda_K.
]

This usually beats the raw Neumann partial sum by a large margin.

## 2. Use Anderson acceleration on the adjoint iteration

Apply Anderson acceleration to

[
\lambda_{k+1}=T(\lambda_k)=b+A\lambda_k.
]

Define residuals

[
r_k=T(\lambda_k)-\lambda_k
=

b-(I-A)\lambda_k.
]

Anderson acceleration forms a linear combination of previous iterates to reduce the residual. In practice, this can be much faster than plain Neumann while still requiring only VJPs.

For a linear problem, Anderson acceleration is closely related to GMRES. So if you already have a matrix-free GMRES implementation, use GMRES. If your codebase already has Anderson acceleration for fixed-point iterations, it is a natural drop-in.

## 3. Precondition the Neumann system

You are solving

[
(I-A)\lambda=b.
]

If you can find an approximate inverse (P\approx (I-A)^{-1}), solve the preconditioned system

[
P(I-A)\lambda = Pb.
]

Equivalently, use an iteration whose effective matrix has a smaller spectral radius.

A simple form is damped Richardson:

[
\lambda_{k+1}
=

\lambda_k+\alpha\bigl(b-(I-A)\lambda_k\bigr).
]

Expanding,

[
\lambda_{k+1}
=

(1-\alpha)\lambda_k+\alpha b+\alpha A\lambda_k.
]

The error satisfies

[
e_{k+1}
=

\bigl(I-\alpha(I-A)\bigr)e_k.
]

So convergence depends on

[
\rho\bigl(I-\alpha(I-A)\bigr).
]

For scalar eigenvalues (a_i) of (A), the transformed eigenvalues are

[
1-\alpha(1-a_i).
]

Choosing (\alpha) well can improve convergence, but if (A) is nonnormal or has complex eigenvalues, this is delicate. GMRES is usually safer.

## 4. Reduce the contraction factor in the primal fixed-point map

Since the adjoint convergence rate depends on (J_x f), the best structural improvement is to make the fixed-point map more contractive.

If the primal equation is

[
x=f(x,\theta),
]

and (|J_x f|\leq q<1), then the Neumann tail roughly scales like

[
|\lambda-\lambda_K|
\lesssim
\frac{q^K}{1-q}|b|.
]

So if (q=0.9), convergence is slow; if (q=0.5), it is fast.

Ways to reduce (q):

* spectral normalization of the fixed-point map,
* smaller step sizes if (f) comes from an optimization update,
* stronger damping in the primal iteration,
* regularization of the Jacobian norm,
* choosing a better equivalent fixed-point formulation.

For example, if

[
x = g(x,\theta)
]

is too weakly contractive, one may use a damped map

[
f_\beta(x,\theta)
=

(1-\beta)x+\beta g(x,\theta).
]

Then

[
J_x f_\beta
=

(1-\beta)I+\beta J_x g.
]

This may improve primal stability, but it does not automatically improve adjoint convergence for all spectra. It helps when it moves the eigenvalues of (J_x f_\beta) closer to zero.

## 5. Use a better truncation criterion

Do not truncate based only on iteration count. Track the residual

[
r_K=b-(I-A)\lambda_K.
]

The true error obeys

[
\lambda-\lambda_K
=

(I-A)^{-1}r_K.
]

Therefore

[
|\lambda-\lambda_K|
\leq
|(I-A)^{-1}||r_K|.
]

If (|A|\leq q<1), then

[
|(I-A)^{-1}|
\leq
\frac{1}{1-q},
]

so

[
|\lambda-\lambda_K|
\leq
\frac{|r_K|}{1-q}.
]

This gives a principled stopping rule.

## 6. Use low-order extrapolation when you want something simpler than GMRES

If you want to keep the Neumann implementation but improve it cheaply, use extrapolation methods such as:

* Aitken acceleration for nearly scalar convergence,
* minimal polynomial extrapolation,
* reduced-rank extrapolation,
* Anderson acceleration with small memory.

For vector-valued adjoints, Anderson with memory (m=5) to (20) is often a strong default.

## Practical recommendation

The cleanest approach is:

[
\boxed{\text{Replace the Neumann sum by a matrix-free GMRES solve of }(I-J_xf^\top)\lambda=\nabla_x\ell.}
]

You only need a function computing

[
u \mapsto u - J_x f^\top u.
]

That is one VJP plus a subtraction. Then compute

[
\nabla_\theta L
=

\partial_\theta \ell
+
\lambda^\top \partial_\theta f.
]

A minimal pseudocode structure is:

```python
def matvec(u):
    # u - J_x f(x*, theta)^T u
    _, vjp_fun = vjp(lambda x: f(x, theta), x_star)
    A_T_u = vjp_fun(u)[0]
    return u - A_T_u

b = grad_x_ell(x_star, theta)

lambda_ = gmres(matvec, b, tol=tol, maxiter=maxiter)

grad_theta = grad_theta_ell(x_star, theta) + vjp_theta_f(lambda_)
```

Use Neumann only when (\rho(J_x f)) is clearly well below (1). If the contraction is weak, Krylov or Anderson acceleration is usually the right fix.
