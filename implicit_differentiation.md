# Implicit Differentiation for Gradient Computation in CCP Log-Likelihood

## Problem Statement

We want to compute the gradient of the log-likelihood with respect to the parameters θ = (δ, τ, λ) for the CCP reconciliation model. The challenge is that the likelihood depends on Pi, which is computed through nested fixed-point iterations:

1. **E** is the fixed point of function g: E = g(E, θ)
2. **Pi** is the fixed point of function h: Pi = h(Pi, E, θ)
3. The log-likelihood is: L(θ) = log(∑ᵢ Pi[root, i])

## Mathematical Framework

### 1. Fixed-Point Equations

From the code, we have:
- **E-step**: E = g(E, θ) where
  ```
  g(E, θ) = p_S * E_s1 * E_s2 + p_D * E² + p_T * E * Ē + p_L
  ```
  where:
  - E_s1 = C₁ @ E (left child extinction)
  - E_s2 = C₂ @ E (right child extinction)
  - Ē = Recipients @ E (average extinction over transfer recipients)
  - p_S, p_D, p_T, p_L are functions of θ

- **Pi-step** (in log space): log(Pi) = h(log(Pi), E, θ)
  - This is more complex due to log-space operations with logsumexp

### 2. Event Probabilities as Functions of Parameters

Given θ = (δ, τ, λ):
```
p_D = δ/(1 + δ + τ + λ)
p_T = τ/(1 + δ + τ + λ)
p_L = λ/(1 + δ + τ + λ)
p_S = 1/(1 + δ + τ + λ)
```

### 3. Implicit Function Theorem

Since (E*, Pi*) are fixed points at convergence:
```
E* - g(E*, θ) = 0
Pi* - h(Pi*, E*, θ) = 0
```

Taking total derivatives with respect to θ:
```
dE*/dθ - ∂g/∂E|₍E*,θ₎ · dE*/dθ - ∂g/∂θ|₍E*,θ₎ = 0
dPi*/dθ - ∂h/∂Pi|₍Pi*,E*,θ₎ · dPi*/dθ - ∂h/∂E|₍Pi*,E*,θ₎ · dE*/dθ - ∂h/∂θ|₍Pi*,E*,θ₎ = 0
```

Solving for the derivatives:
```
dE*/dθ = (I - ∂g/∂E)⁻¹ · ∂g/∂θ
dPi*/dθ = (I - ∂h/∂Pi)⁻¹ · (∂h/∂E · dE*/dθ + ∂h/∂θ)
```

### 4. Gradient of Log-Likelihood

The gradient of the log-likelihood is:
```
dL/dθ = ∂L/∂Pi · dPi*/dθ
```

where:
```
∂L/∂Pi = 1/∑ᵢPi[root,i] · e_root^T
```
(e_root is a one-hot vector for the root clade)

## Efficient Computation via Adjoint Method

### Step 1: Forward Pass
1. Iterate E to convergence: E* = g(E*, θ)
2. Iterate Pi to convergence: Pi* = h(Pi*, E*, θ)
3. Compute L = log(∑ᵢ Pi*[root, i])

### Step 2: Backward Pass (Adjoint Method)

Define adjoint variables:
- λ_Pi = ∂L/∂Pi = 1/∑ᵢPi[root,i] · e_root^T
- λ_E = ∂L/∂E

#### 2.1 Solve for λ_Pi (Adjoint equation for Pi):
```
λ_Pi^T = λ_Pi^T · ∂h/∂Pi + ∂L/∂Pi
```
Rearranging:
```
λ_Pi^T · (I - ∂h/∂Pi) = ∂L/∂Pi
λ_Pi = (I - ∂h/∂Pi)^(-T) · (∂L/∂Pi)^T
```

#### 2.2 Solve for λ_E (Adjoint equation for E):
```
λ_E^T = λ_E^T · ∂g/∂E + λ_Pi^T · ∂h/∂E
```
Rearranging:
```
λ_E = (I - ∂g/∂E)^(-T) · (∂h/∂E)^T · λ_Pi
```

#### 2.3 Compute gradient:
```
dL/dθ = λ_Pi^T · ∂h/∂θ + λ_E^T · ∂g/∂θ
```

## Implementation Strategy for Log-Space

### 1. Computing ∂g/∂E (for E-step)

For E_new = p_S * E_s1 * E_s2 + p_D * E² + p_T * E * Ē + p_L:

```
∂E_new[i]/∂E[j] = p_S * (δᵢⱼ(C₁E)ᵢ(C₂E)ᵢ + Eᵢ(C₁)ᵢⱼ(C₂E)ᵢ + Eᵢ(C₁E)ᵢ(C₂)ᵢⱼ)
                 + p_D * 2 * δᵢⱼ * E[i]
                 + p_T * (δᵢⱼ * Ē[i] + E[i] * Recipients[i,j])
```

### 2. Computing ∂h/∂Pi (for Pi-step in log space)

This is more complex due to logsumexp operations. For numerical stability, we should:
1. Work with automatic differentiation for the forward pass
2. Use custom backward pass that handles -inf values properly

### 3. Computing ∂g/∂θ and ∂h/∂θ

These involve derivatives of event probabilities:
```
∂p_D/∂δ = (1 + τ + λ)/(1 + δ + τ + λ)²
∂p_D/∂τ = -δ/(1 + δ + τ + λ)²
∂p_D/∂λ = -δ/(1 + δ + τ + λ)²
```
(Similar for other probabilities)

## Practical Considerations

### 1. Numerical Stability
- Use log-space operations throughout to avoid underflow
- Handle -inf values carefully in Jacobian computations
- Use iterative solvers (e.g., conjugate gradient) for linear systems

### 2. Computational Efficiency
- The Jacobians ∂g/∂E and ∂h/∂Pi are sparse
- Can use power iteration or Arnoldi method instead of full matrix inversion
- Cache intermediate computations from forward pass

### 3. Verification
- Check gradients using finite differences
- Ensure adjoint equations are solved to sufficient accuracy
- Monitor condition numbers of (I - ∂g/∂E) and (I - ∂h/∂Pi)

## Alternative: Unrolling Fixed-Point Iterations

Instead of implicit differentiation, we could:
1. Unroll a fixed number of iterations
2. Use automatic differentiation through the unrolled computation
3. This is simpler but less accurate if not fully converged

Trade-offs:
- (+) Simpler implementation with PyTorch autograd
- (+) No linear system solves required
- (-) Requires storing intermediate values
- (-) May need many iterations for accurate gradients
- (-) Memory intensive for large problems

## Recommended Approach

For the CCP log-likelihood optimization:

1. **Use implicit differentiation** for accuracy and efficiency
2. **Implement custom autograd Function** that:
   - Forward: runs fixed-point iterations to convergence
   - Backward: solves adjoint equations
3. **Handle -inf values** by masking in Jacobian computations
4. **Use iterative solvers** for the linear systems
5. **Verify with finite differences** on small test cases

This approach will give exact gradients at convergence while being memory efficient and numerically stable.

## Single-Iteration Gradient Approximation

### The Proposed Approach

Instead of computing gradients through the converged fixed point, we consider:
1. Given current (E, Pi), perform one iteration: (E', Pi') = f(E, Pi, θ)
2. Compute likelihood L(Pi')
3. Use ∇_θ L(Pi') as the gradient estimate

The key question: **Does gradient descent using these approximate gradients converge to the true MLE?**

### Mathematical Analysis

#### 1. Gradient Decomposition

The true gradient at parameters θ is:
```
∇_θ L(θ) = ∂L/∂Pi* · dPi*/dθ
```
where Pi* = Pi*(θ) is the converged fixed point.

The single-iteration gradient is:
```
∇̃_θ L(θ) = ∂L/∂Pi' · ∂Pi'/∂θ
```
where Pi' = h(Pi, E, θ) and E' = g(E, θ).

The approximation error is:
```
∇̃_θ L(θ) - ∇_θ L(θ) = ∂L/∂Pi' · ∂Pi'/∂θ - ∂L/∂Pi* · dPi*/dθ
```

#### 2. Contraction Mapping Analysis

If the fixed-point map F(x, θ) = (g(E, θ), h(Pi, E, θ)) is a contraction with constant ρ < 1:
```
||F(x, θ) - F(x*, θ)|| ≤ ρ ||x - x*||
```

Then after k iterations from initial x₀:
```
||x_k - x*|| ≤ ρᵏ ||x₀ - x*||
```

This bounds the difference between Pi' and Pi*, which in turn bounds the gradient approximation error.

#### 3. Descent Property

For gradient descent to converge, we need the approximate gradient to satisfy a descent condition:
```
⟨∇̃_θ L(θ), ∇_θ L(θ)⟩ > 0
```

This ensures the approximate gradient points in a direction of improvement.

### Convergence Analysis

#### Theorem (Informal): Convergence with Single-Iteration Gradients

Under the following conditions:
1. **Contraction**: The fixed-point map F is a contraction with constant ρ < 1
2. **Smoothness**: The likelihood L and fixed-point map F are sufficiently smooth
3. **Initialization**: We start "close enough" to the fixed point

Then gradient descent with single-iteration gradients converges to a neighborhood of the true MLE, where the neighborhood size depends on:
- The contraction constant ρ
- The Lipschitz constants of L and F
- The step size used in gradient descent

#### Key Insights

1. **Near Fixed Point**: When (E, Pi) ≈ (E*, Pi*), the single-iteration gradient is very close to the true gradient, so convergence is assured.

2. **Contraction Helps**: The contraction property ensures that even if we're not at the fixed point, we're moving toward it, which tends to align the approximate gradient with the true gradient.

3. **Trade-off**: We trade exact convergence for computational efficiency. The method may converge to a point that's close to, but not exactly at, the true MLE.

### Connection to Existing Theory

#### 1. Truncated Backpropagation Through Time (TBPTT)

Our approach is analogous to TBPTT in recurrent neural networks:
- Full BPTT: Backpropagate through all time steps (all fixed-point iterations)
- Truncated BPTT: Backpropagate through k steps (k=1 in our case)

TBPTT is known to work well in practice despite the approximation, especially when:
- The dynamics are stable (contraction property)
- The truncation length captures essential dependencies

#### 2. Gradient Descent with Errors

There's established theory on gradient descent with approximate gradients. If the gradient error is bounded:
```
||∇̃_θ L(θ) - ∇_θ L(θ)|| ≤ ε
```

Then gradient descent converges to an ε-neighborhood of the optimum. In our case, ε depends on how far we are from the fixed point.

#### 3. Implicit Function Theorem Perspective

The single-iteration approach can be viewed as a first-order approximation to the implicit function theorem:
```
Pi*(θ + δθ) ≈ Pi*(θ) + ∂h/∂θ · δθ + O(||δθ||²)
```

When we use Pi' instead of Pi*, we're essentially using a different linearization point, which is valid if the fixed-point map is well-behaved.

### Practical Considerations

#### When Single-Iteration Works Well

1. **Near Convergence**: When optimization is close to the optimum, (E, Pi) are near their fixed points
2. **Strong Contraction**: When ρ is small, one iteration gets us close to the fixed point
3. **Smooth Landscape**: When the likelihood is not too sensitive to small changes in Pi

#### When It Might Fail

1. **Far from Fixed Point**: Early in optimization when (E, Pi) are far from convergence
2. **Weak Contraction**: When ρ ≈ 1, many iterations needed for convergence
3. **Sensitive Likelihood**: When small changes in Pi cause large changes in L

#### Hybrid Strategies

1. **Adaptive Iterations**: Use more iterations early in optimization, fewer near convergence
2. **Warm Start**: Maintain (E, Pi) across optimization steps instead of reinitializing
3. **Periodic Full Convergence**: Occasionally run to full convergence to correct accumulated errors

### Empirical Validation Strategy

To verify that single-iteration gradients lead to correct optimization:

1. **Compare Trajectories**: Plot optimization paths with exact vs. approximate gradients
2. **Final Likelihood**: Check if both methods converge to similar likelihood values
3. **Parameter Recovery**: On synthetic data, verify recovery of true parameters
4. **Gradient Alignment**: Measure ⟨∇̃_θ L, ∇_θ L⟩ throughout optimization

### Recommendation

The single-iteration gradient approach is **theoretically sound** under reasonable conditions and offers significant computational advantages:

1. **Memory Efficient**: No need to store intermediate iterations
2. **Simple Implementation**: Direct use of autograd without custom backward pass
3. **Often Sufficient**: In practice, especially with warm starts, often converges to good solutions

However, for **high-accuracy** requirements or **theoretical guarantees**, the implicit differentiation approach remains preferable.

### Implementation Sketch

```python
class SingleIterationCCPLog:
    def __init__(self, species_helpers, ccp_helpers):
        self.species_helpers = species_helpers
        self.ccp_helpers = ccp_helpers
        self.E = None  # Maintain across iterations
        self.log_Pi = None
    
    def forward(self, delta, tau, lambda_param):
        # Convert to probabilities
        p_S, p_D, p_T, p_L = compute_probabilities(delta, tau, lambda_param)
        
        # Single E update
        if self.E is None:
            self.E = torch.zeros(...)
        self.E = E_step(self.E, p_S, p_D, p_T, p_L, ...)
        
        # Single Pi update
        if self.log_Pi is None:
            self.log_Pi = initialize_log_Pi(...)
        self.log_Pi = Pi_update_ccp_log(self.log_Pi, self.E, p_S, p_D, p_T, ...)
        
        # Compute likelihood
        root_Pi = torch.exp(self.log_Pi[root_idx, :])
        return torch.log(root_Pi.sum())
```

This approach maintains (E, Pi) across optimization steps, effectively implementing a warm start strategy that improves gradient quality.