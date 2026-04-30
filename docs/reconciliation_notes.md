# Reconciliation Algorithm Documentation

## Overview
This document describes the GPU-accelerated reconciliation algorithm for phylogenetic trees, implementing the ALE (Amalgamated Likelihood Estimation) algorithm with Conditional Clade Probabilities (CCPs) in log-space.

## Algorithm Flow

### 1. Fixed-Point Structure
The reconciliation involves two nested fixed-point computations:

#### E-step: Extinction Probabilities
- **Function**: `log_E_step` in `src/reconciliation/likelihood.py`
- **Fixed point**: E* = G(E*; τ, δ, λ)
- **Purpose**: Computes probability of gene lineage extinction on each species branch
- **Contraction**: Guaranteed by probabilistic interpretation (probabilities < 1)

#### Pi-update: Likelihood Matrix
- **Function**: `Pi_update_ccp_log` in `src/reconciliation/likelihood.py`  
- **Fixed point**: Π* = F(Π*, E*; τ, δ)
- **Purpose**: Computes likelihood Pi[c,s] of observing clade c on species branch s
- **Contraction**: Guaranteed by event probabilities summing to 1

### 2. Event Model
The algorithm models four types of evolutionary events:

- **Speciation (S)**: Gene follows species tree, probability p_S = 1/(1+δ+τ+λ)
- **Duplication (D)**: Gene duplicates on same branch, probability p_D = δ/(1+δ+τ+λ)
- **Transfer (T)**: Gene transfers to non-ancestral branch, probability p_T = τ/(1+δ+τ+λ)
- **Loss (L)**: Gene lineage goes extinct, probability p_L = λ/(1+δ+τ+λ)

### 3. Likelihood Computation
Final log-likelihood: L(Π*) = logsumexp(Π*[root_clade, :])

## Mathematical Formulation

### Fixed-Point Equations

#### E-step Update
```
E_new = logsumexp([
    p_S * E_s1 * E_s2,           # Speciation
    p_D * E^2,                   # Duplication
    p_T * E * Ebar,              # Transfer
    p_L                          # Loss
])
```

Where:
- E_s1, E_s2: Extinction at left/right species children
- Ebar: Average extinction over recipient branches

#### Pi-update
```
Pi_new[c,s] = logsumexp([
    # Duplication events
    Σ p_D * p(γ',γ''|γ) * Pi[γ',s] * Pi[γ'',s],  # Both survive
    2 * p_D * Pi[c,s] * E[s],                      # One extinct
    
    # Speciation events  
    Σ p_S * p(γ',γ''|γ) * (Pi[γ',s1]*Pi[γ'',s2] + Pi[γ',s2]*Pi[γ'',s1]),
    p_S * (Pi[c,s1]*E[s2] + Pi[c,s2]*E[s1]),      # One extinct
    
    # Transfer events
    Σ p_T * p(γ',γ''|γ) * (Pi[γ',s]*Pibar[γ'',s] + Pi[γ'',s]*Pibar[γ',s]),
    p_T * (Pi[c,s]*Ebar[s] + Pibar[c,s]*E[s])     # One extinct
])
```

### Implicit Differentiation

For computing gradients ∇_{τ,δ,λ} L(Π*) without unrolling:

#### Adjoint System
1. Solve (I - F_Π^T)v = ∂L/∂Π
2. Compute q = F_E^T v  
3. Solve (I - G_E^T)w = q
4. Gradients:
   - ∇_τ L = F_τ^T v + G_τ^T w
   - ∇_δ L = F_δ^T v + G_δ^T w  
   - ∇_λ L = G_λ^T w

Where F_x, G_x denote Jacobians with respect to x.

## Implementation Details

### Numerical Stability

#### Log-Space Computations
- All probabilities maintained in log-space
- Use logsumexp for numerically stable addition
- Custom ScatterLogSumExp autograd for scatter operations

#### Initialization
- E initialized to -log(2) (probability 0.5)
- Pi initialized to -log(2) for all entries
- Leaf clades set based on species mapping

#### Handling Edge Cases
- Zero split probabilities → -inf in log space
- Empty clade-species mappings handled gracefully
- Numerical bounds to prevent underflow

### Custom Autograd Functions

#### ScatterLogSumExp
- Forward: Numerically stable scatter with log-sum-exp trick
- Backward: Softmax-style gradient weights
- Supports functorch transforms (jacrev, vmap)

### Matrix Representations

#### Species Tree Helpers
- `s_C1, s_C2`: Children indices for species nodes
- `Recipients_mat`: Transfer recipient matrix
- `internal_mask`: Boolean mask for internal nodes

#### CCP Helpers
- `split_parents, split_lefts, split_rights`: Clade split structure
- `log_split_probs`: Log probabilities of splits
- `ccp_leaves_mask`: Boolean mask for leaf clades

## Implicit VJP Implementation

### Core Functions

#### solve_adjoint_fixedpoint
Solves (I - J^T)x = rhs using:
- Picard iteration: x_{k+1} = rhs + J^T x_k
- Optional: GMRES for near-marginal contractions
- Convergence criterion: ||x_{k+1} - x_k|| < tol

#### implicit_grad_L_vjp
Main gradient computation:
1. Create VJP closures at fixed points
2. Solve adjoint systems matrix-free
3. Combine gradient components
4. Return gradients for τ, δ, λ

### Memory Efficiency
- O(1) memory vs O(iterations) for unrolling
- Reuses forward computation via VJP closures
- No need to store intermediate iterations

### Numerical Considerations
- Damping factor for stability
- Tolerance settings for convergence
- Handling of -inf and NaN values