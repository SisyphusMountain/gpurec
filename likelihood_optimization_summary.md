# Likelihood Optimization Summary

## Key Mathematical Insights

1. **Fixed-Point Structure**: 
   - E and Π satisfy fixed-point equations: (E,Π) = F((E,Π), θ)
   - E doesn't depend on Π, creating a hierarchical structure
   - This simplifies gradient computation

2. **Sparse Likelihood Dependence**:
   - L only depends on Π[root,:] and E (not the full Π matrix)
   - This sparsity makes adjoint method very efficient

3. **Gradient Computation**:
   - Use implicit differentiation of fixed-point equations
   - Adjoint method avoids computing full Jacobians
   - Solve linear systems: (I - ∂F/∂x)ᵀ v = ∂L/∂x

## PyTorch Implementation Strategy

### Custom Autograd Function
```python
class FixedPointCCPLogFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_params, ...):
        # 1. Compute E* via fixed-point iteration
        # 2. Compute Π* via fixed-point iteration  
        # 3. Compute likelihood L
        # 4. Save intermediates for backward
        return L
    
    @staticmethod
    def backward(ctx, grad_output):
        # 1. Compute ∂L/∂Π (sparse - only root row)
        # 2. Solve adjoint equation for v_Π
        # 3. Compute v_E from v_Π
        # 4. Compute ∂L/∂θ using adjoint formula
        return grad_theta * grad_output
```

### Handling -∞ Values
- Track finite masks throughout computation
- Use masked operations in adjoint solver
- Ensure gradients are zero for -∞ positions

### Optimization Algorithms

1. **Adam with Softplus**:
   - θ = softplus(log_θ) ensures positivity
   - Handles different parameter scales well
   - Good for initial optimization

2. **L-BFGS-B**:
   - Quasi-Newton with box constraints
   - Faster convergence near optimum
   - Requires exact gradients (which we provide)

3. **Newton's Method**:
   - Use finite differences of gradients for Hessian
   - Or implement second-order adjoint method
   - Quadratic convergence near optimum

## Implementation Checklist

- [ ] Implement FixedPointCCPLogFunction with proper masking
- [ ] Implement efficient adjoint solver (conjugate gradient)
- [ ] Add parameter transformations (softplus)
- [ ] Create optimizer wrapper class
- [ ] Add convergence diagnostics
- [ ] Implement checkpointing for debugging
- [ ] Test gradient accuracy with finite differences
- [ ] Benchmark against current implementation

## Key Advantages

1. **Exact Gradients**: No finite difference approximation errors
2. **Memory Efficient**: Adjoint method scales linearly with parameters
3. **GPU Compatible**: All operations use PyTorch tensors
4. **Handles Sparsity**: Exploits structure of the problem
5. **Robust**: Proper handling of log-space and -∞ values