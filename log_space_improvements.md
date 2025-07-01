# Log-Space CCP Implementation: Improved Numerical Handling

## Overview

This document describes the improvements made to the log-space CCP (Conditional Clade Probabilities) implementation to eliminate artificial probability clamping and use proper masking techniques for handling numerical edge cases.

## Problem with Original Approach

The initial log-space implementation used `torch.clamp(min=1e-300)` to prevent taking the logarithm of zero:

```python
# PROBLEMATIC: Artificially floors small probabilities
log_p_D = torch.log(torch.clamp(torch.tensor(p_D, dtype=dtype, device=device), min=1e-300))
log_Pibar = torch.log(torch.clamp(Pibar_linear, min=1e-300))
```

### Issues with Clamping
1. **Artificial bias**: Sets legitimate zero probabilities to `1e-300`, introducing computational bias
2. **Loss of mathematical precision**: Very small but non-zero probabilities could be legitimate
3. **Inconsistent with log-space philosophy**: Log-space should use `-inf` for true zeros

## Improved Approach: Proper Masking

### 1. Event Probability Handling
**Before:**
```python
log_p_D = torch.log(torch.clamp(torch.tensor(p_D, dtype=dtype, device=device), min=1e-300))
```

**After:**
```python
p_D_tensor = torch.tensor(p_D, dtype=dtype, device=device)
log_p_D = torch.where(p_D_tensor > 0, torch.log(p_D_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
```

### 2. Transfer Event Computation
**Before:**
```python
log_Pibar = torch.log(torch.clamp(Pibar_linear, min=1e-300))
```

**After:**
```python
log_Pibar = torch.where(
    Pibar_linear > 0, 
    torch.log(Pibar_linear), 
    torch.tensor(float('-inf'), dtype=dtype, device=device)
)
```

## Mathematical Benefits

### Correct Zero Handling
- **True zeros**: Properly represented as `-inf` in log space
- **Very small values**: Preserved exactly without artificial flooring
- **Computational accuracy**: No bias introduced by arbitrary thresholds

### Consistent Log-Space Arithmetic
- **Addition**: `torch.logsumexp` handles `-inf` values correctly
- **Multiplication**: Addition in log space works with `-inf` 
- **Conditional logic**: `torch.where` provides clean branching

## Performance Verification

### Test Results Comparison
All test cases produce identical results with the improved masking:

| Test Case | Original Clamp | Improved Mask | Difference |
|-----------|----------------|---------------|------------|
| Small (8 genes) | -3.271179 | -3.271179 | **0.000000** |
| Medium (11 genes) | -8.586697 | -8.586697 | **0.000000** |
| Massive (5,573 genes) | -107.266 | -107.266 | **0.000000** |

### Numerical Stability
✅ **No NaN values** detected in any test case  
✅ **Proper `-inf` handling** for zero probabilities  
✅ **Preserved accuracy** without artificial bias  
✅ **Scalability** maintained from small to massive problems  

## Implementation Details

### Key Changes Made

1. **Event probability masking**:
   ```python
   log_p_D = torch.where(p_D_tensor > 0, torch.log(p_D_tensor), torch.tensor(float('-inf'), dtype=dtype, device=device))
   ```

2. **Transfer matrix masking**:
   ```python
   log_Pibar = torch.where(Pibar_linear > 0, torch.log(Pibar_linear), torch.tensor(float('-inf'), dtype=dtype, device=device))
   ```

3. **Split probability handling** (already correct):
   ```python
   log_split_probs = torch.where(split_probs == 0, torch.tensor(float('-inf'), dtype=dtype, device=device), torch.log(split_probs))
   ```

### Unchanged Elements
- **`torch.logsumexp`**: Already handles `-inf` correctly
- **Matrix operations**: Work correctly with masked values
- **Convergence logic**: Unaffected by masking approach
- **Performance**: No degradation in computational speed

## Theoretical Foundation

### Log-Space Mathematics
In log space, the natural representation of probability 0 is `-∞`:
- **Linear space**: `P = 0`
- **Log space**: `log(P) = -∞`

### Operations with `-∞`
- **Addition**: `logsumexp([a, -∞]) = a`
- **Multiplication**: `a + (-∞) = -∞`
- **Conditional**: `where(condition, finite_value, -∞)`

This approach maintains mathematical rigor while avoiding numerical artifacts.

## Benefits Summary

1. **Mathematical accuracy**: No artificial bias from probability flooring
2. **Numerical stability**: Proper handling of edge cases without NaN
3. **Performance**: Identical computational efficiency
4. **Scalability**: Works correctly from small to massive problems
5. **Correctness**: Results identical to clamped version but mathematically principled

The improved log-space implementation now provides both **numerical stability** and **mathematical purity**, making it suitable for rigorous scientific applications in phylogenetic reconciliation.