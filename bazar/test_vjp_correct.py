#!/usr/bin/env python3
"""
Test the correct VJP API usage.
"""

import torch
from torch.autograd.functional import vjp

def test_vjp_correct():
    """Test the correct VJP API usage."""
    print("Testing VJP API correctly...")
    
    # For functions with vector outputs, we need to specify v at vjp call time
    def f(x):
        return x ** 2
    
    x = torch.tensor([1.0, 2.0])
    v = torch.tensor([1.0, 1.0])
    
    # Call vjp with the vector v
    output, grad = vjp(f, x, v)
    print(f"Input: {x}")
    print(f"Output: {output}")
    print(f"Gradient: {grad}")
    print(f"Expected (2x): {2 * x}")
    
    # For scalar outputs, v can be omitted
    def f_scalar(x):
        return (x ** 2).sum()
    
    output_scalar, grad_scalar = vjp(f_scalar, x)
    print(f"\nScalar output: {output_scalar}")
    print(f"Scalar gradient: {grad_scalar}")
    print(f"Expected (2x): {2 * x}")

if __name__ == "__main__":
    test_vjp_correct()