#!/usr/bin/env python3
"""
Test the correct VJP API usage.
"""

import torch
from torch.autograd.functional import vjp

def test_vjp_api():
    """Test the correct VJP API usage."""
    print("Testing VJP API...")
    
    def f(x):
        return x ** 2
    
    x = torch.tensor([1.0, 2.0])
    v = torch.tensor([1.0, 1.0])
    
    # Try the documented way
    output, vjp_fn = vjp(f, x)
    print(f"Output: {output}")
    print(f"VJP function type: {type(vjp_fn)}")
    
    # Call the vjp function
    result = vjp_fn(v)
    print(f"VJP result: {result}")
    print(f"VJP result type: {type(result)}")
    
    if isinstance(result, tuple):
        grad = result[0]
        print(f"Gradient: {grad}")
    else:
        print(f"Gradient (direct): {result}")

if __name__ == "__main__":
    test_vjp_api()