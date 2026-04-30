"""
Debugging utilities for phylogenetic reconciliation.

This module provides functions for debugging tensor operations,
tracking numerical issues, and logging computational progress.
"""

import torch
import logging
from typing import Optional, Dict, Any, List
import sys


def setup_logger(name: str = 'reconciliation', 
                 level: int = logging.DEBUG,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger for debugging reconciliation computations.
    
    Args:
        name: Logger name
        level: Logging level (default: DEBUG)
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Set format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


def tensor_stats(tensor: Optional[torch.Tensor], 
                name: str = "tensor") -> str:
    """
    Get comprehensive statistics about a tensor.
    
    Includes information about shape, data type, device, and numerical properties
    like NaN/Inf counts, min/max values, and sparsity.
    
    Args:
        tensor: Tensor to analyze (can be None)
        name: Name for display
        
    Returns:
        Formatted string with tensor statistics
    """
    if tensor is None:
        return f"{name}: None"
    
    # Basic info
    shape = list(tensor.shape)
    total_elements = tensor.numel()
    device = tensor.device
    dtype = tensor.dtype
    
    # Check for NaN and Inf
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    finite_mask = torch.isfinite(tensor)
    
    nan_count = nan_mask.sum().item()
    inf_count = inf_mask.sum().item()
    finite_count = finite_mask.sum().item()
    
    # Min/max of finite values
    if finite_count > 0:
        finite_values = tensor[finite_mask]
        min_val = finite_values.min().item()
        max_val = finite_values.max().item()
        mean_val = finite_values.mean().item()
        std_val = finite_values.std().item() if finite_count > 1 else 0.0
        stats_str = f"[{min_val:.6e}, {max_val:.6e}], μ={mean_val:.6e}, σ={std_val:.6e}"
    else:
        stats_str = "[no finite values]"
    
    # Special value counts
    neg_inf_count = (tensor == float('-inf')).sum().item()
    pos_inf_count = (tensor == float('inf')).sum().item()
    zero_count = (tensor == 0.0).sum().item()
    
    # Sparsity
    sparsity = zero_count / total_elements if total_elements > 0 else 0
    
    return (f"{name}: shape={shape}, device={device}, dtype={dtype}\n"
            f"  finite={finite_count}/{total_elements} ({100*finite_count/total_elements:.1f}%)\n"
            f"  range={stats_str}\n"
            f"  nan={nan_count}, +inf={pos_inf_count}, -inf={neg_inf_count}\n"
            f"  zeros={zero_count} (sparsity={100*sparsity:.1f}%)")


def check_tensor_health(tensor: torch.Tensor, 
                       name: str = "tensor",
                       raise_on_nan: bool = True,
                       raise_on_inf: bool = False) -> Dict[str, Any]:
    """
    Check tensor for numerical issues and optionally raise errors.
    
    Args:
        tensor: Tensor to check
        name: Name for error messages
        raise_on_nan: If True, raise error on NaN values
        raise_on_inf: If True, raise error on Inf values
        
    Returns:
        Dictionary with health check results
        
    Raises:
        ValueError: If NaN/Inf found and corresponding flag is True
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    has_neg_inf = (tensor == float('-inf')).any().item()
    has_pos_inf = (tensor == float('inf')).any().item()
    
    health = {
        'healthy': not (has_nan or (has_inf and raise_on_inf)),
        'has_nan': has_nan,
        'has_inf': has_inf,
        'has_neg_inf': has_neg_inf,
        'has_pos_inf': has_pos_inf
    }
    
    if has_nan and raise_on_nan:
        raise ValueError(f"NaN detected in {name}:\n{tensor_stats(tensor, name)}")
    
    if has_inf and raise_on_inf:
        raise ValueError(f"Inf detected in {name}:\n{tensor_stats(tensor, name)}")
    
    return health


def log_tensor(tensor: Optional[torch.Tensor], 
              name: str,
              logger: Optional[logging.Logger] = None,
              level: int = logging.DEBUG) -> None:
    """
    Log tensor statistics using a logger.
    
    Args:
        tensor: Tensor to log
        name: Name for display
        logger: Logger instance (uses root logger if None)
        level: Logging level
    """
    if logger is None:
        logger = logging.getLogger()
    
    stats = tensor_stats(tensor, name)
    logger.log(level, stats)


def compare_tensors(tensor1: torch.Tensor,
                   tensor2: torch.Tensor,
                   name1: str = "tensor1",
                   name2: str = "tensor2",
                   rtol: float = 1e-5,
                   atol: float = 1e-8) -> Dict[str, Any]:
    """
    Compare two tensors for numerical differences.
    
    Args:
        tensor1, tensor2: Tensors to compare
        name1, name2: Names for display
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dictionary with comparison results
    """
    if tensor1.shape != tensor2.shape:
        return {
            'equal': False,
            'reason': f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"
        }
    
    # Check exact equality
    exact_equal = torch.equal(tensor1, tensor2)
    
    # Check approximate equality
    close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    
    # Compute differences
    diff = torch.abs(tensor1 - tensor2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Find locations of maximum difference
    if max_diff > 0:
        max_idx = torch.argmax(diff.view(-1))
        max_idx_tuple = torch.unravel_index(max_idx, diff.shape)
        max_loc = tuple(idx.item() for idx in max_idx_tuple)
        val1_at_max = tensor1.view(-1)[max_idx].item()
        val2_at_max = tensor2.view(-1)[max_idx].item()
    else:
        max_loc = None
        val1_at_max = None
        val2_at_max = None
    
    return {
        'exact_equal': exact_equal,
        'allclose': close,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'max_diff_location': max_loc,
        f'{name1}_at_max': val1_at_max,
        f'{name2}_at_max': val2_at_max
    }


def track_gradient_flow(model_or_tensor: torch.Tensor,
                        name: str = "tensor") -> Dict[str, Any]:
    """
    Track gradient flow information for debugging backpropagation.
    
    Args:
        model_or_tensor: Tensor with gradients
        name: Name for display
        
    Returns:
        Dictionary with gradient statistics
    """
    if not model_or_tensor.requires_grad:
        return {'requires_grad': False, 'name': name}
    
    grad = model_or_tensor.grad
    
    if grad is None:
        return {
            'requires_grad': True,
            'has_grad': False,
            'name': name
        }
    
    grad_stats = tensor_stats(grad, f"{name}.grad")
    
    return {
        'requires_grad': True,
        'has_grad': True,
        'grad_stats': grad_stats,
        'name': name
    }


class DebugContext:
    """
    Context manager for debugging specific code sections.
    
    Usage:
        with DebugContext("likelihood_computation") as debug:
            debug.log_tensor(Pi, "Pi_matrix")
            # ... computation ...
            debug.check_health(result, "result")
    """
    
    def __init__(self, name: str, 
                 logger: Optional[logging.Logger] = None,
                 enabled: bool = True):
        """
        Initialize debug context.
        
        Args:
            name: Name of the debugging section
            logger: Logger to use
            enabled: Whether debugging is enabled
        """
        self.name = name
        self.logger = logger or logging.getLogger(name)
        self.enabled = enabled
        self.tensors: Dict[str, torch.Tensor] = {}
        
    def __enter__(self):
        if self.enabled:
            self.logger.debug(f"=== Entering {self.name} ===")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            if exc_type is not None:
                self.logger.error(f"Exception in {self.name}: {exc_val}")
            self.logger.debug(f"=== Exiting {self.name} ===")
    
    def log_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """Log tensor statistics."""
        if self.enabled:
            log_tensor(tensor, name, self.logger)
            self.tensors[name] = tensor
    
    def check_health(self, tensor: torch.Tensor, name: str) -> None:
        """Check tensor health."""
        if self.enabled:
            health = check_tensor_health(tensor, name, raise_on_nan=False)
            if not health['healthy']:
                self.logger.warning(f"Unhealthy tensor {name}: {health}")
    
    def compare(self, name1: str, name2: str) -> None:
        """Compare two stored tensors."""
        if self.enabled and name1 in self.tensors and name2 in self.tensors:
            result = compare_tensors(self.tensors[name1], self.tensors[name2], 
                                    name1, name2)
            self.logger.debug(f"Comparison {name1} vs {name2}: {result}")