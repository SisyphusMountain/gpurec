"""Lean optimizer exports.

The high-level optimization path is the standard PyTorch optimizer interface on
``GeneReconModel.theta``.  The only custom optimizer retained here is the
row-wise batched L-BFGS implementation used for genewise polishing.
"""

from .batched_lbfgs import BatchedLBFGS
from .implicit_grad import implicit_grad_loglik_vjp_wave

__all__ = [
    "BatchedLBFGS",
    "implicit_grad_loglik_vjp_wave",
]
