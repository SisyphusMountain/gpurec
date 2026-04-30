"""Notebook-friendly API for gpurec.

Public exports:
    GeneReconModel — torch.nn.Module wrapping the gradient pipeline.
    ReconStaticState — opaque container holding the precomputed wave layout
        and other static state shared across forward calls.
"""
from .autograd import ReconStaticState, _GeneReconFunction
from .model import GeneReconModel
from .modes import _MODE_MAP, _mode_to_flags, _default_theta_init

__all__ = [
    "GeneReconModel",
    "ReconStaticState",
]
