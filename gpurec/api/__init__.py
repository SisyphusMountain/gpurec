"""Notebook-friendly API for gpurec.

Public exports:
    GeneReconModel — torch.nn.Module wrapping the gradient pipeline.
    ReconStaticState — opaque container holding the precomputed wave layout
        and other static state shared across forward calls.
"""
from .autograd import ReconStaticState, _GeneReconFunction
from .model import GeneReconModel
from .uniform_chunked import UniformChunkedReconModel

__all__ = [
    "GeneReconModel",
    "UniformChunkedReconModel",
    "ReconStaticState",
]
