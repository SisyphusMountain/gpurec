"""Notebook-friendly API for gpurec.

Public exports:
    GeneReconModel — torch.nn.Module wrapping the gradient pipeline.
"""
from .model import GeneReconModel
from .uniform_chunked import UniformChunkedReconModel

__all__ = [
    "GeneReconModel",
    "UniformChunkedReconModel",
]
