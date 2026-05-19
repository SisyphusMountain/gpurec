"""Notebook-friendly API for gpurec.

Public exports:
    GeneReconModel — torch.nn.Module wrapping the gradient pipeline.
"""
from .model import (
    ActiveFamilyBatch,
    BatchMetadata,
    FamilyInput,
    GeneReconModel,
    ReconciliationState,
)
from .uniform_chunked import UniformChunkedReconModel

__all__ = [
    "ActiveFamilyBatch",
    "BatchMetadata",
    "FamilyInput",
    "GeneReconModel",
    "ReconciliationState",
    "UniformChunkedReconModel",
]
