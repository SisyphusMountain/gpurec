"""Notebook-friendly public API for gpurec.

Exports the high-level PyTorch modules plus lightweight metadata objects used
by workflow scripts and diagnostics.
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
