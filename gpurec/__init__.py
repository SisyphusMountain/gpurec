"""
GPU-accelerated phylogenetic reconciliation package.

Notebook-friendly API:
  from gpurec import GeneReconModel
  model = GeneReconModel.from_trees("sp.nwk", ["g1.nwk"], mode="global")

Lower-level access:
  from gpurec.core.model import GeneDataset
  from gpurec.core.likelihood import E_fixed_point, compute_nll
  from gpurec.core.forward import Pi_wave_forward
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


_SOURCE_VERSION = "0.1.0"


def _distribution_version() -> str:
    try:
        return version(__name__)
    except PackageNotFoundError:
        return _SOURCE_VERSION


__version__ = _distribution_version()

_LAZY_EXPORTS = {
    "GeneReconModel": "gpurec.api",
    "UniformChunkedReconModel": "gpurec.api",
    "UniformChunkMetadata": "gpurec.api",
    "ActiveFamilyBatch": "gpurec.api",
    "BatchMetadata": "gpurec.api",
    "FamilyInput": "gpurec.api",
    "ReconciliationState": "gpurec.api",
    "EVENT_KEYS": "gpurec.backtracking",
    "ensure_backtracking_available": "gpurec.backtracking",
    "export_backtracking_input": "gpurec.backtracking",
    "recphyloxml_event_counts": "gpurec.backtracking",
    "sample_recphyloxml": "gpurec.backtracking",
    "sample_recphyloxmls": "gpurec.backtracking",
    "RunConfig": "gpurec.workflow",
    "SamplingConfig": "gpurec.workflow",
    "OptimizationResult": "gpurec.workflow",
    "OptimizationRunner": "gpurec.workflow",
    "SamplingResult": "gpurec.workflow",
    "SamplingRunner": "gpurec.workflow",
    "optimize": "gpurec.workflow",
    "sample": "gpurec.workflow",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'gpurec' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
