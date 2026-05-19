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

_LAZY_EXPORTS = {
    "GeneReconModel": "gpurec.api",
    "UniformChunkedReconModel": "gpurec.api",
    "ActiveFamilyBatch": "gpurec.api",
    "BatchMetadata": "gpurec.api",
    "FamilyInput": "gpurec.api",
    "ReconciliationState": "gpurec.api",
    "EVENT_KEYS": "gpurec.backtracking",
    "export_backtracking_input": "gpurec.backtracking",
    "recphyloxml_event_counts": "gpurec.backtracking",
    "sample_recphyloxml": "gpurec.backtracking",
    "sample_recphyloxmls": "gpurec.backtracking",
    "RunConfig": "gpurec.workflow",
    "SamplingConfig": "gpurec.workflow",
    "OptimizationRunner": "gpurec.workflow",
    "SamplingRunner": "gpurec.workflow",
    "optimize": "gpurec.workflow",
    "sample": "gpurec.workflow",
}

__all__ = [
    "GeneReconModel",
    "UniformChunkedReconModel",
    "ActiveFamilyBatch",
    "BatchMetadata",
    "FamilyInput",
    "ReconciliationState",
    "EVENT_KEYS",
    "export_backtracking_input",
    "recphyloxml_event_counts",
    "sample_recphyloxml",
    "sample_recphyloxmls",
    "RunConfig",
    "SamplingConfig",
    "OptimizationRunner",
    "SamplingRunner",
    "optimize",
    "sample",
]


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
