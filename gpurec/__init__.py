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

from gpurec.api import GeneReconModel, UniformChunkedReconModel
from gpurec.backtracking import (
    EVENT_KEYS,
    export_backtracking_input,
    recphyloxml_event_counts,
    sample_recphyloxml,
    sample_recphyloxmls,
)
from gpurec.workflow import (
    OptimizationRunner,
    RunConfig,
    SamplingConfig,
    SamplingRunner,
    optimize,
    sample,
)

__all__ = [
    "GeneReconModel",
    "UniformChunkedReconModel",
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
