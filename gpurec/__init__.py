"""
GPU-accelerated phylogenetic reconciliation package.

Notebook-friendly API:
  from gpurec import GeneReconModel
  model = GeneReconModel.from_trees("sp.nwk", ["g1.nwk"], mode="global")

Lower-level access:
  from gpurec.core.model import GeneDataset
  from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
  from gpurec.core.forward import Pi_wave_forward
  from gpurec.core.legacy import Pi_fixed_point
"""

__all__: list[str] = []

try:
    from gpurec.api import GeneReconModel  # noqa: F401
    __all__.append("GeneReconModel")
except ImportError:
    # JIT C++ build or torch may be missing in some environments; allow
    # partial import of subpackages even if the high-level API can't load.
    pass
