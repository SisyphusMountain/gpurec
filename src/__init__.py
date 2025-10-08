"""
GPU-accelerated phylogenetic reconciliation package.

This package's public API is under active refactor. To avoid import-time
errors when submodules are not present, the top-level package does not
eagerly import submodules.

Downstream code should import directly from concrete subpackages, e.g.:
  from src.core.model import GeneDataset
  from src.core.likelihood import E_fixed_point, Pi_fixed_point
"""

__all__: list[str] = []
