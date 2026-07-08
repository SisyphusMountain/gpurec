"""Centralized configuration package for gpurec.

Currently exposes the shared dtype-relative tolerance helpers used by the
solver and implicit-gradient code. This package will grow into a single
``GpurecConfig`` home for the config-centralization refactor; later tasks
build on this skeleton.
"""
from gpurec.config.gpurec_config import dtype_rel_tol_default, dtype_rel_tol_floor
from gpurec.config.memory import MemoryOptions
from gpurec.config.newton import NewtonOptions
from gpurec.config.rates import RateBounds

__all__ = [
    "dtype_rel_tol_default", "dtype_rel_tol_floor", "MemoryOptions", "NewtonOptions", "RateBounds",
]
