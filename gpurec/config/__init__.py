"""Centralized configuration package for gpurec.

Exposes the shared dtype-relative tolerance helpers used by the solver and
implicit-gradient code, the per-area option dataclasses, and the top-level
``GpurecConfig`` that composes them.
"""
from gpurec.config.gpurec_config import (
    DEFAULTS_TOML_PATH,
    GpurecConfig,
    dtype_rel_tol_default,
    dtype_rel_tol_floor,
    load_config,
)
from gpurec.config.memory import MemoryOptions
from gpurec.config.newton import NewtonOptions
from gpurec.config.rates import RateBounds
from gpurec.api.solver_options import SolverOptions

__all__ = [
    "dtype_rel_tol_default", "dtype_rel_tol_floor",
    "GpurecConfig", "SolverOptions", "NewtonOptions", "RateBounds", "PenaltyOptions", "MemoryOptions",
    "DEFAULTS_TOML_PATH", "load_config",
]


def __getattr__(name):
    # ``PenaltyOptions`` (re-exported from ``gpurec.solver.penalties``) is
    # resolved lazily on first attribute access (PEP 562) rather than via a
    # top-level import: a top-level ``from gpurec.solver.penalties import
    # PenaltyOptions`` here forces ``gpurec.solver``'s package ``__init__`` to
    # run, which -- via ``value_and_grad`` -> ``gpurec.api._execution`` ->
    # ``gpurec.api._implicit_grad`` -- imports back into a still-partially-
    # initialized ``gpurec.api._implicit_grad`` module (the module that
    # imports *this* package for the dtype-tol helpers in the first place),
    # raising a circular-import ``ImportError``. See the matching note in
    # ``gpurec/config/gpurec_config.py`` for the full chain. By the time any
    # caller actually accesses ``gpurec.config.PenaltyOptions``, package
    # loading has finished and the import below is a plain cached lookup.
    if name == "PenaltyOptions":
        from gpurec.solver.penalties import PenaltyOptions
        return PenaltyOptions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
