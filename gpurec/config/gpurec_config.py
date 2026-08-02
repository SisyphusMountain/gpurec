from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path

import torch

from gpurec.api.solver_options import SolverOptions
from gpurec.config.memory import MemoryOptions
from gpurec.config.newton import NewtonOptions
from gpurec.config.precision import PrecisionOptions
from gpurec.config.rates import RateBounds

# Package-relative (NOT cwd-relative) path to the hand-written TOML mirror of
# the dataclass defaults below. Resolved from this module's own file location
# so it works regardless of the caller's working directory or how gpurec was
# installed.
DEFAULTS_TOML_PATH = Path(__file__).resolve().parent / "defaults.toml"

# NOTE: ``gpurec.solver.penalties`` is imported lazily (inside
# ``_default_penalty_options`` below) rather than at module level. A top-level
# ``from gpurec.solver.penalties import PenaltyOptions`` here forces
# ``gpurec.solver``'s package ``__init__`` to run, which imports
# ``gpurec.solver.value_and_grad`` -> ``gpurec.api._execution`` ->
# ``gpurec.api._implicit_grad``. That module itself imports
# ``gpurec.config`` (for the dtype-tol helpers) at its top, so when this
# module is reached *through* that same ``gpurec.api._implicit_grad`` import
# (the normal path when anything imports plain ``gpurec``), the eager import
# would try to pull a still-partially-initialized ``gpurec.api._implicit_grad``
# module and raise ``ImportError: cannot import name ... (most likely due to a
# circular import)``. Deferring the import to first use (i.e. to when a
# ``GpurecConfig``/``PenaltyOptions`` default is actually constructed, by
# which point the whole package graph has finished loading) breaks the cycle
# without changing any public import path for callers of ``PenaltyOptions``.
# ``SolverOptions`` has no such cycle (``gpurec.api`` is a plain namespace
# package with no ``__init__.py``, and ``solver_options.py`` has no gpurec
# imports of its own), so it stays a normal top-level import.


def _default_penalty_options():
    from gpurec.solver.penalties import PenaltyOptions
    return PenaltyOptions()


def dtype_rel_tol_default(dtype) -> float:
    """Dtype-matched relative-residual target for dtype-aware linear/fixed-point solves.

    The target must sit ABOVE the finite-precision stagnation floor (a few * the
    unit roundoff ``eps``) yet stay BELOW the ~2e-4 downstream gradient atomic-noise
    floor, so the solve is as tight as the working precision *reliably* allows
    without wasting iterations. These are exactly the values used elsewhere in
    the codebase for the same purpose (e.g. the E-adjoint Neumann-series solve
    and ``gpurec.solver.hvp.forward_tangent._default_tol``):

      * fp32: ``1e-6`` (~8.4x fp32 eps 1.19e-7).
      * fp64: ``1e-12`` (~4.5e3x fp64 eps 2.2e-16; tight but trivially reachable).
    """
    return 1e-12 if dtype == torch.float64 else 1e-6


def dtype_rel_tol_floor(dtype) -> float:
    """Tightest relative residual an iteration can reach in a given dtype.

    A small multiple of the unit roundoff. Any requested target below this is
    physically unreachable, so callers clamp up to it (with a warning) rather
    than spin to ``max_iter`` and raise on a solve that is, in fact, fully
    converged.
    """
    return 4.0 * float(torch.finfo(dtype).eps)


def _merge_into(instance, overrides: dict, path: str):
    """Return a copy of ``instance`` with ``overrides`` deep-merged in.

    ``overrides`` may be a partial dict: keys absent from it keep the value
    already present on ``instance``. Nested dataclass fields are merged
    recursively when the override value for that key is itself a dict. Any
    key (at any nesting depth) that does not name a field of the dataclass
    being merged into raises ``ValueError`` -- unknown keys are never
    silently ignored.
    """
    if not isinstance(overrides, dict):
        full = path or "<root>"
        raise TypeError(f"expected a dict of overrides at {full!r}, got {type(overrides).__name__}")
    valid_names = {f.name for f in fields(instance)}
    updates = {}
    for key, value in overrides.items():
        if key not in valid_names:
            full = f"{path}.{key}" if path else key
            raise ValueError(f"unknown config key: {full!r}")
        full_path = f"{path}.{key}" if path else key
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            updates[key] = _merge_into(current, value, full_path)
        elif isinstance(current, tuple) and isinstance(value, list):
            # TOML (and JSON) have no tuple type -- arrays always decode to
            # Python ``list``. Coerce back to ``tuple`` here so a field whose
            # dataclass default is a tuple (e.g. ``PenaltyOptions.lambdas``)
            # round-trips through ``from_dict``/``from_toml`` as a tuple, not a
            # list, and so equality against the dataclass default still holds.
            updates[key] = tuple(value)
        else:
            updates[key] = value
    return replace(instance, **updates)


@dataclass
class GpurecConfig:
    """Top-level configuration composing the per-area option dataclasses.

    Purely a composition root: constructing ``GpurecConfig()`` is identical to
    constructing each of the six sub-option dataclasses with their own
    defaults. No existing dataclass is modified by this module.
    """
    solver: SolverOptions = field(default_factory=SolverOptions)
    precision: PrecisionOptions = field(default_factory=PrecisionOptions)
    newton: NewtonOptions = field(default_factory=NewtonOptions)
    rates: RateBounds = field(default_factory=RateBounds)
    regularizer: PenaltyOptions = field(default_factory=_default_penalty_options)
    memory: MemoryOptions = field(default_factory=MemoryOptions)

    def validate(self) -> None:
        """Delegate to every sub-option's own ``validate`` (when it defines one)."""
        for sub in (
            self.solver,
            self.precision,
            self.newton,
            self.rates,
            self.regularizer,
            self.memory,
        ):
            validate_fn = getattr(sub, "validate", None)
            if callable(validate_fn):
                validate_fn()

    def to_dict(self) -> dict:
        """Nested ``dataclasses.asdict`` of every sub-option."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GpurecConfig":
        """Deep-merge a (possibly partial) dict onto ``GpurecConfig()`` defaults.

        Any key -- top-level or nested -- that does not correspond to an
        existing field raises ``ValueError`` rather than being ignored.
        """
        return _merge_into(cls(), d, path="")

    @classmethod
    def genewise_reference(cls) -> "GpurecConfig":
        """The ``fit_genewise`` recipe's solver + rate-bounds config (see ``gpurec/fit/genewise_fit.py``).

        This is the single source for the recipe's base solver settings (pi_iters/neumann_terms are
        overridden per-tier by the fit loop itself, so they are left at the ``SolverOptions`` default
        here) and its tighter genewise rate box. ``gpurec.fit.genewise_fit._BASE_SOLVER`` is derived
        from this factory's ``.solver`` -- values must never be edited independently in the two places.
        """
        return cls(
            solver=SolverOptions(
                e_max_iter=128, e_tol=1e-8,
                e_adjoint_max_iter=128, e_adjoint_tol=1e-7,
                adjoint_pruning_threshold=1e-6, use_adjoint_pruning=True, pibar_side_threshold=0.0,
            ),
            rates=RateBounds.genewise(),
        )

    @classmethod
    def map_cv_reference(cls) -> "GpurecConfig":
        """The ``map_cv`` CONVERGED solver config (pi_iters=64, neumann_terms=64; see
        ``gpurec/fit/map_cv.py``'s ``_CV_SO``), required for the CV-fitted theta to be a true optimum.
        ``gpurec.fit.map_cv._CV_SO`` is derived from this factory's ``.solver``.
        """
        return cls(
            solver=SolverOptions(
                e_max_iter=128, e_tol=1e-8, pi_iters=64, neumann_terms=64,
                e_adjoint_max_iter=128, e_adjoint_tol=None,
                adjoint_pruning_threshold=1e-6,
                use_adjoint_pruning=True, pibar_side_threshold=0.0,
            ),
        )

    @classmethod
    def optimize_reference(cls) -> "GpurecConfig":
        """``gpurec.fit.optimize.optimize`` has no separate solver recipe -- ``GeneReconModel`` falls
        back to the default ``SolverOptions`` wherever ``optimize()`` is used, so the reference config
        is exactly ``GpurecConfig()`` defaults."""
        return cls()

    @classmethod
    def from_toml(cls, path: str | Path) -> "GpurecConfig":
        """Load a (possibly partial) TOML file and deep-merge it onto ``GpurecConfig()``.

        ``path`` is read in binary mode via :mod:`tomllib` (as required by that
        module) and the resulting dict is passed through :meth:`from_dict`, so a
        user TOML need only list the fields it wants to override -- and any
        unknown key (top-level or nested) still raises ``ValueError`` rather
        than being silently ignored.

        ``gpurec/config/defaults.toml`` (see ``DEFAULTS_TOML_PATH``) is a
        hand-written mirror of every field's dataclass default; the guard test
        ``GpurecConfig.from_toml(DEFAULTS_TOML_PATH) == GpurecConfig()`` is the
        permanent check that the two never drift apart.
        """
        with open(path, "rb") as fh:
            d = tomllib.load(fh)
        return cls.from_dict(d)


def load_config(path: str | Path | None = None) -> GpurecConfig:
    """Return the effective :class:`GpurecConfig`.

    ``path is None`` (the default) returns ``GpurecConfig()`` directly -- the
    hardcoded dataclass defaults, with no file IO. When ``path`` is given, it
    is loaded as a TOML file via :meth:`GpurecConfig.from_toml`, deep-merged
    onto the defaults so the file need only list overrides.
    """
    if path is None:
        return GpurecConfig()
    return GpurecConfig.from_toml(path)
