"""Workflow helpers with lazy public exports."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType


_LAZY_EXPORTS = {
    "RunConfig": ("gpurec.workflow.config", "RunConfig"),
    "SamplingConfig": ("gpurec.workflow.config", "SamplingConfig"),
    "OptimizationResult": ("gpurec.workflow.optimize", "OptimizationResult"),
    "OptimizationRunner": ("gpurec.workflow.optimize", "OptimizationRunner"),
    "SamplingResult": ("gpurec.workflow.sampling", "SamplingResult"),
    "SamplingRunner": ("gpurec.workflow.sampling", "SamplingRunner"),
    "optimize": ("gpurec.workflow.optimize", "optimize"),
    "sample": ("gpurec.workflow.sampling", "sample"),
}

__all__ = list(_LAZY_EXPORTS)

_MISSING = object()


def _resolve_export(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module 'gpurec.workflow' has no attribute {name!r}"
        ) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str):
    return _resolve_export(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


class _WorkflowModule(ModuleType):
    # Child-module imports bind attributes such as ``workflow.optimize`` to the
    # module object; public exports with the same name must continue to resolve
    # to the callable.
    def __getattribute__(self, name: str):
        namespace = ModuleType.__getattribute__(self, "__dict__")
        target = namespace.get("_LAZY_EXPORTS", {}).get(name)
        if target is not None:
            value = namespace.get(name, _MISSING)
            module_name, _ = target
            if not (
                isinstance(value, ModuleType)
                and getattr(value, "__name__", None) == module_name
            ):
                if value is not _MISSING:
                    return value
            return namespace["_resolve_export"](name)
        return ModuleType.__getattribute__(self, name)


sys.modules[__name__].__class__ = _WorkflowModule
