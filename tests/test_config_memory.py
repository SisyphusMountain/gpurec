"""``MemoryOptions`` pins the memory/infra knobs scattered across ``gpurec/core/memory_policy.py``,
``gpurec/solver/value_and_grad.py``, ``gpurec/solver/curvature/gauge.py``, and
``gpurec/solver/hvp/exact.py``. See task-6 brief. No numeric/gradient value depends on these
knobs -- they only gate memory-budget/cache-empty decisions.
"""
import inspect
import os

import pytest

from gpurec.config.memory import MemoryOptions


def test_memory_options_defaults_pinned():
    opts = MemoryOptions()
    assert opts.fraction == 0.85
    assert opts.reserve_gib == 1.0
    assert opts.scratch_tensors == 10
    assert opts.min_free_gib_driver == 4.0
    assert opts.min_free_gib_hvp == 8.0
    assert opts.free_cache_every == 32
    assert opts.grad_avg_k == 1


def test_memory_options_importable_from_config_package():
    from gpurec.config import MemoryOptions as ReExported
    assert ReExported is MemoryOptions


def test_memory_policy_signature_defaults_match_memory_options():
    """cuda_memory_budget_bytes / proposal0_wave_scratch_bytes default kwargs must be sourced
    from MemoryOptions(), not bare copy-pasted literals."""
    from gpurec.core import memory_policy

    opts = MemoryOptions()
    budget_params = inspect.signature(memory_policy.cuda_memory_budget_bytes).parameters
    assert budget_params["default_fraction"].default == opts.fraction
    assert budget_params["default_reserve_gib"].default == opts.reserve_gib

    scratch_params = inspect.signature(memory_policy.proposal0_wave_scratch_bytes).parameters
    assert scratch_params["scratch_tensors"].default == opts.scratch_tensors


def test_value_and_grad_signature_defaults_match_memory_options():
    from gpurec.solver import value_and_grad

    opts = MemoryOptions()
    fc_params = inspect.signature(value_and_grad.free_cuda_cache_if_tight).parameters
    assert fc_params["min_free_gib"].default == opts.min_free_gib_driver

    vg_params = inspect.signature(value_and_grad.make_value_and_grad).parameters
    assert vg_params["grad_avg_K"].default == opts.grad_avg_k


def test_curvature_hvp_rebuild_gate_wired_to_memory_options():
    """curvature.newton_min's HVP-rebuild free_cuda_cache_if_tight call must read
    MemoryOptions().min_free_gib_hvp rather than a bare `min_free_gib=8.0` literal."""
    from gpurec.solver import curvature

    src = inspect.getsource(curvature.gauge.newton_min)
    assert "MemoryOptions" in src
    assert "min_free_gib=8.0" not in src


def test_hvp_exact_free_cache_every_wired_to_memory_options():
    """make_exact_hvp_single's NEWTON_FREE_CACHE_EVERY env-var fallback must be MemoryOptions()'s
    free_cache_every rather than a bare `32` literal. (The reverse-sweep body lives in
    make_exact_hvp_single since make_exact_hvp became a single/multi-batch dispatcher.)"""
    from gpurec.solver.hvp import exact as hvp_exact

    src = inspect.getsource(hvp_exact.make_exact_hvp_single)
    assert "MemoryOptions" in src
    assert "else 32" not in src


def test_memory_policy_env_var_fraction_override_wins():
    """GPUREC_MEMORY_POLICY_FRACTION must still override the MemoryOptions()-sourced default --
    env-var precedence is preserved, not replaced by the dataclass."""
    from gpurec.core import memory_policy

    if not __import__("torch").cuda.is_available():
        pytest.skip("requires CUDA to exercise cuda_memory_budget_bytes")

    old = os.environ.get("GPUREC_MEMORY_POLICY_FRACTION")
    try:
        os.environ.pop("GPUREC_MEMORY_POLICY_FRACTION", None)
        budget_default = memory_policy.cuda_memory_budget_bytes()

        os.environ["GPUREC_MEMORY_POLICY_FRACTION"] = "0.1"
        budget_overridden = memory_policy.cuda_memory_budget_bytes()
    finally:
        if old is None:
            os.environ.pop("GPUREC_MEMORY_POLICY_FRACTION", None)
        else:
            os.environ["GPUREC_MEMORY_POLICY_FRACTION"] = old

    assert budget_default != budget_overridden
