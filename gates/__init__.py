"""Standalone FD / parity / analytic-gradient gate scripts for gpurec.optim.

Run a gate directly, e.g. ``python -m gates._verify_hvp``. These are developer
verification harnesses, NOT part of the shipped ``gpurec`` package and NOT
collected by pytest (they live outside ``testpaths``). ``tests/test_optim_golden.py``
imports ``gates._parity_kbench`` as a fixture helper.
"""
