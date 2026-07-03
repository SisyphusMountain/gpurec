"""Benchmark + AleRax-fidelity helpers (Triton-free, CPU-importable)."""
from gpurec.bench.alerax_io import (
    AleraxRates,
    parse_alerax_likelihoods,
    parse_alerax_parameters,
    norm_family_name,
    global_rates,
)

__all__ = [
    "AleraxRates",
    "parse_alerax_likelihoods",
    "parse_alerax_parameters",
    "norm_family_name",
    "global_rates",
]
