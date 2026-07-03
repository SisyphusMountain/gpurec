"""Parsers for AleRax reference output files.

Pure text; no Triton/CUDA import. All log-likelihoods are NATS (ln), matching AleRax.
The rate columns in ``model_parameters.txt`` are ordered ``D L T`` — the SAME order as
gpurec ``theta = [log2 D, log2 L, log2 T]``, so they map straight to theta (no reorder).
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

AleraxRates = namedtuple("AleraxRates", ["D", "L", "T"])
# AleRax columns D L T == gpurec theta order [log2 D, log2 L, log2 T]; no reorder helper needed.

_SUFFIXES = (".ale", ".txt", ".ufboot", ".newick", ".nwk", ".treelist")


def norm_family_name(name: str) -> str:
    """Strip a known extension and directory so gpurec and AleRax names align."""
    base = str(name).strip().rsplit("/", 1)[-1]
    for suf in _SUFFIXES:
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def parse_alerax_likelihoods(path) -> dict:
    """Parse ``per_fam_likelihoods.txt`` -> {normalized_family: logL_nats}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            out[norm_family_name(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return out


def parse_alerax_parameters(path) -> dict:
    """Parse ``model_parameters.txt`` -> {node: AleraxRates(D, L, T)}.

    Skips the ``# node D L T`` comment header. Node labels may be numeric or strings.
    """
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            d, l, t = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        out[parts[0]] = AleraxRates(D=d, L=l, T=t)
    return out


def global_rates(params: dict) -> AleraxRates:
    """Return the common (D, L, T) triple for a GLOBAL-mode AleRax run.

    Raises ValueError if the run is empty or has per-node rate variation.
    """
    if not params:
        raise ValueError("no rate rows parsed")
    vals = list(params.values())
    first = vals[0]
    for r in vals[1:]:
        if r != first:
            raise ValueError("rates are not global (per-node variation present)")
    return first
