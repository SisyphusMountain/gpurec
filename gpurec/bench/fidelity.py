"""Run gpurec at AleRax's fitted rates and compare per-family log-likelihood.

``compare`` is pure (CPU). ``reconcile_at_alerax_rates`` builds a GeneReconModel and
needs Triton/CUDA. All log-likelihoods are NATS.
"""
from __future__ import annotations

import math
from collections import namedtuple

FidelityReport = namedtuple(
    "FidelityReport",
    ["n_matched", "mean_abs_delta", "rmse", "max_abs_delta", "total_delta", "pearson_r"],
)


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def compare(gpurec_ll: dict, alerax_ll: dict) -> FidelityReport:
    from gpurec.bench.alerax_io import norm_family_name
    g = {norm_family_name(k): v for k, v in gpurec_ll.items()}
    a = {norm_family_name(k): v for k, v in alerax_ll.items()}
    keys = sorted(set(g) & set(a))
    n = len(keys)
    if n == 0:
        nan = float("nan")
        return FidelityReport(0, nan, nan, nan, nan, None)
    deltas = [g[k] - a[k] for k in keys]
    abs_d = [abs(x) for x in deltas]
    r = _pearson([g[k] for k in keys], [a[k] for k in keys]) if n > 2 else None
    return FidelityReport(n, sum(abs_d) / n, math.sqrt(sum(x * x for x in deltas) / n),
                          max(abs_d), sum(deltas), r)


def reconcile_at_alerax_rates(species, genes, rates, *, device="cuda", dtype=None,
                              mode="global", solver_options=None) -> dict:
    """rates: AleraxRates/(D, L, T) in AleRax order. Returns {family: logL_nats}."""
    import torch
    from gpurec.api.model import GeneReconModel
    from gpurec.bench.alerax_io import norm_family_name

    if dtype is None:
        dtype = torch.float64
    D, L, T = float(rates[0]), float(rates[1]), float(rates[2])
    gene_list = genes if isinstance(genes, list) else [genes]
    model = GeneReconModel(species, gene_list, mode=mode, device=device,
                           solver_options=solver_options)
    # gpurec theta order is [log2 D, log2 L, log2 T] == AleRax (D, L, T) column order
    triple = torch.tensor([math.log2(D), math.log2(L), math.log2(T)],
                          dtype=model.theta.dtype, device=model.theta.device)
    with torch.no_grad():
        if model.theta.dim() == 1:
            model.theta.copy_(triple)
        else:
            model.theta.copy_(triple.expand_as(model.theta))
        loss_bits = float(model())
    fam = norm_family_name(gene_list[0])
    return {fam: -loss_bits * math.log(2.0)}
