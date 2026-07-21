import math

import torch

_NEG_INF = float("-inf")
_LN2 = math.log(2.0)


def safe_log2(x: torch.Tensor) -> torch.Tensor:
    positive = x > 0
    return torch.where(positive, torch.log2(torch.where(positive, x, torch.ones_like(x))), x.new_full(x.shape, _NEG_INF))


def survival_by_species_from_E(E: torch.Tensor) -> torch.Tensor:
    """Per-species survival probabilities ``1 - 2**E`` without cancellation."""
    return -torch.expm1(E * _LN2)


def survival_from_E(E: torch.Tensor, origination_probs: torch.Tensor | None = None, *, keepdim: bool = False) -> torch.Tensor:
    """Survival (non-extinction) probability ``1 - sum_s w_s 2^{E_s}``, computed without cancellation.

    ``E`` is log2-extinction (``E_s <= 0``); ``origination_probs`` are the per-species weights ``w_s``
    (summing to 1), or ``None`` for the uniform ``1/S``.  The naive ``1 - mean_s 2^{E_s}`` round-trips
    log -> linear -> log and catastrophically cancels: once ``2^{E_s}`` rounds to ``1.0`` in the working
    dtype (fp32: extinction within ~1.2e-7 of 1) the difference collapses to 0, destroying survival
    probabilities below ~1e-6 and yielding spurious ~1e36 gradients.  The identity

        ``1 - sum_s w_s 2^{E_s} = sum_s w_s (1 - 2^{E_s}) = sum_s w_s (-expm1(E_s * ln2))``

    keeps everything small: ``1 - 2^{E_s} = -expm1(E_s * ln2)`` is the compensated ``2^x - 1`` primitive
    (exact as ``E_s -> 0``), and a weighted sum of non-negative terms cannot cancel.  No clamp: when a
    family is genuinely certain to go extinct (``E_s == 0`` for all ``s``) this returns exactly ``0``.
    """
    surv_s = survival_by_species_from_E(E)
    if origination_probs is None:
        return surv_s.mean(dim=-1, keepdim=keepdim)
    return (origination_probs * surv_s).sum(dim=-1, keepdim=keepdim)


def log2_survival(E: torch.Tensor, origination_probs: torch.Tensor | None = None) -> torch.Tensor:
    """``log2`` of :func:`survival_from_E` -- the survival log-normalizer of the DTL NLL."""
    return safe_log2(survival_from_E(E, origination_probs))


def logsumexp2(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    row_max = x.max(dim=dim, keepdim=True).values
    row_max_safe = torch.where(row_max == _NEG_INF, torch.zeros_like(row_max), row_max)
    total = torch.exp2(x - row_max_safe).sum(dim=dim, keepdim=True)
    result = safe_log2(total) + row_max
    return result if keepdim else result.squeeze(dim)
