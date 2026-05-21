"""Internal origination-prior value objects.

This module is a typed contract around the existing origination probability
semantics.  It owns no likelihood math; validation and normalization still flow
through ``prepare_origination_probs`` so callers can prepare once before passing
probabilities into existing evaluator paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index as integer_index
from typing import Any

import torch

from .likelihood import prepare_origination_probs

__all__ = [
    "OriginationPrior",
    "PreparedOriginationPrior",
    "prepare_origination_prior",
]


def _family_indices_tensor(
    family_indices: Sequence[int],
    *,
    family_count: int,
    device: torch.device,
) -> torch.Tensor:
    indices: list[int] = []
    for raw_index in family_indices:
        if isinstance(raw_index, bool):
            raise ValueError("family_indices entries must be integers, got bool")
        try:
            family_index = integer_index(raw_index)
        except TypeError as exc:
            raise ValueError("family_indices entries must be integers") from exc
        if family_index < 0 or family_index >= family_count:
            raise IndexError(
                f"family index {family_index} out of range for "
                f"{family_count} family-specific origination rows"
            )
        indices.append(int(family_index))
    return torch.as_tensor(indices, dtype=torch.long, device=device)


@dataclass(frozen=True)
class PreparedOriginationPrior:
    """Validated origination weights ready for likelihood call sites."""

    probs: torch.Tensor | None
    species_count: int
    family_count: int | None = None

    @classmethod
    def from_raw(
        cls,
        probs: Any,
        *,
        S: int,
        device: torch.device | str,
        dtype: torch.dtype,
        family_count: int | None = None,
        assume_prepared: bool = False,
    ) -> "PreparedOriginationPrior":
        prepared = prepare_origination_probs(
            probs,
            S=int(S),
            device=device,
            dtype=dtype,
            family_count=family_count,
            assume_prepared=assume_prepared,
        )
        if prepared is None:
            prepared_family_count = None
        elif prepared.ndim == 2:
            prepared_family_count = int(prepared.shape[0])
        else:
            prepared_family_count = None
        return cls(
            probs=prepared,
            species_count=int(S),
            family_count=prepared_family_count,
        )

    @property
    def is_default_uniform(self) -> bool:
        """Whether the prior uses the implicit uniform species distribution."""
        return self.probs is None

    @property
    def is_shared(self) -> bool:
        """Whether one distribution is shared by every family."""
        return self.probs is None or self.probs.ndim == 1

    @property
    def is_family_specific(self) -> bool:
        """Whether the prepared tensor has one row per family."""
        return self.probs is not None and self.probs.ndim == 2

    @property
    def log_weights(self) -> torch.Tensor | None:
        """Log2 view used by root-likelihood implementations."""
        if self.probs is None:
            return None
        return torch.log2(self.probs)

    def select_families(
        self,
        family_indices: Sequence[int],
    ) -> "PreparedOriginationPrior":
        """Return the prepared-prior view for a selected family subset."""
        if not self.is_family_specific:
            return self
        if self.probs is None or self.family_count is None:
            raise RuntimeError("family-specific prior is missing family metadata")
        idx = _family_indices_tensor(
            family_indices,
            family_count=self.family_count,
            device=self.probs.device,
        )
        return PreparedOriginationPrior(
            probs=self.probs.index_select(0, idx),
            species_count=self.species_count,
            family_count=int(idx.numel()),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PreparedOriginationPrior":
        """Move a prepared prior without revalidating probability mass."""
        if self.probs is None:
            return self
        target_device = self.probs.device if device is None else torch.device(device)
        target_dtype = self.probs.dtype if dtype is None else dtype
        if self.probs.device == target_device and self.probs.dtype == target_dtype:
            return self
        return PreparedOriginationPrior.from_raw(
            self.probs,
            S=self.species_count,
            device=target_device,
            dtype=target_dtype,
            family_count=self.family_count,
            assume_prepared=True,
        )


@dataclass(frozen=True)
class OriginationPrior:
    """Raw origination-prior specification awaiting preparation."""

    probs: Any = None
    assume_prepared: bool = False

    @classmethod
    def prepared(cls, probs: Any) -> "OriginationPrior":
        """Wrap weights that have already crossed the validation boundary."""
        return cls(probs=probs, assume_prepared=True)

    def prepare(
        self,
        *,
        S: int,
        device: torch.device | str,
        dtype: torch.dtype,
        family_count: int | None = None,
    ) -> PreparedOriginationPrior:
        """Validate and normalize into a prepared prior."""
        return prepare_origination_prior(
            self,
            S=S,
            device=device,
            dtype=dtype,
            family_count=family_count,
        )


def prepare_origination_prior(
    prior: Any,
    *,
    S: int,
    device: torch.device | str,
    dtype: torch.dtype,
    family_count: int | None = None,
) -> PreparedOriginationPrior:
    """Coerce raw or prepared origination input into a prepared-prior object."""
    if isinstance(prior, PreparedOriginationPrior):
        if int(S) != prior.species_count:
            raise ValueError(
                f"prepared origination prior has S={prior.species_count}, "
                f"got S={int(S)}"
            )
        if (
            family_count is not None
            and prior.is_family_specific
            and int(family_count) != prior.family_count
        ):
            raise ValueError(
                "prepared family-specific origination prior has one row per "
                f"family: expected {family_count}, got {prior.family_count}"
            )
        return prior.to(device=device, dtype=dtype)
    if isinstance(prior, OriginationPrior):
        return PreparedOriginationPrior.from_raw(
            prior.probs,
            S=S,
            device=device,
            dtype=dtype,
            family_count=family_count,
            assume_prepared=prior.assume_prepared,
        )
    return PreparedOriginationPrior.from_raw(
        prior,
        S=S,
        device=device,
        dtype=dtype,
        family_count=family_count,
    )
