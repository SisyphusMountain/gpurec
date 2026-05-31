from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from gpurec.api.model import GeneReconModel


@dataclass(frozen=True)
class BatchFinalCache:
    """Runtime cache for final active-batch loss and gradient rows."""

    loss: torch.Tensor
    grad: torch.Tensor
    ready: torch.Tensor

    @classmethod
    def create(cls, model: GeneReconModel) -> BatchFinalCache:
        return cls(
            loss=torch.empty(
                (int(model.n_families),),
                device=model.theta.device,
                dtype=model.theta.dtype,
            ),
            grad=torch.empty_like(model.theta),
            ready=torch.zeros(
                (int(model.n_families),),
                device=model.theta.device,
                dtype=torch.bool,
            ),
        )

    def cache(
        self,
        *,
        model: GeneReconModel,
        loss_vec: torch.Tensor,
        active_indices: torch.Tensor,
    ) -> None:
        idx = active_indices.to(device=self.ready.device, dtype=torch.long)
        if idx.numel() == 0:
            return
        self.loss.index_copy_(0, idx, loss_vec.detach().index_select(0, idx))
        if model.theta.grad is not None:
            self.grad.index_copy_(
                0,
                idx,
                model.theta.grad.detach().index_select(0, idx),
            )
        self.ready.index_fill_(0, idx, True)

    def invalidate(self, family_indices: Sequence[int] | torch.Tensor) -> None:
        idx = torch.as_tensor(
            family_indices,
            dtype=torch.long,
            device=self.ready.device,
        )
        if idx.numel() == 0:
            return
        self.ready.index_fill_(0, idx, False)

    def cached_final_result(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not bool(self.ready.all().item()):
            return None
        return self.loss.detach().clone(), self.grad.detach().clone()
