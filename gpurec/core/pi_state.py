"""Typed offset sidecar for the canonical row-gauged log-domain Pi state.

The mathematical and lifecycle contract is documented in
``docs/centered_state_contract.md``. Residual matrices remain in the solver
result; this small sidecar can safely remain on a streamed batch static without
retaining every batch's full Pi/Pibar allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class PiState:
    """Offsets paired with Pi/Pibar residuals returned by one forward solve."""

    pi_offset: torch.Tensor
    pibar_offset: torch.Tensor

    def validate(
        self,
        pi_residual: torch.Tensor,
        pibar_residual: torch.Tensor,
        pibar_row_max: torch.Tensor,
        *,
        check_values: bool = True,
    ) -> None:
        pi = pi_residual
        pibar = pibar_residual
        if pi.ndim != 2 or pibar.shape != pi.shape:
            raise ValueError("Pi and Pibar residuals must share shape [rows, species]")
        if pi.dtype not in _SUPPORTED_FLOAT_DTYPES or pibar.dtype != pi.dtype:
            raise TypeError(
                "Pi/Pibar residuals must share dtype torch.float32 or torch.float64"
            )
        if pi.device.type != "cuda" or pibar.device != pi.device:
            raise ValueError("Pi/Pibar residuals must share one CUDA device")
        if not pi.is_contiguous() or not pibar.is_contiguous():
            raise ValueError("Pi/Pibar residuals must be contiguous")
        rows = int(pi.shape[0])
        accumulator_dtype = self.pi_offset.dtype
        if accumulator_dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError("Pi/Pibar offsets must use torch.float32 or torch.float64")
        if pi.dtype == torch.float64 and accumulator_dtype != torch.float64:
            raise TypeError("Pi/Pibar offsets must not be narrower than the residual dtype")
        for name, offset in (
            ("pi_offset", self.pi_offset),
            ("pibar_offset", self.pibar_offset),
        ):
            if offset.ndim != 1 or int(offset.shape[0]) != rows:
                raise ValueError(f"{name} must have shape [{rows}]")
            if offset.dtype != accumulator_dtype:
                raise TypeError(
                    f"{name} must match accumulator dtype {accumulator_dtype}"
                )
            if offset.device != pi.device:
                raise ValueError(f"{name} must be on {pi.device}")
            if not offset.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if pibar_row_max.ndim != 1 or int(pibar_row_max.shape[0]) != rows:
            raise ValueError(f"pibar_row_max must have shape [{rows}]")
        if pibar_row_max.dtype != pi.dtype or pibar_row_max.device != pi.device:
            raise TypeError("pibar_row_max must match the Pi residual dtype and device")
        if not pibar_row_max.is_contiguous():
            raise ValueError("pibar_row_max must be contiguous")
        if check_values:
            # This explicit audit intentionally synchronizes. Production
            # producers call with ``check_values=False`` after structural
            # validation; tests and external/debug consumers get the stronger
            # contract check by default.
            for name, offset in (
                ("pi_offset", self.pi_offset),
                ("pibar_offset", self.pibar_offset),
            ):
                if not bool(torch.isfinite(offset).all().item()):
                    raise ValueError(f"{name} must contain only finite values")
            for name, residual, offset in (
                ("pi_residual", pi, self.pi_offset),
                ("pibar_residual", pibar, self.pibar_offset),
            ):
                valid = torch.isfinite(residual) | torch.isneginf(residual)
                if not bool(valid.all().item()):
                    raise ValueError(f"{name} may contain finite values or -inf only")
                empty = torch.isneginf(residual).all(dim=1)
                if bool((offset[empty] != 0.0).any().item()):
                    raise ValueError(
                        f"all--inf {name} rows must use the canonical zero offset"
                    )

    @staticmethod
    def reconstruct_rows(residual: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Reconstruct rows in the offset accumulator dtype without clamping."""

        if residual.ndim != 2:
            raise ValueError("residual must have shape [rows, species]")
        if residual.dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError("residual must use torch.float32 or torch.float64")
        if offset.ndim != 1 or int(offset.shape[0]) != int(residual.shape[0]):
            raise ValueError("offset must contain one value per residual row")
        if offset.dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError("offset must use torch.float32 or torch.float64")
        if residual.dtype == torch.float64 and offset.dtype != torch.float64:
            raise TypeError("offset must not be narrower than the residual dtype")
        if offset.device != residual.device:
            raise TypeError("offset must be on the residual device")
        return residual.to(dtype=offset.dtype) + offset[:, None]

    def reconstruct_pi(self, pi_residual: torch.Tensor) -> torch.Tensor:
        return self.reconstruct_rows(pi_residual, self.pi_offset)

    def reconstruct_pibar(self, pibar_residual: torch.Tensor) -> torch.Tensor:
        return self.reconstruct_rows(pibar_residual, self.pibar_offset)

    def reconstruct_pibar_row_max(self, pibar_row_max: torch.Tensor) -> torch.Tensor:
        return pibar_row_max.to(dtype=self.pi_offset.dtype) + self.pi_offset
