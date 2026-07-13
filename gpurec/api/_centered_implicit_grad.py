"""First-order centered-Pi adjoint correctness bridge.

This adapter reconstructs centered Pi/Pibar state in fp64 and delegates to the
retained implicit adjoint.  It exists to unblock end-to-end gradient parity and
to provide an oracle for native offset-aware kernels; it is not intended to
meet the centered path's runtime or memory budget.
"""

from __future__ import annotations

import torch

from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.kernels.centered_pi_backward import (
    prepare_centered_backward_reference,
)


@torch.no_grad()
def implicit_grad_loglik_vjp_wave_centered_reference(
    wave_layout,
    species_helpers,
    *,
    Pi_star_wave: torch.Tensor,
    Pibar_star_wave: torch.Tensor,
    uniform_pibar_row_max: torch.Tensor,
    centered_pi_state=None,
    pi_offset: torch.Tensor | None = None,
    pibar_offset: torch.Tensor | None = None,
    **kwargs,
):
    """Run the retained first-order adjoint from centered forward state.

    The three primal keyword names intentionally match
    :func:`implicit_grad_loglik_vjp_wave`, making the adapter a narrow call-site
    substitution.  ``centered_pi_state`` must expose fp64 ``pi_offset`` and
    ``pibar_offset`` tensors, as ``CenteredPiForwardState`` does.
    """

    if centered_pi_state is not None:
        if pi_offset is not None or pibar_offset is not None:
            raise ValueError(
                "pass either centered_pi_state or explicit offsets, not both"
            )
        try:
            pi_offset = centered_pi_state.pi_offset
            pibar_offset = centered_pi_state.pibar_offset
        except AttributeError as exc:
            raise TypeError(
                "centered_pi_state must expose pi_offset and pibar_offset"
            ) from exc
    if pi_offset is None or pibar_offset is None:
        raise ValueError(
            "centered_pi_state or both pi_offset and pibar_offset are required"
        )

    absolute = prepare_centered_backward_reference(
        Pi_star_wave,
        Pibar_star_wave,
        pi_offset,
        pibar_offset,
        uniform_pibar_row_max,
    )

    # The retained kernels choose their arithmetic dtype from Pi.  Passing fp64
    # Pi with fp32 rate/E pointers leaves several Triton loop-carried
    # expressions with mixed types and, even where implicit promotion works,
    # does not provide a coherent fp64 reference.  Promote every floating
    # primal/head input consumed by the implicit adjoint.  Integer topology and
    # family-index tensors are intentionally left unchanged.
    fp64_kwargs = dict(kwargs)
    for name in (
        "E_star",
        "Ebar",
        "E_s1",
        "E_s2",
        "log_pS",
        "log_pD",
        "log_pL",
        "max_transfer_mat",
        "receiver_log_probs",
        "theta",
        "receiver_weights",
        "seed_root",
        "origination_log_probs",
        "origination_probs",
    ):
        value = fp64_kwargs.get(name)
        if torch.is_tensor(value) and value.is_floating_point():
            fp64_kwargs[name] = value.to(dtype=torch.float64)

    warm_v = fp64_kwargs.get("warm_v")
    if warm_v is not None:
        fp64_kwargs["warm_v"] = {
            key: value.to(dtype=torch.float64)
            for key, value in warm_v.items()
        }
    return implicit_grad_loglik_vjp_wave(
        wave_layout,
        species_helpers,
        Pi_star_wave=absolute.pi_absolute,
        Pibar_star_wave=absolute.pibar_absolute,
        uniform_pibar_row_max=absolute.pibar_row_max_absolute,
        **fp64_kwargs,
    )
