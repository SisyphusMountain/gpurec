"""Opt-in snapshot and on-the-spot diagnosis of the batch a solver failure happened on.

A failure raised deep inside the HVP or the E-adjoint says nothing about WHICH families and
WHICH rates produced it, and a multi-hour fit cannot be replayed to get back to that iterate.
When a driver opts in with :func:`use_directory`, the failing batch's rates, its family paths and
the solver settings are written to one ``.pt`` so the failure can be replayed on that batch alone,
and a short report says where the first non-finite value is, so the common case (a non-finite
forward state) needs no replay at all.

Nothing here runs unless a driver opts in: the finiteness checks all synchronise with the GPU and
have no place in a production step.
"""
from __future__ import annotations

from pathlib import Path
import time

import torch

# Set by :func:`use_directory`; ``None`` means every entry point below is a no-op.
_DIRECTORY: Path | None = None


def use_directory(directory) -> None:
    """Turn dumping on (a path) or off (``None``). Called by a benchmark/debug driver only."""
    global _DIRECTORY
    if directory is None:
        _DIRECTORY = None
        return
    _DIRECTORY = Path(directory)
    _DIRECTORY.mkdir(parents=True, exist_ok=True)


def is_enabled() -> bool:
    return _DIRECTORY is not None


def nonfinite_summary(name, tensor) -> str | None:
    """``None`` when every entry is finite, else a one-line count of what is wrong."""
    if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return None
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return None
    nan_count = int(torch.isnan(tensor).sum())
    positive_infinity = int((tensor == float("inf")).sum())
    negative_infinity = int((tensor == -float("inf")).sum())
    return (
        f"{name}: {int((~finite).sum())} of {tensor.numel()} non-finite "
        f"(nan {nan_count}, +inf {positive_infinity}, -inf {negative_infinity})"
    )


def describe_forward_state(static, theta_batch, receiver_weights) -> str:
    """Report where the forward solve for one batch turns non-finite, under each self-loop mode.

    ``-inf`` is a legal Pi/Pibar value (an impossible species lane), so it is counted separately
    from ``nan``/``+inf``, which are not. The two modes are run back to back on the same inputs so
    the report says whether the exact self-loop is the source or merely inherits the problem.
    """
    from gpurec.core.inference.solver import solve_resident_e_pi

    lines = []
    original_mode = static.solver_options.forward_self_loop
    for mode in ("log", "exact"):
        static.solver_options.forward_self_loop = mode
        try:
            with torch.no_grad():
                (
                    E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max,
                    log_pS, log_pD, log_pL, max_transfer, receiver_log_probs,
                ) = solve_resident_e_pi(
                    static, theta_batch, receiver_weights,
                    warm_start_E=None, pi_iters=None, pi_residual_out=None,
                )
        except Exception as failure:  # noqa: BLE001 - a diagnosis must not mask the real error
            lines.append(f"  [{mode}] forward raised {type(failure).__name__}: {failure}")
            continue
        state = static.pi_forward_state
        named = {
            "E": E, "E_s1": E_s1, "E_s2": E_s2, "Ebar": Ebar,
            "log_pS": log_pS, "log_pD": log_pD, "log_pL": log_pL,
            "max_transfer": max_transfer, "receiver_log_probs": receiver_log_probs,
            "root_rows": root_rows, "Pi_residual": pi_wave, "Pibar_residual": pibar_wave,
            "pibar_row_max": pibar_row_max,
            "Pi_offset": None if state is None else state.pi_offset,
            "Pibar_offset": None if state is None else state.pibar_offset,
        }
        problems = [text for text in (nonfinite_summary(n, t) for n, t in named.items()) if text]
        # A whole Pi row of -inf is the shape a scaled-linear-space underflow takes: every lane
        # of that clade lost its mass at once, which the log path cannot do. Count it separately
        # from single impossible lanes, which are ordinary.
        dead_pi = int((~torch.isfinite(pi_wave)).all(dim=1).sum())
        dead_pibar = int((~torch.isfinite(pibar_wave)).all(dim=1).sum())
        problems.append(
            f"all--inf rows: Pi {dead_pi}/{pi_wave.shape[0]}, Pibar {dead_pibar}/{pibar_wave.shape[0]}"
        )
        lines.append(f"  [{mode}] " + "; ".join(problems))
        del E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max
        torch.cuda.empty_cache()
    static.solver_options.forward_self_loop = original_mode
    return "\n".join(lines)


def save_batch(static, theta_batch, receiver_weights, *, species_tree, family_paths, reason,
               extra) -> str | None:
    """Write one batch's replay inputs; return the file path, or ``None`` when dumping is off."""
    if _DIRECTORY is None:
        return None
    indices = [int(index) for index in static.family_indices]
    payload = {
        "reason": reason,
        "species_tree": str(species_tree),
        "family_paths": [str(family_paths[index]) for index in indices],
        "family_indices": indices,
        "theta_batch": theta_batch.detach().cpu(),
        "receiver_weights": receiver_weights.detach().cpu(),
        "solver_options": vars(static.solver_options).copy(),
        "genewise": bool(static.genewise),
        "specieswise": bool(static.specieswise),
        "accumulator_dtype": str(static.accumulator_dtype),
        "model_dtype": str(static.precision_options.model_dtype),
    }
    payload.update(extra)
    path = _DIRECTORY / f"{reason}-{time.strftime('%Y%m%d-%H%M%S')}-{len(indices)}fam.pt"
    torch.save(payload, path)
    return str(path)
