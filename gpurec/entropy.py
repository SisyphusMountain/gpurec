"""Analytical entropy for gpurec reconciliation distributions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Literal

import torch

from gpurec.backtracking import export_backtracking_input
from gpurec.core.log2_utils import logsumexp2
from gpurec.core.species import species_wave_topology

__all__ = [
    "compute_reconciliation_entropy",
    "reconciliation_entropy_from_payload",
]


EntropyMode = Literal["collapsed", "expanded", "both"]

_NEG_INF_CUTOFF = -1.0e290


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be positive")
    return out


def _nonnegative_float(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return out


def _as_log_matrix(
    payload_matrix: Mapping[str, Any],
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    rows = int(payload_matrix["rows"])
    cols = int(payload_matrix["cols"])
    data = torch.as_tensor(payload_matrix["data"], dtype=dtype, device=device)
    expected = rows * cols
    if int(data.numel()) != expected:
        raise ValueError(f"{name} has {int(data.numel())} values, expected {expected}")
    return data.reshape(rows, cols).contiguous()


def _as_log_vector(
    values: Any,
    *,
    name: str,
    length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    out = torch.as_tensor(values, dtype=dtype, device=device)
    if out.ndim != 1 or int(out.numel()) != int(length):
        raise ValueError(f"{name} must have shape [{length}], got {tuple(out.shape)}")
    return out.contiguous()


def _as_index_vector(
    values: Any,
    *,
    name: str,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    out = torch.as_tensor(values, dtype=torch.long, device=device)
    if out.ndim != 1 or int(out.numel()) != int(length):
        raise ValueError(f"{name} must have shape [{length}], got {tuple(out.shape)}")
    return out.contiguous()


def _finite_log_weights(
    values: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    out = values
    if valid is not None:
        out = torch.where(valid, out, torch.full_like(out, float("-inf")))
    return torch.where(out > _NEG_INF_CUTOFF, out, torch.full_like(out, float("-inf")))


def _masked_logsumexp2(values: torch.Tensor, dim: int) -> torch.Tensor:
    return logsumexp2(_finite_log_weights(values), dim=dim)


def _weighted_child_entropy(
    log_weights: torch.Tensor,
    child_entropy: torch.Tensor,
    log_norm: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    """Return ``sum_i p_i * (-log2 p_i + child_entropy_i)``.

    ``log_norm`` is the precomputed log partition for the same candidates.
    Entries below the gpurec negative-infinity sentinel are ignored.
    """

    norm = log_norm.unsqueeze(dim)
    valid = (log_weights > _NEG_INF_CUTOFF) & (norm > _NEG_INF_CUTOFF)
    log_p = torch.where(valid, log_weights - norm, torch.zeros_like(log_weights))
    probs = torch.where(valid, torch.exp2(log_p), torch.zeros_like(log_weights))
    surprisal = torch.where(valid, norm - log_weights, torch.zeros_like(log_weights))
    child = torch.where(valid, child_entropy, torch.zeros_like(child_entropy))
    return (probs * (surprisal + child)).sum(dim=dim)


def _recipient_mask_from_parent(parent: torch.Tensor) -> torch.Tensor:
    s_count = int(parent.numel())
    ancestors = torch.zeros((s_count, s_count), dtype=torch.bool, device=parent.device)
    parent_cpu = parent.detach().cpu().tolist()
    for start in range(s_count):
        cur = start
        seen: set[int] = set()
        while cur >= 0:
            if cur in seen:
                raise ValueError("species parent pointers contain a cycle")
            seen.add(cur)
            ancestors[start, cur] = True
            cur = int(parent_cpu[cur])
    return ~ancestors


def _pibar_entropy(
    *,
    pi: torch.Tensor,
    pibar: torch.Tensor,
    max_transfer: torch.Tensor,
    child_entropy: torch.Tensor,
    recipient_mask: torch.Tensor,
) -> torch.Tensor:
    c_count, s_count = pi.shape
    out = torch.zeros((c_count, s_count), dtype=pi.dtype, device=pi.device)
    for donor in range(s_count):
        recipients = torch.nonzero(recipient_mask[donor], as_tuple=False).flatten()
        if int(recipients.numel()) == 0:
            continue
        log_weights = pi.index_select(1, recipients) + max_transfer[donor]
        child = child_entropy.index_select(1, recipients)
        out[:, donor] = _weighted_child_entropy(
            log_weights,
            child,
            pibar[:, donor],
            dim=1,
        )
    return out


def _ebar_entropy(
    *,
    e: torch.Tensor,
    ebar: torch.Tensor,
    max_transfer: torch.Tensor,
    h_e: torch.Tensor,
    recipient_mask: torch.Tensor,
) -> torch.Tensor:
    s_count = int(e.numel())
    out = torch.zeros((s_count,), dtype=e.dtype, device=e.device)
    for donor in range(s_count):
        recipients = torch.nonzero(recipient_mask[donor], as_tuple=False).flatten()
        if int(recipients.numel()) == 0:
            continue
        log_weights = e.index_select(0, recipients) + max_transfer[donor]
        child = h_e.index_select(0, recipients)
        out[donor] = _weighted_child_entropy(
            log_weights,
            child,
            ebar[donor],
            dim=0,
        )
    return out


def _solve_extinction_entropy(
    *,
    e: torch.Tensor,
    ebar: torch.Tensor,
    log_p_s: torch.Tensor,
    log_p_d: torch.Tensor,
    log_p_l: torch.Tensor,
    max_transfer: torch.Tensor,
    child1: torch.Tensor,
    child2: torch.Tensor,
    recipient_mask: torch.Tensor,
    tol: float,
    max_iters: int,
) -> tuple[torch.Tensor, int, float, bool]:
    h_e = torch.zeros_like(e)
    internal = (child1 >= 0) & (child2 >= 0)
    c1 = child1.clamp_min(0)
    c2 = child2.clamp_min(0)

    for iteration in range(1, max_iters + 1):
        h_ebar = _ebar_entropy(
            e=e,
            ebar=ebar,
            max_transfer=max_transfer,
            h_e=h_e,
            recipient_mask=recipient_mask,
        )
        log_terms = [
            log_p_l,
            log_p_d + 2.0 * e,
            e + ebar,
        ]
        child_terms = [
            torch.zeros_like(e),
            2.0 * h_e,
            h_e + h_ebar,
        ]
        spec_log = torch.full_like(e, float("-inf"))
        spec_child = torch.zeros_like(e)
        if bool(internal.any().item()):
            spec_log = torch.where(
                internal,
                log_p_s + e.index_select(0, c1) + e.index_select(0, c2),
                spec_log,
            )
            spec_child = torch.where(
                internal,
                h_e.index_select(0, c1) + h_e.index_select(0, c2),
                spec_child,
            )
        log_terms.append(spec_log)
        child_terms.append(spec_child)

        logs = torch.stack(log_terms, dim=0)
        children = torch.stack(child_terms, dim=0)
        h_new = _weighted_child_entropy(logs, children, e, dim=0)
        delta = float(torch.max(torch.abs(h_new - h_e)).detach().cpu())
        h_e = h_new
        if delta <= tol:
            return h_e, iteration, delta, True
    return h_e, max_iters, delta, False


def _split_groups(
    splits: list[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[int, list[tuple[int, int, torch.Tensor]]]:
    groups: dict[int, list[tuple[int, int, torch.Tensor]]] = defaultdict(list)
    for split in splits:
        parent = int(split["parent"])
        left = int(split["left"])
        right = int(split["right"])
        log_prob = torch.tensor(float(split["log_prob"]), dtype=torch.float64, device=device)
        groups[parent].append((left, right, log_prob))
    return groups


def _solve_pi_entropy(
    *,
    pi: torch.Tensor,
    pibar: torch.Tensor,
    e: torch.Tensor,
    ebar: torch.Tensor,
    log_p_s: torch.Tensor,
    log_p_d: torch.Tensor,
    max_transfer: torch.Tensor,
    leaf_species: torch.Tensor,
    split_groups: Mapping[int, list[tuple[int, int, torch.Tensor]]],
    child1: torch.Tensor,
    child2: torch.Tensor,
    recipient_mask: torch.Tensor,
    h_e: torch.Tensor | None,
    h_ebar_fixed: torch.Tensor | None,
    expand_extinction: bool,
    tol: float,
    max_iters: int,
) -> tuple[torch.Tensor, int, float, bool]:
    c_count, s_count = pi.shape
    h_pi = torch.zeros_like(pi)
    internal = (child1 >= 0) & (child2 >= 0)
    c1 = child1.clamp_min(0)
    c2 = child2.clamp_min(0)
    zero_s = torch.zeros((s_count,), dtype=pi.dtype, device=pi.device)
    h_e_vec = torch.zeros_like(e) if h_e is None else h_e
    h_ebar = torch.zeros_like(ebar) if h_ebar_fixed is None else h_ebar_fixed

    for iteration in range(1, max_iters + 1):
        h_pibar = _pibar_entropy(
            pi=pi,
            pibar=pibar,
            max_transfer=max_transfer,
            child_entropy=h_pi,
            recipient_mask=recipient_mask,
        )
        h_new = torch.zeros_like(h_pi)

        for clade in range(c_count):
            pi_row = pi[clade]
            log_terms = [
                1.0 + log_p_d + e + pi_row,
                pi_row + ebar,
                pibar[clade] + e,
            ]
            child_terms = [
                h_pi[clade] + (1.0 + h_e_vec if expand_extinction else 0.0),
                h_pi[clade] + (h_ebar if expand_extinction else 0.0),
                h_pibar[clade] + (h_e_vec if expand_extinction else 0.0),
            ]

            spec_left_log = torch.full_like(pi_row, float("-inf"))
            spec_right_log = torch.full_like(pi_row, float("-inf"))
            spec_left_child = zero_s
            spec_right_child = zero_s
            if bool(internal.any().item()):
                child1_pi = pi[clade].index_select(0, c1)
                child2_pi = pi[clade].index_select(0, c2)
                spec_left_log = torch.where(
                    internal,
                    log_p_s + e.index_select(0, c2) + child1_pi,
                    spec_left_log,
                )
                spec_right_log = torch.where(
                    internal,
                    log_p_s + e.index_select(0, c1) + child2_pi,
                    spec_right_log,
                )
                spec_left_child = torch.where(
                    internal,
                    h_pi[clade].index_select(0, c1)
                    + (h_e_vec.index_select(0, c2) if expand_extinction else 0.0),
                    zero_s,
                )
                spec_right_child = torch.where(
                    internal,
                    h_pi[clade].index_select(0, c2)
                    + (h_e_vec.index_select(0, c1) if expand_extinction else 0.0),
                    zero_s,
                )
            log_terms.extend([spec_left_log, spec_right_log])
            child_terms.extend([spec_left_child, spec_right_child])

            leaf_log = torch.full_like(pi_row, float("-inf"))
            leaf = int(leaf_species[clade])
            if leaf >= 0:
                leaf_log[leaf] = log_p_s[leaf]
            log_terms.append(leaf_log)
            child_terms.append(zero_s)

            for left, right, split_log_prob in split_groups.get(clade, []):
                split_log = split_log_prob.to(dtype=pi.dtype)
                left_pi = pi[left]
                right_pi = pi[right]
                left_h = h_pi[left]
                right_h = h_pi[right]
                left_pibar_h = h_pibar[left]
                right_pibar_h = h_pibar[right]

                log_terms.append(split_log + log_p_d + left_pi + right_pi)
                child_terms.append(left_h + right_h)
                log_terms.append(split_log + left_pi + pibar[right])
                child_terms.append(left_h + right_pibar_h)
                log_terms.append(split_log + right_pi + pibar[left])
                child_terms.append(right_h + left_pibar_h)

                split_spec_left_log = torch.full_like(pi_row, float("-inf"))
                split_spec_right_log = torch.full_like(pi_row, float("-inf"))
                split_spec_left_child = zero_s
                split_spec_right_child = zero_s
                if bool(internal.any().item()):
                    left_c1 = left_pi.index_select(0, c1)
                    left_c2 = left_pi.index_select(0, c2)
                    right_c1 = right_pi.index_select(0, c1)
                    right_c2 = right_pi.index_select(0, c2)
                    split_spec_left_log = torch.where(
                        internal,
                        split_log + log_p_s + left_c1 + right_c2,
                        split_spec_left_log,
                    )
                    split_spec_right_log = torch.where(
                        internal,
                        split_log + log_p_s + right_c1 + left_c2,
                        split_spec_right_log,
                    )
                    split_spec_left_child = torch.where(
                        internal,
                        left_h.index_select(0, c1) + right_h.index_select(0, c2),
                        zero_s,
                    )
                    split_spec_right_child = torch.where(
                        internal,
                        right_h.index_select(0, c1) + left_h.index_select(0, c2),
                        zero_s,
                    )
                log_terms.extend([split_spec_left_log, split_spec_right_log])
                child_terms.extend([split_spec_left_child, split_spec_right_child])

            logs = torch.stack(log_terms, dim=0)
            children = torch.stack(child_terms, dim=0)
            h_new[clade] = _weighted_child_entropy(logs, children, pi_row, dim=0)

        delta = float(torch.max(torch.abs(h_new - h_pi)).detach().cpu())
        h_pi = h_new
        if delta <= tol:
            return h_pi, iteration, delta, True
    return h_pi, max_iters, delta, False


def _root_entropy(
    *,
    pi: torch.Tensor,
    root_clade: int,
    h_root_states: torch.Tensor,
    origination_probs: Any,
) -> torch.Tensor:
    s_count = int(pi.shape[1])
    if origination_probs is None:
        log_prior = torch.full(
            (s_count,),
            -math.log2(s_count),
            dtype=pi.dtype,
            device=pi.device,
        )
    else:
        probs = torch.as_tensor(origination_probs, dtype=pi.dtype, device=pi.device)
        if probs.ndim != 1 or int(probs.numel()) != s_count:
            raise ValueError(
                f"origination_probs must have shape [{s_count}], got {tuple(probs.shape)}"
            )
        log_prior = torch.where(
            probs > 0.0,
            torch.log2(torch.where(probs > 0.0, probs, torch.ones_like(probs))),
            torch.full_like(probs, float("-inf")),
        )
    log_weights = log_prior + pi[root_clade]
    log_norm = _masked_logsumexp2(log_weights, dim=0)
    return _weighted_child_entropy(log_weights, h_root_states, log_norm, dim=0)


def _species_tensors_from_model(
    model: Any,
    *,
    s_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        species_helpers = model.static.species_helpers
    except AttributeError as exc:
        raise ValueError(
            "compute_reconciliation_entropy requires a GeneReconModel with "
            "static.species_helpers; use reconciliation_entropy_from_payload "
            "when supplying topology arrays manually"
        ) from exc
    topology = species_wave_topology(species_helpers, device="cpu", S=s_count)
    child1 = topology["sp_child1_cpu"].to(device=device, dtype=torch.long)
    child2 = topology["sp_child2_cpu"].to(device=device, dtype=torch.long)
    parent = topology["sp_parent_cpu"].to(device=device, dtype=torch.long)
    return child1, child2, parent


def reconciliation_entropy_from_payload(
    payload: Mapping[str, Any],
    *,
    species_child1: Any,
    species_child2: Any,
    species_parent: Any,
    mode: EntropyMode = "both",
    tol: float = 1e-10,
    max_iters: int = 10_000,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[str, float | int | bool | str]:
    """Compute analytical reconciliation entropy from a backtracking payload.

    ``payload`` is the dictionary produced by :func:`export_backtracking_input`.
    Species topology arrays must use the same postorder species indexing as the
    payload.  ``species_child1`` and ``species_child2`` use ``-1`` for leaves;
    ``species_parent`` uses ``-1`` for the species-tree root.
    """

    if mode not in ("collapsed", "expanded", "both"):
        raise ValueError("mode must be 'collapsed', 'expanded', or 'both'")
    tol = _nonnegative_float("tol", tol)
    max_iters = _positive_int("max_iters", max_iters)
    target_device = torch.device("cpu" if device is None else device)

    pi = _as_log_matrix(payload["pi"], name="pi", dtype=dtype, device=target_device)
    pibar = _as_log_matrix(payload["pibar"], name="pibar", dtype=dtype, device=target_device)
    if tuple(pibar.shape) != tuple(pi.shape):
        raise ValueError(f"pibar shape {tuple(pibar.shape)} must match pi {tuple(pi.shape)}")
    c_count, s_count = map(int, pi.shape)
    e = _as_log_vector(payload["e"], name="e", length=s_count, dtype=dtype, device=target_device)
    ebar = _as_log_vector(
        payload["ebar"],
        name="ebar",
        length=s_count,
        dtype=dtype,
        device=target_device,
    )
    log_p_s = _as_log_vector(
        payload["log_p_s"],
        name="log_p_s",
        length=s_count,
        dtype=dtype,
        device=target_device,
    )
    log_p_d = _as_log_vector(
        payload["log_p_d"],
        name="log_p_d",
        length=s_count,
        dtype=dtype,
        device=target_device,
    )
    if "log_p_l" in payload:
        log_p_l = _as_log_vector(
            payload["log_p_l"],
            name="log_p_l",
            length=s_count,
            dtype=dtype,
            device=target_device,
        )
    elif mode in ("expanded", "both"):
        raise ValueError("expanded entropy requires payload['log_p_l']")
    else:
        log_p_l = torch.full(
            (s_count,),
            float("-inf"),
            dtype=dtype,
            device=target_device,
        )
    max_transfer = _as_log_vector(
        payload["max_transfer"],
        name="max_transfer",
        length=s_count,
        dtype=dtype,
        device=target_device,
    )
    child1 = _as_index_vector(
        species_child1,
        name="species_child1",
        length=s_count,
        device=target_device,
    )
    child2 = _as_index_vector(
        species_child2,
        name="species_child2",
        length=s_count,
        device=target_device,
    )
    child1 = torch.where(
        (child1 >= 0) & (child1 < s_count),
        child1,
        torch.full_like(child1, -1),
    )
    child2 = torch.where(
        (child2 >= 0) & (child2 < s_count),
        child2,
        torch.full_like(child2, -1),
    )
    parent = _as_index_vector(
        species_parent,
        name="species_parent",
        length=s_count,
        device=target_device,
    )
    leaf_species_values = [
        -1 if value is None else int(value) for value in payload["leaf_species"]
    ]
    leaf_species = _as_index_vector(
        leaf_species_values,
        name="leaf_species",
        length=c_count,
        device=target_device,
    )
    root_clade = int(payload["root_clade"])
    if root_clade < 0 or root_clade >= c_count:
        raise ValueError(f"root_clade {root_clade} outside 0..{c_count}")

    split_groups = _split_groups(list(payload["splits"]), device=target_device)
    recipient_mask = _recipient_mask_from_parent(parent)

    result: dict[str, float | int | bool | str] = {"mode": mode}

    if mode in ("collapsed", "both"):
        h_pi, iterations, delta, converged = _solve_pi_entropy(
            pi=pi,
            pibar=pibar,
            e=e,
            ebar=ebar,
            log_p_s=log_p_s,
            log_p_d=log_p_d,
            max_transfer=max_transfer,
            leaf_species=leaf_species,
            split_groups=split_groups,
            child1=child1,
            child2=child2,
            recipient_mask=recipient_mask,
            h_e=None,
            h_ebar_fixed=None,
            expand_extinction=False,
            tol=tol,
            max_iters=max_iters,
        )
        root_h = _root_entropy(
            pi=pi,
            root_clade=root_clade,
            h_root_states=h_pi[root_clade],
            origination_probs=payload.get("origination_probs"),
        )
        result.update(
            {
                "collapsed_bits": float(root_h.detach().cpu()),
                "collapsed_pi_iterations": iterations,
                "collapsed_pi_delta": delta,
                "collapsed_converged": converged,
            }
        )

    if mode in ("expanded", "both"):
        h_e, e_iterations, e_delta, e_converged = _solve_extinction_entropy(
            e=e,
            ebar=ebar,
            log_p_s=log_p_s,
            log_p_d=log_p_d,
            log_p_l=log_p_l,
            max_transfer=max_transfer,
            child1=child1,
            child2=child2,
            recipient_mask=recipient_mask,
            tol=tol,
            max_iters=max_iters,
        )
        h_ebar = _ebar_entropy(
            e=e,
            ebar=ebar,
            max_transfer=max_transfer,
            h_e=h_e,
            recipient_mask=recipient_mask,
        )
        h_pi, iterations, delta, converged = _solve_pi_entropy(
            pi=pi,
            pibar=pibar,
            e=e,
            ebar=ebar,
            log_p_s=log_p_s,
            log_p_d=log_p_d,
            max_transfer=max_transfer,
            leaf_species=leaf_species,
            split_groups=split_groups,
            child1=child1,
            child2=child2,
            recipient_mask=recipient_mask,
            h_e=h_e,
            h_ebar_fixed=h_ebar,
            expand_extinction=True,
            tol=tol,
            max_iters=max_iters,
        )
        root_h = _root_entropy(
            pi=pi,
            root_clade=root_clade,
            h_root_states=h_pi[root_clade],
            origination_probs=payload.get("origination_probs"),
        )
        result.update(
            {
                "expanded_bits": float(root_h.detach().cpu()),
                "expanded_pi_iterations": iterations,
                "expanded_pi_delta": delta,
                "expanded_pi_converged": converged,
                "e_iterations": e_iterations,
                "e_delta": e_delta,
                "e_converged": e_converged,
            }
        )

    return result


def compute_reconciliation_entropy(
    model: Any,
    *,
    family_index: int = 0,
    mode: EntropyMode = "both",
    tol: float = 1e-10,
    max_iters: int = 10_000,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[str, float | int | bool | str]:
    """Compute analytical entropy for one family of a solved gpurec model.

    The ``collapsed`` result is over the same distribution sampled by gpurec's
    stochastic backtracker.  The ``expanded`` result also includes entropy from
    hidden extinction histories represented by ``E`` and ``Ebar``.
    """

    target_device = torch.device("cpu" if device is None else device)
    payload = export_backtracking_input(model, family_index=family_index)
    s_count = int(payload["pi"]["cols"])
    child1, child2, parent = _species_tensors_from_model(
        model,
        s_count=s_count,
        dtype=dtype,
        device=target_device,
    )
    return reconciliation_entropy_from_payload(
        payload,
        species_child1=child1,
        species_child2=child2,
        species_parent=parent,
        mode=mode,
        tol=tol,
        max_iters=max_iters,
        dtype=dtype,
        device=target_device,
    )
