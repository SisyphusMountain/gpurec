"""Forward pass: Pi_wave_forward and helpers."""

from dataclasses import dataclass

import torch

from .kernels.wave_step import (
    wave_step_uniform_fused_into,
    wave_step_uniform_initial_leaf_fused_into,
    wave_pibar_uniform_parent_fused,
)
from .kernels.dts_fused import dts_fused_parent_reduced
from ._helpers import _nvtx_range
from .extract_parameters import as_family_param, as_family_species
from .log2_utils import logsumexp2
from .species import species_wave_topology

NEG_INF = float("-inf")


@dataclass(frozen=True)
class _PiOutputIntent:
    """Resolved private output contract for ``Pi_wave_forward``."""

    name: str
    emit_original_pi: bool
    emit_root_rows: bool
    emit_wave_ordered_pi: bool
    retain_saved_state: bool

    @property
    def can_skip_final_pibar(self) -> bool:
        return not self.retain_saved_state


def _pi_output_intent(
    *,
    return_original: bool,
    return_root_rows: bool,
) -> _PiOutputIntent:
    """Map legacy output booleans onto an explicit internal output contract."""
    if return_root_rows:
        return _PiOutputIntent(
            name=(
                "legacy_original_root_rows"
                if return_original
                else "root_row_loss_only"
            ),
            emit_original_pi=return_original,
            emit_root_rows=True,
            emit_wave_ordered_pi=False,
            retain_saved_state=False,
        )
    if return_original:
        return _PiOutputIntent(
            name="export_original_and_wave_rows",
            emit_original_pi=True,
            emit_root_rows=False,
            emit_wave_ordered_pi=True,
            retain_saved_state=True,
        )
    return _PiOutputIntent(
        name="training_or_wave_export_state",
        emit_original_pi=False,
        emit_root_rows=False,
        emit_wave_ordered_pi=True,
        retain_saved_state=True,
    )


@dataclass(frozen=True)
class _PiForwardRequest:
    """Intent-named internal request for invoking ``Pi_wave_forward``."""

    intent: _PiOutputIntent

    def run(self, **forward_kwargs):
        return Pi_wave_forward(
            **forward_kwargs,
            return_original=self.intent.emit_original_pi,
            return_root_rows=self.intent.emit_root_rows,
        )


def pi_training_state_request() -> _PiForwardRequest:
    """Return wave-ordered Pi/Pibar state for likelihood plus gradients."""
    return _PiForwardRequest(
        _pi_output_intent(return_original=False, return_root_rows=False)
    )


def pi_root_row_loss_request() -> _PiForwardRequest:
    """Return root rows only for no-gradient loss evaluation."""
    return _PiForwardRequest(
        _pi_output_intent(return_original=False, return_root_rows=True)
    )


def pi_export_state_request(*, original_order: bool) -> _PiForwardRequest:
    """Return export Pi state in the caller-selected clade order."""
    return _PiForwardRequest(
        _pi_output_intent(
            return_original=original_order,
            return_root_rows=False,
        )
    )


# ---------------------------------------------------------------------------
# Cross-clade DTS
# ---------------------------------------------------------------------------

def _compute_dts_cross(Pi, Pibar, meta, sp_child1, sp_child2, log_pD, log_pS,
                       S, device, dtype, active_mask=None,
                       family_idx=None,
                       family_offset=0):
    """Compute DTS cross-clade terms and reduce to [W, S] for one wave."""
    sl = meta['sl']
    sr = meta['sr']
    wlsp = meta['log_split_probs']
    W = meta['W']
    n_eq1 = meta.get('n_eq1', 0)
    ge2_parent_ids = meta.get('ge2_parent_ids', sl[:0])
    covered_parent_rows = int(n_eq1) + int(ge2_parent_ids.numel())
    initialize_out = active_mask is not None or covered_parent_rows != int(W)
    return dts_fused_parent_reduced(
        Pi, Pibar, sl, sr,
        sp_child1, sp_child2,
        log_pD, log_pS, wlsp,
        W,
        n_eq1,
        meta.get('eq1_reduce_idx', sl[:0]),
        meta.get('ge2_ptr', sl.new_zeros((1,), dtype=torch.long)),
        ge2_parent_ids,
        active_mask=active_mask,
        family_idx=family_idx,
        family_offset=family_offset,
        tile_splits=64,
        ge2_max_fanout=meta.get('ge2_max_fanout'),
        initialize_out=initialize_out,
    )


# ---------------------------------------------------------------------------
# Pi wave forward
# ---------------------------------------------------------------------------

def Pi_wave_forward(
    wave_layout,
    species_helpers,
    E,
    Ebar,
    E_s1,
    E_s2,
    log_pS,
    log_pD,
    max_transfer_mat,
    device,
    dtype,
    *,
    fixed_iters: int,
    family_idx: torch.Tensor | None = None,
    return_original: bool = True,
    return_root_rows: bool = False,
    progress_callback=None,
    trace_root_logsumexp: bool = False,
    convergence_tolerance: float = -1.0,
    convergence_check_interval: int = 4,
):
    """Wave-based Pi forward pass with wave-ordered layout (v2).

    Clades are permuted so each wave occupies a contiguous block of Pi[ws:we].
    The self-loop uses zero-copy views instead of gather/scatter.

    Args:
        wave_layout: dict from build_wave_layout() containing permuted indices
                     and precomputed per-wave metadata
        species_helpers: species tree helpers dict
        E, Ebar, E_s1, E_s2: converged E vectors [S] or [G, S]
        log_pS, log_pD: event probabilities (scalar, [S], [G], or [G, S])
        max_transfer_mat: [S] or [G, S] log2-space uniform transfer maxima
        device, dtype: target device and float dtype
        fixed_iters: fixed self-loop iteration count. Must be even so the
            ping-pong state ends with final Pi rows in ``Pi``.
        family_idx: Long[C] clade→family mapping in wave-ordered space.
                    Public model callers pass normalized [G, 1] or [G, S]
                    parameters for genewise mode.  Direct callers should pass
                    [G, 1] or [G, S] rather than bare [G] when G == S so direct
                    forward/backward DTS paths cannot disagree on whether a
                    one-dimensional tensor is species-indexed or family-indexed.
        return_root_rows: if True, gather and return only final root rows as
                          ``Pi_root_rows`` and drop the full wave-ordered Pi
                          reference from the output. This is for inference-only
                          likelihood callers and skips saved Pibar state.
        trace_root_logsumexp: if True, record a GPU-resident
                              ``[fixed_iters, n_roots]`` trace of base-2
                              logsumexp values for root rows.
        convergence_tolerance: when non-negative, treat ``fixed_iters`` as a
                               maximum and stop a wave early when the max row
                               update changes by less than this value at a
                               convergence check.
        convergence_check_interval: check every N local self-loop iterations.
                                    Must be even when convergence is enabled so
                                    final Pi rows remain in ``Pi``.

    Returns:
        dict with 'Pi' (in original clade order when requested),
        'Pi_root_rows' when requested.
    """
    output_intent = _pi_output_intent(
        return_original=return_original,
        return_root_rows=return_root_rows,
    )
    leaf_row_index = wave_layout['leaf_row_index']
    leaf_species_index = wave_layout.get('leaf_species_index')
    wave_metas = wave_layout['wave_metas']

    C = int(wave_layout['C'])
    S = int(species_helpers['S'])
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("The lean forward path requires CUDA.")
    if leaf_species_index is None:
        raise ValueError("The lean forward path requires leaf_species_index in the wave layout.")
    if fixed_iters < 1 or fixed_iters % 2 != 0:
        raise ValueError("The lean forward path requires a positive even fixed_iters value.")
    if convergence_check_interval < 1:
        raise ValueError("convergence_check_interval must be positive")
    if convergence_tolerance >= 0.0 and convergence_check_interval % 2 != 0:
        raise ValueError(
            "Pi convergence checks require an even convergence_check_interval"
        )

    with _nvtx_range("Pi setup tensors"):
        Pi = torch.empty((C, S), dtype=dtype, device=device)
        # Pibar is a ping-pong scratch/output buffer. Each row is written by
        # its wave before any later DTS wave can read it, so an initial full
        # tensor fill only burns memory bandwidth on large tree batches.
        Pibar = torch.empty((C, S), dtype=dtype, device=device)

    batched = family_idx is not None
    if batched:
        family_idx = family_idx.to(device=device, dtype=torch.long).contiguous()

    if batched:
        family_rows = int(E.shape[0]) if E.ndim == 2 else None
        param_kwargs = dict(S=S, device=target_device, dtype=dtype, family_rows=family_rows)
        E_family = as_family_species(E, name="E", **param_kwargs)
        Ebar_family = as_family_species(Ebar, name="Ebar", **param_kwargs)
        E_s1_family = as_family_species(E_s1, name="E_s1", **param_kwargs)
        E_s2_family = as_family_species(E_s2, name="E_s2", **param_kwargs)
        mt_family = as_family_species(
            max_transfer_mat.squeeze(-1), name="max_transfer_mat", **param_kwargs
        )
        log_pD_param = as_family_param(log_pD, name="log_pD", **param_kwargs)
        log_pS_param = as_family_param(log_pS, name="log_pS", **param_kwargs)
        log_pD_family = as_family_species(log_pD, name="log_pD", **param_kwargs)
        log_pS_family = as_family_species(log_pS, name="log_pS", **param_kwargs)
    else:
        E_family = Ebar_family = E_s1_family = E_s2_family = None
        mt_family = None
        log_pD_param = log_pD
        log_pS_param = log_pS
        log_pD_family = log_pD
        log_pS_family = log_pS

    uniform_pibar_row_max = (
        torch.empty((C,), dtype=dtype, device=device)
        if output_intent.retain_saved_state else None
    )
    max_wave_W = max((int(meta.get('W', 0)) for meta in wave_metas), default=0)
    max_diff_scratch = (
        torch.empty((max_wave_W,), dtype=dtype, device=device)
        if convergence_tolerance >= 0.0 and max_wave_W > 0
        else None
    )

    with _nvtx_range("Pi setup species helpers"):
        species_topology = species_wave_topology(species_helpers, S=S, device=device)
        sp_child1 = species_topology["sp_child1"]
        sp_child2 = species_topology["sp_child2"]
        sp_parent = species_topology["sp_parent"]
        sp_subtree_start = species_topology["sp_subtree_start"]
        sp_subtree_end = species_topology["sp_subtree_end"]
        max_ancestor_depth = int(species_topology["max_ancestor_depth"])

    with _nvtx_range("Pi setup DTS constants"):
        if batched:
            DL_const = 1.0 + log_pD_family + E_family
            SL1_const = log_pS_family + E_s2_family
            SL2_const = log_pS_family + E_s1_family
        else:
            DL_const = 1.0 + log_pD + E
            SL1_const = log_pS + E_s2
            SL2_const = log_pS + E_s1

    with _nvtx_range("Pi setup uniform tensors"):
        mt_squeezed = max_transfer_mat.squeeze(-1) if max_transfer_mat.ndim > 1 else max_transfer_mat
        if batched:
            DL_const = DL_const.contiguous()
            SL1_const = SL1_const.contiguous()
            SL2_const = SL2_const.contiguous()
            Ebar_family = Ebar_family.contiguous()
            E_family = E_family.contiguous()
            mt_family = mt_family.contiguous()
            uniform_leaf_logp = log_pS_family.contiguous()
        else:
            uniform_leaf_logp = log_pS.expand(S).contiguous() if log_pS.ndim == 0 else log_pS.contiguous()

    if batched:
        wave_consts = (DL_const, SL1_const, SL2_const, Ebar_family, E_family, mt_family)
    else:
        wave_consts = (DL_const, SL1_const, SL2_const, Ebar, E, mt_squeezed)

    root_clade_ids_for_skip = None
    if output_intent.can_skip_final_pibar or trace_root_logsumexp:
        root_clade_ids_for_skip = wave_layout.get('root_clade_ids_cpu')
        if root_clade_ids_for_skip is None:
            root_clade_ids_for_skip = [
                int(r) for r in wave_layout['root_clade_ids'].detach().cpu().tolist()
            ]

    root_logsumexp_trace = None
    roots_by_wave = None
    if trace_root_logsumexp:
        root_ids = wave_layout['root_clade_ids'].to(device=device, dtype=torch.long)
        root_logsumexp_trace = torch.full(
            (fixed_iters, root_ids.numel()),
            NEG_INF,
            dtype=dtype,
            device=device,
        )
        roots_by_wave = [None] * len(wave_metas)
        if root_clade_ids_for_skip is None:
            raise RuntimeError("root clade ids are required for root logsumexp tracing")
        for wave_index, meta in enumerate(wave_metas):
            ws = meta['start']
            we = meta['end']
            family_positions = [
                family_pos
                for family_pos, root_id in enumerate(root_clade_ids_for_skip)
                if ws <= root_id < we
            ]
            if family_positions:
                pos = torch.tensor(family_positions, dtype=torch.long, device=device)
                rows = root_ids[pos]
                roots_by_wave[wave_index] = (rows, pos)

    def _can_skip_final_pibar(ws: int, we: int, W: int) -> bool:
        if root_clade_ids_for_skip is None:
            return False
        roots_in_wave = 0
        for root_id in root_clade_ids_for_skip:
            if ws <= root_id < we:
                roots_in_wave += 1
        return roots_in_wave == W

    fuse_final_pibar = True
    specialize_nonleaf_leaf_term = True

    def _progress(event: str, wave_index: int | None = None,
                  local_iter: int | None = None, meta=None) -> None:
        if progress_callback is not None:
            progress_callback(
                event,
                wave_index,
                len(wave_metas),
                local_iter,
                fixed_iters,
                meta,
            )

    pi_wave_iterations = [fixed_iters] * len(wave_metas)
    pi_wave_converged = [False] * len(wave_metas)
    pi_wave_deltas: list[float | None] = [None] * len(wave_metas)

    def _run_wave_self_loop(meta, dts_r, leaf_wt, DL_w, SL1_w, SL2_w,
                            Ebar_w, E_w, mt_w, wave_index):
        ws = meta['start']
        we = meta['end']
        W = meta['W']
        has_leaf_term = (
            not specialize_nonleaf_leaf_term
            or int(meta.get('phase', 1)) == 1
        )
        initial_nonleaf_input = None
        for local_iter in range(fixed_iters):
            iteration_count = local_iter + 1
            pi_in = Pi if (local_iter % 2 == 0) else Pibar
            pi_out = Pibar if (local_iter % 2 == 0) else Pi
            initial_nonleaf_fast_path = local_iter == 0 and not has_leaf_term
            needs_final_pibar = (
                local_iter == fixed_iters - 1
                and not _can_skip_final_pibar(ws, we, W)
            )
            store_final_pibar = fuse_final_pibar and needs_final_pibar
            check_convergence = (
                max_diff_scratch is not None
                and iteration_count % convergence_check_interval == 0
            )
            if initial_nonleaf_fast_path:
                can_read_local_initial = (
                    dts_r is not None
                    and root_logsumexp_trace is None
                    and not check_convergence
                )
                if can_read_local_initial:
                    initial_nonleaf_input = dts_r
                else:
                    pi_out_rows = pi_out.narrow(0, int(ws), int(W))
                    if dts_r is None:
                        pi_out_rows.fill_(torch.finfo(dtype).min)
                    else:
                        pi_out_rows.copy_(dts_r)
                    initial_nonleaf_input = None
            elif (
                local_iter == 0
                and has_leaf_term
                and dts_r is None
                and leaf_species_index is not None
                and not check_convergence
            ):
                wave_step_uniform_initial_leaf_fused_into(
                    pi_out, ws, W, S,
                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                    sp_child1, sp_child2,
                    sp_subtree_start, sp_subtree_end,
                    leaf_species_index,
                    uniform_leaf_logp,
                    family_idx=family_idx if batched else None,
                    family_indexed_consts=batched,
                )
            else:
                step_pi_in = pi_in
                step_input_ws = None
                if local_iter == 1 and initial_nonleaf_input is not None:
                    step_pi_in = initial_nonleaf_input
                    step_input_ws = 0
                wave_step_uniform_fused_into(
                    step_pi_in, pi_out, Pibar, ws, W, S,
                    mt_w, DL_w, Ebar_w, E_w, SL1_w, SL2_w,
                    sp_child1, sp_child2, sp_parent, max_ancestor_depth,
                    leaf_wt, dts_r,
                    leaf_species_idx=leaf_species_index,
                    leaf_logp=uniform_leaf_logp,
                    family_idx=family_idx if batched else None,
                    family_indexed_consts=batched,
                    store_final_pibar=store_final_pibar,
                    final_pibar_row_max=uniform_pibar_row_max if store_final_pibar else None,
                    compute_diff=check_convergence,
                    max_diff_out=max_diff_scratch[:W] if check_convergence else None,
                    has_leaf_term=has_leaf_term,
                    initial_state=local_iter == 0,
                    input_ws=step_input_ws,
                )
            if root_logsumexp_trace is not None:
                root_entry = roots_by_wave[wave_index]
                if root_entry is not None:
                    root_rows, root_positions = root_entry
                    root_logsumexp_trace[local_iter, root_positions] = logsumexp2(
                        pi_out[root_rows],
                        dim=-1,
                    )
            _progress("pi_iter", wave_index, local_iter + 1, meta)
            if check_convergence:
                if max_diff_scratch is None:
                    raise RuntimeError(
                        "internal error: missing Pi convergence scratch buffer"
                    )
                max_diff = float(max_diff_scratch[:W].amax().detach().cpu())
                pi_wave_deltas[wave_index] = max_diff
                if max_diff < convergence_tolerance:
                    pi_wave_iterations[wave_index] = iteration_count
                    pi_wave_converged[wave_index] = True
                    if not _can_skip_final_pibar(ws, we, W):
                        wave_pibar_uniform_parent_fused(
                            Pi, Pibar, ws, W, S,
                            mt_w, sp_parent, max_ancestor_depth,
                            row_max_out=uniform_pibar_row_max,
                            family_idx=family_idx if batched else None,
                            family_indexed_consts=batched,
                        )
                    return
            if needs_final_pibar and not store_final_pibar:
                wave_pibar_uniform_parent_fused(
                    Pi, Pibar, ws, W, S,
                    mt_w, sp_parent, max_ancestor_depth,
                    row_max_out=uniform_pibar_row_max,
                    family_idx=family_idx if batched else None,
                    family_indexed_consts=batched,
                )

    with _nvtx_range("Pi wave forward v2"):
        _progress("start")
        for wave_index, meta in enumerate(wave_metas):
            _progress("wave_start", wave_index, None, meta)
            if meta['has_splits']:
                _progress("dts_start", wave_index, None, meta)
                dts_r = _compute_dts_cross(
                    Pi, Pibar, meta, sp_child1, sp_child2,
                    log_pD_param, log_pS_param, S, device, dtype,
                    family_idx=family_idx if batched else None,
                    family_offset=meta['start'],
                )
            else:
                dts_r = None

            DL_w, SL1_w, SL2_w, Ebar_w, E_w, mt_w = wave_consts
            _run_wave_self_loop(
                meta, dts_r, uniform_leaf_logp, DL_w, SL1_w, SL2_w,
                Ebar_w, E_w, mt_w, wave_index,
            )
        _progress("done")

    with _nvtx_range("Pi finalize permute"):
        if output_intent.emit_root_rows:
            Pi_root_rows = Pi[wave_layout['root_clade_ids']]
        else:
            Pi_root_rows = None

        if output_intent.emit_original_pi:
            perm = wave_layout['perm']
            Pi_orig = Pi[perm]
        else:
            Pi_orig = None

    Pi_wave_ordered = Pi if output_intent.emit_wave_ordered_pi else None

    return {
        'Pi': Pi_orig,
        'Pi_root_rows': Pi_root_rows,
        'Pi_wave_ordered': Pi_wave_ordered,
        'Pibar_wave_ordered': Pibar if output_intent.retain_saved_state else None,
        'uniform_pibar_row_max': (
            uniform_pibar_row_max if output_intent.retain_saved_state else None
        ),
        'root_logsumexp_trace': root_logsumexp_trace,
        'Pi_wave_iterations': pi_wave_iterations,
        'Pi_wave_converged': pi_wave_converged,
        'Pi_wave_convergence_deltas': pi_wave_deltas,
        'Pi_max_iterations': max(pi_wave_iterations, default=0),
    }
