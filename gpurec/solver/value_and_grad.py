"""Value-and-gradient closure over gpurec's batched streaming evaluators.

Port of kernel-bench ``newton/vg.py``, re-pointed from kbench's single-``static``
``run_forward``/``run_backward`` pair onto gpurec's multi-batch ``stream_batches``
(scalar loss + gradient). The optimization variable is the flat ``theta`` vector
(``theta.reshape(-1)``); the gradient layout is elementwise-identical, so
flatten/unflatten is a plain ``reshape``.

``make_value_and_grad`` is the single ``f(theta_vec) -> (loss, grad)`` contract the
whole optimization layer sits on. Unlike kbench, gpurec carries the E warm-start on
each ``static.warm_E`` and refreshes it internally, so the ``warm_E`` thread is a
no-op kept only for signature parity.

Terminology rename applied at the kbench -> gpurec port boundary: item->family,
col->receiver, state->species, solve_e_pi->solve_resident_e_pi,
col_weights->receiver_weights, max_coupling->max_transfer,
col_log_probs->receiver_log_probs.
"""

from __future__ import annotations

import torch

from gpurec.api._batch_state import _BatchStatic
from gpurec.api._execution import stream_batches
from gpurec.config.memory import MemoryOptions
from gpurec.core.inference.solver import nll_from_root_rows, solve_resident_e_pi
from gpurec.solver.penalties import (
    tv_prior_and_grad, origination_penalty_and_grad, group_expand, group_reduce,
    DEFAULT_TV_EPS,
)

# Names of the forward-solve intermediates the exact-HVP / tangent path consumes,
# in the order ``solve_resident_e_pi`` returns them. gpurec rename of kbench's
# FORWARD_SAVED_NAMES: max_coupling -> max_transfer, col_log_probs -> receiver_log_probs.
#
# Row-gauged storage has no extra tuple entry: ``forward_solve`` snapshots the
# exact sidecar produced by this solve under ``saved["pi_state"]``.
# Keeping it out of this tuple preserves the long-standing return contract and
# ensures derivative consumers never read offsets from a later solve through
# the mutable batch-static attribute.
FORWARD_SAVED_NAMES = (
    "E", "E_s1", "E_s2", "Ebar", "root_rows", "pi_wave", "pibar_wave",
    "pibar_row_max", "log_pS", "log_pD", "log_pL", "max_transfer", "receiver_log_probs",
)


def free_cuda_cache_if_tight(min_free_gib: float = MemoryOptions().min_free_gib_driver):
    """Release the caching allocator's pool to the driver when driver-free memory runs low.

    The hand-written backward gates its scratch on ``torch.cuda.mem_get_info`` (driver-free
    bytes), which the caching allocator does not replenish on tensor free -- without this,
    long optimization loops on the big problems trip the gate spuriously.
    """
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info()
        if free_b < min_free_gib * (1024 ** 3):
            torch.cuda.empty_cache()


def _as_static_list(batch_statics):
    """Accept either a single ``_BatchStatic`` or a list of them; return a list."""
    if isinstance(batch_statics, _BatchStatic):
        return [batch_statics]
    return list(batch_statics)


def forward_solve(batch_statics, theta, receiver_weights, *, warm_E=None):
    """Run the forward solve at ``theta``; return ``(loss_tensor, saved)``.

    For a SINGLE batch (the exact-HVP / Newton-polish regime) ``saved`` is the dict of
    forward intermediates the second-order kernels consume (keyed by
    :data:`FORWARD_SAVED_NAMES`). For multiple batches ``saved`` is None -- the HVP path is
    single-batch; multi-batch optimization uses :func:`make_value_and_grad`.
    """
    statics = _as_static_list(batch_statics)
    with torch.no_grad():
        if len(statics) == 1:
            static = statics[0]
            theta_b = (
                theta.index_select(0, static.family_index_tensor) if static.genewise else theta
            )
            out = solve_resident_e_pi(
                static, theta_b, receiver_weights,
                warm_start_E=warm_E if warm_E is not None else static.warm_E,
            )
            saved = dict(zip(FORWARD_SAVED_NAMES, out))
            saved["pi_state"] = getattr(static, "pi_forward_state", None)
            if saved["pi_state"] is None:
                raise RuntimeError("Pi forward did not publish its row-offset state")
            loss = nll_from_root_rows(saved["root_rows"], saved["E"])
            return loss, saved
        loss, _g, _gr, _go = stream_batches(
            statics, theta, receiver_weights, torch.zeros_like(receiver_weights),
            genewise=statics[0].genewise, need_grad=False,
        )
        return loss, None


def make_value_and_grad(batch_statics, receiver_weights, *, theta_shape=None,
                        grad_avg_K: int = MemoryOptions().grad_avg_k,
                        prior=None, tree_penalty=None, optimize_receiver: bool = False,
                        origination_weights=None, optimize_origination: bool = False,
                        tv_penalty=None, origination_penalty=None, group_index=None):
    """Return ``f(theta_vec, *, warm_E=None, want_grad=True) -> (loss, g_vec, saved, warm_E_out)``.

    The single ``(theta_vec -> value, grad)`` contract the gpurec optimization layer sits on.
    ``batch_statics`` is ``model.batch_statics`` (a length-1 list for a single batch).
    ``loss`` is a Python float; ``g_vec`` is a flat tensor matching ``theta_vec`` (or None when
    ``want_grad=False``). The genewise/specieswise mode is read off the batch statics.

    ``optimize_receiver`` switches the optimization variable to the joint vector
    ``z = [theta.reshape(-1); alpha]`` of length ``theta.numel() + S``, where ``S =
    receiver_weights.numel()`` and ``alpha`` are the per-species receiver logits (in R^S). When set,
    ``f`` splits ``z`` into ``theta = z[:theta.numel()].reshape(theta_shape)`` and ``alpha =
    z[theta.numel():]``, passes ``alpha`` as ``receiver_weights`` into the forward/backward EACH call
    (the closure ``receiver_weights`` arg is then used only to fix ``S``/dtype/device, not as a stale
    value), and returns ``g_z = cat([g_theta_flat (+ theta penalties), grad_receiver])``. The
    receiver-weight block of the gradient is the production VJP's ``grad_receiver`` (already a correct,
    alpha-space, gauge-respecting ``dNLL/dalpha``: the softmax Jacobian alpha->w is applied inside the
    head autograd). ``theta`` penalties (``prior``/``tree_penalty``) do NOT touch the alpha block.
    Default ``False`` preserves the legacy theta-only contract (``theta_vec`` is just ``theta``).

    ``warm_E`` is accepted for signature parity with kernel-bench but is a no-op: gpurec carries
    the E warm-start on each ``static.warm_E`` and refreshes it internally every evaluation.
    ``saved``/``warm_E_out`` are returned as None (the streaming evaluator does forward+backward
    in one pass; there is nothing to hand back).

    ``grad_avg_K`` averages the (atomically nondeterministic) backward over K passes to suppress
    its noise floor -- used by the HVP/CG path; default 1 (no extra cost).

    ``prior`` adds the centered MAP / ridge term ``(lam/2)||theta - theta_ref||^2`` to loss and
    gradient: pass ``(lam: float, theta_ref: tensor)`` (``theta_ref`` is flattened).

    ``tree_penalty`` adds the Sanderson-style autocorrelated-rates (GBM) roughness term
    ``(lam_tree/2) * sum_{edges (child,parent)} ||theta[child] - theta[parent]||^2`` -- a tree
    graph-Laplacian quadratic that shrinks each species' rates toward its PARENT's (no arbitrary
    center; lam_tree -> infinity gives the clock with the common rate set by the data). Pass
    ``(lam_tree: float, sp_parent: int tensor [S])`` where ``sp_parent[s]`` is the parent species
    index (``< 0`` at the root). This is the penalty the CV homotopy sweeps lam_tree over; it
    composes with ``prior`` (both can be set). Its Hessian is ``lam_tree * L`` (PSD), so it keeps
    the certified-PD structure of the centered ridge.
    """
    statics = _as_static_list(batch_statics)
    genewise = statics[0].genewise
    S = int(receiver_weights.numel())  # number of receiver logits; kept EXPLICIT (theta may be [F,3])
    if theta_shape is None:
        theta_shape = (S, 3)
    theta_shape = tuple(theta_shape)
    theta_numel = 1
    for _d in theta_shape:
        theta_numel *= int(_d)
    lam = None if prior is None else float(prior[0])
    theta_ref_flat = None if prior is None else prior[1].detach().reshape(-1).contiguous()

    # Per-species origination logits. When optimize_origination, the live values are carried in the
    # tail block of z (after theta and, if present, alpha); otherwise this fixed (default uniform)
    # vector is used. Enters ONLY the NLL aggregation, so it adds no kernel/fixed-point cost.
    base_origination = (
        receiver_weights.new_zeros(S)
        if origination_weights is None
        else origination_weights.detach().reshape(-1).to(device=receiver_weights.device, dtype=receiver_weights.dtype)
    )

    # GBM / tree-Laplacian penalty: precompute the edge (child, parent) index pair once.
    lam_tree = None if tree_penalty is None else float(tree_penalty[0])
    tp_child = tp_parent = None
    if tree_penalty is not None:
        sp_parent = tree_penalty[1].detach().reshape(-1).long()
        tp_child = (sp_parent >= 0).nonzero(as_tuple=True)[0].contiguous()   # [E] non-root species
        tp_parent = sp_parent[tp_child].contiguous()                         # [E] their parents

    tv_lam = tv_sp_parent = None
    tv_eps = DEFAULT_TV_EPS
    if tv_penalty is not None:
        tv_lam, tv_sp_parent = tv_penalty[0], tv_penalty[1].detach().reshape(-1)
        tv_eps = tv_penalty[2] if len(tv_penalty) > 2 else DEFAULT_TV_EPS

    n_groups = None
    if group_index is not None:
        group_index = group_index.detach().reshape(-1).long()
        n_groups = int(group_index.max().item()) + 1

    def f(theta_vec: torch.Tensor, *, warm_E=None, want_grad: bool = True):
        zvec = theta_vec.detach().reshape(-1)
        # joint layout: [theta_numel] [+ S alpha if optimize_receiver] [+ S origination if optimize_origination]
        tvec = zvec[:theta_numel]
        off = theta_numel
        if optimize_receiver:
            recv = zvec[off:off + S].contiguous()  # LIVE per-species receiver logits
            off += S
        else:
            recv = receiver_weights
        if optimize_origination:
            orig = zvec[off:off + S].contiguous()  # LIVE per-species origination logits
            off += S
        else:
            orig = base_origination
        theta = tvec.reshape(theta_shape).contiguous()
        theta_expanded = group_expand(theta, group_index)  # [S,3] (identity if group_index is None)
        if want_grad:
            # the backward's scratch gate reads driver-free memory; return any stale cached
            # blocks (e.g. from another dtype's stage) before it runs
            free_cuda_cache_if_tight()
        loss, g_theta, g_recv, g_orig = stream_batches(
            statics, theta_expanded, recv, orig, genewise=genewise, need_grad=want_grad,
            need_origination_grad=optimize_origination,
        )
        loss_val = float(loss)
        d = None
        if lam is not None:
            d = theta_expanded.reshape(-1) - theta_ref_flat.to(device=tvec.device, dtype=tvec.dtype)
            loss_val = loss_val + 0.5 * lam * float((d * d).sum())
        tdiff = None
        if lam_tree is not None:
            ts = theta_expanded  # [S, 3]
            tdiff = ts.index_select(0, tp_child) - ts.index_select(0, tp_parent)  # [E, 3]
            loss_val = loss_val + 0.5 * lam_tree * float((tdiff * tdiff).sum())
        if tv_lam is not None:
            tv_pen, _ = tv_prior_and_grad(theta_expanded, tv_sp_parent, tv_lam, tv_eps)
            loss_val = loss_val + float(tv_pen)
        if origination_penalty is not None and optimize_origination:
            o_pen, _ = origination_penalty_and_grad(orig, origination_penalty)
            loss_val = loss_val + float(o_pen)
        g_vec = None
        if want_grad:
            if int(grad_avg_K) > 1:
                acc = g_theta
                acc_recv = g_recv
                acc_orig = g_orig
                for _ in range(int(grad_avg_K) - 1):
                    _l, gk, grk, gok = stream_batches(
                        statics, theta_expanded, recv, orig, genewise=genewise, need_grad=True,
                        need_origination_grad=optimize_origination,
                    )
                    acc = acc + gk
                    if optimize_receiver:
                        acc_recv = acc_recv + grk
                    if optimize_origination:
                        acc_orig = acc_orig + gok
                g_theta = acc / float(grad_avg_K)
                if optimize_receiver:
                    g_recv = acc_recv / float(grad_avg_K)
                if optimize_origination:
                    g_orig = acc_orig / float(grad_avg_K)
            g_theta_full = g_theta.reshape(theta_expanded.shape)
            if lam is not None:
                g_theta_full = g_theta_full + (lam * d).reshape(theta_expanded.shape)
            if lam_tree is not None:
                # d/dtheta of (lam_tree/2) sum_e ||theta[c]-theta[p]||^2: +g on child, -g on parent
                gpen = torch.zeros_like(theta_expanded)
                step = lam_tree * tdiff
                gpen.index_add_(0, tp_child, step)
                gpen.index_add_(0, tp_parent, -step)
                g_theta_full = g_theta_full + gpen
            if tv_lam is not None:
                _, tv_g = tv_prior_and_grad(theta_expanded, tv_sp_parent, tv_lam, tv_eps)
                g_theta_full = g_theta_full + tv_g
            g_vec = group_reduce(g_theta_full, group_index, n_groups).reshape(-1)
            # alpha / origination blocks: stream_batches already returns dNLL/dalpha and dNLL/dorig
            # (softmax Jacobians applied inside the head autograd); theta penalties don't touch them.
            if optimize_receiver:
                g_vec = torch.cat([g_vec.reshape(-1), g_recv.reshape(-1)])
            if optimize_origination:
                if origination_penalty is not None:
                    _, o_g = origination_penalty_and_grad(orig, origination_penalty)
                    g_orig = g_orig + o_g
                g_vec = torch.cat([g_vec.reshape(-1), g_orig.reshape(-1)])
            g_vec = g_vec.contiguous()
        return loss_val, g_vec, None, None

    return f
