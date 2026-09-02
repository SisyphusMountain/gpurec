"""H100 sweep of the Triton launch parameter ``num_warps`` for the five hottest gpurec kernels.

Only ``num_warps`` moves. Every block size and every ``tl.constexpr`` stays at its current value,
so each kernel does exactly the same arithmetic on exactly the same tiles -- ``num_warps`` only
changes how many hardware warps the scheduler gives one Triton program. The five launch sites each
read a dedicated module-level constant (added for this sweep), and this script rebinds that constant
between runs; Triton recompiles the kernel automatically on the first launch at a new value.

Four of the five kernels run inside one loss+gradient call and are timed that way; the fifth
(``_apply_reconciliation_self_loop_jvp_iterations_kernel``) only runs inside a Hessian-vector
product, so it is timed with one ``_analytic_hessian`` call.

Correctness gate. Because the arithmetic is unchanged, the forward per-family negative
log-likelihood vector must come back bitwise identical to the default setting. The gradient uses
atomic adds, whose summation order is not reproducible, so it is compared against a measured noise
floor: the largest absolute difference between two gradient calls made with identical settings.
A setting passes when its largest absolute gradient difference from the default is at most twice
that noise floor.

Usage (from the repo root on the cluster, env.sh sourced):

  python -u benchmark/cc/sweep_num_warps.py \
      --species $CC_SPECIES --families $CC_FAMILIES --limit 300 --clade-budget 315000 \
      --theta-init 0.0 --warps 2,4,8,16 --grad-reps 3 --phase all
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Fixed-point iteration count and Neumann series length for the solve. These match the genewise
# recipe's first tier (the setting the nsys profile that motivated this sweep was taken at); they
# are not knobs of the sweep, because changing them would change which kernels dominate.
PI_ITERS = 16
NEUMANN_TERMS = 16


def _targets():
    """Return the sweep registry: key -> (module, constant name, kernel name, which timer)."""
    from gpurec.core.kernels import pi_forward, wave_backward, wave_tangent

    return {
        "update_reconciliation": (
            pi_forward, "_NUM_WARPS_UPDATE_RECONCILIATION",
            "_update_reconciliation_likelihood_kernel", "grad",
        ),
        "self_loop_transpose": (
            wave_backward, "_NUM_WARPS_SELF_LOOP_TRANSPOSE",
            "_apply_reconciliation_self_loop_transpose_kernel", "grad",
        ),
        "prepare_self_loop_vjp": (
            wave_backward, "_NUM_WARPS_PREPARE_SELF_LOOP_VJP",
            "_prepare_reconciliation_self_loop_vjp_kernel", "grad",
        ),
        "transfer_subtree_vjp": (
            wave_backward, "_NUM_WARPS_TRANSFER_SUBTREE_VJP",
            "_accumulate_transfer_subtree_vjp_kernel", "grad",
        ),
        "self_loop_jvp_iters": (
            wave_tangent, "_NUM_WARPS_SELF_LOOP_JVP_ITERS",
            "_apply_reconciliation_self_loop_jvp_iterations_kernel", "hessian",
        ),
    }


def _timed_grad(model, theta):
    """Run one loss+gradient call; return (seconds, per-family NLL vector, gradient)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_vector, grad, _extra = model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, loss_vector.detach().clone(), grad.detach().clone()


def _timed_hessian(model, theta):
    """Run one analytic-Hessian call (3 HVP probes); return (seconds, [G,3,3] curvature)."""
    from gpurec.fit.genewise_fit import _analytic_hessian

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    H = _analytic_hessian(model, theta, PI_ITERS)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, H.detach().clone()


def _max_abs_diff(a, b):
    """Largest absolute element-wise difference between two tensors, as a Python float."""
    return float((a.double() - b.double()).abs().max())


def _run_grad_phase(model, theta, warps, grad_reps, rows):
    """Time the four gradient-path kernels at each num_warps value; append result rows."""
    targets = _targets()
    grad_keys = [k for k, v in targets.items() if v[3] == "grad"]

    # Warm-up at the default settings: compiles every kernel and settles the allocator.
    warm_s, _, _ = _timed_grad(model, theta)
    print(f"[sweep] grad warm-up (defaults) {warm_s:.2f}s", flush=True)

    base_times = []
    base_nll = None
    base_grad = None
    for rep in range(grad_reps):
        s, nll, g = _timed_grad(model, theta)
        base_times.append(s)
        if rep == 0:
            base_nll, base_grad = nll, g
    base_s = min(base_times)

    # Atomics noise floor: one more gradient call with identical settings.
    _s, noise_nll, noise_grad = _timed_grad(model, theta)
    noise = _max_abs_diff(noise_grad, base_grad)
    nll_repeatable = torch.equal(noise_nll, base_nll)
    print(f"[sweep] baseline (all kernels at default): grad min {base_s:.3f}s of "
          f"{[round(x, 3) for x in base_times]}; atomics noise = {noise:.3e}; "
          f"forward NLL repeatable across identical calls = {nll_repeatable}; "
          f"grad absmax = {float(base_grad.abs().max()):.4f}", flush=True)

    for key in grad_keys:
        module, const, kernel_name, _which = targets[key]
        default = getattr(module, const)
        for v in warps:
            if v == default:
                rows.append(dict(kernel=kernel_name, key=key, num_warps=v, is_default=True,
                                 seconds=base_s, speedup=1.0, fwd_equal=True,
                                 grad_diff=0.0, noise=noise, note="baseline"))
                continue
            setattr(module, const, v)
            try:
                _w, _n, _g = _timed_grad(model, theta)          # compile + warm-up at this value
                times = []
                nll = None
                g = None
                for _rep in range(grad_reps):
                    s, nll, g = _timed_grad(model, theta)
                    times.append(s)
                best = min(times)
                rows.append(dict(kernel=kernel_name, key=key, num_warps=v, is_default=False,
                                 seconds=best, speedup=base_s / best,
                                 fwd_equal=bool(torch.equal(nll, base_nll)),
                                 grad_diff=_max_abs_diff(g, base_grad), noise=noise, note=""))
            except Exception as exc:                             # compile / launch failure
                rows.append(dict(kernel=kernel_name, key=key, num_warps=v, is_default=False,
                                 seconds=float("nan"), speedup=float("nan"), fwd_equal=False,
                                 grad_diff=float("nan"), noise=noise,
                                 note=f"FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:80]}"))
            finally:
                setattr(module, const, default)
            print(f"[sweep] {kernel_name} num_warps={v}: {rows[-1]}", flush=True)
            torch.cuda.empty_cache()


def _run_hessian_phase(model, theta, warps, rows):
    """Time the Hessian-only tangent kernel at each num_warps value; append result rows."""
    targets = _targets()
    hess_keys = [k for k, v in targets.items() if v[3] == "hessian"]

    warm_s, _H = _timed_hessian(model, theta)
    del _H
    torch.cuda.empty_cache()
    print(f"[sweep] hessian warm-up (defaults) {warm_s:.2f}s", flush=True)

    base_s, base_H = _timed_hessian(model, theta)
    torch.cuda.empty_cache()
    noise_s, noise_H = _timed_hessian(model, theta)
    hess_noise = _max_abs_diff(noise_H, base_H)
    del noise_H
    torch.cuda.empty_cache()
    print(f"[sweep] hessian baseline {base_s:.3f}s (repeat {noise_s:.3f}s); "
          f"curvature noise between two identical calls = {hess_noise:.3e}; "
          f"H absmax = {float(base_H.abs().max()):.4f}", flush=True)

    for key in hess_keys:
        module, const, kernel_name, _which = targets[key]
        default = getattr(module, const)
        for v in warps:
            if v == default:
                rows.append(dict(kernel=kernel_name, key=key, num_warps=v, is_default=True,
                                 seconds=base_s, speedup=1.0, fwd_equal=True,
                                 grad_diff=0.0, noise=hess_noise, note="baseline (hessian)"))
                continue
            setattr(module, const, v)
            try:
                _w, _H = _timed_hessian(model, theta)            # compile + warm-up at this value
                del _H
                torch.cuda.empty_cache()
                s, H = _timed_hessian(model, theta)
                rows.append(dict(kernel=kernel_name, key=key, num_warps=v, is_default=False,
                                 seconds=s, speedup=base_s / s,
                                 fwd_equal=bool(torch.equal(H, base_H)),
                                 grad_diff=_max_abs_diff(H, base_H), noise=hess_noise,
                                 note="hessian: 'fwd equal'/'diff' compare the [G,3,3] curvature"))
                del H
            except Exception as exc:
                rows.append(dict(kernel=kernel_name, key=key, num_warps=v, is_default=False,
                                 seconds=float("nan"), speedup=float("nan"), fwd_equal=False,
                                 grad_diff=float("nan"), noise=hess_noise,
                                 note=f"FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:80]}"))
            finally:
                setattr(module, const, default)
                torch.cuda.empty_cache()
            print(f"[sweep] {kernel_name} num_warps={v}: {rows[-1]}", flush=True)


def _print_table(rows):
    """Print the final result table."""
    head = (f"{'kernel':<52} {'warps':>5} {'seconds':>9} {'speedup':>8} {'fwd bitwise':>11} "
            f"{'diff vs default':>16} {'noise':>11}  note")
    print("\n" + "=" * len(head), flush=True)
    print(head, flush=True)
    print("=" * len(head), flush=True)
    for r in rows:
        print(f"{r['kernel']:<52} {r['num_warps']:>5} {r['seconds']:>9.3f} {r['speedup']:>8.3f} "
              f"{('yes' if r['fwd_equal'] else 'NO'):>11} {r['grad_diff']:>16.3e} "
              f"{r['noise']:>11.3e}  {r['note']}", flush=True)
    print("=" * len(head), flush=True)

    print("\n[sweep] verdict (>= 5% faster AND forward bitwise-equal AND diff <= 2x noise):", flush=True)
    any_win = False
    for r in rows:
        if r["is_default"] or not (r["seconds"] == r["seconds"]):
            continue
        faster = r["speedup"] >= 1.05
        safe = r["fwd_equal"] and r["grad_diff"] <= 2.0 * r["noise"]
        if faster and safe:
            any_win = True
            print(f"[sweep]   ADOPT {r['kernel']} num_warps={r['num_warps']}: "
                  f"{(r['speedup'] - 1.0) * 100:.1f}% faster, checks pass", flush=True)
        elif faster:
            print(f"[sweep]   reject {r['kernel']} num_warps={r['num_warps']}: "
                  f"{(r['speedup'] - 1.0) * 100:.1f}% faster BUT checks fail "
                  f"(bitwise={r['fwd_equal']}, diff={r['grad_diff']:.3e} vs 2x noise "
                  f"{2.0 * r['noise']:.3e})", flush=True)
    if not any_win:
        print("[sweep]   none -- keep every current default", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True, help="listfile: one .ale path per line")
    ap.add_argument("--limit", required=True, type=int, help="0 = all families; N = first N")
    ap.add_argument("--clade-budget", required=True, type=int, help="clades per batch")
    ap.add_argument("--theta-init", required=True, type=float,
                    help="every theta entry is set to this log2-rate (0.0 = the nsys profile point)")
    ap.add_argument("--warps", required=True, help="comma-separated num_warps values, e.g. 2,4,8,16")
    ap.add_argument("--grad-reps", required=True, type=int,
                    help="timed loss+gradient calls per configuration; the minimum is reported")
    ap.add_argument("--phase", required=True, choices=("grad", "hessian", "all"))
    args = ap.parse_args()

    warps = [int(x) for x in args.warps.split(",") if x.strip()]
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    # Cold adjoint, exactly as benchmark/cc/test_grad_scaling.py --warm 0 does it.
    os.environ.pop("GPUREC_WARM_ADJOINT", None)
    solver_options = SolverOptions(**{**dict(_BASE_SOLVER), "pi_iters": PI_ITERS,
                                      "neumann_terms": NEUMANN_TERMS})

    t0 = time.perf_counter()
    model = GeneReconModel(args.species, paths, mode="genewise", device="cuda",
                           dtype=torch.float32, solver_options=solver_options,
                           clade_budget=args.clade_budget)
    model.receiver_weights.requires_grad_(False)
    torch.cuda.synchronize()
    print(f"[sweep] build {time.perf_counter() - t0:.1f}s families={len(paths)} "
          f"batches={len(model.batch_statics)} warm_ok={getattr(model, 'warm_adjoint_ok', None)} "
          f"warps={warps} phase={args.phase} theta_init={args.theta_init}", flush=True)

    defaults = {k: getattr(v[0], v[1]) for k, v in _targets().items()}
    print(f"[sweep] current defaults: {defaults}", flush=True)

    theta = torch.full((len(paths), 3), args.theta_init, device="cuda", dtype=torch.float32)

    rows = []
    try:
        if args.phase in ("grad", "all"):
            _run_grad_phase(model, theta, warps, args.grad_reps, rows)
        if args.phase in ("hessian", "all"):
            _run_hessian_phase(model, theta, warps, rows)
    finally:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
