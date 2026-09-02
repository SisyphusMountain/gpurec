"""Driver for the per-wave BACKWARD kernel work: ncu probe, gradient timing, exactness check.

Every mode builds the same model -- ``--limit`` families in list order, exact forward self-loop,
exact adjoint self-loop, the fit's own solver settings -- and evaluates the genewise gradient at
either the fitted theta from a ``run_genewise`` ``.pt`` or at a flat theta.

Modes
  probe   one gradient, nothing timed. This is the process Nsight Compute attaches to
          (see benchmark/cc/run_ncu_bwd.sh).
  nsys    one gradient inside a CUDA-profiler range, so an Nsight Systems capture started with
          ``--capture-range=cudaProfilerApi`` reports that ONE gradient and nothing else. The
          per-kernel share of a gradient is very different from the share of a gradient plus a
          Hessian, which is what benchmark/cc/nsys_grad.py captures.
  time    three warm gradients per theta, reports the mean wall time and the per-batch spread.
  check   two IDENTICAL gradient calls per theta, dumps per-family NLL, the full [F,3] gradient of
          both calls and every per-batch gradient block to a ``.pt``. The difference between the
          two calls is the run-to-run atomics noise; the dump is what ``compare`` diffs against a
          dump taken from the other kernel version.
  compare max-abs differences between two ``check`` dumps, printed next to each dump's own noise.
  warps   one gradient per candidate ``num_warps`` for the two hottest backward kernels, with the
          gradient difference from the default printed against that noise floor. NOTE what this
          measured on the H100: every non-default value moves the gradient by 3 to 7 while the
          noise floor is 1.2e-3, because ``valid_receiver_mass = total_receiver_mass -
          ancestor_sum`` in the self-loop VJP kernel cancels, and ``num_warps`` changes the order
          of the 2013-term sum that makes the first of those two numbers.

Two checkouts, one job. ``compare`` and the A/B timings above are meant to be run as
``PYTHONPATH=<other checkout> python benchmark/cc/bwd_kernels.py ...`` twice in the same job, so
both versions see the same GPU and the same thermal state.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch


def _load_theta(theta_pt, paths, device):
    saved = torch.load(theta_pt, map_location="cpu")
    pos = {p: i for i, p in enumerate(saved["paths"])}
    return torch.stack([saved["theta"][pos[p]] for p in paths]).float().to(device)


def _build(args):
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]

    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    os.environ.pop("GPUREC_WARM_ADJOINT", None)
    so = SolverOptions(**{
        **_BASE_SOLVER,
        "pi_iters": 16,
        "neumann_terms": 16,
        "forward_self_loop": "exact",
        "adjoint_self_loop": "exact",
    })
    model = GeneReconModel(
        args.species, paths, mode="genewise", device="cuda", dtype=torch.float32,
        solver_options=so, clade_budget=args.clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    theta_fit = _load_theta(args.theta_pt, paths, "cuda")
    thetas = {"fitted": theta_fit, "flat-6": torch.full_like(theta_fit, -6.0)}
    return model, paths, thetas


def _capture_per_batch(model):
    """Record each batch's gradient block, in call order, for the next gradient evaluation."""
    from gpurec.api import _execution

    blocks = []
    original = _execution.evaluate_static_loss_vector_grad

    def wrapped(static, *a, **k):
        out = original(static, *a, **k)
        grad = out[1]
        blocks.append(None if grad is None else grad.detach().clone().cpu())
        return out

    _execution.evaluate_static_loss_vector_grad = wrapped
    return blocks, (lambda: setattr(_execution, "evaluate_static_loss_vector_grad", original))


def _mode_probe(args):
    model, paths, thetas = _build(args)
    theta = thetas[args.theta]
    model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    print(f"[probe] {len(paths)} families, {len(model.batch_statics)} batches, theta={args.theta}", flush=True)
    return 0


def _mode_nsys(args):
    """One gradient inside a CUDA-profiler range, so Nsight Systems reports that gradient alone.

    A warm-up gradient runs first (outside the range) so no Triton compile lands inside it.
    """
    model, paths, thetas = _build(args)
    theta = thetas[args.theta]
    model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    print(f"[nsys] {len(paths)} families, {len(model.batch_statics)} batches, theta={args.theta}", flush=True)
    torch.cuda.profiler.start()
    t = time.perf_counter()
    model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t
    torch.cuda.profiler.stop()
    print(f"[nsys] captured gradient took {elapsed:.3f}s", flush=True)
    return 0


def _mode_time(args):
    model, paths, thetas = _build(args)
    from gpurec.api import _execution

    per_batch = []
    original = _execution.evaluate_static_loss_vector_grad

    def timed(static, *a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = original(static, *a, **k)
        torch.cuda.synchronize()
        per_batch.append(time.perf_counter() - t)
        return out

    _execution.evaluate_static_loss_vector_grad = timed
    print(f"[time] {len(paths)} families, {len(model.batch_statics)} batches", flush=True)
    for name in ("fitted", "flat-6"):
        theta = thetas[name]
        model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)  # warm-up
        reps = []
        for _rep in range(3):
            per_batch.clear()
            torch.cuda.synchronize()
            t = time.perf_counter()
            model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
            torch.cuda.synchronize()
            reps.append(time.perf_counter() - t)
            spread = sorted(per_batch)
            print(f"[time] {name} rep{_rep}: {reps[-1]:.3f}s; per-batch min/med/max = "
                  f"{spread[0]:.3f}/{statistics.median(spread):.3f}/{spread[-1]:.3f}s", flush=True)
        print(f"[time] {name} MEAN of 3 = {statistics.mean(reps):.3f}s", flush=True)
    _execution.evaluate_static_loss_vector_grad = original
    return 0


def _mode_warps(args):
    """Time one gradient per candidate ``num_warps`` for the two hottest backward kernels.

    Only the launch shape moves: block sizes and every other compile-time constant stay put, so
    each program does the same arithmetic on the same tiles. What does move is the order in which
    a program's 2013 species are summed into its per-row totals, so the gradient shifts by float32
    rounding; each candidate's largest gradient difference from the default is printed next to the
    run-to-run atomics noise measured from two identical calls at the default.
    """
    from gpurec.core.kernels import wave_backward

    model, paths, thetas = _build(args)
    theta = thetas[args.theta]

    def timed():
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss_vec, grad, _ = model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t, loss_vec.detach().clone(), grad.detach().clone()

    timed()  # warm-up: compile every kernel at the default settings
    base_times = []
    base_nll = base_grad = None
    for rep in range(3):
        seconds, nll, grad = timed()
        base_times.append(seconds)
        if rep == 0:
            base_nll, base_grad = nll, grad
    _s, noise_nll, noise_grad = timed()
    noise = (noise_grad - base_grad).abs().max().item()
    base = min(base_times)
    print(f"[warps] {len(paths)} families, theta={args.theta}: default gradient min {base:.3f}s of "
          f"{[round(x, 3) for x in base_times]}; atomics noise {noise:.3e}; "
          f"NLL repeatable = {bool(torch.equal(noise_nll, base_nll))}", flush=True)

    sweep = {
        "_NUM_WARPS_PREPARE_SELF_LOOP_VJP": (4, 8, 16, 32),
        "_NUM_WARPS_TRANSFER_SUBTREE_VJP": (2, 4, 8, 16),
    }
    for name, candidates in sweep.items():
        default = getattr(wave_backward, name)
        for value in candidates:
            if value == default:
                print(f"[warps] {name}={value}: {base:.3f}s (default)", flush=True)
                continue
            setattr(wave_backward, name, value)
            try:
                timed()  # compile at this setting
                runs = [timed() for _ in range(3)]
                seconds = min(r[0] for r in runs)
                nll, grad = runs[-1][1], runs[-1][2]
                print(f"[warps] {name}={value}: {seconds:.3f}s  speedup {base / seconds:.3f}x; "
                      f"NLL identical = {bool(torch.equal(nll, base_nll))}; "
                      f"grad max|diff| {(grad - base_grad).abs().max().item():.3e} "
                      f"(noise {noise:.3e})", flush=True)
            except Exception as exc:
                print(f"[warps] {name}={value}: FAILED {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:120]}", flush=True)
            finally:
                setattr(wave_backward, name, default)
            torch.cuda.empty_cache()
    return 0


def _mode_check(args):
    model, paths, thetas = _build(args)
    dump = {"paths": paths, "n_batches": len(model.batch_statics)}
    for name in ("fitted", "flat-6"):
        theta = thetas[name]
        entry = {}
        for call in (0, 1):
            blocks, restore = _capture_per_batch(model)
            loss_vec, grad_theta, _ = model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
            restore()
            torch.cuda.synchronize()
            entry[f"nll{call}"] = loss_vec.detach().cpu()
            entry[f"grad{call}"] = grad_theta.detach().cpu()
            entry[f"blocks{call}"] = blocks
        noise = (entry["grad0"] - entry["grad1"]).abs().max().item()
        nll_noise = (entry["nll0"] - entry["nll1"]).abs().max().item()
        print(f"[check] {name}: |grad| max {entry['grad0'].abs().max().item():.6e}; "
              f"run-to-run grad noise {noise:.6e}; nll noise {nll_noise:.6e}", flush=True)
        dump[name] = entry
    torch.save(dump, args.out)
    print(f"[check] wrote {args.out}", flush=True)
    return 0


def _mode_compare(args):
    a = torch.load(args.a, map_location="cpu")
    b = torch.load(args.b, map_location="cpu")
    for name in ("fitted", "flat-6"):
        ea, eb = a[name], b[name]
        noise_a = (ea["grad0"] - ea["grad1"]).abs().max().item()
        noise_b = (eb["grad0"] - eb["grad1"]).abs().max().item()
        dg = (ea["grad0"] - eb["grad0"]).abs().max().item()
        dnll = (ea["nll0"] - eb["nll0"]).abs().max().item()
        blocks = [
            (x - y).abs().max().item()
            for x, y in zip(ea["blocks0"], eb["blocks0"])
            if x is not None and y is not None
        ]
        db = max(blocks) if blocks else float("nan")
        scale = ea["grad0"].abs().max().item()
        print(f"[compare] {name}: full-grad max|A-B| = {dg:.6e}  (noise A {noise_a:.6e}, "
              f"B {noise_b:.6e}, |grad| scale {scale:.6e})", flush=True)
        print(f"[compare] {name}: per-batch max|A-B| = {db:.6e} over {len(blocks)} batches", flush=True)
        print(f"[compare] {name}: per-family NLL max|A-B| = {dnll:.6e}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("probe", "time", "check", "compare", "nsys", "warps"))
    ap.add_argument("--species", required=False, default=None)
    ap.add_argument("--families", required=False, default=None)
    ap.add_argument("--limit", required=False, type=int, default=100)
    ap.add_argument("--clade-budget", required=False, type=int, default=315_000)
    ap.add_argument("--theta-pt", required=False, default=None)
    ap.add_argument("--theta", required=False, default="fitted", choices=("fitted", "flat-6"))
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--a", required=False, default=None)
    ap.add_argument("--b", required=False, default=None)
    args = ap.parse_args()
    if args.mode == "compare":
        return _mode_compare(args)
    for required in ("species", "families", "theta_pt"):
        if getattr(args, required) is None:
            ap.error(f"--{required.replace('_', '-')} is required for mode {args.mode}")
    if args.mode == "check" and args.out is None:
        ap.error("--out is required for mode check")
    return {"probe": _mode_probe, "time": _mode_time, "check": _mode_check,
            "nsys": _mode_nsys, "warps": _mode_warps}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
