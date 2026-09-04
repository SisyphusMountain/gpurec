"""Count Triton re-specializations (one JIT compile each) and check they do not change results.

Three modes, all on a genewise model built exactly like the production first tier
(pi_iters=16, neumann_terms=16, receiver weights frozen):

  --mode diag    record, for every Triton kernel launched during forward + gradient + one
                 ``_analytic_hessian``, the specialization tuple Triton computes per launch
                 (the thing its compile cache is keyed on). Reports per kernel: launches,
                 distinct tuples, and WHICH argument names take more than one value. Also
                 counts the directories the run left in TRITON_CACHE_DIR per kernel name.
  --mode verify  save per-family NLL, the batch-0 forward Pi rows, two identical gradients
                 and two identical Hessians to a .pt file, for a bitwise before/after diff.
  --mode time    mean of 3 warm loss+gradient calls and 3 warm ``_analytic_hessian`` calls.

  --compare A.pt B.pt   print the before/after diff of two ``--mode verify`` files.

Usage:
  python benchmark/cc/spec_variants.py --species S.nwk --families LIST.txt --limit 100 \
      --clade-budget 315000 --mode diag --out /path/out.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
import time

import torch

# Kernels named in the cache-growth report; printed first in the diag summary.
REPORTED_KERNELS = (
    "_transfer_subtree_vjp_directional_derivative_kernel",
    "_accumulate_transfer_subtree_vjp_kernel",
    "_stage_multiple_gene_split_event_reduction_kernel",
    "_exact_tree_pi_self_loop_kernel",
    "_update_reconciliation_likelihood_kernel",
    "_finalize_multiple_gene_split_event_reduction_kernel",
    "_reduce_single_gene_split_events_kernel",
    "_reconciliation_self_loop_transpose_series_kernel",
)

_records: dict[str, dict] = {}


def install_spec_recorder() -> None:
    """Wrap Triton's kernel launch so every launch's specialization tuple is recorded.

    Triton builds that tuple from the argument list (pointer 16-byte alignment, integer
    ``== 1`` / divisible-by-16, constexpr values) and uses it as the compile-cache key, so
    two launches with different tuples are two separate JIT compiles.
    """
    from triton.runtime import driver
    from triton.runtime.jit import JITFunction

    original_run = JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        try:
            device = driver.active.get_current_device()
            binder = self.device_caches[device][4]
            _bound, specialization, _options = binder(*args, **kwargs)
            name = getattr(self, "__name__", None) or self.fn.__name__
            record = _records.get(name)
            if record is None:
                record = _records[name] = {
                    "arg_names": list(self.arg_names),
                    "specs": collections.Counter(),
                }
            record["specs"][tuple(str(entry) for entry in specialization)] += 1
        except Exception as exc:  # recording must never break the run
            _records.setdefault("__recorder_errors__", {"arg_names": [], "specs": collections.Counter()})
            _records["__recorder_errors__"]["specs"][(repr(exc),)] += 1
        return original_run(self, *args, grid=grid, warmup=warmup, **kwargs)

    JITFunction.run = run


def summarize_records() -> dict:
    """Per kernel: launch count, distinct specialization tuples, and the arguments that vary."""
    out = {}
    for name, record in _records.items():
        if name == "__recorder_errors__":
            continue
        specs = record["specs"]
        arg_names = record["arg_names"]
        varying = []
        for position, arg_name in enumerate(arg_names):
            values = {spec[position] for spec in specs if position < len(spec)}
            if len(values) > 1:
                sample = sorted(values)
                varying.append({
                    "arg": arg_name,
                    "position": position,
                    "n_values": len(values),
                    "values": sample[:8],
                })
        out[name] = {
            "launches": sum(specs.values()),
            "distinct_specializations": len(specs),
            "varying_args": varying,
        }
    return out


def count_cache_entries(cache_dir: str) -> dict[str, int]:
    """Count compiled-kernel directories in a Triton cache dir, grouped by kernel name."""
    counts: collections.Counter = collections.Counter()
    if not os.path.isdir(cache_dir):
        return {}
    for entry in os.scandir(cache_dir):
        if not entry.is_dir():
            continue
        names = set()
        for f in os.scandir(entry.path):
            base = f.name
            if base.startswith("__grp__"):
                base = base[len("__grp__"):]
            stem = base.split(".")[0]
            if stem:
                names.add(stem)
        for stem in names:
            counts[stem] += 1
    return dict(counts)


def build_model(species: str, paths: list[str], clade_budget: int):
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER

    options = SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16})
    t0 = time.perf_counter()
    model = GeneReconModel(
        species, paths, mode="genewise", device="cuda", dtype=torch.float32,
        solver_options=options, clade_budget=clade_budget,
    )
    model.receiver_weights.requires_grad_(False)
    print(f"[build] {time.perf_counter() - t0:.1f}s families={len(paths)} "
          f"batches={len(model.batch_statics)}", flush=True)
    return model


def tensor_fingerprint(t: torch.Tensor) -> str:
    """Bit-exact 16-hex-digit fingerprint of a tensor's raw bytes.

    Hashed in 8192-row chunks: the forward Pi table is [clades, species] and can be gigabytes,
    and a single host copy of it would dominate the run's memory.
    """
    t = t.detach().contiguous()
    digest = hashlib.sha256()
    if t.ndim == 0:
        t = t.reshape(1)
    for start in range(0, t.shape[0], 8192):
        digest.update(t[start:start + 8192].cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def forward_pi_rows(model, theta):
    """Batch-0 forward Pi (the wave-ordered [clades, species] table) plus its fingerprint."""
    from gpurec.solver.value_and_grad import forward_solve

    static = model.batch_statics[0]
    _loss, saved = forward_solve([static], theta, model.receiver_weights.detach())
    pi = saved["pi_wave"].detach()
    return pi, tensor_fingerprint(pi)


def run_diag(model, theta, cache_dir: str) -> dict:
    from gpurec.fit.genewise_fit import _analytic_hessian

    loss_vec, _g, _gr = model.genewise_loss_vector_and_grad(theta=theta, need_grad=False)
    torch.cuda.synchronize()
    print(f"[diag] forward nll_sum={float(loss_vec.sum()):.6f}", flush=True)
    model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    print("[diag] gradient done", flush=True)
    _analytic_hessian(model, theta, 16)
    torch.cuda.synchronize()
    print("[diag] hessian done", flush=True)
    return {"kernels": summarize_records(), "cache_entries": count_cache_entries(cache_dir)}


def run_verify(model, theta) -> dict:
    from gpurec.fit.genewise_fit import _analytic_hessian

    loss_vec, _g, _gr = model.genewise_loss_vector_and_grad(theta=theta, need_grad=False)
    pi, pi_hash = forward_pi_rows(model, theta)
    grads = []
    for _ in range(2):
        _l, g, _gr = model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        grads.append(g.detach().cpu().clone())
    hessians = [_analytic_hessian(model, theta, 16).detach().cpu().clone() for _ in range(2)]
    torch.cuda.synchronize()
    return {
        "loss_vec": loss_vec.detach().cpu().clone(),
        "loss_hash": tensor_fingerprint(loss_vec),
        "pi_hash": pi_hash,
        "pi_head": pi[: min(8, pi.shape[0])].detach().cpu().clone(),
        "grads": grads,
        "hessians": hessians,
    }


def run_time(model, theta, repeats: int) -> dict:
    from gpurec.fit.genewise_fit import _analytic_hessian

    model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
    torch.cuda.synchronize()
    grad_times = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t = time.perf_counter()
        model.genewise_loss_vector_and_grad(theta=theta, need_grad=True)
        torch.cuda.synchronize(); grad_times.append(time.perf_counter() - t)
    _analytic_hessian(model, theta, 16)
    torch.cuda.synchronize()
    hess_times = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t = time.perf_counter()
        _analytic_hessian(model, theta, 16)
        torch.cuda.synchronize(); hess_times.append(time.perf_counter() - t)
    return {
        "grad_times": grad_times,
        "hess_times": hess_times,
        "grad_mean": statistics.mean(grad_times),
        "hess_mean": statistics.mean(hess_times),
        "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    }


def compare(path_a: str, path_b: str) -> int:
    a = torch.load(path_a, weights_only=False)
    b = torch.load(path_b, weights_only=False)
    print(f"[cmp] A={path_a}\n[cmp] B={path_b}")
    print(f"[cmp] per-family NLL: hash A={a['loss_hash']} B={b['loss_hash']} "
          f"bitwise_equal={torch.equal(a['loss_vec'], b['loss_vec'])} "
          f"max_abs_diff={float((a['loss_vec'] - b['loss_vec']).abs().max()):.3e}")
    print(f"[cmp] forward Pi:    hash A={a['pi_hash']} B={b['pi_hash']} "
          f"bitwise_equal={a['pi_hash'] == b['pi_hash']} "
          f"head_bitwise_equal={torch.equal(a['pi_head'], b['pi_head'])}")
    for label in ("grads", "hessians"):
        noise_a = float((a[label][0] - a[label][1]).abs().max())
        noise_b = float((b[label][0] - b[label][1]).abs().max())
        scale = float(a[label][0].abs().max())
        cross = float((a[label][0] - b[label][0]).abs().max())
        print(f"[cmp] {label}: run-to-run noise A={noise_a:.3e} B={noise_b:.3e} "
              f"A-vs-B={cross:.3e} scale={scale:.3e} "
              f"within_noise={cross <= max(noise_a, noise_b) or cross == 0.0} "
              f"bitwise_equal={torch.equal(a[label][0], b[label][0])}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--species")
    ap.add_argument("--families")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--clade-budget", type=int)
    ap.add_argument("--mode", choices=("diag", "verify", "time"))
    ap.add_argument("--repeats", type=int, help="warm repeats for --mode time")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)

    for required in ("species", "families", "limit", "clade_budget", "mode", "out"):
        if getattr(args, required) is None:
            ap.error(f"--{required.replace('_', '-')} is required outside --compare")

    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]

    cache_dir = os.environ.get("TRITON_CACHE_DIR", "")
    print(f"[env] mode={args.mode} families={len(paths)} clade_budget={args.clade_budget} "
          f"cache_dir={cache_dir} gpurec={os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}",
          flush=True)

    if args.mode == "diag":
        install_spec_recorder()

    os.environ["GPUREC_WARM_ADJOINT"] = "1"
    model = build_model(args.species, paths, args.clade_budget)
    theta = torch.full((len(paths), 3), -6.0, device="cuda")

    if args.mode == "diag":
        result = run_diag(model, theta, cache_dir)
        summary = result["kernels"]
        cache = result["cache_entries"]
        print("\n[diag] kernel                                                   launches  distinct  cache_dirs")
        ordered = sorted(summary.items(), key=lambda kv: -kv[1]["distinct_specializations"])
        for name, info in ordered:
            print(f"[diag] {name:<58} {info['launches']:>8} {info['distinct_specializations']:>9} "
                  f"{cache.get(name, 0):>11}")
        print("\n[diag] varying arguments (these are what force a re-compile):")
        for name, info in ordered:
            if not info["varying_args"]:
                continue
            print(f"[diag] {name}: distinct={info['distinct_specializations']}")
            for entry in info["varying_args"]:
                print(f"[diag]     arg#{entry['position']:<3} {entry['arg']:<32} "
                      f"n_values={entry['n_values']}  e.g. {entry['values']}")
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1, default=str)
        print(f"\n[diag] wrote {args.out}", flush=True)
    elif args.mode == "verify":
        torch.save(run_verify(model, theta), args.out)
        print(f"[verify] wrote {args.out}", flush=True)
    else:
        if args.repeats is None:
            ap.error("--repeats is required for --mode time")
        result = run_time(model, theta, args.repeats)
        print(f"[time] loss+grad mean={result['grad_mean']:.3f}s of "
              f"{[round(x, 3) for x in result['grad_times']]}")
        print(f"[time] hessian   mean={result['hess_mean']:.3f}s of "
              f"{[round(x, 3) for x in result['hess_times']]}")
        print(f"[time] peak={result['peak_gib']:.1f}GiB")
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
