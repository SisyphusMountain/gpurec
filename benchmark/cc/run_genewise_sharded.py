"""Run the genewise fit on several GPUs at once by splitting the family list into shards.

The genewise fit (``gpurec.fit.dtl_fit.fit_dtl(species, paths, "genewise")``) optimizes every gene
family independently -- family f's rates affect only family f -- so cutting the family list into K
disjoint pieces and fitting each piece on its own GPU gives exactly the same answer as one big run:
the total negative log-likelihood is the sum over the shards, and the wall time is the slowest shard
instead of the sum of all of them.

This script only splits, launches and merges. The actual fitting is done by the existing one-GPU
driver ``benchmark/cc/run_genewise.py``, run once per shard as a subprocess with
``CUDA_VISIBLE_DEVICES`` pointing at that shard's GPU.

Shards are balanced by the SUM OF SQUARED CLADE COUNTS, not by family count: the per-family
preprocessing cost grows roughly like the square of the number of clades in the family's conditional
clade distribution, so a handful of huge families would otherwise pin one GPU while the others idle.
The split uses greedy longest-processing-time assignment (sort families heaviest-first, give each one
to whichever shard is currently lightest).

Outputs, for tag T and K shards, all inside --out-dir:
  T.shard<k>.txt    the family list handed to shard k
  T.shard<k>.log    that shard's stdout+stderr (the run_genewise.py progress output)
  T.shard<k>.json   that shard's headline numbers (written by run_genewise.py)
  T.shard<k>.pt     that shard's fitted theta + its own family paths (written by run_genewise.py)
  T.json            merged headline numbers (summed NLL, parent wall time, per-shard summaries)
  T.pt              merged theta [F,3], rows put back in the order of the ORIGINAL family list

Usage:
  python benchmark/cc/run_genewise_sharded.py --species S.nwk --families LIST.txt --limit 0 \
      --clade-counts clade_counts.txt --n-shards 4 --out-dir DIR --tag NAME
``--limit 0`` means all families in the list; ``--limit N`` keeps the first N (smoke runs).
``--clade-counts`` is a text file whose lines are "<clade_count> <relative_path>"; the basename of
that relative path must match the basename of the corresponding entry in the families listfile.
On CC-IN2P3 that file is /sps/biometr/emarsot/gpurec-data/coleman/clade_counts.txt.
"""
from __future__ import annotations

import argparse
import builtins
import heapq
import json
import os
import subprocess
import sys
import time

import torch

REPO_LEVEL_PRINT = builtins.print
_T0 = time.perf_counter()

# Name of the existing one-GPU driver this script shells out to. It lives next to this file; it is
# not a setting because there is exactly one genewise driver and calling any other program here
# would silently change what the shards compute.
DRIVER_BASENAME = "run_genewise.py"


def _timed_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    REPO_LEVEL_PRINT(f"[{time.perf_counter() - _T0:9.2f}s]", *args, **kwargs)


def read_family_list(list_path, limit):
    """Read the families listfile (one .ale path per line, '#' comments skipped).

    ``limit`` of 0 keeps every line; a positive N keeps the first N.
    """
    paths = [ln.strip() for ln in open(list_path) if ln.strip() and not ln.startswith("#")]
    if limit > 0:
        paths = paths[:limit]
    return paths


def read_clade_counts(counts_path):
    """Read the clade-counts file into {basename: clade_count}.

    Each line is "<clade_count> <relative_path>". Only the basename of the relative path is kept,
    because the families listfile stores absolute paths into a different directory layout. A basename
    that appears twice with two different counts is an error rather than a silent last-one-wins.
    """
    counts = {}
    with open(counts_path) as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{counts_path}:{lineno}: expected '<clade_count> <relative_path>', got {line!r}")
            count = int(fields[0])
            name = os.path.basename(fields[1])
            if name in counts and counts[name] != count:
                raise ValueError(
                    f"{counts_path}:{lineno}: {name} appears twice with different clade counts "
                    f"({counts[name]} and {count})")
            counts[name] = count
    return counts


def weights_for_paths(paths, clade_counts):
    """Per-family (clade_count, clade_count**2) for the given family paths.

    The squared count is the balancing weight: per-family preprocessing cost is roughly quadratic in
    the number of clades. A family with no entry in the clade-counts file is a hard error -- guessing
    a weight would quietly unbalance the shards.
    """
    missing = [p for p in paths if os.path.basename(p) not in clade_counts]
    if missing:
        raise KeyError(
            f"{len(missing)} families have no clade count (first 5: {missing[:5]}); "
            f"regenerate the clade-counts file so it covers the whole family list")
    clades = [clade_counts[os.path.basename(p)] for p in paths]
    return clades, [c * c for c in clades]


def build_shards(paths, weights, n_shards):
    """Split ``paths`` into ``n_shards`` index lists balanced by ``weights`` (greedy LPT).

    Sort the families heaviest-first, then hand each one to whichever shard currently has the
    smallest total weight. Returns a list of ``n_shards`` lists of indices into ``paths``; each inner
    list is sorted ascending so a shard's own family list keeps the original relative order.
    Ties are broken by the family's position in the input list, so the split is deterministic.
    """
    if n_shards < 1:
        raise ValueError(f"--n-shards must be at least 1, got {n_shards}")
    if len(paths) != len(weights):
        raise ValueError(f"got {len(paths)} paths but {len(weights)} weights")
    if n_shards > len(paths):
        raise ValueError(
            f"--n-shards {n_shards} exceeds the {len(paths)} families to fit; "
            f"every shard must get at least one family")
    order = sorted(range(len(paths)), key=lambda i: (-weights[i], i))
    heap = [(0, k) for k in range(n_shards)]
    heapq.heapify(heap)
    shards = [[] for _ in range(n_shards)]
    for i in order:
        load, k = heapq.heappop(heap)
        shards[k].append(i)
        heapq.heappush(heap, (load + weights[i], k))
    return [sorted(s) for s in shards]


def launch_shards(python_exe, driver_path, species, out_dir, tag, shard_list_paths):
    """Start one run_genewise.py subprocess per shard, each pinned to its own GPU.

    Shard k gets ``CUDA_VISIBLE_DEVICES=k`` (a copy of the parent environment with that one key
    replaced), so inside the child ``cuda:0`` is physical GPU k. Its stdout and stderr both go to
    ``<out_dir>/<tag>.shard<k>.log``. Returns a list of (Popen, log file handle, start time).
    """
    running = []
    for k, list_path in enumerate(shard_list_paths):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(k)
        cmd = [python_exe, driver_path,
               "--species", species,
               "--families", list_path,
               "--limit", "0",
               "--out-dir", out_dir,
               "--tag", f"{tag}.shard{k}"]
        log_path = os.path.join(out_dir, f"{tag}.shard{k}.log")
        log_fh = open(log_path, "w")
        print(f"[sharded] launching shard {k} on CUDA_VISIBLE_DEVICES={k} -> {log_path}")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        running.append((proc, log_fh, time.perf_counter()))
    return running


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True, help="listfile: one .ale path per line")
    ap.add_argument("--limit", required=True, type=int, help="0 = all families; N = first N of the list")
    ap.add_argument("--clade-counts", required=True,
                    help="text file, lines '<clade_count> <relative_path>'; basenames must match --families")
    ap.add_argument("--n-shards", required=True, type=int, help="number of shards = number of GPUs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    builtins.print = _timed_print

    paths = read_family_list(args.families, args.limit)
    clade_counts = read_clade_counts(args.clade_counts)
    clades, weights = weights_for_paths(paths, clade_counts)
    shards = build_shards(paths, weights, args.n_shards)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[sharded] {len(paths)} families, {args.n_shards} shards, species={args.species}, "
          f"torch {torch.__version__}")
    shard_list_paths = []
    for k, idxs in enumerate(shards):
        sum_clades = sum(clades[i] for i in idxs)
        sum_sq = sum(weights[i] for i in idxs)
        print(f"[sharded] shard {k}: {len(idxs)} families, sum_clades={sum_clades}, "
              f"sum_squared_clades={sum_sq}")
        list_path = os.path.join(args.out_dir, f"{args.tag}.shard{k}.txt")
        with open(list_path, "w") as fh:
            for i in idxs:
                fh.write(paths[i] + "\n")
        shard_list_paths.append(list_path)

    driver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DRIVER_BASENAME)
    if not os.path.exists(driver_path):
        raise FileNotFoundError(f"one-GPU driver not found: {driver_path}")

    t0 = time.perf_counter()
    running = launch_shards(sys.executable, driver_path, args.species, args.out_dir, args.tag,
                            shard_list_paths)
    rcs = []
    shard_walls = []
    for k, (proc, log_fh, t_start) in enumerate(running):
        rc = proc.wait()
        shard_wall = time.perf_counter() - t_start
        log_fh.close()
        rcs.append(rc)
        shard_walls.append(shard_wall)
        print(f"[sharded] shard {k} exit={rc} wall={shard_wall:.1f}s")
    wall = time.perf_counter() - t0

    # Merge. A shard counts as usable only if it exited 0 AND left both of its output files.
    failed_shards = []
    per_shard = []
    theta_by_path = {}
    nll_bits = 0.0
    nll_nats = 0.0
    n_families = 0
    for k in range(args.n_shards):
        json_path = os.path.join(args.out_dir, f"{args.tag}.shard{k}.json")
        pt_path = os.path.join(args.out_dir, f"{args.tag}.shard{k}.pt")
        if rcs[k] != 0 or not os.path.exists(json_path) or not os.path.exists(pt_path):
            failed_shards.append(k)
            print(f"[sharded] shard {k} FAILED (exit={rcs[k]}, json={os.path.exists(json_path)}, "
                  f"pt={os.path.exists(pt_path)}) -- see {args.tag}.shard{k}.log")
            continue
        with open(json_path) as fh:
            s = json.load(fh)
        per_shard.append({
            "shard": k, "exit_code": rcs[k], "shard_wall_s": shard_walls[k],
            "n_families": s["n_families"], "wall_s": s["wall_s"], "nll_bits": s["nll_bits"],
            "n_steps": s["n_steps"], "n_builds": s["n_builds"], "peak_gib": s["peak_gib"],
            "certificate": s["certificate"],
        })
        nll_bits += s["nll_bits"]
        nll_nats += s["nll_nats"]
        n_families += s["n_families"]
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        for row, p in enumerate(blob["paths"]):
            theta_by_path[p] = blob["theta"][row]

    # Put the fitted rows back into the order of the ORIGINAL family list. Families whose shard
    # failed simply have no row; the remaining ones keep their original relative order.
    merged_paths = [p for p in paths if p in theta_by_path]
    if merged_paths:
        theta = torch.stack([theta_by_path[p] for p in merged_paths])
    else:
        theta = torch.empty(0, 3)
    torch.save({"theta": theta, "paths": merged_paths}, os.path.join(args.out_dir, f"{args.tag}.pt"))

    summary = {
        "tag": args.tag, "n_shards": args.n_shards, "n_families": n_families, "wall_s": wall,
        "nll_bits": nll_bits, "nll_nats": nll_nats,
        "per_shard": per_shard, "failed_shards": failed_shards,
    }
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    n_steps = sum(s["n_steps"] for s in per_shard)
    n_builds = sum(s["n_builds"] for s in per_shard)
    peak_gib = max([s["peak_gib"] for s in per_shard], default=0.0)
    slowest = max(shard_walls, default=0.0)
    print(f"[driver] DONE wall={wall:.1f}s slowest_shard={slowest:.1f}s families={n_families} "
          f"nll_bits={nll_bits:.3f} nll_nats={nll_nats:.3f} steps={n_steps} builds={n_builds} "
          f"peak={peak_gib:.2f}GiB failed_shards={failed_shards}")
    return 1 if failed_shards else 0


if __name__ == "__main__":
    sys.exit(main())
