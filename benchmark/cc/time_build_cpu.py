"""Time the CPU-side model build for N families, stage by stage (no GPU needed).

Stages: (1) Rust `preprocess_dataset` (parses .ale files, builds CCPs, packs batches, plans waves,
serializes JSON); (2) `json.loads` of that payload; (3) the Python-side conversion of every batch's
wave plan into tensors (`build_wave_layout_from_plan`, device=cpu).

Usage: python benchmark/cc/time_build_cpu.py --species S --families LIST --limit N --clade-budget B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--limit", required=True, type=int)
    ap.add_argument("--clade-budget", required=True, type=int)
    args = ap.parse_args()
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    if args.limit > 0:
        paths = paths[: args.limit]
    print(f"[build] {len(paths)} families, clade_budget={args.clade_budget}, cpus={os.cpu_count()}", flush=True)

    from gpurec.core.scheduling.batching import _load_native_module, build_wave_layout_from_plan

    native = _load_native_module()
    t0 = time.perf_counter()
    s = native.preprocess_dataset(args.species, paths, 300, args.clade_budget, "depth_first_fit", 8192, None)
    t1 = time.perf_counter()
    print(f"[build] rust preprocess_dataset: {t1 - t0:.2f}s  json payload {len(s) / 2**20:.1f} MiB", flush=True)
    raw = json.loads(s)
    t2 = time.perf_counter()
    print(f"[build] json.loads: {t2 - t1:.2f}s", flush=True)
    del s
    batches = raw["batches"]
    layouts = raw["batch_wave_layouts"]
    clades = [sum(len(raw["families"][i]["clade_parent"]) if "clade_parent" in raw["families"][i] else 0 for i in b) for b in batches]
    print(f"[build] batches={len(batches)} families/batch={[len(b) for b in batches][:12]}...", flush=True)
    t3 = time.perf_counter()
    n_waves = []
    for plan in layouts:
        wl = build_wave_layout_from_plan({"plan": plan["plan"], "log_split_probs_sorted": plan["log_split_probs_sorted"]}
                                         if "plan" in plan else plan, device="cpu")
        n_waves.append(len(wl["wave_metas"]) if isinstance(wl, dict) and "wave_metas" in wl else -1)
    t4 = time.perf_counter()
    print(f"[build] python wave-layout conversion (all batches): {t4 - t3:.2f}s  waves/batch={n_waves[:12]}...", flush=True)
    fam0 = raw["families"][0]
    print(f"[build] family payload keys: {list(fam0.keys())[:20]}", flush=True)
    print(f"[build] layout payload keys: {list(layouts[0].keys())[:20]}", flush=True)
    print(f"[build] TOTAL {t4 - t0:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
