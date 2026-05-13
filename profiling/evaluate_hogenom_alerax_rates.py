from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from gpurec import GeneReconModel
from gpurec.core.model import GeneDataset, parse_alerax_family_file


def _read_reference_likelihoods(path: Path) -> tuple[list[str], dict[str, float]]:
    names: list[str] = []
    values: dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        name, value = line.split()[:2]
        names.append(name)
        values[name] = float(value)
    return names, values


def _read_family_rates(path: Path) -> tuple[float, float, float]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"checkpoint file has no rate line: {path}")
    values = [float(x) for x in lines[1].split()]
    if len(values) < 3:
        raise ValueError(f"checkpoint rate line has fewer than 3 values: {path}")
    return values[0], values[1], values[2]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "reference_rank",
        "family_file_rank",
        "family",
        "D",
        "L",
        "T",
        "alerax_loglik_nats",
        "gpurec_loglik_nats",
        "diff_nats",
        "abs_diff_nats",
        "gpurec_nll_bits",
        "chunk",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alerax = [float(r["alerax_loglik_nats"]) for r in rows]
    gpurec = [float(r["gpurec_loglik_nats"]) for r in rows]
    diff = [float(r["diff_nats"]) for r in rows]
    abs_diff = [abs(x) for x in diff]

    lo = min(min(alerax), min(gpurec))
    hi = max(max(alerax), max(gpurec))
    pad = 0.02 * (hi - lo if hi > lo else 1.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(alerax, gpurec, s=10, alpha=0.65, linewidths=0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1)
    ax.set_xlabel("AleRax log-likelihood (nats)")
    ax.set_ylabel("GPUREC log-likelihood (nats)")
    ax.set_title("HOGENOM AleRax Rates: GPUREC vs AleRax")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "likelihood_scatter.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diff, bins=80, color="#4477AA", alpha=0.85)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("GPUREC - AleRax log-likelihood (nats)")
    ax.set_ylabel("Families")
    ax.set_title("Likelihood Difference Distribution")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "diff_histogram.png", dpi=160)
    plt.close(fig)

    ranks = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(ranks, abs_diff, s=9, alpha=0.7, linewidths=0)
    ax.set_yscale("log")
    ax.set_xlabel("Family rank in AleRax likelihood file")
    ax.set_ylabel("|difference| (nats)")
    ax.set_title("Absolute Likelihood Difference by Family")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "abs_diff_by_family.png", dpi=160)
    plt.close(fig)


def _summarize(rows: list[dict[str, object]], elapsed_s: float, args) -> dict[str, object]:
    alerax_ll = [float(r["alerax_loglik_nats"]) for r in rows]
    gpurec_ll = [float(r["gpurec_loglik_nats"]) for r in rows]
    diffs = [float(r["diff_nats"]) for r in rows]
    abs_diffs = [abs(x) for x in diffs]
    worst = max(rows, key=lambda r: float(r["abs_diff_nats"]))
    rmse = math.sqrt(sum(x * x for x in diffs) / len(diffs))
    within_1e_2 = sum(x <= 1e-2 for x in abs_diffs)
    within_1e_1 = sum(x <= 1e-1 for x in abs_diffs)
    return {
        "families": len(rows),
        "elapsed_s": elapsed_s,
        "fixed_iters_E": args.fixed_iters_e,
        "fixed_iters_Pi": args.fixed_iters_pi,
        "chunk_size": args.chunk_size,
        "dtype": args.dtype,
        "device": args.device,
        "species_tree": str(args.species_tree),
        "total_alerax_loglik_nats": sum(alerax_ll),
        "total_gpurec_loglik_nats": sum(gpurec_ll),
        "total_diff_nats": sum(diffs),
        "max_abs_diff_nats": max(abs_diffs),
        "mean_abs_diff_nats": sum(abs_diffs) / len(abs_diffs),
        "rmse_nats": rmse,
        "within_0.01_nats": within_1e_2,
        "within_0.1_nats": within_1e_1,
        "worst_family": {
            "family": worst["family"],
            "alerax_loglik_nats": worst["alerax_loglik_nats"],
            "gpurec_loglik_nats": worst["gpurec_loglik_nats"],
            "diff_nats": worst["diff_nats"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hogenom-dir",
        type=Path,
        default=Path("tests/data/HOGENOM/hogenom"),
    )
    parser.add_argument(
        "--alerax-output",
        type=Path,
        default=Path("tests/data/HOGENOM/hogenom/output_alerax_corrected"),
    )
    parser.add_argument(
        "--species-tree",
        type=Path,
        default=None,
        help="Species tree used for evaluation. Defaults to hogenom_S.tree.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/data/HOGENOM/hogenom/output_gpurec_alerax_rate_check"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--fixed-iters-e", type=int, default=16)
    parser.add_argument("--fixed-iters-pi", type=int, default=6)
    parser.add_argument("--max-wave-size", type=int, default=32768)
    parser.add_argument("--refresh-preprocess-cache", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    families_file = args.hogenom_dir / "hogenom_families.local.txt"
    species_tree = (
        args.species_tree
        if args.species_tree is not None
        else args.hogenom_dir / "hogenom_S.tree"
    )
    args.species_tree = species_tree
    ref_path = args.alerax_output / "per_fam_likelihoods.txt"
    checkpoint_dir = args.alerax_output / "checkpoint"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref_names, ref_likelihoods = _read_reference_likelihoods(ref_path)
    family_names, tree_paths, leaf_maps = parse_alerax_family_file(families_file)
    by_name = {
        name: (idx, paths, mapping)
        for idx, (name, paths, mapping) in enumerate(
            zip(family_names, tree_paths, leaf_maps)
        )
    }
    names = [name for name in ref_names if name in by_name]
    if args.max_families is not None:
        names = names[: args.max_families]
    if not names:
        raise RuntimeError("no overlapping families to evaluate")

    preprocess_cache = args.out_dir / "preprocess_cache"
    rows: list[dict[str, object]] = []
    t0 = perf_counter()
    ln2 = math.log(2.0)

    for chunk_idx, start in enumerate(range(0, len(names), args.chunk_size)):
        chunk_names = names[start : start + args.chunk_size]
        chunk_tree_paths = [by_name[name][1] for name in chunk_names]
        chunk_leaf_maps = [by_name[name][2] for name in chunk_names]
        rates = [_read_family_rates(checkpoint_dir / f"{name}.txt") for name in chunk_names]
        rate_tensor = torch.tensor(rates, dtype=dtype, device=device)
        theta_init = torch.log2(rate_tensor)

        dataset = GeneDataset(
            species_tree_path=str(species_tree),
            gene_tree_paths=chunk_tree_paths,
            genewise=True,
            specieswise=False,
            dtype=dtype,
            device=device,
            preprocess_cache_dir=preprocess_cache,
            refresh_preprocess_cache=args.refresh_preprocess_cache,
            family_names=chunk_names,
            leaf_species_maps=chunk_leaf_maps,
        )
        model = GeneReconModel(
            dataset=dataset,
            mode="genewise",
            theta_init=theta_init,
            fixed_iters_E=args.fixed_iters_e,
            fixed_iters_Pi=args.fixed_iters_pi,
            max_wave_size=args.max_wave_size,
        )
        with torch.no_grad():
            nll_bits = model.nll_per_family().detach().cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        for local_idx, (name, bits, rate) in enumerate(zip(chunk_names, nll_bits, rates)):
            gpurec_ll = -float(bits) * ln2
            alerax_ll = ref_likelihoods[name]
            diff = gpurec_ll - alerax_ll
            rows.append(
                {
                    "reference_rank": start + local_idx,
                    "family_file_rank": by_name[name][0],
                    "family": name,
                    "D": rate[0],
                    "L": rate[1],
                    "T": rate[2],
                    "alerax_loglik_nats": alerax_ll,
                    "gpurec_loglik_nats": gpurec_ll,
                    "diff_nats": diff,
                    "abs_diff_nats": abs(diff),
                    "gpurec_nll_bits": float(bits),
                    "chunk": chunk_idx,
                }
            )
        elapsed = perf_counter() - t0
        print(
            f"chunk {chunk_idx + 1}/{math.ceil(len(names) / args.chunk_size)} "
            f"families={len(rows)}/{len(names)} elapsed_s={elapsed:.1f}",
            flush=True,
        )

    elapsed = perf_counter() - t0
    rows.sort(key=lambda row: int(row["reference_rank"]))
    _write_csv(args.out_dir / "comparison.csv", rows)
    top = sorted(rows, key=lambda row: float(row["abs_diff_nats"]), reverse=True)[:50]
    _write_csv(args.out_dir / "top_abs_diff.csv", top)
    summary = _summarize(rows, elapsed, args)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_results(rows, args.out_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
