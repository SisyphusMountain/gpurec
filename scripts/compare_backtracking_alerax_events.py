#!/usr/bin/env python
"""Compare gpurec Rust backtracking event counts against AleRax samples."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpurec import GeneReconModel, recphyloxml_event_counts, sample_recphyloxmls
from gpurec.backtracking import EVENT_KEYS
from gpurec.core.model import parse_alerax_family_file


def parse_alerax_event_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        counts[key.strip()] = int(float(value.strip()))
    return counts


def _read_rate_row(
    path: Path,
    *,
    columns: tuple[int, int, int],
    expected_format: str,
) -> tuple[float, float, float]:
    if not path.exists():
        raise FileNotFoundError(f"missing AleRax rate file: {path}")
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 2:
        raise ValueError(
            f"{path} must contain a header row and at least one rate row "
            f"with columns: {expected_format}"
        )
    first_row = rows[1]
    if len(first_row) <= max(columns):
        raise ValueError(
            f"{path} first rate row must have columns: {expected_format}; "
            f"got {first_row!r}"
        )
    try:
        rates = tuple(float(first_row[column]) for column in columns)
    except ValueError as exc:
        raise ValueError(
            f"could not parse D/L/T rates from {path}; expected columns: "
            f"{expected_format}; got {first_row!r}"
        ) from exc
    return rates


def load_rates(output_dir: Path, family_name: str | None = None) -> tuple[float, float, float]:
    if family_name is not None:
        family_rates = output_dir / "model_parameters" / f"{family_name}_rates.txt"
        if family_rates.exists():
            return _read_rate_row(
                family_rates,
                columns=(0, 1, 2),
                expected_format="D L T",
            )

    params = output_dir / "model_parameters" / "model_parameters.txt"
    return _read_rate_row(
        params,
        columns=(1, 2, 3),
        expected_format="node D L T",
    )


def summarize(counts: list[dict[str, int]]) -> dict[str, tuple[int, float, int]]:
    return {
        key: (
            min(row.get(key, 0) for row in counts),
            statistics.mean(row.get(key, 0) for row in counts),
            max(row.get(key, 0) for row in counts),
        )
        for key in EVENT_KEYS
    }


def compare_family(
    *,
    dataset: Path,
    output_dir: Path,
    family_index: int,
    samples: int,
    seed: int,
    fixed_iters_pi: int,
    max_iters_e: int,
    tol_e: float,
    backtrack_binary: Path | None,
    families_file: Path | None,
    species_tree: Path | None,
    preprocess_cache_dir: Path | None,
) -> list[tuple[str, str, int, float, int, int, float, int, float]]:
    if families_file is None:
        family_name = f"family_{family_index:04d}"
    else:
        names, _tree_paths, _leaf_maps = parse_alerax_family_file(
            families_file,
            start=family_index,
            max_families=1,
        )
        family_name = names[0]

    alerax_dir = output_dir / "reconciliations" / "all"
    alerax_counts = [
        parse_alerax_event_counts(path)
        for path in sorted(alerax_dir.glob(f"{family_name}_eventCounts_*.txt"))
    ]
    if not alerax_counts:
        raise FileNotFoundError(f"no AleRax event counts found for {family_name} in {alerax_dir}")

    duplication, loss, transfer = load_rates(output_dir, family_name if families_file else None)
    if families_file is None:
        gene_tree = dataset / f"g_{family_index:04d}.nwk"
        if not gene_tree.exists():
            gene_tree = dataset / f"g_{family_index}.nwk"
        if not gene_tree.exists():
            raise FileNotFoundError(gene_tree)

        model = GeneReconModel.from_trees(
            str(dataset / "sp.nwk"),
            [str(gene_tree)],
            mode="global",
            device="cuda",
            dtype=torch.float64,
            theta_init_rates=(duplication, loss, transfer),
            fixed_iters_Pi=fixed_iters_pi,
            max_iters_E=max_iters_e,
            tol_E=tol_e,
        )
    else:
        if species_tree is None:
            raise ValueError("--species-tree is required when --families-file is used")
        model = GeneReconModel.from_alerax_families(
            str(species_tree),
            families_file,
            mode="global",
            start=family_index,
            max_families=1,
            device="cuda",
            dtype=torch.float64,
            theta_init_rates=(duplication, loss, transfer),
            preprocess_cache_dir=preprocess_cache_dir,
            fixed_iters_Pi=fixed_iters_pi,
            max_iters_E=max_iters_e,
            tol_E=tol_e,
        )
    gpurec_counts = [
        recphyloxml_event_counts(xml)
        for xml in sample_recphyloxmls(
            model,
            family_index=0,
            num_samples=samples,
            seed=seed,
            max_events=100_000,
            backtrack_binary=backtrack_binary,
        )
    ]
    alerax_summary = summarize(alerax_counts)
    gpurec_summary = summarize(gpurec_counts)

    rows = []
    for key in EVENT_KEYS:
        a_min, a_mean, a_max = alerax_summary[key]
        g_min, g_mean, g_max = gpurec_summary[key]
        rows.append((family_name, key, a_min, a_mean, a_max, g_min, g_mean, g_max, g_mean - a_mean))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("tests/data/test_trees_100"))
    parser.add_argument("--output-name", default="output_global")
    parser.add_argument("--families", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-iters-pi", type=int, default=6)
    parser.add_argument("--max-iters-e", type=int, default=4000)
    parser.add_argument("--tol-e", type=float, default=1e-10)
    parser.add_argument("--backtrack-binary", type=Path)
    parser.add_argument("--families-file", type=Path)
    parser.add_argument("--species-tree", type=Path)
    parser.add_argument("--preprocess-cache-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for gpurec's retained forward path")

    output_dir = args.dataset / args.output_name
    print(
        "family\tevent\talerax_min\talerax_mean\talerax_max\t"
        "gpurec_min\tgpurec_mean\tgpurec_max\tmean_delta",
        flush=True,
    )
    for family_index in range(args.start, args.start + args.families):
        for row in compare_family(
            dataset=args.dataset,
            output_dir=output_dir,
            family_index=family_index,
            samples=args.samples,
            seed=args.seed,
            fixed_iters_pi=args.fixed_iters_pi,
            max_iters_e=args.max_iters_e,
            tol_e=args.tol_e,
            backtrack_binary=args.backtrack_binary,
            families_file=args.families_file,
            species_tree=args.species_tree,
            preprocess_cache_dir=args.preprocess_cache_dir,
        ):
            family, key, a_min, a_mean, a_max, g_min, g_mean, g_max, delta = row
            print(
                f"{family}\t{key}\t{a_min}\t{a_mean:.6g}\t{a_max}\t"
                f"{g_min}\t{g_mean:.6g}\t{g_max}\t{delta:.6g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
