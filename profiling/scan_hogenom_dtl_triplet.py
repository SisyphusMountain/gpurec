#!/usr/bin/env python3
"""Scan same-rate DTL likelihoods for GPUREC and AleRax on HOGENOM families."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

import torch

from gpurec import GeneReconModel


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "hogenom_bench"


def family_name(fid: int) -> str:
    return f"family_{fid:04d}"


def parse_alerax_nll_bits(path: Path, fid: int) -> float:
    fam = family_name(fid)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == fam:
            return -float(parts[1]) / math.log(2.0)
    raise RuntimeError(f"missing {fam} in {path}")


def run_alerax_fixed(
    *,
    alerax: str,
    fid: int,
    rate: float,
    fam_file: Path,
    out_dir: Path,
) -> float:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        alerax,
        "-s",
        "sp.nwk",
        "-f",
        str(fam_file.relative_to(DATA)),
        "-p",
        str(out_dir.relative_to(DATA)),
        "--model-parametrization",
        "GLOBAL",
        "--d",
        f"{rate:.17g}",
        "--l",
        f"{rate:.17g}",
        "--t",
        f"{rate:.17g}",
        "--fix-rates",
        "--species-tree-search",
        "SKIP",
        "-g",
        "0",
    ]
    result = subprocess.run(cmd, cwd=DATA, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"AleRax failed for {family_name(fid)} rate={rate}\n"
            f"stdout:\n{result.stdout[-3000:]}\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )
    return parse_alerax_nll_bits(out_dir / "per_fam_likelihoods.txt", fid)


def gpurec_nll_bits(fid: int, rate: float, *, cache_dir: Path) -> float:
    model = GeneReconModel.from_trees(
        species_tree=str(DATA / "sp.nwk"),
        gene_trees=[str(DATA / f"g_{fid}.nwk")],
        mode="global",
        pibar_mode="uniform",
        device="cuda",
        dtype=torch.float32,
        theta_init_rates=(rate, rate, rate),
        preprocess_cache_dir=str(cache_dir),
        fixed_iters_Pi=None,
        max_iters_Pi=2000,
        tol_Pi=1e-7,
        max_iters_E=2000,
        tol_E=1e-8,
        use_pruning=False,
        max_wave_size=32768,
    )
    with torch.no_grad():
        return float(model.nll().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="*", type=int, default=[631, 276, 0])
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--min-positive", type=float, default=1e-10)
    parser.add_argument("--out-dir", type=Path, default=DATA / "diagnostics" / "dtl_triplet_scan")
    parser.add_argument("--alerax", default=shutil.which("alerax") or "alerax")
    args = parser.parse_args()

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = DATA / ".gpurec_triplet_scan_cache"

    labels = [2.0 * i / (args.points - 1) for i in range(args.points)]
    rows: list[dict[str, object]] = []

    for fid in args.families:
        fam = family_name(fid)
        fam_dir = args.out_dir / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        fam_file = fam_dir / "families.txt"
        fam_file.write_text(f"[FAMILIES]\n- {fam}\ngene_tree = g_{fid}.nwk\n", encoding="utf-8")

        for label in labels:
            rate = args.min_positive if label == 0.0 else label
            point_dir = fam_dir / f"rate_{label:.3f}".replace(".", "_")
            alerax_nll = run_alerax_fixed(
                alerax=args.alerax,
                fid=fid,
                rate=rate,
                fam_file=fam_file,
                out_dir=point_dir,
            )
            gpurec_nll = gpurec_nll_bits(fid, rate, cache_dir=cache_dir)
            row = {
                "family_id": fid,
                "family": fam,
                "rate_label": label,
                "rate_used": rate,
                "D": rate,
                "L": rate,
                "T": rate,
                "alerax_nll_bits": alerax_nll,
                "gpurec_nll_bits": gpurec_nll,
                "gpurec_minus_alerax_bits": gpurec_nll - alerax_nll,
                "abs_diff_bits": abs(gpurec_nll - alerax_nll),
            }
            rows.append(row)
            print(
                f"{fam} x={label:.2f}: "
                f"AleRax={alerax_nll:.6f} GPUREC={gpurec_nll:.6f} "
                f"diff={gpurec_nll - alerax_nll:.6f}",
                flush=True,
            )

    out_csv = args.out_dir / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
