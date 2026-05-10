#!/usr/bin/env python3
"""Controlled hogenom checks for GPUREC/AleRax rate discrepancies."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "hogenom_bench"
DEFAULT_FAMILIES = [631, 956, 410, 208, 276, 0]


def family_name(fid: int) -> str:
    return f"family_{fid:04d}"


def parse_rates(path: Path) -> tuple[float, float, float]:
    lines = path.read_text().splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            return float(parts[1]), float(parts[2]), float(parts[3])
    raise RuntimeError(f"no rate row in {path}")


def parse_single_ll(path: Path, fid: int) -> float:
    fam = family_name(fid)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == fam:
            return float(parts[1])
    raise RuntimeError(f"missing {fam} in {path}")


def write_family_file(path: Path, fid: int) -> None:
    path.write_text(
        f"[FAMILIES]\n- {family_name(fid)}\ngene_tree = g_{fid}.nwk\n",
        encoding="utf-8",
    )


def write_rates_file(path: Path, D: float, L: float, T: float) -> None:
    path.write_text(f"# D L T\n{D:.17g} {L:.17g} {T:.17g}\n", encoding="utf-8")


def run_alerax(cmd: list[str], *, cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "AleRax failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="*", type=int, default=DEFAULT_FAMILIES)
    parser.add_argument(
        "--alerax",
        default=shutil.which("alerax") or str(ROOT / "extra" / "AleRax_modified" / "build" / "bin" / "alerax"),
    )
    parser.add_argument("--out-dir", type=Path, default=DATA / "diagnostics" / "single_family_alerax")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gene-tree-rooting", choices=["UNIFORM", "ROOTED", "MAD"], default="UNIFORM")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()

    gpurec = pd.read_csv(DATA / "gpurec" / "model_rates.csv")
    alerax_batch = pd.read_csv(DATA / "output_global" / "model_rates.csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for fid in args.families:
        fam = family_name(fid)
        work = args.out_dir / fam
        opt_out = work / "alerax_opt"
        fixed_out = work / "alerax_fixed_gpurec"
        start_gp_out = work / "alerax_start_gpurec_opt"
        rates_dir = work / "gpurec_rates"

        if args.force and work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        rates_dir.mkdir(exist_ok=True)

        fam_file = work / "families.txt"
        write_family_file(fam_file, fid)

        gp_row = gpurec.loc[gpurec["family_id"] == fid].iloc[0]
        gp_D, gp_L, gp_T = float(gp_row["D"]), float(gp_row["L"]), float(gp_row["T"])
        gp_nll = float(gp_row["nll"])
        write_rates_file(rates_dir / f"{fam}_rates.txt", gp_D, gp_L, gp_T)

        if not (opt_out / "per_fam_likelihoods.txt").exists():
            run_alerax(
                [
                    args.alerax,
                    "-s",
                    "sp.nwk",
                    "-f",
                    str(fam_file.relative_to(DATA)),
                    "-p",
                    str(opt_out.relative_to(DATA)),
                    "--model-parametrization",
                    "GLOBAL",
                    "--species-tree-search",
                    "SKIP",
                    "--gene-tree-samples",
                    "0",
                    "--gene-tree-rooting",
                    args.gene_tree_rooting,
                ],
                cwd=DATA,
            )

        if not (fixed_out / "per_fam_likelihoods.txt").exists():
            run_alerax(
                [
                    args.alerax,
                    "-s",
                    "sp.nwk",
                    "-f",
                    str(fam_file.relative_to(DATA)),
                    "-p",
                    str(fixed_out.relative_to(DATA)),
                    "--model-parametrization",
                    "GLOBAL",
                    "--d",
                    f"{gp_D:.17g}",
                    "--l",
                    f"{gp_L:.17g}",
                    "--t",
                    f"{gp_T:.17g}",
                    "--fix-rates",
                    "--species-tree-search",
                    "SKIP",
                    "--gene-tree-samples",
                    "0",
                    "--gene-tree-rooting",
                    args.gene_tree_rooting,
                ],
                cwd=DATA,
            )

        if not (start_gp_out / "per_fam_likelihoods.txt").exists():
            run_alerax(
                [
                    args.alerax,
                    "-s",
                    "sp.nwk",
                    "-f",
                    str(fam_file.relative_to(DATA)),
                    "-p",
                    str(start_gp_out.relative_to(DATA)),
                    "--model-parametrization",
                    "GLOBAL",
                    "--d",
                    f"{gp_D:.17g}",
                    "--l",
                    f"{gp_L:.17g}",
                    "--t",
                    f"{gp_T:.17g}",
                    "--species-tree-search",
                    "SKIP",
                    "--gene-tree-samples",
                    "0",
                    "--gene-tree-rooting",
                    args.gene_tree_rooting,
                ],
                cwd=DATA,
            )

        opt_D, opt_L, opt_T = parse_rates(opt_out / "model_parameters" / "model_parameters.txt")
        opt_ll = parse_single_ll(opt_out / "per_fam_likelihoods.txt", fid)
        fixed_ll = parse_single_ll(fixed_out / "per_fam_likelihoods.txt", fid)
        start_gp_D, start_gp_L, start_gp_T = parse_rates(
            start_gp_out / "model_parameters" / "model_parameters.txt"
        )
        start_gp_ll = parse_single_ll(start_gp_out / "per_fam_likelihoods.txt", fid)
        opt_nll = -opt_ll / math.log(2.0)
        fixed_nll = -fixed_ll / math.log(2.0)
        start_gp_nll = -start_gp_ll / math.log(2.0)

        if fid < len(alerax_batch):
            batch_row = alerax_batch.iloc[fid]
            batch_D = float(batch_row["D"])
            batch_L = float(batch_row["L"])
            batch_T = float(batch_row["T"])
        else:
            batch_D = batch_L = batch_T = math.nan

        row = {
            "family_id": fid,
            "family": fam,
            "gpurec_D": gp_D,
            "gpurec_L": gp_L,
            "gpurec_T": gp_T,
            "gpurec_nll_bits": gp_nll,
            "alerax_batch_D": batch_D,
            "alerax_batch_L": batch_L,
            "alerax_batch_T": batch_T,
            "alerax_single_D": opt_D,
            "alerax_single_L": opt_L,
            "alerax_single_T": opt_T,
            "alerax_single_nll_bits": opt_nll,
            "alerax_fixed_gpurec_nll_bits": fixed_nll,
            "alerax_start_gpurec_D": start_gp_D,
            "alerax_start_gpurec_L": start_gp_L,
            "alerax_start_gpurec_T": start_gp_T,
            "alerax_start_gpurec_nll_bits": start_gp_nll,
            "gpurec_minus_alerax_fixed_bits": gp_nll - fixed_nll,
            "alerax_fixed_minus_single_bits": fixed_nll - opt_nll,
            "alerax_start_gpurec_minus_single_bits": start_gp_nll - opt_nll,
        }
        rows.append(row)
        print(
            f"{fam}: batch=({batch_D:.6g},{batch_L:.6g},{batch_T:.6g}) "
            f"single=({opt_D:.6g},{opt_L:.6g},{opt_T:.6g}) "
            f"start_gp=({start_gp_D:.6g},{start_gp_L:.6g},{start_gp_T:.6g}) "
            f"gpurec=({gp_D:.6g},{gp_L:.6g},{gp_T:.6g}) "
            f"nll gp={gp_nll:.3f} fixed={fixed_nll:.3f} "
            f"single={opt_nll:.3f} start_gp={start_gp_nll:.3f}",
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
