#!/usr/bin/env python
"""gpurax vs GeneRax — RF-to-true-tree benchmark.

Paired, per amino-acid length L:
  true gene tree --pyvolve LG(L)--> alignment
  alignment --FastTree -lg--> ML start tree      (sequence-only floor; shared start)
  alignment + start + species tree --GeneRax(UndatedDTL,SPR)--> corrected tree
  alignment + start + species tree --gpurax(UndatedDTL,SPR)--> corrected tree
  normalized RF-to-truth (ete3, unrooted, internal labels stripped) for all three.

Both tools see the IDENTICAL alignment, start tree, species tree and rec model;
only the reconciliation engine + search differ. GeneRax and gpurax need different
mapping-file formats, so each gets its own mapping + families file (everything
else shared).

Run in .venv (has torch, pyvolve, ete3, gpurec, gpurax). External binaries:
FastTree (phylo env) and the prebuilt GeneRax. Resumable per (family, L).

Usage:
  .venv/bin/python experiments/gpurax_vs_generax/run_benchmark.py \
      [--n-families 1] [--lengths 55,120,300] [--max-spr-radius 5] [--np 2]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/enzo/Documents/git/gpurec/consolidate-release")
GL = REPO / "experiments" / "ghost_lineages"
FT = "/home/enzo/miniforge3/envs/phylo/bin/FastTree"
GENERAX = str(GL / "tools" / "GeneRax" / "build" / "bin" / "generax")
SP = str(GL / "results" / "sim_allgenes" / "species_tree.newick")
GENES = GL / "results" / "sim_allgenes" / "genes"
FAMILIES_LIST = GL / "results" / "sim_inferred_L1000" / "families.txt"
OUT = REPO / "experiments" / "gpurax_vs_generax" / "results"


# ---------- tree helpers (inlined; see eff_lib.norm_rf) ----------
def strip_internal(nwk):
    """Drop internal-node labels/support so RF compares topology only."""
    return re.sub(r"\)[A-Za-z0-9_]+:", "):", nwk.strip())


def norm_rf(true_nwk, inf_nwk):
    from ete3 import Tree

    r = Tree(strip_internal(true_nwk), format=1).robinson_foulds(
        Tree(inf_nwk.strip(), format=1), unrooted_trees=True)
    return r[0] / r[1] if r[1] else 0.0


def leaves_of(nwk):
    from ete3 import Tree

    return [lf.name for lf in Tree(nwk.strip(), format=1).get_leaves()]


def species_of(leaf):
    """Leaf names are '<species>_<gene>'; species is the prefix before '_'."""
    return leaf.split("_", 1)[0]


# ---------- stage 1: alignment + FastTree start ----------
def align_and_fasttree(name, L, true_nwk, adir, fdir, seed):
    import numpy as np
    import pyvolve

    fa = adir / f"{name}.fasta"
    ft = fdir / f"{name}.nwk"
    if not fa.exists():
        np.random.seed(seed)
        pyvolve.Evolver(
            partitions=pyvolve.Partition(models=pyvolve.Model("LG"), size=L),
            tree=pyvolve.read_tree(tree=strip_internal(true_nwk)),
        )(seqfile=str(fa), ratefile=None, infofile=None)
    if not ft.exists():
        with open(ft, "w") as fo:
            subprocess.run([FT, "-lg", "-quiet", "-nopr", str(fa)],
                           stdout=fo, stderr=subprocess.DEVNULL, check=True)
    return fa, ft


# ---------- mapping files (two formats) ----------
def write_generax_map(name, start_nwk, mdir):
    """GeneRax .link: 'species:gene1;gene2;...' per species."""
    by_sp = {}
    for lf in leaves_of(start_nwk):
        by_sp.setdefault(species_of(lf), []).append(lf)
    p = mdir / f"{name}.link"
    p.write_text("".join(f"{sp}:{';'.join(gs)}\n" for sp, gs in by_sp.items()))
    return p


def write_gpurax_map(name, start_nwk, mdir):
    """gpurax/Treerecs: 'gene species' per line."""
    p = mdir / f"{name}.map"
    p.write_text("".join(f"{lf} {species_of(lf)}\n" for lf in leaves_of(start_nwk)))
    return p


def write_families(path, name, start, aln, mapping):
    path.write_text(
        "[FAMILIES]\n"
        f"- {name}\n"
        f"starting_gene_tree = {start}\n"
        f"alignment = {aln}\n"
        f"mapping = {mapping}\n"
        "subst_model = LG\n")
    return path


# ---------- stage 2a: GeneRax ----------
def run_generax(names, starts, alns, ldir, max_radius, np_ranks):
    mdir = ldir / "gx_map"; mdir.mkdir(exist_ok=True)
    gxout = ldir / "gx_out"
    fam = ldir / "gx_families.txt"
    have = {n for n in names if (gxout / "results" / n / "geneTree.newick").exists()}
    if len(have) < len(names):
        with open(fam, "w") as fo:
            fo.write("[FAMILIES]\n")
            for n in names:
                write_generax_map(n, starts[n].read_text(), mdir)
                fo.write(f"- {n}\nstarting_gene_tree = {starts[n]}\n"
                         f"alignment = {alns[n]}\nmapping = {mdir/f'{n}.link'}\n"
                         "subst_model = LG\n")
        shutil.rmtree(gxout, ignore_errors=True)  # GeneRax refuses to overwrite
        cmd = ["mpirun", "-np", str(np_ranks), GENERAX, "--families", str(fam),
               "--species-tree", SP, "--rec-model", "UndatedDTL", "--per-family-rates",
               "--geneSearchStrategy", "SPR", "--max-spr-radius", str(max_radius),
               "--unrooted-gene-tree", "--seed", "1", "--prefix", str(gxout)]
        t0 = time.time()
        with open(ldir / "generax.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"    GeneRax: {time.time()-t0:.0f}s", flush=True)
    out = {}
    for n in names:
        gp = gxout / "results" / n / "geneTree.newick"
        if gp.exists():
            out[n] = gp.read_text().strip()
    return out


# ---------- stage 2b: gpurax ----------
def run_gpurax(names, starts, alns, ldir, max_radius):
    mdir = ldir / "gpx_map"; mdir.mkdir(exist_ok=True)
    gpxout = ldir / "gpx_out"
    fam = ldir / "gpx_families.txt"
    have = {n for n in names if (gpxout / f"{n}.reconciled.nwk").exists()}
    if len(have) < len(names):
        with open(fam, "w") as fo:
            fo.write("[FAMILIES]\n")
            for n in names:
                write_gpurax_map(n, starts[n].read_text(), mdir)
                fo.write(f"- {n}\nstarting_gene_tree = {starts[n]}\n"
                         f"alignment = {alns[n]}\nmapping = {mdir/f'{n}.map'}\n"
                         "subst_model = LG\n")
        shutil.rmtree(gpxout, ignore_errors=True)
        gpxout.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "gpurax", "-f", str(fam), "-s", SP,
               "-p", str(gpxout), "-r", "UndatedDTL",
               "--max-spr-radius", str(max_radius), "--device", "cuda"]
        t0 = time.time()
        with open(ldir / "gpurax.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=str(REPO))
        print(f"    gpurax:  {time.time()-t0:.0f}s", flush=True)
    out = {}
    for n in names:
        gp = gpxout / f"{n}.reconciled.nwk"
        if gp.exists():
            out[n] = gp.read_text().strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-families", type=int, default=1)
    ap.add_argument("--lengths", default="55,120,300")
    ap.add_argument("--max-spr-radius", type=int, default=5)
    ap.add_argument("--np", type=int, default=2, help="GeneRax MPI ranks")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    all_names = [ln.strip() for ln in open(FAMILIES_LIST) if ln.strip()]
    names = all_names[:args.n_families]
    true = {n: (GENES / f"{n}.nwk").read_text().strip() for n in names}
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"families={names}  lengths={lengths}  radius={args.max_spr_radius}", flush=True)

    results = {}  # (name, L) -> dict
    for L in lengths:
        print(f"\n=== L={L} ===", flush=True)
        ldir = OUT / f"L{L}"; adir = ldir / "align"; fdir = ldir / "fasttree"
        for d in (adir, fdir):
            d.mkdir(parents=True, exist_ok=True)
        starts, alns, rf_start = {}, {}, {}
        for i, n in enumerate(names):
            fa, ft = align_and_fasttree(n, L, true[n], adir, fdir,
                                        (L * 1_000_003 + i) % 2**31)
            alns[n], starts[n] = fa, ft
            rf_start[n] = norm_rf(true[n], ft.read_text())
        gx = run_generax(names, starts, alns, ldir, args.max_spr_radius, args.np)
        gpx = run_gpurax(names, starts, alns, ldir, args.max_spr_radius)
        for n in names:
            row = dict(name=n, L=L, rf_start=rf_start[n],
                       rf_generax=norm_rf(true[n], gx[n]) if n in gx else None,
                       rf_gpurax=norm_rf(true[n], gpx[n]) if n in gpx else None,
                       rf_gpurax_vs_generax=(norm_rf(gx[n], gpx[n])
                                             if n in gx and n in gpx else None))
            results[f"{n}@L{L}"] = row
            print(f"  {n}: start={row['rf_start']:.3f}  "
                  f"generax={row['rf_generax']}  gpurax={row['rf_gpurax']}  "
                  f"(gpx-vs-gx={row['rf_gpurax_vs_generax']})", flush=True)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    _summarize(results, lengths)


def _summarize(results, lengths):
    import statistics as st
    print("\n=== summary (mean normalized RF-to-true) ===")
    print(f"{'L':>5} {'n':>3} {'start':>7} {'generax':>8} {'gpurax':>7}   win/tie/loss (gpx vs gx)")
    for L in lengths:
        rows = [r for r in results.values() if r["L"] == L
                and r["rf_generax"] is not None and r["rf_gpurax"] is not None]
        if not rows:
            print(f"{L:>5}  (no complete rows)")
            continue
        ms = st.mean(r["rf_start"] for r in rows)
        mg = st.mean(r["rf_generax"] for r in rows)
        mp = st.mean(r["rf_gpurax"] for r in rows)
        w = sum(r["rf_gpurax"] < r["rf_generax"] - 1e-9 for r in rows)
        t = sum(abs(r["rf_gpurax"] - r["rf_generax"]) <= 1e-9 for r in rows)
        loss = len(rows) - w - t
        print(f"{L:>5} {len(rows):>3} {ms:>7.3f} {mg:>8.3f} {mp:>7.3f}   {w}/{t}/{loss}")


if __name__ == "__main__":
    main()
