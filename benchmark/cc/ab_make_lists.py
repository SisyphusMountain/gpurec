"""Build the family list files for the archaea / HOGENOM old-vs-new comparison.

Writes plain one-path-per-line list files (the format `run_genewise.py --families` and
`gpurec.fit.genewise_fit._resolve_gene_trees` both read) and prints, for every list, how many
families went in, how many were kept, and why any were dropped. Nothing is silently skipped.

A gene leaf is named `<Species>_<rest>` (archaea `Mhung_839`, HOGENOM `CLOCE_1.PE2286`), so the
species a leaf belongs to is the text before the FIRST underscore -- the same rule the Rust
preprocessor and AleRax both use. "Covered species" of a family = the distinct species in its
representative (first) gene tree.

Lists written:
  archaea-all    every .ale file in the archaea directory
  archaea-ge4sp  the subset covering at least --min-species species
  hogenom-1055   the 1055-family subset, in the order of the name list
  hogenom-full   every family named in the AleRax per-family likelihood file

Usage (all arguments required; there are no defaults):
  python benchmark/cc/ab_make_lists.py \
      --archaea-ale-dir DIR --archaea-out-all F --archaea-out-ge4sp F \
      --hogenom-families-root DIR --hogenom-gene-tree-name ufboot1000.MFP.geneTree.newick \
      --hogenom-name-list F --hogenom-alerax-likelihoods F \
      --hogenom-out-1055 F --hogenom-out-full F --min-species 4
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# A leaf token is `<Species>_<rest>` and appears right after a '(' or a ',' in the Newick string.
# Same split-on-first-underscore rule as the Rust preprocessor (lib.rs: leaf_name.split_once('_')).
_SPECIES = re.compile(r"[(,]([A-Za-z][A-Za-z0-9]*)_")


def _representative_tree_line(path: str) -> str:
    """The family's representative gene tree: line 2 of an .ale file, line 1 of a Newick sample."""
    with open(path) as handle:
        first = handle.readline()
        if first.startswith("#"):          # .ale: '#constructor_string' then the tree
            return handle.readline()
        return first


def _covered_species(path: str) -> set[str]:
    return set(_SPECIES.findall(_representative_tree_line(path)))


def _write(out_path: str, paths: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as handle:
        for path in paths:
            handle.write(path + "\n")


def _report(label: str, considered: int, kept: list[str], missing: list[str],
            thin: list[str], min_species: int, out_path: str) -> None:
    print(f"[lists] {label}: considered={considered} kept={len(kept)} "
          f"missing_file={len(missing)} below_{min_species}_species={len(thin)} -> {out_path}",
          flush=True)
    for name in missing[:10]:
        print(f"[lists]   missing: {name}", flush=True)
    for name in thin[:10]:
        print(f"[lists]   < {min_species} species: {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archaea-ale-dir", required=True)
    parser.add_argument("--archaea-out-all", required=True)
    parser.add_argument("--archaea-out-ge4sp", required=True)
    parser.add_argument("--hogenom-families-root", required=True)
    parser.add_argument("--hogenom-gene-tree-name", required=True)
    parser.add_argument("--hogenom-name-list", required=True)
    parser.add_argument("--hogenom-alerax-likelihoods", required=True)
    parser.add_argument("--hogenom-out-1055", required=True)
    parser.add_argument("--hogenom-out-full", required=True)
    parser.add_argument("--min-species", required=True, type=int)
    args = parser.parse_args()

    ale_files = sorted(glob.glob(os.path.join(args.archaea_ale_dir, "*.ale")))
    _write(args.archaea_out_all, ale_files)
    print(f"[lists] archaea-all: {len(ale_files)} .ale files -> {args.archaea_out_all}", flush=True)
    ge4, thin = [], []
    for path in ale_files:
        (ge4 if len(_covered_species(path)) >= args.min_species else thin).append(path)
    _write(args.archaea_out_ge4sp, ge4)
    _report("archaea-ge4sp", len(ale_files), ge4, [], [os.path.basename(p) for p in thin],
            args.min_species, args.archaea_out_ge4sp)

    def hogenom_path(family: str) -> str:
        return os.path.join(args.hogenom_families_root, family, "gene_trees",
                            args.hogenom_gene_tree_name)

    def build(names: list[str], out_path: str, label: str) -> None:
        kept, missing, thin_names = [], [], []
        for name in names:
            path = hogenom_path(name)
            if not os.path.exists(path):
                missing.append(name)
            elif len(_covered_species(path)) < args.min_species:
                thin_names.append(name)
            else:
                kept.append(path)
        _write(out_path, kept)
        _report(label, len(names), kept, missing, thin_names, args.min_species, out_path)

    names_1055 = [line.strip() for line in open(args.hogenom_name_list)
                  if line.strip() and not line.startswith("#")]
    build(names_1055, args.hogenom_out_1055, "hogenom-1055")

    names_full = [line.split()[0] for line in open(args.hogenom_alerax_likelihoods)
                  if line.strip() and not line.startswith("#")]
    build(names_full, args.hogenom_out_full, "hogenom-full")
    return 0


if __name__ == "__main__":
    sys.exit(main())
