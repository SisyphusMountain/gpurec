"""Derive per-genome size (gene content = number of gene families in which a genome is present)
for the 60 archaeal genomes, by parsing the `#leaf-id` block of each `.ale` amalgamation file.
A leaf is named `<genome>_<seqid>` (e.g. `Mhung_839`), so the genome is the prefix before the
first `_`; a genome is "present" in a family iff at least one of that family's leaves carries its
prefix. Writes genome_sizes.json (genome -> #families present) and verifies the count, range, and
genome set are reproducible. This is the covariate used by archaea_exposure_cv.py / exposure_*.py.
"""
import os, sys, json, glob

RW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # gpurec repo root
sys.path.insert(0, f"{RW}/experiments/sanderson_cv")
from run_cv import ARCHAEA_ROOT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ALE_GLOB = str(ARCHAEA_ROOT / "ale_gene_tree_distributions/main_families_ge4seq/*.ale")


def genomes_in_ale(path):
    """The set of genome prefixes appearing in one .ale file's #leaf-id block."""
    genomes = set()
    with open(path) as fh:
        in_block = False
        for line in fh:
            line = line.strip()
            if line == "#leaf-id":
                in_block = True
                continue
            if in_block:
                if line.startswith("#") or not line:   # next section / blank ends the block
                    break
                leaf = line.split("\t")[0].split()[0]   # "<genome>_<seqid>\t<id>"
                genomes.add(leaf.split("_")[0])
    return genomes


def main():
    files = sorted(glob.glob(ALE_GLOB))
    assert files, f"no .ale files under {ALE_GLOB}"
    sizes = {}
    for f in files:
        for g in genomes_in_ale(f):
            sizes[g] = sizes.get(g, 0) + 1
    vals = sorted(sizes.values())
    med = vals[len(vals) // 2]
    print(f"parsed {len(files)} families; {len(sizes)} distinct genomes", flush=True)
    print(f"gene content (# families present): min {vals[0]} median {med} max {vals[-1]}", flush=True)
    top = sorted(sizes.items(), key=lambda kv: -kv[1])[:5]
    print(f"top-5 largest genomes: {top}", flush=True)

    out = os.path.join(HERE, "genome_sizes.json")
    if os.path.exists(out):                              # reproducibility check against the committed file
        old = json.load(open(out))
        same = old == sizes
        print(f"[verify] matches existing genome_sizes.json: {same}"
              f"{'' if same else '  (DIFFERS — overwriting)'}", flush=True)
    json.dump(sizes, open(out, "w"), indent=0)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
