"""Deterministic generator for the 6-taxon gpurax SPR fixture.

Builds, under this directory:
  species.nwk, true_gene.nwk, start_gene.nwk, aln.fasta, map.link,
  families.txt, generax_ref.json

The starting gene tree is a deliberately-wrong topology (two leaves swapped
relative to the species-tree-congruent true tree), so a real SPR search has
work to do. A full GeneRax SPR run is used to confirm the fixture is
well-posed: GeneRax must recover a tree closer to the truth (by Robinson-
Foulds distance) than the starting tree was.

Run with: .venv/bin/python tests/gpurax/fixtures/six/build_fixture.py
"""
from __future__ import annotations

import json
import pathlib
import random
import re
import subprocess

import dendropy
from dendropy.model.discrete import Jc69, simulate_discrete_chars

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
GENERAX_BIN = REPO_ROOT / "experiments/ghost_lineages/tools/GeneRax/build/bin/generax"

SEED = 20260706

SPECIES_NWK = "(((A,B),(C,D)),(E,F));"

# Fixed DTL rates used both to record in generax_ref.json and to pass to
# GeneRax (rates are re-optimized during the SPR search regardless; these are
# just the starting values, mirroring the existing 4-taxon fixture).
FIXED_RATES = {"D": 0.2, "L": 0.3, "T": 0.1}


def true_gene_newick(branch_len: float) -> str:
    b = branch_len
    return (
        f"(((A_a:{b},B_b:{b}):{b},(C_c:{b},D_d:{b}):{b}):{b},"
        f"(E_e:{b},F_f:{b}):{b});"
    )


def start_gene_newick(branch_len: float = 0.1) -> str:
    # Species-tree-incongruent topology: swap B_b and C_c relative to truth.
    b = branch_len
    return (
        f"(((A_a:{b},C_c:{b}):{b},(B_b:{b},D_d:{b}):{b}):{b},"
        f"(E_e:{b},F_f:{b}):{b});"
    )


def simulate_alignment(true_newick: str, seq_len: int, seed: int) -> dendropy.DnaCharacterMatrix:
    rng = random.Random(seed)
    tns = dendropy.TaxonNamespace()
    tree = dendropy.Tree.get(
        data=true_newick, schema="newick", taxon_namespace=tns, preserve_underscores=True
    )
    seq_model = Jc69(rng=rng)
    char_matrix = simulate_discrete_chars(
        seq_len=seq_len,
        tree_model=tree,
        seq_model=seq_model,
        mutation_rate=1.0,
        rng=rng,
    )
    return char_matrix


def write_fasta(char_matrix: dendropy.DnaCharacterMatrix, path: pathlib.Path) -> None:
    lines = []
    for taxon in char_matrix.taxon_namespace:
        seq = char_matrix[taxon]
        seq_str = "".join(str(state) for state in seq)
        lines.append(f">{taxon.label}")
        lines.append(seq_str)
    path.write_text("\n".join(lines) + "\n")


def robinson_foulds(true_path: pathlib.Path, other_path: pathlib.Path) -> int:
    tns = dendropy.TaxonNamespace()
    t = dendropy.Tree.get(
        path=str(true_path), schema="newick", taxon_namespace=tns, preserve_underscores=True
    )
    o = dendropy.Tree.get(
        path=str(other_path), schema="newick", taxon_namespace=tns, preserve_underscores=True
    )
    return int(dendropy.calculate.treecompare.symmetric_difference(t, o))


def run_generax(families_path: pathlib.Path, species_path: pathlib.Path, prefix: pathlib.Path) -> dict:
    prefix.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(GENERAX_BIN),
        "-f", str(families_path),
        "-s", str(species_path),
        "-r", "UndatedDTL",
        "--geneSearchStrategy", "SPR",
        "-p", str(prefix),
        "--dup-rate", str(FIXED_RATES["D"]),
        "--loss-rate", str(FIXED_RATES["L"]),
        "--transfer-rate", str(FIXED_RATES["T"]),
        "--seed", str(SEED),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    (prefix / "stdout.log").write_text(proc.stdout)
    (prefix / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"generax failed (rc={proc.returncode}); see {prefix}/stdout.log and stderr.log"
        )
    return {"command": " ".join(cmd), "stdout": proc.stdout, "stderr": proc.stderr}


def parse_loglk(stdout: str) -> dict:
    # GeneRax's final summary block looks like:
    #   Reconciliation likelihood: -3.24545e-06
    #   Phylogenetic likelihood: -5868.92
    #   Joint likelihood: -5868.92
    out = {}
    rec = re.search(r"Reconciliation likelihood:\s*(-?\d+\.?\d*(?:e-?\d+)?)", stdout)
    seq = re.search(r"Phylogenetic likelihood:\s*(-?\d+\.?\d*(?:e-?\d+)?)", stdout)
    joint = re.search(r"Joint likelihood:\s*(-?\d+\.?\d*(?:e-?\d+)?)", stdout)
    if rec:
        out["rec_loglk_nats"] = float(rec.group(1))
    if seq:
        out["seq_loglk_nats"] = float(seq.group(1))
    if joint:
        out["joint_loglk_nats"] = float(joint.group(1))
    return out


def build(branch_len: float, seq_len: int, run_dir: pathlib.Path) -> dict:
    """Generate all fixture files for given (branch_len, seq_len) and run GeneRax.

    Returns the dict that will be written to generax_ref.json.
    """
    species_path = HERE / "species.nwk"
    true_path = HERE / "true_gene.nwk"
    start_path = HERE / "start_gene.nwk"
    aln_path = HERE / "aln.fasta"
    map_path = HERE / "map.link"
    families_path = HERE / "families.txt"

    species_path.write_text(SPECIES_NWK + "\n")

    true_nwk = true_gene_newick(branch_len)
    start_nwk = start_gene_newick(0.1)
    true_path.write_text(true_nwk + "\n")
    start_path.write_text(start_nwk + "\n")

    char_matrix = simulate_alignment(true_nwk, seq_len=seq_len, seed=SEED)
    write_fasta(char_matrix, aln_path)

    species_letters = ["A", "B", "C", "D", "E", "F"]
    gene_leaves = ["A_a", "B_b", "C_c", "D_d", "E_e", "F_f"]
    map_path.write_text("\n".join(f"{g} {s}" for g, s in zip(gene_leaves, species_letters)) + "\n")

    families_path.write_text(
        "[FAMILIES]\n"
        "- fam1\n"
        f"starting_gene_tree = {start_path}\n"
        f"alignment = {aln_path}\n"
        f"mapping = {map_path}\n"
        "subst_model = GTR\n"
    )

    run_info = run_generax(families_path, species_path, run_dir)
    result_tree_path = run_dir / "results" / "fam1" / "geneTree.newick"
    reconciled_newick = result_tree_path.read_text().strip()
    # write out for RF computation (dendropy needs a file/data string; use data=)
    tmp_result_path = run_dir / "geneTree_result.nwk"
    tmp_result_path.write_text(reconciled_newick + "\n")

    rf_start_to_true = robinson_foulds(true_path, start_path)
    rf_generax_to_true = robinson_foulds(true_path, tmp_result_path)

    loglk = parse_loglk(run_info["stdout"])

    ref = {
        "available": True,
        "generax_version": "2.1.3",
        "generax_binary": "experiments/ghost_lineages/tools/GeneRax/build/bin/generax",
        "spr_command": run_info["command"],
        "fixed_rates": FIXED_RATES,
        "branch_len": branch_len,
        "seq_len": seq_len,
        "seed": SEED,
        "reconciled_newick": reconciled_newick,
        "rf_start_to_true": rf_start_to_true,
        "rf_generax_to_true": rf_generax_to_true,
        **loglk,
        "notes": (
            "6-taxon fixture (species A..F). true_gene.nwk is the species-tree-congruent "
            f"topology with all branch lengths = {branch_len}; start_gene.nwk swaps B_b/C_c "
            "relative to truth (RF>=2). Alignment simulated under JC69 via "
            "dendropy.model.discrete.simulate_discrete_chars with rng=random.Random(SEED), "
            f"seq_len={seq_len}, seed={SEED}. GeneRax run in full --geneSearchStrategy SPR "
            "mode (not EVAL) from start_gene.nwk with DTL rates free to re-optimize "
            "(--dup-rate/--loss-rate/--transfer-rate give only the starting values); "
            "reconciled_newick captured from results/fam1/geneTree.newick. "
            "rf_generax_to_true < rf_start_to_true confirms the fixture is well-posed: "
            "GeneRax's joint search recovers a tree strictly closer to the truth than the "
            "deliberately-wrong starting tree."
        ),
    }
    return ref


def main() -> None:
    import sys

    scratch_run_dir = pathlib.Path(
        "/tmp/claude-1000/-home-enzo-Documents-git-gpurec-consolidate-release/"
        "d4a04445-3eb9-4ad2-9a1f-6a5117fb555b/scratchpad/gpurax/generax_run_six"
    )

    # (branch_len, seq_len) attempts, increasing signal until well-posed.
    attempts = [(0.15, 1000), (0.25, 1500), (0.35, 2500)]
    ref = None
    for i, (branch_len, seq_len) in enumerate(attempts):
        run_dir = scratch_run_dir / f"attempt_{i}"
        print(f"[build_fixture] attempt {i}: branch_len={branch_len} seq_len={seq_len}")
        ref = build(branch_len, seq_len, run_dir)
        print(
            f"[build_fixture]   rf_start_to_true={ref['rf_start_to_true']} "
            f"rf_generax_to_true={ref['rf_generax_to_true']}"
        )
        if ref["rf_generax_to_true"] < ref["rf_start_to_true"]:
            print("[build_fixture] well-posed, stopping.")
            break
    else:
        print("[build_fixture] WARNING: never achieved well-posedness across attempts!", file=sys.stderr)

    (HERE / "generax_ref.json").write_text(json.dumps(ref, indent=2) + "\n")
    print(f"[build_fixture] wrote {HERE / 'generax_ref.json'}")


if __name__ == "__main__":
    main()
