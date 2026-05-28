#!/usr/bin/env python3
"""Generate a deterministic public end-to-end tutorial dataset for gpurec."""

from __future__ import annotations

import argparse
from pathlib import Path


def _balanced_newick(labels: list[str], *, root_name: str) -> str:
    def _subtree(items: list[str]) -> str:
        if len(items) == 1:
            return f"{items[0]}"
        mid = len(items) // 2
        return f"({_subtree(items[:mid])},{_subtree(items[mid:])})"

    tree = _subtree(labels)
    return f"{tree}{root_name}:0;"


def _build_tutorial_inputs(output_dir: Path, species_count: int) -> None:
    species_names = [f"sp{idx:03d}" for idx in range(1, species_count + 1)]
    species_tree = output_dir / "sp.nwk"
    species_tree.write_text(
        f"{_balanced_newick(species_names, root_name="Root")}\n",
        encoding="utf-8",
    )

    gene_trees = [
        [
            "sp001", "sp002", "sp003", "sp004",
            "sp005", "sp006", "sp007", "sp008",
        ],
        [
            "sp009", "sp010", "sp011", "sp012",
            "sp013", "sp014", "sp015", "sp016",
        ],
    ]
    families = [
        ("tutorial_family_1", "g1.nwk"),
        ("tutorial_family_2", "g2.nwk"),
    ]

    family_entries: list[str] = ["[FAMILIES]\n"]
    mapping_lines: list[str] = []

    for family_index, (family_name, gene_tree_name) in enumerate(families):
        leaves = gene_trees[family_index]
        mapped_leaves = []
        gene_lines: list[str] = []
        for leaf_index, species_name in enumerate(leaves):
            gene_name = f"{species_name}_f{family_index+1}_{leaf_index+1}"
            mapped_leaves.append(gene_name)
            gene_lines.append(f"{species_name}: {gene_name}\n")
        mapping_lines.extend(gene_lines)

        gene_tree = output_dir / gene_tree_name
        gene_tree.write_text(
            _balanced_newick(mapped_leaves, root_name="Gene") + "\n",
            encoding="utf-8",
        )
        family_entries.extend(
            (
                f"- {family_name}\n",
                f"starting_gene_tree = {gene_tree_name}\n",
                "mapping = gene_map.tsv\n\n",
            )
        )

    (output_dir / "gene_map.tsv").write_text(
        "".join(mapping_lines),
        encoding="utf-8",
    )
    (output_dir / "families.txt").write_text(
        "".join(family_entries),
        encoding="utf-8",
    )

    (output_dir / "run.json").write_text(
        """{
  \"species_tree\": \"sp.nwk\",
  \"families_file\": \"families.txt\",
  \"out_dir\": \"output_gpurec\",
  \"mode\": \"genewise\",
  \"dtype\": \"fp32\",
  \"device\": \"cuda\",
  \"steps\": 8,
  \"batch_packing\": \"depth_first_fit\",
  \"family_chunk_size\": 0
}
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
    )
    parser.add_argument("--species-count", type=int, default=261)

    args = parser.parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.species_count <= 256:
        raise ValueError("species_count must be greater than 256")

    _build_tutorial_inputs(output_dir, species_count=args.species_count)


if __name__ == "__main__":
    main()
