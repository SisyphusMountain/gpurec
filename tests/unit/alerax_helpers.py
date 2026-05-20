from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TinyAleraxInputs:
    root: Path
    species_tree: Path
    families_file: Path
    gene_tree: Path
    mapping_file: Path


def write_tiny_alerax_inputs(
    root: Path,
    *,
    family_name: str = "tiny_family",
    species_tree_name: str = "sp.nwk",
    gene_tree_name: str = "gene.nwk",
    mapping_name: str = "gene.map",
    species_tree: str = "(A:1,B:1)Root;\n",
    gene_tree: str = "(a:1,b:1);\n",
    mapping: str = "A:a\nB:b\n",
    family_lines: Sequence[str] | None = None,
) -> TinyAleraxInputs:
    species_tree_path = root / species_tree_name
    gene_tree_path = root / gene_tree_name
    mapping_path = root / mapping_name
    families_path = root / "families.txt"

    species_tree_path.write_text(species_tree, encoding="utf-8")
    gene_tree_path.write_text(gene_tree, encoding="utf-8")
    mapping_path.write_text(mapping, encoding="utf-8")

    if family_lines is None:
        family_lines = (
            f"starting_gene_tree = {gene_tree_name}",
            f"mapping = {mapping_name}",
        )
    families_path.write_text(
        "\n".join(["[FAMILIES]", f"- {family_name}", *family_lines, ""]),
        encoding="utf-8",
    )

    return TinyAleraxInputs(
        root=root,
        species_tree=species_tree_path,
        families_file=families_path,
        gene_tree=gene_tree_path,
        mapping_file=mapping_path,
    )
