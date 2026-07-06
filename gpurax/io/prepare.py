"""Apply the gene->species mapping file (`parse_mapping`) to a gene tree and
its alignment BEFORE they reach gpurec/libpll.

gpurec derives a leaf's species from the substring before the first `_` in
the leaf name (see `crates/gpurec-preprocess/src/lib.rs`). Real GeneRax
datasets often use gene leaf names that don't encode species this way (e.g.
`g1`, `geneid_00123`), which crashes gpurec ("species not found") or causes
silent mis-reconciliation. GeneRax's mapping file already records gene ->
species for exactly this case; `prepare_family` relabels gene-tree leaves
(and the matching alignment headers, so libpll can still pair sequences to
tips by name) to `<species>_<gene>` using that mapping.
"""

import pathlib

import dendropy

from gpurax.io.families import parse_mapping


def prepare_family(family, tree_path, workdir):
    """Return (tree_path, alignment_path), relabeled if `family.mapping` is set.

    If `family.mapping` is None, the inputs are assumed to already encode
    species as a leaf-name prefix, and the original paths are returned
    unchanged.

    Relabel rule (must be a no-op on already-prefixed inputs): a gene leaf
    `g` mapped to `species` is left unchanged if `g == species` or
    `g.startswith(species + "_")` (already gpurec-compatible); otherwise it
    is renamed to `f"{species}_{g}"`.
    """
    if family.mapping is None:
        return tree_path, family.alignment

    mapping = parse_mapping(family.mapping)

    for species in mapping.values():
        if "_" in species:
            raise ValueError(
                f"family {family.name}: species name {species!r} (from mapping "
                f"{family.mapping}) contains '_', which gpurec cannot recover from "
                "a relabeled leaf name (species are split on the first '_'); "
                "rename the species in the mapping file to remove the '_'"
            )

    rename = {}
    for gene, species in mapping.items():
        if gene == species or gene.startswith(species + "_"):
            rename[gene] = gene
        else:
            rename[gene] = f"{species}_{gene}"

    tree = dendropy.Tree.get(path=tree_path, schema="newick", preserve_underscores=True)
    for leaf in tree.leaf_node_iter():
        name = leaf.taxon.label
        if name not in mapping:
            raise KeyError(
                f"family {family.name}: gene tree leaf {name!r} has no entry in "
                f"mapping file {family.mapping}"
            )
        leaf.taxon.label = rename[name]

    out_tree = pathlib.Path(workdir) / f"{family.name}.mapped.nwk"
    tree.write(
        path=str(out_tree), schema="newick",
        suppress_rooting=True, unquoted_underscores=True,
    )

    out_aln = pathlib.Path(workdir) / f"{family.name}.mapped.fasta"
    _relabel_fasta(family.alignment, rename, family.name, out_aln)

    return str(out_tree), str(out_aln)


def _relabel_fasta(alignment_path, rename, family_name, out_path):
    lines = []
    for line in open(alignment_path):
        if line.startswith(">"):
            header = line[1:].strip()
            if header not in rename:
                raise KeyError(
                    f"family {family_name}: alignment header {header!r} has no "
                    f"entry in the mapping file"
                )
            lines.append(f">{rename[header]}\n")
        else:
            lines.append(line)
    pathlib.Path(out_path).write_text("".join(lines))
