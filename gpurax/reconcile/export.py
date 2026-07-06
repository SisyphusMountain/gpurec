import pathlib
import xml.etree.ElementTree as ET


def write_newick(out_dir, name, newick_str):
    p = pathlib.Path(out_dir) / f"{name}.reconciled.nwk"
    p.write_text(newick_str.strip() + "\n")
    return str(p)

def write_scores(out_dir, rows):
    p = pathlib.Path(out_dir) / "scores.tsv"
    cols = ["name", "joint", "rec", "seq"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r[c]) for c in cols))
    p.write_text("\n".join(lines) + "\n")
    return str(p)


class _NwkNode:
    __slots__ = ("name", "children")

    def __init__(self, name, children):
        self.name = name
        self.children = children


def _parse_newick(newick_str):
    """Minimal recursive-descent Newick parser (no external deps).

    Ignores branch lengths; keeps node labels (leaf and, if present, internal).
    """
    text = newick_str.strip()
    if text.endswith(";"):
        text = text[:-1]
    n = len(text)
    pos = 0

    def parse_label():
        nonlocal pos
        start = pos
        while pos < n and text[pos] not in ",():;":
            pos += 1
        return text[start:pos].strip()

    def skip_branch_length():
        nonlocal pos
        if pos < n and text[pos] == ":":
            pos += 1
            parse_label()  # numeric branch length, discarded

    def parse_clade():
        nonlocal pos
        children = []
        if pos < n and text[pos] == "(":
            pos += 1
            while True:
                children.append(parse_clade())
                if pos < n and text[pos] == ",":
                    pos += 1
                    continue
                break
            if pos < n and text[pos] == ")":
                pos += 1
        name = parse_label()
        skip_branch_length()
        return _NwkNode(name, children)

    return parse_clade()


def _postorder(node, out):
    for child in node.children:
        _postorder(child, out)
    out.append(node)
    return out


def _species_index_to_name(species_newick):
    """Index -> species name, using gpurec's own numbering scheme: species nodes
    are indexed 0..S-1 in postorder (children before parent) of the species tree
    (see `build_species_output` in crates/gpurec-preprocess/src/lib.rs). Unnamed
    internal nodes (the common case for species trees with no internal labels)
    have no real name available; callers should fall back to `species_<idx>`.
    """
    root = _parse_newick(species_newick)
    nodes = _postorder(root, [])
    return {idx: node.name for idx, node in enumerate(nodes) if node.name}, root


def _species_name(index_to_name, species_index):
    name = index_to_name.get(species_index)
    return name if name else f"species_{species_index}"


def _build_sptree_clade(node):
    clade = ET.Element("clade")
    ET.SubElement(clade, "name").text = node.name if node.name else "NULL"
    for child in node.children:
        clade.append(_build_sptree_clade(child))
    return clade


_EVENT_TAGS = {
    "leaf": "leaf",
    "speciation": "speciation",
    "duplication": "duplication",
    "loss": "loss",
}


def _transfer_destination_index(recon_nodes, node_index):
    """The recipient species of a `transfer` node: the child whose species index
    differs from the transfer node's own (donor) species index. Mirrors the
    donor/recipient child layout produced by the backtracker (see
    `crates/gpurec-backtrack/src/lib.rs`, `TransferLossDonor` / `SplitTransfer`).
    """
    _event, donor_species, children = recon_nodes[node_index]
    for child_index in children:
        if child_index == -1:
            continue
        _child_event, child_species, _child_children = recon_nodes[child_index]
        if child_species != donor_species:
            return child_species
    return donor_species


def _build_genetree_clade(recon_nodes, node_index, index_to_name):
    event, species_index, children = recon_nodes[node_index]
    clade = ET.Element("clade")
    ET.SubElement(clade, "name").text = "NULL"
    events_rec = ET.SubElement(clade, "eventsRec")

    if event == "transfer":
        dest_index = _transfer_destination_index(recon_nodes, node_index)
        ET.SubElement(
            events_rec, "transferBack",
            destinationSpecies=_species_name(index_to_name, dest_index),
        )
    elif event in _EVENT_TAGS:
        ET.SubElement(
            events_rec, _EVENT_TAGS[event],
            speciesLocation=_species_name(index_to_name, species_index),
        )
    else:
        raise ValueError(f"unknown reconciliation event {event!r}")

    for child_index in children:
        if child_index != -1:
            clade.append(_build_genetree_clade(recon_nodes, child_index, index_to_name))
    return clade


def _notung_node_label(event, species_index, node_index, index_to_name):
    if event == "leaf":
        return _species_name(index_to_name, species_index)
    if event == "loss":
        return f"LOST_{_species_name(index_to_name, species_index)}"
    return f"n{node_index}"


def _notung_nhx_comment(recon_nodes, node_index, index_to_name):
    """Build the `[&&NHX:...]` comment for one node, matching the tag keys
    GeneRax's own NHX writer emits (`printEventNHX` in
    `ext/GeneRaxCore/src/IO/ReconciliationWriter.cpp`): `S=` species label,
    `D=Y/N` duplication flag, `H=Y/N` transfer(horizontal) flag, and for
    transfer nodes a `@<donorSpecies>@<destSpecies>` suffix appended directly
    after `H=`. Branch lengths (`B=` in GeneRax) are omitted: `recon_nodes`
    carries no branch-length information.
    """
    event, species_index, _children = recon_nodes[node_index]
    species_label = _species_name(index_to_name, species_index)
    comment = f"[&&NHX:S={species_label}"
    comment += ":D=" + ("Y" if event == "duplication" else "N")
    comment += ":H=" + ("Y" if event == "transfer" else "N")
    if event == "transfer":
        dest_index = _transfer_destination_index(recon_nodes, node_index)
        dest_label = _species_name(index_to_name, dest_index)
        comment += f"@{species_label}@{dest_label}"
    comment += "]"
    return comment


def _build_notung_newick(recon_nodes, node_index, index_to_name):
    event, species_index, children = recon_nodes[node_index]
    left_index, right_index = children
    comment = _notung_nhx_comment(recon_nodes, node_index, index_to_name)
    if left_index == -1 and right_index == -1:
        label = _notung_node_label(event, species_index, node_index, index_to_name)
        return label + comment
    left_str = _build_notung_newick(recon_nodes, left_index, index_to_name)
    right_str = _build_notung_newick(recon_nodes, right_index, index_to_name)
    return f"({left_str},{right_str})n{node_index}" + comment


def write_notung(out_dir, name, gene_newick, recon_nodes, species_newick=None):
    """Write a Notung-style NHX-annotated Newick reconciliation, matching the
    tag conventions of GeneRax's own NHX writer (`saveReconciliationNHX` in
    `ext/GeneRaxCore/src/IO/ReconciliationWriter.cpp`). The tree topology and
    per-node events come entirely from `recon_nodes` (node 0 = root), mirroring
    `write_recphyloxml`; `gene_newick` is accepted for API symmetry but its
    topology is not re-parsed here.
    """
    if species_newick is not None:
        index_to_name, _species_root = _species_index_to_name(species_newick)
    else:
        index_to_name = {}

    nhx_newick = _build_notung_newick(recon_nodes, 0, index_to_name) + ";"
    p = pathlib.Path(out_dir) / f"{name}.notung.nhx"
    p.write_text(nhx_newick + "\n")
    return str(p)


def write_recphyloxml(out_dir, name, gene_newick, recon_nodes, species_newick):
    """Write a RecPhyloXML reconciliation, matching the tag/attribute names used
    by GeneRax's own writer (`ext/GeneRaxCore/src/IO/ReconciliationWriter.cpp`):
    a `<recPhylo>` root with a `<spTree>` (species tree, phyloXML `<clade>`
    nesting) and a `<recGeneTree>` (gene tree, one `<clade>` per reconciliation
    node carrying an `<eventsRec>` with the node's event element).
    """
    index_to_name, species_root = _species_index_to_name(species_newick)

    root = ET.Element(
        "recPhylo",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://www.recg.org ./recGeneTreeXML.xsd",
            "xmlns": "http://www.recg.org",
        },
    )

    sp_tree = ET.SubElement(root, "spTree")
    sp_phylogeny = ET.SubElement(sp_tree, "phylogeny")
    sp_phylogeny.append(_build_sptree_clade(species_root))

    gene_tree = ET.SubElement(root, "recGeneTree")
    gene_phylogeny = ET.SubElement(gene_tree, "phylogeny", rooted="true")
    gene_phylogeny.append(_build_genetree_clade(recon_nodes, 0, index_to_name))

    ET.indent(root)
    p = pathlib.Path(out_dir) / f"{name}.recphyloxml.xml"
    with p.open("w", encoding="utf-8") as fh:
        ET.ElementTree(root).write(fh, xml_declaration=True, encoding="unicode")
    return str(p)
