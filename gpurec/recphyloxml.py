"""Shared RecPhyloXML traversal helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator


EVENT_KEYS = ("S", "SL", "D", "DL", "T", "TL", "L", "Leaf")

_PRIMARY_EVENT_NAMES = frozenset(
    {"speciation", "duplication", "branchingOut", "loss", "leaf"}
)


def local_name(tag: str) -> str:
    """Return an XML tag without its namespace prefix."""
    return tag.rsplit("}", 1)[-1]


def direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if local_name(child.tag) == name]


def events_for_clade(clade: ET.Element) -> list[ET.Element]:
    for child in clade:
        if local_name(child.tag) == "eventsRec":
            return list(child)
    return []


def primary_event_name(clade: ET.Element) -> str | None:
    for event in events_for_clade(clade):
        name = local_name(event.tag)
        if name in _PRIMARY_EVENT_NAMES:
            return name
    return None


def iter_rec_gene_tree_clades(xml: str) -> Iterator[ET.Element]:
    root = ET.fromstring(xml)
    for rec_gene_tree in root.iter():
        if local_name(rec_gene_tree.tag) != "recGeneTree":
            continue
        first_clade = next(
            (
                element
                for element in rec_gene_tree.iter()
                if local_name(element.tag) == "clade"
            ),
            None,
        )
        if first_clade is not None:
            yield first_clade
