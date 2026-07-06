from collections import namedtuple

Family = namedtuple("Family", "name starting_tree alignment mapping model")

_KEY_MAP = {
    "starting_gene_tree": "starting_tree",
    "alignment": "alignment",
    "mapping": "mapping",
    "subst_model": "model",
}


def parse_families(path):
    fams, cur = [], None
    for line in open(path):
        s = line.strip()
        if not s or s == "[FAMILIES]":
            continue
        if s.startswith("- "):
            if cur:
                fams.append(_finish(cur))
            cur = {"name": s[2:].strip()}
        elif "=" in s and cur is not None:
            k, v = (x.strip() for x in s.split("=", 1))
            cur[_KEY_MAP.get(k, k)] = v
    if cur:
        fams.append(_finish(cur))
    return fams


def _finish(d):
    return Family(d["name"], d.get("starting_tree"), d.get("alignment"),
                  d.get("mapping"), d.get("model", "GTR"))


def parse_mapping(path):
    """Parse a GeneRax gene->species mapping file (Treerecs per-line format:
    `gene species` per line, one pair per line)."""
    m = {}
    for line in open(path):
        s = line.split()
        if len(s) == 2:
            m[s[0]] = s[1]
    return m
