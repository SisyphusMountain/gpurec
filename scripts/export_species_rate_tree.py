"""Export the species tree + per-node DTL rates for plotting in R.

The model indexes species-tree nodes by the post-order position used by the Rust
preprocessor (``rustree``'s ``postorder_indices``).  We re-derive that ordering by
parsing the Newick with children in file order, then (optionally) assert the
reconstructed topology matches the model's ``sp_parent/sp_child1/sp_child2`` arrays.
If it matches, the ``gp_idx -> species name`` mapping is correct by construction.

Outputs two files the R script consumes:
  * ``<out_dir>/species_rate_tree.nwk`` — every node labelled by its model index
    (``gp_idx``); leaves carry the model index too, names live in the CSV.
  * ``<out_dir>/species_rate_rates.csv`` — one row per node:
    ``label,name,is_leaf,parent,rate_D,rate_L,rate_T,log2_D,log2_L,log2_T``
    (``label`` == ``gp_idx`` so R can join on it; rate = ``2**theta``).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


class _Node:
    __slots__ = ("name", "children")

    def __init__(self):
        self.name = ""
        self.children = []


def _parse_newick(text: str) -> _Node:
    text = text.strip()
    pos = 0

    def parse() -> _Node:
        nonlocal pos
        nd = _Node()
        if text[pos] == "(":
            pos += 1
            while True:
                nd.children.append(parse())
                if text[pos] == ",":
                    pos += 1
                    continue
                if text[pos] == ")":
                    pos += 1
                    break
        label = re.match(r"[^():,;]*", text[pos:]).group(0)
        pos += len(label)
        if ":" in label:
            label = label.split(":")[0]
        if pos < len(text) and text[pos] == ":":
            bl = re.match(r":[^():,;]*", text[pos:]).group(0)
            pos += len(bl)
        nd.name = label
        return nd

    return parse()


def export_rate_tree(species_tree_path, theta, out_dir, species_helpers=None):
    """Write the gp-indexed Newick + per-node rate table; return (nwk, csv) paths.

    theta: [S, 3] array/tensor of log2-rates, columns [D, L, T].
    species_helpers: optional dict with sp_parent/sp_child1/sp_child2 for verification.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theta = np.asarray(theta.detach().cpu() if hasattr(theta, "detach") else theta, dtype=float)
    assert theta.ndim == 2 and theta.shape[1] == 3, f"theta must be [S,3], got {theta.shape}"
    S = theta.shape[0]

    root = _parse_newick(Path(species_tree_path).read_text())

    postorder: list[_Node] = []
    parent_of: dict[int, _Node] = {}

    def walk(nd: _Node):
        for c in nd.children:
            parent_of[id(c)] = nd
            walk(c)
        postorder.append(nd)

    walk(root)
    if len(postorder) != S:
        raise ValueError(f"parsed {len(postorder)} nodes but theta has S={S}")

    gp = {id(nd): i for i, nd in enumerate(postorder)}
    sp_parent = np.full(S, -1, int)
    sp_c1 = np.full(S, S, int)
    sp_c2 = np.full(S, S, int)
    name = [""] * S
    for nd in postorder:
        i = gp[id(nd)]
        name[i] = nd.name
        if id(nd) in parent_of:
            sp_parent[i] = gp[id(parent_of[id(nd)])]
        if nd.children:
            if len(nd.children) != 2:
                raise ValueError("species tree must be binary")
            sp_c1[i] = gp[id(nd.children[0])]
            sp_c2[i] = gp[id(nd.children[1])]

    if species_helpers is not None:
        def arr(k):
            v = species_helpers[k]
            return v.detach().cpu().numpy().astype(int) if hasattr(v, "detach") else np.asarray(v, int)
        for key, rec in (("sp_parent", sp_parent), ("sp_child1", sp_c1), ("sp_child2", sp_c2)):
            mism = int((arr(key) != rec).sum())
            if mism:
                raise AssertionError(f"{key} mismatch vs model ({mism}) — gp_idx mapping unverified")

    # gp-indexed Newick (label every node by its model index; branch length 1)
    def emit(nd: _Node) -> str:
        i = gp[id(nd)]
        body = ("(" + ",".join(emit(c) for c in nd.children) + ")" + str(i)) if nd.children else str(i)
        return body + (":1" if id(nd) in parent_of else "")

    nwk_path = out_dir / "species_rate_tree.nwk"
    nwk_path.write_text(emit(root) + ";\n")

    rates = np.exp2(theta)
    csv_path = out_dir / "species_rate_rates.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "name", "is_leaf", "parent",
                    "rate_D", "rate_L", "rate_T", "log2_D", "log2_L", "log2_T"])
        for i in range(S):
            is_leaf = int(sp_c1[i] == S)
            disp = name[i] if (is_leaf and name[i]) else str(i)
            w.writerow([i, disp, is_leaf, int(sp_parent[i]),
                        rates[i, 0], rates[i, 1], rates[i, 2],
                        theta[i, 0], theta[i, 1], theta[i, 2]])

    return nwk_path, csv_path


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("species_tree")
    ap.add_argument("theta_json", help="JSON with 'final_theta' [S,3] (e.g. a notebook fit file)")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    payload = json.load(open(args.theta_json))
    theta = np.asarray(payload["final_theta"], float)
    nwk, csv_p = export_rate_tree(args.species_tree, theta, args.out_dir)
    print("wrote", nwk)
    print("wrote", csv_p)
