"""
Summary writers for sampled reconciliations.

Provides:
- write_transfer_frequencies: aggregates transfer counts across scenarios
- write_events_with_labels: dumps events with species and gene leaf labels
"""
from typing import List, Dict, Tuple
import csv


EVENT_NAMES = {
    0: "S",
    1: "SL",
    2: "D",
    3: "DL",
    4: "T",
    5: "TL",
    6: "L",
    7: "Leaf",
    8: "Invalid",
}


def _sp_name(species_helpers: Dict, idx: int) -> str:
    if idx < 0:
        return ""
    # Prefer list under 'names'; else mapping 'sp_names_by_idx'
    if 'names' in species_helpers:
        names = species_helpers['names']
        try:
            return names[idx]
        except Exception:
            return str(idx)
    sp_map = species_helpers.get('sp_names_by_idx', {})
    return sp_map.get(idx, str(idx))


def write_transfer_frequencies(
    scenarios: List[Dict],
    species_helpers: Dict,
    out_path: str,
    normalize: bool = True,
) -> None:
    counts: Dict[Tuple[str, str], int] = {}
    for sc in scenarios:
        for ev in sc["events"]:
            et = int(ev["type"])  # 4=T, 5=TL
            if et in (4, 5):
                src = int(ev["species_node"])
                dst = int(ev["dest_species_node"]) if ev["dest_species_node"] is not None else -1
                if dst < 0:
                    continue  # TL dest lost or invalid
                key = (_sp_name(species_helpers, src), _sp_name(species_helpers, dst))
                counts[key] = counts.get(key, 0) + 1
    total = max(1, len(scenarios))
    # Sort by descending count
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from", "to", "count", "freq"])
        for (src, dst), c in items:
            freq = c / total if normalize else c
            w.writerow([src, dst, c, freq])


def write_events_with_labels(
    scenarios: List[Dict],
    species_helpers: Dict,
    ccp_helpers: Dict,
    out_path: str,
) -> None:
    """Dump events with species labels and optional gene leaf labels.

    If ccp_helpers contains 'clade_leaf_labels' (list/array of length C), we use it
    to label leaf events by clade id. Otherwise, gene_label is left empty.
    """
    clade_leaf_labels = None
    if isinstance(ccp_helpers, dict) and 'clade_leaf_labels' in ccp_helpers:
        clade_leaf_labels = ccp_helpers['clade_leaf_labels']
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "idx", "type", "species", "dest_species", "gene_node", "cid", "left_cid", "right_cid", "gene_label"])
        for s_idx, sc in enumerate(scenarios):
            for i, ev in enumerate(sc["events"]):
                et = int(ev["type"])  # numeric
                species = int(ev["species_node"]) if ev["species_node"] is not None else -1
                dest = int(ev["dest_species_node"]) if ev["dest_species_node"] is not None else -1
                species_label = _sp_name(species_helpers, species)
                dest_label = _sp_name(species_helpers, dest)
                cid = int(ev.get("cid", -1))
                left_cid = int(ev.get("left_cid", -1))
                right_cid = int(ev.get("right_cid", -1))
                gene_label = ""
                if et == 7 and cid >= 0 and clade_leaf_labels is not None:
                    try:
                        gene_label = clade_leaf_labels[cid]
                        # Normalize non-leaf entries to empty
                        if gene_label is None:
                            gene_label = ""
                        if not isinstance(gene_label, str):
                            gene_label = str(gene_label)
                    except Exception:
                        gene_label = ""
                w.writerow([
                    s_idx,
                    i,
                    EVENT_NAMES.get(et, str(et)),
                    species_label,
                    dest_label,
                    int(ev["gene_node"]),
                    cid,
                    left_cid,
                    right_cid,
                    gene_label,
                ])


def write_donated_transfers_per_branch(
    scenarios: List[Dict],
    species_helpers: Dict,
    out_path: str,
    normalize: bool = True,
) -> None:
    """Aggregate donated transfers per species branch (parent->child).

    - Counts both T and TL events as donations from the source lineage (species_node).
    - Maps each source node to its incoming branch (parent->child). For root (no parent), parent is empty.
    - Normalizes by number of sampled scenarios if normalize=True.
    """
    names = species_helpers.get('names', [])
    S = int(species_helpers.get('S', len(names)))
    # Reconstruct parent mapping using compact helpers
    sP = species_helpers['s_P_indexes']
    sC12 = species_helpers['s_C12_indexes']
    # tensors -> lists
    sP = sP[:sP.shape[0]//2] # Only take first part
    sP = sP.detach().cpu().tolist()
    sC12 = sC12.detach().cpu().tolist()
    I = len(sP)
    parent = [-1] * S
    for i in range(I):
        p = int(sP[i])
        c1 = int(sC12[i])
        c2 = int(sC12[i + I])
        parent[c1] = p
        parent[c2] = p
    # Aggregate counts per child branch
    counts: Dict[int, int] = {}
    for sc in scenarios:
        for ev in sc['events']:
            et = int(ev['type'])
            if et in (4, 5):  # T, TL
                child = int(ev['species_node'])
                counts[child] = counts.get(child, 0) + 1
    total = max(1, len(scenarios))
    # Prepare rows: only branches with nonzero counts
    rows = []
    for child, c in counts.items():
        par = parent[child]
        par_name = names[par] if (par is not None and par >= 0 and par < len(names)) else ''
        child_name = names[child] if (child >= 0 and child < len(names)) else str(child)
        freq = (c / total) if normalize else c
        rows.append((par_name, child_name, c, freq))
    # Sort by decreasing freq
    rows.sort(key=lambda r: r[3], reverse=True)
    # Write file: if .txt, mimic ALE totalTransfers format: "from to freq" with no header
    if out_path.endswith('.txt'):
        with open(out_path, 'w') as f:
            for par_name, child_name, c, freq in rows:
                f.write(f"{par_name} {child_name} {freq:.4f}\n")
    else:
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['parent', 'child', 'donated_count', 'donated_freq'])
            for r in rows:
                w.writerow(r)


def write_total_transfers_txt(
    scenarios: List[Dict],
    species_helpers: Dict,
    out_path: str,
    normalize: bool = True,
) -> None:
    """Write totalTransfers-style file: "from to freq" per (src,dst) species pair.

    - Counts both T and TL events with a valid destination species.
    - If normalize is True, divides counts by number of sampled scenarios.
    - Sorted by decreasing frequency.
    """
    counts: Dict[Tuple[str, str], int] = {}
    for sc in scenarios:
        for ev in sc.get('events', []):
            et = int(ev.get('type', -1))
            if et in (4, 5):  # T, TL
                src = int(ev.get('species_node', -1))
                dst = int(ev.get('dest_species_node', -1))
                if dst < 0:
                    continue
                key = (_sp_name(species_helpers, src), _sp_name(species_helpers, dst))
                counts[key] = counts.get(key, 0) + 1
    total = max(1, len(scenarios))
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    with open(out_path, 'w') as f:
        for (src, dst), c in items:
            freq = (c / total) if normalize else float(c)
            f.write(f"{src} {dst} {freq:.4f}\n")


def merge_total_transfers_txt(
    files: List[str],
    out_path: str,
    normalize: bool = False,
) -> None:
    """Merge per-family meanTransfers files into a totalTransfers-style file.

    Each input file must have lines: "from to value" with no header.
    If normalize=False (ALE behavior), simply sums values across files.
    If normalize=True, divides the sum by len(files).
    """
    counts: Dict[Tuple[str, str], float] = {}
    for path in files:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) != 3:
                    continue
                src, dst, val = parts[0], parts[1], parts[2]
                try:
                    x = float(val)
                except ValueError:
                    continue
                key = (src, dst)
                counts[key] = counts.get(key, 0.0) + x
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    denom = float(len(files)) if (normalize and len(files) > 0) else 1.0
    with open(out_path, 'w') as f:
        for (src, dst), v in items:
            f.write(f"{src} {dst} {v/denom:.4f}\n")
