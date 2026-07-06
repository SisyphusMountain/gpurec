import pathlib

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
