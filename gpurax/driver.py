"""Top-level orchestration driver: parse families -> Step 0 (starting trees)
-> reconciliation-only warm-up -> main loop (rates <-> joint SPR, radius
1..max) -> final reconciliation -> export.

Mirrors the GeneRax pipeline (spec S3): fixed-tree rate optimization
alternates with joint (sequence + reconciliation) SPR search at increasing
radius, starting from an optional reconciliation-only warm-up round.
"""

import pathlib
import tempfile

import dendropy

from gpurax._seqlik import SeqFamily
from gpurax.io.families import parse_families
from gpurax.io.initial_trees import ensure_starting_tree
from gpurax.io.prepare import prepare_family
from gpurax.rates.optimize import optimize_global_rates
from gpurax.recon.parity import recon_loglk
from gpurax.reconcile.export import (
    write_newick, write_notung, write_recphyloxml, write_scores,
)
from gpurax.reconcile.sample import reconcile_family
from gpurax.search.scorer import JointScorer
from gpurax.search.spr import FamilyState, spr_search

_DEFAULT_RATES = (0.2, 0.3, 0.1)


def _validate_newick(path, name):
    """Parse `path` with a pure-Python Newick parser before it ever reaches
    SeqFamily: a malformed Newick crashes the vendored coraxlib parser with a
    SIGSEGV (not a catchable exception), so any reader-supplied starting tree
    must be validated here first."""
    try:
        dendropy.Tree.get(path=path, schema="newick")
    except Exception as exc:
        raise ValueError(f"malformed gene tree for family {name}: {exc}") from exc


def _build_family_state(family, workdir):
    tree_path = ensure_starting_tree(family, workdir)
    _validate_newick(tree_path, family.name)
    tree_path, alignment_path = prepare_family(family, tree_path, workdir)
    newick = pathlib.Path(tree_path).read_text().strip()
    seqfam = SeqFamily(newick, alignment_path, family.model)
    return FamilyState(family.name, seqfam, _DEFAULT_RATES)


def _write_tree(tmpdir, tag, seqfam):
    path = str(pathlib.Path(tmpdir) / f"{tag}.nwk")
    pathlib.Path(path).write_text(seqfam.newick() + "\n")
    return path


def run(families_path, species_path, out_dir, *, rec_model="UndatedDTL",
        max_spr_radius=5, rec_radius=1, rec_weight=1.0, device="cuda"):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    species_newick = pathlib.Path(species_path).read_text()

    families = parse_families(families_path)
    scorer = JointScorer(species_path, rec_weight=rec_weight, device=device)
    fitted_rates = _DEFAULT_RATES

    with tempfile.TemporaryDirectory() as tmpdir:
        states = [_build_family_state(f, tmpdir) for f in families]

        if rec_radius > 0:
            spr_search(states, scorer, max_radius=rec_radius, tmpdir=tmpdir, warmup=True)

        for r in range(1, max_spr_radius + 1):
            tree_paths = [_write_tree(tmpdir, f"{fs.name}.r{r}", fs.seqfam) for fs in states]
            fitted_rates = optimize_global_rates(species_path, tree_paths, device=device)
            states = [FamilyState(fs.name, fs.seqfam, fitted_rates) for fs in states]
            spr_search(states, scorer, max_radius=r, tmpdir=tmpdir, warmup=False)

        rows = []
        for fs in states:
            final_tree = _write_tree(tmpdir, f"{fs.name}.final", fs.seqfam)
            recon_nodes = reconcile_family(species_path, final_tree, fitted_rates, device=device)
            gene_newick = fs.seqfam.newick()

            write_newick(out_dir, fs.name, gene_newick)
            write_recphyloxml(out_dir, fs.name, gene_newick, recon_nodes, species_newick)
            write_notung(out_dir, fs.name, gene_newick, recon_nodes, species_newick)

            seq = fs.seqfam.seq_loglk(False)
            rec = recon_loglk(species_path, final_tree, *fitted_rates, device=device)
            rows.append({"name": fs.name, "joint": seq + rec_weight * rec, "rec": rec, "seq": seq})

        write_scores(out_dir, rows)
