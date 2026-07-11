"""Deterministic rustree simulation of gpurec-compatible DTL datasets (regen-only benchmark).

Trees are never committed; callers regenerate from a fixed seed. rustree is imported lazily so
the rest of gpurec does not depend on it.

Recipe (established empirically against the ``GeneReconModel`` finite-likelihood gate, see
``tests/regression/test_simulate.py``):

- Simulate the full (extinction-including) species tree, then immediately restrict to its
  ``sample_extant()`` subtree. All DTL simulation happens *on that extant tree*, not on the raw
  birth-death tree: rustree's DTL gene-tree "extant" leaves are genes that reached a species-tree
  leaf without being lost, which is a different notion than "the species survived to the present".
  Simulating on the raw tree (with extinct-lineage tips) therefore produces gene tips labeled by
  species leaves that are not among today's extant species. Simulating directly on the pruned
  extant species tree keeps gene-tip species labels and species-tree leaves in the same
  namespace.
- After simulation, `forest.sample_extant()` prunes each gene tree to genes present at a leaf
  (dropping loss-terminated lineages) and `forest.species_tree` is written as the paired species
  tree, keeping labels consistent by construction.
- genewise has no batch API with per-family rates, so families are simulated one at a time with
  `simulate_dtl` and assembled into a `rustree.GeneForest(species_tree, gene_trees)` (rustree has
  no `GeneForest.from_gene_trees`, but the constructor itself takes exactly this shape).
- specieswise uses `simulate_dtl_batch_with_branch_rates`, which validates that
  `origination_probability` sums to 1.0 (it is a per-branch origination distribution, not an
  independent zero vector); a uniform distribution over `sp_extant.num_nodes()` branches is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Canonical parameters for the committed goldens. Seeds are arbitrary but fixed. 200 leaves / 200
# families is large enough to exercise the real recipes (multi-batch, warm-adjoint) while keeping a
# repeats>1 mint tractable (~minutes/fit rather than the ~hour at 500 leaves).
SIM_PARAMS = {
    "global":      {"seed": 20260709, "n_species": 200, "n_families": 200, "dtl": 0.05},
    "genewise":    {"seed": 20260710, "n_species": 200, "n_families": 200, "dtl": 0.05},
    "specieswise": {"seed": 20260711, "n_species": 200, "n_families": 200, "dtl": 0.05},
}

# Specieswise has no well-posed one-shot MLE, so its perf-golden fits fit_specieswise at a FIXED,
# committed prior precision (a deterministic recipe test at a pinned prior; a full map_cv is too
# expensive/noisy for a golden). See docs/.../2026-07-11-specieswise-recipe-organization-design.md.
SPECIESWISE_GOLDEN_LAM = 10.0


def _write(sp_out, pruned_forest, out_dir: Path) -> tuple[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sp_path = out_dir / "species.nwk"
    sp_out.save_newick(str(sp_path))
    gene_paths = []
    for i, g in enumerate(pruned_forest):
        p = out_dir / f"fam_{i:06d}.nwk"
        nwk = g.to_newick()
        p.write_text(nwk if nwk.rstrip().endswith(";") else nwk + ";")
        gene_paths.append(str(p))
    return str(sp_path), gene_paths


def simulate_dataset(mode, out_dir, *, n_species, n_families, dtl, seed):
    import rustree

    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    sp_full = rustree.simulate_species_tree(n_species, 1.0, 0.5, seed=seed)
    sp_extant = sp_full.sample_extant()

    if mode == "global":
        forest = sp_extant.simulate_dtl_batch(n_families, dtl, dtl, dtl, seed=seed)
    elif mode == "genewise":
        # per-family (D,T,L) ~ lognormal around `dtl` (positive, ~0.5 dex spread); fixed by rng.
        rates = dtl * np.exp(rng.normal(0.0, 0.5, size=(n_families, 3)))
        trees = [sp_extant.simulate_dtl(float(d), float(t), float(l), seed=int(seed + 1 + i))
                 for i, (d, t, l) in enumerate(rates)]
        forest = rustree.GeneForest(sp_extant, trees)
    elif mode == "specieswise":
        n_branch = sp_extant.num_nodes()
        d = (dtl * np.exp(rng.normal(0.0, 0.5, n_branch))).tolist()
        t = (dtl * np.exp(rng.normal(0.0, 0.5, n_branch))).tolist()
        l = (dtl * np.exp(rng.normal(0.0, 0.5, n_branch))).tolist()
        orig = (np.ones(n_branch) / n_branch).tolist()
        forest = sp_extant.simulate_dtl_batch_with_branch_rates(n_families, d, t, l, orig, seed=seed)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    pruned = forest.sample_extant()
    sp_out = pruned.species_tree
    return _write(sp_out, pruned, out_dir)
