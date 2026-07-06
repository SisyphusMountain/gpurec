"""GeneRax "Step 0": build a starting gene tree when a family doesn't supply
one, via LibpllEvaluation::createAndSaveRandomTree (exposed as
`_seqlik.build_starting_tree`)."""

import pathlib

from gpurax import _seqlik


def ensure_starting_tree(family, workdir):
    """Return a path to a starting Newick tree for `family`.

    If `family.starting_tree` is set, returns it unchanged. Otherwise builds
    a random starting tree from `family.alignment` / `family.model` and
    writes it to `workdir/{family.name}.start.nwk`, returning that path. The
    joint SPR search improves this random start later; building a
    parsimony/ML start is out of scope here.
    """
    if family.starting_tree:
        return family.starting_tree
    out = str(pathlib.Path(workdir) / f"{family.name}.start.nwk")
    return _seqlik.build_starting_tree(family.alignment, family.model, out)
