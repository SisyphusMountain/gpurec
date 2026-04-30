"""Sampling helper: feed gpurec-optimized DTL rates into AleRax for reconciliation sampling.

The public entry point is :meth:`GeneReconModel.sample_reconciliations`,
which delegates here. This module exists separately so the labelling
machinery (replicating ``PLLRootedTree::ensureUniqueLabels``) does not
clutter the model class.
"""
from __future__ import annotations

import os
import pathlib
import tempfile
from typing import Any

import torch


# ──────────────────────────────────────────────────────────────────────
# AleRax label replication
# ──────────────────────────────────────────────────────────────────────


def _is_numeric(s: str) -> bool:
    if not s:
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _alerax_label_map(species_tree_path: str) -> dict[str, str]:
    """Build a ``{newick_name → alerax_label}`` mapping for every species
    tree node.

    Replicates ``PLLRootedTree::ensureUniqueLabels`` (see
    ``extra/AleRax_modified/ext/GeneRaxCore/src/trees/PLLRootedTree.cpp:309-339``):

    - Leaves keep their Newick label as-is.
    - Internal nodes: keep the Newick label only if it is non-empty,
      not duplicated, and not purely numeric. Otherwise the label is
      replaced with ``Node_<leftLeaf>_<rightLeaf>_<unique_id>``, where
      ``leftLeaf``/``rightLeaf`` are propagated from the left and right
      children in post-order.

    Uses :mod:`rustree` for Newick parsing — both gpurec and AleRax
    parse the same file, so node names match exactly for nodes that
    survive the renaming pass.
    """
    import rustree

    sp = rustree.parse_species_tree(species_tree_path)
    n = sp.num_nodes()
    nodes = [sp.get_node(i) for i in range(n)]

    root = next(i for i in range(n) if nodes[i].parent is None)

    # Iterative post-order to avoid recursion-depth issues on big trees.
    post_order: list[int] = []
    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        idx, processed = stack.pop()
        if processed:
            post_order.append(idx)
            continue
        stack.append((idx, True))
        node = nodes[idx]
        if node.right_child is not None:
            stack.append((node.right_child, False))
        if node.left_child is not None:
            stack.append((node.left_child, False))

    any_leaf_label: list[str] = [""] * n
    seen: set[str] = set()
    name_to_label: dict[str, str] = {}

    def _get_unique(seen_set: set[str], base: str) -> str:
        i = 0
        while True:
            candidate = f"{base}_{i}"
            if candidate not in seen_set:
                return candidate
            i += 1

    for idx in post_order:
        node = nodes[idx]
        if node.left_child is None:
            # Leaf
            any_leaf_label[idx] = node.name
            name_to_label[node.name] = node.name
            seen.add(node.name)
            continue
        any_leaf_label[idx] = any_leaf_label[node.left_child]
        label = node.name or ""
        if (not label) or (label in seen) or _is_numeric(label):
            base = (
                f"Node_{any_leaf_label[node.left_child]}_"
                f"{any_leaf_label[node.right_child]}"
            )
            label = _get_unique(seen, base)
        # Map the *original* Newick name (which gpurec uses) to the
        # AleRax-renamed label. For leaves the original == renamed.
        original = node.name if node.name else label
        name_to_label[original] = label
        seen.add(label)

    return name_to_label


# ──────────────────────────────────────────────────────────────────────
# Rate-file writers
# ──────────────────────────────────────────────────────────────────────


def _write_specieswise_rates_dir(
    rates: torch.Tensor,
    species_names: list[str],
    species_tree_path: str,
    target_dir: pathlib.Path,
) -> None:
    """Write ``model_parameters.txt`` for AleRax PER-SPECIES mode.

    Parameters
    ----------
    rates : Tensor [S, 3]
        Natural-space ``[D, L, T]`` rates per species tree node, in the
        same order as ``species_names``.
    species_names : list[str]
        gpurec species names (from ``species_helpers['names']``).
    species_tree_path : str
        Path to the Newick file — used to compute AleRax-style labels.
    target_dir : Path
        Output directory; ``<target_dir>/model_parameters.txt`` is
        created.
    """
    if rates.ndim != 2 or rates.shape[1] != 3:
        raise ValueError(
            f"specieswise rates must have shape [S, 3], got {tuple(rates.shape)}"
        )
    if rates.shape[0] != len(species_names):
        raise ValueError(
            f"rates rows ({rates.shape[0]}) do not match species count "
            f"({len(species_names)})"
        )
    name_to_label = _alerax_label_map(species_tree_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "model_parameters.txt"
    rates_cpu = rates.detach().cpu()
    with open(out_path, "w") as f:
        f.write("# node D L T\n")
        for s, name in enumerate(species_names):
            label = name_to_label.get(name)
            if label is None:
                raise KeyError(
                    f"gpurec species name '{name}' has no matching AleRax "
                    f"label (the species tree at {species_tree_path} may "
                    f"have changed since this dataset was built)"
                )
            d, l, t = (float(rates_cpu[s, i]) for i in range(3))
            f.write(f"{label} {d:.10g} {l:.10g} {t:.10g}\n")


def _write_genewise_rates_dir(
    rates: torch.Tensor,
    gene_tree_paths: list[str],
    target_dir: pathlib.Path,
) -> None:
    """Write per-family ``<family>_rates.txt`` files for AleRax PER-FAMILY mode.

    Family names match the file stems of ``gene_tree_paths`` (which is
    how :func:`rustree.reconcile_with_alerax` names them when given
    file paths).
    """
    if rates.ndim != 2 or rates.shape[1] != 3:
        raise ValueError(
            f"genewise rates must have shape [G, 3], got {tuple(rates.shape)}"
        )
    if rates.shape[0] != len(gene_tree_paths):
        raise ValueError(
            f"rates rows ({rates.shape[0]}) do not match family count "
            f"({len(gene_tree_paths)})"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    rates_cpu = rates.detach().cpu()
    for g, gpath in enumerate(gene_tree_paths):
        family_name = pathlib.Path(gpath).stem
        out_path = target_dir / f"{family_name}_rates.txt"
        d, l, t = (float(rates_cpu[g, i]) for i in range(3))
        with open(out_path, "w") as f:
            f.write("# D L T\n")
            f.write(f"{d:.10g} {l:.10g} {t:.10g}\n")


# ──────────────────────────────────────────────────────────────────────
# Public entry point (called from GeneReconModel.sample_reconciliations)
# ──────────────────────────────────────────────────────────────────────


def sample_reconciliations(
    model: Any,
    *,
    num_samples: int = 100,
    output_dir: str | None = None,
    seed: int | None = None,
    keep_output: bool = False,
    alerax_path: str = "alerax",
) -> dict[str, Any]:
    """Run AleRax in pure-sampling mode using ``model``'s optimized rates.

    Supports the three single-axis modes:

    - ``global``      : passes ``--d/--l/--t --fix-rates``.
    - ``specieswise`` : writes ``model_parameters.txt`` with AleRax-style
      labels and passes ``--starting-rates-file --fix-rates``.
    - ``genewise``    : writes per-family ``<family>_rates.txt`` files
      and passes ``--starting-rates-file --fix-rates``.

    The combined ``genewise + specieswise`` mode and ``pairwise``
    transfers have no AleRax equivalent and raise
    :class:`NotImplementedError`.
    """
    import rustree

    ds = model._dataset
    if ds.pairwise:
        raise NotImplementedError(
            "sample_reconciliations does not support pairwise transfer mode."
        )
    if ds.genewise and ds.specieswise:
        raise NotImplementedError(
            "sample_reconciliations does not support combined genewise + "
            "specieswise mode (AleRax has no per-family-per-species rates "
            "storage)."
        )

    rates = model.rates  # natural space; layout matches theta

    # Resolve output directory: if user passed one, write rates files there
    # so they can be inspected after the run.
    tmp_holder: tempfile.TemporaryDirectory | None = None
    if output_dir is None:
        tmp_holder = tempfile.TemporaryDirectory(prefix="gpurec_alerax_")
        out_root = pathlib.Path(tmp_holder.name)
    else:
        out_root = pathlib.Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

    rates_dir = out_root / "starting_rates"
    alerax_run_dir = out_root / "alerax_run"

    kwargs: dict[str, Any] = dict(
        species_tree=str(ds.species_tree_path),
        gene_trees=[str(p) for p in ds.gene_tree_paths],
        num_samples=num_samples,
        output_dir=str(alerax_run_dir),
        seed=seed,
        keep_output=keep_output or (output_dir is not None),
        alerax_path=alerax_path,
        fix_rates=True,
    )

    mode = model._mode
    if mode == "global":
        d, l_, t = (float(x) for x in rates.detach().cpu().tolist())
        kwargs.update(model="GLOBAL", d=d, l=l_, t=t)
    elif mode == "specieswise":
        _write_specieswise_rates_dir(
            rates,
            list(ds.species_helpers["names"]),
            str(ds.species_tree_path),
            rates_dir,
        )
        kwargs.update(
            model="PER-SPECIES",
            starting_rates_file=str(rates_dir),
        )
    elif mode == "genewise":
        _write_genewise_rates_dir(
            rates,
            [str(p) for p in ds.gene_tree_paths],
            rates_dir,
        )
        kwargs.update(
            model="PER-FAMILY",
            starting_rates_file=str(rates_dir),
        )
    else:
        raise NotImplementedError(f"unsupported mode: {mode!r}")

    try:
        return rustree.reconcile_with_alerax(**kwargs)
    finally:
        if tmp_holder is not None and not keep_output:
            tmp_holder.cleanup()
