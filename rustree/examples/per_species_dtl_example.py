#!/usr/bin/env python3
"""
Example demonstrating the per-species DTL simulation model in rustree.

This script shows how to:
1. Simulate a gene tree using the per-species DTL model
2. Compare per-species vs per-gene-copy DTL models
3. Export reconciled trees to XML and Newick formats

Key concept:
  In the standard (per-gene-copy) DTL model, the total event rate scales with
  the number of gene copies: more duplications lead to more copies, which in
  turn produce even more events (a positive feedback loop).

  In the per-species DTL model (Zombi-style), the event rate scales with the
  number of SPECIES that carry at least one gene copy. Duplications within a
  species do not increase the overall event rate. This tends to produce gene
  trees with fewer total events and less runaway growth.
"""

import sys
sys.path.insert(0, "../target/release")

import rustree
import tempfile
import os


def example_single_gene_tree():
    """Simulate a single gene tree with the per-species DTL model."""
    print("=" * 60)
    print("Example 1: Single Per-Species DTL Simulation")
    print("=" * 60)

    # Step 1: Simulate a species tree with ~20 extant species
    species_tree = rustree.simulate_species_tree(
        n=20, lambda_=1.0, mu=0.3, seed=42
    )
    print(f"Species tree: {species_tree.num_leaves()} leaves, "
          f"{species_tree.num_nodes()} total nodes, "
          f"height = {species_tree.tree_height():.4f}")
    print()

    # Step 2: Simulate a gene tree using the per-species model
    # Rates: duplication=0.5, transfer=0.2, loss=0.3 per species per unit time
    gene_tree = species_tree.simulate_dtl_per_species(
        lambda_d=0.5, lambda_t=0.2, lambda_l=0.3, seed=123
    )

    # Step 3: Print basic statistics
    events = gene_tree.count_events()
    print(f"Gene tree nodes:    {gene_tree.num_nodes()}")
    print(f"Extant gene copies: {gene_tree.num_extant()}")
    print(f"Speciations:        {events['speciations']}")
    print(f"Duplications:       {events['duplications']}")
    print(f"Transfers:          {events['transfers']}")
    print(f"Losses:             {events['losses']}")
    print(f"Leaves:             {events['leaves']}")
    print()

    # Step 4: Show extant gene names
    extant_names = gene_tree.extant_gene_names()
    print(f"Extant genes ({len(extant_names)}):")
    for name in extant_names[:10]:
        print(f"  {name}")
    if len(extant_names) > 10:
        print(f"  ... and {len(extant_names) - 10} more")
    print()

    # Step 5: Export to Newick
    newick = gene_tree.to_newick()
    print(f"Newick (first 120 chars): {newick[:120]}...")
    print()

    return species_tree


def example_compare_models(species_tree):
    """Compare per-species and per-gene-copy models using batches."""
    print("=" * 60)
    print("Example 2: Per-Species vs Per-Gene-Copy Model Comparison")
    print("=" * 60)

    lambda_d, lambda_t, lambda_l = 0.5, 0.2, 0.3
    n_trees = 10

    print(f"DTL rates: d={lambda_d}, t={lambda_t}, l={lambda_l}")
    print(f"Simulating {n_trees} gene trees with each model...")
    print()

    # Simulate batches with each model
    per_gene_trees = species_tree.simulate_dtl_batch(
        n=n_trees,
        lambda_d=lambda_d, lambda_t=lambda_t, lambda_l=lambda_l,
        seed=456
    )
    per_species_trees = species_tree.simulate_dtl_per_species_batch(
        n=n_trees,
        lambda_d=lambda_d, lambda_t=lambda_t, lambda_l=lambda_l,
        seed=456
    )

    # Collect statistics
    def summarize(trees, label):
        nodes_list = []
        extant_list = []
        dup_list = []
        trans_list = []
        loss_list = []
        for gt in trees:
            nodes_list.append(gt.num_nodes())
            extant_list.append(gt.num_extant())
            ev = gt.count_events()
            dup_list.append(ev["duplications"])
            trans_list.append(ev["transfers"])
            loss_list.append(ev["losses"])

        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        print(f"  {label}:")
        print(f"    Avg nodes:          {avg(nodes_list):.1f}")
        print(f"    Avg extant genes:   {avg(extant_list):.1f}")
        print(f"    Avg duplications:   {avg(dup_list):.1f}")
        print(f"    Avg transfers:      {avg(trans_list):.1f}")
        print(f"    Avg losses:         {avg(loss_list):.1f}")
        return avg(nodes_list), avg(extant_list)

    pg_nodes, pg_extant = summarize(per_gene_trees, "Per-gene-copy model")
    print()
    ps_nodes, ps_extant = summarize(per_species_trees, "Per-species model")
    print()

    # Explain the difference
    print("Interpretation:")
    print("  The per-gene-copy model rate scales with the number of gene copies,")
    print("  so duplications feed back into higher event rates. The per-species")
    print("  model rate depends only on how many species carry genes, so")
    print("  duplications within a single species do not increase the rate.")
    if ps_nodes < pg_nodes:
        print("  As expected, the per-species model produced smaller gene trees")
        print(f"  on average ({ps_nodes:.0f} vs {pg_nodes:.0f} nodes).")
    else:
        print("  In this particular run the per-species trees were not smaller,")
        print("  which can happen with small samples or low rates.")
    print()


def example_export_formats(species_tree):
    """Demonstrate exporting per-species gene trees to various formats."""
    print("=" * 60)
    print("Example 3: Exporting Per-Species Gene Trees")
    print("=" * 60)

    # Simulate a gene tree (require at least one extant gene for a valid export)
    gene_tree = species_tree.simulate_dtl_per_species(
        lambda_d=0.4, lambda_t=0.1, lambda_l=0.2,
        require_extant=True,
        seed=789
    )
    print(f"Gene tree: {gene_tree.num_nodes()} nodes, "
          f"{gene_tree.num_extant()} extant")
    print()

    # Export to Newick
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.nwk', delete=False
    ) as f:
        newick_path = f.name
    gene_tree.save_newick(newick_path)
    newick_size = os.path.getsize(newick_path)
    print(f"Newick saved to: {newick_path} ({newick_size} bytes)")

    # Export to RecPhyloXML (contains species tree, gene tree, and mapping)
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.recphyloxml', delete=False
    ) as f:
        xml_path = f.name
    gene_tree.save_xml(xml_path)
    xml_size = os.path.getsize(xml_path)
    print(f"RecPhyloXML saved to: {xml_path} ({xml_size} bytes)")
    print()

    # Show a snippet of the XML output
    print("RecPhyloXML excerpt (first 15 lines):")
    with open(xml_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 15:
                print("  ...")
                break
            print(f"  {line.rstrip()}")
    print()

    # Clean up temp files
    os.remove(newick_path)
    os.remove(xml_path)


def example_require_extant():
    """Show the require_extant flag with the per-species model."""
    print("=" * 60)
    print("Example 4: Using require_extant with Per-Species Model")
    print("=" * 60)

    species_tree = rustree.simulate_species_tree(
        n=10, lambda_=1.0, mu=0.3, seed=99
    )

    # With high loss rate, genes may go completely extinct.
    # Setting require_extant=True ensures the returned tree has survivors.
    gene_tree = species_tree.simulate_dtl_per_species(
        lambda_d=0.1, lambda_t=0.1, lambda_l=0.8,
        require_extant=True,
        seed=321
    )

    ev = gene_tree.count_events()
    print(f"High loss rate (d=0.1, t=0.1, l=0.8) with require_extant=True:")
    print(f"  Nodes:        {gene_tree.num_nodes()}")
    print(f"  Extant genes: {gene_tree.num_extant()}")
    print(f"  Events:       S={ev['speciations']} D={ev['duplications']} T={ev['transfers']} L={ev['losses']}")
    print()

    # Also works with batch mode
    trees = species_tree.simulate_dtl_per_species_batch(
        n=5,
        lambda_d=0.1, lambda_t=0.1, lambda_l=0.8,
        require_extant=True,
        seed=654
    )
    print(f"Batch of 5 trees (require_extant=True):")
    for i, gt in enumerate(trees):
        ev = gt.count_events()
        print(f"  Tree {i}: {gt.num_extant()} extant genes, "
              f"events S={ev['speciations']} D={ev['duplications']} T={ev['transfers']} L={ev['losses']}")
    print()


if __name__ == "__main__":
    species_tree = example_single_gene_tree()
    example_compare_models(species_tree)
    example_export_formats(species_tree)
    example_require_extant()

    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
