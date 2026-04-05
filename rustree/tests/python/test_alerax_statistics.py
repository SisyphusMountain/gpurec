#!/usr/bin/env python3
"""
Test script for ALERax integration with summary statistics
"""

import rustree
import sys

def main():
    print("=" * 80)
    print("Testing ALERax Integration with Summary Statistics")
    print("=" * 80)

    # Load species tree
    print("\n1. Loading species tree...")
    species_tree = rustree.parse_species_tree("test_data_3/output_zombi/T/ExtantTree.nwk")
    print(f"   Species tree loaded: {species_tree.num_nodes()} nodes")

    # Reconcile gene trees
    print("\n2. Reconciling gene trees with ALERax...")
    gene_trees = {
        "family_1": "test_data_3/output_zombi/G/Gene_trees/1_prunedtree.nwk",
        "family_2": "test_data_3/output_zombi/G/Gene_trees/2_prunedtree.nwk"
    }

    results = rustree.reconcile_with_alerax(
        species_tree=species_tree,
        gene_trees=gene_trees,
        seed=42,
        num_samples=100
    )

    print(f"   Successfully reconciled {len(results)} families")

    # Test each family result
    for family_name, result in results.items():
        print(f"\n{'=' * 80}")
        print(f"Results for {family_name}")
        print(f"{'=' * 80}")

        # Basic information
        print(f"\nBasic Information:")
        print(f"  Reconciliation samples: {len(result.gene_trees)}")
        print(f"  Duplication rate (D):   {result.duplication_rate:.6f}")
        print(f"  Loss rate (L):          {result.loss_rate:.6f}")
        print(f"  Transfer rate (T):      {result.transfer_rate:.6f}")
        print(f"  Log-likelihood:         {result.likelihood:.2f}")

        # Access summary statistics
        stats = result.statistics
        print(f"\nSummary Statistics:")
        print(f"  Statistics object: {stats}")

        # Mean event counts
        print(f"\nMean Event Counts (across {len(result.gene_trees)} samples):")
        mean_events = stats.mean_event_counts
        print(f"  Speciations:         {mean_events.speciations:.2f}")
        print(f"  Speciation Losses:   {mean_events.speciation_losses:.2f}")
        print(f"  Duplications:        {mean_events.duplications:.2f}")
        print(f"  Duplication Losses:  {mean_events.duplication_losses:.2f}")
        print(f"  Transfers:           {mean_events.transfers:.2f}")
        print(f"  Transfer Losses:     {mean_events.transfer_losses:.2f}")
        print(f"  Losses:              {mean_events.losses:.2f}")
        print(f"  Leaves:              {mean_events.leaves:.2f}")

        # Mean transfers between species
        print(f"\nMean Transfers Between Species:")
        transfers = stats.mean_transfers
        if transfers:
            for source in sorted(transfers.keys()):
                for dest in sorted(transfers[source].keys()):
                    count = transfers[source][dest]
                    print(f"  {source} → {dest}: {count:.2f}")
        else:
            print("  No transfers detected")

        # Events per species
        print(f"\nMean Events Per Species:")
        events_per_species = stats.events_per_species
        for species in sorted(events_per_species.keys()):
            species_events = events_per_species[species]
            print(f"\n  Species: {species}")
            print(f"    Speciations:  {species_events.speciations:.2f}")
            print(f"    Duplications: {species_events.duplications:.2f}")
            print(f"    Transfers:    {species_events.transfers:.2f}")
            print(f"    Losses:       {species_events.losses:.2f}")
            print(f"    Leaves:       {species_events.leaves:.2f}")

        # Verify first reconciliation sample works
        print(f"\nVerifying first reconciliation sample:")
        best_tree = result.gene_trees[0]
        events = best_tree.count_events()
        print(f"  Events in first sample: S={events[0]}, D={events[1]}, T={events[2]}, L={events[3]}")
        print(f"  Number of nodes:        {best_tree.num_nodes()}")
        print(f"  Number of extant genes: {best_tree.num_extant()}")

        # Export to CSV
        print(f"\nExporting to CSV...")
        df = best_tree.to_csv()
        print(f"  CSV DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

    print(f"\n{'=' * 80}")
    print("All tests passed successfully!")
    print(f"{'=' * 80}\n")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
