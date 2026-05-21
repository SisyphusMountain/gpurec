# test_trees_3 Fixture Contract

This is the smallest checked CUDA stochastic-backtracking fixture.  It is not a
general parser fixture and should stay small enough for a single-family smoke.

Contents:

- `sp.nwk`: one rooted binary species tree with 15 postorder species nodes.
- `g.nwk`: one rooted binary gene tree for the same species labels through the
  legacy `Species_gene` prefix mapping.
- `output/reconciliations/totalSpeciesEventCounts.txt`: AleRax-style aggregate
  event counts used as historical context for this tiny scenario.
- `output/reconciliations/totalTransfers.txt`: AleRax-style aggregate transfer
  table.  It is intentionally empty for this fixture.

Pinned expectations:

- `export_backtracking_input()` should produce a `pi` payload with 35 clade
  rows and 15 species columns for this family.
- The exported `root_clade` must be a valid clade row.
- The Rust sampler should emit one `recGeneTree`.
- The sampled reconciliation should contain 10 leaf events and at least one
  non-leaf reconciliation event among speciation, loss, duplication, or
  transfer categories.

Use this fixture only for CUDA integration smoke.  CPU-only Rust CLI checks
should use `tests/fixtures/backtracking/speciation.json` instead.
