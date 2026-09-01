# gpurec local archive

`archive/storage/` contains historical worktrees, experiment output, rescued
changes, repository bundles, and superseded snapshots. It is intentionally
ignored by Git and is not part of the installable package or a fresh clone.

The store exists for recovery and provenance. Maintained source code,
documentation, papers, benchmark recipes, and compact result summaries must
live outside `archive/storage/` in their canonical tracked locations.

Nothing in the archive should be treated as canonical unless a tracked index
explicitly says so. Deleting archive material is a separate cleanup operation
after its replacement or redundancy has been verified.

Current recovery landmarks include:

- `storage/repository-bundles/`: complete Git bundles for the retired
  `kernel-bench` history and the independent ghost-lineage project;
- `storage/retired-workspace/`: recoverable `kernel-bench`, `gergely_version`,
  and `comparison_report` checkouts/snapshots;
- `storage/benchmark-runs/`: large historical benchmark outputs;
- `storage/consolidate-release-untracked/`: former experiment results and
  third-party tool builds;
- `storage/recovery/`: older rescue directories and worktree material.

Small documents that remain useful as dated evidence live in tracked
`internal-audits/`, `internal-reviews/`, and `papers/` rather than in the large
storage area.
