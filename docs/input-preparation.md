# Input Preparation

This guide is for preparing source data before a production `gpurec` run.  It
covers the files the workflow expects, how relative paths are interpreted, and
the validation commands to run before spending GPU time.

## Required Files

A production run starts from:

- One rooted binary species tree Newick file.
- One AleRax-style family file with a `[FAMILIES]` section.
- One or more gene-tree Newick files referenced by the family file.
- Optional mapping files that assign gene leaves to species.
- One flat JSON `RunConfig` file for the CLI.

The species tree must contain exactly one rooted binary tree.  Gene-tree files
may contain one tree or multiple semicolon-delimited tree records.  Branch
lengths are accepted but ignored by the retained parser.  Labels must be
ordinary unquoted Newick labels; quoted labels, comments, NHX metadata, and
embedded delimiter characters are not supported.

## Family File

The family file uses a deliberately small AleRax-style format:

```text
[FAMILIES]
- family_id
starting_gene_tree = relative/path/to/gene-tree.nwk
gene_tree = relative/path/to/bootstrap-or-sample.nwk
mapping = relative/path/to/gene-leaves.map
```

Each family begins with `- name`.  The accepted keys are
`starting_gene_tree`, `gene_tree`, and `mapping`.  At least one gene tree is
required per family.  Both `starting_gene_tree` and `gene_tree` entries are
loaded as gene-tree inputs for that family, so use additional `gene_tree`
lines when a family has multiple sampled or bootstrap trees.  Family names
must be unique.
Production family files should be structured for multiple families, multiple trees per family, and explicit mapping files.

Relative gene-tree and mapping paths are resolved relative to the family file,
not the process working directory.  Put the family file next to its tree
directory, or keep paths explicitly relative to that file's location.

## Leaf To Species Mapping

Use explicit mapping files when gene leaf names do not directly equal species
names.  The mapping format is:

```text
Species:gene1;gene2
Other_species:gene3
```

Each line maps one species to one or more gene leaves.  A gene leaf may appear
in only one species assignment; duplicate gene assignments are rejected during
validation.  Mapping files are optional only when the legacy direct prefix
fallback is sufficient for every gene leaf.

For direct `GeneReconModel.from_trees(...)` calls without an AleRax family
file, the legacy fallback maps `Species_gene` to `Species` and a leaf without
`_` to the full leaf label.  Production users should prefer explicit
`mapping` entries in family files.  For custom direct preprocessing code, the
narrow supported lower-level escape hatch is
`GeneDataset(..., leaf_species_maps=...)`.

## Mapping Conversion Guidance

Common source conventions can be normalized to the same explicit mapping file
shape:

- Treerecs / GeneRax style `Species:gene1;gene2` lines: keep as-is.
- AleRax family `mapping = ...` paths: point each family to one normalized
  mapping file in that same `Species:gene1;gene2` format.
- OrthoFinder-style gene labels (for example `Species_gene`): split at the
  first `_`, then emit explicit mapping lines rather than relying on fallback.
- Simple `gene -> species` TSVs: invert into grouped
  `Species:gene1;gene2;...` entries before preflight validation.

After conversion, run `gpurec validate-inputs --json` (or
`validate-config --check-preprocess`) and treat any missing species or
duplicate gene assignment as an input contract failure.

## JSON Run Config

The CLI accepts a flat JSON config.  Paths inside the JSON config, including
`species_tree`, `families_file`, `out_dir`, and `resume_from`, are resolved
relative to the JSON config file:

```json
{
  "species_tree": "S.tree",
  "families_file": "families.txt",
  "out_dir": "output_gpurec",
  "mode": "genewise",
  "device": "cuda",
  "optimizer": "auto"
}
```

The `optimizer=auto` route keeps the production defaults: `mode=genewise`
uses `hessian-sgd`, `mode=specieswise` uses `adagrad-restarts`, and
`mode=global` uses `adam`. `validate-config`, summaries, and checkpoint route
metadata report `mode_default_optimizer` and `uses_mode_default_optimizer` so
automation can verify whether a config remains on that default. They also
report `final_check_iters` and `final_check_iters_e` so preflight and final
artifacts agree on the likelihood/gradient validation budgets. Generate a
starter config with:

```bash
gpurec config-template --mode genewise --output run.json
gpurec config-template --mode specieswise --output specieswise-run.json
```

`gpurec config-template --mode global` is available for shared-rate
diagnostic configs, but it is outside the strict HOGENOM/`test_trees_1000`
production-route gate and will not pass `--require-production-default-route`.

## Preflight Validation

Run the lightweight config and file-reference validation first:

```bash
gpurec validate-config --config run.json
gpurec validate-config --config run.json --require-mode-default-optimizer
gpurec validate-config --config run.json --require-production-default-route
```

Then ask the retained Rust preprocessing parser to read the selected species
tree, family records, gene trees, and mapping files on CPU:

```bash
gpurec validate-config --config run.json --check-preprocess
```

Workflow preprocessing uses the native Rust `crates/gpurec-preprocess`
extension.  Source checkouts and unpacked source archives can build that native
extension from the included Cargo manifest.  Wheel-only deployments should set
`GPUREC_PREPROCESS_NATIVE_LIB` to a compatible prebuilt extension before using
`--check-preprocess`, `gpurec optimize`, or `gpurec run`.  Run
`gpurec preprocess-check` first to validate `GPUREC_PREPROCESS_NATIVE_LIB`,
`--preprocess-native-lib`, or the source-tree Cargo fallback without reading
dataset files.  The `GPUREC_PREPROCESS_BIN` CLI override is for the subprocess
adapter and profiling helpers, not a workflow model-construction fallback.

The validation output reports the effective optimizer, batch packing, family
chunking, solver budgets, optimizer-specific defaults, and, when
`--check-preprocess` is used, `cuda_backward_ready` for the current species
node count. Treat parser failures as input contract failures; fix the
referenced tree, family, or mapping file before starting `gpurec optimize`.
Use `--require-mode-default-optimizer` when automation should reject explicit
optimizer overrides. Use `--require-production-default-route` for release or
pipeline launch checks that should also reject changed HOGENOM/`test_trees_1000`
route settings, non-default resident batching, stale `final_check_iters_e`, or
stale likelihood/gradient route metadata before spending CUDA time. That
stricter gate is scoped to the retained
genewise `hessian-sgd` and specieswise `adagrad-restarts` profiles; `mode=global`
can pass the mode-default optimizer gate but fails the production-route gate as
outside the shipped HOGENOM/`test_trees_1000` optimizer route.
For production launch checks, add `--require-cuda-backward-ready` so a species
tree that fails the retained CUDA backward size gate exits nonzero.

The checked files under `examples/` are source-checkout and source-archive
fixtures for parser/config validation.  They are useful for confirming command
shape, but they are not CPU optimizer smokes.  The optimized likelihood path
currently requires CUDA, and the retained Pi backward path currently requires
more than 256 postorder species nodes (`S > 256`); tiny fixtures report
`cuda_backward_ready=false`.

## Validation Fixtures

Use `docs/workflow-examples/input-validation-fixtures/` for a quick
`validate-inputs` smoke with both passing and failing cases:

- `valid/`: complete working dataset.
- `duplicate-family-names/`: parser rejection for repeated AleRax family IDs.
- `missing-mapping/`: parser rejection for a missing mapping target.
- `malformed-newick/`: preprocessing rejection for invalid gene tree syntax.
- `unsupported-species-topology/`: preprocessing rejection for non-binary
  species trees.

The bad fixture commands intentionally fail with machine-readable issue entries and
consistent exit status, making them suitable for automation checks.

## Large Family Sets

For large datasets, validate a small family window first, then scale:

```bash
gpurec validate-inputs \
  --species-tree species_tree.nwk \
  --families-file families.txt \
  --start 0 \
  --max-families 25 \
  --check-preprocess \
  --json
```

Use `--start` and `--max-families` to sample the first `N` families (or later
windows) before full optimization.

Use the preprocess report counts as a memory estimate input:

- larger `preprocessed_species_nodes`, per-family clade counts, and split counts
  usually require more GPU memory,
- a single very large family can dominate resident-batch memory.

If memory pressure appears during optimization, tune these run-config controls:

- lower `clade_budget` to force smaller resident batches,
- set a positive `family_chunk_size` so the workflow does not try to keep one
  oversized all-family resident batch.
