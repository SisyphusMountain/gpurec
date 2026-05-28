# Input Validation Fixtures

This directory contains tiny AleRax-style fixtures for `validate-inputs` smoke checks.
Each scenario is fully standalone and uses only local `.nwk` and `.tsv` files.
These checks run without constructing a CUDA model.

- `valid/`: complete, parseable minimal dataset that should pass without preprocess
  checks.
- `duplicate-family-names/`: parser failure for duplicated family names.
- `missing-mapping/`: parser failure because the family `mapping` target does not exist.
- `malformed-newick/`: preprocess failure from an invalid gene-tree Newick string.
- `unsupported-species-topology/`: preprocess failure from a non-binary species tree.

## Good-input smoke

From the repository root:

```bash
python -m gpurec.cli validate-inputs \
  --species-tree docs/workflow-examples/input-validation-fixtures/valid/species_tree.nwk \
  --families-file docs/workflow-examples/input-validation-fixtures/valid/families.txt \
  --json
```

Expected: `valid_inputs: true` in JSON output.

## Failure examples

```bash
python -m gpurec.cli validate-inputs \
  --species-tree docs/workflow-examples/input-validation-fixtures/duplicate-family-names/species_tree.nwk \
  --families-file docs/workflow-examples/input-validation-fixtures/duplicate-family-names/families.txt \
  --json
```

```bash
python -m gpurec.cli validate-inputs \
  --species-tree docs/workflow-examples/input-validation-fixtures/missing-mapping/species_tree.nwk \
  --families-file docs/workflow-examples/input-validation-fixtures/missing-mapping/families.txt \
  --json
```

```bash
python -m gpurec.cli validate-inputs \
  --species-tree docs/workflow-examples/input-validation-fixtures/malformed-newick/species_tree.nwk \
  --families-file docs/workflow-examples/input-validation-fixtures/malformed-newick/families.txt \
  --check-preprocess \
  --json
```

```bash
python -m gpurec.cli validate-inputs \
  --species-tree docs/workflow-examples/input-validation-fixtures/unsupported-species-topology/species_tree.nwk \
  --families-file docs/workflow-examples/input-validation-fixtures/unsupported-species-topology/families.txt \
  --check-preprocess \
  --json
```

All failing commands should return nonzero with machine-readable `issues` entries.

Each issue entry is expected to include: file path, family name, affected label,
expected format, and next action.

Structured reports should cover every family with issue categories such as
missing mapping, duplicate family name, duplicate species mappings, rejected tree,
and species coverage.
