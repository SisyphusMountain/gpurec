#!/usr/bin/env bash
# Scale-benchmark configuration. Edit for your cluster/dataset.
set -euo pipefail

# Dataset selector: "williams" (Zenodo download) or "archaea" (local sibling repo).
DATASET="${DATASET:-archaea}"

REPO="$(git rev-parse --show-toplevel)"
GPUREC_DATA_ROOT="${GPUREC_DATA_ROOT:-$REPO/data/external/benchmarks/large_dataset_capacity/datasets}"

# The full archaea dataset lives in the ignored data store; the 60-taxon tree is
# also vendored as a small test fixture.
ARCHAEA_DATA_DIR="${ARCHAEA_DATA_DIR:-$GPUREC_DATA_ROOT/alerax_archaea_davin2017}"
ARCHAEA_SPECIES_TREE="${ARCHAEA_SPECIES_TREE:-$REPO/tests/data/archaea60/reference_species_tree.newick}"

# Williams et al. 2017 (headline fidelity dataset) is Zenodo-only.
ZENODO_TARBALL_URL="${ZENODO_TARBALL_URL:-https://zenodo.org/record/17360806/files/williams2017.tar.gz}"

# gpurec mode -> AleRax parametrization.
MODE="${MODE:-global}"          # global | specieswise | genewise
case "$MODE" in
  global)      ALERAX_PARAM="GLOBAL" ;;
  specieswise) ALERAX_PARAM="PER-SPECIES" ;;
  genewise)    ALERAX_PARAM="PER-FAMILY" ;;
esac

RESULTS_DIR="${RESULTS_DIR:-$REPO/benchmark/results}"
PY="${PY:-python}"
