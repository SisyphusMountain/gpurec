#!/usr/bin/env bash
# Scale-benchmark configuration. Edit for your cluster/dataset.
set -euo pipefail

# Dataset selector: "williams" (Zenodo download) or "archaea" (local sibling repo).
DATASET="${DATASET:-archaea}"

# archaea60 lives in a sibling repo; the 60-taxon tree is vendored into the base.
ARCHAEA_DATA_DIR="${ARCHAEA_DATA_DIR:-/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_archaea_davin2017}"
ARCHAEA_SPECIES_TREE="${ARCHAEA_SPECIES_TREE:-$(git rev-parse --show-toplevel)/tests/data/archaea60/reference_species_tree.newick}"

# Williams et al. 2017 (headline fidelity dataset) is Zenodo-only.
ZENODO_TARBALL_URL="${ZENODO_TARBALL_URL:-https://zenodo.org/record/17360806/files/williams2017.tar.gz}"

# gpurec mode -> AleRax parametrization.
MODE="${MODE:-global}"          # global | specieswise | genewise
case "$MODE" in
  global)      ALERAX_PARAM="GLOBAL" ;;
  specieswise) ALERAX_PARAM="PER-SPECIES" ;;
  genewise)    ALERAX_PARAM="PER-FAMILY" ;;
esac

RESULTS_DIR="${RESULTS_DIR:-$(git rev-parse --show-toplevel)/benchmark/results}"
PY="${PY:-python}"
