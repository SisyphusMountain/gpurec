#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

log "dataset=$DATASET mode=$MODE alerax_param=$ALERAX_PARAM"
"$PY" -c "import gpurec; print('gpurec import OK')"
if [ "$DATASET" = "archaea" ]; then
  require_file "$ARCHAEA_SPECIES_TREE"
  [ -d "$ARCHAEA_DATA_DIR" ] || { log "archaea data dir absent: $ARCHAEA_DATA_DIR"; exit 1; }
  log "archaea60 ready: $(find "$ARCHAEA_DATA_DIR/ale_gene_tree_distributions" -name '*.ale' | wc -l) .ale families"
else
  log "williams: fetch $ZENODO_TARBALL_URL (not automated here)"
fi
