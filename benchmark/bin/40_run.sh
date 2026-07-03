#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"
mkdir -p "$RESULTS_DIR"

if [ "$DATASET" != "archaea" ]; then log "only archaea wired for 40_run"; exit 1; fi
GENES="$ARCHAEA_DATA_DIR/ale_gene_tree_distributions/main_families_ge4seq"
log "fitting gpurec ($MODE) on archaea60 main families -> $RESULTS_DIR/rates.txt"
"$PY" "$HERE/bench_gpurec_fit.py" --species "$ARCHAEA_SPECIES_TREE" \
  --gene "$GENES" --mode "$MODE" --steps 300 --out "$RESULTS_DIR/rates.txt"
