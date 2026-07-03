#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

# Requires an AleRax output dir ($ALERAX_DIR) to correlate against.
: "${ALERAX_DIR:?set ALERAX_DIR to an AleRax output directory}"
GENES="${GENES:?set GENES to the gene-tree files/dir}"
log "checking gpurec-vs-AleRax fidelity against $ALERAX_DIR"
"$PY" "$HERE/eval_at_alerax_rates.py" --species "$ARCHAEA_SPECIES_TREE" \
  --gene $GENES --alerax-dir "$ALERAX_DIR"
