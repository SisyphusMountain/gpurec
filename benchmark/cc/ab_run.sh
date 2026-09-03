#!/bin/bash
# Run ONE arm of the archaea / HOGENOM old-vs-new comparison on the cluster.
#
# Usage: ab_run.sh CODE MODE SPECIES FAMILIES LIMIT TAG
#   CODE     old = the original checkout /sps/biometr/emarsot/gpurec_base (commit 817007e6)
#            new = the branch-HEAD checkout $CC_REPO
#   MODE     genewise | global
#   SPECIES  species tree newick
#   FAMILIES list file, one gene-tree path per line
#   LIMIT    0 = all families in the list, N = first N
#   TAG      output name; results land in $CC_RUNS/results/<TAG>.json and .pt
#
# Each arm runs its OWN checkout's default recipe -- that is the whole point of the comparison, so
# no solver settings are passed to the old arm and the new arm only names the kernel paths that are
# already its defaults (exact forward and adjoint self-loops, fit_dtl's own genewise start).
#
# GPUREC_MEMORY_POLICY_FRACTION=0.3 on the old arm: the original code sizes its resident
# warm-adjoint cache from the gradient adjoint only and then runs out of memory inside the 3-probe
# Hessian on a 94 GiB H100 (this is exactly the miscount that HEAD's memory_policy fixes with its
# resident_caches argument). Shrinking the policy's memory budget to 30% makes the original code
# decide to run the adjoint cold instead, which is the only way it completes at 500+ families.
set -uo pipefail
CODE=$1; MODE=$2; SPECIES=$3; FAMILIES=$4; LIMIT=$5; TAG=$6
CC_BASE=/sps/biometr/emarsot/gpurec_base
OUT=$CC_RUNS/results
mkdir -p "$OUT"

echo "[ab_run] code=$CODE mode=$MODE tag=$TAG limit=$LIMIT families=$FAMILIES start=$(date -Is)"
if [ "$CODE" = "old" ]; then
    export PYTHONPATH=$CC_BASE
    export GPUREC_MEMORY_POLICY_FRACTION=0.3
    cd "$CC_BASE" || exit 1
    if [ "$MODE" = "genewise" ]; then
        $CC_PY "$CC_BASE/benchmark/cc/run_genewise.py" --species "$SPECIES" --families "$FAMILIES" \
            --limit "$LIMIT" --out-dir "$OUT" --tag "$TAG"
    else
        $CC_PY "$CC_REPO/benchmark/cc/run_global.py" --species "$SPECIES" --families "$FAMILIES" \
            --limit "$LIMIT" --out-dir "$OUT" --tag "$TAG"
    fi
else
    export PYTHONPATH=$CC_REPO
    cd "$CC_REPO" || exit 1
    if [ "$MODE" = "genewise" ]; then
        $CC_PY "$CC_REPO/benchmark/cc/run_genewise.py" --species "$SPECIES" --families "$FAMILIES" \
            --limit "$LIMIT" --out-dir "$OUT" --tag "$TAG" \
            --forward-self-loop exact --adjoint-self-loop exact --init-rate none
    else
        $CC_PY "$CC_REPO/benchmark/cc/run_global.py" --species "$SPECIES" --families "$FAMILIES" \
            --limit "$LIMIT" --out-dir "$OUT" --tag "$TAG"
    fi
fi
RC=$?
echo "[ab_run] tag=$TAG rc=$RC end=$(date -Is)"
exit $RC
