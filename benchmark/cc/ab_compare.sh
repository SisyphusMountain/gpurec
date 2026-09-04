#!/bin/bash
# Score two fitted thetas with ONE common solver and, when a reference exists, cross-check both
# against AleRax's per-family log-likelihoods.
#
# Usage: ab_compare.sh DATASET LEFT_TAG RIGHT_TAG NAME_RULE
#   DATASET    archaea | hogenom1055 | hogenomfull
#   LEFT_TAG   result tag of the first fit  (by convention: the OLD code)
#   RIGHT_TAG  result tag of the second fit (by convention: the NEW code)
#   NAME_RULE  none          -> no AleRax reference for this dataset, skip the cross-check
#              family-dir    -> HOGENOM: <root>/<FAMILY>/gene_trees/<tree>.newick  -> <FAMILY>
#              basename-noext-> archaea: <dir>/<FAMILY>.ale                        -> <FAMILY>
#
# The scoring solver is the converged reference one (pi_iters 64, neumann_terms 64, exact forward
# self-loop) for BOTH thetas, so any difference reported here is about where the two fits LANDED,
# not about how they were evaluated.
set -uo pipefail
source /sps/biometr/emarsot/gpurec/benchmark/cc/env.sh
DATASET=$1; LEFT_TAG=$2; RIGHT_TAG=$3; NAME_RULE=$4
OUT=$CC_RUNS/results
ALERAX=$CC_REPO/benchmarks/hogenom-cpu-vs-gpu/results/alerax_hogenom_combined_likelihoods.txt
ARCHAEA_SPECIES=/sps/biometr/emarsot/gpurec-data/archaea/species_reference/reference_species_tree.newick
HOGENOM_SPECIES="/sps/biometr/emarsot/gpurec-data/hogenom/hogenom/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick"

case "$DATASET" in
    # clade_budget is a per-batch clade cap; Pi is [clade_budget, S] fp32, so the archaea budget can
    # be generous at S=119 while HOGENOM at S=1331 keeps the 900,000 used by the recorded AleRax
    # cross-check (900,000 x 1331 x 4 B = 4.8 GiB of forward Pi per batch).
    archaea)     SPECIES=$ARCHAEA_SPECIES; LIST=$CC_RUNS/ablists/archaea_all.txt;       BUDGET=315000 ;;
    hogenom1055) SPECIES=$HOGENOM_SPECIES; LIST=$CC_RUNS/ablists/hogenom_1055_all.txt;  BUDGET=900000 ;;
    hogenomfull) SPECIES=$HOGENOM_SPECIES; LIST=$CC_RUNS/ablists/hogenom_full.txt;      BUDGET=900000 ;;
    *) echo "unknown dataset: $DATASET" >&2; exit 2 ;;
esac

export PYTHONPATH=$CC_REPO
cd "$CC_REPO" || exit 1
echo "[ab_compare] dataset=$DATASET left=$LEFT_TAG right=$RIGHT_TAG start=$(date -Is)"
$CC_PY benchmark/cc/compare_fit_thetas.py --species "$SPECIES" --families "$LIST" \
    --left "$OUT/$LEFT_TAG.pt" --right "$OUT/$RIGHT_TAG.pt" \
    --clade-budget $BUDGET --pi-iters 64 --neumann-terms 64 --forward-self-loop exact --top 10 \
    | tee "$OUT/cmp_${LEFT_TAG}__${RIGHT_TAG}.txt"

if [ "$NAME_RULE" != "none" ]; then
    CMP=$OUT/$RIGHT_TAG.pt.cmp_nll.pt
    $CC_PY benchmark/cc/xcheck_alerax.py --label "$LEFT_TAG" \
        --nll-source "$CMP" --nll-key left_nll --paths-source "$OUT/$LEFT_TAG.pt" \
        --name-rule "$NAME_RULE" --alerax-likelihoods "$ALERAX" --out "$OUT/xcheck_$LEFT_TAG.json"
    $CC_PY benchmark/cc/xcheck_alerax.py --label "$RIGHT_TAG" \
        --nll-source "$CMP" --nll-key right_nll --paths-source "$OUT/$LEFT_TAG.pt" \
        --name-rule "$NAME_RULE" --alerax-likelihoods "$ALERAX" --out "$OUT/xcheck_$RIGHT_TAG.json"
fi
echo "[ab_compare] done end=$(date -Is)"
