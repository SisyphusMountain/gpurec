#!/bin/bash
# Score ONE fitted theta with the converged reference solver and cross-check it against AleRax.
# Used for the full 10,869-family HOGENOM set, where only the new code is fitted, so there is no
# second theta for compare_fit_thetas.py to compare against.
#
# Usage: ab_xcheck_single.sh DATASET TAG NAME_RULE      (arguments as in ab_compare.sh)
set -uo pipefail
source /sps/biometr/emarsot/gpurec/benchmark/cc/env.sh
DATASET=$1; TAG=$2; NAME_RULE=$3
OUT=$CC_RUNS/results
ALERAX=$CC_REPO/benchmarks/hogenom-cpu-vs-gpu/results/alerax_hogenom_combined_likelihoods.txt
HOGENOM_SPECIES="/sps/biometr/emarsot/gpurec-data/hogenom/hogenom/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick"
ARCHAEA_SPECIES=/sps/biometr/emarsot/gpurec-data/archaea/species_reference/reference_species_tree.newick

case "$DATASET" in
    archaea)     SPECIES=$ARCHAEA_SPECIES; LIST=$CC_RUNS/ablists/archaea_all.txt;      BUDGET=315000 ;;
    hogenom1055) SPECIES=$HOGENOM_SPECIES; LIST=$CC_RUNS/ablists/hogenom_1055_all.txt; BUDGET=900000 ;;
    hogenomfull) SPECIES=$HOGENOM_SPECIES; LIST=$CC_RUNS/ablists/hogenom_full.txt;     BUDGET=900000 ;;
    *) echo "unknown dataset: $DATASET" >&2; exit 2 ;;
esac

export PYTHONPATH=$CC_REPO
cd "$CC_REPO" || exit 1
echo "[ab_xcheck_single] dataset=$DATASET tag=$TAG start=$(date -Is)"
$CC_PY benchmark/cc/score_per_family.py --species "$SPECIES" --families "$LIST" \
    --theta "$OUT/$TAG.pt" --clade-budget $BUDGET --pi-iters 64 --neumann-terms 64 \
    --forward-self-loop exact --out "$OUT/scored_$TAG.pt"
$CC_PY benchmark/cc/xcheck_alerax.py --label "$TAG" \
    --nll-source "$OUT/scored_$TAG.pt" --nll-key nll_bits --paths-source "$OUT/scored_$TAG.pt" \
    --name-rule "$NAME_RULE" --alerax-likelihoods "$ALERAX" --out "$OUT/xcheck_$TAG.json"
echo "[ab_xcheck_single] done end=$(date -Is)"
