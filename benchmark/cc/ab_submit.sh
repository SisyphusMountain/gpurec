#!/bin/bash
# Submit one H100 job that runs a sequence of comparison arms back to back.
#
# Usage: ab_submit.sh JOBNAME HH:MM:SS ARM [ARM ...]
#   ARM = CODE,MODE,DATASET,LIMIT,TAG
#     CODE    old | new           (which checkout provides the gpurec package)
#     MODE    genewise | global
#     DATASET archaea | hogenom1055 | hogenomfull
#     LIMIT   0 = every family in the list, N = the first N
#     TAG     result name; $CC_RUNS/results/<TAG>.json and .pt
#
# The dataset paths below are hard-wired: they are facts about where the data sits on this cluster,
# not settings, and both codes must be pointed at byte-identical inputs for the comparison to mean
# anything. The family lists are the ones `ab_make_lists.py` writes.
set -uo pipefail
source /sps/biometr/emarsot/gpurec/benchmark/cc/env.sh

ARCHAEA_SPECIES=/sps/biometr/emarsot/gpurec-data/archaea/species_reference/reference_species_tree.newick
HOGENOM_SPECIES="/sps/biometr/emarsot/gpurec-data/hogenom/hogenom/runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick"

NAME=$1
TIME=$2
shift 2
CMD=""
for ARM in "$@"; do
    IFS=, read -r CODE MODE DATASET LIMIT TAG <<< "$ARM"
    case "$DATASET" in
        archaea)      SPECIES=$ARCHAEA_SPECIES; LIST=$CC_RUNS/ablists/archaea_all.txt ;;
        hogenom1055)  SPECIES=$HOGENOM_SPECIES; LIST=$CC_RUNS/ablists/hogenom_1055_all.txt ;;
        hogenomfull)  SPECIES=$HOGENOM_SPECIES; LIST=$CC_RUNS/ablists/hogenom_full.txt ;;
        *) echo "unknown dataset: $DATASET" >&2; exit 2 ;;
    esac
    CMD="$CMD bash $CC_REPO/benchmark/cc/ab_run.sh $CODE $MODE '$SPECIES' $LIST $LIMIT $TAG ;"
done
bash "$CC_REPO/benchmark/cc/sbatch_h100.sh" "$NAME" "$TIME" "$CMD"
