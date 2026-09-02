#!/bin/bash
# What bwd_job.sbatch runs. Edit this file, rsync it, resubmit the same sbatch file.
# Current contents: the run-to-run atomics noise of the gradient at both thetas, then the two
# reference genewise fits.
set -uo pipefail
REPO=$CC_REPO
OUT=$CC_RUNS/bwd
mkdir -p $OUT

echo "############ CHECK 100 families: run-to-run gradient noise at both thetas ############"
$CC_PY -u $REPO/benchmark/cc/bwd_kernels.py --mode check --species $CC_SPECIES \
  --families $CC_FAMILIES --limit 100 --theta-pt $CC_RUNS/results/full_v3.pt --out $OUT/check_head.pt

echo "############ FIT 40 families (bwd_smoke40) ############"
$CC_PY -u $REPO/benchmark/cc/run_genewise.py --species $CC_SPECIES --families $CC_FAMILIES --limit 40 \
  --forward-self-loop exact --adjoint-self-loop exact --init-rate none \
  --out-dir $CC_RUNS/results --tag bwd_smoke40

echo "############ FIT 500 families (bwd_500) ############"
$CC_PY -u $REPO/benchmark/cc/run_genewise.py --species $CC_SPECIES --families $CC_FAMILIES --limit 500 \
  --forward-self-loop exact --adjoint-self-loop exact --init-rate none \
  --out-dir $CC_RUNS/results --tag bwd_500
