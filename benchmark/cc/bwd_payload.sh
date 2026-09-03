#!/bin/bash
# What bwd_job.sbatch runs. Edit this file, rsync it, resubmit the same sbatch file.
#
# Final verification of the additive valid-receiver mass plus the 8-warp transfer-subtree launch:
# float64 oracle, run-to-run atomics noise, a same-job A/B of one gradient at 500 families, and
# the two reference genewise fits.
set -uo pipefail
BASE=/sps/biometr/emarsot/gpurec_bwd_base
NEW=/sps/biometr/emarsot/gpurec_bwd
OUT=$CC_RUNS/bwd
mkdir -p $OUT
COMMON="--species $CC_SPECIES --families $CC_FAMILIES --limit 100 --theta-pt $CC_RUNS/results/full_v3.pt"

for DT in float64 float32; do
  echo "############ CHECK 100 families, new, $DT ############"
  PYTHONPATH=$NEW $CC_PY -u $NEW/benchmark/cc/bwd_kernels.py --mode check $COMMON \
    --dtype $DT --out $OUT/final_new_${DT}.pt
done
echo "############ COMPARE fp32 old vs new ############"
$CC_PY -u $NEW/benchmark/cc/bwd_kernels.py --mode compare \
  --a $OUT/check_base_float32.pt --b $OUT/final_new_float32.pt
echo "############ ORACLE: fp32 old and new against the float64 pair ############"
$CC_PY -u $NEW/benchmark/cc/bwd_kernels.py --mode oracle \
  --oracle $OUT/check_base_float64.pt --oracle-b $OUT/final_new_float64.pt \
  --a $OUT/check_base_float32.pt --b $OUT/final_new_float32.pt

echo "############ TIME 500 families, BASELINE ############"
PYTHONPATH=$BASE $CC_PY -u $NEW/benchmark/cc/bwd_kernels.py --mode time --species $CC_SPECIES \
  --families $CC_FAMILIES --limit 500 --theta-pt $CC_RUNS/results/full_v3.pt
echo "############ TIME 500 families, NEW ############"
PYTHONPATH=$NEW $CC_PY -u $NEW/benchmark/cc/bwd_kernels.py --mode time --species $CC_SPECIES \
  --families $CC_FAMILIES --limit 500 --theta-pt $CC_RUNS/results/full_v3.pt

echo "############ FIT 40 families (bwd_smoke40) ############"
$CC_PY -u $NEW/benchmark/cc/run_genewise.py --species $CC_SPECIES --families $CC_FAMILIES --limit 40 \
  --forward-self-loop exact --adjoint-self-loop exact --init-rate none --clade-budget 0 \
  --out-dir $CC_RUNS/results --tag bwd_smoke40
echo "############ FIT 500 families (bwd_500) ############"
$CC_PY -u $NEW/benchmark/cc/run_genewise.py --species $CC_SPECIES --families $CC_FAMILIES --limit 500 \
  --forward-self-loop exact --adjoint-self-loop exact --init-rate none --clade-budget 0 \
  --out-dir $CC_RUNS/results --tag bwd_500
