#!/bin/bash
# Fitted-theta vs flat-theta gradient profile: an Nsight Systems capture of one gradient per theta,
# then a separate un-profiled process that counts the iterations of the two syncing E fixed-point
# loops. Two processes because the counting monkey-patches would distort the nsys timings.
#
# Usage (from $CC_REPO, inside a job):
#   bash benchmark/cc/run_nsys_theta.sh LIMIT CLADE_BUDGET THETA_PT OUTBASE
set -uo pipefail
LIMIT=$1
CLADE_BUDGET=$2
THETA_PT=$3
OUTBASE=$4

source /sps/biometr/emarsot/gpurec/benchmark/cc/env.sh
cd "$CC_REPO" || exit 1

NSYS_BIN=""
for candidate in "$NSYS" $(command -v nsys) /opt/nvidia/nsight-systems/*/target-linux-x64/nsys \
                 /usr/local/cuda/bin/nsys /usr/local/bin/nsys; do
    if [ -x "$candidate" ]; then NSYS_BIN="$candidate"; break; fi
done
if [ -z "$NSYS_BIN" ]; then
    echo "[nsys] NOT FOUND (looked at \$NSYS=$NSYS, PATH, /opt/nvidia/nsight-systems/*, /usr/local)."
    exit 3
fi
echo "[nsys] binary=$NSYS_BIN"; "$NSYS_BIN" --version

echo "=== part 1: nsys capture ==="
"$NSYS_BIN" profile --trace=cuda,nvtx,osrt --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
    -o "$OUTBASE" \
    "$CC_PY" -u benchmark/cc/nsys_theta.py \
        --species "$CC_SPECIES" --families "$CC_FAMILIES" \
        --limit "$LIMIT" --clade-budget "$CLADE_BUDGET" --theta-pt "$THETA_PT" \
        --mode nsys --forward-self-loop "${FORWARD_MODE:-linear}" --out "$OUTBASE.nsys.json"
echo "[nsys] profile rc=$?"

"$NSYS_BIN" stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
    --output "$OUTBASE" "$OUTBASE.nsys-rep"
echo "[nsys] stats rc=$?"

echo "=== part 2: E fixed-point iteration counts (no profiler, monkey-patched counters) ==="
"$CC_PY" -u benchmark/cc/nsys_theta.py \
    --species "$CC_SPECIES" --families "$CC_FAMILIES" \
    --limit "$LIMIT" --clade-budget "$CLADE_BUDGET" --theta-pt "$THETA_PT" \
    --mode count --forward-self-loop "${FORWARD_MODE:-linear}" --out "$OUTBASE.count.json"
echo "[count] rc=$?"

ls -la "$OUTBASE"*
