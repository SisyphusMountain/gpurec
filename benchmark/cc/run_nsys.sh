#!/bin/bash
# Nsight-Systems capture of one genewise loss+gradient call and one curvature call.
#
# The $NSYS path in env.sh does not exist on the gpu_h100 compute nodes, so this script searches for
# an nsys binary before running, prints where it found one (or every place it looked, when it found
# none), then writes <outbase>.nsys-rep and the cuda_gpu_kern_sum / cuda_api_sum / nvtx_sum CSVs
# beside it.
#
# Usage (from $CC_REPO, inside a job):  bash benchmark/cc/run_nsys.sh LIMIT CLADE_BUDGET HESSIAN OUTBASE
#   HESSIAN is "library" or "streamed" (see nsys_grad.py --hessian).
set -uo pipefail
LIMIT=$1
CLADE_BUDGET=$2
HESSIAN=$3
OUTBASE=$4

source /sps/biometr/emarsot/gpurec/benchmark/cc/env.sh
cd "$CC_REPO" || exit 1

NSYS_BIN=""
for candidate in "$NSYS" $(command -v nsys) /opt/nvidia/nsight-systems/*/target-linux-x64/nsys \
                 /usr/local/cuda/bin/nsys /usr/local/bin/nsys; do
    if [ -x "$candidate" ]; then NSYS_BIN="$candidate"; break; fi
done
if [ -z "$NSYS_BIN" ]; then
    NSYS_BIN=$(find /opt /usr/local /usr/lib64 /usr/share /cvmfs -maxdepth 6 -name nsys -type f 2>/dev/null | head -1)
fi
if [ -z "$NSYS_BIN" ]; then
    echo "[nsys] NOT FOUND. Looked at \$NSYS=$NSYS, PATH, /opt/nvidia/nsight-systems/*, /usr/local/cuda/bin."
    echo "[nsys] /opt contents:"; ls -la /opt 2>/dev/null
    echo "[nsys] nvidia dirs:"; ls -la /opt/nvidia 2>/dev/null
    echo "[nsys] cuda dirs:"; ls -d /usr/local/cuda* 2>/dev/null
    exit 3
fi
echo "[nsys] binary=$NSYS_BIN"
"$NSYS_BIN" --version

"$NSYS_BIN" profile --trace=cuda,nvtx,osrt --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
    -o "$OUTBASE" \
    "$CC_PY" -u benchmark/cc/nsys_grad.py \
        --species "$CC_SPECIES" --families "$CC_FAMILIES" \
        --limit "$LIMIT" --clade-budget "$CLADE_BUDGET" --hessian "$HESSIAN"
echo "[nsys] profile rc=$?"

"$NSYS_BIN" stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
    --output "$OUTBASE" "$OUTBASE.nsys-rep"
echo "[nsys] stats rc=$?"
ls -la "$OUTBASE"*
