#!/bin/bash
# Shared paths for gpurec jobs on the CC-IN2P3 cluster (ssh cc). Source this from job scripts.
export CC_REPO=/sps/biometr/emarsot/gpurec
export CC_PY=/sps/biometr/emarsot/envs/gpurec-h100/bin/python
export CC_DATA=/sps/biometr/emarsot/gpurec-data/coleman
export CC_RUNS=/sps/biometr/emarsot/gpurec-runs
export CC_SPECIES=$CC_DATA/ReferenceTree.nwk
export CC_FAMILIES=$CC_DATA/families_no_largest.txt
# Run the repo checkout, not any installed copy; keep ~/.local out of the picture.
export PYTHONPATH=$CC_REPO
export PYTHONNOUSERSITE=1
# Persistent Triton kernel cache shared across jobs (compiles once per kernel/shape).
export TRITON_CACHE_DIR=/sps/biometr/emarsot/.triton-cache-h100
# Nsight Systems on the gpu_h100 nodes (verified job 57472282: 2026.1.3). The old
# /opt/nvidia/nsight-systems/2025.6.3/... path does not exist there and fails with rc=127.
export NSYS=/usr/local/bin/nsys
export NCU=/usr/local/cuda/bin/ncu
mkdir -p "$CC_RUNS" "$TRITON_CACHE_DIR"
