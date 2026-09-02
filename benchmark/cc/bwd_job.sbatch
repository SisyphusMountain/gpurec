#!/bin/bash
#SBATCH --job-name=bwdjob
#SBATCH --partition=gpu_h100
#SBATCH --account=biometr
#SBATCH --licenses=sps
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --gres=gpu:h100:1 --mem=96G --time=00:40:00
#SBATCH --output=/sps/biometr/emarsot/gpurec-runs/%x-%j.out
set -uo pipefail
export CC_REPO=/sps/biometr/emarsot/gpurec_bwd
source $CC_REPO/benchmark/cc/env.sh
cd $CC_REPO
echo "[job] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) start=$(date -Is)"
bash $CC_REPO/benchmark/cc/bwd_payload.sh
echo "[job] exit=$? end=$(date -Is)"
