#!/usr/bin/env bash
#SBATCH --job-name=gpurec-end-to-end
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../end-to-end-tutorial"

export OMP_NUM_THREADS=1

run_json=run.json

if [[ -f output_gpurec/checkpoints/latest.pt ]]; then
  resume_args=(--resume-from output_gpurec/checkpoints/latest.pt)
else
  resume_args=()
fi

gpurec validate-config \
  --config "$run_json" \
  --check-preprocess \
  --require-mode-default-optimizer \
  --require-cuda-backward-ready

gpurec optimize \
  --config "$run_json" \
  "${resume_args[@]}" \
  --require-mode-default-optimizer \
  --require-final-check-ok

if [[ -f output_gpurec/checkpoints/best.pt ]]; then
  gpurec summary-info --summary output_gpurec/summary.json --require-final-check-ok
  gpurec sample \
    --checkpoint output_gpurec/checkpoints/best.pt \
    --samples 2 \
    --seed 42 \
    --sample-out-dir output_gpurec
fi
