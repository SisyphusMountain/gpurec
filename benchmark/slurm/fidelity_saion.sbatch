#!/usr/bin/env bash
#SBATCH --job-name=gpurec-fidelity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate 2>/dev/null || true
DATASET=archaea MODE=global bash benchmark/bin/70_check_fidelity.sh
