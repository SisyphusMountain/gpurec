#!/bin/bash
#SBATCH --job-name=sol_coleman_final
#SBATCH --partition=gpu_h100_interactive
#SBATCH --account=biometr
#SBATCH --licenses=sps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --time=00:15:00

set -euo pipefail
campaign_repo=$1
campaign_em2_repo=$2
campaign_python=/sps/biometr/emarsot/envs/gpurec-h100/bin/python
campaign_data=/sps/biometr/emarsot/gpurec-data/coleman
export PYTHONPATH="$campaign_repo"
export PYTHONNOUSERSITE=1
export TRITON_CACHE_DIR=/sps/biometr/emarsot/.triton-cache-h100
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=32
cd "$campaign_repo"
printf '[campaign] host=%s job=%s start=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$(date -Is)"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
sha256sum gpurec/fit/genewise_fit.py gpurec/fit/em_warmup.py gpurec/fit/dtl_fit.py \
    gpurec/api/_implicit_grad.py gpurec/api/_execution.py gpurec/api/model.py \
    benchmark/cc/run_genewise.py experiments/coleman_sol_20260906/audit_fits.py
"$campaign_python" benchmark/cc/run_genewise.py \
    --species "$campaign_data/ReferenceTree.nwk" \
    --families "$campaign_data/families_all.txt" \
    --limit 0 --clade-budget 0 --init-rate none \
    --warmup-method em --em-steps 2 \
    --out-dir "$campaign_repo/results" --tag em2_full_b
"$campaign_python" experiments/coleman_sol_20260906/audit_fits.py \
    --species "$campaign_data/ReferenceTree.nwk" \
    --baseline "$campaign_em2_repo/results/adam_full_b.pt" \
    --candidate "$campaign_repo/results/em2_full_b.pt" \
    --candidate-repeat "$campaign_em2_repo/results/em_full_a.pt" --repeat-baseline \
    --out "$campaign_repo/results/audit_em2_repeat.json"
printf '[campaign] end=%s\n' "$(date -Is)"
