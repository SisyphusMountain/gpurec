#!/bin/bash
#SBATCH --job-name=sol_coleman_em2
#SBATCH --partition=gpu_h100_interactive
#SBATCH --account=biometr
#SBATCH --licenses=sps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --time=00:40:00

set -euo pipefail
campaign_repo=$1
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
sha256sum gpurec/fit/genewise_fit.py gpurec/fit/em_warmup.py gpurec/api/_implicit_grad.py gpurec/config/gpurec_config.py "$campaign_data/families_all.txt" "$campaign_data/ReferenceTree.nwk"
for campaign_case in em_full_a adam_full_b em_full_b; do
    campaign_method=${campaign_case%%_*}
    "$campaign_python" benchmark/cc/run_genewise.py \
        --species "$campaign_data/ReferenceTree.nwk" \
        --families "$campaign_data/families_all.txt" \
        --limit 0 --clade-budget 0 --init-rate none \
        --warmup-method "$campaign_method" \
        --out-dir "$campaign_repo/results" --tag "$campaign_case"
done
"$campaign_python" experiments/coleman_sol_20260906/audit_fits.py \
    --species "$campaign_data/ReferenceTree.nwk" \
    --baseline "$campaign_repo/results/adam_full_b.pt" \
    --candidate "$campaign_repo/results/em_full_b.pt" \
    --out "$campaign_repo/results/audit_em_b.json"
printf '[campaign] end=%s\n' "$(date -Is)"
