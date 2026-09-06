#!/bin/bash
#SBATCH --job-name=sol_coleman_base
#SBATCH --partition=gpu_h100
#SBATCH --account=biometr
#SBATCH --licenses=sps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --time=00:35:00

set -euo pipefail
campaign_repo=$1
shift
if [ "$#" -eq 0 ]; then
    set -- baseline_full_a baseline_full_b
fi
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
sha256sum gpurec/fit/genewise_fit.py gpurec/api/_implicit_grad.py gpurec/config/gpurec_config.py "$campaign_data/families_all.txt" "$campaign_data/ReferenceTree.nwk"
for campaign_tag in "$@"; do
    "$campaign_python" benchmark/cc/run_genewise.py \
        --species "$campaign_data/ReferenceTree.nwk" \
        --families "$campaign_data/families_all.txt" \
        --limit 0 --clade-budget 0 --init-rate none \
        --out-dir "$campaign_repo/results" --tag "$campaign_tag"
done
printf '[campaign] end=%s\n' "$(date -Is)"
