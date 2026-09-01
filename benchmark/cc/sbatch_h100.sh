#!/bin/bash
# Submit a one-GPU H100 job on CC-IN2P3.
# Usage: sbatch_h100.sh JOBNAME HH:MM:SS 'command to run'   (runs from $CC_REPO with env.sh sourced)
set -euo pipefail
NAME=$1; TIME=$2; CMD=$3
GPUS=${GPUS:-1}
MEM=${MEM:-96G}
CPUS=${CPUS:-12}
PART=${PART:-gpu_h100}
source /sps/biometr/emarsot/gpurec/benchmark/cc/env.sh
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --partition=$PART
#SBATCH --account=biometr
#SBATCH --licenses=sps
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=$CPUS --gres=gpu:h100:$GPUS --mem=$MEM --time=$TIME
#SBATCH --output=$CC_RUNS/%x-%j.out
set -uo pipefail
source $CC_REPO/benchmark/cc/env.sh
cd \$CC_REPO
echo "[job] host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) start=\$(date -Is)"
$CMD
echo "[job] exit=\$? end=\$(date -Is)"
EOT
