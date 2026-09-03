#!/bin/bash
# Submit an H100 job that runs an arbitrary command from $CC_REPO with env.sh sourced.
# A thin wrapper over sbatch_h100.sh so the comparison/cross-check steps can be queued the same way
# the fit arms are. Usage: ab_submit_cmd.sh JOBNAME HH:MM:SS 'command'
set -uo pipefail
source /sps/biometr/emarsot/gpurec/benchmark/cc/env.sh
bash "$CC_REPO/benchmark/cc/sbatch_h100.sh" "$1" "$2" "$3"
