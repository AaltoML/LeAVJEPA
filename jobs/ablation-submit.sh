#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(ours inv-only sigreg-only no-local joint masked random-drop audio-only)
for arm in "${ARMS[@]}"; do
  sbatch --export=ALL,ARM="$arm" -J "ablation-$arm" \
    -o "slurm_logs/%j_ablation-$arm.out" -e "slurm_logs/%j_ablation-$arm.err" \
    ablation-vggsound.job
done
